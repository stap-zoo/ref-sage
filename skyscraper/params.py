# params.py
# ---------------------------------------------------------------------------
# Parameter definition for Skyscraper: the SkyscraperParams class.
#
# SkyscraperParams is the single source of truth for an instance. It sanitizes
# the user-facing parameters and expands them into a fully-specified instance
# that the permutation, hash modes, instances and tests consume. Any value the
# user omits is filled in by the matching _init_* helper (or, for r/c/d, by the
# shared derive_rate_capacity_digest). Settings that depart from the recommended
# ones raise a ParamRecommendationWarning rather than an error.
#
# Skyscraper is a 2-branch Feistel over an (extension) field GF(p^n). It has no
# linear layer and no power-map permutation S-box; its two round functions are:
#   * Square: x -> x^2 * sigma_inv + rc   (sigma_inv is a Montgomery constant)
#   * Bar:    x -> Bar(x) + rc            (a lookup/bit-manipulation S-box)
# The squaring over GF(p^n) is captured -- exactly as in XHash -- by the
# coordinate polynomials of x -> x^2 (cpolys), so the permutation never needs to
# construct an extension field: addition and scalar (constant) multiplication of
# extension elements are componentwise, and squaring is the cpolys evaluation.
# The extension field is built ONCE here, only to derive cpolys from fmod.
# ---------------------------------------------------------------------------

# Structural imports
import warnings
from recommendations import ParamRecommendationWarning
from types import SimpleNamespace

# Math specific imports
from math import gcd, prod
from hashlib import sha256
from sage.all import GF, Integer, PolynomialRing

# Custom imports
from utils.lut import monolith_lut8
from utils.sampler import XOFFieldElementSampler
from utils.matrix import map_nested
from utils.mode import derive_rate_capacity_digest
from utils.poly import univ_from_list, power_map_coordinate_polys, poly_to_aos, map_coeffs

class SkyscraperParams:
    LABEL = "Skyscraper"

    def __init__(
        self,
        p:          int,
        n:          int = 1,
        si:         list[int] = None,
        fmod:       list[int] = None,
        cpolys:     list = None,
        R:          int = 18,
        bar_rounds: set = None, # TODO pass like this or maybe other method?
        LUTs:       dict[int, list[int]] = None,
        rcons:      list[list[int]] = None,
        r:          int = None,
        c:          int = None,
        d:          int = None,
        kappa:      int = 128,
    ):
        """
        Parameters
        ----------
        p          : field characteristic (prime); base field F = GF(p)
        n          : extension degree
        fmod       : coefficients (low->high) of a degree-n irreducible modulus over F,
                     defining GF(p^n); None means the prime field (n = 1)
        cpolys     : coordinate polynomials in Fp[x1,...,xn] describing squaring in GF(p^n) = Fp[x]/fmod;
                     derived from fmod if not provided. n is taken from cpolys/fmod.
                     cpolys[i] is the i-th coordinate polynomial, given as a list of terms tuple[int, tuple[int]],
                     where each term-tuple stores the coefficient and the exponent tuple. AoS format, see utils.poly.
        si         : mixed-radix bases for the Bar decompose/compose (e.g. [256]*32)
        R          : number of Feistel rounds (default 18)
        bar_rounds : round indices that use the Bar function (default {6,7,10,11}); all other rounds use Square
        montgomery : multiply the squaring by sigma_inv (Montgomery constant) if True
        LUTs       : per-radix lookup tables for Bar; generated via _init_LUTs if not provided
        rcons      : R x n round constants; generated via _init_rcons (SHA256) if not provided
        r          : sponge rate (over base-field elements); r + c == 2*n
        c          : sponge capacity (over base-field elements)
        d          : sponge digest size (over base-field elements)
        kappa      : target security level in bits (default 128)
        """

        # Input sanitization
        SkyscraperParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.n = n
        self.F = GF(p) # base field, not extension field
        self.bits = int(p).bit_length()
        self.kappa = kappa

        # Extension degree n and Feistel state of 2 branches => t = 2*n base-field elements.
        if cpolys is not None:
            self.n = len(cpolys)
        elif fmod is not None:
            self.n = len(fmod) - 1
        else:
            self.n = 1
        self.t = 2 * self.n

        # Montgomery constant
        machine_bit = ((self.bits + 63) // 64) * 64 # bit-width of field element in memory
        self.mont_R = self.to_field(2**machine_bit)           # Montgomery constant
        self.mont_R_inv = self.mont_R ** (-1)

        # Feistel round function: Bar
        self.m = (self.bits + 7) // 8                      # use m 8bit lookup tables in Bar
        self.si = si if si is not None else [2**8] * self.m          # mixed-radix bases
        self.rot = self.m // 2                                       # rotation amount
        self.LUTs = LUTs if LUTs is not None else self._init_LUTs()  # only forward

        # Feistel round function: Square 
        self.fmod = fmod
        self.cpolys = self._init_cpolys(cpolys) # coordinate polynomials of x -> x^2 over GF(p^n) (AoS, field coeffs)

        # Round schedule
        self.R = R
        self.bar_rounds = set(bar_rounds) if bar_rounds is not None else {6, 7, 10, 11}

        # Round constants (R rows of n coordinates; first and last rows are zero)
        self.rcons = map_nested(rcons if rcons is not None else self._init_rcons(), self.to_field)

        # Hash modes (state size for the modes is the flat 2*n base-field state)
        self.r, self.c, self.d = derive_rate_capacity_digest(self.kappa, self.t, r, c, d)

        # Final consistency checks
        self._parameter_sanitization()

    # ---------------------------------------------------------------------------
    # Small field conversion helpers
    # ---------------------------------------------------------------------------

    def from_field(self, el) -> Integer:
        return Integer(el)

    def to_field(self, n: int):
        return self.F(n)

    # ---------------------------------------------------------------------------
    # Input sanitization and security requirements
    # ---------------------------------------------------------------------------

    @staticmethod
    def _input_sanitization(params):
        """Validate the raw constructor arguments: hard checks raise, recommendation
        deviations warn (ParamRecommendationWarning) but do not raise."""

        # --- Hard checks (must always hold) ---
        if params.p == 2:
            raise NotImplementedError("Characteristic 2 not implemented")
        if params.fmod is not None and len(params.fmod) != params.n + 1:
            raise ValueError(f"degree of fmod must equal the extension degree + 1. got {len(params.fmod)}, expected {params.n + 1}")
        if params.cpolys is not None:
            if len(params.cpolys) != params.n:
                raise ValueError(f"len(cpolys) must equal the extension degree. got {len(params.cpolys)}, expected {params.n}")
            if not all(len(exps) == params.n for poly in params.cpolys for _coeff, exps in poly):
                raise ValueError(f"all exponent tuples must have length n = {self.n}")
        

    def _parameter_sanitization(self):
        """Validate the fully-derived object."""

        # --- Hard checks (must always hold) ---
        if len(self.cpolys) != self.n:
            raise ValueError(f"expected {self.n} coordinate polynomials, got {len(self.cpolys)}")
        if len(self.rcons) < self.R or not all(len(row) == self.n for row in self.rcons):
            raise ValueError(f"round constants size wrong: expected >= {self.R} rows of {self.n} coords")
        if self.m != len(self.si):
            raise ValueError(f"wrong number of table lookups: expected {self.m}, git {len(self.si)}")
        if self.m < 2 or self.m % 2 != 0:
            # The Bar rotation is by m//2 over the flattened chunks; m must be even and >= 2.
            raise ValueError(f"m must be even >= 2, got {self.m}")
        if prod(self.si) < self.p:
            raise ValueError(f"si decomposition does not cover the field: prod(si) = {prod(self.si)} < p")

        # --- Warnings (recommended, not required) ---
        if self.bits < 31:
            warnings.warn(f"TOY VERSION: field is only {self.bits} bits, recommended at least 31", ParamRecommendationWarning, stacklevel=2)
        if not (self.n in (1, 2, 3) and self.si == [256] * 32 and self.R == 18):
            warnings.warn(
                "Non-official Skyscraper parameters (recommended: n in {1,2,3}, si=[256]*32, R=18)",
                ParamRecommendationWarning, stacklevel=2,
            )

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_LUTs(self) -> dict[int, list[int]]:
        # si entries are the radix bases (e.g. 256), mapping to the 8-bit Bar lookup table respectively.
        LUTs = {}
        for s in set(self.si):
            if s == 2**8:
                LUTs[s] = monolith_lut8
            else:
                raise NotImplementedError(f"Bar LUT generation not implemented for base si={s}.")
        return LUTs

    def _init_cpolys(self, cpolys):
        """Coordinate polynomials (AoS, field-element coefficients) of x -> x^2 over GF(p^n).

        If `cpolys` is supplied it is validated and its integer coefficients lifted into F.
        Otherwise it is derived from `fmod`: GF(p^n) = F[x]/fmod is constructed (the ONLY
        place an extension field is built) and power_map_coordinate_polys(Fn, 2) is taken.
        """
        if cpolys is not None:
            return [map_coeffs(poly, self.to_field) for poly in cpolys]

        if self.n == 1:
            # Prime field: the single coordinate of x -> x^2 is just x0^2.
            return [[(self.to_field(1), (2,))]]

        if self.fmod is None:
            raise ValueError("must supply fmod (or cpolys) for n > 1")

        f = univ_from_list(PolynomialRing(self.F, 'x').gen(), self.fmod)
        if not f.is_irreducible() or f.degree() != self.n:
            raise ValueError(f"fmod = {f} is not irreducible of degree {self.n} over {self.F}")
        Fn = self.F.extension(f, name='X')
        return [poly_to_aos(poly) for poly in power_map_coordinate_polys(Fn, 2)]

    def _init_rcons(self) -> list[list[int]]:
        """Rxn round constants: constant j (0 <= j <= n*(R-2)) is sampled from a fresh SHA256 sampler seeded with the 32-byte value
        (i.to_bytes(4) || "Skyscraper" || zero-pad to 32). The first and last Feistel rounds add nothing (their constants are zero by construction).
        No Montgomery scaling is performed, round constants are assumed to be in Montgomery space already.
        Matches round constants of reference implementation from https://github.com/Skyscraper-Hash/skyscraper-sage."""
        
        label = self.LABEL.encode("ascii")
        rcons = [0] * self.n  # no round constant addition in first Feistel round
        for i in range((self.R - 2) * self.n):
            seed = int(i).to_bytes(4, "big") + label + bytes(32 - 4 - len(label))
            sampler = XOFFieldElementSampler(seed=seed, p=self.p, xof="sha256", sampling="mod", n_bytes=32, endianess="big")
            rcons.append(sampler.next())
        rcons += [0] * self.n  # no round constant addition in last Feistel round
        return [rcons[i:i + self.n] for i in range(0, len(rcons), self.n)]

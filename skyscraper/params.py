# params.py
# ---------------------------------------------------------------------------
# Parameter definition for Skyscraper: the SkyscraperParams class.
#
# SkyscraperParams is the single source of truth for an instance. It sanitizes
# the user-facing parameters and expands them into a fully-specified instance
# that the permutation, hash modes, instances and tests consume. Any value the
# user omits is filled in by the matching _init_* helper (or, for r/c/d, by the
# shared resolve_sponge_params). Settings that depart from the recommended
# ones raise a ParamRecommendationWarning rather than an error.
#
# Skyscraper is a 2-branch Feistel over an (extension) field GF(p^n). It has no
# linear layer -- so, deliberately, no _init_mat: the matrix-generation slot of
# the common params contract is filled by _init_cpolys (the squaring coordinate
# polynomials below). There is also no power-map permutation S-box; the two
# round functions are:
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
from recommendations import ParamRecommendationWarning, recommend
from types import SimpleNamespace

# Math specific imports
from math import gcd, prod
from hashlib import sha256
from sage.all import GF, Integer, PolynomialRing

# Custom imports
from monolith.params import MONOLITH_LUT8   # Skyscraper's Bar reuses Monolith's 8-bit chi table
from utils.sampler import XOFFieldElementSampler
from utils.matrix import map_nested
from utils.mode import resolve_sponge_params
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
        R:          int = None,
        bar_rounds: set = None, # TODO pass like this or maybe other method?
        LUTs:       dict[int, list[int]] = None,
        rcons:      list[list[int]] = None,
        # Sponge parameters (derived if not provided)
        r:          int = None,
        c:          int = None,
        d:          int = None,
        # Target security level (default 128 bits)
        kappa:      int = 128,
        toy:        bool = False,
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
        R          : number of Feistel rounds; the spec-fixed 18 via _init_rounds if not provided
        bar_rounds : round indices that use the Bar function (default {6,7,10,11}); all other rounds use Square
        montgomery : multiply the squaring by sigma_inv (Montgomery constant) if True
        LUTs       : per-radix lookup tables for Bar; generated via _init_LUTs if not provided
        rcons      : R x n round constants; generated via _init_cons (SHA256) if not provided
        r          : rate (number of outer state elements absorbed/squeezed per sponge step); derived if not provided
        c          : capacity (number of inner state elements for sponge); derived if not provided
        d          : digest size for generic fixed-output sponge (number of output elements); derived if not provided
        kappa      : target security level in bits (default 128)
        toy        : if True, recommendation-level checks warn instead of raising (default False)
        """

        # Input sanitization
        SkyscraperParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.n = n
        self.t = 2*self.n  # Feistel state of 2 branches => t = 2*n base-field elements
        self.F = GF(p) # base field, not extension field
        self.bits = int(p).bit_length()
        self.kappa = kappa
        self.toy = toy

        # Sponge parameters
        self.r, self.c, self.d = resolve_sponge_params(kappa=self.kappa, p=self.p, t=self.t, r=r, c=c, d=d, toy=toy)

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
        self.R = R if R is not None else self._init_rounds()
        self.bar_rounds = set(bar_rounds) if bar_rounds is not None else {6, 7, 10, 11}

        # Round constants (R rows of n coordinates; first and last rows are zero)
        self.rcons = map_nested(rcons if rcons is not None else self._init_cons(), self.to_field)

        # Parameter sanitization
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
            recommend(f"TOY VERSION: field is only {self.bits} bits, recommended at least 31", self.toy)
        if not (self.n in (1, 2, 3) and self.si == [256] * 32 and self.R == 18):
            warnings.warn(
                "Non-official Skyscraper parameters (recommended: n in {1,2,3}, si=[256]*32, R=18)",
                ParamRecommendationWarning, stacklevel=2,
            )

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_rounds(self) -> int:
        """The round number is fixed by the spec: 18 Feistel rounds for every
        instance (https://eprint.iacr.org/2025/058), independent of the field."""
        return 18

    def _init_LUTs(self) -> dict[int, list[int]]:
        # si entries are the radix bases (e.g. 256), mapping to the 8-bit Bar lookup
        # table (Monolith's chi table, which Skyscraper's spec reuses) respectively.
        SKYSCRAPER_LUTS = {2**8: MONOLITH_LUT8}
        LUTs = {}
        for s in set(self.si):
            if s not in SKYSCRAPER_LUTS:
                raise NotImplementedError(f"Error: Not implemented -- Bar LUT generation for base si={s}")
            LUTs[s] = SKYSCRAPER_LUTS[s]
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

    def _init_cons(self) -> list[list[int]]:
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

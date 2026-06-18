# params.py
# ---------------------------------------------------------------------------
# Parameter definition for Arion: the ArionParams class.
#
# ArionParams is the single source of truth for an instance. It sanitizes the
# user-facing parameters and expands them into a fully-specified instance that
# every other file (permutation, hash modes, instances, tests) consumes. Any
# value the user omits is filled in by the matching _init_* helper (or, for
# r/c/d, by the shared derive_rate_capacity_digest). Settings that depart from
# the recommended ones raise a ParamRecommendationWarning rather than an error.
# ---------------------------------------------------------------------------

# Structural imports
from recommendations import ParamRecommendationWarning
from types import SimpleNamespace

# Math specific imports
import warnings
from math import gcd
from sage.all import GF, Integer, legendre_symbol

# Custom imports
from utils.matrix import simple_circulant_matrix, map_nested, invert_matrix
from utils.sampler import XOFFieldElementSampler
from utils.mode import derive_rate_capacity_digest


class ArionParams:
    def __init__(
        self,
        p:           int,
        t:           int,
        R:           int,
        alpha1:      int = None,
        alpha2:      int = None,
        alpha1_inv:  int = None,
        alpha2_inv:  int = None,
        coeffs_g:    list[list[list[int]]] = None,
        coeffs_h:    list[list[int]] = None,
        rcons:       list[list[int]] = None,
        M:           list[list[int]] = None,
        r:           int = None,
        c:           int = None,
        d:           int = None,
        kappa:       int = 128,
    ):
        """
        Parameters
        ----------
        p             : field characteristic (prime)
        t             : permutation state size (branches)
        R             : number of rounds
        alpha1        : exponent of the power permutation applied to v_0..v_{t-2} in GTDS; smallest positive integer coprime to p-1 if not provided
        alpha2        : exponent of the power permutation applied to v_{t-1} in GTDS; arbitrary positive integer coprime to p-1 if not provided
        alpha1_inv    : alpha1^{-1} mod (p-1); computed via _init_alpha1_inv if not provided
        alpha2_inv    : alpha2^{-1} mod (p-1); computed via _init_alpha2_inv if not provided
        coeffs_g      : R*(t-1) [a, b] pairs for the rational maps g_i (legendre_symbol(a^2 - 4*b, p) == -1); generated via _init_constants if not provided
        coeffs_h      : R*(t-1) constants for the linear maps h_i; generated via _init_constants if not provided
        rcons         : Rxt affine round constants; generated via _init_constants if not provided
        M             : MDS matrix (txt); generated via _init_M if not provided
        r             : rate (number of outer state elements absorbed/squeezed per sponge step);
                        derived from kappa/t via derive_rate_capacity_digest if not provided
        c             : capacity (number of inner state elements); derived if not provided
        d             : digest size (number of output elements); derived if not provided
        kappa         : target security level in bits (default 128)
        """

        # Input sanitization
        ArionParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Rounds (set before _init_constants, whose derivation depends on R)
        self.R = R

        # Non-linear layer (GTDS)
        self.alpha1 = alpha1 if alpha1 is not None else self._init_alpha1()
        self.alpha2 = alpha2 if alpha2 is not None else self._init_alpha2()
        self.alpha1_inv = alpha1_inv if alpha1_inv is not None else self._init_alpha1_inv()
        self.alpha2_inv = alpha2_inv if alpha2_inv is not None else self._init_alpha2_inv()
        # coeffs_g, coeffs_h and rcons share one SHAKE256 stream, so they are derived together.
        if coeffs_g is None or coeffs_h is None or rcons is None:
            _coeffs_g, _coeffs_h, _rcons = self._init_constants()
            rcons = rcons if rcons is not None else _rcons
            coeffs_g = coeffs_g if coeffs_g is not None else _coeffs_g
            coeffs_h = coeffs_h if coeffs_h is not None else _coeffs_h
        self.coeffs_g = map_nested(coeffs_g, self.to_field)
        self.coeffs_h = map_nested(coeffs_h, self.to_field)

        # Hash modes
        self.r, self.c, self.d = derive_rate_capacity_digest(self.kappa, self.t, r, c, d)

        # Affine layer
        self.M = map_nested(M if M is not None else self._init_M(), self.to_field)
        self.M_inv = invert_matrix(self.M)
        self.rcons = map_nested(rcons, self.to_field)

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
        if params.t < 2:
            raise ValueError(f"state size t must be at least 2. Got {params.t}")
        if params.alpha1 is not None and gcd(params.alpha1, params.p - 1) != 1:
            raise ValueError("alpha1 power map does not define a permutation (gcd(alpha1, p-1) != 1)")
        if params.alpha2 is not None and gcd(params.alpha2, params.p - 1) != 1:
            raise ValueError("alpha2 power map does not define a permutation (gcd(alpha2, p-1) != 1)")

        # --- Warnings (recommended, not required) ---
        field_bits = int(params.p).bit_length()
        if field_bits < 31:
            warnings.warn(f"TOY VERSION: field is only {field_bits} bits", ParamRecommendationWarning, stacklevel=2)

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_alpha1(self) -> int:
        alpha1 = 2
        while gcd(alpha1, self.p - 1) != 1:
            alpha1 += 1
        return alpha1

    def _init_alpha2(self) -> int:
        """While paper states that alpha2 can be an arbitrary integer coprime to p - 1, it
        states 121 <= alpha <= 257 in section 2.4 and choses the smallest of [257, 161, 129, 125, 123, 121]
        in the reference implementation, see https://github.com/sca-research/Arion.
        These values are chosen large for Gröbner basis resistance (degree-alpha2 relations per round)
        but with short addition chains for cheap y^alpha2 = x verification."""
        for alpha2 in (257, 161, 129, 125, 123, 121):
            if gcd(alpha2, self.p - 1) == 1:
                return alpha2
        raise ValueError("no valid alpha2 found in [257, 161, 129, 125, 123, 121]")

    def _init_alpha1_inv(self) -> int:
        return pow(self.alpha1, -1, self.p - 1)

    def _init_alpha2_inv(self) -> int:
        return pow(self.alpha2, -1, self.p - 1)

    def _init_M(self) -> list[list[int]]:
        return simple_circulant_matrix(self.t)

    def _init_constants(self):
        # Deterministic constant generation via SHAKE256, so coeffs_g/coeffs_h/rcons
        # can be reproduced from (p, t, R) instead of relying on Sage's unseeded random_element(), as
        # used by the reference implementation in https://github.com/sca-research/Arion.
        seed = f"Arion({self.p},{self.t},{self.R})".encode("ascii")

        rcons = XOFFieldElementSampler(seed=seed + b"aff", p=self.p, xof="shake_256", sampling="mod").grid(self.R, self.t)
        coeffs_h = XOFFieldElementSampler(seed=seed + b"h", p=self.p, xof="shake_256", sampling="mod").grid(self.R, self.t - 1)

        # coeffs_g: pairs [a, b] with legendre_symbol(a^2 - 4*b, p) == -1, rejection-sampled
        # from a candidate pool drawn from the same SHAKE stream. Sampled flat (round-major)
        # since the rejection breaks the row alignment of the stream, then reshaped.
        n = self.R * (self.t - 1)
        candidates = XOFFieldElementSampler(seed=seed + b"g", p=self.p, xof="shake_256", sampling="mod").grid(4 * n, 2)
        coeffs_g = []
        for a, b in candidates:
            if legendre_symbol(a**2 - 4 * b, self.p) == -1:
                coeffs_g.append([a, b])
                if len(coeffs_g) == n:
                    break
        if len(coeffs_g) < n:
            raise RuntimeError("not enough valid coeffs_g candidates; widen the candidate pool")

        # Reshape to [round][branch]
        w = self.t - 1
        coeffs_g = [coeffs_g[r * w:(r + 1) * w] for r in range(self.R)]
        return coeffs_g, coeffs_h, rcons

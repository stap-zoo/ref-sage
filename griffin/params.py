# params.py
# ---------------------------------------------------------------------------
# Parameter definition for Griffin: the GriffinParams class.
#
# GriffinParams is the single source of truth for an instance. It takes the
# user-facing parameters and expands them into a fully-specified instance:
# it sanitizes the inputs and stores/derives every value the permutation and
# hash modes consume. Any value the user omits is filled in by the matching
# _init_* helper (or, for r/c/d, by the shared derive_rate_capacity_digest).
# Settings that depart from the recommended ones raise a
# ParamRecommendationWarning rather than an error.
# ---------------------------------------------------------------------------

# Structural imports
from recommendations import ParamRecommendationWarning
from types import SimpleNamespace

# Math specific imports
import warnings
from math import gcd
from sage.all import GF, Integer, legendre_symbol

# Custom imports
from utils import m4_to_block_circulant_matrix, circulant, XOFFieldElementSampler, map_to_field, invert_matrix
from modes import derive_rate_capacity_digest


class GriffinParams:
    def __init__(
        self,
        p:         int,
        t:         int,
        R:         int,
        alpha:     int,
        alpha_inv: int = None,
        rcons:     list[list[int]] = None,
        coeffs_G:  list[list[int]] = None,
        M:         list[list[int]] = None,
        r:         int = None,
        c:         int = None,
        d:         int = None,
        kappa:     int = 128,
    ):
        """
        Parameters
        ----------
        p         : field characteristic (prime)
        t         : permutation state size; must be 3 or a multiple of 4
        R         : number of rounds
        alpha     : non-linear layer exponent (3, 5, or 7)
        alpha_inv : alpha^{-1} mod (p-1); computed via _init_alpha_inv if not provided
        rcons     : (R-1)xt round constants (the final round has none); generated via SHAKE128 if not provided
        coeffs_G  : (t-2) [a, b] pairs for the quadratic maps G_i; generated via SHAKE128 if not provided
        M         : mixing matrix (txt); generated via _init_M if not provided
        r         : rate (number of outer state elements absorbed/squeezed per sponge step);
                    derived from kappa/t via derive_rate_capacity_digest if not provided
        c         : capacity (number of inner state elements); derived if not provided
        d         : digest size (number of output elements); derived if not provided
        kappa     : target security level in bits (default 128)
        """

        # Input sanitization
        GriffinParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Rounds (set before _init_constants, whose derivation depends on R)
        self.R = R

        # Non-linear layer
        self.alpha = alpha
        self.alpha_inv = alpha_inv if alpha_inv is not None else self._init_alpha_inv()
        # rcons and coeffs_G share one SHAKE128 stream, so they are derived together.
        if rcons is None or coeffs_G is None:
            _rcons, _coeffs_G = self._init_constants()
            rcons = rcons if rcons is not None else _rcons
            coeffs_G = coeffs_G if coeffs_G is not None else _coeffs_G
        self.coeffs_G = map_to_field(coeffs_G, self.to_field)

        # Hash modes
        self.r, self.c, self.d = derive_rate_capacity_digest(self.kappa, self.t, r, c, d)

        # Linear layer
        self.M = map_to_field(M if M is not None else self._init_M(), self.to_field)
        self.M_inv = invert_matrix(self.M)

        # Round constants: pad with a zero row so AffineLayer can uniformly index
        # rcons[round_idx] for round_idx in 0..R-1 (the final round has none).
        self.rcons = map_to_field(rcons, self.to_field) + [[self.F.zero()] * self.t]

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
        if not (params.t == 3 or params.t % 4 == 0):
            raise ValueError(f"state size t must be 3 or a multiple of 4. Got {params.t}")
        if params.alpha not in (3, 5, 7):
            raise ValueError(f"alpha must be 3, 5, or 7. Got {params.alpha}")
        if gcd(params.alpha, params.p - 1) != 1:
            raise ValueError("power map does not define a permutation (gcd(alpha, p-1) != 1)")

        # --- Warnings (recommended, not required) ---
        field_bits = int(params.p).bit_length()
        if field_bits < 31:
            warnings.warn(f"TOY VERSION: field is only {field_bits} bits", ParamRecommendationWarning, stacklevel=2)

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_alpha_inv(self) -> int:
        return pow(self.alpha, -1, self.p - 1)

    def _init_M(self) -> list[list[int]]:
        if self.t == 3:
            return circulant([2, 1, 1])
        else:
            return m4_to_block_circulant_matrix(self.t)

    def _init_constants(self):
        """Derive the (R-1)xt round constants and the (t-2) quadratic-map [a, b] pairs from one
        SHAKE128 stream seeded with "Griffin" || p (little-endian 64-bit limbs)."""
        # Initialize the sampler, seeded with "Griffin" followed by the field characteristic serialized as little-endian 64-bit limbs.
        n_bytes = ((self.p.bit_length() + 63) // 64) * 8
        seed = b"Griffin" + self.p.to_bytes(n_bytes, "little")
        reader = XOFFieldElementSampler(seed=seed, p=self.p, xof="shake_128", sampling="bitmask")

        # Generate round constants. The last round has no round constants.
        rcons = [[reader.next() for _ in range(self.t)] for _ in range(self.R - 1)]

        # generate coefficients for the quadratic maps L_i
        coeffs = []

        # random a/b: distinct, non-zero, and legendre_symbol(a^2 - 4*b, p) == -1
        while True:
            a = reader.next_nonzero()
            b = reader.next_nonzero()
            while a == b:
                b = reader.next_nonzero()
            if legendre_symbol(a**2 - 4 * b, self.p) == -1:
                coeffs.append([a, b])
                break

        # remaining a_i = i*a, b_i = i^2*b, resampling b_i if a_i == b_i
        a_0, b_0 = coeffs[0]
        for i in range(2, self.t - 1):
            a_i = (a_0 * i) % self.p
            b_i = (b_0 * i * i) % self.p
            while a_i == b_i:
                b_i = reader.next_nonzero()
            coeffs.append([a_i, b_i])

        return rcons, coeffs

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
import warnings
from recommendations import ParamRecommendationWarning
from types import SimpleNamespace

# Math specific imports
from math import gcd
from sage.all import GF, Integer, legendre_symbol

# Custom imports
from utils.matrix import m4_to_block_circulant_matrix, circulant, map_nested, invert_matrix
from utils.sampler import XOFFieldElementSampler
from utils.mode import derive_rate_capacity_digest

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Pinned mixing matrix for t = 3 (https://eprint.iacr.org/2022/403, Section 4.2);
# every other official state size (t a multiple of 4) uses the generic
# M4-block-circulant construction in _init_mat.
GRIFFIN_M = {3: circulant([2, 1, 1])}


class GriffinParams:
    def __init__(
        self,
        p:         int,
        t:         int,
        alpha:     int,
        R:         int = None,
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
        alpha     : non-linear layer exponent (3, 5, or 7)
        R         : number of rounds; derived via _init_rounds if not provided (not yet implemented)
        alpha_inv : alpha^{-1} mod (p-1); computed via _init_alpha_inv if not provided
        rcons     : (R-1)xt round constants (the final round has none); generated via SHAKE128 if not provided
        coeffs_G  : (t-2) [a, b] pairs for the quadratic maps G_i; generated via SHAKE128 if not provided
        M         : mixing matrix (txt); generated via _init_mat if not provided
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

        # Rounds (set before _init_cons, whose derivation depends on R)
        self.R = R if R is not None else self._init_rounds()

        # Non-linear layer
        self.alpha = alpha
        self.alpha_inv = alpha_inv if alpha_inv is not None else self._init_alpha_inv()
        # rcons and coeffs_G share one SHAKE128 stream, so they are derived together.
        if rcons is None or coeffs_G is None:
            _rcons, _coeffs_G = self._init_cons()
            rcons = rcons if rcons is not None else _rcons
            coeffs_G = coeffs_G if coeffs_G is not None else _coeffs_G
        self.coeffs_G = map_nested(coeffs_G, self.to_field)

        # Hash modes
        self.r, self.c, self.d = derive_rate_capacity_digest(self.kappa, self.t, r, c, d)

        # Linear layer
        self.M = map_nested(M if M is not None else self._init_mat(), self.to_field)
        self.M_inv = invert_matrix(self.M)

        # Round constants: pad with a zero row so AffineLayer can uniformly index
        # rcons[round_idx] for round_idx in 0..R-1 (the final round has none).
        self.rcons = map_nested(rcons, self.to_field) + [[self.F.zero()] * self.t]

        # Parameter sanitization: validate the fully-constructed (stored/derived) values
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

    def _parameter_sanitization(self):
        """Validate the fully-constructed parameter object (stored/derived values):
        hard checks raise, recommendation deviations warn (ParamRecommendationWarning)."""

        # --- Hard checks (must always hold) ---
        if len(self.M) != self.t or any(len(row) != self.t for row in self.M):
            raise ValueError(f"M must be a {self.t} x {self.t} matrix")
        if len(self.rcons) != self.R or any(len(row) != self.t for row in self.rcons):
            raise ValueError(f"rcons (incl. the zero padding row) must be an {self.R} x {self.t} grid")
        if len(self.coeffs_G) != self.t - 2 or any(len(pair) != 2 for pair in self.coeffs_G):
            raise ValueError(f"coeffs_G must hold {self.t - 2} [a, b] pairs")

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_rounds(self) -> int:
        """Derive the round number from the target security level kappa.
        TODO: implement the round-number criterion of the Griffin paper
        (https://eprint.iacr.org/2022/403, Section 5: Groebner basis bound with
        a 20% security margin); until then R must be passed explicitly."""
        raise NotImplementedError("Error: Not implemented -- round number derivation for Griffin")

    def _init_alpha_inv(self) -> int:
        return pow(self.alpha, -1, self.p - 1)

    def _init_mat(self) -> list[list[int]]:
        """The pinned t = 3 matrix (module constant GRIFFIN_M), otherwise the
        generic M4 block-circulant construction for t a multiple of 4."""
        if self.t in GRIFFIN_M:
            return GRIFFIN_M[self.t]
        return m4_to_block_circulant_matrix(self.t)

    def _init_cons(self):
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

# params.py
# ---------------------------------------------------------------------------
# Parameter definition for Griffin: the GriffinParams class.
#
# GriffinParams is the single source of truth for an instance. It sanitizes
# user-facing parameters and expands them into a fully-specified instance that
# the permutation, hash modes, instances and tests consume. Any value the user
# omits is filled in by the matching _init_* helper. Settings that depart from 
# the recommended ones raise a ParamRecommendationWarning rather than an error.
# ---------------------------------------------------------------------------

# Structural imports
from recommendations import recommend
from types import SimpleNamespace

# Math specific imports
from math import gcd, ceil, log2
from sage.all import GF, Integer, legendre_symbol

# Custom imports
from utils.matrix import m4_to_block_circulant_matrix, circulant, map_nested, invert_matrix
from utils.sampler import XOFFieldElementSampler
from utils.mode import SpongeLE
from utils.complexities import gb_comp

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
        # Sponge parameters (derived if not provided)
        r:         int = None,
        c:         int = None,
        d:         int = None,
        # Target security level (default 128 bits)
        kappa:     int = 128,
        toy:       bool = False,
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
        r         : rate (number of outer state elements absorbed/squeezed per sponge step); derived if not provided
        c         : capacity (number of inner state elements for sponge); derived if not provided
        d         : digest size for generic fixed-output sponge (number of output elements); derived if not provided
        kappa     : target security level in bits (default 128)
        toy       : if True, recommendation-level checks warn instead of raising (default False)
        """

        # Input sanitization
        GriffinParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa
        self.toy = toy

        # Sponge parameters
        self.sponge = SpongeLE(kappa=kappa, p=p, t=t, r=r, c=c, d=d, to_field=self.to_field, toy=toy)

        # Non-linear layer
        self.alpha = alpha
        self.alpha_inv = alpha_inv if alpha_inv is not None else self._init_alpha_inv()

        # Rounds (set before _init_cons, whose derivation depends on R)
        self.R = R if R is not None else self._init_rounds()

        # rcons and coeffs_G share one SHAKE128 stream, so they are derived together.
        if rcons is None or coeffs_G is None:
            _rcons, _coeffs_G = self._init_cons()
            rcons = rcons if rcons is not None else _rcons
            coeffs_G = coeffs_G if coeffs_G is not None else _coeffs_G
        self.coeffs_G = map_nested(coeffs_G, self.to_field)

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
        if params.alpha not in (3, 5, 7, 11):
            raise ValueError(f"alpha must be 3, 5, or 7. Got {params.alpha}")
        if gcd(params.alpha, params.p - 1) != 1:
            raise ValueError("power map does not define a permutation (gcd(alpha, p-1) != 1)")

        # --- Warnings (recommended, not required) ---
        field_bits = int(params.p).bit_length()
        if field_bits < 31:
            recommend(f"TOY VERSION: field is only {field_bits} bits", params.toy)

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

    # ---------------------------------------------------------------------------
    # Security analysis helpers (Section 5.2 & Section 6)
    # ---------------------------------------------------------------------------

    def _differential_comp(self, R: int) -> float:
        """log2 of the upper bound on any R-round differential trail probability."""
        return (R/2.5) * log2(self.p / (self.alpha - 1))
    
    def _groebner_intermediate_comp(self, R: int) -> float:
        """log2 of the upper bound on R-round Groebner-basis attack (F4). 
        Intermediate variables, see Equation 8 of https://eprint.iacr.org/2022/403.pdf"""
        n = 1 + self.t * R  # number of variables/equations in the system
        dreg = self.alpha * R  # estimated lower bound on dreg
        return gb_comp(dreg=dreg, nv=n, w=2) 

    def _groebner_partial_intermediate_comp(self, R: int) -> float:
        """log2 of the upper bound on R-round Groebner-basis attack (F4). 
        Partial intermediate variables, see Equation 9 of https://eprint.iacr.org/2022/403.pdf"""
        n = 1 + R  # number of variables/equations in the system
        dreg = self.alpha ** R  # estimated lower bound on dreg
        return gb_comp(dreg=dreg, nv=n, w=2) 

    def _init_rounds(self) -> int:
        """Derive the round number from the target security level kappa.
        Round-number criterion of the Griffin paper (https://eprint.iacr.org/2022/403, 
        Section 5.2: Groebner basis bound with a 20% security margin)"""

        target = self.kappa
        
        gb_attacks = [self._groebner_intermediate_comp, self._groebner_partial_intermediate_comp]
        for R_gb in range(1, 10_000):
            complexities = [attack(R_gb) for attack in gb_attacks]
            if min(complexities) >= target:
                break
        R_gb += 1

        # Round numbers for differential attacks (_differential_comp) expliclty stated in paper (page 21)
        # Not present in first version (https://eprint.iacr.org/archive/2022/403/1648711416.pdf),
        # but in published version (https://link.springer.com/chapter/10.1007/978-3-031-38548-3_19)
        R_diff = ceil(2.5 * target / (log2(self.p) - log2(self.alpha - 1)))

        return ceil(1.2 * max(6, R_diff, 1 + R_gb)) # 20% security margin

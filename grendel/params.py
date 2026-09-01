# params.py
# GrendelParams: the fully-specified parameter set for Grendel (single source of truth per instance).

# Structural imports
from recommendations import recommend
from types import SimpleNamespace

# Math specific imports
from math import gcd, ceil, floor, log2
from sage.all import GF, Integer

# Custom imports
from utils.matrix import map_nested, invert_matrix, vandermonde_mds_matrix
from utils.sampler import XOFFieldElementSampler
from utils.complexities import gb_comp2

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Security margin on the attack work factors (Sec. 5.7): the round number R is
# the smallest N such that every applicable attack costs at least 2^(MARGIN * kappa).
MARGIN = 1.25

# ---------------------------------------------------------------------------
# Parameter definition
# ---------------------------------------------------------------------------
class GrendelParams:
    def __init__(
        self,
        p:     int,
        t:     int,
        alpha: int = None,
        R:     int = None,
        g:     int = None,
        M:     list[list[int]] = None,
        rcons: list[list[int]] = None,
        # Mode of operation: sponge params dict dict(r=.., c=.., d=..) (Grendel is sponge-only,
        # so comp stays None). r + c == t; d = 1 for Grendel's root-finding analysis.
        sponge: dict = None,
        comp:   dict = None,
        # Target security level (default 128 bits)
        kappa: int = 128,
        toy: bool = False,
    ):
        """
        Parameters
        ----------
        p     : field characteristic (prime)
        t     : state size in field elements (the paper's m; the design requires t >= 2)
        r     : sponge rate; together with c it must satisfy r + c == t
        c     : sponge capacity
        d     : digest size (the paper's output_length, in field elements)
        alpha : exponent of the power-map part of the S-box; 2 if p = 3 mod 4, otherwise
                the smallest integer > 2 coprime with p-1 if not provided (Sec. 4.1)
        R     : number of rounds; derived from the attack complexities of Sec. 5.6/5.7 if not provided
        g     : primitive element of F_p^* seeding the MDS matrix; smallest one if not provided
        M     : t x t MDS matrix; derived from the systematic Reed-Solomon generator matrix (Sec. 4.5, Algorithm 4) if not provided
        rcons : R x t round constants c^(0), ..., c^(R-1); generated via _init_cons (SHAKE256, Sec. 4.6, Algorithm 5) if not provided
        r     : rate (number of outer state elements absorbed/squeezed per sponge step); derived if not provided
        c     : capacity (number of inner state elements for sponge); derived if not provided
        d     : digest size for generic fixed-output sponge (number of output elements); derived if not provided
        kappa : target security level in bits (default 128)
        toy   : if True, recommendation-level checks warn instead of raising (default False)
        """

        # Input sanitization
        GrendelParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa
        self.toy = toy

        # Mode of operation: per-mode param dicts (consumed by the mode functions, not the
        # permutation). Grendel's security analysis reads the resolved sizes below.
        self.sponge, self.comp = sponge, comp

        # Non-linear layer
        # The S-box S(x) = x^alpha * legendre(x) equals the single power map x^e with e = alpha + (p-1)/2.
        self.alpha = alpha if alpha is not None else self._init_alpha()
        self.e = self.alpha + (p - 1) // 2
        self.e_inv = pow(self.e, -1, p - 1)

        # Round number
        self.R = R if R is not None else self._init_rounds()

        # Linear layer
        self.g = self.to_field(g) if g is not None else self.F.multiplicative_generator()
        self.M = map_nested(M if M is not None else self._init_mat(), self.to_field)
        self.M_inv = invert_matrix(self.M)

        # Round constants
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
        if params.t < 2:
            raise ValueError(f"state size t must be at least 2. Got {params.t}")
        if params.alpha is not None and gcd(params.alpha + (params.p - 1) // 2, params.p - 1) != 1:
            raise ValueError("S-box x^alpha * legendre(x) = x^(alpha + (p-1)/2) does not define a permutation")

        # --- Warnings (recommended, not required) --- 
        p, t, kappa = params.p, params.t, params.kappa
        field_bits = int(p).bit_length()
        if field_bits < 31:
            msg = f"TOY VERSION: field is only {field_bits} bits"
            recommend(msg, params.toy)

    def _parameter_sanitization(self):
        """Validate the fully-constructed parameter object (stored/derived values):
        hard checks raise, recommendation deviations warn (ParamRecommendationWarning)."""

        # --- Hard checks (must always hold) ---
        if len(self.M) != self.t or any(len(row) != self.t for row in self.M):
            raise ValueError(f"M must be a {self.t} x {self.t} matrix")
        if len(self.rcons) != self.R or any(len(row) != self.t for row in self.rcons):
            raise ValueError(f"rcons must be an {self.R} x {self.t} grid (one t-vector per round)")
        
        # --- Warnings (recommended, not required) ---
        if self._integral_comp() < self.kappa:
            msg = f"TOY VERSION: integral attacks cost only 2^{self._integral_comp():.1f} < 2^{self.kappa}"
            recommend(msg, self.toy)
        if self.sponge["d"] == 1 and self._rootfinding_comp() < self.kappa:
            msg = f"TOY VERSION: root-finding attacks cost only 2^{self._rootfinding_comp():.1f} < 2^{self.kappa}"
            recommend(msg, self.toy)

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------
    def _init_alpha(self) -> int:
        """The power-map exponent of Sec. 4.1: 2 when p = 3 mod 4 (where squaring only
        loses the sign information the Legendre symbol restores), otherwise the smallest
        integer greater than 2 that is coprime with p-1."""
        if self.p % 4 == 3:
            return 2
        for alpha in range(3, self.p):
            if gcd(alpha, self.p - 1) == 1:
                return alpha

    def _init_mat(self) -> list[list[int]]:
        """The t x t MDS matrix of Sec. 4.5 (Algorithm 4): row-reduce the t x 2t
        Reed-Solomon generator matrix G[i][j] = g^(i*j) to systematic form (I | M^T)
        and return the transpose of the right half. This is the shared Vandermonde
        echelon construction with the RPO-style transpose."""
        return vandermonde_mds_matrix(self.p, self.t, self.g, transpose=True)

    def _init_cons(self) -> list[list[int]]:
        """The round constants of Sec. 4.6 (Algorithm 5).
        This is exactly the XOF sampler's big-endian "mod" strategy (its chunk width ceil(bits/8) + 1 equals the paper's w)."""
        seed = f"grendel-{self.p}-{self.t}-{self.kappa}".encode()
        sampler = XOFFieldElementSampler(seed=seed, p=self.p, xof="shake_256", sampling="mod", endianess="big")
        return sampler.grid(self.R, self.t)
    
    # ---------------------------------------------------------------------------
    # Security analysis helpers (Section 5)
    # ---------------------------------------------------------------------------
    def _integral_comp(self) -> float:
        """Log2 of subspace sum and propagation attacks. Independent of round number R"""
        subspace_propagation = ceil((self.t + 1) / 2) * log2(self.p)
        subspace_sum = self.t * log2(self.p)
        return min(subspace_propagation, subspace_sum)
    
    def _rootfinding_comp(self) -> float:
        """log2 of root-finding attack for 1 output. Independent of round number R."""
        assert(self.sponge["d"] == 1)
        return log2(self.p)

    def _guessing_legendre_rootfinding_comp(self, R: int) -> float:
        """log2 of the upper bound on any R-round root-finding attack with known Legendre symbols."""
        return (R * self.t - self.sponge["c"]) + R * log2(self.alpha) + 2 * log2(R)

    def _linear_comp(self, R: int) -> float:
        """log2 of the upper bound on any R-round linear trail probability."""
        return (R//2) * log2(self.p / (2*self.alpha))
    
    def _differential_comp(self, R: int) -> float:
        """log2 of the upper bound on any R-round differential trail probability."""
        return (R//2) * log2(self.p / (4*self.alpha - 2))
    
    def _groebner_comp(self, R: int) -> float:
        """log2 of the upper bound on any R-round Groebner-basis attack."""
        # Without known Legendre symbols
        n = 2*(R * self.t - self.sponge["c"]) # number of variables/equations in the system
        dreg = (1 + n//2 * (self.alpha + 3)) // 8 # Equation 30/31
        return gb_comp2(dreg=dreg, nv=n, w=2)

    def _guessing_legendre_groebner_comp(self, R: int) -> float:
        """log2 of the upper bound on any R-round Groebner-basis attack."""
        # With known Legendre symbols
        # Approximating the Legendre symbols as uniform random variables across {−1, 1}, 
        # the attacker has to guess O(2^n) times before his guess is correct.
        n = R * self.t - self.sponge["c"] # number of variables/equations in the system
        dreg = (1 + n * (self.alpha - 1)) // 9 # Equation 35/36
        return n + gb_comp2(dreg=dreg, nv=n, w=2) 

    def _init_rounds(self) -> int:
        """Smallest round number R such that every applicable attack of Table 1 costs 
        at least 2^(MARGIN * kappa) field operations (Sec. 5.7)"""

        target = MARGIN * self.kappa
        attacks = [self._guessing_legendre_rootfinding_comp, self._linear_comp, self._differential_comp,
                   self._groebner_comp, self._guessing_legendre_groebner_comp]

        for R in range(1, 10_000):
            complexities = [attack(R) for attack in attacks]
            if min(complexities) >= target:
                return R

        raise ValueError(f"no round number below 10000 reaches the target security level "
                         f"2^{target:.1f} (is the field large enough for kappa = {self.kappa}?)")
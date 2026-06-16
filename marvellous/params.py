from utils import vandermonde_mds_matrix, rpo_mds_matrix, XOFFieldElementSampler, map_to_field, invert_matrix
from complexities import gb_comp
from sage.all import GF, Integer, matrix, vector
from math import ceil, floor, gcd, log

# ---------------------------------------------------------------------------
# Rescue
# ---------------------------------------------------------------------------

class RescueParams:
    def __init__(
        self,
        p:         int,
        t:         int,
        d:         int,
        alpha:     int = None,
        alpha_inv: int = None,
        R:         int = None,
        g:         int = None,
        M:         list[list[int]] = None,
        rcons:     list[list[int]] = None,
        r:         int = None,
        c:         int = None,
        kappa:     int = 128,
    ):
        """
        Parameters
        ----------
        p         : field characteristic (prime)
        t         : permutation state size
        d         : digest size (number of output elements)
        alpha     : power-map exponent for S-Box
        alpha_inv : power-map exponent for inverse S-Box, i.e., alpha^{-1} mod (p-1)
        R         : number of rounds; computed from kappa if not provided
        g         : a primitive element of GF(p) (e.g. Field.generator); required if M is not given
        M         : MDS matrix (txt); generated via vandermonde_mds_matrix if not provided
        rcons     : (2*R+1)xt round-constants; generated via SHAKE256 if not provided
        r         : rate (defaults to t - c)
        c         : capacity (number of inner state elements); rate r = t - c
        kappa     : target security level in bits
        """
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Hash modes
        self.c = c
        self.r = r if r is not None else t - self.c
        self.d = d

        # Non-linear layer
        self.alpha = alpha if alpha is not None else self._init_alpha()
        self.alpha_inv = alpha_inv if alpha_inv is not None else pow(self.alpha, -1, p - 1)

        # Rounds
        self.R = R if R is not None else self._init_rounds()

        # Linear Layer
        self.g = g if g is not None else self.F.multiplicative_generator() # smallest primitive element

        M = M if M is not None else self._init_mds()
        self.M = map_to_field(M, self.to_field)
        self.M_inv = invert_matrix(self.M)

        self.rcons = rcons if rcons is not None else self._init_rcons()
        self.rcons = map_to_field(self.rcons, self.to_field)

    # ---------------------------------------------------------------------------
    # Small helpers
    # ---------------------------------------------------------------------------

    def from_field(self, el) -> Integer:
        return Integer(el)

    def to_field(self, n: int):
        return self.F(n)

    def _init_alpha(self) -> int:
        for alpha in range(3, self.p):
            if gcd(alpha, self.p - 1) == 1:
                return alpha

    def _init_mds(self) -> list[list[int]]:
        return vandermonde_mds_matrix(self.p, self.t, self.g, transpose=False)
    
    def _init_rcons(self) -> list[list[int]]:
        # Round constants created via the Rescue key schedule, where key-schedule material is sampled via
        # SHAKE256, t rows at a time, until t consecutive rows form an invertible txt matrix. The two rows
        # following that block become the initial constant and the constants-schedule's additive constant.
        seed = b"winteriscoming"
        num_blocks = 1
        while True:
            rows = XOFFieldElementSampler(seed=seed, p=self.p, xof="shake_256", sampling="mod").grid(num_blocks * self.t + 2, self.t)
            for i in range(0, len(rows) - self.t - 1, self.t):
                constants_matrix = rows[i:i + self.t]
                if matrix(self.F, constants_matrix).is_invertible():
                    initial_constant = rows[i + self.t]
                    constants_constant = rows[i + self.t + 1]
                    return self._key_schedule(constants_matrix, initial_constant, constants_constant)
            num_blocks *= 2

    def _key_schedule(self, Mc: list[list[int]], c0: list[int], c: list[int]) -> list[list[int]]:
        """Rescue's key-schedule (Application of Rescue BlockCipher with key=0)"""
        M = matrix(self.F, self.M)
        Mc = matrix(self.F, Mc)
        c = vector(self.F, c)

        state = vector(self.F, c0)
        key_injection = vector(self.F, c0)

        result = [list(state)]
        for r in range(2 * self.R):
            state = state.apply_map(lambda x: x**self.alpha_inv) if r % 2 == 0 else state.apply_map(lambda x: x**self.alpha)
            key_injection = Mc * key_injection + c
            state = M * state + key_injection
            result.append(list(state))

        return result

    def _l0(self) -> int:
        """The maximal number of rounds that can be generically attacked"""
        # Following Table 1 in https://eprint.iacr.org/2019/426.pdf
        R_differential = ceil((2 * self.kappa) / ((self.t + 1) * floor(log(self.p / (self.alpha - 1),2))))
        R_interpol = 3
        return max(R_differential, R_interpol)

    def _l1(self) -> int:
        """Instance-specific number of rounds that can be attacked by a Gröbner basis attack"""
        # Following Equation (9) in https://eprint.iacr.org/2019/426.pdf
        nvar = lambda r : self.t * r + self.d # number of variables/equations
        dcon = lambda r : floor(0.5 * (self.alpha - 1) * self.t * r + 2) # extrapolation for observed solving degree
        R = 1
        while gb_comp(dreg=dcon(R), nv=nvar(R), w=2) < self.kappa:
            R += 1
        return R

    def _init_rounds(self) -> int:
        """Round number derivation, including 100% security margin"""
        return 2 * ceil(max(5, self._l0(), self._l1()))

# ---------------------------------------------------------------------------
# Rescue Prime
# ---------------------------------------------------------------------------

class RescuePrimeParams(RescueParams):
    """Same parameters as Rescue, with some simplifications:
        - round constants: instead of deriving constants through the block cipher's key schedule with
        the zero key, constants are generated directly by expanding a seed string with SHAKE-256
        - security margin: reduced from 100% to 50%
    """
    LABEL = "Rescue-XLIX"
    
    def _l1(self) -> int:
        """Instance-specific number of rounds that can be attacked by a Gröbner basis attack"""
        nvar = lambda r : self.t * (r-1) + self.d # number of variables/equations
        dcon = lambda r : floor(0.5 * (self.alpha - 1) * self.t * (r - 1) + 2) # extrapolation for observed solving degree
        R = 1
        while gb_comp(dreg=dcon(R), nv=nvar(R), w=2) < self.kappa:
            R += 1
        return R

    def _init_rounds(self) -> int:
        """Round number derivation, including 50% security margin"""
        # Rescue Prime: 50% security margin over the Groebner-basis bound.
        return ceil(1.5 * max(5, self._l1()))

    def _init_mds(self) -> list[list[int]]:
        return vandermonde_mds_matrix(self.p, self.t, self.g, transpose=True)

    def _init_rcons(self) -> list[list[int]]:
        seed = f"{self.LABEL}({self.p},{self.t},{self.c},{self.kappa})".encode("ascii")
        return XOFFieldElementSampler(seed=seed, p=self.p, xof="shake_256", sampling="mod").grid(2 * self.R, self.t)

# ---------------------------------------------------------------------------
# Rescue Prime Optimized (RPO)
# ---------------------------------------------------------------------------

class RescuePrimeOptimizedParams(RescuePrimeParams):
    """Same parameters as RescuePrime, with some adaptations:
        - matrix: Circulant MDS matrix instead of the Vandermonde-derived one, chosen so matrix-vector products 
        can be computed fast via Karatsuba or NTT-based polynomial multiplication (the field is NTT-friendly)
        - security margin: similar to RescuePrime, but reduced by one round
    """
    LABEL = "RPO"

    def _init_rounds(self) -> int:
        # RPO paper states that 1 round less compared to RP is fine for proposed instance
        return ceil(1.5 * max(5, self._l1())) - 1

    def _init_mds(self) -> list[list[int]]:
        return rpo_mds_matrix(self.t)

# ---------------------------------------------------------------------------
# XHASH
# ---------------------------------------------------------------------------

# TODO
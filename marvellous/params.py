# params.py
# ---------------------------------------------------------------------------
# Parameter definitions for the Marvellous family: RescueParams and its
# subclasses RescuePrimeParams and RescuePrimeOptimizedParams (RPO).
#
# Each params class is the single source of truth for an instance. It sanitizes
# user-facing parameters and expands them into a fully-specified instance that
# the permutation, hash modes, instances and tests consume. Any value the user
# omits is filled in by the matching _init_* helper. Settings that depart from 
# the recommended ones raise a ParamRecommendationWarning rather than an error.
# ---------------------------------------------------------------------------

# Structural imports
import warnings
from recommendations import ParamRecommendationWarning, recommend
from types import SimpleNamespace

# Math specific imports
from math import ceil, floor, gcd, log
from sage.all import GF, Integer, matrix, vector, flatten, PolynomialRing

# Custom imports
from utils.matrix import vandermonde_mds_matrix, map_nested, invert_matrix, circulant
from utils.sampler import XOFFieldElementSampler
from utils.mode import SpongeRescue, SpongeRPO, Sponge2
from utils.complexities import gb_comp
from utils.field import GOLDILOCKS, MERSENNE31
from utils.poly import poly_to_aos, map_coeffs, univ_from_list, power_map_coordinate_polys, diff_polys_list

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# RPO's circulant MDS rows (https://eprint.iacr.org/2022/1577, Section 3.1),
# chosen so matrix-vector products can be computed fast via Karatsuba or
# NTT-based polynomial multiplication. Reused by XHash (t = 12) and, outside
# this module, by Tip4' (tip5/params.py imports this constant).
RPO_MDS_ROWS = {
    12: [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8],
    16: [256, 2, 1073741824, 2048, 16777216, 128, 8, 16, 524288, 4194304, 1, 268435456, 1, 1024, 2, 8192],
}

# XHash's efficient MDS from a Reed-Solomon code for t = 24
# (https://hackmd.io/@sKYgEqCsSZW5mqQfCGUHvA/SkUsv8qAZg), given as a 32x32
# circulant row; the t = 24 instance uses the top-left t x t block.
XHASH_MDS_M31_T32_ROW = [
    185870542, 2144994796, 1696461115, 215190769, 930115258, 766567118, 2003379079, 1770558586,
    1779722644, 434368282, 289154277, 1979813463, 1436360233, 1342944808, 63026005, 903393155,
    1512525948, 105409451, 1072974295, 979558870, 436105640, 2126764826, 1981550821, 636196459,
    645360517, 412540024, 1649351985, 1485803845, 53244687, 719457988, 270924307, 82564914,
]

# ---------------------------------------------------------------------------
# Rescue
# ---------------------------------------------------------------------------

class RescueParams:
    LABEL = None
    SPONGE = SpongeRescue

    def __init__(
        self,
        p:         int,
        t:         int,
        alpha:     int = None,
        alpha_inv: int = None,
        R:         int = None,
        g:         int = None,
        M:         list[list[int]] = None,
        rcons:     list[list[int]] = None,
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
        t         : permutation state size
        alpha     : power-map exponent for the S-Box; smallest valid exponent via _init_alpha if not provided
        alpha_inv : power-map exponent for the inverse S-Box, i.e. alpha^{-1} mod (p-1); computed if not provided
        R         : number of rounds; computed from kappa via _init_rounds if not provided
        g         : a primitive element of GF(p) (e.g. Field.generator); _init_g if not provided
        M         : MDS matrix (txt); generated via _init_mat if not provided
        rcons     : (2*R+1)xt round-constants; generated via _init_cons if not provided
        r         : rate (number of outer state elements absorbed/squeezed per sponge step); derived if not provided
        c         : capacity (number of inner state elements for sponge); derived if not provided
        d         : digest size for generic fixed-output sponge (number of output elements); derived if not provided
        kappa     : target security level in bits (default 128)
        toy       : if True, recommendation-level checks warn instead of raising (default False)
        """
        
        # Input sanitization
        RescueParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa
        self.toy = toy

        # Sponge parameters
        if d is None and r is not None: # TODO double-check
            d = r // 2  
        self.sponge = self.SPONGE(kappa=kappa, p=p, t=t, r=r, c=c, d=d, to_field=self.to_field, toy=toy)

        # Non-linear layer
        self.alpha = alpha if alpha is not None else self._init_alpha()
        self.alpha_inv = alpha_inv if alpha_inv is not None else self._init_alpha_inv()

        # Rounds
        self.R = R if R is not None else self._init_rounds()

        # Linear layer (given matrix is trimmed to txt, i.e., first t rows and cols are taken)
        self.g = g if g is not None else self._init_g()
        self.M = map_nested([row[:self.t] for row in M[:self.t]] if M is not None else self._init_mat(), self.to_field)
        self.M_inv = invert_matrix(self.M)

        # Round constants
        self.rcons = map_nested(rcons if rcons is not None else self._init_cons(), self.to_field)

        # Check final object consistency
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
        deviations warn (ParamRecommendationWarning) but do not raise. Shared by all
        Marvellous subclasses."""

        # --- Hard checks (must always hold) ---
        if params.p == 2:
            raise NotImplementedError("Characteristic 2 not implemented")
        if params.t < 1:
            raise ValueError(f"state size t must be positive. Got {params.t}")
        if params.M != None and (len(params.M) < params.t or any(len(row) < params.t for row in params.M)):
            raise ValueError(f"provided matrix smaller than txt")

        # --- Warnings (recommended, not required) ---
        field_bits = int(params.p).bit_length()
        if field_bits < 31:
            recommend(f"TOY VERSION: field is only {field_bits} bits", params.toy)
    
    def _parameter_sanitization(self):
        """Validate the fully-constructed parameter object (stored/derived values):
        hard checks raise, recommendation deviations warn (ParamRecommendationWarning)."""

        if gcd(self.alpha, self.p - 1) != 1:
            raise ValueError("power map does not define a permutation (gcd(alpha, p-1) != 1)")

        if (len(self.M) != self.t) or not all(len(row) == self.t for row in self.M):
            raise ValueError("M must be txt matrix")

        # Every round is a double round (2 * R), plus final round constant addition
        if (len(self.rcons) < 2 * self.R + 1) or not all (len(ci) == self.t for ci in self.rcons):
            raise ValueError(f"round constants size wrong. Expected at least (2*R+1)xt = {(2*self.R+1)*self.t}, got {len(flatten(self.rcons))}")

    
    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_alpha(self) -> int:
        for alpha in range(3, self.p):
            if gcd(alpha, self.p - 1) == 1:
                return alpha

    def _init_alpha_inv(self) -> int:
        return pow(self.alpha, -1, self.p - 1)

    def _init_g(self) -> int:
        return self.F.multiplicative_generator()  # smallest primitive element

    def _init_mat(self) -> list[list[int]]:
        return vandermonde_mds_matrix(self.p, self.t, self.g, transpose=False)

    def _init_cons(self) -> list[list[int]]:
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
        R_differential = ceil((2 * self.kappa) / ((self.t + 1) * floor(log(self.p / (self.alpha - 1), 2))))
        R_interpol = 3
        return max(R_differential, R_interpol)

    def _l1(self) -> int:
        """Instance-specific number of rounds that can be attacked by a Gröbner basis attack"""
        # Following Equation (9) in https://eprint.iacr.org/2019/426.pdf
        nvar = lambda r: self.t * r + self.sponge.d  # number of variables/equations
        dcon = lambda r: floor(0.5 * (self.alpha - 1) * self.t * r + 2)  # extrapolation for observed solving degree
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
    SPONGE = SpongeRescue

    def _l1(self) -> int:
        """Instance-specific number of rounds that can be attacked by a Gröbner basis attack"""
        nvar = lambda r: self.t * (r - 1) + self.sponge.d  # number of variables/equations
        dcon = lambda r: floor(0.5 * (self.alpha - 1) * self.t * (r - 1) + 2)  # extrapolation for observed solving degree
        R = 1
        while gb_comp(dreg=dcon(R), nv=nvar(R), w=2) < self.kappa:
            R += 1
        return R

    def _init_rounds(self) -> int:
        """Round number derivation, including 50% security margin"""
        # _l0 reused from Rescue, _l1 recalculated for RescuePrime
        # 50% security margin over the Groebner-basis bound.
        return ceil(1.5 * max(5, self._l0(), self._l1()))

    def _init_mat(self) -> list[list[int]]:
        return vandermonde_mds_matrix(self.p, self.t, self.g, transpose=True)

    def _init_cons(self) -> list[list[int]]:
        seed = f"{self.LABEL}({self.p},{self.t},{self.sponge.c},{self.kappa})".encode("ascii")
        return XOFFieldElementSampler(seed=seed, p=self.p, xof="shake_256", sampling="mod").grid(2 * self.R, self.t)

    def _parameter_sanitization(self):
        """Validate the fully-constructed parameter object (stored/derived values):
        hard checks raise, recommendation deviations warn (ParamRecommendationWarning)."""

        if gcd(self.alpha, self.p - 1) != 1:
            raise ValueError("power map does not define a permutation (gcd(alpha, p-1) != 1)")

        if (len(self.M) != self.t) or not all(len(row) == self.t for row in self.M):
            raise ValueError("M must be txt matrix")

        # Every round is a double round (2 * R), no final round constant addition
        if (len(self.rcons) < 2 * self.R) or not all (len(ci) == self.t for ci in self.rcons):
            raise ValueError(f"round constants size wrong. Expected at least (2*R)xt = {(2*self.R)*self.t}, got {len(flatten(self.rcons))}")

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
    SPONGE = SpongeRPO

    def _init_rounds(self) -> int:
        # _l0 reused from Rescue, _l1 reused from RescuePrime
        # RPO paper states that 1 round less compared to RP is fine for proposed instance
        # in XHash (https://eprint.iacr.org/2023/1045.pdf, Section 4.5) authors explicitly mention floor
        return floor(1.5 * max(5, self._l0(), self._l1()))

    def _init_mat(self) -> list[list[int]]:
        """The pinned RPO circulant (module constant RPO_MDS_ROWS, t in {12, 16})."""
        if self.t not in RPO_MDS_ROWS:
            raise NotImplementedError(f"Error: Not implemented -- no RPO MDS row for t={self.t} (only t in {sorted(RPO_MDS_ROWS)})")
        return circulant(row=RPO_MDS_ROWS[self.t])

# ---------------------------------------------------------------------------
# XHASH
# ---------------------------------------------------------------------------

class XHashParams(RescuePrimeOptimizedParams):
    """Same parameters as RescuePrimeOptimized, with some adaptations:
        - matrix: for t=24, use efficient MDS matrices from Reed-Solomon codes 
          (see https://hackmd.io/@sKYgEqCsSZW5mqQfCGUHvA/SkUsv8qAZg) TODO add to matrix derivation strategies
    """
    LABEL = "RPO" # Same label for rcons derivation as for RPO
    SPONGE = Sponge2

    def __init__(
        self, 
        cpolys:  list[list[tuple[int, tuple[int]]]] = None, 
        fmod:    list[int] = None, 
        skipbox: list[int] = None, 
        **kwargs
    ):
        """
        Additional parameters
        ----------
        cpolys  : coordinate polynomials in Fp[a,b,c] describing power map in Fp[x]/fmod; derived if not given
                  cpolys[i] is the i-th coordinate polynomial, given as a list of terms tuple[int, tuple[int]],
                  where each term-tuple stores the coefficient and the exponent tuple. AoS format, see utils.poly.
        fmod    : coefficients (non-sparse) of degree 3 irreducible polynomial in Fp[x] used for field extension
        skipbox : for aggressive versions [mod,rem] such that forward S-Box i is skipped whenever i % mod = rem
        kwargs  : arguments passed to parent class
        """
        super().__init__(**kwargs)
        self._init_sbox_P3(cpolys, fmod)
        self.skipbox = skipbox
        self.skipbox_idx = [] if self.skipbox is None else [i for i in range(self.t) if i % self.skipbox[0] == self.skipbox[1]]

    @property
    def n_rcons(self) -> int:
        """Number of round-constant rows the permutation consumes.

        Every other round is a double SPN round, so the permutation indexes constants over
        range(int(1.5*R)) plus one final row (index -1) used by _post_rounds: int(1.5*R) + 1.
        """
        return int(1.5 * self.R) + 1

    def _parameter_sanitization(self):
        if gcd(self.alpha, self.p - 1) != 1:
            raise ValueError("power map does not define a permutation (gcd(alpha, p-1) != 1)")

        if (len(self.M) != self.t) or not all(len(row) == self.t for row in self.M):
            raise ValueError("M must be txt matrix")

        # At least n_rcons rows are consumed by the permutation (extra supplied rows are allowed).
        if (len(self.rcons) < self.n_rcons) or not all(len(ci) == self.t for ci in self.rcons):
            raise ValueError(f"round constants size wrong. Expected at least n_rcons x t = {self.n_rcons} x {self.t}, got {len(self.rcons)} x t")

        # If not XHash12 of XHash24, undocumented variant
        if not ((self.p == GOLDILOCKS.p and self.t == 12) or (self.p == MERSENNE31.p and self.t == 24)):
            msg = f"TOY VERSION: This is not an officially recommended version"
            recommend(msg, self.toy)

    def _init_mat(self) -> list[list[int]]:
        """The pinned matrices: RPO's circulant for t = 12 (RPO_MDS_ROWS) and the
        Reed-Solomon 32x32 circulant's top-left t x t block for t = 24
        (XHASH_MDS_M31_T32_ROW), matching the truncation the provided-M path
        applies via map_nested."""
        if self.t == 12:
            return circulant(row=RPO_MDS_ROWS[12])
        if self.t == 24:
            full = circulant(row=XHASH_MDS_M31_T32_ROW)
            return [row[:self.t] for row in full[:self.t]]
        raise NotImplementedError(f"Error: Not implemented -- no matrix derivation strategy for t = {self.t}")

    def _init_cons(self) -> list[list[int]]:
        # Same seed scheme as RPO, but generate exactly n_rcons rows (driven by the round
        # structure) rather than the parent's 2*R.
        seed = f"{self.LABEL}({self.p},{self.t},{self.sponge.c},{self.kappa})".encode("ascii")
        return XOFFieldElementSampler(seed=seed, p=self.p, xof="shake_256", sampling="mod").grid(self.n_rcons, self.t)

    def _init_rounds(self) -> int:
        """Round number derivation, including 50% security margin"""
        # One round less than RPO
        return super()._init_rounds() - 1

    def _init_sbox_P3(self, cpolys: list, fmod: list):
        if fmod is None and cpolys is None:
            raise ValueError("must supply either fmod or cpolys")

        # Derive cpolys from fmod, if given
        cpolys_derived = None
        if fmod is not None:
            R = PolynomialRing(self.F, 'x')
            f = univ_from_list(R.gen(), fmod)
            if not f.is_irreducible() or f.degree() != 3:
                raise ValueError(f"fmod = {f} is not irreducible of degree 3 over {self.F}")
            Fn = self.F.extension(f, name='X')
            cpolys_derived = [poly_to_aos(poly) for poly in power_map_coordinate_polys(Fn, self.alpha)]
        
        # Map coefficients of cpolys into field, if given
        cpolys_given = None
        if cpolys is not None:
            if len(cpolys) != 3:
                raise ValueError(f"Wrong number of coordinate polynomials. Expected len(cpolys) = 3, got {len(cpolys)}")
            if not all(len(exps) == 3 for poly in cpolys for coeff, exps in poly):
                raise ValueError("All exponent tuples must have length 3")
            cpolys_given = [map_coeffs(p, self.to_field) for p in cpolys]
        
        # Set coordinate polynomials
        if cpolys_derived is not None and cpolys_given is not None:
            # both supplied -> validate agreement
            n_mondiff, n_cdiff = diff_polys_list(cpolys_derived, cpolys_given)
            if (n_mondiff, n_cdiff) != (0, 0):
                warnings.warn("supplied cpolys do not match those derived from fmod; using fmod-derived cpolys", ParamRecommendationWarning, stacklevel=2)
            self.cpolys = cpolys_derived
            self.fmod = fmod
        elif cpolys_given is not None:
            # only cpolys given, no fmod: trust them, but we cannot certify a modulus
            self.cpolys = cpolys_given
            self.fmod = None
        else: 
            # only fmod was given: use derived cpolys and store the modulus
            self.cpolys = cpolys_derived
            self.fmod = fmod
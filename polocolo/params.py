# params.py
# PolocoloParams: the fully-specified parameter set for Polocolo (single source of truth per instance).

# Structural imports
import warnings
from recommendations import ParamRecommendationWarning, recommend
from types import SimpleNamespace

# Math specific imports
from sage.all import GF, Integer
from math import log2

# Custom imports
from utils.lut import power_residue_sigma, power_residue_lut, power_residue_lut_inv
from utils.matrix import map_nested, invert_matrix, circulant, dl_m44_84_matrix, low_addition_mds_matrix
from utils.sampler import XOFFieldElementSampler
from utils.complexities import uni_solve_comp
from utils.field import BLS12_381_SCALAR, BN254_SCALAR, find_smallest_generator

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Recommended power-residue order m per state size t (Table 1) and the tight
# variant without security margin (Table 7).
RECOMMENDED_M = {3: 1024, 4: 512, 5: 128, 6: 64, 7: 32, 8: 32}
TIGHT_M = {3: 1024, 4: 1024, 6: 64, 8: 32}

# MDS matrices of the linear layer (Appendix A.1): chosen to minimize the number
# of two-term addition gates in a Plonk circuit (Table 4: 5, 8, 13, 17, 24, 31
# additions for t = 3..8). The t = 3, 4 matrices are the low-addition MDS
# matrices of Duval & Leurent (ToSC 2018(2)); the t = 5..8 matrices were found
# with the randomized search of Appendix A.2 (utils.matrix.low_addition_mds_matrix)
# and are pinned here since the search is not seeded. All entries are small
# integers, so the same matrices serve every field.
MDS = {
    3: circulant([2, 1, 1]),
    4: dl_m44_84_matrix(2),
    5: [[    39,     6,    10,    28,     8],
        [   174,    28,    32,    80,    16],
        [   348,    58,    42,    84,     2],
        [    39,     4,    54,   100,    44],
        [   204,    20,   300,   560,   244]],
    6: [[  1011,  1470,    42,   140,   508,  1700],
        [   232,    70,    48,    48,   264,  1280],
        [  4227,  7371,     3,   490,  1420,  2900],
        [  6744, 11760,    60,   844,  2272,  4670],
        [ 13281, 23163,     9,  1540,  4460,  9100],
        [    48,    84,    12,    35,    40,   200]],
    7: [[    3538,     3090,   768,    480,     720,  96, 336],
        [  470862,   470750,  1120,  16380,   94284, 136, 924],
        [10112885, 10113269, 24960, 352496, 2023200, 768, 18048],
        [ 3799380,  3799524,  9024, 132256,  760128, 288, 6783],
        [   94120,    94080,   232,   3276,   18816,   5, 198],
        [ 1357780,  1357788,  3240,  47268,  271632, 101, 2454],
        [  270260,   270260,   640,   9402,   54108,  64, 480]],
    8: [[  3840,   24,  4728, 2952, 258912,  99840,  94222,  74400],
        [  1386,   78,   280, 1218,  32256,  13044,   8120,   6496],
        [  6180,  743, 10416, 4428, 508032, 194858, 193984, 153056],
        [   432,  400,  1920,  144,  73728,  27776,  30400,  23936],
        [ 10122, 1246,  5320, 8526, 346752, 136724, 108570,  86184],
        [   950, 1052,  5424,  240, 202944,  76333,  84683,  66656],
        [  2564,   16,  3072, 1920, 172128,  66380,  62528,  49408],
        [   661,   35,   908,  585,  43008,  16448,  14512,  11456]],
}

# Field labeld for rcons generation: the official fields have a fixed label, any other field falls back to the decimal characteristic.
FIELD_LABELS = {
    BLS12_381_SCALAR.p: "BLS12",
    BN254_SCALAR.p: "BN254",
}

# Largest admissible log2(m) when deriving m: the paper caps the lookup-table
# size at m = 1024 (the largest official m, Table 1). Without this cap the joint
# (min R, then min m) selection would run off to ever larger m, since a larger m
# only ever lowers the algebraic round bound.
M_MAX_LOG = 10

# ---------------------------------------------------------------------------
# Parameter definition
# ---------------------------------------------------------------------------
class PolocoloParams:
    def __init__(
        self,
        p:           int,
        t:           int,
        m:           int = None,
        R:           int = None,
        sigma:       list[int] = None,
        M:           list[list[int]] = None,
        rcons:       list[list[int]] = None,
        g:           int = None,
        field_label: str = None,
        # Modes of operation: sponge params dict dict(r,c,d) (Polocolo is sponge-only).
        sponge:      dict = None,
        comp:        dict = None,
        # Target security level (default 128 bits)
        kappa:       int = 128,
        tight:       bool = False,
        toy:         bool = False,
    ):
        """
        Parameters
        ----------
        p           : field characteristic (prime); officially the BLS12-381 or BN254 scalar field
        t           : state size (number of field elements), officially 3 <= t <= 8
        m           : power-residue order, m | p-1 (a power of two in the official instances);
                      the recommended value for (t, tight) from Table 1 / Table 7 if not provided
        R           : number of rounds; derived from the guessing-power-residue attack bound if
                      not provided (reproduces Table 1 / Table 7)
        sigma       : S-box permutation of {0, ..., m-1}; derived via _init_sigma if not provided
        M           : t x t MDS matrix of the linear layer; the published matrix for t if not provided
        rcons       : R x t round constants c^(0), ..., c^(R-1) (c^(R) = 0 is fixed by the design
                      and not stored); generated via _init_cons (SHAKE128) if not provided
        g           : generator of F_p^*; smallest one if not provided
        field_label : field name in the round-constant seed ("BLS12" / "BN254" for the official
                      fields); derived from p if not provided
        sponge: sponge params dict dict(r, c, d), or None
        comp  : compression params dict, or None (Polocolo is sponge-only)
        kappa       : target security level in bits (default 128)
        tight       : use the tight parameters of Section 6.1 / Table 7 (no security margin,
                      attack bound kappa instead of 1.25*kappa) for the derived m and R
        toy         : if True, recommendation-level checks warn instead of raising (default False)
        """

        # Input sanitization
        PolocoloParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa
        self.tight = tight
        self.toy = toy

        # Modes of operation: per-mode param dicts (consumed by the mode functions).
        self.sponge, self.comp = sponge, comp

        # Non-linear layer: the power-residue S-box S(x) = x^{-1} * T[x^ann] with
        # ann = (p-1)/m, realised as the lookup tables LUT / LUT_inv over sigma.
        # "annihilator" exponent (p-1)/m: raising to it annihilates the subgroup of m-th
        # powers {g^(qm)} -> 1, leaving only the residue-class part g^(r(p-1)/m); i.e.
        # x^ann is the m-th power residue (x/p)_m of Eq. (1).
        self.g = self.to_field(g) if g is not None else self.F.multiplicative_generator()
        self.m = m if m is not None else self._init_m()
        self.ann = (p - 1) // self.m
        self.sigma = list(sigma) if sigma is not None else self._init_sigma()
        self.LUT = {k: self.to_field(v) for k, v in power_residue_lut(p, self.g, self.m, self.sigma).items()}
        self.LUT_inv = {k: self.to_field(v) for k, v in power_residue_lut_inv(p, self.g, self.m, self.sigma).items()}

        # Round number (needs m; set before _init_cons, whose seed contains R)
        self.R = R if R is not None else self._init_rounds()

        # Linear layer
        self.M = map_nested(M if M is not None else self._init_mat(), self.to_field)
        self.M_inv = invert_matrix(self.M)

        # Round constants
        self.field_label = field_label if field_label is not None else self._init_field_label()
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
        m = params.m if params.m is not None else (len(params.sigma) if params.sigma is not None else None)
        if m is not None:
            if m < 2 or (params.p - 1) % m != 0:
                raise ValueError(f"m must divide p-1 (with m >= 2), otherwise the m-th power residue is not well-defined. Got m={m}")
            if params.sigma is not None and sorted(params.sigma) != list(range(m)):
                raise ValueError(f"sigma must be a permutation of range({m})")
        if params.M is not None and not (len(params.M) == params.t and all(len(row) == params.t for row in params.M)):
            raise ValueError(f"M must be a {params.t}x{params.t} matrix")
        if params.rcons is not None and params.R is not None and \
                not (len(params.rcons) == params.R and all(len(row) == params.t for row in params.rcons)):
            raise ValueError(f"rcons must be an {params.R}x{params.t} grid (c^(R) = 0 is implicit)")

        # --- Warnings (recommended, not required) ---
        field_bits = int(params.p).bit_length()
        if field_bits < 31:
            recommend(f"TOY VERSION: field is only {field_bits} bits", params.toy)
        if params.p not in (BLS12_381_SCALAR.p, BN254_SCALAR.p):
            warnings.warn("Polocolo is specified and analyzed for the BLS12-381 and BN254 scalar fields only", ParamRecommendationWarning, stacklevel=2)
        if not 3 <= params.t <= 8:
            warnings.warn(f"official instances use 3 <= t <= 8. Got t={params.t}", ParamRecommendationWarning, stacklevel=2)
        if m is not None and (m & (m - 1)) != 0:
            warnings.warn(f"official instances use a power of two for m (FFT/Plonk friendliness). Got m={m}", ParamRecommendationWarning, stacklevel=2)
        recommended = (TIGHT_M if params.tight else RECOMMENDED_M).get(params.t)
        if m is not None and recommended is not None and m != recommended:
            warnings.warn(f"recommended m for t={params.t} ({'tight' if params.tight else 'with margin'}) is {recommended}. Got m={m}", ParamRecommendationWarning, stacklevel=2)
        if params.kappa != 128:
            warnings.warn(f"official instances target kappa = 128. Got kappa={params.kappa}", ParamRecommendationWarning, stacklevel=2)

    def _parameter_sanitization(self):
        """Validate the fully-constructed parameter object (stored/derived values):
        hard checks raise, recommendation deviations warn (ParamRecommendationWarning)."""

        # --- Hard checks (must always hold) ---
        if len(self.M) != self.t or any(len(row) != self.t for row in self.M):
            raise ValueError(f"M must be a {self.t} x {self.t} matrix")
        if len(self.rcons) != self.R or any(len(row) != self.t for row in self.rcons):
            raise ValueError(f"rcons must be an {self.R} x {self.t} grid (c^(R) = 0 is implicit)")
        if sorted(self.sigma) != list(range(self.m)):
            raise ValueError(f"sigma must be a permutation of range({self.m})")

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_field_label(self) -> str:
        """Field label for official instances. Fallback to the decimal characteristic."""
        return FIELD_LABELS.get(self.p, str(self.p))

    def _init_m(self) -> int:
        """The recommended power-residue order for (t, tight) from Table 1 / Table 7;
        for state sizes outside the tables, warn and fall back to the derivation
        strategy (_derive_m) the tables were produced with."""

        table = TIGHT_M if self.tight else RECOMMENDED_M
        if self.t in table:
            return table[self.t]

        warnings.warn(f"no recommended m for t={self.t} ({'tight' if self.tight else 'with margin'}); "
                      "deriving m from the round-number bounds", ParamRecommendationWarning, stacklevel=2)
        return self._derive_m()

    def _derive_m(self) -> int:
        """The paper's selection strategy behind Tables 1/7: among the powers of two
        m <= 2^M_MAX_LOG dividing p-1 (the power residue needs m | p-1; the official
        instances use powers of two for FFT/Plonk friendliness), pick the one whose
        derived round number is minimal, and among those the smallest m (joint
        (min R, then min m) selection)."""
        candidates = [1 << l for l in range(1, M_MAX_LOG + 1) if (self.p - 1) % (1 << l) == 0]
        if not candidates:
            raise ValueError("p-1 has insufficient 2-adicity for a power-residue table")
        return min(candidates, key=lambda m: (self._init_rounds(m), m))

    def _init_sigma(self) -> list[int]:
        """The S-box permutation sigma, exactly as the tables shipped with the reference implementation 
        (see power_residue_sigma for the two derivation variants and their discrepancy). The interpolation
        conditions the paper imposes on sigma (Section 4.2) hold for every official instance; 
        they are asserted in the test suite rather than re-checked here, since the Lagrange interpolations 
        are expensive at m = 1024."""
        return power_residue_sigma(self.m, self.p, self.g, seed=f"Polocolo-{self.m}", method="shuffle")

    def _init_mat(self) -> list[list[int]]:
        """The published low-addition MDS matrix for t (module constant MDS). For other
        state sizes, fall back to the randomized search the authors used (Appendix A.2):
        start with t gates of slack over the structural minimum and relax the budget by
        t gates whenever the search comes up empty (the success probability at a tight
        budget drops sharply with t) -- the result is a fresh matrix, not a published
        one, hence the warning."""
        if self.t in MDS:
            return MDS[self.t]
        warnings.warn(f"no published Polocolo matrix for t={self.t}; running the randomized low-addition MDS search (Appendix A.2)", ParamRecommendationWarning, stacklevel=2)
        additions = 2 * self.t + (self.t + 1) // 2 + self.t
        while True:
            try:
                return low_addition_mds_matrix(self.t, additions)
            except RuntimeError:
                additions += self.t

    def _init_cons(self) -> list[list[int]]:
        """The R x t round constants c^(0), ..., c^(R-1), drawn from a SHAKE128 stream
        seeded with "Polocolo-{m}-{R}-{t}-{field}" exactly as the reference
        param_gen.sage (big-endian draws whose top byte is right-SHIFTED to the field's
        bit length, with rejection sampling -- the "bitshift" sampler mode). The final
        linear layer has no constant (c^(R) = 0), so no row is drawn for it."""
        seed = f"Polocolo-{self.m}-{self.R}-{self.t}-{self.field_label}".encode()
        sampler = XOFFieldElementSampler(seed=seed, p=self.p, xof="shake_128", sampling="bitshift", endianess="big")
        return sampler.grid(self.R, self.t)

    # ---------------------------------------------------------------------------
    # Security analysis helpers (Section 5)
    # ---------------------------------------------------------------------------
    def _differential_comp(self, R: int, m: int = None) -> float:
        """log2 of the upper bound on any R-round differential trail probability
        (Section 5.1, Lemma 4 + MDS branch number 1+t):
            ((2m^2+2)/p)^((1+t)*floor(R/2)).
        Security against classical differential attacks requires this <= -2.5*kappa
        (2.5 = clustering factor). Also covers the boomerang attack, whose success
        probability is bounded by the cube of the single-round bound. `m` defaults to
        the instance parameter (override for use during _init_m)."""
        if m is None:
            m = self.m
        return (1 + self.t) * (R // 2) * (log2(2 * m * m + 2) - log2(self.p))

    def _linear_comp(self, R: int, m: int = None) -> float:
        """log2 of the upper bound on the R-round linear probability (Section 5.1,
        resting on Conjecture 1, the 2*sqrt(p) Kloosterman-sum bound):
            (16m^2/p)^((1+t)*floor(R/2)).
        Security against linear cryptanalysis requires this <= -kappa."""
        if m is None:
            m = self.m
        return (1 + self.t) * (R // 2) * (log2(16 * m * m) - log2(self.p))

    def _rounds_stat(self, m: int = None) -> int:
        """Round-number floor from statistical attacks (Section 5.1): the minimal R
        satisfying the differential and linear conditions above, maxed with 4 -- the
        structural rebound/truncated-differential floor (each rebound outbound phase
        spans <= 1 round under MDS diffusion, so 4 rounds kill it; prime-independent
        only while the single-round differential bound stays negligible). On the
        official ~254-bit fields both formula minima are 2 and the 4 binds, matching
        the paper. Recommended parameters add 1 round of margin; tight ones none."""
        if m is None:
            m = self.m
        if log2(2 * m * m + 2) >= log2(self.p):
            raise ValueError("m^2 not small relative to p: the single-round bounds are vacuous and no statistical floor can be derived")
        R = 2
        while (self._differential_comp(R, m) > -2.5 * self.kappa or self._linear_comp(R, m) > -self.kappa):
            R += 2          # bounds improve only per pair of rounds (floor(R/2))
        floor = max(4, R)   # rebound structural minimum
        if floor > 4:
            warnings.warn("statistical floor > 4: outside the paper's analyzed regime (small-prime instantiation is an open problem there)", stacklevel=2)
        return floor if self.tight else floor + 1

    def _guessing_power_residue_comp(self, R: int, m: int = None) -> float:
        """log2 complexity of the guessing power residue attack (Section 5.2, Eq. 3)
        against R rounds: guess the m possible power residues of every S-box except
        those bypassed in the first round and via the CICO relation (m^(t(R-1)) guesses,
        Table 5), then solve the resulting univariate equation of degree t^(R-1)."""
        if m is None:
            m = self.m
        return self.t * (R - 1) * log2(m) + uni_solve_comp(self.t ** (R - 1), self.p)

    def _rounds_alg(self, m: int = None) -> int:
        """Round-number minimum from algebraic attacks (Section 5.2): the minimal R
        such that the guessing power residue attack -- the binding algebraic attack,
        dominating the Groebner-basis routes (Table 6) -- costs at least the target:
        2^(1.25*kappa) for the recommended parameters (the paper's 2^160 at kappa=128;
        the 1.25 factor is our extrapolation for other kappa) and 2^kappa for the
        tight parameters of Section 6.1. `m` defaults to the instance parameter
        (override for use during _init_m). Starts at R = 2: the search needs no floor
        of its own -- statistical minima and margin live in _rounds_stat."""
        if m is None:
            m = self.m
        target = self.kappa if self.tight else 1.25 * self.kappa
        R = 2
        while self._guessing_power_residue_comp(R, m) < target:
            R += 1
        return R

    def _init_rounds(self, m: int = None) -> int:
        """Round number of the instance (Section 5): the maximum of the statistical
        floor (incl. the recommended parameters' 1 round of margin; _rounds_stat) and
        the algebraic minimum (_rounds_alg). On the official instances the algebraic
        bound binds everywhere except recommended t=8, where both give 5. Reproduces
        the R columns of Table 1 and Table 7."""
        return max(self._rounds_stat(m), self._rounds_alg(m))
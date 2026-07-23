# params.py
# ---------------------------------------------------------------------------
# Parameter definition for Tip5 (and its TIP4 / TIP4' variants): the Tip5Params class.
#
# Tip5Params is the single source of truth for an instance. It sanitizes 
# user-facing parameters and expands them into a fully-specified instance that
# the permutation, hash modes, instances and tests consume. Any value the user
# omits is filled in by the matching _init_* helper. Settings that depart from 
# the recommended ones raise a ParamRecommendationWarning rather than an error.
#
# Tip4Params and Tip4Prime params implement variants detailed here: 
# https://toposware.com/paper_tip5.pdf
# ---------------------------------------------------------------------------

# Structural imports
from recommendations import recommend
from types import SimpleNamespace

# Math specific imports
from math import gcd
from sage.all import GF, Integer

# Custom imports
from utils.lut import invert_LUT
from utils.sampler import XOFFieldElementSampler
from utils.mode import SpongeCLE
from utils.matrix import map_nested, invert_matrix, circulant
from utils.field import GOLDILOCKS
from marvellous.params import RPO_MDS_ROWS   # Tip4' reuses RPO's circulant MDS

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# First column of Tip5's 16x16 circulant MDS matrix: the 16 little-endian 16-bit
# words of SHA-256("Tip5"), both fixed by the spec, drawn as a grid (n_bytes=2)
# from the XOF sampler. Entries are 16-bit by design (deliberately narrower than
# the field, enabling delayed modular reduction), so no rejection occurs and the
# column is identical over every field with p > 2^16 -- it is computed once here
# over Goldilocks. Reduced state sizes t < 16 (e.g. Tip4' with t = 12) use the
# first t entries.
TIP5_MDS_COLUMN = XOFFieldElementSampler(seed=b"Tip5", p=GOLDILOCKS.p, xof="sha256", sampling="naive", n_bytes=2).grid(1, 16)[0]

# Byte decomposition of a 64-bit value: the mixed-radix bases of the
# split-and-lookup S-box.
TIP5_SI = [256] * 8


class Tip5Params:
    LABEL = "Tip5" # label for deriving rcons

    def __init__(
        self,
        p:         int = GOLDILOCKS.p,
        t:         int = 16,
        R:         int = None,
        u:         int = 4,
        M:         list[list[int]] = None,
        alpha:     int = GOLDILOCKS.alpha,
        alpha_inv: int = GOLDILOCKS.alpha_inv,
        LUT:       list[int] = None,
        rcons:     list[list[int]] = None,
        # Sponge parameters (derived if not provided)
        r:          int = 10,
        c:          int = 6,
        d:          int = 5,
        # Target security level (default 128 bits)
        kappa:      int = 160,
        toy:        bool = False,
    ):
        """
        Parameters
        ----------
        p         : field characteristic (prime, ~64 bits)
        t         : permutation state size
        R         : number of rounds; the spec-fixed 5 via _init_rounds if not provided
        u         : number of decomposition S-boxes in the nonlinear layer (applied to the first u state
                    elements; the remaining t-u elements go through the power map)
        M         : circulant MDS matrix (txt); generated via _init_mat if not provided
        alpha     : power map exponent, coprime with p-1; smallest valid exponent via _init_alpha if not provided
        alpha_inv : alpha^{-1} mod (p-1); computed via _init_alpha_inv if not provided
        LUT       : 256-entry lookup table for the split-and-lookup S-box; generated via _init_LUT if not provided
        rcons     : Rxt round constants; generated via Blake3 (_init_cons) if not provided
        r         : rate (number of outer state elements absorbed/squeezed per sponge step); derived if not provided
        c         : capacity (number of inner state elements for sponge); derived if not provided
        d         : digest size for generic fixed-output sponge (number of output elements); derived if not provided
        kappa     : target security level in bits (default 128)
        toy       : if True, recommendation-level checks warn instead of raising (default False)
        """

        # Input sanitization
        Tip5Params._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa
        self.toy = toy

        # Sponge parameters
        self.sponge = SpongeCLE(kappa=kappa, p=p, t=t, r=r, c=c, d=d, to_field=self.to_field, toy=toy)

        # Montgomery constant
        self.mont_R = self.to_field(2**64) # Montgomery constant
        self.mont_R_inv = self.mont_R ** (-1)

        # Non-linear layer: split-and-lookup S-boxes (S) and power maps (T)
        self.u = u
        self.si = TIP5_SI
        self.LUT = LUT if LUT is not None else self._init_LUT()
        self.LUT_inv = invert_LUT(self.LUT)
        self.alpha = alpha if alpha is not None else self._init_alpha()
        self.alpha_inv = alpha_inv if alpha_inv is not None else self._init_alpha_inv()

        # Rounds (set before _init_cons, whose derivation depends on R)
        self.R = R if R is not None else self._init_rounds()

        # Affine layer
        self.M = map_nested(M if M is not None else self._init_mat(), self.to_field)
        self.M_inv = invert_matrix(self.M)

        # Round constants
        self.rcons = map_nested(rcons if rcons is not None else self._init_cons(), self.to_field)

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
        if params.p.bit_length() != 64:
            raise ValueError("Tip5 is defined over ~64-bit fields")
        if params.t < 1:
            raise ValueError(f"state size t must be positive. Got {params.t}")
        if params.alpha is not None and gcd(params.alpha, params.p - 1) != 1:
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
            raise ValueError(f"rcons must be an {self.R} x {self.t} grid (one t-vector per round)")
        if len(self.LUT) != 256:
            raise ValueError(f"LUT must have 256 entries. Got {len(self.LUT)}")

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_rounds(self) -> int:
        """The round number is fixed by the spec: 5 rounds for Tip5 and its Tip4 /
        Tip4' variants (https://eprint.iacr.org/2023/107, Section 3)."""
        return 5

    def _init_alpha(self) -> int:
        for alpha in range(3, self.p):
            if gcd(alpha, self.p - 1) == 1:
                return alpha

    def _init_alpha_inv(self) -> int:
        return pow(self.alpha, -1, self.p - 1)

    @staticmethod
    def _init_LUT() -> list[int]:
        """Tip5 lookup map L : F_257 -> F_257, x -> (x + 1)^3 - 1.
        Restricts to a permutation on {0, ..., 255} since the only
        element outside 8 bits, 256 = -1 mod 257, is a fixed point."""
        return [(pow(x + 1, 3, 257) - 1) % 257 for x in range(256)]

    def _init_mat(self) -> list[list[int]]:
        """The Tip5 circulant built from the spec-fixed SHA-256("Tip5") column
        (module constant TIP5_MDS_COLUMN), truncated to the first t entries for
        reduced state sizes."""
        # TODO check whether the Monolith/RPO Goldilocks T12 row (RPO_MDS_ROWS[12])
        # should be used instead (KATs were generated with the truncated Tip5 column, matching the sage reference).
        if not 1 <= self.t <= 16:
            raise ValueError(f"t must be in 1..16. Got {self.t}")
        return circulant(col=TIP5_MDS_COLUMN[:self.t])

    def _init_cons(self) -> list[list[int]]:
        """Rxt round constants: constant j is sampled from a fresh Blake3 XOF seeded with
        "Tip5" || byte(j), reading t bytes little-endian and reducing mod p, then scaled
        by 2^-64 so that adding them in Montgomery form is cheap.
        Matches round constants of Rust reference implementation from
        https://github.com/Neptune-Crypto/twenty-first."""
        mont_R_inv = self.from_field(self.mont_R_inv)
        label = self.LABEL.encode("ascii")
        rcons = []
        for r in range(self.R):
            row = []
            for i in range(self.t):
                sampler = XOFFieldElementSampler(seed=label + bytes([i + r * self.t]), p=self.p, xof="blake3", sampling="mod", n_bytes=self.t)
                row.append((sampler.next() * mont_R_inv) % self.p)
            rcons.append(row)
        return rcons


class Tip4Params(Tip5Params):
    LABEL = "Tip4" # label for deriving rcons

    def __init__(self, **kwargs):
        defaults = dict(t=16, r=12, c=4, d=4, kappa=128) # include deviations from Tip5 defaults
        super().__init__(**{**defaults, **kwargs})

class Tip4PrimeParams(Tip4Params):
    LABEL = "Tip4'" # label for deriving rcons

    def __init__(self, **kwargs):
        defaults = dict(t=12, r=8, c=4, d=4, kappa=128) # include deviations from Tip4 defaults
        super().__init__(**{**defaults, **kwargs})
    
    def _init_mat(self):
        """Tip4' uses RPO's circulant MDS (marvellous.params.RPO_MDS_ROWS)."""
        return circulant(row=RPO_MDS_ROWS[self.t])
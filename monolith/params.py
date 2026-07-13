# params.py
# ---------------------------------------------------------------------------
# Parameter definition for Monolith: the MonolithParams class.
#
# MonolithParams is the single source of truth for an instance. It sanitizes
# the user-facing parameters and expands them into a fully-specified instance
# that the permutation, hash modes, instances and tests consume. Any value the
# user omits is filled in by the matching _init_* helper (or, for r/c/d, by the
# shared derive_rate_capacity_digest). Settings that depart from the recommended
# ones raise a ParamRecommendationWarning rather than an error.
# ---------------------------------------------------------------------------

# Structural imports
import struct
import warnings
from recommendations import ParamRecommendationWarning
from types import SimpleNamespace

# Math specific imports
from sage.all import GF, Integer

# Custom imports
from utils.sampler import XOFFieldElementSampler
from utils.lut import invert_LUT, invertible_phi_from_landscape, crotl, compose
from utils.matrix import map_nested, invert_matrix
from utils.mode import derive_rate_capacity_digest

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Monolith's Bar lookup tables (https://eprint.iacr.org/2023/1025, Table A.1),
# built from the generic shift-invariant chi-class constructions in utils/lut.py:
# the invertible phi for the landscape, composed with a 1-bit cyclic rotation.
# The 8-bit table ("*001") serves the 64-bit fields (and is reused by Skyscraper);
# the 7-bit table ("*01") completes the 31-bit Mersenne decomposition.
MONOLITH_LUT8 = [compose(invertible_phi_from_landscape(8, "001*", xi={3}), crotl(8, 1))(x) for x in range(1 << 8)]
MONOLITH_LUT7 = [compose(invertible_phi_from_landscape(7, "01*",  xi={2}), crotl(7, 1))(x) for x in range(1 << 7)]

# Bar tables per decomposition base si.
MONOLITH_LUTS = {2**8: MONOLITH_LUT8, 2**7: MONOLITH_LUT7}


class MonolithParams:
    def __init__(
        self,
        p:     int,
        t:     int,
        si:    list[int],
        u:     int,
        R:     int = None,
        LUTs:  dict[int, list[int]] = None,
        M:     list[list[int]] = None,
        rcons: list[list[int]] = None,
        r:     int = None,
        c:     int = None,
        d:     int = None,
        kappa: int = 128,
    ):
        """
        Parameters
        ----------
        p     : field characteristic (prime)
        t     : permutation state size
        si    : bases for decompose/compose in the Bars layer
        u     : number of decomposition S-boxes in the Bars layer
        R     : number of rounds; derived via _init_rounds if not provided
        LUTs  : per-digit lookup tables used in Bar (one per distinct value in si); generated via _init_LUTs if not provided
        M     : circulant MDS matrix (txt); generated via _init_mat if not provided
        rcons : (R-1)xt round constants; generated via _init_cons (SHAKE128) if not provided
        r     : rate (number of outer state elements absorbed/squeezed per sponge step); derived if not provided
        c     : capacity (number of inner state elements); derived if not provided
        d     : digest size (number of output elements); derived if not provided
        kappa : target security level in bits (default 128)
        """

        # Input sanitization
        MonolithParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Non-linear layers: Bars
        self.si = si
        self.LUTs = LUTs if LUTs is not None else self._init_LUTs()
        self.LUTs_inv = {s: invert_LUT(self.LUTs[s]) for s in self.LUTs}
        self.u = u

        # Hash modes
        self.r, self.c, self.d = derive_rate_capacity_digest(self.kappa, self.t, r, c, d)

        # Rounds (set before _init_cons, whose derivation depends on R)
        self.R = R if R is not None else self._init_rounds()

        # Affine layer
        self.M = map_nested(M if M is not None else self._init_mat(), self.to_field)
        self.M_inv = invert_matrix(self.M)

        # Round constants: pad with a trailing zero row so AffineLayer can uniformly index
        # rcons[round_idx] for round_idx in 0..R-1 (the final round has no round constants).
        rcons = rcons if rcons is not None else self._init_cons()
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
        if params.t < 1:
            raise ValueError(f"state size t must be positive. Got {params.t}")
        if params.M is not None and not (len(params.M) == params.t and all(len(row) == params.t for row in params.M)):
            raise ValueError(f"M must be a {params.t}x{params.t} matrix")

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
        if any(s not in self.LUTs for s in self.si):
            raise ValueError(f"LUTs must cover every base in si; missing {sorted(set(self.si) - set(self.LUTs))}")

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_LUTs(self) -> dict[int, list[int]]:
        # si entries are the radix bases (e.g. 256 / 128), mapping to the 8-bit / 7-bit
        # Bar lookup tables (module constant MONOLITH_LUTS) respectively.
        LUTs = {}
        for s in set(self.si):
            if s not in MONOLITH_LUTS:
                raise NotImplementedError(f"Error: Not implemented -- Bar LUT generation for base si={s}")
            LUTs[s] = MONOLITH_LUTS[s]
        return LUTs

    def _init_cons(self) -> list[list[int]]:
        bits = self.p.bit_length()
        seed = (b"Monolith"
                + bytes([self.t, self.R])
                + (struct.pack('<I', self.p) if bits <= 32 else struct.pack('<Q', self.p))
                + (bytes([8, 8, 8, 7])       if bits <= 32 else bytes([8] * 8)))
        return XOFFieldElementSampler(seed=seed, p=self.p, xof="shake_128", sampling="naive").grid(self.R - 1, self.t)

    def _init_rounds(self) -> int:
        """Derive the round number from the target security level kappa.
        TODO: implement the round-number criterion of the Monolith paper
        (https://eprint.iacr.org/2023/1025, Section 5); until then R must be
        passed explicitly."""
        raise NotImplementedError("Error: Not implemented -- round number derivation for Monolith")

    def _init_mat(self) -> list[list[int]]:
        """Return the t x t circulant MDS matrix.
        TODO: implement the paper's circulant construction (Goldilocks: fixed row;
        Mersenne31: SHAKE128-sampled circulant row, https://eprint.iacr.org/2023/1025
        Section 4.4); until then M must be passed explicitly."""
        raise NotImplementedError("Error: Not implemented -- MDS matrix generation for Monolith")

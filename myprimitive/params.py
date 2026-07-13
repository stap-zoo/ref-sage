# params.py
# ---------------------------------------------------------------------------
# Parameter definition for MyPrimitive: the MyPrimitiveParams class.
#
# MyPrimitiveParams is the single source of truth for an instance. It takes a
# small set of user-facing parameters and expands them into a fully-specified
# instance: it sanitizes the inputs and stores/derives parameters. Every other 
# file (permutation, hash/sponge wrapper, instances, test vectors) consumes a
# MyPrimitiveParams object.
#
# Any value the user omits is filled in by the matching _init_* helper. An
# analyst can override parameters to spin up toy or reduced instances
# for cryptanalysis without editing this file. Settings that depart from the
# recommended ones raise a ParamRecommendationWarning rather than an error.
# ---------------------------------------------------------------------------

# Structural imports
import warnings
from recommendations import ParamRecommendationWarning
from types import SimpleNamespace

# Math specific imports
from math import gcd
from sage.all import GF, Integer

# Custom imports
from utils.matrix import map_nested, invert_matrix, simple_circulant_matrix
from utils.mode import derive_rate_capacity_digest
# Add any other helpers your primitive needs, e.g.:
# from complexities import gb_comp
# from utils import circulant, XOFFieldElementSampler

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# Put primitive-specific field-independent design constants.
# Example: digits of pi used to seed round constants, precomputed MDS rows, etc.
#
# PI_0 = 1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679

# ---------------------------------------------------------------------------
# Parameter definition
# ---------------------------------------------------------------------------
class MyPrimitiveParams:
    def __init__(
        self,
        p:     int,
        t:     int,
        r:     int = None,
        c:     int = None,
        d:     int = None,
        alpha: int = None,
        R:     int = None,
        M:     list[list[int]] = None,
        rcons: list[list[int]] = None,
        kappa: int = 128,
    ):
        """
        Parameters
        ----------
        p     : field characteristic (prime)
        t     : state size (number of field elements)
        alpha : S-box exponent, coprime with p-1; smallest valid exponent if not provided
        R     : number of rounds; derived via _init_rounds() if not provided
        r     : rate (number of outer state elements absorbed/squeezed per sponge step);
                derived from kappa/t via derive_rate_capacity_digest if not provided
        c     : capacity (number of inner state elements); derived if not provided
        d     : digest size (number of output elements); derived if not provided
        M     : t x t MDS matrix for the linear layer; generated via _init_mat() if not provided
        rcons : round constants; generated via _init_cons() if not provided
        kappa : target security level in bits (default 128)
        """

        # Input sanitization
        MyPrimitiveParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Non-linear layer: any values associated to non-linear layer
        self.alpha = alpha if alpha is not None else self._init_alpha()
        self.alpha_inv = pow(self.alpha, -1, p - 1)

        # Hash modes: values specific to sponge/compression modes defined in hash.py.
        # r, c, d are always optional; any omitted value is filled in by the shared
        # derive_rate_capacity_digest helper (the same sponge/compression security relation
        # for every primitive), rather than per-primitive _init_r/_init_c/_init_d.
        self.r, self.c, self.d = derive_rate_capacity_digest(self.kappa, self.t, r, c, d)

        # Round number: given round number or derived one
        self.R = R if R is not None else self._init_rounds()

        # Linear layer: any values associated to linear layer
        # Matrices are generated/provided as list[list[int]], and transformed to list[list[FieldElement]] via map_nested
        self.M = map_nested(M if M is not None else self._init_mat(), self.to_field)
        self.M_inv = invert_matrix(self.M)

        # Round constants: any values associated to round constant addition
        # Round constants are typically generated/provided as list[int], and transformed to list[FieldElement] via map_nested
        self.rcons = map_nested(rcons if rcons is not None else self._init_cons(), self.to_field)

        # Parameter sanitization: validate the fully-constructed (stored/derived) values
        self._parameter_sanitization()

    # ---------------------------------------------------------------------------
    # Small field conversion helpers to provide common framework for all functions
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
        """Validate the raw constructor arguments.
 
        `params` is a struct holding every value passed to __init__, accessed by
        attribute (args.p, args.t, args.r, args.c, args.d, args.alpha, args.R,
        args.M, args.rcons, args.kappa).
 
        Two kinds of checks:
          * hard checks -- invariants that must always hold for the construction
                           to make sense; raise on failure.
          * warnings    -- deviations from the recommended ("official") settings
                           that the primitive can still run with (e.g. a toy
                           instance for cryptanalysis); warn but do NOT raise,
                           provided the design otherwise supports the adaptation.
        """

        # --- Hard checks (must always hold) ---
        # Add the invariants your construction genuinely requires, e.g.:
        if params.p == 2:
            raise NotImplementedError("Characteristic 2 not implemented")
        if params.t < 1:
            raise ValueError(f"state size t must be positive. Got {params.t}")
        if params.alpha is not None and gcd(params.alpha, params.p - 1) != 1:
            raise ValueError(f"power map does not define a permutation")
 
        # --- Warnings (recommended, not required) ---
        # Example: a small field is fine for a toy / cryptanalysis instance but is
        # not secure (or not analyzed) for real use, so warn instead of refusing it.
        field_bits = int(params.p).bit_length()
        if field_bits < 31:
            msg = f"TOY VERSION: field is only {field_bits} bits"
            warnings.warn(msg, ParamRecommendationWarning, stacklevel=2)

    def _parameter_sanitization(self):
        """Validate the fully-constructed parameter object (stored/derived values).

        Runs at the END of __init__, after every _init_* helper has filled in the
        missing values -- the counterpart to the pre-construction _input_sanitization.
        Same split as there:
          * hard checks -- structural invariants of the stored values (shapes,
                           counts, permutation properties); raise on failure.
          * warnings    -- derived values that deviate from the recommended
                           settings; warn with ParamRecommendationWarning.
        """

        # --- Hard checks (must always hold) ---
        if len(self.M) != self.t or any(len(row) != self.t for row in self.M):
            raise ValueError(f"M must be a {self.t} x {self.t} matrix")
        if len(self.rcons) != self.R:
            raise ValueError(f"rcons must have one row per round: expected {self.R}, got {len(self.rcons)}")
        if any(len(row) != self.t for row in self.rcons):
            raise ValueError(f"each rcons row must hold {self.t} elements")

        # --- Warnings (recommended, not required) ---
        # Example: warn if a derived value ends up outside the analyzed range.

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_alpha(self) -> int:
        """Smallest exponent >= 3 that is coprime with p-1 (i.e. an invertible S-box)."""
        for alpha in range(3, self.p):
            if gcd(alpha, self.p - 1) == 1:
                return alpha

    def _init_rounds(self) -> int:
        """Derive round numbers to resist known attacks.
        TODO: replace with your primitive's round number derivation strategy."""
        raise NotImplementedError("Error: Not implemented -- round number derivation for MyPrimitive")

    def _init_mat(self) -> list[list[int]]:
        """Return a t x t MDS matrix over F.
        TODO: replace with your primitive's matrix construction (e.g. a circulant
        search, a Cauchy matrix, or a fixed low-addition family). If you follow a generic
        derivation strategy that might be reusable by other primitives, implement it in
        utils/matrix.py and import. Example: cauchy_mds_matrix(self.p, self.t)."""
        return simple_circulant_matrix(self.t)

    def _init_cons(self):
        """Return the round constants as an R x t grid (one t-vector per round, indexed rcons[r]).
        TODO: replace with your primitive's round-constant derivation (e.g. from the
        digits of pi, a fixed seed, or a counter-based construction). Many primitives use
        a FieldElementSampler from utils/sampler.py to implement reproducible round constant generation.
        This placeholder is a deterministic counter grid -- valid in shape, not cryptographic."""
        return [[r * self.t + i + 1 for i in range(self.t)] for r in range(self.R)]
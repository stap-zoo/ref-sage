# params.py
# MyPrimitiveParams: the fully-specified parameter set for MyPrimitive (single source of truth per instance).

# Structural imports
from recommendations import recommend
from types import SimpleNamespace

# Math specific imports
from math import gcd
from sage.all import GF, Integer

# Custom imports
from utils.matrix import map_nested, invert_matrix, simple_circulant_matrix
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
    """Single instance spec for MyPrimitive, read by the permutation (MyPrimitivePerm) and the
    mode functions (MyPrimitiveHash / MyPrimitiveCompress). Carries the permutation parameters
    plus the per-mode parameter dicts `sponge` and `comp`; the permutation ignores them, the
    mode functions build their mode from them."""

    def __init__(
        self,
        p:     int,
        t:     int,
        alpha: int = None,
        R:     int = None,
        M:     list[list[int]] = None,
        rcons: list[list[int]] = None,
        # Modes of operation: per-mode parameter dicts, or None if the instance defines no
        # such mode. sponge = dict(r=.., c=.., d=..); comp = dict(d=..) / dict(a=..) / ...
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
        t     : state size (number of field elements)
        alpha : S-box exponent, coprime with p-1; smallest valid exponent if not provided
        R     : number of rounds; derived via _init_rounds() if not provided
        M     : t x t MDS matrix for the linear layer; generated via _init_mat() if not provided
        rcons : round constants; generated via _init_cons() if not provided
        sponge: sponge params dict dict(r, c, d) (any omitted -> derived), or None for no sponge
        comp  : compression params dict (digest d and/or arity a, optional matrix M), or None
        kappa : target security level in bits (default 128)
        toy   : if True, recommendation-level checks warn instead of raising (default False)
        """

        # Input sanitization: validate the raw constructor arguments
        self._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings: store the field, state size, and security level
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa
        self.toy = toy

        # Modes of operation: per-mode param dicts (consumed by the mode functions, not the
        # permutation). None means the instance does not define that mode.
        self.sponge, self.comp = sponge, comp

        # Non-linear layer: any values associated to non-linear layer
        self.alpha = alpha if alpha is not None else self._init_alpha()
        self.alpha_inv = pow(self.alpha, -1, p - 1)

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
        attribute (args.p, args.t, args.alpha, args.R, args.M, args.rcons, args.kappa).
 
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
        # Example: a small field is fine for an explicit toy=True / cryptanalysis instance
        # (warns) but is not secure (or not analyzed) for real use, so non-toy instances
        # refuse it outright instead of silently constructing an insecure "real" instance.
        field_bits = int(params.p).bit_length()
        if field_bits < 31:
            msg = f"TOY VERSION: field is only {field_bits} bits"
            recommend(msg, params.toy)

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
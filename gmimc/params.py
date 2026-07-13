# params.py
# ---------------------------------------------------------------------------
# Parameter definition for GMiMC: the GMiMCParams class.
#
# GMiMCParams is the single source of truth for an instance. It sanitizes the
# user-facing parameters and expands them into a fully-specified instance that
# every other file (permutation, hash modes, instances, tests) consumes. Any
# value the user omits is filled in by the matching _init_* helper (or, for
# r/c/d, by the shared derive_rate_capacity_digest). Settings that depart from
# the recommended ones raise a ParamRecommendationWarning rather than an error.
# ---------------------------------------------------------------------------

# Structural imports
import warnings
from recommendations import ParamRecommendationWarning
from types import SimpleNamespace

# Math specific imports
from math import gcd
from sage.all import GF, Integer

# Custom imports
from utils.matrix import simple_circulant_matrix, map_nested, invert_matrix
from utils.sampler import XOFFieldElementSampler
from utils.mode import derive_rate_capacity_digest


class GMiMCParams:
    def __init__(
        self,
        p:           int,
        t:           int,
        R:           int = None,
        M:           list[list[int]] = None,
        alpha:       int = None,
        rcons:       list[int] = None,
        r:           int = None,
        c:           int = None,
        d:           int = None,
        kappa:       int = 128,
    ):
        """
        Parameters
        ----------
        p             : field characteristic (prime)
        t             : permutation state size (branches)
        R             : number of rounds; derived via _init_rounds if not provided (not yet implemented)
        M             : matrix (txt); generated via _init_mat (cyclic-shift permutation matrix) if not provided
        alpha         : exponent of the power-map S-box (its degree)
        rcons         : R affine round constants; generated via _init_cons if not provided
        r             : rate (number of outer state elements absorbed/squeezed per sponge step);
                        derived from kappa/t via derive_rate_capacity_digest if not provided
        c             : capacity (number of inner state elements); derived if not provided
        d             : digest size (number of output elements); derived if not provided
        kappa         : target security level in bits (default 128)
        """

        # Input sanitization
        GMiMCParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Power map
        self.alpha = alpha if alpha is not None else self._init_alpha()

        # Rounds (set before _init_cons, whose derivation depends on R)
        self.R = R if R is not None else self._init_rounds()

        # Hash modes
        self.r, self.c, self.d = derive_rate_capacity_digest(self.kappa, self.t, r, c, d)

        # Affine layer
        self.M = map_nested(M if M is not None else self._init_mat(), self.to_field)
        self.M_inv = invert_matrix(self.M)
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
        if params.p == 2:
            raise NotImplementedError("Characteristic 2 not implemented")
        if params.t <= 1:
            raise ValueError(f"state size t must be greater than 1. Got {params.t}")
        if params.alpha is not None and gcd(params.alpha, params.p - 1) != 1:
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
        if len(self.rcons) < self.R:
            raise ValueError(f"Expected at least {self.R} round constants, got {len(self.rcons)}")

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_rounds(self) -> int:
        """Derive the round number from the target security level kappa.
        TODO: implement the round-number criterion of the GMiMC paper
        (https://eprint.iacr.org/2019/397, Section 5: algebraic and statistical
        attack bounds per variant); until then R must be passed explicitly."""
        raise NotImplementedError("Error: Not implemented -- round number derivation for GMiMC")

    def _init_alpha(self) -> int:
        alpha = 2
        while gcd(alpha, self.p - 1) != 1:
            alpha += 1
        return alpha

    def _init_mat(self) -> list[list[int]]:
        mat = [[0 for _ in range(self.t)] for _ in range(self.t)]
        for i in range(0, self.t - 1):
            mat[i][i + 1] += 1
        mat[self.t - 1][0] += 1
        return mat

    def _init_cons(self):
        # Deterministic constant generation via SHAKE256, so the round constants
        # can be reproduced from (p, t, R) instead of relying on Sage's unseeded random_element().
        seed = f"GMiMC({self.p},{self.t},{self.R})".encode("ascii")

        rcons = XOFFieldElementSampler(seed=seed + b"aff", p=self.p, xof="shake_256", sampling="mod").grid(self.R, 1)
        rcons = [con[0] for con in rcons]

        return rcons

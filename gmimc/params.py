# params.py
# ---------------------------------------------------------------------------
# Parameter definition for GMiMC: the GMiMCParams class.
#
# GMiMCParams is the single source of truth for an instance. It sanitizes 
# user-facing parameters and expands them into a fully-specified instance that
# the permutation, hash modes, instances and tests consume. Any value the user
# omits is filled in by the matching _init_* helper. Settings that depart from 
# the recommended ones raise a ParamRecommendationWarning rather than an error.
# ---------------------------------------------------------------------------

# Structural imports
from recommendations import recommend
from types import SimpleNamespace

# Math specific imports
from math import gcd
from sage.all import GF, Integer

# Custom imports
from utils.matrix import simple_circulant_matrix, map_nested, invert_matrix
from utils.sampler import XOFFieldElementSampler
from utils.mode import SpongeLE


class GMiMCParams:
    def __init__(
        self,
        p:           int,
        t:           int,
        R:           int = None,
        M:           list[list[int]] = None,
        alpha:       int = None,
        rcons:       list[int] = None,
        # Sponge parameters (derived if not provided)
        r:           int = None,
        c:           int = None,
        d:           int = None,
        # Target security level (default 128 bits)
        kappa:       int = 128,
        toy:         bool = False,
    ):
        """
        Parameters
        ----------
        p     : field characteristic (prime)
        t     : permutation state size (branches)
        R     : number of rounds; derived via _init_rounds if not provided (not yet implemented)
        M     : matrix (txt); generated via _init_mat (cyclic-shift permutation matrix) if not provided
        alpha : exponent of the power-map S-box (its degree)
        rcons : R affine round constants; generated via _init_cons if not provided
        r     : rate (number of outer state elements absorbed/squeezed per sponge step); derived if not provided
        c     : capacity (number of inner state elements for sponge); derived if not provided
        d     : digest size for generic fixed-output sponge (number of output elements); derived if not provided
        kappa : target security level in bits (default 128)
        toy   : if True, recommendation-level checks warn instead of raising (default False)
        """

        # Input sanitization
        GMiMCParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa
        self.toy = toy

        # Sponge parameters
        self.sponge = SpongeLE(kappa=kappa, p=p, t=t, r=r, c=c, d=d, to_field=self.to_field, toy=toy)

        # Non-linear layer
        self.alpha = alpha if alpha is not None else self._init_alpha()

        # Rounds (set before _init_cons, whose derivation depends on R)
        self.R = R if R is not None else self._init_rounds()

        # Liner layer
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
        if params.t <= 1:
            raise ValueError(f"state size t must be greater than 1. Got {params.t}")
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

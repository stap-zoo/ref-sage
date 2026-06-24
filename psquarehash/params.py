# params.py
# ---------------------------------------------------------------------------
# Parameter definition for pSquare-hash: the pSquareHashParams class.
#
# pSquareHashParams is the single source of truth for an instance. It sanitizes the
# user-facing parameters and expands them into a fully-specified instance that
# every other file (permutation, hash modes, instances, tests) consumes. Any
# value the user omits is filled in by the matching _init_* helper (or, for
# r/c/d, by the shared derive_rate_capacity_digest). Settings that depart from
# the recommended ones raise a ParamRecommendationWarning rather than an error.
# ---------------------------------------------------------------------------

# Structural imports
from recommendations import ParamRecommendationWarning
from types import SimpleNamespace

# Math specific imports
import warnings
from math import gcd
from sage.all import GF, Integer, legendre_symbol

# Custom imports
from utils.matrix import simple_circulant_matrix, map_nested, invert_matrix
from utils.sampler import XOFFieldElementSampler
from utils.mode import derive_rate_capacity_digest


class pSquareHashParams:
    def __init__(
        self,
        p:           int,
        t:           int,
        R:           int,
        M:           list[list[int]] = None,
        M_IO:           list[list[int]] = None,
        rcons:       list[list[int]] = None,
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
        R             : number of rounds
        M             : matrix (txt)
        M_IO          : Input/Output matrix (txt)
        rcons         : Rxt affine round constants; generated via _init_constants if not provided
        r             : rate (number of outer state elements absorbed/squeezed per sponge step);
                        derived from kappa/t via derive_rate_capacity_digest if not provided
        c             : capacity (number of inner state elements); derived if not provided
        d             : digest size (number of output elements); derived if not provided
        kappa         : target security level in bits (default 128)
        """

        # Input sanitization
        pSquareHashParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Rounds (set before _init_constants, whose derivation depends on R)
        self.R = R

        # Hash modes
        self.r, self.c, self.d = derive_rate_capacity_digest(self.kappa, self.t, r, c, d)

        # Affine layer
        self.M = map_nested(M if M is not None else self._init_M(), self.to_field)
        self.M_inv = invert_matrix(self.M)
        self.M_IO = map_nested(M_IO if M_IO is not None else self._init_M_IO(), self.to_field)
        self.M_IO_inv = invert_matrix(self.M_IO)
        self.rcons = map_nested(rcons if rcons is not None else self._init_rcons(), self.to_field)
        if len(self.rcons) < self.R:
            raise ValueError(f"Expected at least {self.R} round-constant rows, got {len(self.rcons)}")

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
        if params.t % 4 != 0:
            raise ValueError(f"state size t must be multiple of 4. Got {params.t}")

        # --- Warnings (recommended, not required) ---
        field_bits = int(params.p).bit_length()
        if field_bits < 31:
            warnings.warn(f"TOY VERSION: field is only {field_bits} bits", ParamRecommendationWarning, stacklevel=2)

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------
    def _init_M(self) -> list[list[int]]:
        mat = [[0 for _ in range(self.t)] for _ in range(self.t)]
        for i in range(0, self.t // 2):
            mat[i][self.t // 2 + i] += 1
            mat[self.t // 2 + i][i] += 1
        for i in range(2, self.t // 2 - 2, 2):
            mat[self.t // 2 - 2][i] += 1
            mat[self.t // 2 - 1][i + 1] += 1
        for i in range(2, self.t // 2, 2):
            mat[i][self.t // 2 - 2] += 2
            mat[i][self.t // 2 - 1] += 1
            mat[i + 1][self.t // 2 - 2] += 1
            mat[i + 1][self.t // 2 - 1] += 1
        return mat

    def _init_M_IO(self) -> list[list[int]]:
        mat_io = [[0 for _ in range(self.t)] for _ in range(self.t)]
        for i in range(0, self.t):
            mat_io[i][i] += 1
        for i in range(0, self.t // 2):
            mat_io[i][self.t // 2 + i] += 1
            mat_io[self.t // 2 + i][i] += 2
        return mat_io

    def _init_constants(self):
        # Deterministic constant generation via SHAKE256, so coeffs_g/coeffs_h/rcons
        # can be reproduced from (p, t, R) instead of relying on Sage's unseeded random_element().
        seed = f"pSquare-hash({self.p},{self.t},{self.R})".encode("ascii")

        rcons = XOFFieldElementSampler(seed=seed + b"aff", p=self.p, xof="shake_256", sampling="mod").grid(self.R, self.t // 2)

        return rcons

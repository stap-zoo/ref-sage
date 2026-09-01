# params.py
# pSquareHashParams: the fully-specified parameter set for pSquare-hash (single source of truth per instance).

# Structural imports
from recommendations import recommend
from types import SimpleNamespace

# Math specific imports
from math import gcd
from sage.all import GF, Integer, legendre_symbol

# Custom imports
from utils.matrix import simple_circulant_matrix, map_nested, invert_matrix
from utils.sampler import XOFFieldElementSampler


class pSquareHashParams:
    def __init__(
        self,
        p:     int,
        t:     int,
        R:     int = None,
        M:     list[list[int]] = None,
        M_IO:  list[list[int]] = None,
        rcons: list[list[int]] = None,
        # Modes of operation: per-mode parameter dicts, or None if the instance defines no
        # such mode. sponge = dict(r=.., c=.., d=..); comp = dict(a=2) (2-to-1 truncation).
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
        t     : permutation state size (branches)
        R     : number of rounds; derived via _init_rounds if not provided (not yet implemented)
        M     : matrix (txt); generated via _init_mat if not provided
        M_IO  : Input/Output matrix (txt); generated via _init_mat_IO if not provided
        rcons : Rxt affine round constants; generated via _init_cons if not provided
        sponge: sponge params dict dict(r, c, d), or None for no sponge
        comp  : compression params dict (e.g. dict(a=2) for 2-to-1), or None
        kappa : target security level in bits (default 128)
        toy   : if True, recommendation-level checks warn instead of raising (default False)
        """

        # Input sanitization
        pSquareHashParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa
        self.toy = toy

        # Modes of operation: per-mode param dicts (consumed by the mode functions, not the
        # permutation). None means the instance does not define that mode.
        self.sponge, self.comp = sponge, comp

        # Rounds (set before _init_cons, whose derivation depends on R)
        self.R = R if R is not None else self._init_rounds()

        # Affine layer
        self.M = map_nested(M if M is not None else self._init_mat(), self.to_field)
        self.M_inv = invert_matrix(self.M)
        self.M_IO = map_nested(M_IO if M_IO is not None else self._init_mat_IO(), self.to_field)
        self.M_IO_inv = invert_matrix(self.M_IO)
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
        if params.t % 4 != 0:
            raise ValueError(f"state size t must be multiple of 4. Got {params.t}")

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
        if len(self.M_IO) != self.t or any(len(row) != self.t for row in self.M_IO):
            raise ValueError(f"M_IO must be a {self.t} x {self.t} matrix")
        if len(self.rcons) < self.R or any(len(row) != self.t // 2 for row in self.rcons):
            raise ValueError(f"rcons must hold at least {self.R} rows of {self.t // 2} constants each")

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_rounds(self) -> int:
        """Derive the round number from the target security level kappa.
        TODO: implement the round-number criterion of the pSquare paper
        (https://eprint.iacr.org/2026/1129); until then R must be passed explicitly."""
        raise NotImplementedError("Error: Not implemented -- round number derivation for pSquare-hash")

    def _init_mat(self) -> list[list[int]]:
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

    def _init_mat_IO(self) -> list[list[int]]:
        mat_io = [[0 for _ in range(self.t)] for _ in range(self.t)]
        for i in range(0, self.t):
            mat_io[i][i] += 1
        for i in range(0, self.t // 2):
            mat_io[i][self.t // 2 + i] += 1
            mat_io[self.t // 2 + i][i] += 2
        return mat_io

    def _init_cons(self):
        # Deterministic constant generation via SHAKE256, so the round constants
        # can be reproduced from (p, t, R) instead of relying on Sage's unseeded random_element().
        seed = f"pSquare-hash({self.p},{self.t},{self.R})".encode("ascii")

        rcons = XOFFieldElementSampler(seed=seed + b"aff", p=self.p, xof="shake_256", sampling="mod").grid(self.R, self.t // 2)

        return rcons

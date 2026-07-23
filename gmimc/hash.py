# hash.py
# ---------------------------------------------------------------------------
# GMiMC: the permutation (round function) and the hash modes built on it.
#
# Constructed from a fully-specified GMiMCParams object.
# ---------------------------------------------------------------------------

from gmimc.params import GMiMCParams
from utils.matrix import matvecmul, vecadd, vecsub
from utils.mode import compress_davies_meyer


class GMiMC:
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: GMiMCParams):
        self.F = params.F
        self.to_field = params.to_field
        self.from_field = params.from_field
        self.t = params.t
        self.alpha = params.alpha

        # Rounds
        self.R = params.R

        # Linear layers
        self.M = params.M
        self.M_inv = params.M_inv

        # Constants
        self.rcons = params.rcons

        # Hash modes
        self.sponge = params.sponge

    # ---------------------------------------------------------------------------
    # Component functions
    # ---------------------------------------------------------------------------

    def nonlinear_layer(self, x_in: list, r: int) -> list:
        x_out = x_in.copy()
        x_0_pow = (x_out[0] + self.rcons[r])**self.alpha
        for i in range(1, self.t):
            x_out[i] += x_0_pow
        return x_out
    
    def nonlinear_layer_inv(self, x_in: list, r: int) -> list:
        x_out = x_in.copy()
        x_0_pow = (x_out[0] + self.rcons[r])**self.alpha
        for i in range(1, self.t):
            x_out[i] -= x_0_pow
        return x_out

    def linear_layer(self, state: list, r: int) -> list:
        return matvecmul(self.M, state)

    def linear_layer_inv(self, state: list, r: int) -> list:
        return matvecmul(self.M_inv, state)

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps (GMiMC does no work outside the loop: identities)
    # ---------------------------------------------------------------------------

    def _pre_rounds(self, state: list) -> list:
        return state

    def _pre_rounds_inv(self, state: list) -> list:
        return state

    def _post_rounds(self, state: list) -> list:
        return state

    def _post_rounds_inv(self, state: list) -> list:
        return state

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)
        for r in range(self.R):
            state = self.nonlinear_layer(state, r)
            state = self.linear_layer(state, r)
        return self._post_rounds(state)

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for r in reversed(range(self.R)):
            state = self.linear_layer_inv(state, r)
            state = self.nonlinear_layer_inv(state, r)
        return self._pre_rounds_inv(state)

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------

    def compress_2_to_1(self, x1: list, x2: list) -> list:
        """Davies-Meyer 2-to-1 compression, defined when t == 2 * digest size."""
        if self.t != 2 * self.d:
            raise ValueError(f"Compression mode not defined for state size {self.t} and digest size {self.d}.")
        if len(x1) != self.d or len(x2) != self.d:
            raise ValueError(f"Invalid input sizes. Expected ({self.d},{self.d}), got ({len(x1)},{len(x2)})")
        return compress_davies_meyer(
            perm=self.permutation,
            x_m=x1,
            x_c=x2,
            digest_size=self.d,
            to_field=self.to_field,
        )

    def hash_sponge(self, data: list) -> list:
        return self.sponge.hash(self.permutation, data, input_len_fixed=True)

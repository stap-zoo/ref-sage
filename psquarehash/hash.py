# hash.py
# ---------------------------------------------------------------------------
# pSquare-hash: the permutation (round function) and the hash modes built on it.
#
# Constructed from a fully-specified pSquare-hashParams object. 
# ---------------------------------------------------------------------------

from psquarehash.params import pSquareHashParams
from utils.matrix import matvecmul, vecadd, vecsub, add_to_start
from utils.mode import compress_davies_meyer, hash_sponge_safe, pad_zero


class pSquareHash:
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: pSquareHashParams):
        self.F = params.F
        self.to_field = params.to_field
        self.from_field = params.from_field
        self.t = params.t

        # Rounds
        self.R = params.R

        # Linear layers
        self.M = params.M
        self.M_inv = params.M_inv
        self.M_IO = params.M_IO
        self.M_IO_inv = params.M_IO_inv

        # Constants
        self.rcons = params.rcons

        # Hash modes
        self.r = params.r
        self.c = params.c
        self.d = params.d

    # ---------------------------------------------------------------------------
    # Component functions
    # ---------------------------------------------------------------------------
    def feistel(self, x_in: list, feistel_constants: list) -> list:
        y_1 = x_in[1] + feistel_constants[0]
        y_2 = x_in[0] + y_1**2
        y_3 = y_1 + y_2 + feistel_constants[1]
        y_4 = y_2 + y_3**2
        y_5 = y_3 + y_4
        return [y_4, y_5]

    def nonlinear_layer(self, x_in: list, r: int) -> list:
        x_out = x_in.copy()
        for i in range(0, self.t // 2, 2):
            y = self.feistel(x_in[i:i + 2], self.rcons[r][self.t // 2 - (i + 2):self.t // 2 - i])
            x_out[self.t - i - 2] += y[0]
            x_out[self.t - i - 1] += y[1]
        return x_out
    
    def nonlinear_layer_inv(self, x_in: list, r: int) -> list:
        x_out = x_in.copy()
        for i in range(0, self.t // 2, 2):
            y = self.feistel(x_in[i:i + 2], self.rcons[r][self.t // 2 - (i + 2):self.t // 2 - i])
            x_out[self.t - i - 2] -= y[0]
            x_out[self.t - i - 1] -= y[1]
        return x_out

    def linear_layer(self, state: list, r: int) -> list:
        return matvecmul(self.M, state)

    def linear_layer_inv(self, state: list, r: int) -> list:
        return matvecmul(self.M_inv, state)

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps (pSquare-hash does no work outside the loop: identities)
    # ---------------------------------------------------------------------------

    def _pre_rounds(self, state: list) -> list:
        return matvecmul(self.M_IO, state)

    def _pre_rounds_inv(self, state: list) -> list:
        return matvecmul(self.M_IO_inv, state)

    def _post_rounds(self, state: list) -> list:
        return matvecmul(self.M_IO, state)

    def _post_rounds_inv(self, state: list) -> list:
        return matvecmul(self.M_IO_inv, state)

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
        padded_data, _ = pad_zero(data, self.r, self.to_field)
        IV = [self.to_field(len(data))] + [self.F.zero()] * (self.c - 1)
        return hash_sponge_safe(
            perm=self.permutation,
            data=padded_data,
            state_size=self.t,
            rate=self.r,
            capacity=self.c,
            digest_size=self.d,
            IV=IV,
            absorb=add_to_start,
            to_field=self.to_field,
        )

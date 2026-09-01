# hash.py
# pSquare-hash: pSquareHashPerm (permutation) and its mode functions (pSquareHashHash, pSquareHashCompress).

from psquarehash.params import pSquareHashParams
from utils.matrix import matvecmul, vecadd, vecsub
from utils.primitive import Permutation, HashFunction, CompressionFunction


class pSquareHashPerm(Permutation):
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: pSquareHashParams):
        super().__init__(params)  # F, to_field, from_field, t, p, kappa, toy

        # Rounds
        self.R = params.R

        # Linear layers
        self.M = params.M
        self.M_inv = params.M_inv
        self.M_IO = params.M_IO
        self.M_IO_inv = params.M_IO_inv

        # Constants
        self.rcons = params.rcons

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

    def permute(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)
        for r in range(self.R):
            state = self.nonlinear_layer(state, r)
            state = self.linear_layer(state, r)
        return self._post_rounds(state)

    def permute_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for r in reversed(range(self.R)):
            state = self.linear_layer_inv(state, r)
            state = self.nonlinear_layer_inv(state, r)
        return self._pre_rounds_inv(state)


# ---------------------------------------------------------------------------
# Hash / compression functions
#
# pSquareHashPerm above is JUST the permutation. Each mode wraps a permutation:
#     P = pSquareHashPerm(params)
#     H = pSquareHashHash(P, params.sponge)      # length-encoded sponge (fixed-length input)
#     H.hash(data, input_len_fixed=True)
#     C = pSquareHashCompress(P, params.comp)    # 2-to-1 truncation / Davies-Meyer (t -> t/2)
#     C.compress(x1 + x2)
# The 2-to-1 compression (comp=dict(a=2)) is only defined for the t = 2d instances; the
# others set comp=None (no pSquareHashCompress).
# ---------------------------------------------------------------------------

class pSquareHashHash(HashFunction):
    SPONGE_KIND = "le"        # length-encoded sponge (fixed-length input)

class pSquareHashCompress(CompressionFunction):
    COMP_KIND = "trunc"       # 2-to-1 truncation / Davies-Meyer (comp=dict(a=2))

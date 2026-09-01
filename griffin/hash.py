# hash.py
# Griffin: GriffinPerm (permutation) and its mode functions (GriffinHash, GriffinCompress).

from griffin.params import GriffinParams
from utils.matrix import matvecmul, vecadd, vecsub
from utils.primitive import Permutation, HashFunction, CompressionFunction


class GriffinPerm(Permutation):
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: GriffinParams):
        super().__init__(params)  # F, to_field, from_field, t

        # Rounds
        self.R = params.R

        # Non-linear layer
        self.alpha = params.alpha
        self.alpha_inv = params.alpha_inv
        self.coeffs_G = params.coeffs_G

        # Affine layer
        self.M = params.M
        self.M_inv = params.M_inv
        self.rcons = params.rcons

    # ---------------------------------------------------------------------------
    # Component functions
    # ---------------------------------------------------------------------------

    def constant_addition(self, state: list, r: int) -> list:
        return vecadd(state, self.rcons[r])

    def constant_addition_inv(self, state: list, r: int) -> list:
        return vecsub(state, self.rcons[r])

    def linear_layer(self, state: list, r: int) -> list:
        return matvecmul(self.M, state)

    def linear_layer_inv(self, state: list, r: int) -> list:
        return matvecmul(self.M_inv, state)

    def _L(self, i: int, z0, z1, z2):
        """L_i(y0, y1, z) = gamma_i * z0 + z1 + z2, with gamma_i = i - 1"""
        return (i - 1) * z0 + z1 + z2

    def _G(self, i: int, l):
        """G_i(l) = l^2 + alpha_i * l + beta_i, root-free since alpha_i^2 - 4*beta_i is a nonresidue"""
        alpha_i, beta_i = self.coeffs_G[i - 2]   # coeffs_G[k] holds (alpha_{k+2}, beta_{k+2})
        return l ** 2 + alpha_i * l + beta_i

    def nonlinear_layer(self, state: list, r: int) -> list:
        """See https://eprint.iacr.org/2022/403.pdf, Eq. 6 and below.
        Round-independent (same map every round), so r is accepted but unused."""
        y0 = state[0] ** self.alpha_inv        # y_0 = x_0^(1/d)
        y1 = state[1] ** self.alpha            # y_1 = x_1^d
        out = [y0, y1]
        for i in range(2, self.t):             # y_i = x_i * G_i(L_i), Horst step
            z = 0 if i == 2 else state[i - 1]  # no feedback word for i = 2
            out.append(state[i] * self._G(i, self._L(i, y0, y1, z)))
        return out

    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        y0, y1 = state[0], state[1]
        x0 = y0 ** self.alpha                  # inverse of x -> x^(1/d)
        x1 = y1 ** self.alpha_inv              # inverse of x -> x^d
        out = [x0, x1]
        for i in range(2, self.t):
            z = 0 if i == 2 else out[i - 1]    # recovered x_(i-1)
            out.append(state[i] * self._G(i, self._L(i, y0, y1, z))**(-1))
        return out

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps (work done once outside the round loop)
    # ---------------------------------------------------------------------------

    def _pre_rounds(self, state: list) -> list:
        return matvecmul(self.M, state)  # initial matrix multiplication

    def _pre_rounds_inv(self, state: list) -> list:
        return matvecmul(self.M_inv, state)

    def _post_rounds(self, state: list) -> list:
        return state

    def _post_rounds_inv(self, state: list) -> list:
        return state

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
            state = self.constant_addition(state, r)
        return self._post_rounds(state)

    def permute_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for r in reversed(range(self.R)):
            state = self.constant_addition_inv(state, r)
            state = self.linear_layer_inv(state, r)
            state = self.nonlinear_layer_inv(state, r)
        return self._pre_rounds_inv(state)


# ---------------------------------------------------------------------------
# Hash function
#
# GriffinPerm above is JUST the permutation. Each mode is its own function object over it:
#     P = GriffinPerm(params)
#     H = GriffinHash(P, params.sponge)     # length-encoded sponge (fixed-length input)
#     H.hash(data, input_len_fixed=True)
#     C = GriffinCompress(P, params.comp)   # 2-to-1 truncation / Davies-Meyer (t -> t/2)
#     C.compress(x1 + x2)
# The 2-to-1 compression (comp=dict(a=2)) is only defined for even t; the t=3 instances set
# comp=None (no GriffinCompress for them).
# ---------------------------------------------------------------------------

class GriffinHash(HashFunction):
    SPONGE_KIND = "le"        # length-encoded sponge (fixed-length input)

class GriffinCompress(CompressionFunction):
    COMP_KIND = "trunc"       # 2-to-1 truncation / Davies-Meyer (comp=dict(a=2))

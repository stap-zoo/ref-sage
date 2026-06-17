# hash.py
# ---------------------------------------------------------------------------
# Griffin: the permutation (round function) and the hash modes built on it.
#
# Constructed from a fully-specified GriffinParams object.
# ---------------------------------------------------------------------------

from griffin.params import GriffinParams
from utils import matvecmul, vecadd, vecsub, add_to_start
from modes import hash_sponge, pad_zero, compress_davies_meyer


class Griffin:
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: GriffinParams):
        self.F = params.F
        self.to_field = params.to_field
        self.from_field = params.from_field
        self.t = params.t

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

        # Hash modes
        self.r = params.r
        self.c = params.c
        self.d = params.d

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

    def permutation(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)
        for r in range(self.R):
            state = self.nonlinear_layer(state, r)
            state = self.linear_layer(state, r)
            state = self.constant_addition(state, r)
        return self._post_rounds(state)

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for r in reversed(range(self.R)):
            state = self.constant_addition_inv(state, r)
            state = self.linear_layer_inv(state, r)
            state = self.nonlinear_layer_inv(state, r)
        return self._pre_rounds_inv(state)

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------
    
    def compress_2_to_1(self, x1: list, x2:list) -> list:
        """2-to-1 compression defined for small state sizes double the digest size"""
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
        return hash_sponge(
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

# hash.py
# Anemoi: AnemoiPerm (permutation) and its mode functions (AnemoiHash, AnemoiCompress).

from anemoi.params import AnemoiParams
from utils.matrix import matvecmul, vecadd, vecsub
from utils.primitive import Permutation, HashFunction, CompressionFunction


class AnemoiPerm(Permutation):
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: AnemoiParams):
        super().__init__(params)  # F, to_field, from_field, t, p, kappa, toy
        self.l = params.l

        # Rounds
        self.R = params.R

        # Non-linear layer (open Flystel)
        self.alpha = params.alpha
        self.alpha_inv = params.alpha_inv
        self.g = params.g
        self.QUAD = params.QUAD
        self.beta = params.beta
        self.gamma = params.gamma
        self.delta = params.delta

        # Linear layer and round constants
        self.Mx = params.Mx
        self.Mx_inv = params.Mx_inv
        self.My = params.My
        self.My_inv = params.My_inv
        self.C = params.C
        self.D = params.D

    # ---------------------------------------------------------------------------
    # Component functions (state is x || y with x = state[:l], y = state[l:])
    # ---------------------------------------------------------------------------

    def constant_addition(self, state: list, r: int) -> list:
        return vecadd(state[:self.l], self.C[r]) + vecadd(state[self.l:], self.D[r])

    def constant_addition_inv(self, state: list, r: int) -> list:
        return vecsub(state[:self.l], self.C[r]) + vecsub(state[self.l:], self.D[r])

    def linear_layer(self, state: list, r: int) -> list:
        """MDS on the x-lane, MDS on the rotated y-lane, then a pseudo-Hadamard transform.
        Round-independent, so r is accepted but unused."""
        x, y = state[:self.l], state[self.l:]
        x, y = matvecmul(self.Mx, x), matvecmul(self.My, y)
        y = vecadd(y, x)
        x = vecadd(x, y)
        return x + y

    def linear_layer_inv(self, state: list, r: int) -> list:
        x, y = state[:self.l], state[self.l:]
        x = vecsub(x, y)
        y = vecsub(y, x)
        x, y = matvecmul(self.Mx_inv, x), matvecmul(self.My_inv, y)
        return x + y

    def _Q_gamma(self, x):
        return self.beta * x ** self.QUAD + self.gamma
    
    def _Q_delta(self, x):
        return self.beta * x ** self.QUAD + self.delta

    def _open_flystel(self, x, y):
        """Open Flystel H : (x, y) -> (u, v), the evaluation form using the inverse power map.
        See https://eprint.iacr.org/2022/840, Fig. 3a."""
        u = x - self._Q_gamma(y)
        v = y - u ** self.alpha_inv
        u = u + self._Q_delta(v)
        return u, v

    def _open_flystel_inv(self, u, v):
        x = u - self._Q_delta(v)
        y = v + x ** self.alpha_inv
        x = x + self._Q_gamma(y)
        return x, y

    def _closed_flystel(self, y, v):
        """Closed Flystel V : (y, v) -> (x, u), the verification form using the forward power map.
        Equivalent to the open Flystel on consistent values: H(x, y) = (u, v) iff V(y, v) = (x, u).
        See https://eprint.iacr.org/2022/840, Fig. 3b."""
        e = (y - v) ** self.alpha
        x = e + self._Q_gamma(y)
        u = e + self._Q_delta(v)
        return x, u

    def nonlinear_layer(self, state: list, r: int) -> list:
        """Open Flystel applied to each column (x_i, y_i).
        Round-independent, so r is accepted but unused."""
        out = list(state)
        for i in range(self.l):
            out[i], out[self.l + i] = self._open_flystel(out[i], out[self.l + i])
        return out

    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        out = list(state)
        for i in range(self.l):
            out[i], out[self.l + i] = self._open_flystel_inv(out[i], out[self.l + i])
        return out

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps. Anemoi applies a final degenerate linear layer once
    # after the round loop (the round index is irrelevant; linear_layer ignores it).
    # ---------------------------------------------------------------------------

    def _pre_rounds(self, state: list) -> list:
        return state

    def _pre_rounds_inv(self, state: list) -> list:
        return state

    def _post_rounds(self, state: list) -> list:
        return self.linear_layer(state, 0)

    def _post_rounds_inv(self, state: list) -> list:
        return self.linear_layer_inv(state, 0)

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permute(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)
        for r in range(self.R):
            state = self.constant_addition(state, r)
            state = self.linear_layer(state, r)
            state = self.nonlinear_layer(state, r)
        return self._post_rounds(state)

    def permute_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for r in reversed(range(self.R)):
            state = self.nonlinear_layer_inv(state, r)
            state = self.linear_layer_inv(state, r)
            state = self.constant_addition_inv(state, r)
        return self._pre_rounds_inv(state)


# ---------------------------------------------------------------------------
# Hash functions
#
# AnemoiPerm above is JUST the permutation. Each mode is its own function object over it:
#     P = AnemoiPerm(params)
#     H = AnemoiHash(P, params.sponge)     # Hirose sponge:                 H.permute, H.hash
#     C = AnemoiCompress(P, params.comp)   # Jive_2 compression (2l -> l):  C.permute, C.compress
# The arity of the Jive compression comes from the comp dict (e.g. dict(a=2)).
# ---------------------------------------------------------------------------

class AnemoiHash(HashFunction):
    SPONGE_KIND = "hirose"    # Anemoi's Hirose-mode sponge

class AnemoiCompress(CompressionFunction):
    COMP_KIND = "jive"        # Jive_a: a-to-1 (comp=dict(a=2) -> 2l -> l)

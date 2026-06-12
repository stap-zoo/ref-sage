from anemoi.params import AnemoiParams
from utils import matvecmul, vecadd, vecsub
from modes import compress_jive, hash_sponge_hirose, pad_one


class Anemoi:
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: AnemoiParams):
        self.F = params.F
        self.to_field = params.to_field
        self.from_field = params.from_field
        self.l = params.l
        self.t = params.t
        self.kappa = params.kappa

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

        # Hash modes
        self.r = params.r
        self.c = params.c
        self.d = params.d

    # ---------------------------------------------------------------------------
    # Component functions (state is x || y with x = state[:l], y = state[l:])
    # ---------------------------------------------------------------------------

    def ConstantAddition(self, state: list, round_idx: int) -> list:
        return vecadd(state[:self.l], self.C[round_idx]) + vecadd(state[self.l:], self.D[round_idx])

    def ConstantAddition_inv(self, state: list, round_idx: int) -> list:
        return vecsub(state[:self.l], self.C[round_idx]) + vecsub(state[self.l:], self.D[round_idx])

    def LinearLayer(self, state: list) -> list:
        """MDS on the x-lane, MDS on the rotated y-lane, then a pseudo-Hadamard transform."""
        x, y = state[:self.l], state[self.l:]
        x, y = matvecmul(self.Mx, x), matvecmul(self.My, y)
        y = vecadd(y, x)
        x = vecadd(x, y)
        return x + y

    def LinearLayer_inv(self, state: list) -> list:
        x, y = state[:self.l], state[self.l:]
        x = vecsub(x, y)
        y = vecsub(y, x)
        x, y = matvecmul(self.Mx_inv, x), matvecmul(self.My_inv, y)
        return x + y

    def _Q_gamma(self, x):
        return self.beta * x ** self.QUAD + self.gamma
    
    def _Q_delta(self, x):
        return self.beta * x ** self.QUAD + self.delta

    def OpenFlystel(self, x, y):
        """Open Flystel H : (x, y) -> (u, v), the evaluation form using the inverse power map.
        See https://eprint.iacr.org/2022/840, Fig. 3a."""
        u = x - self._Q_gamma(y)
        v = y - u ** self.alpha_inv
        u = u + self._Q_delta(v)
        return u, v

    def OpenFlystel_inv(self, u, v):
        x = u - self._Q_delta(v)
        y = v + x ** self.alpha_inv
        x = x + self._Q_gamma(y)
        return x, y

    def ClosedFlystel(self, y, v):
        """Closed Flystel V : (y, v) -> (x, u), the verification form using the forward power map.
        Equivalent to the open Flystel on consistent values: H(x, y) = (u, v) iff V(y, v) = (x, u).
        See https://eprint.iacr.org/2022/840, Fig. 3b."""
        e = (y - v) ** self.alpha
        x = e + self._Q_gamma(y)
        u = e + self._Q_delta(v)
        return x, u

    def NonLinearLayer(self, state: list) -> list:
        """Open Flystel applied to each column (x_i, y_i)."""
        out = list(state)
        for i in range(self.l):
            out[i], out[self.l + i] = self.OpenFlystel(out[i], out[self.l + i])
        return out

    def NonLinearLayer_inv(self, state: list) -> list:
        out = list(state)
        for i in range(self.l):
            out[i], out[self.l + i] = self.OpenFlystel_inv(out[i], out[self.l + i])
        return out

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        for r in range(self.R):
            state = self.ConstantAddition(state, r)
            state = self.LinearLayer(state)
            state = self.NonLinearLayer(state)
        return self.LinearLayer(state)  # final degenerate round

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self.LinearLayer_inv(state)
        for r in reversed(range(self.R)):
            state = self.NonLinearLayer_inv(state)
            state = self.LinearLayer_inv(state)
            state = self.ConstantAddition_inv(state, r)
        return state

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------

    def compress_2_to_1(self, x: list, y: list) -> list:
        """Jive_2 compression of the x- and y-lane into l elements."""
        if len(x) != self.l or len(y) != self.l:
            raise ValueError(f"Invalid input sizes. Expected ({self.l},{self.l}), got ({len(x)},{len(y)})")
        return compress_jive(
            perm=self.permutation,
            inputs=[x,y],
            b=2,
            to_field=self.to_field,
        )

    def hash_sponge(self, data: list) -> list:
        # Domain separator sigma = 1 for rate-aligned non-empty messages; everything
        # else (including the empty message) is padded with a 1 followed by zeros.
        if len(data) % self.r == 0 and len(data) != 0:
            padded_data, sigma = list(data), 1
        else:
            (padded_data, _), sigma = pad_one(data, self.r, self.to_field), 0
        return hash_sponge_hirose(
            perm=self.permutation,
            data=padded_data,
            state_size=self.t,
            rate=self.r,
            capacity=self.c,
            digest_size=self.d,
            sigma=sigma,
            to_field=self.to_field,
        )

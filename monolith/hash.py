# hash.py
# Monolith: MonolithPerm (permutation) and its mode functions (MonolithHash, MonolithCompress).
# NOTE: uses a lookup-table S-box -- non-generic (cannot run symbolically over a polynomial ring).

from monolith.params import MonolithParams
from utils.matrix import matvecmul, vecadd, vecsub
from utils.lut import mixed_radix_decompose, mixed_radix_compose
from utils.primitive import Permutation, HashFunction, CompressionFunction

class MonolithPerm(Permutation):
    def __init__(self, params: MonolithParams):
        super().__init__(params)  # F, to_field, from_field, t, p, kappa, toy

        # Rounds
        self.R = params.R

        # Non-linear layers: _bars
        self.si = params.si
        self.LUTs = params.LUTs
        self.LUTs_inv = params.LUTs_inv
        self.u = params.u

        # Linear layer
        self.M = params.M
        self.M_inv = params.M_inv

        # Round constants (padded with a trailing zero row in params as the final 
        # round has no round constant addition)
        self.rcons = params.rcons


    # ---------------------------------------------------------------------------
    # Component functions
    # ---------------------------------------------------------------------------

    def constant_addition(self, state: list, r: int) -> list:
        return vecadd(state, self.rcons[r])

    def constant_addition_inv(self, state: list, r: int) -> list:
        return vecsub(state, self.rcons[r])

    def linear_layer(self, state: list, r: int) -> list:
        # Originally called Concrete in Monolith paper
        return matvecmul(self.M, state)

    def linear_layer_inv(self, state: list, r: int) -> list:
        return matvecmul(self.M_inv, state)

    def _bar(self, x):
        # Lookup-table S-box on one element (non-generic; see module note).
        digits = mixed_radix_decompose(x, self.si, self.from_field)
        new_digits = [self.LUTs[s][d] for s, d in zip(self.si, digits)]
        return mixed_radix_compose(new_digits, self.si, self.to_field)

    def _bar_inv(self, x):
        digits = mixed_radix_decompose(x, self.si, self.from_field)
        new_digits = [self.LUTs_inv[s][d] for s, d in zip(self.si, digits)]
        return mixed_radix_compose(new_digits, self.si, self.to_field)

    def _bars(self, state: list, r: int) -> list:
        # Round-independent (_bar applied to the first u branches every round), so r is unused.
        return [self._bar(x) if i < self.u else x for i, x in enumerate(state)]

    def _bars_inv(self, state: list, r: int) -> list:
        return [self._bar_inv(x) if i < self.u else x for i, x in enumerate(state)]

    def _bricks(self, state: list, r: int) -> list:
        # Round-independent, so r is accepted but unused.
        result = [state[0]]
        for i in range(1, self.t):
            result.append(state[i] + state[i - 1] ** 2)
        return result

    def _bricks_inv(self, state: list, r: int) -> list:
        result = [state[0]]
        for i in range(1, self.t):
            result.append(state[i] - result[i - 1] ** 2)
        return result
    
    def nonlinear_layer(self, state: list, r: int) -> list:
        state = self._bars(state, r)
        return self._bricks(state, r)

    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        state = self._bricks_inv(state, r)
        return self._bars_inv(state, r)

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
            state = self.linear_layer(state, r) # linear_layer called "Concrete" in Monolith paper
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
# Hash / compression functions
#
# MonolithPerm above is JUST the permutation. Each mode wraps a permutation:
#     P = MonolithPerm(params)
#     H = MonolithHash(P, params.sponge)      # SAFE sponge
#     C = MonolithCompress(P, params.comp)    # 2-to-1 truncation / Davies-Meyer (t=2d -> d)
# Compression is only defined for the t = 2d instances (comp=dict(a=2)); others comp=None.
# ---------------------------------------------------------------------------

class MonolithHash(HashFunction):
    SPONGE_KIND = "safe"

class MonolithCompress(CompressionFunction):
    COMP_KIND = "trunc"       # 2-to-1 truncation / Davies-Meyer (comp=dict(a=2))

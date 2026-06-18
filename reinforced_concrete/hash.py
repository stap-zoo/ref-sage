# hash.py
# ---------------------------------------------------------------------------
# Reinforced Concrete: the permutation (round function) and the hash modes built
# on it.
#
# Constructed from a fully-specified ReinforcedConcreteParams object; this class
# only *applies* the parameters, never derives or validates them.
#
# NOTE: the _bars component (and its per-element _bar helper) is a lookup-table S-box:
# it decomposes a field element into integer digits and applies a precomputed
# LUT. It is therefore inherently NON-generic -- it operates on the integer
# representation and cannot run symbolically over a polynomial ring, unlike the
# arithmetic component (_bricks, constant_addition, linear_layer). This is a 
# deliberate exception to the "generic component" contract.
# ---------------------------------------------------------------------------

from reinforced_concrete.params import ReinforcedConcreteParams
from utils.matrix import matvecmul, vecadd, vecsub, add_to_start
from utils.lut import mixed_radix_decompose, mixed_radix_compose
from utils.mode import compress_davies_meyer, hash_sponge, pad_zero

class ReinforcedConcrete:
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: ReinforcedConcreteParams):
        self.F = params.F
        self.to_field = params.to_field
        self.from_field = params.from_field
        self.t = params.t

        # Rounds
        self.R_pre = params.R_pre
        self.R_bars = params.R_bars
        self.R = params.R

        # Non-linear layers: _bricks
        self.alpha = params.alpha
        self.alpha_inv = params.alpha_inv
        self.a_coeffs = params.a_coeffs
        self.b_coeffs = params.b_coeffs

        # Non-linear layers: _bars
        self.si = params.si
        self.LUT = params.LUT
        self.LUT_inv = params.LUT_inv

        # Linear layer
        self.M = params.M
        self.M_inv = params.M_inv

        # Round constants
        self.rcons = params.rcons

        # Hash modes
        self.r = params.r
        self.c = params.c
        self.d = params.d

    # ---------------------------------------------------------------------------
    # Component functions
    # ---------------------------------------------------------------------------

    def _is_bars_round(self, r):
        return self.R_pre <= r < self.R_pre + self.R_bars # off-by-one due to initial round constant addition

    def constant_addition(self, state: list, r: int) -> list:
        return vecadd(state, self.rcons[r])

    def constant_addition_inv(self, state: list, r: int) -> list:
        return vecsub(state, self.rcons[r])

    def linear_layer(self, state: list, r: int) -> list:
        return matvecmul(self.M, state)

    def linear_layer_inv(self, state: list, r: int) -> list:
        return matvecmul(self.M_inv, state)

    def _F(self, val, i):
        return val ** 2 + self.a_coeffs[i] * val + self.b_coeffs[i]

    def _bricks(self, state: list, r: int) -> list:
        # Round-independent, so r is accepted but unused.
        result = [state[0] ** self.alpha]
        for i in range(1, self.t):
            result.append(state[i] * self._F(state[i - 1], i - 1))
        return result

    def _bricks_inv(self, state: list, r: int) -> list:
        result = [state[0] ** self.alpha_inv]
        for i in range(1, self.t):
            result.append(state[i] * self._F(result[i - 1], i - 1) ** (-1))
        return result

    def _bar(self, x):
        # Lookup-table S-box on one element (non-generic; see module note).
        digits = mixed_radix_decompose(x, self.si, self.from_field)
        new_digits = [self.LUT[d] for d in digits]
        return mixed_radix_compose(new_digits, self.si, self.to_field)

    def _bar_inv(self, x):
        digits = mixed_radix_decompose(x, self.si, self.from_field)
        new_digits = [self.LUT_inv[d] for d in digits]
        return mixed_radix_compose(new_digits, self.si, self.to_field)

    def _bars(self, state: list, r: int) -> list:
        # Round-independent (_bar applied to every branch each round), so r is unused.
        return [self._bar(x) for x in state]

    def _bars_inv(self, state: list, r: int) -> list:
        return [self._bar_inv(x) for x in state]
    
    def nonlinear_layer(self, state: list, r: int) -> list:
        if self._is_bars_round(r):
            return self._bars(state, r)
        else:
            return self._bricks(state, r)
    
    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        if self._is_bars_round(r):
            return self._bars_inv(state, r)
        else:
            return self._bricks_inv(state, r)

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps (the initial Concrete/affine map, done once)
    # ---------------------------------------------------------------------------

    def _pre_rounds(self, state: list) -> list:
        # "Concrete" layer is applied, that is, linear_layer + constant_addition
        state = self.linear_layer(state, 0)
        return self.constant_addition(state, 0)

    def _pre_rounds_inv(self, state: list) -> list:
        state = self.constant_addition_inv(state, 0)
        return self.linear_layer_inv(state, 0)

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
        for i in range(self.R):
            state = self.nonlinear_layer(state, i)
            state = self.linear_layer(state, i)
            state = self.constant_addition(state, i + 1) # off-by-one due to initial round constant addition
        return self._post_rounds(state)

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for i in reversed(range(self.R)):
            state = self.constant_addition_inv(state, i + 1) # off-by-one due to initial round constant addition
            state = self.linear_layer_inv(state, i)
            state = self.nonlinear_layer_inv(state, i)
        return self._pre_rounds_inv(state)

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------

    def compress_2_to_1(self, x1: list, x2:list) -> list:
        """2-to-1 compression defined for large state sizes triple the digest size"""
        if self.t != 3 * self.d:
            raise ValueError(f"Compression mode not defined for state size {self.t} and digest size {self.d}.")
        if len(x1) != self.d or len(x2) != self.d:
            raise ValueError(f"Invalid input sizes. Expected ({self.d},{self.d}), got ({len(x1)},{len(x2)})")
        return compress_davies_meyer(
            perm=self.permutation, 
            x_m=x1 + x2, 
            x_c=[self.F.zero()] * self.d, 
            digest_size=self.d, 
            to_field=self.to_field
        )

    def hash_sponge(self, data: list) -> list:
        padded_data, _ = pad_zero(data, self.r, self.to_field)
        IV = [self.F.zero()] * self.c
        return hash_sponge(
            perm=self.permutation, 
            data=padded_data, 
            state_size=self.t, 
            rate=self.r, 
            capacity=self.c, 
            digest_size=self.d, 
            IV=IV,
            absorb=add_to_start,
            to_field=self.to_field
        )

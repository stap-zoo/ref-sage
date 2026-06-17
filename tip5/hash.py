# hash.py
# ---------------------------------------------------------------------------
# Tip5 (and TIP4 / TIP4'): the permutation (round function) and the hash modes
# built on it.
#
# Constructed from a fully-specified Tip5Params object; this class only *applies*
# the parameters, never derives or validates them.
#
# NOTE: the split-and-lookup S-box (_sl_sbox / _sl_sbox_inv, used by nonlinear_layer on
# the first u branches) decomposes a field element into bytes and applies a
# precomputed LUT. It is therefore inherently NON-generic -- it operates on the
# integer representation and cannot run symbolically over a polynomial ring,
# unlike the power-map branch of nonlinear_layer and the AffineLayer. This is a
# deliberate exception to the "generic component" contract.
# ---------------------------------------------------------------------------

from tip5.params import Tip5Params, Tip4Params, Tip4PrimeParams
from utils import matvecmul, vecadd, vecsub, mixed_radix_decompose, mixed_radix_compose, add_to_start
from modes import hash_sponge, pad_fixed_length


class Tip5:
    def __init__(self, params: Tip5Params):
        self.F = params.F
        self.t = params.t
        self.to_field = params.to_field
        self.from_field = params.from_field

        # Rounds
        self.R = params.R

        # Non-linear layer: split-and-lookup S-boxes and power maps
        self.u = params.u
        self.si = params.si
        self.LUT = params.LUT
        self.LUT_inv = params.LUT_inv
        self.mont_R = params.mont_R
        self.mont_R_inv = params.mont_R_inv
        self.alpha = params.alpha
        self.alpha_inv = params.alpha_inv

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

    def _sl_sbox(self, x):
        """Split-and-lookup S-box (non-generic; see module note): convert to Montgomery form,
        substitute each byte via the lookup table, convert back."""
        digits = mixed_radix_decompose(self.mont_R * x, self.si, self.from_field)
        new_digits = [self.LUT[d] for d in digits]
        return self.mont_R_inv * mixed_radix_compose(new_digits, self.si, self.to_field)

    def _sl_sbox_inv(self, x):
        digits = mixed_radix_decompose(self.mont_R * x, self.si, self.from_field)
        new_digits = [self.LUT_inv[d] for d in digits]
        return self.mont_R_inv * mixed_radix_compose(new_digits, self.si, self.to_field)

    def nonlinear_layer(self, state: list, r: int) -> list:
        """Split-and-lookup on the first u elements, power map on the rest."""
        return [self._sl_sbox(x) if i < self.u else x ** self.alpha for i, x in enumerate(state)]

    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        return [self._sl_sbox_inv(x) if i < self.u else x ** self.alpha_inv for i, x in enumerate(state)]

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps (Tip5 does no work outside the loop: identities)
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

    def hash_sponge(self, data: list, c_value: int =1) -> list:
        """Fixed-length hash: a single rate-sized block, capacity initialized to c_value."""
        if len(data) != self.r:
            raise ValueError(f"Invalid input size. Expected {self.r}, got {len(data)}")
        padded_data, _ = pad_fixed_length(data, self.r, self.to_field)
        IV = [self.to_field(c_value)] * self.c
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


# ---------------------------------------------------------------------------
# Tip4 and Tip4'
# ---------------------------------------------------------------------------

class Tip4(Tip5):

    def __init__(self, params: Tip4Params):
        super().__init__(params)

    def compress_4_to_1(self, x1: list, x2: list, x3: list, x4: list) -> list:
        b = 4
        m = self.t // b
        inputs = [x1, x2, x3, x4] # b lists, each of size m = t/b
        if any(len(xi) != m for xi in inputs):
            raise ValueError(f"Invalid input sizes. Expected all of length {m}")
        return compress_jive(
            perm=self.permutation,
            inputs=inputs,
            b=b,
            to_field=self.to_field,
        )

class Tip4Prime(Tip5):

    def __init__(self, params: Tip4Params):
        super().__init__(params)

    def compress_3_to_1(self, x1: list, x2: list, x3: list) -> list:
        b = 3
        m = self.t // b
        inputs = [x1, x2, x3] # b lists, each of size m = t/b
        if any(len(xi) != m for xi in inputs):
            raise ValueError(f"Invalid input sizes. Expected all of length {m}")
        return compress_jive(
            perm=self.permutation,
            inputs=inputs,
            b=b,
            to_field=self.to_field,
        )
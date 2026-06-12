from tip5.params import Tip5Params
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

    def SBox(self, x):
        """Split-and-lookup S-box: convert to Montgomery form, substitute each byte via the lookup table, convert back."""
        digits = mixed_radix_decompose(self.mont_R * x, self.si, self.from_field)
        return self.mont_R_inv * mixed_radix_compose([self.LUT[d] for d in digits], self.si, self.to_field)

    def SBox_inv(self, x):
        digits = mixed_radix_decompose(self.mont_R * x, self.si, self.from_field)
        return self.mont_R_inv * mixed_radix_compose([self.LUT_inv[d] for d in digits], self.si, self.to_field)

    def NonLinearLayer(self, state: list) -> list:
        """Split-and-lookup on the first s elements, power map on the rest."""
        return [self.SBox(x) if i < self.u else x ** self.alpha for i, x in enumerate(state)]

    def NonLinearLayer_inv(self, state: list) -> list:
        return [self.SBox_inv(x) if i < self.u else x ** self.alpha_inv for i, x in enumerate(state)]

    def AffineLayer(self, state: list, round_idx: int) -> list:
        """MDS matrix-vector product followed by round-constant addition."""
        return vecadd(matvecmul(self.M, state), self.rcons[round_idx])

    def AffineLayer_inv(self, state: list, round_idx: int) -> list:
        return matvecmul(self.M_inv, vecsub(state, self.rcons[round_idx]))

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        for r in range(self.R):
            state = self.NonLinearLayer(state)
            state = self.AffineLayer(state, r)
        return state

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        for r in reversed(range(self.R)):
            state = self.AffineLayer_inv(state, r)
            state = self.NonLinearLayer_inv(state)
        return state

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

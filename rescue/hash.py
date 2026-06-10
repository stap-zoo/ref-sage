from rescue.params import RescueParams
from utils import matvecmul, vecadd, vecsub, add_to_start
from modes import pad_one, pad_fixed_length, hash_sponge


class Rescue:
    def __init__(self, params: RescueParams):
        self.F = params.F
        self.t = params.t

        # Rounds
        self.R = params.R

        # Non-linear layer
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

        # Field conversion helpers
        self.to_field = params.to_field
        self.from_field = params.from_field

    # ---------------------------------------------------------------------------
    # Component functions
    # ---------------------------------------------------------------------------

    def SBox(self, state: list) -> list:
        return [x ** self.alpha for x in state]

    def SBox_inv(self, state: list) -> list:
        return [x ** self.alpha_inv for x in state]

    def AffineLayer(self, state: list, round_idx: int) -> list:
        state = matvecmul(self.M, state)
        return vecadd(state, self.rcons[round_idx])

    def AffineLayer_inv(self, state: list, round_idx: int) -> list:
        state = vecsub(state, self.rcons[round_idx])
        return matvecmul(self.M_inv, state)

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")
        
        # Initial round constant addition
        state = vecadd(state, self.rcons[0])

        for r in range(self.R):
            state = self.SBox_inv(state)
            state = self.AffineLayer(state, 2 * r + 1)
            state = self.SBox(state)
            state = self.AffineLayer(state, 2 * r + 2)

        return state

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        for r in reversed(range(self.R)):
            state = self.AffineLayer_inv(state, 2 * r + 2)
            state = self.SBox_inv(state)
            state = self.AffineLayer_inv(state, 2 * r + 1)
            state = self.SBox(state)

        return vecsub(state, self.rcons[0])

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------

    def hash_sponge(self, data: list, variable_length: bool=True) -> list:
        """Variable or fixed input length hashing using Sponge mode. Fixed input must be 
        multiple of rate. No domain separation."""
        pad = pad_one if variable_length else pad_fixed_length
        padded_data, _ = pad(data, self.r, self.to_field)
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

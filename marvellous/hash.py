from marvellous.params import RescueParams, RescuePrimeParams, RescuePrimeOptimizedParams
from utils import matvecmul, vecadd, vecsub, add_to_start, replace_start
from modes import pad_one, pad_fixed_length, pad_one_conditional, hash_sponge


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

# ---------------------------------------------------------------------------
# Rescue Prime
# ---------------------------------------------------------------------------

class RescuePrime(Rescue):

    def permutation(self, state: list) -> list:
        """Similar to Rescue.permutation, but no initial round constant addition, and SBox/Sbox_inv order switched."""
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")
        
        for r in range(self.R):
            state = self.SBox(state)
            state = self.AffineLayer(state, 2 * r)
            state = self.SBox_inv(state)
            state = self.AffineLayer(state, 2 * r + 1)

        return state

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        for r in reversed(range(self.R)):
            state = self.AffineLayer_inv(state, 2 * r + 1)
            state = self.SBox(state)
            state = self.AffineLayer_inv(state, 2 * r)
            state = self.SBox_inv(state)

        return state

# ---------------------------------------------------------------------------
# Rescue Prime Optimized (RPO)
# ---------------------------------------------------------------------------

class RescuePrimeOptimized(RescuePrime):

    def permutation(self, state: list) -> list:
        """Similar to RescuePrime.permutation, but order of application of AffineLayer and SBox/SBox_inv switched."""
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        for r in range(self.R):
            state = self.AffineLayer(state, 2 * r)
            state = self.SBox(state)
            state = self.AffineLayer(state, 2 * r + 1)
            state = self.SBox_inv(state)

        return state

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        for r in reversed(range(self.R)):
            state = self.SBox(state)
            state = self.AffineLayer_inv(state, 2 * r + 1)
            state = self.SBox_inv(state)
            state = self.AffineLayer_inv(state, 2 * r)

        return state
    
    def hash_sponge(self, data: list) -> list:
        """Uses domain separation.

        NOTE: the spec's sponge has the capacity in the first c state elements
        and the rate in the remaining r elements (overwritten on each absorption,
        squeeze taken from the rate part). modes.hash_sponge instead places the
        rate first and the capacity last, i.e. capacity and rate are exchanged
        relative to the spec."""
        padded_data, was_aligned = pad_one_conditional(data, self.r, self.to_field)
        IV = [self.to_field(0 if was_aligned else 1)] + [self.F.zero()] * (self.c - 1)
        return hash_sponge(
            perm=self.permutation,
            data=padded_data,
            state_size=self.t,
            rate=self.r,
            capacity=self.c,
            digest_size=self.d,
            IV=IV,
            absorb=replace_start,
            to_field=self.to_field
        )

    def compress_2_to_1(self, x1: list, x2: list) -> list:
        """Merge two digests into one (Merkle node): single-permutation path."""
        if len(x1) != self.r // 2 or len(x2) != self.r // 2:
            raise ValueError(f"Inputs must be digests of {self.r // 2} elements, got {len(x1)} and {len(x2)}.")
        return self.hash_sponge(x1 + x2)

# ---------------------------------------------------------------------------
# XHASH
# ---------------------------------------------------------------------------

# TODO
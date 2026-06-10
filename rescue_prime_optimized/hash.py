from rescue_prime.hash import RescuePrime
from rescue_prime_optimized.params import RescuePrimeOptimizedParams
from modes import pad_one_conditional, hash_sponge
from utils import replace_start

class RescuePrimeOptimized(RescuePrime):
    def __init__(self, params: RescuePrimeOptimizedParams):
        super().__init__(params)

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

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

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------
    
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
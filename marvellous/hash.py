# hash.py
# ---------------------------------------------------------------------------
# Marvellous family: the Rescue permutation and its RescuePrime / RPO variants,
# plus the hash modes built on them.
#
# Each class is constructed from a fully-specified params object and only
# *applies* the parameters (no derivation/validation).
#
# NOTE: Rescue / RescuePrime / RPO use double rounds (each "round" consists of 
# two SPN rounds), while XHash uses triple rounds (each "round" consists of 
# three SPN rounds). The parameter R counts the primitive "rounds", not the SPN rounds.
# ---------------------------------------------------------------------------

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

    def constant_addition(self, state: list, r: int) -> list:
        return vecadd(state, self.rcons[r])

    def constant_addition_inv(self, state: list, r: int) -> list:
        return vecsub(state, self.rcons[r])

    def linear_layer(self, state: list, r: int) -> list:
        return matvecmul(self.M, state)

    def linear_layer_inv(self, state: list, r: int) -> list:
        return matvecmul(self.M_inv, state)

    def nonlinear_layer(self, state: list, r: int) -> list:
        return [x ** self.alpha for x in state]

    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        return [x ** self.alpha_inv for x in state]

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps
    # ---------------------------------------------------------------------------

    def _pre_rounds(self, state: list) -> list:
        return state

    def _pre_rounds_inv(self, state: list) -> list:
        return state

    def _post_rounds(self, state: list) -> list:
        return self.constant_addition(state, -1)  # final round constant addition

    def _post_rounds_inv(self, state: list) -> list:
        return self.constant_addition_inv(state, -1)

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)
        for r in range(2 * self.R): # Rescue uses double rounds
            if r % 2 == 0: # (B) part of double-round
                state = self.constant_addition(state, r)
                state = self.nonlinear_layer_inv(state, r) # backward for even rounds
                state = self.linear_layer(state, r)
                
            else: # (F) part of double-round
                state = self.constant_addition(state, r)
                state = self.nonlinear_layer(state, r) # forward for odd rounds
                state = self.linear_layer(state, r)
        return self._post_rounds(state)

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for r in reversed(range(2 * self.R)):
            if r % 2 == 0:
                state = self.linear_layer_inv(state, r)
                state = self.nonlinear_layer(state, r)
                state = self.constant_addition_inv(state, r)
            else:
                state = self.linear_layer_inv(state, r)
                state = self.nonlinear_layer_inv(state, r)
                state = self.constant_addition_inv(state, r)
        return self._pre_rounds_inv(state)

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------

    def hash_sponge(self, data: list, variable_length: bool = True) -> list:
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

    def __init__(self, params: RescuePrimeParams):
        super().__init__(params)

    def _post_rounds(self, state: list) -> list:
        return state # no final round constant addition (reordered layers)

    def _post_rounds_inv(self, state: list) -> list:
        return state

    def permutation(self, state: list) -> list:
        """Similar to Rescue, but nonlinear_layer/nonlinear_layer_inv order switched 
        for even/odd rounds and round constant addition now at the end of each round 
        (thus no final round constant addition in _post_rounds)."""
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)
        for r in range(2 * self.R): # RescuePrime uses double rounds
            if r % 2 == 0: # (F) part of double-round
                state = self.nonlinear_layer(state, r) # forward for even rounds
                state = self.linear_layer(state, r)
                state = self.constant_addition(state, r)
            else: # (B) part of double-round
                state = self.nonlinear_layer_inv(state, r) # backward for odd rounds
                state = self.linear_layer(state, r)
                state = self.constant_addition(state, r)
        return self._post_rounds(state)

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for r in reversed(range(2 * self.R)):
            if r % 2 == 0:
                state = self.constant_addition_inv(state, r)
                state = self.linear_layer_inv(state, r)
                state = self.nonlinear_layer_inv(state, r)
            else:
                state = self.constant_addition_inv(state, r)
                state = self.linear_layer_inv(state, r)
                state = self.nonlinear_layer(state, r)
        return self._pre_rounds_inv(state)

# ---------------------------------------------------------------------------
# Rescue Prime Optimized (RPO)
# ---------------------------------------------------------------------------

class RescuePrimeOptimized(RescuePrime):

    def __init__(self, params: RescuePrimeOptimizedParams):
        super().__init__(params)

    def permutation(self, state: list) -> list:
        """Similar to RescuePrime, but order of application of AffineLayer and nonlinear_layer/nonlinear_layer_inv switched."""
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)
        for r in range(2 * self.R): # RescuePrimeOptimized uses double rounds
            if r % 2 == 0: # (F) part of double-round
                state = self.linear_layer(state, r)
                state = self.constant_addition(state, r)
                state = self.nonlinear_layer(state, r) # forward for even rounds
            else: # (B) part of double-round
                state = self.linear_layer(state, r)
                state = self.constant_addition(state, r)
                state = self.nonlinear_layer_inv(state, r) # backward for odd rounds
        return self._post_rounds(state)

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for r in reversed(range(2 * self.R)):
            if r % 2 == 0:
                state = self.nonlinear_layer_inv(state, r) 
                state = self.constant_addition_inv(state, r)
                state = self.linear_layer_inv(state, r)
            else:
                state = self.nonlinear_layer(state, r) 
                state = self.constant_addition_inv(state, r)
                state = self.linear_layer_inv(state, r)
        return self._pre_rounds_inv(state)

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

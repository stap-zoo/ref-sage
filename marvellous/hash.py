# hash.py
# ---------------------------------------------------------------------------
# Marvellous family: the Rescue permutation and its RescuePrime / RPO variants,
# plus the hash modes built on them.
#
# Each class is constructed from a fully-specified params object.
#
# NOTE: Rescue / RescuePrime / RPO use double rounds (each "round" consists of 
# two SPN rounds), while XHash uses triple rounds (each "round" consists of 
# three SPN rounds). The parameter R counts the primitive "rounds", not the SPN rounds.
# ---------------------------------------------------------------------------

from marvellous.params import RescueParams, RescuePrimeParams, RescuePrimeOptimizedParams, XHashParams
from utils.matrix import matvecmul, vecadd, vecsub
from utils.poly import eval_aos


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
        self.sponge = params.sponge

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
        # x -> x^alpha, called pi_1 in Rescue paper, pi_0 in XHash paper
        return [x ** self.alpha for x in state] 

    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        # x -> x^(1/alpha), called pi_0 in Rescue paper, pi_1 in XHash paper
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

        # Each round of Rescue is a double SPN round, yielding: (BF)(BF)...(BF)(C)
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
        return self.sponge.hash(self.permutation, data, input_len_fixed=(not variable_length))

# ---------------------------------------------------------------------------
# Rescue Prime
# ---------------------------------------------------------------------------

class RescuePrime(Rescue):

    def __init__(self, params: RescuePrimeParams):
        super().__init__(params)

    def _pre_rounds(self, state: list) -> list:
        return state

    def _pre_rounds_inv(self, state: list) -> list:
        return state

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

        # Each round of RescuePrime is a double SPN round, yielding: (FB)(FB)...(FB)
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

        # Each round of RescuePrime is a double SPN round, yielding: (FB)(FB)...(FB)
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

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------

    def hash_sponge(self, data: list) -> list:
        return self.sponge.hash(self.permutation, data)

    def compress_2_to_1(self, x1: list, x2: list) -> list:
        """Merge two digests into one (Merkle node): single-permutation path."""
        r = self.sponge.r # TODO replace with compression notation
        if len(x1) != r // 2 or len(x2) != r // 2:
            raise ValueError(f"Inputs must be digests of {r // 2} elements, got {len(x1)} and {len(x2)}.")
        return self.hash_sponge(x1 + x2)

# ---------------------------------------------------------------------------
# XHASH
# ---------------------------------------------------------------------------

class XHash(RescuePrimeOptimized):

    def __init__(self, params: XHashParams):
        super().__init__(params)

        # Nonlinear layer (old)
        # For aggressive version with S-box skipping, [mod,rem] != None
        # Indices to be skipped saved in skipbox_idx
        self.skipbox = params.skipbox
        self.skipbox_idx = params.skipbox_idx

        # Nonlinear layer (new)
        # Coordinate polynomials of power map over degree-3 extension field (with modulus fmod), 
        # saved in AoS format (see utils.poly)
        self.cpolys = params.cpolys
        self.fmod = params.fmod

        # TODO make sure t is divisible by 3?
    
    def _pre_rounds(self, state: list) -> list:
        return state

    def _pre_rounds_inv(self, state: list) -> list:
        return state

    def _post_rounds(self, state: list) -> list:
        # final matrix multiplication and round constant addition
        state = self.linear_layer(state, -1)
        return self.constant_addition(state, -1)

    def _post_rounds_inv(self, state: list) -> list:
        state = self.constant_addition_inv(state, -1)
        return self.linear_layer_inv(state, -1)
    
    def _sbox_P3(self, point) -> list:
        """Forward XHash S-box pi_2 over the degree-3 extension field.

        `point` is a list of 3 field coordinates [x0, x1, x2] representing one
        extension-field element x0 + x1*X + x2*X^2. Applying x -> x^alpha in the
        extension is precomputed as 3 coordinate polynomials (self.cpolys, in AoS
        form); evaluating each at `point` gives the 3 output coordinates.
        """
        return [eval_aos(terms, point) for terms in self.cpolys]


    def _sbox_P3_inv(self, point) -> list:
        """Inverse XHash S-box pi_2^{-1} (x -> x^{1/alpha}) over the extension.

        `point` is a list of 3 field coordinates [x0, x1, x2], mirroring _sbox_P3.

        TODO Not implemented: the inverse power map would need its own coordinate
        polynomials (self.cpolys_inv), derived from the inverse exponent 1/alpha mod (p^3 - 1).
        """
        raise NotImplementedError("Inversion of power map over extension field not implemented.")
        # return [eval_aos(terms, point) for terms in self.cpolys_inv]


    def nonlinear_layer(self, state: list, r: int) -> list:
        """Apply the round-`r` non-linear layer to the full state of t elements.

        The schedule mixes two S-boxes by round index:
        - every third round (r % 3 == 2): the XHash extension S-box pi_2, applied to each consecutive triple of state elements;
        - all other rounds: the standard Rescue forward S-box x -> x^alpha, applied element-wise, skipping the positions in self.skipbox_idx.
        Returns a new state list of the same length t.
        """
        if r % 3 == 2:
            # XHash S-box pi_2: group the state into consecutive triples [0,1,2], [3,4,5], ... , 
            # apply the extension S-box to each triple, and flatten the 3-coordinate outputs back into a single flat state list.
            return [coord for i in range(0, self.t, 3) for coord in self._sbox_P3(state[i:i+3])]
        else:
            # Standard Rescue forward S-box pi_1: x -> x^alpha, skipping indices in skipbox_idx
            return [(state[i] if i in self.skipbox_idx else state[i] ** self.alpha)
                    for i in range(self.t)] 

    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        if r % 3 == 2:
            return [coord for i in range(0, self.t, 3) for coord in self._sbox_P3_inv(state[i:i+3])]
        else:
            # Standard Rescue backward S-box pi_1: x -> x^(1/alpha), skipping indices in skipbox_idx
            return [(state[i] if i in self.skipbox_idx else state[i] ** self.alpha_inv) for i in range(self.t)]

    def permutation(self, state: list) -> list:
        """Similar to RescuePrime, but order of application of AffineLayer and nonlinear_layer/nonlinear_layer_inv switched."""
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        # Every other round of XHASH is a double SPN round, yielding: (FB)(P3)...(FB)(P3)(MC)
        state = self._pre_rounds(state)
        for r in range(int(1.5 * self.R)):
            if r % 3 == 0: # (F) part of double-round
                state = self.constant_addition(state, r)
                state = self.linear_layer(state, r)
                state = self.nonlinear_layer(state, r)
            elif r % 3 == 1: # (B) part of double-round
                state = self.linear_layer(state, r)
                state = self.constant_addition(state, r)
                state = self.nonlinear_layer_inv(state, r)
            else: # (P3) part - no linear layer since nonlinear layer does some mixing
                state = self.constant_addition(state, r)
                state = self.nonlinear_layer(state, r)
        return self._post_rounds(state)

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        # Every other round of XHASH is a double SPN round, yielding: (FB)(P3)...(FB)(P3)(MC)
        state = self._pre_rounds(state)
        for r in reversed(range(int(1.5 * self.R))):
            if r % 3 == 0: # (F^-1) part of double-round
                state = self.nonlinear_layer_inv(state, r)
                state = self.linear_layer_inv(state, r)
                state = self.constant_addition_inv(state, r)
            elif r % 3 == 1: # (B^-1) part of double-round
                state = self.nonlinear_layer(state, r)
                state = self.constant_addition_inv(state, r)
                state = self.linear_layer_inv(state, r)
            else: # (P3^-1) part
                state = self.nonlinear_layer_inv(state, r)
                state = self.constant_addition_inv(state, r)
        return self._post_rounds(state)
# hash.py
# Tip5: Tip5Perm, Tip4Perm, Tip4PrimePerm (permutation) and their mode functions (Tip5Hash; Tip4Hash, Tip4Compress; Tip4PrimeHash, Tip4PrimeCompress).
# NOTE: uses a lookup-table S-box -- non-generic (cannot run symbolically over a polynomial ring).

from tip5.params import Tip5Params, Tip4Params, Tip4PrimeParams
from utils.matrix import matvecmul, vecadd, vecsub
from utils.lut import mixed_radix_decompose, mixed_radix_compose
from utils.primitive import Permutation, HashFunction, CompressionFunction


class Tip5Perm(Permutation):
    def __init__(self, params: Tip5Params):
        super().__init__(params)  # F, to_field, from_field, t, p, kappa, toy

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

    def permute(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)
        for r in range(self.R):
            state = self.nonlinear_layer(state, r)
            state = self.linear_layer(state, r)
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
# Tip4 and Tip4' permutations. They share Tip5's round-function *structure* (the same
# component layers and permute loop), but differ in several parameters, all carried by their
# params classes: the state size t (Tip4: 16, Tip4': 12), the MDS matrix (Tip4' uses RPO's
# circulant), the round-constant derivation label (rcons differ), the security level kappa, the
# sponge sizes, and the mode set (Tip4/Tip4' add a Jive compression; Tip5 has none).
# ---------------------------------------------------------------------------

class Tip4Perm(Tip5Perm):
    def __init__(self, params: Tip4Params):
        super().__init__(params)

class Tip4PrimePerm(Tip5Perm):
    def __init__(self, params: Tip4PrimeParams):
        super().__init__(params)


# ---------------------------------------------------------------------------
# Hash / compression functions
#
# The permutations above are JUST permutations; each variant owns its mode functions. All three
# hashes are the same length-encoded ("cle") sponge, called with fixed-length input, e.g.
# H.hash(data, input_len_fixed=True, c_val=1). Only Tip4 / Tip4' define a compression -- Jive,
# with the arity from the comp dict (Tip4: a=4, Tip4': a=3); Tip5 has NO compression:
#     P = Tip5Perm(params);      H = Tip5Hash(P, params.sponge)                                  # Tip5: sponge only
#     P = Tip4Perm(params);      H = Tip4Hash(P, params.sponge);      C = Tip4Compress(P, params.comp)       # Jive_4: 4m -> m
#     P = Tip4PrimePerm(params); H = Tip4PrimeHash(P, params.sponge); C = Tip4PrimeCompress(P, params.comp)  # Jive_3: 3m -> m
# ---------------------------------------------------------------------------

class Tip5Hash(HashFunction):
    SPONGE_KIND = "cle"

class Tip4Hash(HashFunction):
    SPONGE_KIND = "cle"

class Tip4Compress(CompressionFunction):
    COMP_KIND = "jive"        # Jive_4 (comp=dict(a=4))

class Tip4PrimeHash(HashFunction):
    SPONGE_KIND = "cle"

class Tip4PrimeCompress(CompressionFunction):
    COMP_KIND = "jive"        # Jive_3 (comp=dict(a=3))
# hash.py
# Polocolo: PolocoloPerm (permutation) and its mode functions (PolocoloHash).
# NOTE: uses a lookup-table S-box -- non-generic (cannot run symbolically over a polynomial ring).

from polocolo.params import PolocoloParams
from utils.primitive import Permutation, HashFunction
from utils.matrix import matvecmul, vecadd, vecsub


class PolocoloPerm(Permutation):
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: PolocoloParams):
        # General settings (F, to_field, from_field, t, p, kappa, toy copied by Permutation)
        super().__init__(params)

        # Rounds
        self.R = params.R

        # Non-linear layer: power-residue S-box
        self.m = params.m
        self.ann = params.ann          # the "annihilator" exponent (p-1)/m
        self.LUT = params.LUT
        self.LUT_inv = params.LUT_inv

        # Linear layer
        self.M = params.M
        self.M_inv = params.M_inv

        # Round constants
        self.rcons = params.rcons


    # ---------------------------------------------------------------------------
    # Component layers
    # ---------------------------------------------------------------------------

    def constant_addition(self, state: list, r: int) -> list:
        return vecadd(state, self.rcons[r])

    def constant_addition_inv(self, state: list, r: int) -> list:
        return vecsub(state, self.rcons[r])

    def linear_layer(self, state: list, r: int) -> list:
        return matvecmul(self.M, state)

    def linear_layer_inv(self, state: list, r: int) -> list:
        return matvecmul(self.M_inv, state)

    def _sbox(self, x):
        # S(x) = x^{-1} * T[x^((p-1)/m)]  (lookup-based; see module note).
        #
        # x^(p-2) is the field inverse extended with 0 -> 0, and T[0] = 0, so the
        # S(0) = 0 case of the specification needs no branch.
        #
        # The lookup table T has m+1 entries, indexed by the m-th power residue
        # x^((p-1)/m), which takes the m+1 values {g^(r(p-1)/m) : 0 <= r < m} u {0}:
        #
        #     T[g^(r*(p-1)/m)] = g^(r*m + sigma(r) + r),      T[0] = 0,
        #
        # where sigma is a fixed permutation on {0, ..., m-1}. Writing x = g^(q*m + r), 
        # the trailing +r in the table exponent cancels the -r from x^(-1) = g^(-q*m - r), 
        # so that
        #
        #     x^(-1) * T[...] = g^(-q*m - r) * g^(r*m + sigma(r) + r)
        #                     = g^(-q*m + r*m + sigma(r)),
        #
        # which is exactly S(x) from Eq. (1) of the Polocolo spec (Sec. 3.2).
        return x ** (self.p - 2) * self.LUT[self.from_field(x ** self.ann)]

    def _sbox_inv(self, y):
        # The inverse S-box has the same shape with the table T_inv keyed by the
        # power residue of y (see utils.lut.power_residue_lut_inv for the algebra).
        return y ** (self.p - 2) * self.LUT_inv[self.from_field(y ** self.ann)]

    def nonlinear_layer(self, state: list, r: int) -> list:
        # Substitution step: the S-box is applied to every state element.
        return [self._sbox(x) for x in state]

    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        return [self._sbox_inv(y) for y in state]

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps
    # ---------------------------------------------------------------------------

    def _pre_rounds(self, state: list) -> list:
        return state

    def _pre_rounds_inv(self, state: list) -> list:
        return state

    def _post_rounds(self, state: list) -> list:
        # LinLayer^(R): the final linear layer carries no round constant (c^(R) = 0,
        # it would not affect security), so it is the bare matrix product.
        return matvecmul(self.M, state)

    def _post_rounds_inv(self, state: list) -> list:
        return matvecmul(self.M_inv, state)

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permute(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)
        for r in range(self.R):
            state = self.linear_layer(state, r)
            state = self.constant_addition(state, r)
            state = self.nonlinear_layer(state, r)
        return self._post_rounds(state)

    def permute_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for r in reversed(range(self.R)):
            state = self.nonlinear_layer_inv(state, r)
            state = self.constant_addition_inv(state, r)
            state = self.linear_layer_inv(state, r)
        return self._pre_rounds_inv(state)


# ---------------------------------------------------------------------------
# Hash function
#
# PolocoloPerm above is JUST the permutation. Polocolo is sponge-only (no compression):
#     P = PolocoloPerm(params)
#     H = PolocoloHash(P, params.sponge)
#     H.hash(data, input_len_fixed=True)
# ---------------------------------------------------------------------------

class PolocoloHash(HashFunction):
    SPONGE_KIND = "sponge2"



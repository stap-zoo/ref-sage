# hash.py
# ---------------------------------------------------------------------------
# Polocolo: the permutation (round function) and the hash mode built on it.
#
# Constructed from a fully-specified PolocoloParams object; this class only
# *applies* the parameters, never derives or validates them. The design goal is
# fidelity to the specification (https://eprint.iacr.org/2025/926, Section 4.2),
# not speed.
#
# The nonlinear layer applies the power-residue S-box 
# 
#   S(x) = x^{-1} * T[x^((p-1)/m)] 
# 
# to every state element.
#
# NOTE: the S-box (_sbox / _sbox_inv) is a lookup-table component: it keys a
# precomputed table by the integer value of the power residue x^((p-1)/m). It is
# therefore inherently NON-generic -- it branches on the value of a state element
# and cannot run symbolically over a polynomial ring, unlike constant_addition
# and linear_layer. This is a deliberate exception to the "generic component"
# contract, shared with the other lookup-based schemes (Reinforced Concrete,
# Monolith, Skyscraper); algebraic models replace the lookup by the constraint
# x*y = g^((m+1)r + sigma(r)) for a guessed power residue r, or by the
# interpolating polynomials f/h of utils.lut.sigma_conditions_hold.
# ---------------------------------------------------------------------------

from polocolo.params import PolocoloParams
from utils.matrix import matvecmul, vecadd, vecsub, add_to_start
from utils.mode import hash_sponge, pad_zero


class Polocolo:
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: PolocoloParams):
        # General settings
        self.F = params.F
        self.to_field = params.to_field
        self.from_field = params.from_field
        self.p = params.p
        self.t = params.t

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

        # Hash modes
        self.r = params.r
        self.c = params.c
        self.d = params.d

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

    def permutation(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)
        for r in range(self.R):
            state = self.linear_layer(state, r)
            state = self.constant_addition(state, r)
            state = self.nonlinear_layer(state, r)
        return self._post_rounds(state)

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for r in reversed(range(self.R)):
            state = self.nonlinear_layer_inv(state, r)
            state = self.constant_addition_inv(state, r)
            state = self.linear_layer_inv(state, r)
        return self._pre_rounds_inv(state)

    # ---------------------------------------------------------------------------
    # Hash modes
    #
    # Polocolo is turned into a hash function via the standard sponge (Section 2);
    # for the official ~255-bit fields a capacity of one element gives 128-bit
    # security, so r = t-1, c = 1, d = 1 (derived in params).
    # ---------------------------------------------------------------------------

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
            to_field=self.to_field,
        )

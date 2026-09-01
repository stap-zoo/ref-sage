# hash.py
# Grendel: GrendelPerm (permutation) and its mode functions (GrendelHash).

from grendel.params import GrendelParams
from utils.matrix import matvecmul, vecadd, vecsub
from utils.primitive import Permutation, HashFunction

class GrendelPerm(Permutation):
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: GrendelParams):
        # Copy the fully-specified values out of the params object. This class is
        # a pure consumer of params; nothing is derived or checked here.

        # General settings (F, to_field, from_field, t, p, kappa, toy copied by Permutation)
        super().__init__(params)

        # Rounds
        self.R = params.R

        # Non-linear layer
        self.alpha = params.alpha
        self.e = params.e
        self.e_inv = params.e_inv

        # Linear layer
        self.M = params.M
        self.M_inv = params.M_inv

        # Round constants
        self.rcons = params.rcons

    # ---------------------------------------------------------------------------
    # Component layers
    # ---------------------------------------------------------------------------

    def _legendre(self, x):
        # Legendre symbol of x as a field element, via Euler's criterion:
        # x^((p-1)/2) = -1 / 0 / 1 for a quadratic non-residue / zero / quadratic residue
        return x ** ((self.p - 1) // 2)

    def _sbox(self, x):
        # The power map with possible sign flip
        return x ** self.alpha * self._legendre(x)

    def _sbox_inv(self, y):
        # S equals the single power map x^(alpha + (p-1)/2) = x^e, so the inverse
        # is the power map with the inverse exponent e_inv = e^(-1) mod p-1
        return y ** self.e_inv

    def nonlinear_layer(self, state: list, r: int) -> list:
        # Substitution step: the S-box is applied to every state element.
        return [self._sbox(x) for x in state]

    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        return [self._sbox_inv(y) for y in state]

    def linear_layer(self, state: list, r: int) -> list:
        return matvecmul(self.M, state)

    def linear_layer_inv(self, state: list, r: int) -> list:
        return matvecmul(self.M_inv, state)

    def constant_addition(self, state: list, r: int) -> list:
        return vecadd(state, self.rcons[r])

    def constant_addition_inv(self, state: list, r: int) -> list:
        return vecsub(state, self.rcons[r])

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps
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
# Hash function
#
# GrendelPerm above is JUST the permutation. Grendel is sponge-only (no compression):
#     P = GrendelPerm(params)
#     H = GrendelHash(P, params.sponge)   # Bertoni et al. sponge
#     H.hash(data)
# ---------------------------------------------------------------------------

class GrendelHash(HashFunction):
    SPONGE_KIND = "plain"     # Bertoni et al. sponge with pad10*

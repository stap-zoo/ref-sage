# hash.py
# ---------------------------------------------------------------------------
# Grendel: the permutation (round function) and the sponge hash built on it.
#
# Construct it from a fully-specified GrendelParams object; this class only
# *applies* the parameters, it never derives or validates them (that already
# happened in params.py). The design goal is fidelity to the specification
# (https://eprint.iacr.org/2021/984), not speed.
#
# The nonlinear layer applies the "low-degree power map with possible sign flip":
#
#     S(x) = x^alpha * legendre(x)
#
# where legendre(x) = x^((p-1)/2) in {-1, 0, 1} is the Legendre symbol of x
# (Euler's criterion): -1 for a quadratic non-residue, 0 for 0, and 1 for a
# quadratic residue. The sign flip costs little to evaluate but raises the
# S-box degree to alpha + (p-1)/2, which is what defends against algebraic
# attacks at a small number of rounds.
#
# NOTE: every component uses only ring operations (+, -, *, **) and never branches 
# on the value of a state element (the Legendre symbol is computed as the power 
# x^((p-1)/2), not via a value-based case split). Note that the resulting degree is 
# of the order of p, so symbolic evaluation is only practical for very few rounds
# over tiny fields; algebraic models instead keep low-degree relations by
# introducing helper variables for the Legendre symbols (Sec. 5.4, Eq. 27/28).
# ---------------------------------------------------------------------------

from grendel.params import GrendelParams
from utils.matrix import matvecmul, vecadd, vecsub, add_to_start
from utils.mode import hash_sponge, pad_one


class Grendel:
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: GrendelParams):
        # Copy the fully-specified values out of the params object. This class is
        # a pure consumer of params; nothing is derived or checked here.

        # General settings
        self.p = params.p
        self.F = params.F
        self.to_field = params.to_field
        self.from_field = params.from_field
        self.t = params.t
        self.kappa = params.kappa

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

        # Hash modes
        self.r = params.r
        self.c = params.c
        self.d = params.d

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

    def permutation(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)
        for r in range(self.R):
            state = self.nonlinear_layer(state, r)
            state = self.linear_layer(state, r)
            state = self.constant_addition(state, r)
        return self._post_rounds(state)

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for r in reversed(range(self.R)):
            state = self.constant_addition_inv(state, r)
            state = self.linear_layer_inv(state, r)
            state = self.nonlinear_layer_inv(state, r)
        return self._pre_rounds_inv(state)

    # ---------------------------------------------------------------------------
    # Hash modes
    #
    # Grendel is turned into a hash function via the standard sponge (Sec. 4.3, 4.4);
    # ---------------------------------------------------------------------------

    def hash_sponge(self, data: list) -> list:
        padded_data, _ = pad_one(data, self.r, self.to_field)
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

    def compress(self, data: list) -> list:
        # Grendel defines no dedicated compression function (sponge only).
        raise NotImplementedError

    def compress_2_to_1(self, x: list, y: list) -> list:
        # Grendel defines no dedicated compression function (sponge only).
        raise NotImplementedError

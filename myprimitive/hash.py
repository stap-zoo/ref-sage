# hash.py
# MyPrimitive: MyPrimitivePerm (permutation) and its mode functions (MyPrimitiveHash, MyPrimitiveCompress).

from myprimitive.params import MyPrimitiveParams
from utils.matrix import matvecmul, vecadd, vecsub
from utils.primitive import Permutation, HashFunction, CompressionFunction

class MyPrimitivePerm(Permutation):
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: MyPrimitiveParams):
        # Copy the fully-specified values out of the params object. This class is
        # a pure consumer of params; nothing is derived or checked here.

        # General settings (F, to_field, from_field, t, p, kappa, toy copied by Permutation)
        super().__init__(params)

        # Rounds
        self.R = params.R

        # Non-linear layer
        self.alpha = params.alpha
        self.alpha_inv = params.alpha_inv

        # Linear layer
        self.M = params.M
        self.M_inv = params.M_inv

        # Round constants
        self.rcons = params.rcons

    # ---------------------------------------------------------------------------
    # Component layers
    #
    # Each layer is one SPN operation applied to the whole state, with a uniform
    # (state, r) -> state signature. The round index r is passed to EVERY layer
    # even when unused: it lets any layer become round-dependent without changing
    # the call sites, which keeps a common framework across all SPN-based schemes
    # (it may look obsolete for primitives whose layers never vary by round).
    #
    # Keep every layer GENERIC: use only +, -, *, ** so the same code runs on
    # field elements AND on polynomials over F (see the header note). Never branch
    # on a state element's VALUE; branching on the round index r is fine.
    # ---------------------------------------------------------------------------

    def constant_addition(self, state: list, r: int) -> list:
        # Round-dependent by nature: the constants added are the round-r constants.
        return vecadd(state, self.rcons[r])

    def constant_addition_inv(self, state: list, r: int) -> list:
        return vecsub(state, self.rcons[r])

    def linear_layer(self, state: list, r: int) -> list:
        # Round-INdependent here (same matrix M every round), so r is ignored --
        # but still accepted, for the uniform layer interface.
        return matvecmul(self.M, state)

    def linear_layer_inv(self, state: list, r: int) -> list:
        return matvecmul(self.M_inv, state)

    # You may want a helper for the non-linear layer's per-element S-box; the aim
    # is to stay expressive and close to the spec, e.g.:
    # def _sbox(self, x):
    #     return x ** self.alpha

    def nonlinear_layer(self, state: list, r: int) -> list:
        # Example of a layer whose BEHAVIOUR depends on the round: even rounds
        # apply the forward power map x**alpha, odd rounds apply x**alpha_inv.
        # (A primitive using the same S-box every round would ignore r here; the
        # branch is kept to demonstrate the round-dependent case.)
        if r % 2 == 0:
            return [x ** self.alpha for x in state]
        else:
            return [x ** self.alpha_inv for x in state]

    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        # Undo nonlinear_layer for the SAME round r by inverting the exponent the
        # forward layer used (even used **alpha -> invert with **alpha_inv, etc.).
        if r % 2 == 0:
            return [x ** self.alpha_inv for x in state]
        else:
            return [x ** self.alpha for x in state]

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps (the work done once outside the round loop). Defined by
    # each concrete permutation; _pre/_post must be mutual inverses of their _inv
    # counterparts. These are one-off (not per-round), so they take no round index.
    # ---------------------------------------------------------------------------

    def _pre_rounds(self, state: list) -> list:
        # Example: an initial linear map applied before the first round.
        return matvecmul(self.M, state)

    def _pre_rounds_inv(self, state: list) -> list:
        return matvecmul(self.M_inv, state)

    def _post_rounds(self, state: list) -> list:
        # Identity here; override if your spec does work after the last round.
        return state

    def _post_rounds_inv(self, state: list) -> list:
        return state

    # ---------------------------------------------------------------------------
    # Permutation
    #
    # The round loop fixes the ORDER in which the layers run within each round
    # (e.g. ARK -> S -> M). This is a separate concern from each layer's per-round
    # behaviour, which lives inside the layer (keyed by r). Here the order itself
    # also differs by round parity, purely to show that ordering can be
    # round-dependent too -- most primitives use one fixed order for all rounds.
    # ---------------------------------------------------------------------------

    def permute(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)

        for r in range(self.R):
            # The ORDER of the layers in round r is defined here; WHAT each layer
            # does in round r is defined inside the layer itself.
            if r % 2 == 0:
                state = self.linear_layer(state, r)
                state = self.nonlinear_layer(state, r)
                state = self.constant_addition(state, r)
            else:
                state = self.constant_addition(state, r)
                state = self.nonlinear_layer(state, r)
                state = self.linear_layer(state, r)

        return self._post_rounds(state)

    def permute_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        # Run everything backwards: undo the post step, walk the rounds from last
        # to first, and within each round apply the inverse layers in the REVERSE
        # of the forward order (the last layer applied is the first undone), then
        # undo the pre step.
        state = self._post_rounds_inv(state)

        for r in reversed(range(self.R)):
            if r % 2 == 0:
                # forward order was: linear -> nonlinear -> constant_addition
                state = self.constant_addition_inv(state, r)
                state = self.nonlinear_layer_inv(state, r)
                state = self.linear_layer_inv(state, r)
            else:
                # forward order was: constant_addition -> nonlinear -> linear
                state = self.linear_layer_inv(state, r)
                state = self.nonlinear_layer_inv(state, r)
                state = self.constant_addition_inv(state, r)

        return self._pre_rounds_inv(state)


# ---------------------------------------------------------------------------
# Hash / compression functions
#
# MyPrimitivePerm above is JUST the permutation. Each mode of operation is its own function
# object wrapping a permutation, loaded independently:
#     P = MyPrimitivePerm(params)
#     H = MyPrimitiveHash(P, params.sponge)      # plain (pad10*) sponge:  H.permute, H.hash
#     C = MyPrimitiveCompress(P, params.comp)    # truncation compression: C.permute, C.compress
# The class pins only the mode KIND; the sizes come from the params dict passed in, which must
# be non-None (else construction raises). A second sponge variant would just be another
# HashFunction subclass with a different SPONGE_KIND.
# ---------------------------------------------------------------------------

class MyPrimitiveHash(HashFunction):
    SPONGE_KIND = "plain"      # Bertoni et al. sponge with pad10*

class MyPrimitiveCompress(CompressionFunction):
    COMP_KIND = "trunc"        # truncation compression mode M = I_{d x t}
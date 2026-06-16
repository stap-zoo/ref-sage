# permutation.py
# ---------------------------------------------------------------------------
# MyPrimitive: the permutation (round function) and the hash modes built on it.
#
# Construct it from a fully-specified MyPrimitiveParams object; this class only
# *applies* the parameters, it never derives or validates them (that already
# happened in params.py). The design goal is fidelity to the specification, not
# speed: the code should read like the paper so it can be checked against it and
# reused for cryptanalysis. Efficiency is explicitly a non-goal.
#
# Cryptanalysis / symbolic evaluation: the components are written GENERICALLY --
# they use only ring operations (+, -, *, **) and never inspect or branch on the
# VALUE of a state element (branching on the round index r is fine; r is a known
# integer). So the same permutation runs not only on field elements but on
# elements of a polynomial ring over F: pass the generators of
# PolynomialRing(F, 'x', t) as the state to get the output as polynomials in the
# inputs (for degree growth, Groebner-basis modeling, ...). Never call to_field /
# from_field on intermediate values inside a component -- that would coerce a
# symbolic input back into F and break this.
#
# Layout:
#   * Component layers (constant_addition, linear_layer, nonlinear_layer, ...):
#     each is ONE SPN operation on the whole state. Every layer takes the round
#     index r, so a layer whose behaviour changes per round can branch on it;
#     round-independent layers simply ignore r. The uniform (state, r) -> state
#     signature lets the round loop treat every layer identically. Each forward
#     layer has an `_inv` partner that undoes it for the same r.
#   * Pre-/post-round steps (_pre_rounds / _post_rounds): one-off work done once
#     outside the loop (e.g. an initial linear map). Not per-round, so no r.
#   * permutation / permutation_inv: the round loop.
#   * Hash modes: sponge / compression wrappers around the permutation.
#
# Two ORTHOGONAL notions of "round-dependence" deliberately live in separate
# places -- keep them separate when you adapt this:
#   - WHAT a layer does in round r             -> inside the layer, keyed by r
#   - In WHICH ORDER the layers run in round r -> in the permutation loop
# ---------------------------------------------------------------------------

from myprimitive.params import MyPrimitiveParams
from utils import matvecmul, vecadd, vecsub, add_to_start
from modes import hash_sponge, pad_zero


class MyPrimitive:
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: MyPrimitiveParams):
        # Copy the fully-specified values out of the params object. This class is
        # a pure consumer of params; nothing is derived or checked here.

        # General settings
        self.F = params.F
        self.to_field = params.to_field
        self.from_field = params.from_field
        self.t = params.t
        self.kappa = params.kappa

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

        # Hash modes
        self.r = params.r
        self.c = params.c
        self.d = params.d

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

    def permutation(self, state: list) -> list:
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

    def permutation_inv(self, state: list) -> list:
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
    # Hash modes
    #
    # Thin wrappers that turn the permutation into a hash / compression function.
    # The mode logic and padding rules live in modes.py and are shared across
    # primitives; these methods just call into it with this primitive's
    # parameters. Not every primitive defines every mode.
    # ---------------------------------------------------------------------------

    def hash_sponge(self, data: list) -> list:
        # Sponge hash using rate self.r and capacity self.c, squeezing self.d
        # elements (self.r + self.c == self.t).
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

    def compress(self, data: list) -> list:
        # General compression function (e.g. Jive mode, or truncation with a
        # feed-forward). Not all primitives define this.
        raise NotImplementedError

    def compress_2_to_1(self, x: list, y: list) -> list:
        # 2-to-1 compression (or 3-to-1, etc.), defined either via the `compress`
        # function above or via the sponge.
        raise NotImplementedError
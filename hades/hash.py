# hash.py
# ---------------------------------------------------------------------------
# Hades family: the abstract HadesLikePermutation and its concrete Poseidon /
# Poseidon2 / Neptune permutations, plus the hash modes built on them.
#
# Each class is constructed from a fully-specified params object and only
# *applies* the parameters (no derivation/validation).
# ---------------------------------------------------------------------------

from utils import matvecmul, vecadd, vecsub, add_to_start
from modes import hash_sponge, pad_zero, compress_davies_meyer
import warnings
from recommendations import ModeRecommendationWarning


class HadesLikePermutation:
    """Abstract Hades-strategy permutation: external (full) rounds, then internal (partial)
    rounds, then external rounds, around a state of t branches. Each round is ARK -> S-box ->
    matrix (round constant rcons[r] added before the S-box).

    The work done once outside the round loop is left to the subclass via _pre_rounds /
    _post_rounds (and their inverses): a leading external matrix (Poseidon2, Neptune), output
    whitening (Neptune, which is S -> M -> ARK and so applies its final constant here), or
    nothing (Poseidon). The engine itself stays agnostic.

    Subclasses (Poseidon, Poseidon2, Neptune) otherwise differ only in the matrices/constants
    carried by their params and, for Neptune, in the external S-box. The S-box hits all t
    branches in external rounds and the first u branches in internal rounds.
    """

    def __init__(self, params):
        self.F = params.F
        self.to_field = params.to_field
        self.from_field = params.from_field
        self.t = params.t
        self.alpha = params.alpha
        self.u = params.u 

        # Rounds
        self.R = params.R
        self.R_ext_beg = params.R_ext_beg
        self.R_int = params.R_int
        self.R_ext_end = params.R_ext_end

        # Linear layers
        self.alpha_inv = params.alpha_inv
        self.M_ext = params.M_ext
        self.M_int = params.M_int
        self.M_ext_inv = params.M_ext_inv
        self.M_int_inv = params.M_int_inv

        # Round constants (the per-round engine grid the round loop indexes; ARK before each
        # S-box). Variants that whiten the output read the trailing row in _post_rounds.
        self.rcons = params.rcons

        # Hash modes
        self.r = params.r
        self.c = params.c
        self.d = params.d

    # ---------------------------------------------------------------------------
    # Component functions
    # ---------------------------------------------------------------------------

    def _is_internal_round(self, r):
        return self.R_ext_beg <= r < self.R_ext_beg + self.R_int

    def constant_addition(self, state: list, r: int) -> list:
        return vecadd(state, self.rcons[r])

    def constant_addition_inv(self, state: list, r: int) -> list:
        return vecsub(state, self.rcons[r])

    def linear_layer(self, state: list, r: int) -> list:
        if self._is_internal_round(r):
            return matvecmul(self.M_int, state)
        else:
            return matvecmul(self.M_ext, state)

    def linear_layer_inv(self, state: list, r: int) -> list:
        if self._is_internal_round(r):
            return matvecmul(self.M_int_inv, state)
        else:
            return matvecmul(self.M_ext_inv, state)

    def nonlinear_layer(self, state: list, r: int) -> list:
        if self._is_internal_round(r):
            return [x ** self.alpha for x in state[:self.u]] + state[self.u:]
        else:
            return [x ** self.alpha for x in state]

    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        if self._is_internal_round(r):
            return [x ** self.alpha_inv for x in state[:self.u]] + state[self.u:]
        else:
            return [x ** self.alpha_inv for x in state]

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps (the work done once outside the round loop). Defined by each
    # concrete permutation; _pre/_post must be mutual inverses of _pre_inv/_post_inv.
    # ---------------------------------------------------------------------------

    def _pre_rounds(self, state: list) -> list:
        return state

    def _post_rounds(self, state: list) -> list:
        return state

    def _pre_rounds_inv(self, state: list) -> list:
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
            state = self.constant_addition(state, r)
            state = self.nonlinear_layer(state, r)
            state = self.linear_layer(state, r)

        return self._post_rounds(state)

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)

        for r in reversed(range(self.R)):
            state = self.linear_layer_inv(state, r)
            state = self.nonlinear_layer_inv(state, r)
            state = self.constant_addition_inv(state, r)

        return self._pre_rounds_inv(state)

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------

    def compress_2_to_1(self, x1: list, x2: list) -> list:
        """Davies-Meyer 2-to-1 compression, defined when t == 2 * digest size."""
        if self.t != 2 * self.d:
            raise ValueError(f"Compression mode not defined for state size {self.t} and digest size {self.d}.")
        if len(x1) != self.d or len(x2) != self.d:
            raise ValueError(f"Invalid input sizes. Expected ({self.d},{self.d}), got ({len(x1)},{len(x2)})")
        return compress_davies_meyer(
            perm=self.permutation,
            x_m=x1,
            x_c=x2,
            digest_size=self.d,
            to_field=self.to_field,
        )

    def hash_sponge(self, data: list) -> list:
        padded_data, _ = pad_zero(data, self.r, self.to_field)
        IV = [self.to_field(len(data))] + [self.F.zero()] * (self.c - 1)
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


# ---------------------------------------------------------------------------
# Concrete permutations
# ---------------------------------------------------------------------------

class Poseidon(HadesLikePermutation):
    """Poseidon: a single MDS matrix for external and internal rounds, power-map S-box,
    full-width round constants before each S-box. No leading matrix."""

    def compress_2_to_1(self, x1: list, x2: list) -> list:
        warnings.warn("Poseidon does not define a compression mode; using the generic "
            "truncated-feed-forward construction. This is an unanalyzed extension, "
            "not part of the Poseidon specification.",
            UserWarning,
            stacklevel=2,
        )
        return super().compress_2_to_1(x1, x2)

class Poseidon2(HadesLikePermutation):
    """Poseidon2: a leading external matrix, distinct external/internal matrices, power-map
    S-box, internal-round constants on branch 0 only. Leading external matrix."""

    def _pre_rounds(self, state: list) -> list:
        return matvecmul(self.M_ext, state)

    def _pre_rounds_inv(self, state: list) -> list:
        return matvecmul(self.M_ext_inv, state)

class Neptune(HadesLikePermutation):
    """Neptune: external rounds apply a quadratic pair-wise S-box (open-Flystel map with
    alpha = beta = 1 and constant gamma) instead of the power map; internal rounds and the
    linear layers follow the Hades template. A leading external matrix is applied up front;
    its S->M->ARK round order needs no leading constant."""

    def __init__(self, params):
        super().__init__(params)
        self.lm_alpha = params.lm_alpha
        self.lm_alpha_inv = params.lm_alpha_inv
        self.lm_beta = params.lm_beta
        self.lm_gamma = params.lm_gamma
        self.lm_M = params.lm_M
        self.lm_M_inv = params.lm_M_inv

    def _pre_rounds(self, state: list) -> list:
        return matvecmul(self.M_ext, state)

    def _pre_rounds_inv(self, state: list) -> list:
        return matvecmul(self.M_ext_inv, state)

    def _post_rounds(self, state: list) -> list:
        # last operation is an AddRoundConstants (output whitening, no trailing bare matrix).
        return vecadd(state, self.rcons[-1])

    def _post_rounds_inv(self, state: list) -> list:
        return vecsub(state, self.rcons[-1])

    @staticmethod
    def _S_F(x, y, alpha, beta):
        """S_F lifting of Lai-Massey construction F(x,y) = alpha*x + beta*(x - y)^2:
        (x, y) -> (alpha*x + beta*s, alpha*y + beta*s), s = (x - y)^2."""
        s = beta * (x - y) ** 2
        return alpha * x + s, alpha * y + s

    @staticmethod
    def _S_F_inv(x, y, alpha, beta):
        """Inverse. x - y = alpha*(x - y) is preserved (independent of beta),
        so the same s = beta*((x - y)/alpha)^2 is recoverable, then divide alpha out."""
        alpha_inv = alpha ** (-1)
        s = beta * ((x - y) * alpha_inv) ** 2
        return (x - s) * alpha_inv, (y - s) * alpha_inv

    def _sbox(self, x, y):
        """Neptune external Lai-Massey-like S-box,  see Eq. (22) of https://eprint.iacr.org/2021/1695.pdf."""
        x, y = self._S_F(x, y, self.lm_alpha, self.lm_beta)
        x, y = matvecmul(self.lm_M, [x,y])
        x, y = vecadd([x, y], [self.lm_gamma, self.F.zero()])
        x, y = self._S_F(x, y, self.lm_alpha, self.lm_beta)
        return vecadd([x, y], [-self.lm_alpha * self.lm_gamma, self.F.zero()])
    
    def _sbox_inv(self, x, y):
        x, y = vecsub([x, y], [-self.lm_alpha * self.lm_gamma, self.F.zero()])
        x, y = self._S_F_inv(x, y, self.lm_alpha, self.lm_beta)
        x, y = matvecmul(self.lm_M_inv, vecsub([x, y], [self.lm_gamma, self.F.zero()]))
        return self._S_F_inv(x, y, self.lm_alpha, self.lm_beta)

    def nonlinear_layer(self, state: list, r: int) -> list:
        if self._is_internal_round(r): # same as Poseidon2
            return [x ** self.alpha for x in state[:self.u]] + state[self.u:] 
        else: # Apply quadratic pair-wise S-box
            out = []
            for i in range(0, self.t, 2):
                y1, y2 = self._sbox(state[i], state[i + 1])
                out += [y1, y2]
            return out

    def nonlinear_layer_inv(self, state: list, r: int) -> list:
        if self._is_internal_round(r):
            return [x ** self.alpha_inv for x in state[:self.u]] + state[self.u:]
        else:
            out = []
            for i in range(0, self.t, 2):
                x1, x2 = self._sbox_inv(state[i], state[i + 1])
                out += [x1, x2]
            return out

    def compress_2_to_1(self, x1: list, x2: list) -> list:
        msg = "Neptune does not define a compression mode; using truncated-feed-forward construction."
        warnings.warn(msg, ModeRecommendationWarning, stacklevel=2)
        return super().compress_2_to_1(x1, x2)
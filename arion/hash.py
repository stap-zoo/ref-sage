# hash.py
# ---------------------------------------------------------------------------
# Arion: the permutation (round function) and the hash modes built on it.
#
# Constructed from a fully-specified ArionParams object. 
# ---------------------------------------------------------------------------

from arion.params import ArionParams
from utils.matrix import matvecmul, vecadd, vecsub

class Arion:
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: ArionParams):
        self.F = params.F
        self.to_field = params.to_field
        self.from_field = params.from_field
        self.t = params.t

        # Rounds
        self.R = params.R

        # Non-linear layer (GTDS)
        self.alpha1 = params.alpha1
        self.alpha2 = params.alpha2
        self.alpha1_inv = params.alpha1_inv
        self.alpha2_inv = params.alpha2_inv
        self.coeffs_g = params.coeffs_g
        self.coeffs_h = params.coeffs_h

        # Affine layer
        self.M = params.M
        self.M_inv = params.M_inv
        self.rcons = params.rcons

        # Hash modes
        self.sponge = params.sponge

    # ---------------------------------------------------------------------------
    # Component functions
    # ---------------------------------------------------------------------------
    def constant_addition(self, state: list, r: int) -> list:
        # Round-dependent by nature: the constants added are the round-r constants.
        return vecadd(state, self.rcons[r])

    def constant_addition_inv(self, state: list, r: int) -> list:
        return vecsub(state, self.rcons[r])
    
    def linear_layer(self, state: list, r: int) -> list:
        return matvecmul(self.M, state)

    def linear_layer_inv(self, state: list, r: int) -> list:
        return matvecmul(self.M_inv, state)

    def _g(self, sigma, r: int, i: int):
        """g_i(x) = x^2 + a*x + b; root-free in F_p since a^2 - 4*b is a quadratic nonresidue."""
        a, b = self.coeffs_g[r][i]
        return sigma ** 2 + a * sigma + b

    def _h(self, sigma, r: int, i: int):
        """h_i(x) = x^2 + c*x (may have roots, is never inverted)."""
        c = self.coeffs_h[r][i]
        return sigma ** 2 + c * sigma

    def nonlinear_layer(self, x: list, r: int) -> list:
        """Generalized Triangular Dynamical System, see https://arxiv.org/pdf/2303.04639, Def 1"""
        y = [None] * self.t
        y[-1] = x[-1] ** self.alpha2_inv
        sigma = x[-1] + y[-1]                          # sigma_{n,n}
        for i in range(self.t - 2, -1, -1):
            # calculate y_i = x_i^alpha1 * g_i + h_i
            y[i] = x[i] ** self.alpha1 * self._g(sigma, r, i) + self._h(sigma, r, i)
            # extend sigma_{i+1,n} to sigma_{i,n}
            sigma += x[i] + y[i]                      
        return y

    def nonlinear_layer_inv(self, y: list, r: int) -> list:
        """Inverse Generalized Triangular Dynamical System."""
        x = [None] * self.t
        x[-1] = y[-1] ** self.alpha2
        sigma = y[-1] + x[-1]                          # sigma_{n,n}
        for i in range(self.t - 2, -1, -1):
            # revert y_i = x_i^alpha1 * g_i + h_i from the outside in
            t1 = y[i] - self._h(sigma, r, i)           # = x_i^alpha1 * g_i
            t2 = t1 * self._g(sigma, r, i)**(-1)       # = x_i^alpha1
            x[i] = t2 ** self.alpha1_inv               # = x_i
            # extend sigma_{i+1,n} to sigma_{i,n}
            sigma += y[i] + x[i]                       
        return x

    def AffineLayer(self, state: list, r: int) -> list:
        return vecadd(matvecmul(self.M, state), self.rcons[r])

    def AffineLayer_inv(self, state: list, r: int) -> list:
        return matvecmul(self.M_inv, vecsub(state, self.rcons[r]))

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps (Arion does no work outside the loop: identities)
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
    # ---------------------------------------------------------------------------

    def hash_sponge(self, data: list) -> list:
        return self.sponge.hash(self.permutation, data)

from arion.params import ArionParams
from utils import matvecmul, vecadd, vecsub, add_to_start
from modes import hash_sponge, pad_zero


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
        self.r = params.r
        self.c = params.c
        self.d = params.d

    # ---------------------------------------------------------------------------
    # Component functions
    # ---------------------------------------------------------------------------

    def _g(self, sigma, r: int, i: int):
        """g_i(x) = x^2 + a*x + b; root-free in F_p since a^2 - 4*b is a quadratic nonresidue."""
        a, b = self.coeffs_g[r][i]
        return sigma ** 2 + a * sigma + b

    def _h(self, sigma, r: int, i: int):
        """h_i(x) = x^2 + c*x (may have roots, is never inverted)."""
        c = self.coeffs_h[r][i]
        return sigma ** 2 + c * sigma

    def GTDS(self, x: list, r: int) -> list:
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

    def GTDS_inv(self, y: list, r: int) -> list:
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

    def AffineLayer(self, state: list, round_idx: int) -> list:
        return vecadd(matvecmul(self.M, state), self.rcons[round_idx])

    def AffineLayer_inv(self, state: list, round_idx: int) -> list:
        return matvecmul(self.M_inv, vecsub(state, self.rcons[round_idx]))

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        for r in range(self.R):
            state = self.GTDS(state, r)
            state = self.AffineLayer(state, r)
        return state

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        for r in reversed(range(self.R)):
            state = self.AffineLayer_inv(state, r)
            state = self.GTDS_inv(state, r)
        return state

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------

    def hash_sponge(self, data: list) -> list:
        padded_data, was_aligned = pad_zero(data, self.r, self.to_field)
        IV = [self.F.zero()] * self.c if was_aligned else [self.to_field(len(data))] + [self.F.zero()] * (self.c-1)
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

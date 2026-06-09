from sage.all import GF, Integer, Matrix

from monolith.params import MonolithParams
from utils import matvecmul, mixed_radix_decompose, mixed_radix_compose, invert_LUT
from modes import compress_davies_meyer, hash_sponge_safe, pad_zero

class Monolith:
    def __init__(self, params: MonolithParams):
        self.F = GF(params.p)
        self.t = params.t

        # Rounds
        self.R = params.R

        # Non-linear layers: Bars
        self.si = params.si
        self.LUTs = params.LUTs
        self.LUTs_inv = {s: invert_LUT(self.LUTs[s]) for s in self.LUTs}
        self.u = params.u

        # Affine layer
        # According to reference implementation, no round constant addition in 
        self.M = params.M
        self.M_inv = [list(row) for row in Matrix(self.F, self.M).inverse()]
        self.rcons = [[self.F.zero()] * self.t] + \
            [[self.to_field(rc) for rc in row] for row in params.rcons] +\
            [[self.F.zero()] * self.t]    

        # Hash modes
        self.r = params.r
        self.c = params.c
        self.d = params.d

    # ---------------------------------------------------------------------------
    # Small helpers
    # ---------------------------------------------------------------------------

    def from_field(self, el) -> Integer:
        return Integer(el)

    def to_field(self, n: int):
        return self.F(n)

    # ---------------------------------------------------------------------------
    # Component functions
    # ---------------------------------------------------------------------------

    def AffineLayer(self, state: list, round_idx: int) -> list:
        """MDS matrix-vector product followed by round-constant addition. 
        Matrix multiplication originally called Concrete in Monolith paper."""
        result = matvecmul(self.M, state)
        for i, rc in enumerate(self.rcons[round_idx]):
            result[i] = result[i] + rc
        return result
    
    def AffineLayer_inv(self, state: list, round_idx: int) -> list:
        sub = [state[i] - self.rcons[round_idx][i] for i in range(len(state))]
        return matvecmul(self.M_inv, sub)

    def Bars(self, state: list) -> list:
        result = list(state)
        for i in range(self.u):
            digits = mixed_radix_decompose(state[i], self.si, self.from_field)
            new_digits = [self.LUTs[s][d] for s, d in zip(self.si, digits)]
            result[i] = mixed_radix_compose(new_digits, self.si, self.to_field)
        return result
    
    def Bars_inv(self, state: list) -> list:
        result = list(state)
        for i in range(self.u):
            digits = mixed_radix_decompose(state[i], self.si, self.from_field)
            new_digits = [self.LUTs_inv[s][d] for s, d in zip(self.si, digits)]
            result[i] = mixed_radix_compose(new_digits, self.si, self.to_field)
        return result

    def Bricks(self, state: list) -> list:
        result = [state[0]]
        for i in range(1, self.t):
            result.append(state[i] + state[i - 1] ** 2)
        return result
    
    def Bricks_inv(self, state: list) -> list:
        result = [state[0]]
        for i in range(1, self.t):
            result.append(state[i] - result[i - 1] ** 2)
        return result

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self.AffineLayer(state, 0)

        for i in range(1, self.R + 1):
            state = self.Bars(state)
            state = self.Bricks(state)
            state = self.AffineLayer(state, i)

        return state
    
    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        for i in range(self.R, 0, -1):
            state = self.AffineLayer_inv(state, i)
            state = self.Bricks_inv(state)
            state = self.Bars_inv(state)
            
        state = self.AffineLayer_inv(state, 0)
        return state

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------

    def compress_2_to_1(self, x1: list, x2:list) -> list:
        """2-to-1 compression defined for small state sizes double the digest size"""
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
        return hash_sponge_safe(
            perm=self.permutation,
            data=data,
            state_size=self.t,
            rate=self.r,
            capacity=self.c,
            digest_size=self.d,
            pad=pad_zero,
            to_field=self.to_field,
        )

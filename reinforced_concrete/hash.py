from sage.all import GF, Integer, Matrix

from reinforced_concrete.params import ReinforcedConcreteParams
from utils import matvecmul, mixed_radix_decompose, mixed_radix_compose, invert_LUT
from modes import compress_davies_meyer, hash_sponge, pad_zero

class ReinforcedConcrete:
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: ReinforcedConcreteParams):
        self.F = GF(params.p)
        self.t = params.t

        # Rounds
        self.R_pre = params.R_pre
        self.R_bars = params.R_bars
        self.R = params.R

        # Non-linear layers: Bricks
        self.alpha = params.alpha
        self.alpha_inv = params.alpha_inv if params.alpha_inv is not None else pow(alpha, -1, p - 1)
        self.a_coeffs = [self.to_field(a) for a in params.a_coeffs]
        self.b_coeffs = [self.to_field(b) for b in params.b_coeffs]

        # Non-linear layers: Bars
        self.si = params.si
        self.LUT = params.LUT
        self.LUT_inv = invert_LUT(self.LUT)

        # Affine layer
        self.M = params.M
        self.M_inv = [list(row) for row in Matrix(self.F, self.M).inverse()]
        self.rcons = [[self.to_field(rc) for rc in row] for row in params.rcons]
        
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
        Matrix multiplication and round constant addition originally called Concrete in RC paper."""
        result = matvecmul(self.M, state)
        for i, rc in enumerate(self.rcons[round_idx]):
            result[i] = result[i] + rc
        return result

    def AffineLayer_inv(self, state: list, round_idx: int) -> list:
        sub = [state[i] - self.rcons[round_idx][i] for i in range(len(state))]
        return matvecmul(self.M_inv, sub)

    def Fi(self, val, i):
        return val ** 2 + self.a_coeffs[i] * val + self.b_coeffs[i]

    def Bricks(self, state: list) -> list:
        result = [state[0] ** self.alpha]
        for i in range(1, self.t):
            result.append(state[i] * self.Fi(state[i - 1], i - 1))
        return result

    def Bricks_inv(self, state: list) -> list:
        result = [state[0] ** self.alpha_inv]
        for i in range(1, self.t):
            result.append(state[i] * self.Fi(result[i - 1], i - 1) ** (-1))
        return result

    def Bars(self, state: list) -> list:
        result = []
        for el in state:
            digits = mixed_radix_decompose(el, self.si, self.from_field)
            digits = [self.LUT[d] for d in digits]
            result.append(mixed_radix_compose(digits, self.si, self.to_field))
        return result

    def Bars_inv(self, state: list) -> list:
        result = []
        for el in state:
            digits = mixed_radix_decompose(el, self.si, self.from_field)
            digits = [self.LUT_inv[d] for d in digits]
            result.append(mixed_radix_compose(digits, self.si, self.to_field))
        return result

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self.AffineLayer(state, 0)

        for i in range(1, self.R_pre + 1):
            state = self.Bricks(state)
            state = self.AffineLayer(state, i)

        for i in range(self.R_pre + 1, self.R_pre + self.R_bars + 1):
            state = self.Bars(state)
            state = self.AffineLayer(state, i)

        for i in range(self.R_pre + self.R_bars + 1, self.R + 1):
            state = self.Bricks(state)
            state = self.AffineLayer(state, i)

        return state

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        for i in range(self.R, self.R_pre + self.R_bars, -1):
            state = self.AffineLayer_inv(state, i)
            state = self.Bricks_inv(state)

        for i in range(self.R_pre + self.R_bars, self.R_pre, -1):
            state = self.AffineLayer_inv(state, i)
            state = self.Bars_inv(state)

        for i in range(self.R_pre, 0, -1):
            state = self.AffineLayer_inv(state, i)
            state = self.Bricks_inv(state)

        state = self.AffineLayer_inv(state, 0)
        return state

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------

    def compress_2_to_1(self, x1: list, x2:list) -> list:
        """2-to-1 compression defined for large state sizes triple the digest size"""
        if self.t != 3 * self.d:
            raise ValueError(f"Compression mode not defined for state size {self.t} and digest size {self.d}.")
        if len(x1) != self.d or len(x2) != self.d:
            raise ValueError(f"Invalid input sizes. Expected ({self.d},{self.d}), got ({len(x1)},{len(x2)})")
        return compress_davies_meyer(
            perm=self.permutation, 
            x_m=x1 + x2, 
            x_c=[self.F.zero()] * self.d, 
            digest_size=self.d, 
            to_field=self.to_field
        )

    def hash_sponge(self, data: list) -> list:
        return hash_sponge(
            perm=self.permutation, 
            data=data, 
            state_size=self.t, 
            rate=self.r, 
            capacity=self.c, 
            digest_size=self.d, 
            pad=pad_zero, 
            to_field=self.to_field
        )

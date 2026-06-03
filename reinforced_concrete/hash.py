from sage.all import GF, Integer, Matrix

from reinforced_concrete.params import ReinforcedConcreteParams
from utils import matvecmul
from modes import compress_davies_meyer, hash_sponge, pad_zero


class ReinforcedConcrete:
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------
    
    def __init__(self, params: ReinforcedConcreteParams):
        self.F               = GF(params.p)
        self.alpha           = params.alpha
        self.pre_rounds      = params.pre_rounds
        self.bars_rounds     = params.bars_rounds
        self.total_rounds    = params.total_rounds
        self.matrix          = params.mds_matrix
        self.state_size      = params.state_size
        self.digest_size     = params.digest_size
        self.si              = params.si
        self.lut             = params.lut
        self.lut_inv         = params.lut_inv
        self.alpha_inv       = params.alpha_inv if params.alpha_inv is not None else pow(self.alpha, -1, int(self.F.characteristic()) - 1)
        self.matrix_inv      = [list(row) for row in Matrix(self.F, self.matrix).inverse()]
        self.round_constants = [[self.to_field(rc) for rc in row] for row in params.round_constants]
        self.a_coeffs        = [self.to_field(a) for a in params.a_coeffs]
        self.b_coeffs        = [self.to_field(b) for b in params.b_coeffs]

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

    def concrete(self, state: list, round_idx: int) -> list:
        """MDS matrix-vector product followed by round-constant addition, in-place."""
        result = matvecmul(self.matrix, state)
        for i, rc in enumerate(self.round_constants[round_idx]):
            result[i] = result[i] + rc
        return result
    
    def concrete_inv(self, state: list, round_idx: int) -> list:
        sub = [state[i] - self.round_constants[round_idx][i] for i in range(len(state))]
        return matvecmul(self.matrix_inv, sub)

    def Fi(self, val, i):
        return val ** 2 + self.a_coeffs[i] * val + self.b_coeffs[i]

    def bricks(self, state: list) -> list:
        new0 = state[0] ** self.alpha
        new1 = state[1] * self.Fi(state[0], 0)
        new2 = state[2] * self.Fi(state[1], 1)
        return [new0, new1, new2]
    
    def bricks_inv(self, state: list) -> list:
        x0 = state[0] ** self.alpha_inv
        x1 = state[1] * self.Fi(x0, 0) ** (-1)
        x2 = state[2] * self.Fi(x1, 1) ** (-1)
        return [x0, x1, x2]

    def decompose(self, val) -> list[int]:
        """Decompose a field element into mixed-radix digits w.r.t. si."""
        n = self.from_field(val)
        res = [0] * len(self.si)
        for i in range(len(self.si) - 1, 0, -1):
            n, res[i] = divmod(n, self.si[i])
        res[0] = n
        return res

    def compose(self, vals: list[int]):
        """Recompose a field element from mixed-radix digits w.r.t. si."""
        result = vals[0]
        for val, s in zip(vals[1:], self.si[1:]):
            result = result * s + val
        return self.to_field(result)

    def bars(self, state: list) -> list:
        result = []
        for el in state:
            digits = self.decompose(el)
            digits = [self.lut[d] for d in digits]
            result.append(self.compose(digits))
        return result

    def bars_inv(self, state: list) -> list:
        result = []
        for el in state:
            digits = self.decompose(el)
            digits = [self.lut_inv[d] for d in digits]
            result.append(self.compose(digits))
        return result

    # ---------------------------------------------------------------------------
    # Permutation and hash function
    # ---------------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        if len(state) != self.state_size:
            raise ValueError(f"Invalid state size. Expected {self.state_size}, got {len(state)}")

        # Initial concrete
        state = self.concrete(state, 0)

        # pre-rounds (bricks)
        for i in range(1, self.pre_rounds + 1):
            state = self.bricks(state)
            state = self.concrete(state, i)

        # middle-rounds (bars)
        for i in range(self.pre_rounds + 1, self.pre_rounds + self.bars_rounds + 1):
            state = self.bars(state)
            state = self.concrete(state, i)

        # post-rounds (bricks)
        for i in range(self.pre_rounds + self.bars_rounds + 1, self.total_rounds + 1):
            state = self.bricks(state)
            state = self.concrete(state, i)

        return state

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.state_size:
            raise ValueError(f"Invalid state size. Expected {self.state_size}, got {len(state)}")

        # Undo post-rounds (bricks)
        for i in range(self.total_rounds, self.pre_rounds + self.bars_rounds, -1):
            state = self.concrete_inv(state, i)
            state = self.bricks_inv(state)

        # Undo middle-rounds (bars)
        for i in range(self.pre_rounds + self.bars_rounds, self.pre_rounds, -1):
            state = self.concrete_inv(state, i)
            state = self.bars_inv(state)

        # Undo pre-rounds (bricks)
        for i in range(self.pre_rounds, 0, -1):
            state = self.concrete_inv(state, i)
            state = self.bricks_inv(state)

        # Undo initial concrete
        state = self.concrete_inv(state, 0)
        return state

    def compress(self, x_m: list, x_c: list) -> list:
        return compress_davies_meyer(perm=self.permutation, x_m=x_m, x_c=x_c, digest_size=self.digest_size, to_field=self.to_field)

    def hash_sponge(self, data: list) -> list:
        capacity = self.state_size - self.digest_size
        rate = self.state_size - capacity
        return hash_sponge(perm=self.permutation, data=data, state_size=self.state_size, rate=rate, capacity=capacity, digest_size=self.digest_size, pad=pad_zero, to_field=self.to_field)

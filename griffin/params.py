from sage.all import GF, Integer, matrix, legendre_symbol
from math import gcd

from utils import m4_to_block_circulant_matrix, circulant, ShakeReader


class GriffinParams:
    def __init__(
        self,
        p:         int,
        t:         int,
        R:         int,
        alpha:     int,
        alpha_inv: int = None,
        rcons:     list[list[int]] = None,
        coeffs_G:  list[list[int]] = None,
        M:         list[list[int]] = None,
        r:         int = None,
        c:         int = None,
        d:         int = None,
        kappa:     int = 128,
    ):
        """
        Parameters
        ----------
        p         : field characteristic (prime)
        t         : permutation state size; must be 3 or a multiple of 4
        R         : number of rounds
        alpha     : non-linear layer exponent (3, 5, or 7)
        alpha_inv : alpha^{-1} mod (p-1); computed if not provided
        rcons     : (R-1)xt round constants (the final round has no round constants); generated via SHAKE128 if not provided
        coeffs_G  : (t-2) [a, b] pairs for the quadratic maps G_i; generated via SHAKE128 if not provided
        M         : mixing matrix (txt); generated via griffin_matrix(t) if not provided
        r         : rate (number of outer state elements absorbed/squeezed per sponge step)
        c         : capacity (number of inner state elements)
        d         : digest size (number of output elements)
        kappa     : target security level in bits (default 128)
        """
        assert t == 3 or t % 4 == 0, "t must be 3 or a multiple of 4"
        assert alpha in (3, 5, 7), "alpha must be 3, 5, or 7"

        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Rounds
        self.R = R

        # Derive constants if not provided
        if rcons is None or coeffs_G is None:
            _rcons, _coeffs = self._init_constants()
            rcons = rcons if rcons is not None else _rcons
            coeffs_G = coeffs_G if coeffs_G is not None else _coeffs

        # Non-linear layer
        self.alpha = alpha if alpha is not None else self._init_alpha()
        self.alpha_inv = alpha_inv if alpha_inv is not None else pow(alpha, -1, p - 1)
        self.coeffs_G = [[self.to_field(a), self.to_field(b)] for a, b in coeffs_G]

        # Affine layer
        M = M if M is not None else self._init_matrix()
        self.M = [[self.to_field(x) for x in row] for row in M]
        self.M_inv = [list(row) for row in matrix(self.F, self.M).inverse()]

        # Pad with a zero row so AffineLayer can uniformly index rcons[round_idx]
        # for round_idx in 0..R-1 (the final round has no round constants).
        self.rcons = [[self.to_field(x) for x in row] for row in rcons] + [[self.F.zero()] * self.t]

        # Hash modes
        self.r = r
        self.c = c
        self.d = d

    # ---------------------------------------------------------------------------
    # Small helpers
    # ---------------------------------------------------------------------------

    def from_field(self, el) -> Integer:
        return Integer(el)

    def to_field(self, n: int):
        return self.F(n)
    
    def _init_alpha(self) -> int:
        for alpha in range(3, self.p):
            if gcd(alpha, self.p - 1) == 1:
                return alpha

    def _init_matrix(self):
        if self.t == 3:
            return circulant([2, 1, 1])
        else:
            return m4_to_block_circulant_matrix(self.t)

    def _init_constants(self):
        # Initialize ShakeReader, seeded with "Griffin" followed by the field characteristic serialized as little-endian 64-bit limbs.
        n_bytes = ((self.p.bit_length() + 63) // 64) * 8
        seed = b"Griffin" + self.p.to_bytes(n_bytes, "little")
        reader = ShakeReader(seed, self.p)

        # Generate round constants. The last round has no round constants.
        rcons = [[reader.field_element() for _ in range(self.t)] for _ in range(self.R - 1)]
        
        # generate coefficients for the quadratic maps L_i
        coeffs = []

        # random a/b: distinct, non-zero, and legendre_symbol(a^2 - 4*b, p) == -1
        while True:
            a = reader.nonzero_field_element()
            b = reader.nonzero_field_element()
            while a == b:
                b = reader.nonzero_field_element()
            if legendre_symbol(a**2 - 4 * b, self.p) == -1:
                coeffs.append([a, b])
                break

        # remaining a_i = i*a, b_i = i^2*b, resampling b_i if a_i == b_i
        a_0, b_0 = coeffs[0]
        for i in range(2, self.t - 1):
            a_i = (a_0 * i) % self.p
            b_i = (b_0 * i * i) % self.p
            while a_i == b_i:
                b_i = reader.nonzero_field_element()
            coeffs.append([a_i, b_i])

        return rcons, coeffs

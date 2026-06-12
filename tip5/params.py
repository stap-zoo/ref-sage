from math import gcd

from sage.all import GF, Integer, matrix

from utils import invert_LUT, FieldElementSampler, tip5_mds_matrix


class Tip5Params:
    def __init__(
        self,
        p:         int,
        t:         int,
        R:         int,
        u:         int,
        d:         int,
        M:         list[list[int]] = None,
        alpha:     int = None,
        alpha_inv: int = None,
        LUT:       list[int] = None,
        rcons:     list[list[int]] = None,
        r:         int = None,
        c:         int = None,
        kappa:     int = 128,
    ):
        """
        Parameters
        ----------
        p         : field characteristic (prime, ~64 bits)
        t         : permutation state size
        R         : number of rounds
        alpha     : power map exponent, coprime with p-1
        alpha_inv : alpha^{-1} mod (p-1); computed if not provided
        u         : number of decomposition S-boxes in the nonlinear layer (applied to the first u state
                    elements; the remaining t-u elements go through the power map)
        M         : circulant MDS matrix (txt)
        d         : digest size (number of output elements)
        LUT       : 256-entry lookup table for the split-and-lookup S-box; generated via _init_lut if not provided
        rcons     : Rxt round constants; generated via Blake3 if not provided
        r         : rate (number of outer state elements; the hash is fixed-length, one block)
        c         : capacity (number of inner state elements)
        kappa     : target security level in bits (default 128)
        """
        assert p.bit_length() == 64, "Tip5 is defined over ~64-bit fields"

        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Rounds
        self.R = R

        # Non-linear layer: split-and-lookup S-boxes (S) and power maps (T)
        self.u = u
        self.si = [256] * 8  # byte decomposition of a 64-bit value
        self.LUT = LUT if LUT is not None else self._init_lut()
        self.LUT_inv = invert_LUT(self.LUT)
        self.mont_R = self.to_field(2**64)
        self.mont_R_inv = self.mont_R ** (-1)

        self.alpha = alpha if alpha is not None else self._init_alpha()
        self.alpha_inv = alpha_inv if alpha_inv is not None else pow(self.alpha, -1, p - 1)

        # Affine layer
        M = M if M is not None else self._init_mds()
        self.M = [[self.to_field(x) for x in row] for row in M]
        self.M_inv = [list(row) for row in matrix(self.F, self.M).inverse()]

        rcons = rcons if rcons is not None else self._init_rcons(p, t, R)
        self.rcons = [[self.to_field(x) for x in row] for row in rcons]

        # Hash modes
        self.d = d
        self.r = r
        self.c = c

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

    @staticmethod
    def _init_lut() -> list[int]:
        """Tip5 lookup map L : F_257 -> F_257, x -> (x + 1)^3 - 1.
        Restricts to a permutation on {0, ..., 255} since the only
        element outside 8 bits, 256 = -1 mod 257, is a fixed point."""
        return [(pow(x + 1, 3, 257) - 1) % 257 for x in range(256)]

    def _init_mds(self) -> list[list[int]]:
        # TODO check whether the Monolith/RPO Goldilocks T12 row circulant(row=[7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8])
        # should be used instead (KATs were generated with the truncated Tip5 column below, matching the sage reference).
        return tip5_mds_matrix(self.t)

    @staticmethod
    def _init_rcons(p: int, t: int, R: int) -> list[list[int]]:
        """Rxt round constants: constant j is sampled from a fresh Blake3 XOF seeded with
        "Tip5" || byte(j), reading t bytes little-endian and reducing mod p, then scaled
        by 2^-64 so that adding them in Montgomery form is cheap. 
        Matches round constants of Rust reference implementation from 
        https://github.com/Neptune-Crypto/twenty-first."""
        mont_R_inv = pow(2**64, -1, p)
        rcons = []
        for r in range(R):
            row = []
            for i in range(t):
                sampler = FieldElementSampler(b"Tip5" + bytes([i + r * t]), p, xof="blake3", sampling="mod", n_bytes=t)
                row.append((sampler.next() * mont_R_inv) % p)
            rcons.append(row)
        return rcons

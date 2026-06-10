from sage.all import GF, Integer, Matrix

from utils import sample_from_shake_128, invert_LUT

class ReinforcedConcreteParams:
    def __init__(
        self,
        p:         int,
        t:         int,
        R_pre:     int,
        R_bars:    int,
        R_post:    int,
        alpha:     int,
        si:        list[int],
        LUT:       list[int],
        COEFFS:    list[int],
        M:         list[list[int]],
        d:         int,
        r:         int = None,
        c:         int = None,
        alpha_inv: int = None,
        rcons:     list[list[int]] = None,
        kappa:     int = 128,
    ):
        """
        Parameters
        ----------
        p         : field characteristic (prime)
        t         : permutation state size
        R_pre     : number of Bricks+Concrete rounds before the Bars layer(s)
        R_bars    : number of Bars+Concrete rounds in the middle (after R_pre, before R_post)
        R_post    : number of Bricks+Concrete rounds after the Bars layer(s)
        alpha     : power-map exponent for the first state element in Bricks
        si        : bases for decompose/compose in the Bars layer
        LUT       : per-digit lookup table used in Bar (same LUT can be applied to all chunks due to padding in _pad_LUT)
        COEFFS    : Bricks polynomial coefficients
        M         : MDS matrix (txt)
        d         : digest size (number of output elements)
        r         : rate (number of outer state elements absorbed/squeezed per sponge step)
        c         : capacity (number of inner state elements)
        alpha_inv : alpha^{-1} mod (p-1)
        rcons     : Rxt round constants; generated via SHAKE128 if not provided
        kappa     : target security level in bits (default 128)
        """
        assert len(LUT) <= 0xFFFF
        assert len(COEFFS) == 2 and all(len(row) == t - 1 for row in COEFFS)
        assert len(M) == t and all(len(row) == t for row in M) if M is not None else True

        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Rounds
        if R_pre is None or R_bars is None or R_post is None:
            R_pre, R_bars, R_post = self._init_rounds()
        self.R_pre = R_pre
        self.R_bars = R_bars
        self.R_post = R_post
        self.R = R_pre + R_bars + R_post

        # Non-linear layers: Bricks
        self.alpha = alpha
        self.alpha_inv = alpha_inv if alpha_inv is not None else pow(alpha, -1, p - 1)
        self.a_coeffs = [self.to_field(a) for a in COEFFS[0]]
        self.b_coeffs = [self.to_field(b) for b in COEFFS[1]]

        # Non-linear layers: Bars
        self.si = list(si)
        self.LUT = self._pad_LUT(LUT, max(si)) if LUT is not None else self._pad_LUT(self._init_lut(), max(si))
        self.LUT_inv = invert_LUT(self.LUT)

        # Affine layer
        M = M if M is not None else self._init_mds()
        self.M = [[self.to_field(x) for x in row] for row in M]
        self.M_inv = [list(row) for row in Matrix(self.F, self.M).inverse()]

        self.rcons = rcons if rcons is not None else self._init_rcons()
        self.rcons = [[self.to_field(x) for x in row] for row in self.rcons]

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

    def _init_lut(self) -> list[int]:
        # TODO Implement LUT generation via MiMC, as described in the paper.
        raise NotImplementedError(f"LUT creation currently not implemented.")

    def _init_rcons(self) -> list[list[int]]:
        n_bytes = (self.p.bit_length() + 7) // 8
        seed = b"ReinforcedConcrete" + self.p.to_bytes(n_bytes, "little")
        return sample_from_shake_128(seed, self.p, self.R + 1, self.t, sampling="bitmask")
    
    def _init_rounds(self, R_pre, R_bars, R_post) -> (int,int,int):
        # TODO implement
        raise NotImplementedError("Automatic round number derivation not implemented for RC.")

    def _init_mds(self):
        # TODO implement
        raise NotImplementedError("MDS matrix generation not implemented for RC.")

    @staticmethod
    def _pad_LUT(LUT: list[int], max_si: int) -> list[int]:
        out = list(LUT)
        for i in range(len(LUT), max_si):
            out.append(i)
        return out


from sage.all import GF, Integer, Matrix

from utils import FieldElementSampler, invert_LUT

class MonolithParams:
    def __init__(
        self,
        p:     int,
        t:     int,
        R:     int,
        si:    list[int],
        LUTs:  dict[int, list[int]],
        u:     int,
        M:     list[list[int]],
        d:     int,
        r:     int = None,
        c:     int = None,
        rcons: list[list[int]] = None,
        kappa: int = 128,
    ):
        """
        Parameters
        ----------
        p     : field characteristic (prime)
        t     : permutation state size
        R     : number of rounds
        si    : bases for decompose/compose in the Bars layer
        LUTs  : per-digit lookup tables used in Bar (one per distinct value in si)
        u     : number of decomposition S-boxes in the Bars layer
        M     : circulant MDS matrix (txt)
        d     : digest size (number of output elements)
        r     : rate (number of outer state elements absorbed/squeezed per sponge step)
        c     : capacity (number of inner state elements)
        rcons : (R-1)xt round constants; generated via SHAKE128 if not provided
        kappa : target security level in bits (default 128)
        """
        assert len(M) == t and all(len(row) == t for row in M) if M is not None else True

        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Rounds
        self.R = R if R is not None else self._init_rounds()

        # Non-linear layers: Bars
        self.si = si
        self.LUTs = LUTs if LUTs is not None else {s: self._compute_lut(s) for s in set(si)}
        self.LUTs_inv = {s: invert_LUT(self.LUTs[s]) for s in self.LUTs}
        self.u = u

        # Affine layer
        M = M if M is not None else self._init_mds()
        self.M = [[self.to_field(x) for x in row] for row in M]
        self.M_inv = [list(row) for row in Matrix(self.F, self.M).inverse()]

        self.rcons = rcons if rcons is not None else self._init_rcons()
        self.rcons = [[self.to_field(x) for x in row] for row in self.rcons]

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
    
    def _init_luts(self) -> dict[int, list[int]]:
        LUTs = {}
        for s in set(self.si):
            if s == 8:
                LUTs[s] = compute_lut_8()
            elif s == 7:
                LUTs[s] = compute_lut_7()
            else:
                raise NotImplementedError(f"Unsupported si value: {s}.")
        return LUTs
        
    def _init_rcons(self) -> list[list[int]]:
        import struct
        bits = self.p.bit_length()
        seed = (b"Monolith"
                + bytes([self.t, self.R])
                + (struct.pack('<I', self.p) if bits <= 32 else struct.pack('<Q', self.p))
                + (bytes([8, 8, 8, 7])       if bits <= 32 else bytes([8] * 8)))
        return FieldElementSampler(seed, self.p, xof="shake_128", sampling="naive").grid(self.R - 1, self.t)
    
    def _init_rounds(self) -> int:
        # TODO implement
        raise NotImplementedError("Automatic round number derivation not implemented for Monolith.")

    def _init_mds(self):
        # TODO implement
        raise NotImplementedError("MDS matrix generation not implemented for Monolith.")


def compute_lut_8() -> list[int]:
    table = []
    for x in range(256):
        l1 = ((x & 0x80) >> 7) | ((x & 0x7F) << 1)
        l2 = ((x & 0xC0) >> 6) | ((x & 0x3F) << 2)
        l3 = ((x & 0xE0) >> 5) | ((x & 0x1F) << 3)
        tmp = (x ^ ((~l1) & l2 & l3)) & 0xFF
        table.append(((tmp & 0x80) >> 7) | ((tmp & 0x7F) << 1))
    return table

def compute_lut_7() -> list[int]:
    table = []
    for x in range(128):
        l1 = ((x >> 6) | (x << 1)) & 0x7F
        l2 = ((x >> 5) | (x << 2)) & 0x7F
        tmp = (x ^ ((~l1) & l2)) & 0x7F
        table.append(((tmp >> 6) | (tmp << 1)) & 0x7F)
    return table

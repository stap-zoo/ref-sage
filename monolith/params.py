# params.py
# ---------------------------------------------------------------------------
# Parameter definition for Monolith: the MonolithParams class.
#
# MonolithParams is the single source of truth for an instance. It sanitizes
# the user-facing parameters and expands them into a fully-specified instance
# that the permutation, hash modes, instances and tests consume. Any value the
# user omits is filled in by the matching _init_* helper (or, for r/c/d, by the
# shared derive_rate_capacity_digest). Settings that depart from the recommended
# ones raise a ParamRecommendationWarning rather than an error.
# ---------------------------------------------------------------------------

# Structural imports
from recommendations import ParamRecommendationWarning
from types import SimpleNamespace

# Math specific imports
import warnings
from sage.all import GF, Integer

# Custom imports
from utils import XOFFieldElementSampler, invert_LUT, map_to_field, invert_matrix
from modes import derive_rate_capacity_digest


class MonolithParams:
    def __init__(
        self,
        p:     int,
        t:     int,
        si:    list[int],
        u:     int,
        R:     int = None,
        LUTs:  dict[int, list[int]] = None,
        M:     list[list[int]] = None,
        rcons: list[list[int]] = None,
        r:     int = None,
        c:     int = None,
        d:     int = None,
        kappa: int = 128,
    ):
        """
        Parameters
        ----------
        p     : field characteristic (prime)
        t     : permutation state size
        si    : bases for decompose/compose in the Bars layer
        u     : number of decomposition S-boxes in the Bars layer
        R     : number of rounds; derived via _init_R if not provided
        LUTs  : per-digit lookup tables used in Bar (one per distinct value in si); generated via _init_LUTs if not provided
        M     : circulant MDS matrix (txt); generated via _init_M if not provided
        rcons : (R-1)xt round constants; generated via _init_rcons (SHAKE128) if not provided
        r     : rate (number of outer state elements absorbed/squeezed per sponge step); derived if not provided
        c     : capacity (number of inner state elements); derived if not provided
        d     : digest size (number of output elements); derived if not provided
        kappa : target security level in bits (default 128)
        """

        # Input sanitization
        MonolithParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Non-linear layers: Bars
        self.si = si
        self.LUTs = LUTs if LUTs is not None else self._init_LUTs()
        self.LUTs_inv = {s: invert_LUT(self.LUTs[s]) for s in self.LUTs}
        self.u = u

        # Hash modes
        self.r, self.c, self.d = derive_rate_capacity_digest(self.kappa, self.t, r, c, d)

        # Rounds (set before _init_rcons, whose derivation depends on R)
        self.R = R if R is not None else self._init_R()

        # Affine layer
        self.M = map_to_field(M if M is not None else self._init_M(), self.to_field)
        self.M_inv = invert_matrix(self.M)

        # Round constants: pad with a trailing zero row so AffineLayer can uniformly index
        # rcons[round_idx] for round_idx in 0..R-1 (the final round has no round constants).
        rcons = rcons if rcons is not None else self._init_rcons()
        self.rcons = map_to_field(rcons, self.to_field) + [[self.F.zero()] * self.t]

    # ---------------------------------------------------------------------------
    # Small field conversion helpers
    # ---------------------------------------------------------------------------

    def from_field(self, el) -> Integer:
        return Integer(el)

    def to_field(self, n: int):
        return self.F(n)

    # ---------------------------------------------------------------------------
    # Input sanitization and security requirements
    # ---------------------------------------------------------------------------

    @staticmethod
    def _input_sanitization(params):
        """Validate the raw constructor arguments: hard checks raise, recommendation
        deviations warn (ParamRecommendationWarning) but do not raise."""

        # --- Hard checks (must always hold) ---
        if params.p == 2:
            raise NotImplementedError("Characteristic 2 not implemented")
        if params.t < 1:
            raise ValueError(f"state size t must be positive. Got {params.t}")
        if params.M is not None and not (len(params.M) == params.t and all(len(row) == params.t for row in params.M)):
            raise ValueError(f"M must be a {params.t}x{params.t} matrix")

        # --- Warnings (recommended, not required) ---
        field_bits = int(params.p).bit_length()
        if field_bits < 31:
            warnings.warn(f"TOY VERSION: field is only {field_bits} bits", ParamRecommendationWarning, stacklevel=2)

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_LUTs(self) -> dict[int, list[int]]:
        # si entries are the radix bases (e.g. 256 / 128), mapping to the 8-bit / 7-bit
        # Bar lookup tables respectively.
        LUTs = {}
        for s in set(self.si):
            if s == 256:
                LUTs[s] = compute_lut_8()
            elif s == 128:
                LUTs[s] = compute_lut_7()
            else:
                raise NotImplementedError(f"Bar LUT generation not implemented for base si={s}.")
        return LUTs

    def _init_rcons(self) -> list[list[int]]:
        import struct
        bits = self.p.bit_length()
        seed = (b"Monolith"
                + bytes([self.t, self.R])
                + (struct.pack('<I', self.p) if bits <= 32 else struct.pack('<Q', self.p))
                + (bytes([8, 8, 8, 7])       if bits <= 32 else bytes([8] * 8)))
        return XOFFieldElementSampler(seed=seed, p=self.p, xof="shake_128", sampling="naive").grid(self.R - 1, self.t)

    def _init_R(self) -> int:
        # TODO implement
        raise NotImplementedError("Automatic round number derivation not implemented for Monolith.")

    def _init_M(self):
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

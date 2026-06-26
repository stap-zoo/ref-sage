"""Deterministic field-element samplers.

Pseudo-random samplers that turn a seed into a reproducible stream of field elements,
used to generate round constants (and similar parameters) deterministically from a
fixed seed (XOF-based and LFSR-based variants).
"""

from hashlib import shake_128, shake_256, sha256
from blake3 import blake3
from math import ceil

# ---------------------------------------------------------------------------
# Sample field elements
# ---------------------------------------------------------------------------

# Byte-stream sources: each entry maps a name to a constructor taking the seed and returning an
# object whose digest(n) yields the first n bytes of the stream. shake/blake3 are true XOFs
# (unbounded); sha256 is a fixed 32-byte digest exposed through the same interface (bounded -- the
# sampler raises if more than 32 bytes are drawn), used for Tip5's MDS column.

class _SHA256:
    """SHA-256 as a bounded (32-byte) byte source with the XOF digest(n) interface."""
    def __init__(self, seed: bytes):
        self._bytes = sha256(seed).digest()

    def digest(self, n: int) -> bytes:
        return self._bytes[:n]   # capped at 32 bytes; XOFFieldElementSampler raises if it needs more

XOFS = {
    "shake_128": shake_128,
    "shake_256": shake_256,
    "blake3": blake3,
    "sha256": _SHA256,
}

class FieldElementSampler:
    """Abstract base for deterministic field-element samplers in [0, p). A subclass supplies the
    raw candidate value via _draw_candidate(); this base maps it to a field element according to
    the sampling strategy and exposes next / next_nonzero / grid:

    sampling="bitmask" : draw the field's serialized width and trim to its exact bit-length, reject (resample) if >= p.
    sampling="naive"   : draw the field's serialized width but do NOT trim to the exact bit-length, reject if >= p
                         (simpler, higher rejection rate than bitmask).
    sampling="mod"     : draw a slightly wider value and reduce mod p (no rejection).

    The concrete width/encoding of a "draw" is the subclass's business (XOF byte chunks vs LFSR
    bit collection); only the reject-vs-reduce decision lives here. The strategy can be switched
    mid-stream via set_sampling -- e.g. Poseidon draws round constants with "bitmask" then
    switches to "mod" to draw the MDS matrix off the same sampler.
    """

    SAMPLINGS = ("bitmask", "naive", "mod")

    def __init__(self, p: int):
        self.p = p

    def set_sampling(self, sampling: str) -> None:
        """Set/switch the sampling strategy and recompute the derived draw parameters. Subclasses
        set up their underlying stream first, then call this from __init__."""
        if sampling not in self.SAMPLINGS:
            raise ValueError(f"Unknown sampling strategy: {sampling}. Use one of {self.SAMPLINGS}.")
        self.sampling = sampling
        self.reduce_mod = (sampling == "mod")
        self._configure_sampling()

    def _configure_sampling(self) -> None:
        """Recompute the strategy-dependent draw width/mask (subclass-specific)."""
        raise NotImplementedError

    def _draw_candidate(self) -> int:
        """One raw candidate integer from the underlying stream (subclass-specific width)."""
        raise NotImplementedError

    def next(self) -> int:
        """Sample the next field element from the stream."""
        while True:
            val = self._draw_candidate()
            if self.reduce_mod:
                return val % self.p
            if val < self.p:    # rejection sampling
                return val

    def next_nonzero(self) -> int:
        """Sample the next field element from the stream, skipping zeros."""
        while True:
            val = self.next()
            if val != 0:
                return val

    def grid(self, num_rows: int, num_cols: int) -> list[list[int]]:
        """Sample a num_rows x num_cols grid of field elements."""
        return [[self.next() for _ in range(num_cols)] for _ in range(num_rows)]


class XOFFieldElementSampler(FieldElementSampler):
    """Field-element sampler reading a seeded XOF byte stream, cut into fixed-size little-endian
    chunks (see FieldElementSampler for the sampling strategies):

    sampling="bitmask" : reads ceil(p.bit_length()/8) bytes, zeroes bits above p.bit_length() in the last byte.
                         Matches the Rust field_element_from_shake / ff::PrimeField::from_repr from
                         https://extgit.isec.tugraz.at/krypto/zkfriendlyhashzoo/-/blob/master/plain_impls/src/fields/utils.rs?ref_type=heads.
                         Used in Reinforced Concrete and Griffin (with xof="shake_128").
    sampling="naive"   : reads ceil(p.bit_length()/8) bytes like bitmask, but leaves the top byte unmasked,
                         so it rejects whenever the value lands in [p, 256^n_bytes) -- the plain "read the field's
                         byte width and reject" approach. Used in Monolith (with xof="shake_128"). A deliberately
                         different read width (an efficiency choice of the primitive) is an explicit n_bytes override.
    sampling="mod"     : reads ceil(p.bit_length()/8)+1 bytes. Matches Rescue Prime / RPO's get_round_constants from
                         https://github.com/KULeuven-COSIC/Marvellous and https://github.com/ASDiscreteMathematics/rpo.
                         Used in Rescue, Rescue Prime / RPO, and Arion (with xof="shake_256").

    n_bytes overrides the strategy's chunk size, e.g. for Tip5's round constants (xof="blake3", sampling="mod"),
    which read t bytes per element instead of ceil(p.bit_length()/8)+1.
    """

    def __init__(self, *, seed: bytes, p: int, sampling: str, xof: str = "shake_128", n_bytes: int = None, endianess="little"):
        super().__init__(p)
        if xof not in XOFS:
            raise ValueError(f"Unknown XOF: {xof}. Use one of {sorted(XOFS)}.")
        self._xof = XOFS[xof](seed)
        self._n_bytes_override = n_bytes
        self.set_sampling(sampling)        # sets n_bytes / mask (+ reduce_mod)
        self.endianess = endianess

        self._pos = 0
        self._size = max(1024, self.n_bytes * 64)
        self._buf = self._xof.digest(self._size)

    def _configure_sampling(self) -> None:
        bits = self.p.bit_length()
        if self.sampling == "bitmask":
            self.n_bytes = ceil(bits / 8)
            mod = bits % 8
            self.mask = ((1 << mod) - 1) if mod != 0 else 0xFF
        elif self.sampling == "naive":
            self.n_bytes = ceil(bits / 8)
            self.mask = 0xFF
        else:  # "mod"
            self.n_bytes = ceil(bits / 8) + 1
            self.mask = 0xFF
        if self._n_bytes_override is not None:
            self.n_bytes = self._n_bytes_override

    def _ensure(self, n: int) -> None:
        """Grow the buffered XOF output until at least n unread bytes are available.
        An XOF's longer digest is an extension of its shorter one, so re-requesting
        the doubled size extends the stream while keeping every previously read
        position valid. A bounded source (e.g. sha256) stops growing -- raise then."""
        while len(self._buf) - self._pos < n:
            self._size *= 2
            grown = self._xof.digest(self._size)
            if len(grown) == len(self._buf):
                raise ValueError("byte source exhausted (bounded digest cannot provide more)")
            self._buf = grown

    def _draw_candidate(self) -> int:
        self._ensure(self.n_bytes)
        raw = bytearray(self._buf[self._pos:self._pos + self.n_bytes])
        self._pos += self.n_bytes
        raw[-1] &= self.mask
        return int.from_bytes(raw, self.endianess)


class LFSRFieldElementSampler(FieldElementSampler):
    """Field-element sampler reading an LFSR bit stream, used by Poseidon / Poseidon2 to derive
    round constants (and, in the original spec, the MDS matrix) deterministically. The Grain LFSR
    of Poseidon (Appendix E of https://eprint.iacr.org/2019/458) is the instance with taps
    [0, 13, 23, 38, 51, 62] over an 80-bit state.

    The LFSR has a `state_size`-bit register, seeded with `seed_bits`, and is warmed up by `warmup`
    discarded steps (default 2*state_size -- two full passes through the register, i.e. 160 for the
    80-bit Grain LFSR). Per draw, candidate bits are collected MSB-first and mapped to a field element
    per the sampling strategy (see FieldElementSampler). `taps` are the feedback tap indices. The
    seed-bit layout is primitive-specific (it encodes the instance parameters) and is built by the
    caller -- see the Poseidon/Poseidon2 classes in hades/params.py. `shrink` selects the output-bit
    extraction:

      shrink=True  : self-shrinking generator -- read pairs (b0, b1) and keep b1 iff b0 == 1.
                     Used by the original generate_parameters_grain.sage (hadeshash layout).
      shrink=False : the raw clocked bit is used directly (khovratovich/poseidon-tools layout).
    """

    def __init__(self, *, seed_bits: list[int], p: int, taps: list[int], state_size: int,
                 sampling: str, warmup: int = None, shrink: bool = False):
        super().__init__(p)
        if len(seed_bits) != state_size:
            raise ValueError(f"seed_bits length {len(seed_bits)} does not match state_size {state_size}")
        self.n = p.bit_length()
        self.taps = taps
        self.state_size = state_size
        self._shrink = shrink
        self._state = list(seed_bits)
        for _ in range(warmup if warmup is not None else 2 * state_size):   # warm up
            self._next_bit()
        self.set_sampling(sampling)        # sets _cand_bits (+ reduce_mod)

    def _configure_sampling(self) -> None:
        # candidate width per sampling strategy (the bit analogue of the XOF byte widths)
        self._cand_bits = (32 if self.n <= 32 else 64) if self.sampling == "naive" else self.n

    @staticmethod
    def to_bits(value: int, width: int) -> list[int]:
        """Big-endian (MSB-first) bit decomposition of `value` into `width` bits.
        Helper for callers assembling the seed."""
        return [(value >> (width - 1 - i)) & 1 for i in range(width)]

    def _next_bit(self) -> int:
        s = self._state
        new = 0
        for i in self.taps:
            new ^= s[i]
        self._state = s[1:] + [new]
        return new

    def _next_output_bit(self) -> int:
        """One output bit: self-shrinking (read pairs (b0, b1), keep b1 iff b0 == 1) when
        shrink is set, otherwise the raw clocked bit."""
        if not self._shrink:
            return self._next_bit()
        while True:
            b0 = self._next_bit()
            b1 = self._next_bit()
            if b0 == 1:
                return b1

    def _draw_candidate(self) -> int:
        val = 0
        for _ in range(self._cand_bits):
            val = (val << 1) | self._next_output_bit()
        return val

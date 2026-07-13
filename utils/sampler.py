"""Deterministic field-element / integer samplers for arithmetization-oriented primitives.

Pseudo-random samplers that turn a seed into a reproducible stream of field elements,
used to generate round constants (and similar parameters) deterministically from a
fixed seed (XOF-based and LFSR-based variants).

Three classes:

  FieldElementSampler      -- abstract base: sampling strategies, next / next_nonzero /
                              grid for field elements, and randint for bounded uniform
                              integers (HashTape semantics).
  XOFFieldElementSampler   -- byte-tape source (SHAKE128/256, BLAKE3, ...): Reinforced
                              Concrete, Griffin, Monolith, Rescue (Prime / RPO), Arion,
                              Tip5, Polocolo.
  LFSRFieldElementSampler  -- bit-stream source (Grain LFSR): Poseidon / Poseidon2.

The split of responsibilities:

  * The SUBCLASS owns the raw stream and knows how to produce one candidate integer of
    a requested bit width (`_draw_candidate`).
  * The BASE class owns the reject-vs-reduce decision (`next`) and the bounded-integer
    rejection loop (`randint`).
"""
from hashlib import shake_128, shake_256, sha256
from blake3 import blake3
from math import ceil

# Byte-stream sources: each entry maps a name to a constructor taking the seed and returning an
# object whose digest(n) yields the first n bytes of the stream. shake/blake3 are true XOFs
# (unbounded; blake3's digest(length=32, *, seek=0) takes the length positionally, so it fits
# the hashlib interface directly). sha256 is a fixed 32-byte digest exposed through the same
# interface (bounded -- the sampler raises if more than 32 bytes are drawn), used for Tip5's
# MDS column.
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
    """Abstract base for deterministic samplers over [0, p). A subclass supplies raw
    candidate integers via _draw_candidate(); this base maps them to field elements
    according to the sampling strategy and exposes next / next_nonzero / grid / randint.

    Sampling strategies (how a raw draw becomes a field element in `next`):

      sampling="bitmask" : draw the field's serialized width and trim to its exact
                           bit-length, reject (resample) if >= p.
      sampling="bitshift": like bitmask, but trim by right-SHIFTING the most significant
                           byte instead of masking it (keeps its high bits, drops its low
                           bits), reject if >= p. Only meaningful for byte-granular
                           sources; bit-granular sources trim nothing (see subclasses).
      sampling="naive"   : draw the field's serialized width but do NOT trim to the exact
                           bit-length, reject if >= p (simpler, higher rejection rate
                           than bitmask).
      sampling="mod"     : draw a slightly wider value and reduce mod p (no rejection).

    The concrete width/encoding of a "draw" is the subclass's business (XOF byte chunks
    vs LFSR bit collection); only the reject-vs-reduce decision lives here. The strategy
    can be switched mid-stream via set_sampling -- e.g. Poseidon draws round constants
    with "bitmask" then switches to "mod" to draw the MDS matrix off the same sampler.

    randint(mod) draws bounded uniform integers off the SAME stream (advancing the same
    position), with deliberately different semantics from next():

      * per-call width (mod-1).bit_length() instead of the field width -- so e.g.
        randint(1024) is a 10-bit draw. Note that for a power of two mod it never rejects, 
        whereas mod.bit_length() would reject half the draws.
      * ALWAYS rejects, never mod-reduces, regardless of the configured strategy --
        uniformity of randint must not depend on how field elements are encoded.
      * bypasses any subclass width override (n_bytes and the like): those are
        field-element encoding quirks, not part of "give me k uniform bits".

    This matches the HashTape of Polocolo's param_gen.sage when the sampler is a
    big-endian bitshift XOF sampler; used there to rejection-sample the S-box
    permutation sigma. CAUTION: for reproducibility, the exact SEQUENCE of next()
    and randint() calls (and their arguments) is part of a derivation's specification
    -- the stream position advances by data-dependent amounts.
    """

    SAMPLINGS = ("bitmask", "bitshift", "naive", "mod")

    def __init__(self, p: int):
        self.p = p

    def set_sampling(self, sampling: str) -> None:
        """Set/switch the sampling strategy and recompute the derived draw parameters.
        Subclasses set up their underlying stream first, then call this from __init__."""
        if sampling not in self.SAMPLINGS:
            raise ValueError(f"Unknown sampling strategy: {sampling}. Use one of {self.SAMPLINGS}.")
        self.sampling = sampling
        self.reduce_mod = (sampling == "mod")
        self._configure_sampling()

    def _configure_sampling(self) -> None:
        """Recompute the strategy-dependent draw width/mask (subclass-specific)."""
        raise NotImplementedError

    def _draw_candidate(self, bits: int = None) -> int:
        """One raw candidate integer from the underlying stream.

        bits=None : the field's configured width per the current strategy (used by next).
        bits=k    : exactly a k-bit candidate in [0, 2^k) (used by randint), independent
                    of strategy-level width overrides.
        """
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

    def randint(self, mod: int) -> int:
        """Uniform integer in [0, mod) by rejection sampling on (mod-1).bit_length()-bit
        draws. Always rejects (never mod-reduces), whatever the configured strategy.
        Reproduces Polocolo's HashTape.randint when the sampler is
        XOFFieldElementSampler(sampling="bitshift", endianess="big")."""
        if mod <= 1:
            if mod < 1:
                raise ValueError("mod must be >= 1")
            return 0            # would be a 0-bit draw; consume nothing
        bits = (mod - 1).bit_length()
        while True:
            val = self._draw_candidate(bits)
            if val < mod:
                return val


class XOFFieldElementSampler(FieldElementSampler):
    """Field-element sampler reading a seeded XOF byte stream, cut into fixed-size
    chunks (see FieldElementSampler for the sampling strategies):

    sampling="bitmask" : reads ceil(bits/8) bytes, zeroes bits above the bit-length in the
                         most significant byte. Matches the Rust field_element_from_shake /
                         ff::PrimeField::from_repr from
                         https://extgit.isec.tugraz.at/krypto/zkfriendlyhashzoo/-/blob/master/plain_impls/src/fields/utils.rs?ref_type=heads.
                         Used in Reinforced Concrete and Griffin (with xof="shake_128").
    sampling="bitshift": reads ceil(bits/8) bytes and right-shifts the MOST significant byte
                         by the excess bits (keeping its high bits) instead of masking it.
                         Matches the HashTape squeeze of Polocolo's param_gen.sage
                         (seq[0] >>= (-size % 8) on a big-endian read),
                         https://github.com/KAIST-CryptLab/Polocolo. Used in Polocolo
                         (with xof="shake_128", endianess="big").
    sampling="naive"   : reads ceil(bits/8) bytes like bitmask, but leaves the top byte
                         unmasked, so it rejects whenever the value lands in
                         [p, 256^n_bytes) -- the plain "read the field's byte width and
                         reject" approach. Used in Monolith (with xof="shake_128"). A
                         deliberately different read width (an efficiency choice of the
                         primitive) is an explicit n_bytes override.
    sampling="mod"     : reads ceil(bits/8)+1 bytes. Matches Rescue Prime / RPO's
                         get_round_constants from
                         https://github.com/KULeuven-COSIC/Marvellous and
                         https://github.com/ASDiscreteMathematics/rpo.
                         Used in Rescue, Rescue Prime / RPO, and Arion (with
                         xof="shake_256").

    n_bytes overrides the strategy's chunk size for FIELD-ELEMENT draws only, e.g. for
    Tip5's round constants (xof="blake3", sampling="mod"), which read t bytes per element
    instead of ceil(p.bit_length()/8)+1. randint() ignores the override (its width is the
    per-call (mod-1).bit_length()).

    randint() note: the trim rule of the current strategy applies to randint draws too
    (same bytes, different top-byte treatment), so randint streams differ between
    "bitmask" and "bitshift" samplers even though both are uniform. Polocolo's sigma
    derivation is specified on the (bitshift, big-endian) stream.
    """

    def __init__(self, *, seed: bytes, p: int, sampling: str, xof: str = "shake_128",
                 n_bytes: int = None, endianess: str = "little"):
        super().__init__(p)
        if xof not in XOFS:
            raise ValueError(f"Unknown XOF: {xof}. Use one of {sorted(XOFS)}.")
        self._xof = XOFS[xof](seed)
        self._n_bytes_override = n_bytes
        self.set_sampling(sampling)        # sets n_bytes / mask / shift (+ reduce_mod)
        self.endianess = endianess

        self._pos = 0
        self._size = max(1024, self.n_bytes * 64)
        self._buf = self._xof.digest(self._size)

    def _trim_params(self, bits: int) -> tuple:
        """(n_bytes, mask, shift) for a `bits`-bit draw under the current strategy.
        `mask` applies to the most significant byte unless the strategy is "bitshift",
        in which case `shift` does (see _draw_candidate)."""
        n_bytes = ceil(bits / 8)
        if self.sampling == "bitmask":
            mod8 = bits % 8
            return n_bytes, ((1 << mod8) - 1) if mod8 != 0 else 0xFF, 0
        if self.sampling == "bitshift":
            return n_bytes, 0xFF, (-bits) % 8
        if self.sampling == "naive":
            return n_bytes, 0xFF, 0
        return n_bytes + 1, 0xFF, 0        # "mod": one byte wider, no trim

    def _configure_sampling(self) -> None:
        self.n_bytes, self.mask, self.shift = self._trim_params(self.p.bit_length())
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

    def _draw_candidate(self, bits: int = None) -> int:
        if bits is None:   # field-element draw: configured (possibly overridden) width
            n_bytes, mask, shift = self.n_bytes, self.mask, self.shift
        else:              # randint draw: per-call width, override does not apply
            n_bytes, mask, shift = self._trim_params(bits)
        self._ensure(n_bytes)
        raw = bytearray(self._buf[self._pos:self._pos + n_bytes])
        self._pos += n_bytes
        if self.sampling == "bitshift":
            # trim the MOST significant byte by shifting (endianess decides which byte that is)
            raw[0 if self.endianess == "big" else -1] >>= shift
        else:
            raw[-1 if self.endianess == "little" else 0] &= mask
        return int.from_bytes(raw, self.endianess)


class LFSRFieldElementSampler(FieldElementSampler):
    """Field-element sampler reading an LFSR bit stream, used by Poseidon / Poseidon2 to
    derive round constants (and, in the original spec, the MDS matrix) deterministically.
    The Grain LFSR of Poseidon (Appendix E of https://eprint.iacr.org/2019/458) is the
    instance with taps [0, 13, 23, 38, 51, 62] over an 80-bit state.

    The LFSR has a `state_size`-bit register, seeded with `seed_bits`, and is warmed up by
    `warmup` discarded steps (default 2*state_size -- two full passes through the register,
    i.e. 160 for the 80-bit Grain LFSR). Per draw, candidate bits are collected MSB-first
    and mapped to a field element per the sampling strategy (see FieldElementSampler).
    `taps` are the feedback tap indices. The seed-bit layout is primitive-specific (it
    encodes the instance parameters) and is built by the caller -- see the
    Poseidon/Poseidon2 classes in hades/params.py. `shrink` selects the output-bit
    extraction:

      shrink=True  : self-shrinking generator -- read pairs (b0, b1) and keep b1 iff
                     b0 == 1. Used by the original generate_parameters_grain.sage
                     (hadeshash layout).
      shrink=False : the raw clocked bit is used directly (khovratovich/poseidon-tools
                     layout).

    Strategy widths for field-element draws: "naive" collects a machine-word width
    (32/64 bits) and rejects; every other strategy collects exactly p.bit_length() bits
    ("bitmask" and "bitshift" coincide here -- a bit-granular source has no excess to
    trim -- and "mod" reduces the n-bit draw mod p, slight bias and all, matching the
    Poseidon script variants; unlike the XOF "mod", it does NOT draw wider).

    randint() on this source is strategy-independent: it collects exactly
    (mod-1).bit_length() output bits and rejects -- no trimming exists at bit
    granularity. Note that each randint call advances the register by a variable
    amount (rejections; and with shrink=True each output bit consumes a geometrically
    distributed number of clocked bits, mean 4), so the call sequence is part of any
    derivation's specification.
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

    def _draw_candidate(self, bits: int = None) -> int:
        val = 0
        for _ in range(self._cand_bits if bits is None else bits):
            val = (val << 1) | self._next_output_bit()
        return val

# class FieldElementSampler:
#     """Abstract base for deterministic field-element samplers in [0, p). A subclass supplies the
#     raw candidate value via _draw_candidate(); this base maps it to a field element according to
#     the sampling strategy and exposes next / next_nonzero / grid:

#     sampling="bitmask" : draw the field's serialized width and trim to its exact bit-length, reject (resample) if >= p.
#     sampling="bitshift": like bitmask, but trim by right-SHIFTING the most significant byte instead of
#                          masking it (keeps its high bits, drops its low bits), reject if >= p.
#     sampling="naive"   : draw the field's serialized width but do NOT trim to the exact bit-length, reject if >= p
#                          (simpler, higher rejection rate than bitmask).
#     sampling="mod"     : draw a slightly wider value and reduce mod p (no rejection).

#     The concrete width/encoding of a "draw" is the subclass's business (XOF byte chunks vs LFSR
#     bit collection); only the reject-vs-reduce decision lives here. The strategy can be switched
#     mid-stream via set_sampling -- e.g. Poseidon draws round constants with "bitmask" then
#     switches to "mod" to draw the MDS matrix off the same sampler.
#     """

#     SAMPLINGS = ("bitmask", "bitshift", "naive", "mod")

#     def __init__(self, p: int):
#         self.p = p

#     def set_sampling(self, sampling: str) -> None:
#         """Set/switch the sampling strategy and recompute the derived draw parameters. Subclasses
#         set up their underlying stream first, then call this from __init__."""
#         if sampling not in self.SAMPLINGS:
#             raise ValueError(f"Unknown sampling strategy: {sampling}. Use one of {self.SAMPLINGS}.")
#         self.sampling = sampling
#         self.reduce_mod = (sampling == "mod")
#         self._configure_sampling()

#     def _configure_sampling(self) -> None:
#         """Recompute the strategy-dependent draw width/mask (subclass-specific)."""
#         raise NotImplementedError

#     def _draw_candidate(self) -> int:
#         """One raw candidate integer from the underlying stream (subclass-specific width)."""
#         raise NotImplementedError

#     def next(self) -> int:
#         """Sample the next field element from the stream."""
#         while True:
#             val = self._draw_candidate()
#             if self.reduce_mod:
#                 return val % self.p
#             if val < self.p:    # rejection sampling
#                 return val

#     def next_nonzero(self) -> int:
#         """Sample the next field element from the stream, skipping zeros."""
#         while True:
#             val = self.next()
#             if val != 0:
#                 return val

#     def grid(self, num_rows: int, num_cols: int) -> list[list[int]]:
#         """Sample a num_rows x num_cols grid of field elements."""
#         return [[self.next() for _ in range(num_cols)] for _ in range(num_rows)]


# class XOFFieldElementSampler(FieldElementSampler):
#     """Field-element sampler reading a seeded XOF byte stream, cut into fixed-size little-endian
#     chunks (see FieldElementSampler for the sampling strategies):

#     sampling="bitmask" : reads ceil(p.bit_length()/8) bytes, zeroes bits above p.bit_length() in the last byte.
#                          Matches the Rust field_element_from_shake / ff::PrimeField::from_repr from
#                          https://extgit.isec.tugraz.at/krypto/zkfriendlyhashzoo/-/blob/master/plain_impls/src/fields/utils.rs?ref_type=heads.
#                          Used in Reinforced Concrete and Griffin (with xof="shake_128").
#     sampling="bitshift": reads ceil(p.bit_length()/8) bytes and right-shifts the MOST significant byte by
#                          the excess bits (keeping its high bits) instead of masking it. Matches the HashTape
#                          squeeze of Polocolo's param_gen.sage (seq[0] >>= (-size % 8) on a big-endian read),
#                          https://github.com/KAIST-CryptLab/Polocolo. Used in Polocolo
#                          (with xof="shake_128", endianess="big").
#     sampling="naive"   : reads ceil(p.bit_length()/8) bytes like bitmask, but leaves the top byte unmasked,
#                          so it rejects whenever the value lands in [p, 256^n_bytes) -- the plain "read the field's
#                          byte width and reject" approach. Used in Monolith (with xof="shake_128"). A deliberately
#                          different read width (an efficiency choice of the primitive) is an explicit n_bytes override.
#     sampling="mod"     : reads ceil(p.bit_length()/8)+1 bytes. Matches Rescue Prime / RPO's get_round_constants from
#                          https://github.com/KULeuven-COSIC/Marvellous and https://github.com/ASDiscreteMathematics/rpo.
#                          Used in Rescue, Rescue Prime / RPO, and Arion (with xof="shake_256").

#     n_bytes overrides the strategy's chunk size, e.g. for Tip5's round constants (xof="blake3", sampling="mod"),
#     which read t bytes per element instead of ceil(p.bit_length()/8)+1.
#     """

#     def __init__(self, *, seed: bytes, p: int, sampling: str, xof: str = "shake_128", n_bytes: int = None, endianess="little"):
#         super().__init__(p)
#         if xof not in XOFS:
#             raise ValueError(f"Unknown XOF: {xof}. Use one of {sorted(XOFS)}.")
#         self._xof = XOFS[xof](seed)
#         self._n_bytes_override = n_bytes
#         self.set_sampling(sampling)        # sets n_bytes / mask (+ reduce_mod)
#         self.endianess = endianess

#         self._pos = 0
#         self._size = max(1024, self.n_bytes * 64)
#         self._buf = self._xof.digest(self._size)

#     def _configure_sampling(self) -> None:
#         bits = self.p.bit_length()
#         self.shift = 0
#         if self.sampling == "bitmask":
#             self.n_bytes = ceil(bits / 8)
#             mod = bits % 8
#             self.mask = ((1 << mod) - 1) if mod != 0 else 0xFF
#         elif self.sampling == "bitshift":
#             self.n_bytes = ceil(bits / 8)
#             self.mask = 0xFF
#             self.shift = (-bits) % 8
#         elif self.sampling == "naive":
#             self.n_bytes = ceil(bits / 8)
#             self.mask = 0xFF
#         else:  # "mod"
#             self.n_bytes = ceil(bits / 8) + 1
#             self.mask = 0xFF
#         if self._n_bytes_override is not None:
#             self.n_bytes = self._n_bytes_override

#     def _ensure(self, n: int) -> None:
#         """Grow the buffered XOF output until at least n unread bytes are available.
#         An XOF's longer digest is an extension of its shorter one, so re-requesting
#         the doubled size extends the stream while keeping every previously read
#         position valid. A bounded source (e.g. sha256) stops growing -- raise then."""
#         while len(self._buf) - self._pos < n:
#             self._size *= 2
#             grown = self._xof.digest(self._size)
#             if len(grown) == len(self._buf):
#                 raise ValueError("byte source exhausted (bounded digest cannot provide more)")
#             self._buf = grown

#     def _draw_candidate(self) -> int:
#         self._ensure(self.n_bytes)
#         raw = bytearray(self._buf[self._pos:self._pos + self.n_bytes])
#         self._pos += self.n_bytes
#         if self.sampling == "bitshift":
#             # trim the MOST significant byte by shifting (endianess decides which byte that is)
#             raw[0 if self.endianess == "big" else -1] >>= self.shift
#         else:
#             raw[-1] &= self.mask
#         return int.from_bytes(raw, self.endianess)


# class LFSRFieldElementSampler(FieldElementSampler):
#     """Field-element sampler reading an LFSR bit stream, used by Poseidon / Poseidon2 to derive
#     round constants (and, in the original spec, the MDS matrix) deterministically. The Grain LFSR
#     of Poseidon (Appendix E of https://eprint.iacr.org/2019/458) is the instance with taps
#     [0, 13, 23, 38, 51, 62] over an 80-bit state.

#     The LFSR has a `state_size`-bit register, seeded with `seed_bits`, and is warmed up by `warmup`
#     discarded steps (default 2*state_size -- two full passes through the register, i.e. 160 for the
#     80-bit Grain LFSR). Per draw, candidate bits are collected MSB-first and mapped to a field element
#     per the sampling strategy (see FieldElementSampler). `taps` are the feedback tap indices. The
#     seed-bit layout is primitive-specific (it encodes the instance parameters) and is built by the
#     caller -- see the Poseidon/Poseidon2 classes in hades/params.py. `shrink` selects the output-bit
#     extraction:

#       shrink=True  : self-shrinking generator -- read pairs (b0, b1) and keep b1 iff b0 == 1.
#                      Used by the original generate_parameters_grain.sage (hadeshash layout).
#       shrink=False : the raw clocked bit is used directly (khovratovich/poseidon-tools layout).
#     """

#     def __init__(self, *, seed_bits: list[int], p: int, taps: list[int], state_size: int,
#                  sampling: str, warmup: int = None, shrink: bool = False):
#         super().__init__(p)
#         if len(seed_bits) != state_size:
#             raise ValueError(f"seed_bits length {len(seed_bits)} does not match state_size {state_size}")
#         self.n = p.bit_length()
#         self.taps = taps
#         self.state_size = state_size
#         self._shrink = shrink
#         self._state = list(seed_bits)
#         for _ in range(warmup if warmup is not None else 2 * state_size):   # warm up
#             self._next_bit()
#         self.set_sampling(sampling)        # sets _cand_bits (+ reduce_mod)

#     def _configure_sampling(self) -> None:
#         # candidate width per sampling strategy (the bit analogue of the XOF byte widths)
#         self._cand_bits = (32 if self.n <= 32 else 64) if self.sampling == "naive" else self.n

#     @staticmethod
#     def to_bits(value: int, width: int) -> list[int]:
#         """Big-endian (MSB-first) bit decomposition of `value` into `width` bits.
#         Helper for callers assembling the seed."""
#         return [(value >> (width - 1 - i)) & 1 for i in range(width)]

#     def _next_bit(self) -> int:
#         s = self._state
#         new = 0
#         for i in self.taps:
#             new ^= s[i]
#         self._state = s[1:] + [new]
#         return new

#     def _next_output_bit(self) -> int:
#         """One output bit: self-shrinking (read pairs (b0, b1), keep b1 iff b0 == 1) when
#         shrink is set, otherwise the raw clocked bit."""
#         if not self._shrink:
#             return self._next_bit()
#         while True:
#             b0 = self._next_bit()
#             b1 = self._next_bit()
#             if b0 == 1:
#                 return b1

#     def _draw_candidate(self) -> int:
#         val = 0
#         for _ in range(self._cand_bits):
#             val = (val << 1) | self._next_output_bit()
#         return val

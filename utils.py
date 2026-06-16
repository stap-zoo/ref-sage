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

    def __init__(self, *, seed: bytes, p: int, sampling: str, xof: str = "shake_128", n_bytes: int = None):
        super().__init__(p)
        if xof not in XOFS:
            raise ValueError(f"Unknown XOF: {xof}. Use one of {sorted(XOFS)}.")
        self._xof = XOFS[xof](seed)
        self._n_bytes_override = n_bytes
        self.set_sampling(sampling)        # sets n_bytes / mask (+ reduce_mod)

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
        return int.from_bytes(raw, "little")


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

# ---------------------------------------------------------------------------
# Field conversion
# ---------------------------------------------------------------------------

def map_to_field(x, to_field):
    """Recursively map `to_field` over a nested list of ints / field elements, preserving
    structure: lists are descended into, any non-list element is converted via `to_field`.
    Covers the vector, matrix, and deeper (e.g. list-of-[a,b]-pairs) cases uniformly."""
    if isinstance(x, list):
        return [map_to_field(e, to_field) for e in x]
    return to_field(x)

# ---------------------------------------------------------------------------
# Matrix utils
# ---------------------------------------------------------------------------

def circulant(row: list = None, *, col: list = None) -> list[list]:
    """Build a (right-)circulant matrix: each row is a cyclic right-shift of the previous one.

    Specify exactly one of:
      row -- the first row    [r0, r1, ..., r_{n-1}]
      col -- the first column [c0, c1, ..., c_{n-1}] (keyword-only)

    Row/column conversion: for a right-circulant with first row r, the first
    column is [r0, r_{n-1}, r_{n-2}, ..., r1] -- the transformation is its own
    inverse, so col input is converted by the same reversal-after-first-element.
    """
    if (row is None) == (col is None):
        raise ValueError("specify exactly one of 'row' or 'col'")
    if col is not None:
        row = [col[0]] + col[:0:-1]   # keep first element, reverse the rest
    n = len(row)
    return [row[(n - i) % n:] + row[:(n - i) % n] for i in range(n)]

def invert_matrix(M: list[list], field=None) -> list[list]:
    """Invert a square matrix given as a list-of-rows and return the inverse in the same form.
    Entries may be ints (then `field` must be given, e.g. GF(p)) or Sage field elements
    (then `field` is inferred)."""
    from sage.all import matrix
    m = matrix(field, M) if field is not None else matrix(M)
    return [list(row) for row in m.inverse()]

def matvecmul(matrix: list[list], vec: list) -> list:
    """Matrix-vector product over any ring. Returns a new list."""
    return [sum(m * v for m, v in zip(row, vec)) for row in matrix]

def vecadd(vec1: list, vec2: list) -> list:
    """Vector-vector addition over any ring. Returns a new list."""
    return [v1 + v2 for v1, v2 in zip(vec1, vec2)]

def vecsub(vec1: list, vec2: list) -> list:
    """Vector-vector addition over any ring. Returns a new list."""
    return [v1 - v2 for v1, v2 in zip(vec1, vec2)]

def add_to_start(state: list, block: list) -> list:
    """New state with block added element-wise into the first len(block) positions."""
    assert len(state) >= len(block)
    return [s + b for s, b in zip(state, block)] + state[len(block):]

def replace_start(state: list, block: list) -> list:
    """New state with the first len(block) positions overwritten by block."""
    assert len(state) >= len(block)
    return list(block) + state[len(block):]

def is_mds(M: list[list], field=None) -> bool:
    """A matrix is MDS iff all its minors (of every order) are non-zero.
    Accepts a list-of-lists; entries may be ints (then `field` must be given,
    e.g. GF(p)) or Sage field elements (then `field` is inferred)."""
    from sage.all import matrix
    m = matrix(field, M) if field is not None else matrix(M)
    return all(minor != 0 for k in range(1, m.nrows() + 1) for minor in m.minors(k))

# ---------------------------------------------------------------------------
# Mixed-radix utils
# ---------------------------------------------------------------------------

def mixed_radix_decompose(val, si: list[int], from_field) -> list[int]:
    n = from_field(val)
    res = [0] * len(si)
    for i in range(len(si) - 1, 0, -1):
        n, res[i] = divmod(n, si[i])
    res[0] = n
    return res

def mixed_radix_compose(digits: list[int], si: list[int], to_field):
    result = digits[0]
    for digit, s in zip(digits[1:], si[1:]):
        result = result * s + digit
    return to_field(result)

# ---------------------------------------------------------------------------
# Lookup table utils
# ---------------------------------------------------------------------------

def invert_LUT(LUT: list[int]) -> list[int]:
    inv = [0] * len(LUT)
    for i, v in enumerate(LUT):
        inv[v] = i
    return inv

# ---------------------------------------------------------------------------
# MDS Matrix generation methods
# ---------------------------------------------------------------------------

# --- Pseudo-Hadamard transform ---------------------------------------------
def pht_matrix(alpha) -> list[list]:
    """Generalized pseudo-Hadamard transform [[1, a], [a, 1 + a^2]]:
    2 additions, 2 multiplications; the classic PHT [[1,1],[1,2]] at alpha = 1.
    Anemoi's M_2 with alpha = g (Figure 7a in https://eprint.iacr.org/2022/840.pdf).
    det = 1 identically, so it is MDS iff all entries are nonzero: alpha != 0 and alpha^2 != -1."""
    one = alpha**0
    return [[one,   alpha              ],
            [alpha, one + alpha * alpha]]

def pht_apply(x: list, alpha) -> list:
    """Evaluate pht_matrix(alpha) @ x: 2 additions, 2 multiplications."""
    x0 = x[0] + alpha * x[1]
    x1 = x[1] + alpha * x0
    return [x0, x1]

# --- Duval-Leurent low-addition MDS matrices -------------------------------
# DL18: https://tosc.iacr.org/index.php/ToSC/article/view/888
# Catalog labels read M^{adds,muls}_{dim,depth}; function names follow m{dim}{depth}_{adds}{muls}.
# All are generically MDS but can degenerate for specific (field, alpha):
# Verify concrete instances with is_mds.

def dl_m33_52_matrix(alpha) -> list[list]:
    """DL18 M^{5,2}_{3,3} (Fig. 6): 3x3, depth 3, 5 additions, 2 multiplications.
    Anemoi's M_3 with alpha = g^i, smallest i passing is_mds."""
    one = alpha**0
    return [[one + alpha, one, one + alpha],
            [one,         one, alpha      ],
            [alpha,       one, one        ]]

def dl_m33_52_apply(x: list, alpha) -> list:
    """Evaluate dl_m33_52_matrix(alpha) @ x: 5 additions, 2 multiplications."""
    x0, x1, x2 = x
    t = x0 + alpha * x2
    z2 = (x2 + x1) + alpha * x0
    return [t + z2, x1 + t, z2]

def dl_m46_83_matrix(alpha) -> list[list]:
    """DL18 M^{8,3}_{4,6} (Fig. 8): 4x4, depth 6, 8 additions, 3 multiplications.
    Anemoi's M_4 with alpha = g^i, smallest i passing is_mds"""
    one = alpha**0
    a, a2 = alpha, alpha * alpha
    return [[one,     one + a,     a,       a          ],
            [a2,      a2 + a,      one + a, one + a + a],
            [a2,      a2,          one,     one + a    ],
            [one + a, one + a + a, a,       one + a    ]]

def dl_m46_83_apply(x: list, alpha) -> list:
    """Evaluate dl_m46_83_matrix(alpha) @ x: 8 additions, 3 multiplications.
    Anemoi's M_4 with alpha = g^i, smallest i passing is_mds."""
    x0, x1, x2, x3 = x
    x0 = x0 + x1
    x2 = x2 + x3
    x3 = x3 + alpha * x0
    x1 = alpha * (x1 + x2)
    x0 = x0 + x1
    x2 = x2 + alpha * x3
    x1 = x1 + x2
    x3 = x3 + x0
    return [x0, x1, x2, x3]

def dl_m44_84_matrix(alpha) -> list[list]:
    """DL18 M^{8,4}_{4,4}: 4x4, depth 4, 8 additions, 4 multiplications.
    Griffin's/Poseidon's M_4 with alpha = 2 (MDS for all primes p > 2^31 at alpha = 2)."""
    one = alpha**0
    a, a2 = alpha, alpha * alpha
    return [[a2 + one, a2 + a + one, one,      a + one     ],
            [a2,       a2 + a,       one,      one         ],
            [one,      a + one,      a2 + one, a2 + a + one],
            [one,      one,          a2,       a2 + a      ]]

def dl_m44_84_apply(x: list, alpha) -> list:
    """Evaluate dl_m44_84_matrix(alpha) @ x: 8 additions, 4 multiplications
    by alpha / alpha^2 (shifts when alpha = 2)."""
    x0, x1, x2, x3 = x
    a, a2 = alpha, alpha * alpha
    t0 = x0 + x1
    t1 = x2 + x3
    t2 = a * x1 + t1
    t3 = a * x3 + t0
    t4 = a2 * t1 + t3
    t5 = a2 * t0 + t2
    return [t3 + t5, t5, t2 + t4, t4]

# --- Matrix generation strategies ------------------------------------------
def m4_to_block_circulant_matrix(t: int, M4: list[list[int]] = None) -> list[list[int]]:
    """Construct the txt matrix M = circ(2,1,...,1) (x) M4 for t = 4*t_, i.e. a block-circulant matrix 
    with 4x4 blocks: block (i,j) is 2*M4 on the diagonal and M4 off-diagonal. If M4 is not given, use the 
    Duval-Leurent matrix M_{4,4}^{8,4} (alpha=2), which is MDS for all p > 2^31.

    Note: the result is intentionally NOT MDS as a whole, but provides full diffusion in one round at ~O(t) cost.

    This method is used by Griffin and Poseidon2 (external matrix).
    """

    if t % 4 != 0:
        raise ValueError("t must be a multiple of 4")

    if M4 is None:
        # [[5, 7, 1, 3],[4, 6, 1, 1],[1, 3, 5, 7],[1, 1, 4, 6]]
        # See Figure 13 in https://tosc.iacr.org/index.php/ToSC/article/view/888/839
        M4 = dl_m44_84_matrix(alpha=2)
    
    if t == 4:
        return M4

    M = [[0] * t for _ in range(t)]
    for row in range(t):
        for col in range(t):
            val = M4[row % 4][col % 4]
            if row // 4 == col // 4:
                val *= 2
            M[row][col] = val
    
    return M

def vandermonde_mds_matrix(p: int, t: int, generator: int, transpose: bool = False) -> list[list[int]]:
    """Build a txt MDS matrix from the right half of the echelon form of the tx2t Vandermonde matrix 
    V[i][j] = generator^(i*j), 0 <= i < t, 0 <= j < 2t. `generator` must be a primitive element of GF(p).

    Note: dense MDS matrix with unstructured full-size entries

    This method is used by Rescue (transpose=False) and Rescue Prime / RPO (transpose=True)
    """
    from sage.all import GF, matrix

    F = GF(p)
    g = F(generator)
    V = matrix(F, [[g**(i * j) for j in range(2 * t)] for i in range(t)])
    M = V.echelon_form()[:, t:]
    if transpose:
        M = M.transpose()
    return [[int(x) for x in row] for row in M]


def simple_circulant_matrix(t: int) -> list[list[int]]:
    """MDS guaranteed only for t in {2, 3, 4} with p > 130.
    See https://arxiv.org/pdf/2303.04639 (Arion paper), Remark 4."""
    return circulant(row=list(range(1, t + 1)))

def ones_plus_diag_matrix(diag: list[int]) -> list[list[int]]:
    """The t x t matrix J + diag(diag): all-ones with diag[i] added on the diagonal.
    Poseidon2's internal matrix M_I = J + diag(MAT_DIAG_M_1)."""
    t = len(diag)
    M = [[1] * t for _ in range(t)]
    for i in range(t):
        M[i][i] = M[i][i] + diag[i]
    return M

def cauchy_matrix(xs: list[int], ys: list[int], p: int) -> list[list[int]]:
    """Cauchy matrix M[i][j] = 1 / (xs[i] + ys[j]) over GF(p). Raises ZeroDivisionError if any
    xs[i] + ys[j] is congruent to 0 mod p (the modular inverse is undefined, i.e. these xs, ys
    do not form a valid Cauchy matrix)."""
    M = []
    for x in xs:
        row = []
        for y in ys:
            denom = (x + y) % p
            if denom == 0:
                raise ZeroDivisionError("Cauchy denominator xs[i] + ys[j] = 0 mod p")
            row.append(pow(denom, -1, p))
        M.append(row)
    return M

def cauchy_mds_matrix(p: int, t: int, *, xs: list[int] = None, ys: list[int] = None,
                      sampler: "FieldElementSampler" = None) -> list[list[int]]:
    """t x t Cauchy MDS matrix M[i][j] = 1 / (xs[i] + ys[j]) over GF(p). The xs, ys come from
    one of three sources:

      - `sampler` given: draw the 2t elements as a 2-row grid (xs, ys = sampler.grid(2, t)),
        resampling the whole batch until all 2t values are distinct and the Cauchy matrix is
        defined. This is the reference generate_parameters_grain.sage create_mds_p construction;
        the sampler's current strategy applies (Poseidon switches it to "mod" beforehand), and any
        FieldElementSampler works.
      - `xs` and `ys` given: used directly.
      - neither: the fixed indices xs = 0..t-1, ys = -t..-(2t-1), i.e. M[i][j] = 1 / (i - t - j),
        matching github.com/khovratovich/poseidon-tools (the "ethereum" Poseidon strategy; needs p > 2t).
    """
    if sampler is not None:
        while True:
            xs, ys = sampler.grid(2, t)
            if len(set(xs + ys)) != 2 * t:           # values must be distinct
                continue
            try:
                return cauchy_matrix(xs, ys, p)
            except ZeroDivisionError:                # some xs[i] + ys[j] = 0; resample the batch
                continue
    if xs is None or ys is None:
        xs, ys = list(range(t)), [-t - j for j in range(t)]
    return cauchy_matrix(xs, ys, p)

def tip5_mds_matrix(p: int, t: int = 16) -> list[list[int]]:
    """Tip5's 16x16 circulant MDS matrix. The first column is the 16 little-endian 16-bit words of
    SHA-256("Tip5") -- the 32-byte digest split into 16 two-byte words (both fixed by the spec),
    drawn as a grid (n_bytes=2) from the XOF sampler. Entries are 16-bit by design (deliberately
    narrower than the field), enabling delayed modular reduction. `p` is only the sampler's field;
    since every entry is < 2^16 < p no rejection occurs, so the matrix is identical over Goldilocks
    2^64 - 2^32 + 1 and Mersenne 2^31 - 1 (reused verbatim by Tip4 with t = 16 and Monolith-31 with
    t = 16). For reduced state sizes t < 16 (e.g. Tip4' with t = 12) the circulant is built from the
    first t entries of the column."""
    if not 1 <= t <= 16:
        raise ValueError(f"t must be in 1..16. Got {t}")
    column = XOFFieldElementSampler(seed=b"Tip5", p=p, xof="sha256", sampling="naive", n_bytes=2).grid(1, 16)[0]
    return circulant(col=column[:t])

def rpo_mds_matrix(t: int) -> list[list[int]]:
    if t == 12:
        return circulant(row=[7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8])
    elif t == 16:
        return circulant(row=[256, 2, 1073741824, 2048, 16777216, 128, 8, 16, 524288, 4194304, 1, 268435456, 1, 1024, 2, 8192])
    else:
        raise ValueError(f"t must be in {{12,16}}. Got {t}")
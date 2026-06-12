from hashlib import shake_128, shake_256, sha256
from blake3 import blake3
from math import ceil

# ---------------------------------------------------------------------------
# Sample field elements
# ---------------------------------------------------------------------------

# Extendable-output functions usable as byte streams: each entry maps a name to a
# constructor taking the seed and returning an object whose digest(n) yields the
# first n bytes of the stream.

XOFS = {
    "shake_128": shake_128,
    "shake_256": shake_256,
    "blake3": blake3,
}

class FieldElementSampler:
    """Stateful XOF reader, sampling field elements in [0, p) one at a time.

    Cuts the byte stream of the seeded XOF into fixed-size little-endian chunks and maps each chunk to a field element according to the sampling strategy:

    sampling="bitmask" : reads ceil(p.bit_length()/8) bytes, zeroes bits above p.bit_length() in the last byte, rejects (resamples) if >= p.
                         Matches the Rust field_element_from_shake / ff::PrimeField::from_repr from
                         https://extgit.isec.tugraz.at/krypto/zkfriendlyhashzoo/-/blob/master/plain_impls/src/fields/utils.rs?ref_type=heads.
                         Used in Reinforced Concrete and Griffin (with xof="shake_128").
    sampling="naive"   : reads a fixed-width word (4 bytes if p fits in 32 bits, 8 bytes otherwise), rejects if >= p.
                         Used in Monolith (with xof="shake_128").
    sampling="mod"     : reads ceil(p.bit_length()/8)+1 bytes, reduces mod p (no rejection).
                         Matches Rescue Prime / RPO's get_round_constants from 
                         https://github.com/KULeuven-COSIC/Marvellous and https://github.com/ASDiscreteMathematics/rpo.
                         Used in Rescue, Rescue Prime / RPO, and Arion (with xof="shake_256").

    n_bytes overrides the strategy's chunk size, e.g. for Tip5's round constants (xof="blake3", sampling="mod"), 
    which read t bytes per element instead of ceil(p.bit_length()/8)+1.
    """

    def __init__(self, seed: bytes, p: int, xof: str = "shake_128", sampling: str = "bitmask", n_bytes: int = None):
        if xof not in XOFS:
            raise ValueError(f"Unknown XOF: {xof}. Use one of {sorted(XOFS)}.")
        self.p = p
        self._xof = XOFS[xof](seed)

        bits = p.bit_length()
        if sampling == "bitmask":
            self.n_bytes = ceil(bits / 8)
            mod = bits % 8
            self.mask = ((1 << mod) - 1) if mod != 0 else 0xFF
            self.reduce_mod = False
        elif sampling == "naive":
            self.n_bytes = 4 if bits <= 32 else 8
            self.mask = 0xFF
            self.reduce_mod = False
        elif sampling == "mod":
            self.n_bytes = ceil(bits / 8) + 1
            self.mask = 0xFF
            self.reduce_mod = True
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling}. Use 'bitmask', 'naive', or 'mod'.")

        if n_bytes is not None:
            self.n_bytes = n_bytes

        self._pos = 0
        self._size = max(1024, self.n_bytes * 64)
        self._buf = self._xof.digest(self._size)

    def _ensure(self, n: int) -> None:
        """Grow the buffered XOF output until at least n unread bytes are available.
        An XOF's longer digest is an extension of its shorter one, so re-requesting
        the doubled size extends the stream while keeping every previously read
        position valid."""
        while len(self._buf) - self._pos < n:
            self._size *= 2
            self._buf = self._xof.digest(self._size)

    def next(self) -> int:
        """Sample the next field element from the stream."""
        while True:
            self._ensure(self.n_bytes)
            raw = bytearray(self._buf[self._pos:self._pos + self.n_bytes])
            self._pos += self.n_bytes
            raw[-1] &= self.mask
            val = int.from_bytes(raw, "little")
            if self.reduce_mod:
                return val % self.p
            if val < self.p: # rejection sampling
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

def is_mds(M: list[list], field=None) -> bool:
    """A matrix is MDS iff all its minors (of every order) are non-zero.
    Accepts a list-of-lists; entries may be ints (then `field` must be given,
    e.g. GF(p)) or Sage field elements (then `field` is inferred)."""
    from sage.all import matrix
    m = matrix(field, M) if field is not None else matrix(M)
    return all(minor != 0 for k in range(1, m.nrows() + 1) for minor in m.minors(k))

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

def tip5_mds_matrix(t: int = 16) -> list[list[int]]:
    """Tip5's 16x16 circulant MDS matrix. First column corresponds to 16-bit little-endian
    chunks of SHA-256("Tip5"). Entries are 16-bit by construction, enabling delayed modular reduction.
    Reused verbatim by Tip4 (t = 16) and Monolith-31 (t = 16);
    MDS over both Goldilocks 2^64 - 2^32 + 1 and Mersenne 2^31 - 1.
    For reduced state sizes t < 16 (e.g. Tip4' with t = 12), the circulant is built
    from the first t entries of the column, as in the sage reference implementation."""
    if not 1 <= t <= 16:
        raise ValueError(f"t must be in 1..16. Got {t}")
    digest = sha256(b"Tip5").digest()
    first_column = [int.from_bytes(digest[2 * i: 2 * i + 2], "little") for i in range(16)]
    return circulant(col=first_column[:t])
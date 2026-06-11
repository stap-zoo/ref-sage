from hashlib import shake_128, shake_256
from math import ceil

# ---------------------------------------------------------------------------
# Sample randomness
# ---------------------------------------------------------------------------

def sample_from_shake_128(seed: bytes, p: int, num_rows: int, num_cols: int, sampling: str) -> list[list[int]]:
    """Sample a num_rows x num_cols grid of field elements in [0, p) via SHAKE128.

    sampling="bitmask" : reads ceil(p.bit_length()/8) bytes, zeroes bits above p.bit_length() in the last byte, rejects if >= p.
                         Matches the Rust field_element_from_shake / ff::PrimeField::from_repr from https://extgit.isec.tugraz.at/krypto/zkfriendlyhashzoo/-/blob/master/plain_impls/src/fields/utils.rs?ref_type=heads.
    sampling="naive"   : reads a fixed-width word (4 bytes if p fits in 32 bits, 8 bytes otherwise) as little-endian, rejects if >= p.
    
    This sampling method is used in Reinforced Concrete (sampling="bitmask") and Monolith (sampling="naive").
    """
    buf = shake_128(seed).digest(10_000)
    pos = 0

    if sampling == "bitmask":
        bits    = p.bit_length()
        n_bytes = ceil(bits / 8)
        mod     = bits % 8
        mask    = ((1 << mod) - 1) if mod != 0 else 0xFF

        def _read() -> int:
            nonlocal pos
            while True:
                raw      = bytearray(buf[pos : pos + n_bytes])
                pos     += n_bytes
                raw[-1] &= mask
                val      = int.from_bytes(raw, "little")
                if val < p:
                    return val

    elif sampling == "naive":
        word = 4 if p.bit_length() <= 32 else 8

        def _read() -> int:
            nonlocal pos
            while True:
                val  = int.from_bytes(buf[pos : pos + word], "little")
                pos += word
                if val < p:
                    return val

    else:
        raise ValueError(f"Unknown sampling strategy: {sampling!r}. Use 'bitmask' or 'naive'.")

    return [[_read() for _ in range(num_cols)] for _ in range(num_rows)]

def sample_from_shake_256(seed: bytes, p: int, num_rows: int, num_cols: int, sampling: str) -> list[list[int]]:
    """Sample a num_rows x num_cols grid of field elements in [0, p) via SHAKE256.

    sampling="mod" : reads ceil(p.bit_length()/8)+1 bytes per element as little-endian,
                      reduces mod p (no rejection). Matches Rescue Prime / RPO's
                      get_round_constants.

    This sampling method is used in Rescue Prime / RPO.
    """
    if sampling == "mod":
        bytes_per_int = (p.bit_length() + 7) // 8 + 1

        def _read(buf: bytes, i: int) -> int:
            chunk = buf[bytes_per_int * i: bytes_per_int * (i + 1)]
            return int.from_bytes(chunk, "little") % p

    else:
        raise ValueError(f"Unknown sampling strategy: {sampling!r}. Use 'mod'.")

    buf = shake_256(seed).digest(bytes_per_int * num_rows * num_cols)
    flat = [_read(buf, i) for i in range(num_rows * num_cols)]
    return [flat[i * num_cols:(i + 1) * num_cols] for i in range(num_rows)]

# ---------------------------------------------------------------------------
# Stateful SHAKE XOF reader
# ---------------------------------------------------------------------------

class ShakeReader:
    """Stateful SHAKE128 XOF reader, sampling field elements one at a time.

    Matches `field_element_from_shake` / `field_element_from_shake_without_0`
    from https://extgit.isec.tugraz.at/krypto/zkfriendlyhashzoo/-/blob/master/plain_impls/src/fields/utils.rs:
    reads ceil(p.bit_length()/8) bytes, masks the unused high bits of the last
    byte, interprets as little-endian, and rejects (resamples) if the result
    is >= p (or == 0 for `nonzero_field_element`).
    """

    def __init__(self, seed: bytes, p: int):
        self.p = p
        self._shake = shake_128(seed)

        bits = p.bit_length()
        self.n_bytes = ceil(bits / 8)
        mod = bits % 8
        self.mask = ((1 << mod) - 1) if mod != 0 else 0xFF

        self._buf = b""
        self._pos = 0
        self._size = max(1024, self.n_bytes * 64)
        self._buf = self._shake.digest(self._size)

    def _ensure(self, n: int) -> None:
        while len(self._buf) - self._pos < n:
            self._size *= 2
            self._buf = self._shake.digest(self._size)

    def field_element(self) -> int:
        while True:
            self._ensure(self.n_bytes)
            raw = bytearray(self._buf[self._pos:self._pos + self.n_bytes])
            self._pos += self.n_bytes
            raw[-1] &= self.mask
            val = int.from_bytes(raw, "little")
            if val < self.p:
                return val

    def nonzero_field_element(self) -> int:
        while True:
            val = self.field_element()
            if val != 0:
                return val

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
        M4 = [[5, 7, 1, 3],[4, 6, 1, 1],[1, 3, 5, 7],[1, 1, 4, 6]]
    
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
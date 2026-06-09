from hashlib import shake_128
from math import ceil

# ---------------------------------------------------------------------------
# Round constant generation
# ---------------------------------------------------------------------------

def sample_round_constants_from_shake_128(
    seed:     bytes,
    p:        int,
    num_rows: int,
    num_cols: int,
    sampling: str,
) -> list[list[int]]:
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


# ---------------------------------------------------------------------------
# Matrix utils
# ---------------------------------------------------------------------------
def circulant(row: list) -> list[list]:
    """Build a circulant matrix from its first row (each subsequent row is a cyclic right-shift)."""
    n = len(row)
    return [row[(n - i) % n:] + row[:(n - i) % n] for i in range(n)]


def matvecmul(matrix: list[list], vec: list) -> list:
    """Matrix-vector product over any ring. Returns a new list."""
    return [sum(m * v for m, v in zip(row, vec)) for row in matrix]

def add_in_place(dst: list, src: list) -> None:
    assert len(dst) >= len(src)
    for i, x in enumerate(src):
        dst[i] += x


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
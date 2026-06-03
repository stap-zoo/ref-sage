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

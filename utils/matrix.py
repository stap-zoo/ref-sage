"""Matrix and vector utilities.

Basic list-of-rows matrix/vector arithmetic, a generic recursive mapper over nested
lists (`map_nested`), and constructors for the linear layers used by the permutations
in this repo (circulant / PHT / diffusion-layer / MDS matrices), plus an MDS check.
"""

from utils.sampler import XOFFieldElementSampler
from sage.all import GF, matrix

# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------

def map_nested(x, fun):
    """Recursively apply `fun` to every leaf of a nested list, preserving structure:
    lists are descended into, any non-list element is passed through `fun`. Covers the
    vector, matrix, and deeper (e.g. list-of-[a,b]-pairs) cases uniformly. Commonly used
    to coerce nested int data into field elements, but `fun` can be any leaf-wise map."""
    if isinstance(x, list):
        return [map_nested(e, fun) for e in x]
    return fun(x)

def invert_matrix(M: list[list], field=None) -> list[list]:
    """Invert a square matrix given as a list-of-rows and return the inverse in the same form.
    Entries may be ints (then `field` must be given, e.g. GF(p)) or Sage field elements
    (then `field` is inferred)."""
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

def add_at(state: list, block: list, off: int = 0) -> list:
    """New state with block added element-wise into positions [off, off+len(block))."""
    assert off + len(block) <= len(state)
    return (state[:off] + [state[off + i] + b for i, b in enumerate(block)] + state[off + len(block):])

def replace_at(state: list, block: list, off: int = 0) -> list:
    """New state with positions [off, off+len(block)) overwritten by block."""
    assert off + len(block) <= len(state)
    return state[:off] + list(block) + state[off + len(block):]

# ---------------------------------------------------------------------------
# Generation
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
    Griffin's/Poseidon's/Polocolo's M_4 with alpha = 2 (MDS for all primes p > 2^31 at alpha = 2)."""
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

    This method is used by Rescue (transpose=False), Rescue Prime / RPO (transpose=True), and Grendel (transpose=True).
    """
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

# Tip5's MDS column derivation and RPO's pinned circulant rows moved to their
# owning primitives (tip5/params.py TIP5_MDS_COLUMN, marvellous/params.py
# RPO_MDS_ROWS): pinned design data belongs to the primitive, only generic
# constructions stay here.

def low_addition_mds_matrix(t: int, additions: int, coeff_max: int = 8, seed=None, max_trials: int = 500, attempts: int = 100) -> list[list[int]]:
    """Search for a t x t integer MDS matrix computable with at most `additions`
    two-term addition gates (Polocolo, https://eprint.iacr.org/2025/926, Algorithm 1 /
    MDS_generator.sage). Every vector built below is a linear combination of the input
    coordinates and costs exactly one gate `w = c1*u + c2*v` with coefficients in
    [1, coeff_max]; the matrix rows are the last t vectors accepted, so the whole
    product M*x evaluates with `additions` addition gates and only small-constant
    multiplications.

    Three stages, as in the reference:
      1. start from the t unit vectors (free) and ceil(t/2) pairwise sums (one gate each),
      2. expand by `additions - t - ceil(t/2)` random combinations of two previous
         vectors (not both unit vectors), kept only if linearly independent so far,
      3. draw the final t rows one at a time, kept only if they have no zero entry and
         the partial matrix stays MDS-extendable (all minors of every order non-zero,
         checked over the integers); at most max_trials draws, then restart (up to
         `attempts` times).

    The search is randomized; pass `seed` to make it reproducible. Polocolo's published
    t = 5..8 matrices come from unseeded runs of this search, so they cannot be
    regenerated -- they are pinned as constants in polocolo/params.py, and this function
    is the strategy for producing matrices at other state sizes. The integer check is
    valid over F_p as long as no minor is divisible by p (guaranteed for large p with
    these small entries); re-verify concrete instances with is_mds over their field.

    Success probability drops sharply near the minimal budget of t + ceil(t/2) + t
    gates: a couple of gates of slack finds a matrix in well under a second, while the
    paper's tightest published budgets (e.g. t = 6 with 17 additions) need tens of
    thousands of `attempts` -- the authors ran open-ended searches."""
    import random
    from sage.all import QQ

    rng = random.Random(seed)
    n_init_gates = (t + 1) // 2
    n_expand = additions - t - n_init_gates
    if n_expand < t:
        # the final rows are combinations of expansion vectors only, so the expansion
        # stage must span the full t-dimensional space: at least t gates on top of the
        # ceil(t/2) initial and t final ones (t = 5 with 13 additions is the paper's minimum)
        raise ValueError(f"additions={additions} too small: need >= {n_init_gates + 2 * t} for t={t}")

    def combine(u, v, c1, c2):
        return [c1 * a + c2 * b for a, b in zip(u, v)]

    def rand_coeff():
        return rng.randrange(1, coeff_max + 1)

    def int_det(m):
        """Fraction-free (Bareiss) determinant of a small integer matrix. Pure Python:
        the search evaluates millions of tiny minors, where the per-call overhead of
        Sage matrices dominates by orders of magnitude."""
        m = [row[:] for row in m]
        n, sign, prev = len(m), 1, 1
        for i in range(n - 1):
            if m[i][i] == 0:
                for r in range(i + 1, n):
                    if m[r][i] != 0:
                        m[i], m[r] = m[r], m[i]
                        sign = -sign
                        break
                else:
                    return 0
            for r in range(i + 1, n):
                for c in range(i + 1, n):
                    m[r][c] = (m[r][c] * m[i][i] - m[r][i] * m[i][c]) // prev
            prev = m[i][i]
        return sign * m[-1][-1]

    def minors_with_new_row_nonzero(rows):
        """Rows before the last are assumed already validated, so only the minors
        (of every order) that involve the newly added last row are checked."""
        from itertools import combinations
        last = len(rows) - 1
        for k in range(1, len(rows) + 1):
            for ri in combinations(range(last), k - 1):
                for ci in combinations(range(t), k):
                    sub = [[rows[i][j] for j in ci] for i in (*ri, last)]
                    if int_det(sub) == 0:
                        return False
        return True

    for _ in range(attempts):
        # stage 1: unit vectors + ceil(t/2) pairwise sums (pairs (0,1), (2,3), ...; odd t closes with (0, t-1))
        pool = [[int(i == j) for j in range(t)] for i in range(t)]
        pairs = [(2 * i, 2 * i + 1) for i in range(t // 2)] + ([(0, t - 1)] if t % 2 == 1 else [])
        for i, j in pairs:
            pool.append(combine(pool[i], pool[j], rand_coeff(), rand_coeff()))

        # stage 2: expansion vectors, linearly independent among themselves
        expansion = []
        while len(expansion) < n_expand:
            idx1, idx2 = rng.randrange(len(pool)), rng.randrange(len(pool))
            if idx1 == idx2 or (idx1 < t and idx2 < t):  # not the same, not two unit vectors
                continue
            v = combine(pool[idx1], pool[idx2], rand_coeff(), rand_coeff())
            # as in the reference: keep the expansion set at full rank (min(rows, t))
            if matrix(QQ, expansion + [v]).rank() != min(len(expansion) + 1, t):
                continue
            expansion.append(v)
            pool.append(v)

        # stage 3: final rows; drawn from the expansion vectors and the rows accepted so far
        seed_pool = list(expansion)
        rows = []
        for _ in range(max_trials):
            if len(rows) == t:
                break
            idx1, idx2 = rng.randrange(len(seed_pool)), rng.randrange(len(seed_pool))
            if idx1 == idx2:
                continue
            v = combine(seed_pool[idx1], seed_pool[idx2], rand_coeff(), rand_coeff())
            if 0 in v or not minors_with_new_row_nonzero(rows + [v]):
                continue
            rows.append(v)
            seed_pool.append(v)

        if len(rows) == t:
            return rows

    raise RuntimeError(f"no {t}x{t} MDS matrix with {additions} additions found "
                       f"within {attempts} attempts of {max_trials} trials each")

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

def is_mds(M: list[list], field=None) -> bool:
    """A matrix is MDS iff all its minors (of every order) are non-zero.
    Accepts a list-of-lists; entries may be ints (then `field` must be given,
    e.g. GF(p)) or Sage field elements (then `field` is inferred)."""
    m = matrix(field, M) if field is not None else matrix(M)
    return all(minor != 0 for k in range(1, m.nrows() + 1) for minor in m.minors(k))
"""Lookup-table utilities for LUT-based constructions.

Helpers used by primitives whose S-box is realised as a small lookup table over digits
(e.g. Reinforced Concrete, Monolith, Tip5, Skyscraper): mixed-radix decomposition of a
field element into per-digit values and back, inversion of a lookup table, and the
small-n Chi (Bar) S-boxes those constructions share. Also the power-residue tables of
Polocolo, whose lookup is keyed by field-element values rather than dense digit indices.
"""

import random
from utils.sampler import XOFFieldElementSampler

# ---------------------------------------------------------------------------
# Mixed-radix decomposition
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
# Lookup table inversion
# ---------------------------------------------------------------------------

def invert_LUT(LUT: list[int]) -> list[int]:
    inv = [0] * len(LUT)
    for i, v in enumerate(LUT):
        inv[v] = i
    return inv

# ---------------------------------------------------------------------------
# Chi (Bar) S-boxes
#
# Builds the invertible nonlinear shift-invariant map phi from Daemen, "Cipher
# and Hash Function Design" (1995), Ch. 6.6 (Shift-Invariant Transformations /
# Nonlinear Transformations with finite neighborhood) and Table A.1 ("Invertible
# phi specified by a single landscape"). 
# https://cs.ru.nl/~joan/papers/JDA_Thesis_1995.pdf
#
# A complementing landscape (CL) specifies a single local map: at each bit, the
# origin is complemented when its neighborhood matches the pattern. Applying
# that one local map at every position of a cyclic n-bit chunk lifts it to
# a transformation on the whole chunk. Invertibility is n-dependent and read
# off the landscape's xi-set: phi is a permutation iff no element of xi divides
# n.
#
# This is the Chi (Bar) construction used by Skyscraper and Monolith, where the
# chunk map is additionally composed with a bit rotation (itself a shift-
# invariant translation, leaving phi's equivalence class and properties intact).
# Realised as a precomputed lookup table indexed by the chunk value; the inverse
# table is obtained via invert_LUT.
# ---------------------------------------------------------------------------

def invertible_phi_from_landscape(n: int, cl: str, xi: set[int]):
    """
    Build a non-linear shift-invariant transformation phi specified by a single complementing landscape (CL).

    cl: a single CL string as printed in Table A.1, e.g. "*01" (chi), "*10", "0*01", "0*-10". Read left to right:
            '*' -> origin (offset 0); the cell whose bit gets flipped / XORed
            '1' -> invert this bit
            '0' -> do not invert this bit
            '-' -> don't-care
        Symbols right of '*' are offsets +1, +2, ...; left are -1, -2, ...

    xi: the landscape's invertibility set, taken as-is from Table A.1
        (e.g. {2} for chi, {3}, {4}, or set() for the unconditional rows).
        Not computed here. phi is a permutation on n bits iff no element of xi divides n.

    Returns phi: a callable mapping a n-bit integer state to its n-bit image, applying the local map at 
    every position of the cyclic array.

    Scope: single-landscape rules only.
    """

    # Check invertibility rule (assuming xi is correctly given)
    if not all(n % m != 0 for m in xi):
        raise ValueError(f"landscape {cl} is not invertible at n={n} (some element of xi={xi} divides n)")

    # Parse complementing landscape (CL)
    if cl.count('*') != 1:
        raise ValueError("CL string must contain exactly one origin '*'")
    origin = cl.index('*')
    constraints = {}
    for idx, sym in enumerate(cl):
        if sym in ('0', '1'):
            constraints[idx - origin] = int(sym)
        elif sym not in ('*', '-'):
            raise ValueError(f"bad CL symbol {sym}")
    

    items = tuple(constraints.items())   # close over the parsed rule
    mask = (1 << n) - 1

    # result of running the local map at every bit position of the cyclic array
    def phi(x):
        x &= mask
        def get(pos):
            return (x >> (pos % n)) & 1
        y = 0
        for i in range(n):
            flip = 1
            for offset, bitinv in items:
                bit = get(i + offset)
                flip &= (bit ^ 1) if bitinv == 1 else bit
            y |= (get(i) ^ flip) << i
        return y

    return phi


# n-bit S-boxes used in Bar layer of Monolith and Skyscraper, built as composed shift-invariant 
# transformations: a chi-style landscape map (Daemen Table A.1) followed by an outer rotation.
#
# The two boxes use different landscapes so each stays invertible at its n-bit width
# (phi is a permutation iff no element of xi divides n):
#   - n=7: canonical chi "*01",     xi={2} -> invertible since 7 is odd
#   - n=8: 3-term landscape "*001", xi={3} -> invertible since 3 does not divide 8
#
# crotl(n, 1) is a cyclic left-rotation by one (a shift-invariant translation), so
# composing it with chi keeps the map a permutation and stays in chi's class.

def crotl(n: int, b: int):
    """Return the cyclic-left-rotation-by-b transformation on n-bit states.
    This is Daemen's translation tau; itself a shift-invariant transformation."""
    mask = (1 << n) - 1
    def rot(x):
        x &= mask
        return ((x << b) | (x >> (n - b))) & mask
    return rot

def compose(*fns):
    """Compose unary transformations left-to-right: compose(f, g)(x) == g(f(x))."""
    def composed(x):
        for f in fns:
            x = f(x)
        return x
    return composed

# The concrete Monolith Bar tables built from these constructions live at module
# level of monolith/params.py (MONOLITH_LUT8 / MONOLITH_LUT7): pinned design data
# belongs to the owning primitive, only the generic builders stay here.


# ---------------------------------------------------------------------------
# Power-residue S-box tables (Polocolo)
#
# Polocolo's S-box (https://eprint.iacr.org/2025/926, Section 3.2) is
#
#     S(0) = 0,   S(x) = x^{-1} * T[ x^((p-1)/m) ]
#
# for m | p-1 (a power of two in the recommended instances), a generator g of
# F_p^*, and a permutation sigma of {0, ..., m-1}. Writing x = g^(qm+r), the
# m-th power residue x^((p-1)/m) = g^(r(p-1)/m) takes only m+1 distinct values
# (including 0), and the table maps each of them:
#
#     T[0] = 0,   T[ g^(r(p-1)/m) ] = g^((m+1)r + sigma(r)).
#
# Unlike the other LUT-based constructions in this file (Reinforced Concrete,
# Monolith, Tip5), whose tables are indexed by small DENSE integers (per-digit
# values, 7/8-bit chunks) and therefore stored as list[int], Polocolo's T is
# indexed by the power-residue VALUE (m+1 scattered full-size field elements).
# There is no dense index without an extra value->index map (a discrete log
# restricted to m values, i.e. exactly the lookup being avoided), so the tables
# here are dicts {int: int}; the reference implementation uses a
# HashMap<Scalar, Scalar> for the same reason.
# ---------------------------------------------------------------------------

def power_residue_sigma(m: int, p: int, g: int, method: str, seed: str = None) -> list[int]:
    """Derive Polocolo's S-box permutation sigma of {0, ..., m-1} from a seed
    (default "Polocolo-{m}", the seed of the official instances).

    Two derivation variants exist in the Polocolo reference material
    (https://github.com/KAIST-CryptLab/Polocolo), and they DISAGREE:

    method="shuffle" : sigma = Fisher-Yates shuffle of [0, ..., m-1] driven by Python's
                       random.Random(seed) (version-stable string seeding). This is what
                       the lookup tables shipped with the reference implementation
                       (plain/polocolo_luts.rs) were actually generated with: it
                       reproduces all six shipped tables (m = 32 ... 1024) exactly.
    method="hashtape": the derivation published in the authors' param_gen.sage.
                       Rejection-sample distinct values from a SHAKE128 tape seeded with
                       `seed`, and retry the whole permutation until sigma satisfies the
                       two interpolation conditions of the paper (Section 4.2, checked
                       via sigma_conditions_hold; requires `p` and `g`). This does NOT
                       reproduce the shipped tables.

    Every shipped sigma also happens to satisfy the paper's
    interpolation conditions (asserted in the test suite), so the two variants
    differ only in which valid sigma they pick.
    """
    if seed is None:
        seed = f"Polocolo-{m}"

    if method == "shuffle":
        rng = random.Random(seed)
        while True:
            sigma = list(range(m))
            rng.shuffle(sigma)
            if sigma_conditions_hold(p, g, m, sigma):
                return sigma

    if method == "hashtape":
        tape = XOFFieldElementSampler(seed=seed.encode(), p=p, sampling="bitshift", endianess="big", xof="shake_128")
        while True:
            sigma, seen = [], set()
            while len(sigma) < m:
                num = tape.randint(m)
                if num in seen:
                    continue
                seen.add(num)
                sigma.append(num)
            if sigma_conditions_hold(p, g, m, sigma):
                return sigma

    raise ValueError(f"Unknown sigma derivation method: {method}. Use 'shuffle' or 'hashtape'.")


def sigma_conditions_hold(p: int, g: int, m: int, sigma: list[int]) -> bool:
    """The two constraints Polocolo imposes on sigma (Section 4.2): the functions

        f: g^r          -> g^(rm + sigma(r))        (r = 0, ..., m-1)
        h: g^(r(p-1)/m) -> g^(r(m+1) + sigma(r))    (r = 0, ..., m-1)

    must both interpolate to polynomials of maximum degree m-1 with ALL m
    coefficients non-zero (dense), so that the S-box has no low-degree /
    sparse univariate structure an algebraic attack could exploit."""
    from sage.all import GF

    F = GF(p)
    g = F(g)
    R = F["x"]
    k = (p - 1) // m

    f_points = [(g ** r, g ** (r * m + sigma[r])) for r in range(m)]
    f_poly = R.lagrange_polynomial(f_points)
    if f_poly.degree() != m - 1 or 0 in f_poly.coefficients(sparse=False):
        return False

    h_points = [(g ** (r * k), g ** (r * (m + 1) + sigma[r])) for r in range(m)]
    h_poly = R.lagrange_polynomial(h_points)
    if h_poly.degree() != m - 1 or 0 in h_poly.coefficients(sparse=False):
        return False

    return True


def power_residue_lut(p: int, g: int, m: int, sigma: list[int]) -> dict[int, int]:
    """The forward S-box table T with T[0] = 0 and T[g^(r(p-1)/m)] = g^((m+1)r + sigma(r)),
    so that S(x) = x^{-1} * T[x^((p-1)/m)] (and S(0) = 0)."""
    k = (p - 1) // m
    T = {0: 0}
    for r in range(m):
        T[pow(g, r * k, p)] = pow(g, (m + 1) * r + sigma[r], p)
    return T


def power_residue_lut_inv(p: int, g: int, m: int, sigma: list[int]) -> dict[int, int]:
    """The table T_inv of the INVERSE S-box, which has the same shape as S itself:
    S^{-1}(y) = y^{-1} * T_inv[y^((p-1)/m)]. For y = S(g^(qm+r)) = g^(-qm + rm + sigma(r))
    the power residue of y is g^(sigma(r)(p-1)/m), and y^{-1} * g^((m+1)r + sigma(r))
    recovers g^(qm+r) = x, so T_inv[g^(sigma(r)(p-1)/m)] = g^((m+1)r + sigma(r))."""
    k = (p - 1) // m
    T_inv = {0: 0}
    for r in range(m):
        T_inv[pow(g, sigma[r] * k, p)] = pow(g, (m + 1) * r + sigma[r], p)
    return T_inv

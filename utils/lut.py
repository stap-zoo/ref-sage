"""Lookup-table utilities for LUT-based constructions.

Helpers used by primitives whose S-box is realised as a small lookup table over digits
(e.g. Reinforced Concrete, Monolith, Tip5, Skyscraper): mixed-radix decomposition of a
field element into per-digit values and back, inversion of a lookup table, and the
small-n Chi (Bar) S-boxes those constructions share.
"""

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

monolith_phi8 = compose(invertible_phi_from_landscape(8, "001*", xi={3}), crotl(8, 1)) # "*001" in Table A.1
monolith_lut8 = [monolith_phi8(x) for x in range(1 << 8)]

monolith_phi7 = compose(invertible_phi_from_landscape(7, "01*",  xi={2}), crotl(7, 1)) # "*01" in Table A.1
monolith_lut7 = [monolith_phi7(x) for x in range(1 << 7)]

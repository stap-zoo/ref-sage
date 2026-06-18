"""Lookup-table utilities for LUT-based constructions.

Helpers used by primitives whose S-box is realised as a small lookup table over digits
(e.g. Reinforced Concrete, Monolith, Tip5): mixed-radix decomposition of a field element
into per-digit values and back, and inversion of a lookup table.
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

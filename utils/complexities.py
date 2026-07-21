"""Attack-complexity estimators.

log2 cost estimates of algebraic attacks (e.g. Groebner-basis), used when arguing the
security level of a parameter set.
"""

from math import comb, log2


def gb_comp(dreg: int, nv: int, w: int = 2) -> float:
    """log2 cost of a Groebner-basis attack: w * log2(C(dreg + nv, nv)),
    where dreg is the regularity degree and nv the number of variables,
    and w is the linear-algebra exponent of the cost of computing the basis."""
    return w * log2(comb(dreg + nv, nv))


def gb_comp2(dreg: int, nv: int, w: int = 2) -> float:
    """log2 cost of a Groebner-basis attack: w * log2(C(dreg + nv, dreg)),
    where dreg is the regularity degree and nv the number of variables,
    and w is the linear-algebra exponent of the cost of computing the basis."""
    return w * log2(comb(dreg + nv, dreg))



def uni_solve_comp(d: int, p: int) -> float:
    """log2 cost of solving one univariate equation of degree d over F_p:
    d * log2(d) * (log2(d) + log2(p) * log2(log2(d))), the field-operation count
    of Yang et al. (https://eprint.iacr.org/2024/1414) used e.g. in the guessing
    power-residue attack on Polocolo (https://eprint.iacr.org/2025/926, Eq. 3)."""
    ld = log2(d)
    return log2(d * ld * (ld + log2(p) * log2(ld)))

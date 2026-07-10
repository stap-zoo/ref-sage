"""Attack-complexity estimators.

log2 cost estimates of algebraic attacks (e.g. Groebner-basis), used when arguing the
security level of a parameter set.
"""

from math import comb, log


def gb_comp(dreg: int, nv: int, w: int = 2) -> float:
    """log2 cost of a Groebner-basis attack: w * log2(C(dreg + nv, nv)),
    where dreg is the regularity degree and nv the number of variables,
    and w is the linear-algebra exponent of the cost of computing the basis."""
    return w * log(comb(dreg + nv, nv), 2)


def uni_solve_comp(d: int, p: int) -> float:
    """log2 cost of solving one univariate equation of degree d over F_p:
    d * log2(d) * (log2(d) + log2(p) * log2(log2(d))), the field-operation count
    of Yang et al. (https://eprint.iacr.org/2024/1414) used e.g. in the guessing
    power-residue attack on Polocolo (https://eprint.iacr.org/2025/926, Eq. 3)."""
    ld = log(d, 2)
    return log(d * ld * (ld + log(p, 2) * log(ld, 2)), 2)

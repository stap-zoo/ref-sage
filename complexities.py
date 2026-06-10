from math import comb, log


def gb_comp(dreg: int, nv: int, w: int = 2) -> float:
    """log2 cost of a Groebner-basis attack: w * log2(C(dreg + nv, nv)),
    where dreg is the regularity degree and nv the number of variables,
    and w is the linear-algebra exponent of the cost of computing the basis."""
    return w * log(comb(dreg + nv, nv), 2)

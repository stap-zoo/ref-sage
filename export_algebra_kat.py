"""Export known-answer test vectors for the gnark-hashes `algebra` gadgets.

Usage:
    sage -python export_algebra_kat.py            # prints JSON to stdout

"""

import json
import sys
import types

# The reference sampler imports blake3, which is optional and often absent; none
# of the gadgets here touch it, so stub it out before importing utils.matrix.
try:
    import blake3  # noqa: F401
except ImportError:
    stub = types.ModuleType("blake3")
    stub.blake3 = None
    sys.modules["blake3"] = stub

from sage.all import GF

from utils.field import BN254_SCALAR, BLS12_381_SCALAR
from utils.matrix import matvecmul
from hades.hash import Neptune

FIELDS = {
    "bn254": BN254_SCALAR.p,
    "bls12-381": BLS12_381_SCALAR.p,
}

BIG = 0x1cfe73f7a0f9d3b2e5c4a196857d0e2b4f6180c9a3d5f7e02143658b7c9daf01


def hx(el):
    """Field element (or int) -> 0x-hex of its canonical integer representative."""
    return hex(int(el))


def sbox_vectors():
    """x^degree for a spread of degrees (0, 1, and odd non-powers-of-two that
    exercise square-and-multiply) and inputs (including 0 and 1)."""
    out = []
    for fname, p in FIELDS.items():
        F = GF(p)
        for degree in (0, 1, 2, 3, 5, 7, 11):
            for xi in (0, 1, 2, 3, 42, BIG):
                x = F(xi)
                out.append({
                    "field": fname,
                    "x": hx(x),
                    "degree": degree,
                    "y": hx(x ** degree),
                })
    return out


def powermap_vectors():
    """Full power-map layer and the partial (first-u) variant over a small state."""
    out = []
    state_ints = [1, 2, 3, 4, 5]
    u = 2
    for fname, p in FIELDS.items():
        F = GF(p)
        for degree in (3, 5):
            state = [F(v) for v in state_ints]
            full = [v ** degree for v in state]
            partial = [v ** degree for v in state[:u]] + state[u:]
            out.append({
                "field": fname,
                "degree": degree,
                "u": u,
                "state": [hx(v) for v in state],
                "full": [hx(v) for v in full],
                "partial": [hx(v) for v in partial],
            })
    return out


# (rows x cols matrix, vector) test cases. Chosen to exercise the zero-skip and
# unit-coefficient fast paths and a non-square (rows != cols) shape, plus a large
# coefficient for reduction.
LINEAR_CASES = [
    # 3x3 dense, mixes 0 / 1 / other coefficients.
    ([[1, 0, 2], [0, 1, 0], [3, 4, 5]], [1, 2, 3]),
    # 3x2 (more rows than columns): output length follows the matrix, not v.
    ([[2, 3], [0, 1], [7, 0]], [5, 6]),
    # 2x2 with a coefficient reduced from a large integer.
    ([[BIG, 1], [1, BIG]], [BIG, 7]),
]


def linear_vectors():
    out = []
    for fname, p in FIELDS.items():
        F = GF(p)
        for M_ints, v_ints in LINEAR_CASES:
            M = [[F(c) for c in row] for row in M_ints]
            v = [F(x) for x in v_ints]
            res = matvecmul(M, v)
            out.append({
                "field": fname,
                "matrix": [[hx(c) for c in row] for row in M],
                "vec": [hx(x) for x in v],
                "out": [hx(x) for x in res],
            })
    return out


# (a, b) lifting constants and (x, y) inputs for SF. Includes Neptune's own
# a = b = 1 setting and a generic (a, b).
SF_CONSTS = [(1, 1), (3, 2)]
SF_INPUTS = [(1, 2), (0, 0), (7, 13), (BIG, 42)]


def sf_vectors():
    out = []
    for fname, p in FIELDS.items():
        F = GF(p)
        for a, b in SF_CONSTS:
            for xi, yi in SF_INPUTS:
                x, y = F(xi), F(yi)
                u, v = Neptune._S_F(x, y, F(a), F(b))
                out.append({
                    "field": fname,
                    "a": hx(F(a)), "b": hx(F(b)),
                    "x": hx(x), "y": hx(y),
                    "u": hx(u), "v": hx(v),
                })
    return out


# (a, b, gamma, 2x2 matrix) parameter sets for the full Lai-Massey S-box.
LM_PARAMS = [
    (1, 1, 7, [[2, 3], [5, 1]]),
    (3, 2, 11, [[1, 0], [0, 1]]),
]
LM_INPUTS = [(1, 2), (0, 0), (7, 13), (BIG, 42)]


def lai_massey(F, x, y, a, b, gamma, M):
    """Reference Neptune._sbox sequence, using the reference's SF and matvecmul."""
    a, b, gamma = F(a), F(b), F(gamma)
    Mf = [[F(c) for c in row] for row in M]
    x, y = Neptune._S_F(x, y, a, b)
    x, y = matvecmul(Mf, [x, y])
    x, y = x + gamma, y
    x, y = Neptune._S_F(x, y, a, b)
    x, y = x - a * gamma, y
    return x, y


def laimassey_vectors():
    out = []
    for fname, p in FIELDS.items():
        F = GF(p)
        for a, b, gamma, M in LM_PARAMS:
            for xi, yi in LM_INPUTS:
                x, y = F(xi), F(yi)
                u, v = lai_massey(F, x, y, a, b, gamma, M)
                out.append({
                    "field": fname,
                    "a": hx(F(a)), "b": hx(F(b)), "gamma": hx(F(gamma)),
                    "m": [[hx(F(c)) for c in row] for row in M],
                    "x": hx(x), "y": hx(y),
                    "u": hx(u), "v": hx(v),
                })
    return out


def main():
    doc = {
        "_comment": (
            "algebra-gadget known-answer test vectors for gnark-hashes/algebra "
            "(Pow/PowerMap, MatVecMul, SF/LaiMassey). Generated by "
            "ref/export_algebra_kat.py using the reference's own matvecmul and "
            "Neptune._S_F. Inputs may exceed p; the Go side reduces mod p."
        ),
        "fields": {name: hex(p) for name, p in FIELDS.items()},
        "sbox": sbox_vectors(),
        "powermap": powermap_vectors(),
        "linear": linear_vectors(),
        "sf": sf_vectors(),
        "laimassey": laimassey_vectors(),
    }
    print(json.dumps(doc, indent=2))


if __name__ == "__main__":
    main()

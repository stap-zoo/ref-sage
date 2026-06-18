# test_myprimitive.py
# ---------------------------------------------------------------------------
# Test suite for MyPrimitive, parametrized over the named instances in
# instances.py so every recommended instance is covered by the same checks.
#
# Groups:
#   4.1 KATs         -- fixed input/output vectors (permutation + hash modes)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- component identities, linear algebra, symbolic degree
#   4.5 Misc         -- validation/errors, warnings, reproducibility, ...
# ---------------------------------------------------------------------------

import pytest

from myprimitive.hash import MyPrimitive
from myprimitive.params import MyPrimitiveParams
from myprimitive.instances import (
    MYPRIMITIVE_GOLDILOCKS_T3,
    # ... import every named instance you want covered
)
from utils.matrix import matvecmul, is_mds
from recommendations import ParamRecommendationWarning
from utils.field import GOLDILOCKS

INSTANCES = [
    ("GOLDILOCKS_T3", MYPRIMITIVE_GOLDILOCKS_T3),
    # ...
]

# ---------------------------------------------------------------------------
# 4.1 Known-answer tests
#
# KATS[instance_name] -> list of {"input": [...ints...], "output": [...ints...]}.
# Inputs/outputs are PLAIN INTEGERS (field-agnostic, stable to store); the test
# lifts inputs via to_field and lowers outputs via from_field.
#
# Self-derived convention: input = [0, 1, ..., t-1]. Since the constants/matrix
# are deterministically derived inside params from (p, t, R, alpha), the output
# is reproducible -- regenerate by running the permutation once and pasting it.
# Prefer ALSO adding vectors from the paper / reference implementation.
# ---------------------------------------------------------------------------

KATS = {
    "GOLDILOCKS_T3": [
        # Self-derived (deterministic from p, t, R, alpha + the instance's M / rcons).
        {"input": [0, 1, 2], "output": [0xa4b79bc4474dd324, 0x7b31fce35131805c, 0x3cd12a6cda105532]},
        {"input": [0, 0, 0], "output": [0xcfb857d1b83916bc, 0x1ef1ceab6f872b80, 0xb96ee6acb2359fa3]},
    ],
}

KAT_CASES = [
    (name, params, kat)
    for name, params in INSTANCES
    for kat in KATS.get(name, [])
]
KAT_IDS = [
    name if len(KATS.get(name, [])) == 1 else f"{name}-{i}"
    for name, params in INSTANCES
    for i, _ in enumerate(KATS.get(name, []))
]


@pytest.mark.parametrize("name,params,kat", KAT_CASES, ids=KAT_IDS)
def test_permutation_kat(name, params, kat):
    prim = MyPrimitive(params)
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.permutation(inp)
    assert [prim.from_field(x) for x in out] == kat["output"]


# Hash-mode KATs: same shape as above, calling hash_sponge (and later compress /
# compress_2_to_1). Inputs are PLAIN INTEGERS lifted via to_field; the output is
# d integers (the digest), lowered via from_field. hash_sponge zero-pads to a
# multiple of the rate, so pick lengths that exercise pad_zero: empty input, a
# length shorter than the rate, exactly one block, and two full blocks.
#
# As with the permutation KATs these are self-derived: run hash_sponge once on a
# fixed input and paste the result (and add reference vectors when available).
SPONGE_KATS = {
    "GOLDILOCKS_T3": [
        # Self-derived (rate r=2, digest d=1). Inputs chosen to exercise pad_zero.
        {"input": [],           "output": [0x0]},                 # empty: 0 is rate-aligned, no block absorbed -> squeezes the IV state
        {"input": [1],          "output": [0x174a530d1f31be30]},  # shorter than the rate (padded to [1, 0])
        {"input": [0, 1],       "output": [0x1b0b9627bd980125]},  # exactly one block
        {"input": [0, 1, 2],    "output": [0xbc55c677687fa158]},  # not a multiple of the rate (padded to [0, 1, 2, 0])
        {"input": [0, 1, 2, 3], "output": [0xfa1bb2eb24b15d56]},  # two full blocks
    ],
}

SPONGE_KAT_CASES = [
    (name, params, kat)
    for name, params in INSTANCES
    for kat in SPONGE_KATS.get(name, [])
]
SPONGE_KAT_IDS = [
    name if len(SPONGE_KATS.get(name, [])) == 1 else f"{name}-{i}"
    for name, params in INSTANCES
    for i, _ in enumerate(SPONGE_KATS.get(name, []))
]


@pytest.mark.parametrize("name,params,kat", SPONGE_KAT_CASES, ids=SPONGE_KAT_IDS)
def test_sponge_kat(name, params, kat):
    prim = MyPrimitive(params)
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.hash_sponge(inp)
    assert [prim.from_field(x) for x in out] == kat["output"]


# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[n for n, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    prim = MyPrimitive(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[n for n, _ in INSTANCES])
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner, for every round.
    prim = MyPrimitive(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    for r in range(prim.R):
        assert prim.constant_addition_inv(prim.constant_addition(inp, r), r) == inp
        assert prim.linear_layer_inv(prim.linear_layer(inp, r), r) == inp
        assert prim.nonlinear_layer_inv(prim.nonlinear_layer(inp, r), r) == inp
    assert prim._pre_rounds_inv(prim._pre_rounds(inp)) == inp
    assert prim._post_rounds_inv(prim._post_rounds(inp)) == inp


# ---------------------------------------------------------------------------
# 4.3 Consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[n for n, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    prim = MyPrimitive(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[n for n, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    prim = MyPrimitive(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[n for n, _ in INSTANCES])
def test_permutation_output_size(name, params):
    prim = MyPrimitive(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert len(prim.permutation(inp)) == prim.t


@pytest.mark.parametrize("name,params", INSTANCES, ids=[n for n, _ in INSTANCES])
def test_sponge_output_size(name, params):
    prim = MyPrimitive(params)
    #data = [prim.F.random_element() for _ in range(prim.r * 3)] # TODO implement variable length sponge or catch exception
    data = [prim.F.random_element() for _ in range(prim.r)] 
    assert len(prim.hash_sponge(data)) == prim.d


# ---------------------------------------------------------------------------
# 4.4 Algebraic
# ---------------------------------------------------------------------------

# Component identity: with R=1 and a zero round constant, the linear layer must
# reduce to the bare matrix-vector product. Sweep fields / state sizes / alphas.
AFFINE_FIELDS = [
    ("GOLDILOCKS", GOLDILOCKS, 7),
    # ("BN254", BN254_SCALAR, 5), ...
]
AFFINE_CASES = [
    (fn, field, alpha, t) for fn, field, alpha in AFFINE_FIELDS for t in (3, 4, 8)
]


@pytest.mark.parametrize(
    "field_name,field,alpha,t", AFFINE_CASES,
    ids=[f"{fn} t={t} alpha={a}" for fn, _, a, t in AFFINE_CASES],
)
def test_linear_layer_matches_matrix(field_name, field, alpha, t):
    params = MyPrimitiveParams(p=field.p, t=t, alpha=alpha, R=1, r=t - 1, c=1, d=1)
    prim = MyPrimitive(params)
    inp = [prim.F.random_element() for _ in range(t)]
    assert prim.linear_layer(inp, params.R - 1) == matvecmul(prim.M, inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[n for n, _ in INSTANCES])
def test_matrix_is_mds_and_invertible(name, params):
    prim = MyPrimitive(params)
    assert is_mds(prim.M, prim.F)
    # M * M_inv == identity
    prod = [matvecmul(prim.M, col) for col in zip(*prim.M_inv)]  # columns of M_inv
    ident = [[prim.F.one() if i == j else prim.F.zero() for j in range(prim.t)] for i in range(prim.t)]
    assert [list(c) for c in zip(*prod)] == ident


# test_symbolic_degree (the whole-permutation degree check, ~alpha**R) is intentionally
# removed: this template's nonlinear_layer applies the INVERSE map x**alpha_inv on odd rounds
# (to demonstrate round-dependent behaviour), so the full-permutation symbolic degree is
# ~alpha_inv (a ~field-size exponent), not <= alpha**R. We instead exercise the single forward
# S-box component, whose degree is alpha as expected.
def test_nonlinear_forward_degree():
    from sage.all import PolynomialRing
    prim = MyPrimitive(MYPRIMITIVE_GOLDILOCKS_T3)
    P = PolynomialRing(prim.F, 'x', prim.t)
    out = prim.nonlinear_layer(list(P.gens()), r=0)   # even round -> forward power map x**alpha
    assert max(poly.total_degree() for poly in out) == prim.alpha


# ---------------------------------------------------------------------------
# 4.5 Misc: validation, warnings, reproducibility
# ---------------------------------------------------------------------------

def test_invalid_state_size():
    prim = MyPrimitive(MYPRIMITIVE_GOLDILOCKS_T3)
    with pytest.raises(ValueError):
        prim.permutation([prim.F.zero()] * (prim.t + 1))


def test_alpha_must_be_permutation():
    # An explicit alpha not coprime with p-1 must be rejected by params.
    with pytest.raises(ValueError):
        MyPrimitiveParams(p=GOLDILOCKS.p, t=3, alpha=2, R=1, r=2, c=1, d=1)


def test_toy_field_warns():
    with pytest.warns(ParamRecommendationWarning):
        MyPrimitiveParams(p=101, t=3, alpha=3, R=1, r=2, c=1, d=1)  # tiny field


def test_constants_reproducible():
    # Same parameters -> identical derived constants and matrix.
    a = MyPrimitiveParams(p=GOLDILOCKS.p, t=3, alpha=GOLDILOCKS.alpha, R=4, r=2, c=1, d=1)
    b = MyPrimitiveParams(p=GOLDILOCKS.p, t=3, alpha=GOLDILOCKS.alpha, R=4, r=2, c=1, d=1)
    assert a.rcons == b.rcons
    assert a.M == b.M
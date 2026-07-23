# test_arion.py
# ---------------------------------------------------------------------------
# Test suite for Arion, parametrized over the named instances in instances.py.
#
# Groups:
#   4.1 KATs         -- fixed input/output vectors (permutation)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- MDS check, derivation vs. pinned instance
#   4.5 Misc         -- validation/errors, warnings, reproducibility
# ---------------------------------------------------------------------------

import warnings

import pytest

from arion.hash import Arion
from arion.params import ArionParams
from arion.instances import ARION_BLS12_T3
from utils.field import BLS12_381_SCALAR
from utils.matrix import matvecmul, is_mds
from recommendations import ParamRecommendationWarning

INSTANCES = [
    ("BLS12_T3", ARION_BLS12_T3),
]

# ---------------------------------------------------------------------------
# 4.1 Known-answer test vectors (computed with this implementation; constants
# are deterministically derived via ArionParams._init_cons, so this KAT is
# self-reproducible from (p, t, R, alpha1, alpha2) alone)
# ---------------------------------------------------------------------------

KATS = {
    "BLS12_T3": {
        "input": [1, 2, 3],
        "output": [
            8659761558258982444581154623385831633595691755172963733105476337475047514700,
            4791269626370762720123406288230192995551974039613211374308744809811237991818,
            52352966164464576463303472127080635169313939454784230048314196193104401844767,
        ],
    },
}


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_kat(name, params):
    prim = Arion(params)
    kat = KATS[name]
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.permutation(inp)
    assert [prim.from_field(x) for x in out] == kat["output"]


# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    prim = Arion(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner, for every round.
    prim = Arion(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    for r in range(prim.R):
        assert prim.linear_layer_inv(prim.linear_layer(inp, r), r) == inp
        assert prim.constant_addition_inv(prim.constant_addition(inp, r), r) == inp
        assert prim.nonlinear_layer_inv(prim.nonlinear_layer(inp, r), r) == inp
    assert prim._pre_rounds_inv(prim._pre_rounds(inp)) == inp
    assert prim._post_rounds_inv(prim._post_rounds(inp)) == inp


# ---------------------------------------------------------------------------
# 4.3 Consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    prim = Arion(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    prim = Arion(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_sponge_output_size(name, params):
    prim = Arion(params)
    data = [prim.F.random_element() for _ in range(prim.sponge.r)]
    assert len(prim.hash_sponge(data)) == prim.sponge.d
    data = [prim.F.random_element() for _ in range(prim.sponge.r * 3)]
    assert len(prim.hash_sponge(data)) == prim.sponge.d


# ---------------------------------------------------------------------------
# 4.4 Algebraic / derivation
# ---------------------------------------------------------------------------

# TODO: add a symbolic degree-growth bound through GTDS once settled.

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_matrix_is_mds_and_invertible(name, params):
    assert is_mds(params.M, params.F)
    prod = [matvecmul(params.M, col) for col in zip(*params.M_inv)]  # columns of M_inv
    ident = [[params.F.one() if i == j else params.F.zero() for j in range(params.t)] for i in range(params.t)]
    assert [list(c) for c in zip(*prod)] == ident


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_generated_matches_instance(name, params):
    # Rebuilding the params without M / constants must reproduce the pinned instance.
    derived = ArionParams(p=params.p, t=params.t, R=params.R,
                          alpha1=params.alpha1, alpha2=params.alpha2,
                          r=params.sponge.r, c=params.sponge.c, d=params.sponge.d)
    assert derived.M == params.M
    assert derived.rcons == params.rcons
    assert derived.coeffs_g == params.coeffs_g
    assert derived.coeffs_h == params.coeffs_h


@pytest.mark.skip(reason="_init_rounds is a stub (round number derivation not implemented for Arion)")
@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_rounds_derivation_matches_instance(name, params):
    derived = ArionParams(p=params.p, t=params.t,
                          alpha1=params.alpha1, alpha2=params.alpha2,
                          r=params.sponge.r, c=params.sponge.c, d=params.sponge.d)  # R omitted -> _init_rounds
    assert derived.R == params.R


# ---------------------------------------------------------------------------
# 4.5 Misc: validation, warnings, reproducibility
# ---------------------------------------------------------------------------

def test_invalid_state_size():
    prim = Arion(ARION_BLS12_T3)
    with pytest.raises(ValueError):
        prim.permutation([prim.F.zero()] * (prim.t + 1))


def test_alpha_must_be_permutation():
    # An explicit alpha1 not coprime with p-1 must be rejected by params.
    with pytest.raises(ValueError):
        ArionParams(p=BLS12_381_SCALAR.p, t=3, R=6, alpha1=4, alpha2=257, r=2, c=1, d=2)


def test_toy_field_warns():
    with pytest.warns(ParamRecommendationWarning):
        ArionParams(p=101, t=3, R=6, r=2, c=1, d=1, toy=True)  # tiny field


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_recommended_instance_no_warning(name, params):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParamRecommendationWarning)
        ArionParams(p=params.p, t=params.t, R=params.R,
                    alpha1=params.alpha1, alpha2=params.alpha2,
                    r=params.sponge.r, c=params.sponge.c, d=params.sponge.d)


def test_constants_reproducible():
    # Same parameters -> identical derived constants and matrix.
    x = ArionParams(p=BLS12_381_SCALAR.p, t=3, R=6, alpha1=5, alpha2=257, r=2, c=1, d=2)
    y = ArionParams(p=BLS12_381_SCALAR.p, t=3, R=6, alpha1=5, alpha2=257, r=2, c=1, d=2)
    assert x.rcons == y.rcons
    assert x.coeffs_g == y.coeffs_g
    assert x.coeffs_h == y.coeffs_h
    assert x.M == y.M

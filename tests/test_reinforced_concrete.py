# test_reinforced_concrete.py
# ---------------------------------------------------------------------------
# Test suite for Reinforced Concrete, parametrized over the named instances.
#
# Groups:
#   4.1 KATs         -- fixed input/output vectors (permutation)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- derivation vs. pinned instance
#   4.5 Misc         -- validation/errors, warnings, reproducibility
# ---------------------------------------------------------------------------

import warnings

import pytest

from reinforced_concrete.hash import ReinforcedConcrete
from reinforced_concrete.params import ReinforcedConcreteParams
from reinforced_concrete.instances import RC_BLS12_T3, RC_BN254_T3, RC_ST_T3
from recommendations import ParamRecommendationWarning

INSTANCES = [
    ("BN254_T3", RC_BN254_T3),
    ("BLS12_T3", RC_BLS12_T3),
    ("ST_T3", RC_ST_T3),
]

# ---------------------------------------------------------------------------
# 4.1 Known-answer test vectors (from Rust reference implementation)
# ---------------------------------------------------------------------------

KATS = {
    "BN254_T3": [
        {
            "input": [0, 1, 2],
            "output": [
                0x2510ddf9405eebaa4d9a4e0a821bffc80ed439355c500985797becf45403e42e,
                0x1e8fd5b981b3b2d1cff86e3d99a9dbed002afdd7a29726de8f4d645d7841eafd,
                0x2c37d92c6d2b6831006bf8b53614f4f5fcc3ee6c5dff9d36a8460625d7ee6907,
            ],
        },
    ],
    "BLS12_T3": [
        {
            "input": [0, 1, 2],
            "output": [
                0x737df8e5a548189a0d77821a907def6736ea6512ba4633f1001f27d8f242913c,
                0x579c286d69635c6e3136f76e99775b478b29412a05516ac6201527abbb3ea098,
                0x5abe7c734229be9122f936d919f8babb74b36b1ca98f133b00256e29be115aa8,
            ],
        },
    ],
    "ST_T3": [
        {
            "input": [0, 1, 2],
            "output": [
                0x0026b02ce8c46a43773c7b8e2335642224aec1f72d060697faa4c3e99c7b524e,
                0x0314c340fa9da579d2b3466947836d130616e1ca35f1884ab36ed5a8d2e9212e,
                0x02ebb8984a6b0d773bf79e7b24bb3b722313e9c4e5be7cd36e279ed0d22a918a,
            ],
        },
    ],
}

KAT_CASES = [
    (name, params, kat)
    for name, params in INSTANCES
    for kat in KATS[name]
]
KAT_IDS = [
    name if len(KATS[name]) == 1 else f"{name}-{i}"
    for name, params in INSTANCES
    for i, _ in enumerate(KATS[name])
]


@pytest.mark.parametrize("name,params,kat", KAT_CASES, ids=KAT_IDS)
def test_permutation_kat(name, params, kat):
    prim = ReinforcedConcrete(params)
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.permutation(inp)
    assert [prim.from_field(x) for x in out] == kat["output"]


# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    prim = ReinforcedConcrete(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner. round_idx must index
    # into rcons; AffineLayer is the only round-dependent layer here.
    prim = ReinforcedConcrete(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    for r in range(prim.R):
        assert prim.linear_layer_inv(prim.linear_layer(inp, r), r) == inp
        assert prim.constant_addition_inv(prim.constant_addition(inp, r), r) == inp
        assert prim._bricks_inv(prim._bricks(inp, r), r) == inp
        assert prim._bars_inv(prim._bars(inp, r), r) == inp
        assert prim.nonlinear_layer_inv(prim.nonlinear_layer(inp, r), r) == inp
    assert prim._pre_rounds_inv(prim._pre_rounds(inp)) == inp
    assert prim._post_rounds_inv(prim._post_rounds(inp)) == inp


# ---------------------------------------------------------------------------
# 4.3 Consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    prim = ReinforcedConcrete(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    prim = ReinforcedConcrete(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_compress_output_size(name, params):
    prim = ReinforcedConcrete(params)
    x_m = [prim.F.random_element() for _ in range(prim.d)]
    x_c = [prim.F.random_element() for _ in range(prim.d)]
    assert len(prim.compress_2_to_1(x_m, x_c)) == prim.d


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_sponge_output_size(name, params):
    prim = ReinforcedConcrete(params)
    #data = [prim.F.random_element() for _ in range(prim.r * 3)] # TODO implement variable length sponge or catch exception
    data = [prim.F.random_element() for _ in range(prim.r)] 
    assert len(prim.hash_sponge(data)) == prim.d


# ---------------------------------------------------------------------------
# 4.4 Algebraic: derivation vs. pinned instance
# ---------------------------------------------------------------------------

def _instance_ints(params):
    """The instance's M and rcons lowered back to plain integers (constructor format)."""
    M = [[int(params.from_field(x)) for x in row] for row in params.M]
    rcons = [[int(params.from_field(x)) for x in row] for row in params.rcons]
    return M, rcons


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_rcons_generated_matches_instance(name, params):
    # Rebuilding the params without rcons must reproduce the pinned instance
    # (M is supplied: its generation is still a stub).
    M, _ = _instance_ints(params)
    derived = ReinforcedConcreteParams(
        p=params.p, t=params.t, alpha=params.alpha, alpha_inv=params.alpha_inv,
        R_pre=params.R_pre, R_bars=params.R_bars, R_post=params.R_post,
        si=params.si, M=M, r=params.r, c=params.c, d=params.d)
    assert derived.rcons == params.rcons


@pytest.mark.skip(reason="_init_mat is a stub (MDS matrix generation not implemented for Reinforced Concrete)")
@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_matrix_derivation_matches_instance(name, params):
    derived = ReinforcedConcreteParams(
        p=params.p, t=params.t, alpha=params.alpha, alpha_inv=params.alpha_inv,
        R_pre=params.R_pre, R_bars=params.R_bars, R_post=params.R_post,
        si=params.si, r=params.r, c=params.c, d=params.d)  # M omitted -> _init_mat
    assert derived.M == params.M


@pytest.mark.skip(reason="_init_rounds is a stub (round number derivation not implemented for Reinforced Concrete)")
@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_rounds_derivation_matches_instance(name, params):
    M, _ = _instance_ints(params)
    derived = ReinforcedConcreteParams(
        p=params.p, t=params.t, alpha=params.alpha, alpha_inv=params.alpha_inv,
        si=params.si, M=M, r=params.r, c=params.c, d=params.d)  # rounds omitted -> _init_rounds
    assert (derived.R_pre, derived.R_bars, derived.R_post) == (params.R_pre, params.R_bars, params.R_post)


# ---------------------------------------------------------------------------
# 4.5 Misc: validation, warnings, reproducibility
# ---------------------------------------------------------------------------

def test_invalid_state_size():
    prim = ReinforcedConcrete(RC_BN254_T3)
    with pytest.raises(ValueError):
        prim.permutation([prim.F.zero()] * (prim.t + 1))


def test_alpha_must_be_permutation():
    # An explicit alpha not coprime with p-1 must be rejected by params.
    M, _ = _instance_ints(RC_BN254_T3)
    with pytest.raises(ValueError):
        ReinforcedConcreteParams(
            p=RC_BN254_T3.p, t=3, alpha=2, R_pre=3, R_bars=1, R_post=3,
            si=RC_BN254_T3.si, M=M, r=2, c=1, d=1)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_recommended_instance_no_warning(name, params):
    M, _ = _instance_ints(params)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParamRecommendationWarning)
        ReinforcedConcreteParams(
            p=params.p, t=params.t, alpha=params.alpha, alpha_inv=params.alpha_inv,
            R_pre=params.R_pre, R_bars=params.R_bars, R_post=params.R_post,
            si=params.si, M=M, r=params.r, c=params.c, d=params.d)


def test_constants_reproducible():
    # Same parameters -> identical derived constants, LUT, and Bricks coefficients.
    M, _ = _instance_ints(RC_BN254_T3)
    kwargs = dict(p=RC_BN254_T3.p, t=3, alpha=RC_BN254_T3.alpha,
                  R_pre=3, R_bars=1, R_post=3, si=RC_BN254_T3.si, M=M,
                  r=2, c=1, d=1)
    a = ReinforcedConcreteParams(**kwargs)
    b = ReinforcedConcreteParams(**kwargs)
    assert a.rcons == b.rcons
    assert a.LUT == b.LUT
    assert a.a_coeffs == b.a_coeffs and a.b_coeffs == b.b_coeffs

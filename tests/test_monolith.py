# test_monolith.py
# ---------------------------------------------------------------------------
# Test suite for Monolith, parametrized over the named instances in instances.py.
#
# Groups:
#   4.1 KATs         -- fixed input/output vectors (permutation)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- lookup-table generation, derivation vs. pinned instance
#   4.5 Misc         -- validation/errors, warnings, compression availability
# ---------------------------------------------------------------------------

import warnings

import pytest

from monolith.hash import MonolithPerm, MonolithHash
from monolith.instances import (
    MONOLITH_M31_T16,
    MONOLITH_M31_T24,
    MONOLITH_GOLDILOCKS_T8,
    MONOLITH_GOLDILOCKS_T12,
    LUT_8,
    LUT_7,
)
from monolith.params import MonolithParams, MONOLITH_LUT8, MONOLITH_LUT7
from recommendations import ParamRecommendationWarning

INSTANCES = [
    ("M31_T16", MONOLITH_M31_T16),
    ("M31_T24", MONOLITH_M31_T24),
    ("GOLDILOCKS_T8", MONOLITH_GOLDILOCKS_T8),
    ("GOLDILOCKS_T12", MONOLITH_GOLDILOCKS_T12),
]

# ---------------------------------------------------------------------------
# Known-answer test vectors (from Rust reference implementation)
# ---------------------------------------------------------------------------

KATS = {
    "M31_T16": [
        {
            "input": list(range(16)),
            "output": [
                609156607, 290107110, 1900746598, 1734707571, 2050994835, 1648553244,
                1307647296, 1941164548, 1707113065, 1477714255, 1170160793, 93800695,
                769879348, 375548503, 1989726444, 1349325635,
            ],
        },
    ],
    "M31_T24": [
        {
            "input": list(range(24)),
            "output": [
                2067773075, 1832201932, 1944824478, 1823377759, 1441396277, 2131077448,
                2132180368, 1432941899, 1347592327, 1652902071, 1809291778, 1684517779,
                785982444, 1037200378, 1316286130, 1391154514, 1760346031, 1412575993,
                2108791223, 1657735769, 219740691, 1165267731, 505815021, 2080295871,
            ],
        },
    ],
    "GOLDILOCKS_T8": [
        {
            "input": list(range(8)),
            "output": [
                3656442354255169651, 1088199316401146975, 22941152274975507,
                14434181924633355796, 6981961052218049719, 16492720827407246378,
                17986182688944525029, 9161400698613172623,
            ],
        },
    ],
    "GOLDILOCKS_T12": [
        {
            "input": list(range(12)),
            "output": [
                5867581605548782913, 588867029099903233, 6043817495575026667,
                805786589926590032, 9919982299747097782, 6718641691835914685,
                7951881005429661950, 15453177927755089358, 974633365445157727,
                9654662171963364206, 6281307445101925412, 13745376999934453119,
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

# ---------------------------------------------------------------------------
# 4.4 Algebraic: lookup table verification
# ---------------------------------------------------------------------------

def test_lut_8_matches_computed():
    assert LUT_8 == MONOLITH_LUT8

def test_lut_7_matches_computed():
    assert LUT_7 == MONOLITH_LUT7

# ---------------------------------------------------------------------------
# 4.1 KAT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params,kat", KAT_CASES, ids=KAT_IDS)
def test_permutation_kat(name, params, kat):
    P = MonolithPerm(params)
    inp = [P.to_field(x) for x in kat["input"]]
    out = P.permute(inp)
    assert [P.from_field(x) for x in out] == kat["output"]

# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    P = MonolithPerm(params)
    inp = [P.F.random_element() for _ in range(P.t)]
    assert P.permute_inv(P.permute(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner, for every round.
    P = MonolithPerm(params)
    inp = [P.F.random_element() for _ in range(P.t)]
    for r in range(P.R):
        assert P.linear_layer_inv(P.linear_layer(inp, r), r) == inp
        assert P.constant_addition_inv(P.constant_addition(inp, r), r) == inp
        assert P._bricks_inv(P._bricks(inp, r), r) == inp
        assert P._bars_inv(P._bars(inp, r), r) == inp
        assert P.nonlinear_layer_inv(P.nonlinear_layer(inp, r), r) == inp
    assert P._pre_rounds_inv(P._pre_rounds(inp)) == inp
    assert P._post_rounds_inv(P._post_rounds(inp)) == inp

# ---------------------------------------------------------------------------
# 4.3 Consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    P = MonolithPerm(params)
    inp = [P.F.random_element() for _ in range(P.t)]
    assert P.permute(inp) == P.permute(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    P = MonolithPerm(params)
    inp1 = [P.F.random_element() for _ in range(P.t)]
    inp2 = [P.F.random_element() for _ in range(P.t)]
    while inp1 == inp2:
        inp2 = [P.F.random_element() for _ in range(P.t)]
    assert P.permute(inp1) != P.permute(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_sponge_output_size(name, params):
    P = MonolithPerm(params)
    H = MonolithHash(P, params.sponge)
    data = [P.F.random_element() for _ in range(params.sponge["r"])]
    assert len(H.hash(data)) == H.sponge.d
    data = [P.F.random_element() for _ in range(params.sponge["r"] * 3)]
    assert len(H.hash(data)) == H.sponge.d


# ---------------------------------------------------------------------------
# 4.5 Misc: validation, compression availability
# ---------------------------------------------------------------------------

def test_invalid_state_size():
    P = MonolithPerm(MONOLITH_M31_T16)
    with pytest.raises(ValueError):
        P.permute([P.F.zero()] * (P.t + 1))


# ---------------------------------------------------------------------------
# 4.4 (cont.) Derivation vs. pinned instance
# ---------------------------------------------------------------------------

def _instance_ints(params):
    """The instance's M and rcons lowered back to plain integers (constructor format)."""
    M = [[int(params.from_field(x)) for x in row] for row in params.M]
    # Strip the trailing zero padding row the constructor appends.
    rcons = [[int(params.from_field(x)) for x in row] for row in params.rcons[:-1]]
    return M, rcons


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_rcons_generated_matches_instance(name, params):
    # Rebuilding the params without rcons must reproduce the pinned instance
    # (M and LUTs are supplied: their generation is still a stub).
    M, _ = _instance_ints(params)
    derived = MonolithParams(p=params.p, t=params.t, R=params.R, u=params.u,
                             si=params.si, LUTs=params.LUTs, M=M,
                             sponge=dict(r=params.sponge["r"], c=params.sponge["c"], d=params.sponge["d"]))
    assert derived.rcons == params.rcons


@pytest.mark.skip(reason="_init_mat is a stub (MDS matrix generation not implemented for Monolith)")
@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_matrix_derivation_matches_instance(name, params):
    derived = MonolithParams(p=params.p, t=params.t, R=params.R, u=params.u,
                             si=params.si, LUTs=params.LUTs,
                             sponge=dict(r=params.sponge["r"], c=params.sponge["c"], d=params.sponge["d"]))  # M omitted -> _init_mat
    assert derived.M == params.M


@pytest.mark.skip(reason="_init_rounds is a stub (round number derivation not implemented for Monolith)")
@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_rounds_derivation_matches_instance(name, params):
    M, _ = _instance_ints(params)
    derived = MonolithParams(p=params.p, t=params.t, u=params.u,
                             si=params.si, LUTs=params.LUTs, M=M,
                             sponge=dict(r=params.sponge["r"], c=params.sponge["c"], d=params.sponge["d"]))  # R omitted -> _init_rounds
    assert derived.R == params.R


# ---------------------------------------------------------------------------
# 4.5 (cont.) Warnings, reproducibility
# ---------------------------------------------------------------------------

def test_toy_field_warns():
    # 13-bit Mersenne prime; M supplied because _init_mat is a stub.
    with pytest.warns(ParamRecommendationWarning):
        MonolithParams(p=8191, t=4, R=3, u=2, si=[128, 128],
                       M=[[1, 2, 3, 4], [4, 1, 2, 3], [3, 4, 1, 2], [2, 3, 4, 1]],
                       sponge=dict(r=2, c=2, d=2), toy=True)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_recommended_instance_no_warning(name, params):
    M, _ = _instance_ints(params)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParamRecommendationWarning)
        MonolithParams(p=params.p, t=params.t, R=params.R, u=params.u,
                       si=params.si, LUTs=params.LUTs, M=M,
                       sponge=dict(r=params.sponge["r"], c=params.sponge["c"], d=params.sponge["d"]))


def test_constants_reproducible():
    # Same parameters -> identical derived constants.
    M, _ = _instance_ints(MONOLITH_GOLDILOCKS_T8)
    kwargs = dict(p=MONOLITH_GOLDILOCKS_T8.p, t=8, R=6, u=4, si=[256] * 8,
                  M=M, sponge=dict(r=4, c=4, d=4))
    a = MonolithParams(**kwargs)
    b = MonolithParams(**kwargs)
    assert a.rcons == b.rcons

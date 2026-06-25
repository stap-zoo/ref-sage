# test_monolith.py
# ---------------------------------------------------------------------------
# Test suite for Monolith, parametrized over the named instances in instances.py.
#
# Groups:
#   4.1 KATs         -- fixed input/output vectors (permutation)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- lookup-table generation
#   4.5 Misc         -- validation/errors, compression availability
# ---------------------------------------------------------------------------

import pytest

from monolith.hash import Monolith
from monolith.instances import (
    MONOLITH_M31_T16,
    MONOLITH_M31_T24,
    MONOLITH_GOLDILOCKS_T8,
    MONOLITH_GOLDILOCKS_T12,
    LUT_8,
    LUT_7,
)
from monolith.params import monolith_lut8, monolith_lut7

INSTANCES = [
    ("M31_T16", MONOLITH_M31_T16),
    ("M31_T24", MONOLITH_M31_T24),
    ("GOLDILOCKS_T8", MONOLITH_GOLDILOCKS_T8),
    ("GOLDILOCKS_T12", MONOLITH_GOLDILOCKS_T12),
]

# Instances where t == 2*d, so compress_2_to_1 is defined
COMPRESS_INSTANCES = [
    ("M31_T16", MONOLITH_M31_T16),
    ("GOLDILOCKS_T8", MONOLITH_GOLDILOCKS_T8),
]

# Instances where t != 2*d, so compress_2_to_1 raises ValueError
NO_COMPRESS_INSTANCES = [
    ("M31_T24", MONOLITH_M31_T24),
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
    assert LUT_8 == monolith_lut8

def test_lut_7_matches_computed():
    assert LUT_7 == monolith_lut7

# ---------------------------------------------------------------------------
# 4.1 KAT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params,kat", KAT_CASES, ids=KAT_IDS)
def test_permutation_kat(name, params, kat):
    prim = Monolith(params)
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.permutation(inp)
    assert [prim.from_field(x) for x in out] == kat["output"]

# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    prim = Monolith(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner, for every round.
    prim = Monolith(params)
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
    prim = Monolith(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    prim = Monolith(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_sponge_output_size(name, params):
    prim = Monolith(params)
    #data = [prim.F.random_element() for _ in range(prim.r * 3)]  # TODO implement variable length sponge or catch exception
    data = [prim.F.random_element() for _ in range(prim.r)] 
    assert len(prim.hash_sponge(data)) == prim.d


# ---------------------------------------------------------------------------
# 4.5 Misc: validation, compression availability
# ---------------------------------------------------------------------------

def test_invalid_state_size():
    prim = Monolith(MONOLITH_M31_T16)
    with pytest.raises(ValueError):
        prim.permutation([prim.F.zero()] * (prim.t + 1))


@pytest.mark.parametrize("name,params", COMPRESS_INSTANCES, ids=[name for name, _ in COMPRESS_INSTANCES])
def test_compress_output_size(name, params):
    prim = Monolith(params)
    half = prim.t // 2
    x1 = [prim.F.random_element() for _ in range(half)]
    x2 = [prim.F.random_element() for _ in range(half)]
    assert len(prim.compress_2_to_1(x1, x2)) == prim.d


@pytest.mark.parametrize("name,params", NO_COMPRESS_INSTANCES, ids=[name for name, _ in NO_COMPRESS_INSTANCES])
def test_compress_not_defined(name, params):
    prim = Monolith(params)
    half = prim.t // 2
    x1 = [prim.F.random_element() for _ in range(half)]
    x2 = [prim.F.random_element() for _ in range(half)]
    with pytest.raises(ValueError):
        prim.compress_2_to_1(x1, x2)

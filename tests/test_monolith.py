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
from monolith.params import compute_lut_8, compute_lut_7

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
# Lookup table verification
# ---------------------------------------------------------------------------

def test_lut_8_matches_computed():
    assert LUT_8 == compute_lut_8()

def test_lut_7_matches_computed():
    assert LUT_7 == compute_lut_7()

# ---------------------------------------------------------------------------
# KAT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params,kat", KAT_CASES, ids=KAT_IDS)
def test_permutation_kat(name, params, kat):
    m = Monolith(params)
    inp = [m.to_field(x) for x in kat["input"]]
    out = m.permutation(inp)
    assert [m.from_field(x) for x in out] == kat["output"]

# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    m = Monolith(params)
    inp = [m.F.random_element() for _ in range(m.t)]
    assert m.permutation(inp) == m.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    m = Monolith(params)
    inp1 = [m.F.random_element() for _ in range(m.t)]
    inp2 = [m.F.random_element() for _ in range(m.t)]
    while inp1 == inp2:
        inp2 = [m.F.random_element() for _ in range(m.t)]
    assert m.permutation(inp1) != m.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    m = Monolith(params)
    inp = [m.F.random_element() for _ in range(m.t)]
    assert m.permutation_inv(m.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_sponge_output_size(name, params):
    m = Monolith(params)
    data = [m.F.random_element() for _ in range(m.r * 3)]
    assert len(m.hash_sponge(data)) == m.d


@pytest.mark.parametrize("name,params", COMPRESS_INSTANCES, ids=[name for name, _ in COMPRESS_INSTANCES])
def test_compress_output_size(name, params):
    m = Monolith(params)
    half = m.t // 2
    x1 = [m.F.random_element() for _ in range(half)]
    x2 = [m.F.random_element() for _ in range(half)]
    assert len(m.compress_2_to_1(x1, x2)) == m.d


@pytest.mark.parametrize("name,params", NO_COMPRESS_INSTANCES, ids=[name for name, _ in NO_COMPRESS_INSTANCES])
def test_compress_not_defined(name, params):
    m = Monolith(params)
    half = m.t // 2
    x1 = [m.F.random_element() for _ in range(half)]
    x2 = [m.F.random_element() for _ in range(half)]
    with pytest.raises(ValueError):
        m.compress_2_to_1(x1, x2)

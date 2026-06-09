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

# ---------------------------------------------------------------------------
# Known-answer test vectors (from Rust reference implementation)
# ---------------------------------------------------------------------------

M31_T16_KATS = [
    {
        "input": list(range(16)),
        "output": [
            609156607, 290107110, 1900746598, 1734707571, 2050994835, 1648553244,
            1307647296, 1941164548, 1707113065, 1477714255, 1170160793, 93800695,
            769879348, 375548503, 1989726444, 1349325635,
        ],
    },
]

M31_T24_KATS = [
    {
        "input": list(range(24)),
        "output": [
            2067773075, 1832201932, 1944824478, 1823377759, 1441396277, 2131077448,
            2132180368, 1432941899, 1347592327, 1652902071, 1809291778, 1684517779,
            785982444, 1037200378, 1316286130, 1391154514, 1760346031, 1412575993,
            2108791223, 1657735769, 219740691, 1165267731, 505815021, 2080295871,
        ],
    },
]

GOLDILOCKS_T8_KATS = [
    {
        "input": list(range(8)),
        "output": [
            3656442354255169651, 1088199316401146975, 22941152274975507,
            14434181924633355796, 6981961052218049719, 16492720827407246378,
            17986182688944525029, 9161400698613172623,
        ],
    },
]

GOLDILOCKS_T12_KATS = [
    {
        "input": list(range(12)),
        "output": [
            5867581605548782913, 588867029099903233, 6043817495575026667,
            805786589926590032, 9919982299747097782, 6718641691835914685,
            7951881005429661950, 15453177927755089358, 974633365445157727,
            9654662171963364206, 6281307445101925412, 13745376999934453119,
        ],
    },
]

# ---------------------------------------------------------------------------
# Lookup table verification
# ---------------------------------------------------------------------------

def test_lut_8_matches_computed():
    assert LUT_8 == compute_lut_8()

def test_lut_7_matches_computed():
    assert LUT_7 == compute_lut_7()

# ---------------------------------------------------------------------------
# M31 T=16 tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def m_m31_t16():
    return Monolith(MONOLITH_M31_T16)

@pytest.mark.parametrize("kat", M31_T16_KATS)
def test_m31_t16_kat(m_m31_t16, kat):
    inp = [m_m31_t16.to_field(x) for x in kat["input"]]
    out = m_m31_t16.permutation(inp)
    assert [m_m31_t16.from_field(x) for x in out] == kat["output"]

def test_m31_t16_permutation_deterministic(m_m31_t16):
    inp = [m_m31_t16.F.random_element() for _ in range(m_m31_t16.t)]
    assert m_m31_t16.permutation(inp) == m_m31_t16.permutation(inp)

def test_m31_t16_permutation_roundtrip(m_m31_t16):
    inp = [m_m31_t16.F.random_element() for _ in range(m_m31_t16.t)]
    assert m_m31_t16.permutation_inv(m_m31_t16.permutation(inp)) == inp

def test_m31_t16_compress_output_size(m_m31_t16):
    half = m_m31_t16.t // 2
    x1 = [m_m31_t16.F.random_element() for _ in range(half)]
    x2 = [m_m31_t16.F.random_element() for _ in range(half)]
    assert len(m_m31_t16.compress_2_to_1(x1, x2)) == m_m31_t16.d

def test_m31_t16_sponge_output_size(m_m31_t16):
    data = [m_m31_t16.F.random_element() for _ in range(m_m31_t16.r * 3)]
    assert len(m_m31_t16.hash_sponge(data)) == m_m31_t16.d

# ---------------------------------------------------------------------------
# M31 T=24 tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def m_m31_t24():
    return Monolith(MONOLITH_M31_T24)

@pytest.mark.parametrize("kat", M31_T24_KATS)
def test_m31_t24_kat(m_m31_t24, kat):
    inp = [m_m31_t24.to_field(x) for x in kat["input"]]
    out = m_m31_t24.permutation(inp)
    assert [m_m31_t24.from_field(x) for x in out] == kat["output"]

def test_m31_t24_permutation_deterministic(m_m31_t24):
    inp = [m_m31_t24.F.random_element() for _ in range(m_m31_t24.t)]
    assert m_m31_t24.permutation(inp) == m_m31_t24.permutation(inp)

def test_m31_t24_permutation_roundtrip(m_m31_t24):
    inp = [m_m31_t24.F.random_element() for _ in range(m_m31_t24.t)]
    assert m_m31_t24.permutation_inv(m_m31_t24.permutation(inp)) == inp

def test_m31_t24_compress_output_size(m_m31_t24):
    half = m_m31_t24.t // 2
    x1 = [m_m31_t24.F.random_element() for _ in range(half)]
    x2 = [m_m31_t24.F.random_element() for _ in range(half)]
    with pytest.raises(ValueError):
        m_m31_t24.compress_2_to_1(x1, x2)

def test_m31_t24_sponge_output_size(m_m31_t24):
    data = [m_m31_t24.F.random_element() for _ in range(m_m31_t24.r * 3)]
    assert len(m_m31_t24.hash_sponge(data)) == m_m31_t24.d

# ---------------------------------------------------------------------------
# Goldilocks T=8 tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def m_goldilocks_t8():
    return Monolith(MONOLITH_GOLDILOCKS_T8)

@pytest.mark.parametrize("kat", GOLDILOCKS_T8_KATS)
def test_goldilocks_t8_kat(m_goldilocks_t8, kat):
    inp = [m_goldilocks_t8.to_field(x) for x in kat["input"]]
    out = m_goldilocks_t8.permutation(inp)
    assert [m_goldilocks_t8.from_field(x) for x in out] == kat["output"]

def test_goldilocks_t8_permutation_deterministic(m_goldilocks_t8):
    inp = [m_goldilocks_t8.F.random_element() for _ in range(m_goldilocks_t8.t)]
    assert m_goldilocks_t8.permutation(inp) == m_goldilocks_t8.permutation(inp)

def test_goldilocks_t8_permutation_roundtrip(m_goldilocks_t8):
    inp = [m_goldilocks_t8.F.random_element() for _ in range(m_goldilocks_t8.t)]
    assert m_goldilocks_t8.permutation_inv(m_goldilocks_t8.permutation(inp)) == inp

def test_goldilocks_t8_compress_output_size(m_goldilocks_t8):
    half = m_goldilocks_t8.t // 2
    x1 = [m_goldilocks_t8.F.random_element() for _ in range(half)]
    x2 = [m_goldilocks_t8.F.random_element() for _ in range(half)]
    assert len(m_goldilocks_t8.compress_2_to_1(x1, x2)) == m_goldilocks_t8.d

def test_goldilocks_t8_sponge_output_size(m_goldilocks_t8):
    data = [m_goldilocks_t8.F.random_element() for _ in range(m_goldilocks_t8.r * 3)]
    assert len(m_goldilocks_t8.hash_sponge(data)) == m_goldilocks_t8.d

# ---------------------------------------------------------------------------
# Goldilocks T=12 tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def m_goldilocks_t12():
    return Monolith(MONOLITH_GOLDILOCKS_T12)

@pytest.mark.parametrize("kat", GOLDILOCKS_T12_KATS)
def test_goldilocks_t12_kat(m_goldilocks_t12, kat):
    inp = [m_goldilocks_t12.to_field(x) for x in kat["input"]]
    out = m_goldilocks_t12.permutation(inp)
    assert [m_goldilocks_t12.from_field(x) for x in out] == kat["output"]

def test_goldilocks_t12_permutation_deterministic(m_goldilocks_t12):
    inp = [m_goldilocks_t12.F.random_element() for _ in range(m_goldilocks_t12.t)]
    assert m_goldilocks_t12.permutation(inp) == m_goldilocks_t12.permutation(inp)

def test_goldilocks_t12_permutation_roundtrip(m_goldilocks_t12):
    inp = [m_goldilocks_t12.F.random_element() for _ in range(m_goldilocks_t12.t)]
    assert m_goldilocks_t12.permutation_inv(m_goldilocks_t12.permutation(inp)) == inp

def test_goldilocks_t12_compress_output_size(m_goldilocks_t12):
    half = m_goldilocks_t12.t // 2
    x1 = [m_goldilocks_t12.F.random_element() for _ in range(half)]
    x2 = [m_goldilocks_t12.F.random_element() for _ in range(half)]
    with pytest.raises(ValueError):
        m_goldilocks_t12.compress_2_to_1(x1, x2)

def test_goldilocks_t12_sponge_output_size(m_goldilocks_t12):
    data = [m_goldilocks_t12.F.random_element() for _ in range(m_goldilocks_t12.r * 3)]
    assert len(m_goldilocks_t12.hash_sponge(data)) == m_goldilocks_t12.d

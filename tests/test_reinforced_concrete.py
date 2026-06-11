import pytest

from reinforced_concrete.hash import ReinforcedConcrete
from reinforced_concrete.instances import RC_BLS12_T3, RC_BN254_T3, RC_ST_T3

INSTANCES = [
    ("BN254_T3", RC_BN254_T3),
    ("BLS12_T3", RC_BLS12_T3),
    ("ST_T3", RC_ST_T3),
]

# ---------------------------------------------------------------------------
# Known-answer test vectors (from Rust reference implementation)
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
    rc = ReinforcedConcrete(params)
    inp = [rc.to_field(x) for x in kat["input"]]
    out = rc.permutation(inp)
    assert [rc.from_field(x) for x in out] == kat["output"]


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    rc = ReinforcedConcrete(params)
    inp = [rc.F.random_element() for _ in range(rc.t)]
    assert rc.permutation(inp) == rc.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    rc = ReinforcedConcrete(params)
    inp1 = [rc.F.random_element() for _ in range(rc.t)]
    inp2 = [rc.F.random_element() for _ in range(rc.t)]
    while inp1 == inp2:
        inp2 = [rc.F.random_element() for _ in range(rc.t)]
    assert rc.permutation(inp1) != rc.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    rc = ReinforcedConcrete(params)
    inp = [rc.F.random_element() for _ in range(rc.t)]
    assert rc.permutation_inv(rc.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_compress_output_size(name, params):
    rc = ReinforcedConcrete(params)
    x_m = [rc.F.random_element() for _ in range(rc.d)]
    x_c = [rc.F.random_element() for _ in range(rc.d)]
    assert len(rc.compress_2_to_1(x_m, x_c)) == rc.d


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_sponge_output_size(name, params):
    rc = ReinforcedConcrete(params)
    data = [rc.F.random_element() for _ in range(rc.r * 3)]
    assert len(rc.hash_sponge(data)) == rc.d

import pytest
from reinforced_concrete.hash import ReinforcedConcrete
from reinforced_concrete.instances import RC_BLS_PARAMS, RC_BN_PARAMS, RC_ST_PARAMS

# ---------------------------------------------------------------------------
# Known-answer test vectors (from Rust reference implementation)
# ---------------------------------------------------------------------------

BN254_KATS = [
    {
        "input": [0, 1, 2],
        "output": [
            0x2510ddf9405eebaa4d9a4e0a821bffc80ed439355c500985797becf45403e42e,
            0x1e8fd5b981b3b2d1cff86e3d99a9dbed002afdd7a29726de8f4d645d7841eafd,
            0x2c37d92c6d2b6831006bf8b53614f4f5fcc3ee6c5dff9d36a8460625d7ee6907,
        ],
    },
]

BLS12_KATS = [
    {
        "input": [0, 1, 2],
        "output": [
            0x737df8e5a548189a0d77821a907def6736ea6512ba4633f1001f27d8f242913c,
            0x579c286d69635c6e3136f76e99775b478b29412a05516ac6201527abbb3ea098,
            0x5abe7c734229be9122f936d919f8babb74b36b1ca98f133b00256e29be115aa8,
        ],
    },
]

ST_KATS = [
    {
        "input": [0, 1, 2],
        "output": [
            0x0026b02ce8c46a43773c7b8e2335642224aec1f72d060697faa4c3e99c7b524e,
            0x0314c340fa9da579d2b3466947836d130616e1ca35f1884ab36ed5a8d2e9212e,
            0x02ebb8984a6b0d773bf79e7b24bb3b722313e9c4e5be7cd36e279ed0d22a918a,
        ],
    },
]

# ---------------------------------------------------------------------------
# BN254 tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rc_bn():
    return ReinforcedConcrete(RC_BN_PARAMS)

@pytest.mark.parametrize("kat", BN254_KATS)
def test_bn254_kat(rc_bn, kat):
    inp = [rc_bn.to_field(x) for x in kat["input"]]
    out = rc_bn.permutation(inp)
    assert [rc_bn.from_field(x) for x in out] == kat["output"]

def test_bn254_compose_roundtrip(rc_bn):
    for _ in range(5):
        x = rc_bn.F.random_element()
        assert rc_bn.compose(rc_bn.decompose(x)) == x

def test_bn254_permutation_deterministic(rc_bn):
    inp = [rc_bn.F.random_element() for _ in range(3)]
    assert rc_bn.permutation(inp) == rc_bn.permutation(inp)

def test_bn254_compress_output_size(rc_bn):
    rate = rc_bn.state_size - rc_bn.digest_size
    x_m = [rc_bn.F.random_element() for _ in range(rate)]
    x_c = [rc_bn.F.random_element() for _ in range(rc_bn.digest_size)]
    assert len(rc_bn.compress(x_m, x_c)) == rc_bn.digest_size

def test_bn254_permutation_roundtrip(rc_bn):
    inp = [rc_bn.F.random_element() for _ in range(rc_bn.state_size)]
    assert rc_bn.permutation_inv(rc_bn.permutation(inp)) == inp

def test_bn254_sponge_output_size(rc_bn):
    rate = rc_bn.state_size - rc_bn.digest_size
    data = [rc_bn.F.random_element() for _ in range(rate * 3)]
    assert len(rc_bn.hash_sponge(data))    == rc_bn.digest_size

# ---------------------------------------------------------------------------
# BLS12 tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rc_bls():
    return ReinforcedConcrete(RC_BLS_PARAMS)

@pytest.mark.parametrize("kat", BLS12_KATS)
def test_bls12_kat(rc_bls, kat):
    inp = [rc_bls.to_field(x) for x in kat["input"]]
    out = rc_bls.permutation(inp)
    assert [rc_bls.from_field(x) for x in out] == kat["output"]

def test_bls12_compose_roundtrip(rc_bls):
    for _ in range(5):
        x = rc_bls.F.random_element()
        assert rc_bls.compose(rc_bls.decompose(x)) == x

def test_bls12_permutation_deterministic(rc_bls):
    inp = [rc_bls.F.random_element() for _ in range(3)]
    assert rc_bls.permutation(inp) == rc_bls.permutation(inp)

def test_bls12_compress_output_size(rc_bls):
    rate = rc_bls.state_size - rc_bls.digest_size
    x_m = [rc_bls.F.random_element() for _ in range(rate)]
    x_c = [rc_bls.F.random_element() for _ in range(rc_bls.digest_size)]
    assert len(rc_bls.compress(x_m, x_c)) == rc_bls.digest_size

def test_bls12_permutation_roundtrip(rc_bls):
    inp = [rc_bls.F.random_element() for _ in range(rc_bls.state_size)]
    assert rc_bls.permutation_inv(rc_bls.permutation(inp)) == inp

def test_bls12_sponge_output_size(rc_bls):
    rate = rc_bls.state_size - rc_bls.digest_size
    data = [rc_bls.F.random_element() for _ in range(rate * 3)]
    assert len(rc_bls.hash_sponge(data))    == rc_bls.digest_size

# ---------------------------------------------------------------------------
# ST tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rc_st():
    return ReinforcedConcrete(RC_ST_PARAMS)

@pytest.mark.parametrize("kat", ST_KATS)
def test_st_kat(rc_st, kat):
    inp = [rc_st.to_field(x) for x in kat["input"]]
    out = rc_st.permutation(inp)
    assert [rc_st.from_field(x) for x in out] == kat["output"]

def test_st_compose_roundtrip(rc_st):
    for _ in range(5):
        x = rc_st.F.random_element()
        assert rc_st.compose(rc_st.decompose(x)) == x

def test_st_permutation_deterministic(rc_st):
    inp = [rc_st.F.random_element() for _ in range(3)]
    assert rc_st.permutation(inp) == rc_st.permutation(inp)

def test_st_compress_output_size(rc_st):
    rate = rc_st.state_size - rc_st.digest_size
    x_m = [rc_st.F.random_element() for _ in range(rate)]
    x_c = [rc_st.F.random_element() for _ in range(rc_st.digest_size)]
    assert len(rc_st.compress(x_m, x_c)) == rc_st.digest_size

def test_st_permutation_roundtrip(rc_st):
    inp = [rc_st.F.random_element() for _ in range(rc_st.state_size)]
    assert rc_st.permutation_inv(rc_st.permutation(inp)) == inp

def test_st_sponge_output_size(rc_st):
    rate = rc_st.state_size - rc_st.digest_size
    data = [rc_st.F.random_element() for _ in range(rate * 3)]
    assert len(rc_st.hash_sponge(data)) == rc_st.digest_size

import pytest

from rescue_prime.hash import RescuePrime
from rescue_prime.instances import (
    RESCUE_PRIME_BLS12_T3,
    RESCUE_PRIME_BN254_T3,
    RESCUE_PRIME_ST_T3,
    RESCUE_PRIME_GOLDILOCKS_T8,
    RESCUE_PRIME_GOLDILOCKS_T12,
)

# ---------------------------------------------------------------------------
# Known-answer test vectors (from the Sage reference implementation)
# https://github.com/KULeuven-COSIC/Marvellous/blob/master/rescue_prime.sage
# ---------------------------------------------------------------------------

T3_KATS = {
    "BLS12": {
        "input": [0, 1, 2],
        "output": [
            0x2e1183b4ae571061ed9514118392ede2904ae1376d61653de09083cf0b31abce,
            0x38f9e521c67c329a53403dd42999b19c3bfe355e594752c87ada74da35c74b85,
            0x69a193e3c2734c26d85d191a1e521c1bc8024c9047bb5c79835ed5cfc2d8440e,
        ],
    },
    "BN254": {
        "input": [0, 1, 2],
        "output": [
            0xdc30ccd5d64e5bea071e99087ef86d433eb156aa0500a823298f9bb05328bd2,
            0x189893368d5815608c56e44cc67f7e821e093bb6254a0553f9ff69f4d99debc8,
            0x1acafc768221448ebc51fa2cd1e3c9b2044a0c04f3509d833b0a82c7e3462610,
        ],
    },
    "ST": {
        "input": [0, 1, 2],
        "output": [
            0x0b20c5d0d501d2b597b193e377c29d9fbc9185ee63c85055ec47b60f18fde12,
            0x067e87512fd2301814777307dbec5aa04bfecc351d9635c76729bc961d81a6d,
            0x03764f69a9a92eca9b3f96043cfaa5855a0e38d61fef60a46308ef93d62954e2,
        ],
    },
}

GOLDILOCKS_T8_KAT = {
    "input": list(range(8)),
    "output": [
        0x78611c23bb3f3511, 0x747ca7c6adfb6053, 0x72bab842bedc7f2b, 0xff382886d0643ff1,
        0x53364e0ade11b65c, 0xdd7d94314e8b2d24, 0x70f59074a73ebd6f, 0x115d7141e8c75cdd,
    ],
}

GOLDILOCKS_T12_KAT = {
    "input": list(range(12)),
    "output": [
        0xccd94518a9af0782, 0xf7ae608ea3308620, 0xf56dd53fae1f5876, 0x11e7b12aedd8ca86,
        0x869f9c3f93cd5630, 0x6ffe37312e58ac20, 0xac42b1f88aa27570, 0x312f6b96f7611c8a,
        0xf8b19bd51a741b7e, 0x9d1c158cfa1b7a12, 0x62ae69ae877e1e51, 0xce62641553ffe1bc,
    ],
}

INSTANCES = [
    ("BLS12", RESCUE_PRIME_BLS12_T3),
    ("BN254", RESCUE_PRIME_BN254_T3),
    ("ST", RESCUE_PRIME_ST_T3),
]

# ---------------------------------------------------------------------------
# Round count
# ---------------------------------------------------------------------------

def test_init_rounds():
    assert RESCUE_PRIME_BLS12_T3.R == 14
    assert RESCUE_PRIME_BN254_T3.R == 14
    assert RESCUE_PRIME_ST_T3.R == 18
    assert RESCUE_PRIME_GOLDILOCKS_T8.R == 8
    assert RESCUE_PRIME_GOLDILOCKS_T12.R == 8


# ---------------------------------------------------------------------------
# t=3 tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES)
def test_t3_permutation_kat(name, params):
    rp = RescuePrime(params)
    kat = T3_KATS[name]
    inp = [rp.to_field(x) for x in kat["input"]]
    out = rp.permutation(inp)
    assert [rp.from_field(x) for x in out] == kat["output"]


@pytest.mark.parametrize("name,params", INSTANCES)
def test_t3_permutation_deterministic(name, params):
    rp = RescuePrime(params)
    inp = [rp.F.random_element() for _ in range(rp.t)]
    assert rp.permutation(inp) == rp.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES)
def test_t3_permutation_distinct_inputs(name, params):
    rp = RescuePrime(params)
    inp1 = [rp.F.random_element() for _ in range(rp.t)]
    inp2 = [rp.F.random_element() for _ in range(rp.t)]
    assert rp.permutation(inp1) != rp.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES)
def test_t3_hash_output_size(name, params):
    rp = RescuePrime(params)
    data = [rp.F.random_element() for _ in range(rp.r * 2)]
    assert len(rp.hash_sponge(data)) == rp.r


@pytest.mark.parametrize("name,params", INSTANCES)
def test_t3_permutation_roundtrip(name, params):
    rp = RescuePrime(params)
    inp = [rp.F.random_element() for _ in range(rp.t)]
    assert rp.permutation_inv(rp.permutation(inp)) == inp


# ---------------------------------------------------------------------------
# Goldilocks T=8 / T=12 tests
# ---------------------------------------------------------------------------

def test_goldilocks_t8_permutation_kat():
    rp = RescuePrime(RESCUE_PRIME_GOLDILOCKS_T8)
    inp = [rp.to_field(x) for x in GOLDILOCKS_T8_KAT["input"]]
    out = rp.permutation(inp)
    assert [rp.from_field(x) for x in out] == GOLDILOCKS_T8_KAT["output"]


def test_goldilocks_t12_permutation_kat():
    rp = RescuePrime(RESCUE_PRIME_GOLDILOCKS_T12)
    inp = [rp.to_field(x) for x in GOLDILOCKS_T12_KAT["input"]]
    out = rp.permutation(inp)
    assert [rp.from_field(x) for x in out] == GOLDILOCKS_T12_KAT["output"]


def test_goldilocks_t8_permutation_roundtrip():
    rp = RescuePrime(RESCUE_PRIME_GOLDILOCKS_T8)
    inp = [rp.F.random_element() for _ in range(rp.t)]
    assert rp.permutation_inv(rp.permutation(inp)) == inp


def test_goldilocks_t12_permutation_roundtrip():
    rp = RescuePrime(RESCUE_PRIME_GOLDILOCKS_T12)
    inp = [rp.F.random_element() for _ in range(rp.t)]
    assert rp.permutation_inv(rp.permutation(inp)) == inp

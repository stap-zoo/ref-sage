import pytest

from hades.hash import Poseidon
from hades.params import PoseidonParams
from hades.instances import (
    POSEIDON_BN254_T3,
    POSEIDON_BLS12_T3,
    POSEIDON_BLS12_T2,
    POSEIDON_ST_T3,
    POSEIDON_GOLDILOCKS_T8,
    POSEIDON_GOLDILOCKS_T12,
    POSEIDON_MERSENNE_T16,
    POSEIDON_MERSENNE_T24,
)
from fields import BN254_SCALAR
from hades_reference import POSEIDON as REF, CIRCOM_BN254_T3_RC_ROW0

INSTANCES = [
    ("BN254_T3",       POSEIDON_BN254_T3),
    ("BLS12_T3",       POSEIDON_BLS12_T3),
    ("BLS12_T2",       POSEIDON_BLS12_T2),
    ("ST_T3",          POSEIDON_ST_T3),
    ("GOLDILOCKS_T8",  POSEIDON_GOLDILOCKS_T8),
    ("GOLDILOCKS_T12", POSEIDON_GOLDILOCKS_T12),
    ("MERSENNE_T16",   POSEIDON_MERSENNE_T16),
    ("MERSENNE_T24",   POSEIDON_MERSENNE_T24),
]
IDS = [name for name, _ in INSTANCES]
PARAMS = dict(INSTANCES)

# ---------------------------------------------------------------------------
# Known-answer test vectors (from the upstream Rust reference permutation() tests,
# https://extgit.isec.tugraz.at/krypto/zkfriendlyhashzoo)
# ---------------------------------------------------------------------------

KATS = {
    "BN254_T3": {
        "input": [0, 1, 2],
        "output": [
            0x2e72c60509a284872f62830b58ed8524a58c362dd3ddb98b2767f36b566596bd,
            0x180a812301272545f79ae1012b0425162a1833ac39101e070732f4d8a8bc4718,
            0x1828343d70eed99aae404e3ea58209f45743f3d54983fe250ce1526a9d8cf88e,
        ],
    },
    "BLS12_T3": {
        "input": [0, 1, 2],
        "output": [
            0x22a1f1595d99e4a04fc0a5b16be51a844a7cb5b5d69627ebbd1ee8142e7532ce,
            0x5f7ae6d6c380c90510de9c045ee75163eae24054ba8cd88d254cd1c343f43176,
            0x1b7e4da7d1ac6accb2e0470a83ba87d7bb585f4ba8c9a34f936faf3b3dfc695b,
        ],
    },
    "GOLDILOCKS_T12": {
        "input": list(range(12)),
        "output": [
            0xe9ad770762f48ef5, 0xc12796961ddc7859, 0xa61b71de9595e016, 0xead9e6aa583aafa3,
            0x93e297beff76e95b, 0x53abd3c5c2a0e924, 0xf3bc50e655c74f51, 0x246cac41b9a45d84,
            0xcc7f9314b2341f4f, 0xf5f071587c83415c, 0x09486cf35116fba3, 0x9d82aaf136b5c38a,
        ],
    },
}


@pytest.mark.parametrize("name", list(KATS), ids=list(KATS))
def test_permutation_kat(name):
    p = Poseidon(PARAMS[name])
    inp = [p.to_field(x) for x in KATS[name]["input"]]
    out = p.permutation(inp)
    assert [int(p.from_field(x)) for x in out] == KATS[name]["output"]


@pytest.mark.parametrize("name", list(REF), ids=list(REF))
def test_generated_matches_reference(name):
    """Round constants and MDS matrix derived by the "iaik"/"circom" strategies reproduce the
    upstream reference constants exactly (validates LFSRFieldElementSampler + the sampled cauchy_mds_matrix)."""
    params = PARAMS[name]
    rc = [int(params.from_field(x)) for row in params.rcons for x in row]
    mds = [int(params.from_field(x)) for row in params.M for x in row]
    assert rc == REF[name]["rc"]
    assert mds == REF[name]["mds"]


def test_circom_round_constants():
    """The "circom" strategy reproduces the published iden3/circomlib constants
    (https://github.com/iden3/circomlib) for BN254, t=3."""
    params = PoseidonParams(p=BN254_SCALAR.p, t=3, alpha=5, R_ext=8, R_int=57, r=2, c=1, d=1, version="circom")
    row0 = [int(params.from_field(x)) for x in params.rcons[0]]
    assert row0 == CIRCOM_BN254_T3_RC_ROW0


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_roundtrip(name, params):
    p = Poseidon(params)
    inp = [p.F.random_element() for _ in range(p.t)]
    assert p.permutation_inv(p.permutation(inp)) == inp
    assert p.permutation(p.permutation_inv(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_deterministic(name, params):
    p = Poseidon(params)
    inp = [p.F.random_element() for _ in range(p.t)]
    assert p.permutation(inp) == p.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_distinct_inputs(name, params):
    p = Poseidon(params)
    inp1 = [p.F.random_element() for _ in range(p.t)]
    inp2 = [p.F.random_element() for _ in range(p.t)]
    while inp1 == inp2:
        inp2 = [p.F.random_element() for _ in range(p.t)]
    assert p.permutation(inp1) != p.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_sponge_output_size(name, params):
    p = Poseidon(params)
    data = [p.F.random_element() for _ in range(p.r * 3)]
    assert len(p.hash_sponge(data)) == p.d

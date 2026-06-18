# test_poseidon.py
# ---------------------------------------------------------------------------
# Test suite for Poseidon, parametrized over the named instances in instances.py.
#
# Groups:
#   4.1 KATs         -- fixed input/output vectors (permutation)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- generated constants/MDS match the reference
#   4.5 Misc         -- validation/errors
# ---------------------------------------------------------------------------

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
from utils.field import BN254_SCALAR
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
# 4.1 Known-answer test vectors (from the upstream Rust reference permutation() tests,
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
    prim = Poseidon(PARAMS[name])
    inp = [prim.to_field(x) for x in KATS[name]["input"]]
    out = prim.permutation(inp)
    assert [int(prim.from_field(x)) for x in out] == KATS[name]["output"]


# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_roundtrip(name, params):
    prim = Poseidon(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp
    assert prim.permutation(prim.permutation_inv(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner. Use a representative
    # external and internal round index.
    prim = Poseidon(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    ext_idx, int_idx = 0, prim.R_ext_beg
    for r in [ext_idx, int_idx]:
        assert prim.nonlinear_layer_inv(prim.nonlinear_layer(inp, r), r) == inp
        assert prim.linear_layer_inv(prim.linear_layer(inp, r), r) == inp
        assert prim.constant_addition_inv(prim.constant_addition(inp, r), r) == inp
    assert prim._pre_rounds_inv(prim._pre_rounds(inp)) == inp
    assert prim._post_rounds_inv(prim._post_rounds(inp)) == inp


# ---------------------------------------------------------------------------
# 4.3 Consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_deterministic(name, params):
    prim = Poseidon(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_distinct_inputs(name, params):
    prim = Poseidon(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_sponge_output_size(name, params):
    prim = Poseidon(params)
    #data = [prim.F.random_element() for _ in range(prim.r * 3)] # TODO implement variable length sponge or catch exception
    data = [prim.F.random_element() for _ in range(prim.r)]
    assert len(prim.hash_sponge(data)) == prim.d

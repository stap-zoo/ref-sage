import pytest

from hades.hash import Poseidon2
from hades.instances import (
    POSEIDON2_BLS12_T2,
    POSEIDON2_BLS12_T3,
    POSEIDON2_BLS12_T4,
    POSEIDON2_BLS12_T8,
    POSEIDON2_BN254_T3,
    POSEIDON2_GOLDILOCKS_T8,
    POSEIDON2_GOLDILOCKS_T12,
    POSEIDON2_GOLDILOCKS_T16,
    POSEIDON2_GOLDILOCKS_T20,
    POSEIDON2_MERSENNE_T16,
    POSEIDON2_MERSENNE_T24,
)
from hades_reference import POSEIDON2 as REF

INSTANCES = [
    ("BLS12_T2",       POSEIDON2_BLS12_T2),
    ("BLS12_T3",       POSEIDON2_BLS12_T3),
    ("BLS12_T4",       POSEIDON2_BLS12_T4),
    ("BLS12_T8",       POSEIDON2_BLS12_T8),
    ("BN254_T3",       POSEIDON2_BN254_T3),
    ("GOLDILOCKS_T8",  POSEIDON2_GOLDILOCKS_T8),
    ("GOLDILOCKS_T12", POSEIDON2_GOLDILOCKS_T12),
    ("GOLDILOCKS_T16", POSEIDON2_GOLDILOCKS_T16),
    ("GOLDILOCKS_T20", POSEIDON2_GOLDILOCKS_T20),
    ("MERSENNE_T16",   POSEIDON2_MERSENNE_T16),
    ("MERSENNE_T24",   POSEIDON2_MERSENNE_T24),
]
IDS = [name for name, _ in INSTANCES]
PARAMS = dict(INSTANCES)

# Known-answer vectors from the upstream HorizenLabs reference permutation() tests
# (https://github.com/HorizenLabs/poseidon2).
KATS = {
    "GOLDILOCKS_T12": {
        "input": list(range(12)),
        "output": [
            0xed3dbcc4ff1e8d33, 0xfb85eac6ac91a150, 0xd41e1e237ed3e2ef, 0x5e289bf0a4c11897,
            0x4398b20f93e3ba6b, 0x5659a48ffaf2901d, 0xe44d81e89a88f8ae, 0x08efdb285f8c3dbc,
            0x294ab7503297850e, 0xa11c61f4870b9904, 0xa6855c112cc08968, 0x17c6d53d2fb3e8c1,
        ],
    },
    "BLS12_T2": {
        "input": [0, 1],
        "output": [
            0x50f38c87fbf14be6e91d0d911b52dc8c1b19fe439348c427514a8b59bdf92f62,
            0x3222c2d9d80f8be5aff518685e66ae4648cc76243d1ca077101bebb2ee245d30,
        ],
    },
    "BLS12_T3": {
        "input": [0, 1, 2],
        "output": [
            0x562af4b3710cdba6cea53e1f73325b21bb97ac810943b74d863d87163ee8042e,
            0x4674eba4cef166510c0d7a9ddf08cf813637bc2081e2c40c5047dce7ecdf2b95,
            0x0cf55ec35287dca6195eb6dd43e9ac1aba8857b4d3e4501be8bd8e9946a8dc54,
        ],
    },
}


@pytest.mark.parametrize("name", list(KATS), ids=list(KATS))
def test_permutation_kat(name):
    prim = Poseidon2(PARAMS[name])
    inp = [prim.to_field(x) for x in KATS[name]["input"]]
    out = prim.permutation(inp)
    assert [int(prim.from_field(x)) for x in out] == KATS[name]["output"]


@pytest.mark.parametrize("name", list(REF), ids=list(REF))
def test_generated_matches_reference(name):
    """Generated round constants (external rounds full width, internal rounds branch 0 only)
    and internal matrix M_I = J + diag(mat_diag) reproduce the upstream reference constants."""
    params = PARAMS[name]
    rc = [int(params.from_field(x)) for row in params.rcons for x in row]
    mi = [int(params.from_field(x)) for row in params.M_int for x in row]
    assert rc == REF[name]["rc"]
    assert mi == REF[name]["mat_internal"]


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_roundtrip(name, params):
    prim = Poseidon2(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp
    assert prim.permutation(prim.permutation_inv(inp)) == inp

@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner. Use a representative
    # external and internal round index.
    prim = Poseidon2(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    ext_idx, int_idx = 0, prim.R_ext_beg
    for r in [ext_idx, int_idx]:
        assert prim.nonlinear_layer_inv(prim.nonlinear_layer(inp, r), r) == inp
        assert prim.linear_layer_inv(prim.linear_layer(inp, r), r) == inp
        assert prim.constant_addition_inv(prim.constant_addition(inp, r), r) == inp
    assert prim._pre_rounds_inv(prim._pre_rounds(inp)) == inp
    assert prim._post_rounds_inv(prim._post_rounds(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_deterministic(name, params):
    prim = Poseidon2(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_distinct_inputs(name, params):
    prim = Poseidon2(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_sponge_output_size(name, params):
    prim = Poseidon2(params)
    #data = [prim.F.random_element() for _ in range(prim.r * 3)] # TODO implement variable length sponge or catch exception
    data = [prim.F.random_element() for _ in range(prim.r)]
    assert len(prim.hash_sponge(data)) == prim.d

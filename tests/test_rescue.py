# test_rescue.py
# ---------------------------------------------------------------------------
# Test suite for Rescue, parametrized over the named instances in instances.py.
#
# Groups:
#   4.1 KATs         -- fixed input/output vectors (permutation + sponge)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- MDS matrix / round-constant generation
#   4.5 Misc         -- validation/errors
# ---------------------------------------------------------------------------

import pytest

from marvellous.hash import Rescue
from marvellous.instances import (
    RESCUE_BLS12_T3,
    RESCUE_BN254_T3,
    RESCUE_ST_T3,
    RESCUE_GOLDILOCKS_T12,
)

from utils import vandermonde_mds_matrix, XOFFieldElementSampler
from fields import BLS12_381_SCALAR, GOLDILOCKS

INSTANCES = [
    ("BLS12_T3", RESCUE_BLS12_T3),
    ("BN254_T3", RESCUE_BN254_T3),
    ("ST_T3", RESCUE_ST_T3),
    ("GOLDILOCKS_T12", RESCUE_GOLDILOCKS_T12),
]

# ---------------------------------------------------------------------------
# 4.1 Known-answer test vectors (from the Sage reference implementation)
# https://github.com/KULeuven-COSIC/Marvellous/blob/master/instance_generatoprim.sage
# ---------------------------------------------------------------------------

PERMUTATION_KATS = {
    "BLS12_T3": [
        {
            "input": [0, 1, 2],
            "output": [
                0x3a6da171e7d612f45c04bff4fb100efd6d85fbbdc78b49872947ca7c5be9e87a,
                0x65843d0bfa54f9891aedd014bed810acb9a7c9613c724855b312a568d9fa3b7a,
                0x6a724ef437d9280246325184cfa3844a90553cf5b01484a170a5d92a996e4f06,
            ],
        },
    ],
    "BN254_T3": [
        {
            "input": [0, 1, 2],
            "output": [
                0x2262a88c8b641065446e84e3fa132210312c076e9b056c972986dbc775c42e89,
                0x19185415565c3847d80a72e8341b1ddbeb10d8f119a2132e127370e34b6b312e,
                0x16c41359b114c56cf7b00c6a22de88a9bac5e1d5957cf02790a75a66d4566060,
            ],
        },
    ],
    "ST_T3": [
        {
            "input": [0, 1, 2],
            "output": [
                0x02778ceedfc5dca26a811aed72ce59f5a6721fc65f5f61d2dd8fbd333b8a6ebb,
                0x039c581707ec0a5ed908301088b8ebbd1ea815c8db2368d64101b7306272c45f,
                0x018bba1c61eedababe3d7c81ba1ee000b514c4400dd9a23bd3440765da3bfb09,
            ],
        },
    ],
    "GOLDILOCKS_T12": [
        {
            "input": list(range(12)),
            "output": [
                0xee199b1c0c2ec165, 0x0d3003cfd2966617, 0x6bea04edea2f6628, 0x58cc421aac1c5ad5,
                0x614a0cd7ae5aba77, 0xbf5843fd7b332792, 0x4bb813cbbc4b829f, 0x204ffe8417eeb6e8,
                0x0730a4af90216d62, 0x69e9b9704fb47cc9, 0x5b046dcd07061b36, 0x8c30ff6723b16945,
            ],
        },
    ],
}

SPONGE_KATS = {
    "GOLDILOCKS_T12": [
        {
            "input": list(range(11)),
            "output": [
                0x3232bbf92f36b4ca, 0x4dc27ad23dfee93b, 0xd7f07aeea8a9c222, 0xf8654c877d2cf694,
                0x424fba831e07ca3a, 0xd8d8e2e0687dd981, 0x3b6d6919cd9a7e8a, 0xac1a59ad3b320ba7,
                0xa10e69ab735edbd3, 0x4e9d93a7ba766427, 0xbde43d3fbc71eba4,
            ],
        },
    ],
}

PERMUTATION_KAT_CASES = [
    (name, params, kat)
    for name, params in INSTANCES
    for kat in PERMUTATION_KATS[name]
]
PERMUTATION_KAT_IDS = [
    name if len(PERMUTATION_KATS[name]) == 1 else f"{name}-{i}"
    for name, params in INSTANCES
    for i, _ in enumerate(PERMUTATION_KATS[name])
]

SPONGE_KAT_CASES = [
    (name, params, kat)
    for name, params in INSTANCES
    for kat in SPONGE_KATS.get(name, [])
]
SPONGE_KAT_IDS = [
    name if len(SPONGE_KATS[name]) == 1 else f"{name}-{i}"
    for name, params in INSTANCES
    for i, _ in enumerate(SPONGE_KATS.get(name, []))
]

# --- 4.1 (cont.) Round-constant KAT ---

# RESCUE_ST_PARAMS.round_constants[0], from rescue/rust/rescue/rescue_instance_st.rs (RC3[0])
RESCUE_RC3_ST_FIRST = [
    0x02a44b57f6e9b0e2e8b817dac3698c80fb839be67095dff247fa5c7e9dbcedcf,
    0x011a45e31139695c94ba33cd73c90040fd0c31b012d02fd1501a72b3f9001ade,
    0x02486100be21d1eba5a331ff3b7a3a85698329ed00647810bf7c9ce76060c905,
]


def test_st_round_constants_kat():
    assert [RESCUE_ST_T3.from_field(x) for x in RESCUE_ST_T3.rcons[0]] == RESCUE_RC3_ST_FIRST


# --- 4.1 (cont.) Permutation and sponge KATs ---

@pytest.mark.parametrize("name,params,kat", PERMUTATION_KAT_CASES, ids=PERMUTATION_KAT_IDS)
def test_permutation_kat(name, params, kat):
    prim = Rescue(params)
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.permutation(inp)
    assert [prim.from_field(x) for x in out] == kat["output"]


@pytest.mark.parametrize("name,params,kat", SPONGE_KAT_CASES, ids=SPONGE_KAT_IDS)
def test_sponge_kat(name, params, kat):
    prim = Rescue(params)
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.hash_sponge(inp, variable_length=False)
    assert [prim.from_field(x) for x in out] == kat["output"]


# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    prim = Rescue(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partneprim.
    prim = Rescue(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    for r in range(2 * prim.R):
        assert prim.constant_addition_inv(prim.constant_addition(inp, r), r) == inp
        assert prim.linear_layer_inv(prim.linear_layer(inp, r), r) == inp
        assert prim.nonlinear_layer_inv(prim.nonlinear_layer(inp, r), r) == inp
    assert prim._pre_rounds_inv(prim._pre_rounds(inp)) == inp
    assert prim._post_rounds_inv(prim._post_rounds(inp)) == inp


# ---------------------------------------------------------------------------
# 4.3 Consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    prim = Rescue(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    prim = Rescue(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_sponge_output_size(name, params):
    prim = Rescue(params)
    #data = [prim.F.random_element() for _ in range(prim.r * 3)] # TODO implement variable length sponge or catch exception
    data = [prim.F.random_element() for _ in range(prim.r // 2)]
    assert len(prim.hash_sponge(data)) == prim.d


# ---------------------------------------------------------------------------
# 4.4 Algebraic: parameter generation
# ---------------------------------------------------------------------------

def test_vandermonde_mds_matrix_bls12_rescue():
    M = vandermonde_mds_matrix(BLS12_381_SCALAR.p, 3, BLS12_381_SCALAR.generator, transpose=False)
    expected = [[RESCUE_BLS12_T3.from_field(x) for x in row] for row in RESCUE_BLS12_T3.M]
    assert M == expected


# ---------------------------------------------------------------------------
# 4.5 Misc: validation
# ---------------------------------------------------------------------------

def test_invalid_state_size():
    prim = Rescue(RESCUE_BLS12_T3)
    with pytest.raises(ValueError):
        prim.permutation([prim.F.zero()] * (prim.t + 1))

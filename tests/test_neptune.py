import pytest

from hades.hash import Neptune
from hades.instances import (
    NEPTUNE_BN254_T4,
    NEPTUNE_BLS12_T4,
    NEPTUNE_BLS12_T2,
    NEPTUNE_ST_T4,
    NEPTUNE_GOLDILOCKS_T8,
    NEPTUNE_GOLDILOCKS_T12,
)

INSTANCES = [
    ("BN254_T4",       NEPTUNE_BN254_T4),
    ("BLS12_T4",       NEPTUNE_BLS12_T4),
    ("BLS12_T2",       NEPTUNE_BLS12_T2),
    ("ST_T4",          NEPTUNE_ST_T4),
    ("GOLDILOCKS_T8",  NEPTUNE_GOLDILOCKS_T8),
    ("GOLDILOCKS_T12", NEPTUNE_GOLDILOCKS_T12),
]
IDS = [name for name, _ in INSTANCES]

# ---------------------------------------------------------------------------
# Known-answer vectors.
# The Neptune reference (IAIK zk-friendly-hash-zoo,
# https://extgit.isec.tugraz.at/krypto/zkfriendlyhashzoo) ships no permutation KATs and derives
# all constants from SHAKE128 at runtime; these vectors are self-derived (input = [0, ..., t-1])
# and serve as regression anchors. Correctness rests on the structural checks below, which mirror
# the reference's own matmul_equalities / build_mi tests and its external-S-box definition, plus
# the roundtrip test against the analytically derived inverse.
# ---------------------------------------------------------------------------

KATS = {
    "BLS12_T4": [
        0x46f98d9b2baea2ab33081ff8039081df4ec56124b75ca074a2398facb9c8c960,
        0x2f115b88b6f3581e1355d4f9f721cef8c1a76f3378f5daaf388bc83056823e6b,
        0x0b2200ca2987f209cea71a744515554bb7ecaddcd933dd43d02187aaf0bab2c2,
        0x124af253a8d98cf4a597daabc6f6e091b731f589a67921461a73b52846eaa9eb,
    ],
    "GOLDILOCKS_T8": [
        0xfcb7ba304be25038, 0x0a5a73194390d17c, 0x8abc1f0b13d091e3, 0xbb1133cb41dbed00,
        0x18b8f4187427e45d, 0x16cb835a452ddbce, 0x4540254fd8105059, 0xacf6ede72c4578cf,
    ],
    "GOLDILOCKS_T12": [
        0xbab8e46b6ff6682d, 0x6600d1b598b67767, 0x3386b1d0d66cc351, 0xb0c6f65f89e86e16,
        0x7fa447c3314c2977, 0xe77e4584e08e0c21, 0xdf5be09996a589b9, 0xb3c93376b340c02b,
        0x2574e8ac7013c1cd, 0x597afc22783e2ba8, 0x71936a7b28e8474f, 0xf455d814a657ee7d,
    ],
}


@pytest.mark.parametrize("name", list(KATS), ids=list(KATS))
def test_permutation_kat(name):
    params = dict(INSTANCES)[name]
    prim = Neptune(params)
    out = prim.permutation([prim.to_field(i) for i in range(prim.t)])
    assert [int(prim.from_field(x)) for x in out] == KATS[name]


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_external_matrix_is_split(name, params):
    """M_E only couples even-even and odd-odd branches (entry zero iff row+col is odd),
    matching the Neptune reference (https://extgit.isec.tugraz.at/krypto/zkfriendlyhashzoo)."""
    t = params.t
    for i in range(t):
        for j in range(t):
            is_zero = int(params.from_field(params.M_ext[i][j])) == 0
            assert is_zero == ((i + j) % 2 == 1)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_internal_matrix_is_j_plus_diag(name, params):
    """M_I = J + diag(mu - 1): every off-diagonal entry is 1 (build_mi)."""
    t = params.t
    for i in range(t):
        for j in range(t):
            if i != j:
                assert int(params.from_field(params.M_int[i][j])) == 1


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_roundtrip(name, params):
    prim = Neptune(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp
    assert prim.permutation(prim.permutation_inv(inp)) == inp

@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner. Use a representative
    # external and internal round index.
    prim = Neptune(params)
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
    prim = Neptune(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_distinct_inputs(name, params):
    prim = Neptune(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_sponge_output_size(name, params):
    prim = Neptune(params)
    #data = [prim.F.random_element() for _ in range(prim.r * 3)] # TODO implement variable length sponge or catch exception
    data = [prim.F.random_element() for _ in range(prim.r)]
    assert len(prim.hash_sponge(data)) == prim.d

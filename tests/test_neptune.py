# test_neptune.py
# ---------------------------------------------------------------------------
# Test suite for Neptune, parametrized over the named instances in
# hades/instances.py so every recommended instance is covered by the same checks.
#
# Groups:
#   4.1 KATs         -- self-derived regression vectors (see note below)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- structural matrix checks mirroring the reference's tests
#   4.5 Misc         -- validation/errors, warnings, reproducibility
# ---------------------------------------------------------------------------

import warnings

import pytest

from hades.hash import Neptune
from hades.params import NeptuneParams
from hades.instances import (
    NEPTUNE_BN254_T4,
    NEPTUNE_BLS12_T4,
    NEPTUNE_BLS12_T2,
    NEPTUNE_ST_T4,
    NEPTUNE_GOLDILOCKS_T8,
    NEPTUNE_GOLDILOCKS_T12,
)
from recommendations import ParamRecommendationWarning

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
# 4.1 Known-answer tests
#
# The Neptune reference (IAIK zk-friendly-hash-zoo,
# https://extgit.isec.tugraz.at/krypto/zkfriendlyhashzoo) ships no permutation KATs and derives
# all constants from SHAKE128 at runtime; these vectors are self-derived (input = [0, ..., t-1])
# and serve as regression anchors. Correctness rests on the structural checks below, which mirror
# the reference's own matmul_equalities / build_mi tests and its external-S-box definition, plus
# the roundtrip test against the analytically derived inverse.
# ---------------------------------------------------------------------------

KATS = {
    "BLS12_T4": [
        {"input": [0, 1, 2, 3], "output": [
            0x46f98d9b2baea2ab33081ff8039081df4ec56124b75ca074a2398facb9c8c960,
            0x2f115b88b6f3581e1355d4f9f721cef8c1a76f3378f5daaf388bc83056823e6b,
            0x0b2200ca2987f209cea71a744515554bb7ecaddcd933dd43d02187aaf0bab2c2,
            0x124af253a8d98cf4a597daabc6f6e091b731f589a67921461a73b52846eaa9eb,
        ]},
    ],
    "GOLDILOCKS_T8": [
        {"input": [0, 1, 2, 3, 4, 5, 6, 7], "output": [
            0xfcb7ba304be25038, 0x0a5a73194390d17c, 0x8abc1f0b13d091e3, 0xbb1133cb41dbed00,
            0x18b8f4187427e45d, 0x16cb835a452ddbce, 0x4540254fd8105059, 0xacf6ede72c4578cf,
        ]},
    ],
    "GOLDILOCKS_T12": [
        {"input": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "output": [
            0xbab8e46b6ff6682d, 0x6600d1b598b67767, 0x3386b1d0d66cc351, 0xb0c6f65f89e86e16,
            0x7fa447c3314c2977, 0xe77e4584e08e0c21, 0xdf5be09996a589b9, 0xb3c93376b340c02b,
            0x2574e8ac7013c1cd, 0x597afc22783e2ba8, 0x71936a7b28e8474f, 0xf455d814a657ee7d,
        ]},
    ],
}

KAT_CASES = [
    (name, params, kat)
    for name, params in INSTANCES
    for kat in KATS.get(name, [])
]
KAT_IDS = [
    name if len(KATS.get(name, [])) == 1 else f"{name}-{i}"
    for name, params in INSTANCES
    for i, _ in enumerate(KATS.get(name, []))
]


@pytest.mark.parametrize("name,params,kat", KAT_CASES, ids=KAT_IDS)
def test_permutation_kat(name, params, kat):
    prim = Neptune(params)
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.permutation(inp)
    assert [int(prim.from_field(x)) for x in out] == kat["output"]


# ---------------------------------------------------------------------------
# 4.4 Algebraic: structural matrix checks
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4.3 Consistency
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4.5 Misc: validation, warnings, reproducibility
# ---------------------------------------------------------------------------

def test_invalid_state_size():
    prim = Neptune(NEPTUNE_GOLDILOCKS_T8)
    with pytest.raises(ValueError):
        prim.permutation([prim.F.zero()] * (prim.t + 1))


def test_odd_state_size_rejected():
    with pytest.raises(ValueError):
        NeptuneParams(p=NEPTUNE_GOLDILOCKS_T8.p, t=3, alpha=7, R_ext=2, R_int=2, r=2, c=1, d=1)


def test_toy_field_warns():
    with pytest.warns(ParamRecommendationWarning):
        NeptuneParams(p=101, t=4, alpha=3, R_ext=2, R_int=2, r=2, c=2, d=2, toy=True)  # tiny field


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_recommended_instance_no_warning(name, params):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParamRecommendationWarning)
        NeptuneParams(p=params.p, t=params.t, alpha=params.alpha,
                      R_ext=params.R_ext, R_int=params.R_int,
                      r=params.r, c=params.c, d=params.d)


def test_constants_reproducible():
    # Same parameters -> identical derived constants and matrices.
    kwargs = dict(p=NEPTUNE_GOLDILOCKS_T8.p, t=8, alpha=7, R_ext=6, R_int=38, r=4, c=4, d=4)
    a = NeptuneParams(**kwargs)
    b = NeptuneParams(**kwargs)
    assert a.rcons == b.rcons
    assert a.M_ext == b.M_ext
    assert a.M_int == b.M_int

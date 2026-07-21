# test_psquarehash.py
# ---------------------------------------------------------------------------
# Test suite for pSquare-hash, parametrized over the named instances in instances.py.
#
# Groups:
#   4.1 KATs         -- fixed input/output vectors (permutation)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- derivation vs. pinned instance
#   4.5 Misc         -- validation/errors, warnings, compression availability
# ---------------------------------------------------------------------------

import warnings

import pytest

from psquarehash.hash import pSquareHash
from psquarehash.params import pSquareHashParams
from psquarehash.instances import (
    PSQUAREHASH_MERSENNE_T16,
    PSQUAREHASH_MERSENNE_T24,
)
from recommendations import ParamRecommendationWarning

INSTANCES = [
    ("M31_T16", PSQUAREHASH_MERSENNE_T16),
    ("M31_T24", PSQUAREHASH_MERSENNE_T24),
]

# Instances where t == 2*d, so compress_2_to_1 is defined
COMPRESS_INSTANCES = [
    ("M31_T16", PSQUAREHASH_MERSENNE_T16),
]

# Instances where t != 2*d, so compress_2_to_1 raises ValueError
NO_COMPRESS_INSTANCES = [
    ("M31_T24", PSQUAREHASH_MERSENNE_T24),
]

# ---------------------------------------------------------------------------
# Known-answer test vectors (from Rust reference implementation)
# ---------------------------------------------------------------------------

KATS = {
    "M31_T16": [
        {
            "input": [
                0x78066d6b, 0x68a24eb4, 0x2d12aacd, 0x42bb7df4,
                0x3a85ecf4, 0x010084b5, 0x28e3f4fb, 0x41514a49,
                0x0e904f42, 0x0981bfd9, 0x3309b9ac, 0x19f408ff,
                0x1f3202d0, 0x2ebbcc8c, 0x261b659f, 0x22171a32,
            ],
            "output": [
                0x0d31650f, 0x5b324b40, 0x02fb8ac7, 0x555c9139,
                0x00a60cba, 0x1b61b003, 0x33e1c0ad, 0x48d970a2,
                0x0876e39d, 0x3a6f9513, 0x36afab87, 0x4d85ef87,
                0x277b1cee, 0x70debee3, 0x1337395b, 0x35ea5bad,
            ],
        },
    ],
    "M31_T24": [
        {
            "input": [
                0x78066d6b, 0x68a24eb4, 0x2d12aacd, 0x42bb7df4,
                0x3a85ecf4, 0x010084b5, 0x28e3f4fb, 0x41514a49,
                0x0e904f42, 0x0981bfd9, 0x3309b9ac, 0x19f408ff,
                0x1f3202d0, 0x2ebbcc8c, 0x261b659f, 0x22171a32,
                0x2b77fbfb, 0x57d3e692, 0x47dbb2c4, 0x5f803d52,
                0x7791f988, 0x6988c314, 0x283918dd, 0x32a8ab7b
            ],
            "output": [
                0x2e50ad42, 0x117b015c, 0x0c4610fb, 0x636c99be,
                0x3b2635cc, 0x15323786, 0x36ba41ac, 0x788cb4d8,
                0x0f7b751a, 0x7608c969, 0x6ff8eeda, 0x6ed27e30,
                0x55e7b993, 0x63506d14, 0x42032061, 0x31bbe5f5,
                0x7179d761, 0x5871965a, 0x16497d76, 0x2237878d,
                0x637ca7d2, 0x13752294, 0x0831c440, 0x18bf0647,
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


# ---------------------------------------------------------------------------
# 4.1 KAT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params,kat", KAT_CASES, ids=KAT_IDS)
def test_permutation_kat(name, params, kat):
    prim = pSquareHash(params)
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.permutation(inp)
    assert [prim.from_field(x) for x in out] == kat["output"]

# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    prim = pSquareHash(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner, for every round.
    prim = pSquareHash(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    for r in range(prim.R):
        assert prim.nonlinear_layer_inv(prim.nonlinear_layer(inp, r), r) == inp
        assert prim.linear_layer_inv(prim.linear_layer(inp, r), r) == inp
    assert prim._pre_rounds_inv(prim._pre_rounds(inp)) == inp
    assert prim._post_rounds_inv(prim._post_rounds(inp)) == inp

# ---------------------------------------------------------------------------
# 4.3 Consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    prim = pSquareHash(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    prim = pSquareHash(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_sponge_output_size(name, params):
    prim = pSquareHash(params)
    #data = [prim.F.random_element() for _ in range(prim.r * 3)]  # TODO implement variable length sponge or catch exception
    data = [prim.F.random_element() for _ in range(prim.r)] 
    assert len(prim.hash_sponge(data)) == prim.d


# ---------------------------------------------------------------------------
# 4.4 Misc: validation, compression availability
# ---------------------------------------------------------------------------

def test_invalid_state_size():
    prim = pSquareHash(PSQUAREHASH_MERSENNE_T16)
    with pytest.raises(ValueError):
        prim.permutation([prim.F.zero()] * (prim.t + 1))


@pytest.mark.parametrize("name,params", COMPRESS_INSTANCES, ids=[name for name, _ in COMPRESS_INSTANCES])
def test_compress_output_size(name, params):
    prim = pSquareHash(params)
    half = prim.t // 2
    x1 = [prim.F.random_element() for _ in range(half)]
    x2 = [prim.F.random_element() for _ in range(half)]
    assert len(prim.compress_2_to_1(x1, x2)) == prim.d


@pytest.mark.parametrize("name,params", NO_COMPRESS_INSTANCES, ids=[name for name, _ in NO_COMPRESS_INSTANCES])
def test_compress_not_defined(name, params):
    prim = pSquareHash(params)
    half = prim.t // 2
    x1 = [prim.F.random_element() for _ in range(half)]
    x2 = [prim.F.random_element() for _ in range(half)]
    with pytest.raises(ValueError):
        prim.compress_2_to_1(x1, x2)


# ---------------------------------------------------------------------------
# 4.4 Algebraic: derivation vs. pinned instance
# ---------------------------------------------------------------------------

def _instance_rcons_ints(params):
    return [[int(params.from_field(x)) for x in row] for row in params.rcons]


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_matrices_generated_match_instance(name, params):
    # The instances derive M and M_IO via _init_mat / _init_mat_IO; rebuilding
    # must reproduce them.
    derived = pSquareHashParams(p=params.p, t=params.t, R=params.R,
                                rcons=_instance_rcons_ints(params),
                                r=params.r, c=params.c, d=params.d)
    assert derived.M == params.M
    assert derived.M_IO == params.M_IO


@pytest.mark.skip(reason="_init_cons (SHAKE256 fallback) does not reproduce the reference constants pinned in instances.py; TODO: implement the paper's round-constant derivation")
@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_cons_generated_matches_instance(name, params):
    derived = pSquareHashParams(p=params.p, t=params.t, R=params.R,
                                r=params.r, c=params.c, d=params.d)  # rcons omitted -> _init_cons
    assert derived.rcons == params.rcons


@pytest.mark.skip(reason="_init_rounds is a stub (round number derivation not implemented for pSquare-hash)")
@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_rounds_derivation_matches_instance(name, params):
    derived = pSquareHashParams(p=params.p, t=params.t,
                                rcons=_instance_rcons_ints(params),
                                r=params.r, c=params.c, d=params.d)  # R omitted -> _init_rounds
    assert derived.R == params.R


# ---------------------------------------------------------------------------
# 4.5 (cont.) Warnings, reproducibility
# ---------------------------------------------------------------------------

def test_toy_field_warns():
    with pytest.warns(ParamRecommendationWarning):
        pSquareHashParams(p=8191, t=4, R=6, r=2, c=2, d=2, toy=True)  # tiny field


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_recommended_instance_no_warning(name, params):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParamRecommendationWarning)
        pSquareHashParams(p=params.p, t=params.t, R=params.R,
                          rcons=_instance_rcons_ints(params),
                          r=params.r, c=params.c, d=params.d)


def test_constants_reproducible():
    # Same parameters -> identical derived constants and matrices.
    kwargs = dict(p=PSQUAREHASH_MERSENNE_T16.p, t=16, R=52, r=8, c=8, d=8)
    a = pSquareHashParams(**kwargs)
    b = pSquareHashParams(**kwargs)
    assert a.rcons == b.rcons
    assert a.M == b.M
    assert a.M_IO == b.M_IO

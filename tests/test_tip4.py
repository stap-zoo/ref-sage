# test_tip4.py
# ---------------------------------------------------------------------------
# Test suite for Tip4, parametrized over the named instance in instances.py.
# Tip4 shares the Tip5 permutation (t=16, R=5); only the sponge parameters differ.
#
# Groups:
#   4.1 KATs         -- fixed input/output vectors (permutation + hash)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- lookup table / matrix generation
#   4.5 Misc         -- validation/errors
# ---------------------------------------------------------------------------

import warnings

import pytest

from tip5.hash import Tip4
from tip5.params import Tip4Params
from tip5.instances import TIP4, LOOKUP_TABLE
from recommendations import ParamRecommendationWarning

INSTANCES = [
    ("TIP4", TIP4),
]

# ---------------------------------------------------------------------------
# 4.1 Known-answer test vectors, generated with the sage implementation from
# https://github.com/isec-tugraz/ca-tip5family-monolith/blob/main/Tip5.sage
# ---------------------------------------------------------------------------

PERM_KATS = {
    "TIP4": [
        {
            "input": list(range(16)),
            "output": [
                14273019456630489802, 12225354657803044645, 18223679466392555512, 4879234115918641111,
                198243361942729835, 6697571774370475124, 3935892719377798608, 2781322532457452310,
                7475933807446249354, 7334965145562953054, 1275437117587945070, 2445375571864276273,
                17005006372293520413, 9537835648539327419, 12703602725074524970, 5428520427373770602,
            ],
        },
    ],
}

HASH_KATS = {
    "TIP4": [
        {
            "input": list(range(12)),
            "output": [
                15841355890359640929, 14241254185087876088, 7403730774983924736, 1993907345493395678,
            ],
        },
    ],
}

PERM_KAT_CASES = [
    (name, params, kat)
    for name, params in INSTANCES
    for kat in PERM_KATS[name]
]
PERM_KAT_IDS = [
    name if len(PERM_KATS[name]) == 1 else f"{name}-{i}"
    for name, params in INSTANCES
    for i, _ in enumerate(PERM_KATS[name])
]

HASH_KAT_CASES = [
    (name, params, kat)
    for name, params in INSTANCES
    for kat in HASH_KATS[name]
]
HASH_KAT_IDS = [
    name if len(HASH_KATS[name]) == 1 else f"{name}-{i}"
    for name, params in INSTANCES
    for i, _ in enumerate(HASH_KATS[name])
]


@pytest.mark.parametrize("name,params,kat", PERM_KAT_CASES, ids=PERM_KAT_IDS)
def test_permutation_kat(name, params, kat):
    prim = Tip4(params)
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.permutation(inp)
    assert [prim.from_field(x) for x in out] == kat["output"]


@pytest.mark.parametrize("name,params,kat", HASH_KAT_CASES, ids=HASH_KAT_IDS)
def test_hash_kat(name, params, kat):
    prim = Tip4(params)
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.hash_sponge(inp)
    assert [prim.from_field(x) for x in out] == kat["output"]


# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    prim = Tip4(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner, for every round.
    prim = Tip4(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    for r in range(prim.R):
        assert prim.linear_layer_inv(prim.linear_layer(inp, r), r) == inp
        assert prim.constant_addition_inv(prim.constant_addition(inp, r), r) == inp
        assert prim.nonlinear_layer_inv(prim.nonlinear_layer(inp, r), r) == inp
    assert prim._pre_rounds_inv(prim._pre_rounds(inp)) == inp
    assert prim._post_rounds_inv(prim._post_rounds(inp)) == inp


# ---------------------------------------------------------------------------
# 4.3 Consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    prim = Tip4(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    prim = Tip4(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_hash_output_size(name, params):
    prim = Tip4(params)
    data = [prim.F.random_element() for _ in range(prim.r)]
    assert len(prim.hash_sponge(data)) == prim.d


# ---------------------------------------------------------------------------
# 4.4 Algebraic: lookup table and matrix generation
# ---------------------------------------------------------------------------

def test_params_derive_constants():
    """Omitting LUT/M derives the values used by the hardcoded TIP4 instance."""
    derived = Tip4Params()
    assert TIP4.LUT == derived.LUT
    assert TIP4.M == derived.M


# ---------------------------------------------------------------------------
# 4.5 Misc: validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_hash_rejects_wrong_length(name, params):
    prim = Tip4(params)
    data = [prim.F.random_element() for _ in range(prim.r + 1)]
    with pytest.raises(ValueError):
        prim.hash_sponge(data)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_invalid_state_size(name, params):
    prim = Tip4(params)
    with pytest.raises(ValueError):
        prim.permutation([prim.F.zero()] * (prim.t + 1))


# ---------------------------------------------------------------------------
# 4.5 (cont.) Validation, warnings
# ---------------------------------------------------------------------------

def test_non_64bit_field_rejected():
    # The design is fixed to ~64-bit fields; smaller (toy) fields are a hard
    # error here, not a recommendation warning.
    with pytest.raises(ValueError):
        Tip4Params(p=8191)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_recommended_instance_no_warning(name, params):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParamRecommendationWarning)
        Tip4Params(p=params.p, t=params.t, R=params.R, u=params.u,
              r=params.r, c=params.c, d=params.d, kappa=params.kappa)

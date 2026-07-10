# test_tip4prime.py
# ---------------------------------------------------------------------------
# Test suite for Tip4Prime (TIP4'), parametrized over the named instance.
# TIP4' uses a reduced state of t=12 and the RPO circulant MDS matrix.
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

from tip5.hash import Tip4Prime
from tip5.params import Tip4PrimeParams
from tip5.instances import TIP4_PRIME
from recommendations import ParamRecommendationWarning

INSTANCES = [
    ("TIP4_PRIME", TIP4_PRIME),
]

# ---------------------------------------------------------------------------
# 4.1 Known-answer test vectors.
# Self-derived (input = [0, 1, ..., t-1]): TIP4' uses the RPO circulant MDS
# matrix (Tip4PrimeParams._init_mat = circulant(RPO_MDS_ROWS[12])), which differs from the
# truncated Tip5 column used by the sage reference, so these are regenerated from
# this implementation rather than taken from the reference. Reproduce by running
# the permutation / hash_sponge once on the fixed input and pasting the result.
# ---------------------------------------------------------------------------

PERM_KATS = {
    "TIP4_PRIME": [
        {
            "input": list(range(12)),
            "output": [
                12219262814323383363, 1883521465846805839, 8306103264085130535, 14834307083748268728,
                4517165057084710919, 15350310310229153987, 6598810059700262591, 8413036671015552750,
                13576490751299249828, 8242510587915824313, 4031652832455123165, 4347466615197134883,
            ],
        },
    ],
}

HASH_KATS = {
    "TIP4_PRIME": [
        {
            "input": list(range(8)),
            "output": [
                1892580624205133764, 4984391443128303411, 9127970292838149648, 14882338902370023817,
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
    prim = Tip4Prime(params)
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.permutation(inp)
    assert [prim.from_field(x) for x in out] == kat["output"]


@pytest.mark.parametrize("name,params,kat", HASH_KAT_CASES, ids=HASH_KAT_IDS)
def test_hash_kat(name, params, kat):
    prim = Tip4Prime(params)
    inp = [prim.to_field(x) for x in kat["input"]]
    out = prim.hash_sponge(inp)
    assert [prim.from_field(x) for x in out] == kat["output"]


# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    prim = Tip4Prime(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner, for every round.
    prim = Tip4Prime(params)
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
    prim = Tip4Prime(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    prim = Tip4Prime(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_hash_output_size(name, params):
    prim = Tip4Prime(params)
    data = [prim.F.random_element() for _ in range(prim.r)]
    assert len(prim.hash_sponge(data)) == prim.d


# ---------------------------------------------------------------------------
# 4.4 Algebraic: lookup table and matrix generation
# ---------------------------------------------------------------------------

def test_params_derive_constants():
    """Omitting LUT/M derives the values used by the hardcoded TIP4' instance
    (M is the RPO circulant via Tip4PrimeParams._init_mat)."""
    derived = Tip4PrimeParams()
    assert TIP4_PRIME.LUT == derived.LUT
    assert TIP4_PRIME.M == derived.M


# ---------------------------------------------------------------------------
# 4.5 Misc: validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_hash_rejects_wrong_length(name, params):
    prim = Tip4Prime(params)
    data = [prim.F.random_element() for _ in range(prim.r + 1)]
    with pytest.raises(ValueError):
        prim.hash_sponge(data)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_invalid_state_size(name, params):
    prim = Tip4Prime(params)
    with pytest.raises(ValueError):
        prim.permutation([prim.F.zero()] * (prim.t + 1))


# ---------------------------------------------------------------------------
# 4.5 (cont.) Validation, warnings
# ---------------------------------------------------------------------------

def test_non_64bit_field_rejected():
    # The design is fixed to ~64-bit fields; smaller (toy) fields are a hard
    # error here, not a recommendation warning.
    with pytest.raises(ValueError):
        Tip4PrimeParams(p=8191)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_recommended_instance_no_warning(name, params):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParamRecommendationWarning)
        Tip4PrimeParams(p=params.p, t=params.t, R=params.R, u=params.u,
              r=params.r, c=params.c, d=params.d, kappa=params.kappa)

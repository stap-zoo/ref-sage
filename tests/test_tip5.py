# test_tip5.py
# ---------------------------------------------------------------------------
# Test suite for Tip5, parametrized over the named instance in instances.py.
#
# Groups:
#   4.1 KATs         -- fixed input/output vectors (permutation + hash)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- lookup table / round-constant / matrix generation
#   4.5 Misc         -- validation/errors
# ---------------------------------------------------------------------------

import warnings

import pytest

from tip5.hash import Tip5Perm, Tip5Hash
from tip5.params import Tip5Params
from tip5.instances import TIP5, LOOKUP_TABLE
from recommendations import ParamRecommendationWarning

INSTANCES = [
    ("TIP5", TIP5),
]

# ---------------------------------------------------------------------------
# 4.1 Known-answer test vectors
# The permutation KAT is generated with the sage implementation from
# https://github.com/isec-tugraz/ca-tip5family-monolith/blob/main/Tip5.sage;
# the hash KATs come from the Rust reference implementation from
# https://github.com/Neptune-Crypto/twenty-first.
# ---------------------------------------------------------------------------

PERM_KATS = {
    "TIP5": [
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
    # hash10 test vectors from the Rust reference implementation
    # https://github.com/Neptune-Crypto/twenty-first/blob/master/twenty-first/src/math/tip5.rs
    "TIP5": [
        {
            "input": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "output": [
                941080798860502477, 5295886365985465639, 14728839126885177993,
                10358449902914633406, 14220746792122877272,
            ],
        },
        {
            "input": [
                941080798860502477, 5295886365985465639, 14728839126885177993,
                10358449902914633406, 14220746792122877272, 0, 0, 0, 0, 0,
            ],
            "output": [
                15888421881075650037, 8699648354187865464, 6719068786850902915,
                16188941274693647820, 4768361305800190493,
            ],
        },
        {
            "input": [
                941080798860502477, 15888421881075650037, 8699648354187865464,
                6719068786850902915, 16188941274693647820, 4768361305800190493, 0, 0, 0, 0,
            ],
            "output": [
                11494362724359741120, 2984169814429715553, 11021746812971026026,
                5102281498552384717, 5023112854146751042,
            ],
        },
        {
            "input": [
                941080798860502477, 15888421881075650037, 11494362724359741120,
                2984169814429715553, 11021746812971026026, 5102281498552384717,
                5023112854146751042, 0, 0, 0,
            ],
            "output": [
                627201255727529993, 2530132417472465719, 15134374672529870482,
                10586143339158028166, 13810271029904013559,
            ],
        },
        {
            "input": [
                941080798860502477, 15888421881075650037, 11494362724359741120,
                627201255727529993, 2530132417472465719, 15134374672529870482,
                10586143339158028166, 13810271029904013559, 0, 0,
            ],
            "output": [
                4790238723037855394, 13717377209729127271, 8994982932799814404,
                18004412270774820131, 5877166878145340765,
            ],
        },
        {
            "input": [
                941080798860502477, 15888421881075650037, 11494362724359741120,
                627201255727529993, 4790238723037855394, 13717377209729127271,
                8994982932799814404, 18004412270774820131, 5877166878145340765, 0,
            ],
            "output": [
                16959020643814878453, 12118009629857908438, 10239930869937551135,
                6889489196156760098, 5774309862903741805,
            ],
        },
        {
            "input": [
                941080798860502477, 15888421881075650037, 11494362724359741120,
                627201255727529993, 4790238723037855394, 16959020643814878453,
                12118009629857908438, 10239930869937551135, 6889489196156760098, 5774309862903741805,
            ],
            "output": [
                10869784347448351760, 1853783032222938415, 6856460589287344822,
                17178399545409290325, 7650660984651717733,
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
    P = Tip5Perm(params)
    inp = [P.to_field(x) for x in kat["input"]]
    out = P.permute(inp)
    assert [P.from_field(x) for x in out] == kat["output"]


@pytest.mark.parametrize("name,params,kat", HASH_KAT_CASES, ids=HASH_KAT_IDS)
def test_hash_kat(name, params, kat):
    P = Tip5Perm(params)
    H = Tip5Hash(P, params.sponge)
    inp = [P.to_field(x) for x in kat["input"]]
    out = H.hash(inp, input_len_fixed=True, c_val=1)
    assert [P.from_field(x) for x in out] == kat["output"]


# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    P = Tip5Perm(params)
    inp = [P.F.random_element() for _ in range(P.t)]
    assert P.permute_inv(P.permute(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner, for every round.
    P = Tip5Perm(params)
    inp = [P.F.random_element() for _ in range(P.t)]
    for r in range(P.R):
        assert P.linear_layer_inv(P.linear_layer(inp, r), r) == inp
        assert P.constant_addition_inv(P.constant_addition(inp, r), r) == inp
        assert P.nonlinear_layer_inv(P.nonlinear_layer(inp, r), r) == inp
    assert P._pre_rounds_inv(P._pre_rounds(inp)) == inp
    assert P._post_rounds_inv(P._post_rounds(inp)) == inp


# ---------------------------------------------------------------------------
# 4.3 Consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    P = Tip5Perm(params)
    inp = [P.F.random_element() for _ in range(P.t)]
    assert P.permute(inp) == P.permute(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    P = Tip5Perm(params)
    inp1 = [P.F.random_element() for _ in range(P.t)]
    inp2 = [P.F.random_element() for _ in range(P.t)]
    while inp1 == inp2:
        inp2 = [P.F.random_element() for _ in range(P.t)]
    assert P.permute(inp1) != P.permute(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_hash_output_size(name, params):
    P = Tip5Perm(params)
    H = Tip5Hash(P, params.sponge)
    data = [P.F.random_element() for _ in range(params.sponge["r"])]
    assert len(H.hash(data, input_len_fixed=True, c_val=1)) == H.sponge.d


# ---------------------------------------------------------------------------
# 4.4 Algebraic: lookup table, round constant and matrix generation
# ---------------------------------------------------------------------------

def test_lookup_table_matches_computed():
    assert LOOKUP_TABLE == Tip5Params._init_LUT()


def test_params_derive_constants():
    """Omitting LUT/rcons/M derives constants identical to the hardcoded TIP5 ones."""
    pytest.importorskip("blake3")
    P = Tip5Perm(Tip5Params())
    assert TIP5.LUT == P.LUT
    assert TIP5.rcons == P.rcons
    assert TIP5.M == P.M


# ---------------------------------------------------------------------------
# 4.5 Misc: validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_invalid_state_size(name, params):
    P = Tip5Perm(params)
    with pytest.raises(ValueError):
        P.permute([P.F.zero()] * (P.t + 1))


# ---------------------------------------------------------------------------
# 4.5 (cont.) Validation, warnings
# ---------------------------------------------------------------------------

def test_non_64bit_field_rejected():
    # The design is fixed to ~64-bit fields; smaller (toy) fields are a hard
    # error here, not a recommendation warning.
    with pytest.raises(ValueError):
        Tip5Params(p=8191)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_recommended_instance_no_warning(name, params):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParamRecommendationWarning)
        Tip5Params(p=params.p, t=params.t, R=params.R, u=params.u,
              sponge=dict(r=params.sponge["r"], c=params.sponge["c"], d=params.sponge["d"]), kappa=params.kappa)

import pytest

from tip5.hash import Tip5
from tip5.instances import (
    TIP5,
    TIP4,
    TIP4_PRIME,
    LOOKUP_TABLE,
    MDS_FIRST_COLUMN,
)
from tip5.params import Tip5Params
from utils import circulant
from fields import GOLDILOCKS

INSTANCES = [
    ("TIP5", TIP5),
    ("TIP4", TIP4),
    ("TIP4_PRIME", TIP4_PRIME),
]

# ---------------------------------------------------------------------------
# Known-answer test vectors
# Permutation KATs and the TIP4/TIP4' hash KATs are generated with a sage implementation from
# https://github.com/isec-tugraz/ca-tip5family-monolith/blob/main/Tip5.sage
# the TIP5 hash KATs come from the Rust reference implementation from 
# https://github.com/Neptune-Crypto/twenty-first.
# ---------------------------------------------------------------------------

PERM_KATS = {
    "TIP5": [
        {
            "input": list(range(16)),
            "output": [
                14273019456630489802,
                12225354657803044645,
                18223679466392555512,
                4879234115918641111,
                198243361942729835,
                6697571774370475124,
                3935892719377798608,
                2781322532457452310,
                7475933807446249354,
                7334965145562953054,
                1275437117587945070,
                2445375571864276273,
                17005006372293520413,
                9537835648539327419,
                12703602725074524970,
                5428520427373770602,
            ],
        },
    ],
    "TIP4": [
        {
            "input": list(range(16)),
            "output": [
                14273019456630489802,
                12225354657803044645,
                18223679466392555512,
                4879234115918641111,
                198243361942729835,
                6697571774370475124,
                3935892719377798608,
                2781322532457452310,
                7475933807446249354,
                7334965145562953054,
                1275437117587945070,
                2445375571864276273,
                17005006372293520413,
                9537835648539327419,
                12703602725074524970,
                5428520427373770602,
            ],
        },
    ],
    "TIP4_PRIME": [
        {
            "input": list(range(12)),
            "output": [
                14052082272586664442,
                5999147200624552423,
                5130878526965287601,
                11195056982911222609,
                12444002341736099458,
                2835395984450030227,
                18343297692531478784,
                1896548992427398174,
                10786360897180253413,
                3891560815945122708,
                18270876542082115058,
                15946224005925976610,
            ],
        },
    ],
}

HASH_KATS = {
    # TIP5: hash10 test vectors from the Rust reference implementation
    # https://github.com/Neptune-Crypto/twenty-first/blob/master/twenty-first/src/math/tip5.rs
    "TIP5": [
        {
            "input": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "output": [
                941080798860502477,
                5295886365985465639,
                14728839126885177993,
                10358449902914633406,
                14220746792122877272,
            ],
        },
        {
            "input": [
                941080798860502477,
                5295886365985465639,
                14728839126885177993,
                10358449902914633406,
                14220746792122877272,
                0,
                0,
                0,
                0,
                0,
            ],
            "output": [
                15888421881075650037,
                8699648354187865464,
                6719068786850902915,
                16188941274693647820,
                4768361305800190493,
            ],
        },
        {
            "input": [
                941080798860502477,
                15888421881075650037,
                8699648354187865464,
                6719068786850902915,
                16188941274693647820,
                4768361305800190493,
                0,
                0,
                0,
                0,
            ],
            "output": [
                11494362724359741120,
                2984169814429715553,
                11021746812971026026,
                5102281498552384717,
                5023112854146751042,
            ],
        },
        {
            "input": [
                941080798860502477,
                15888421881075650037,
                11494362724359741120,
                2984169814429715553,
                11021746812971026026,
                5102281498552384717,
                5023112854146751042,
                0,
                0,
                0,
            ],
            "output": [
                627201255727529993,
                2530132417472465719,
                15134374672529870482,
                10586143339158028166,
                13810271029904013559,
            ],
        },
        {
            "input": [
                941080798860502477,
                15888421881075650037,
                11494362724359741120,
                627201255727529993,
                2530132417472465719,
                15134374672529870482,
                10586143339158028166,
                13810271029904013559,
                0,
                0,
            ],
            "output": [
                4790238723037855394,
                13717377209729127271,
                8994982932799814404,
                18004412270774820131,
                5877166878145340765,
            ],
        },
        {
            "input": [
                941080798860502477,
                15888421881075650037,
                11494362724359741120,
                627201255727529993,
                4790238723037855394,
                13717377209729127271,
                8994982932799814404,
                18004412270774820131,
                5877166878145340765,
                0,
            ],
            "output": [
                16959020643814878453,
                12118009629857908438,
                10239930869937551135,
                6889489196156760098,
                5774309862903741805,
            ],
        },
        {
            "input": [
                941080798860502477,
                15888421881075650037,
                11494362724359741120,
                627201255727529993,
                4790238723037855394,
                16959020643814878453,
                12118009629857908438,
                10239930869937551135,
                6889489196156760098,
                5774309862903741805,
            ],
            "output": [
                10869784347448351760,
                1853783032222938415,
                6856460589287344822,
                17178399545409290325,
                7650660984651717733,
            ],
        },
    ],
    "TIP4": [
        {
            "input": list(range(12)),
            "output": [
                15841355890359640929,
                14241254185087876088,
                7403730774983924736,
                1993907345493395678,
            ],
        },
    ],
    "TIP4_PRIME": [
        {
            "input": list(range(8)),
            "output": [
                6065955114939091268,
                8580883933753702893,
                157271259785212968,
                5835718442440307983,
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

# ---------------------------------------------------------------------------
# Lookup table and round constant verification
# ---------------------------------------------------------------------------

def test_lookup_table_matches_computed():
    assert LOOKUP_TABLE == Tip5Params._init_lut()


@pytest.mark.parametrize(
    "params",
    [TIP5, TIP4, TIP4_PRIME],
    ids=["TIP5", "TIP4", "TIP4_PRIME"],
)
def test_params_derive_constants(params):
    """Omitting LUT and rcons derives constants identical to the hardcoded TIP5 ones."""
    pytest.importorskip("blake3")
    h = Tip5(Tip5Params(
        p=params.p, t=params.t, R=params.R, alpha=params.alpha, 
        u=params.u, d=params.d, r=params.r, c=params.c,
    ))
    assert params.LUT == h.LUT
    assert params.rcons == h.rcons
    assert params.M == h.M

# ---------------------------------------------------------------------------
# KAT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params,kat", PERM_KAT_CASES, ids=PERM_KAT_IDS)
def test_permutation_kat(name, params, kat):
    h = Tip5(params)
    inp = [h.to_field(x) for x in kat["input"]]
    out = h.permutation(inp)
    assert [h.from_field(x) for x in out] == kat["output"]


@pytest.mark.parametrize("name,params,kat", HASH_KAT_CASES, ids=HASH_KAT_IDS)
def test_hash_kat(name, params, kat):
    h = Tip5(params)
    inp = [h.to_field(x) for x in kat["input"]]
    out = h.hash_sponge(inp)
    assert [h.from_field(x) for x in out] == kat["output"]

# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    h = Tip5(params)
    inp = [h.F.random_element() for _ in range(h.t)]
    assert h.permutation(inp) == h.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    h = Tip5(params)
    inp1 = [h.F.random_element() for _ in range(h.t)]
    inp2 = [h.F.random_element() for _ in range(h.t)]
    while inp1 == inp2:
        inp2 = [h.F.random_element() for _ in range(h.t)]
    assert h.permutation(inp1) != h.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    h = Tip5(params)
    inp = [h.F.random_element() for _ in range(h.t)]
    assert h.permutation_inv(h.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_hash_output_size(name, params):
    h = Tip5(params)
    data = [h.F.random_element() for _ in range(h.r)]
    assert len(h.hash_sponge(data)) == h.d


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_hash_rejects_wrong_length(name, params):
    h = Tip5(params)
    data = [h.F.random_element() for _ in range(h.r + 1)]
    with pytest.raises(ValueError):
        h.hash_sponge(data)

import pytest

from griffin.hash import Griffin
from griffin.params import GriffinParams
from griffin.instances import (
    GRIFFIN_BN254_T3,
    GRIFFIN_BLS12_T3,
    GRIFFIN_ST_T3,
    GRIFFIN_GOLDILOCKS_T8,
    GRIFFIN_GOLDILOCKS_T12,
)
from utils import matvecmul
from fields import BN254_SCALAR, BLS12_381_SCALAR, ST, GOLDILOCKS

INSTANCES = [
    ("BN254_T3", GRIFFIN_BN254_T3),
    ("BLS12_T3", GRIFFIN_BLS12_T3),
    ("ST_T3", GRIFFIN_ST_T3),
    ("GOLDILOCKS_T8", GRIFFIN_GOLDILOCKS_T8),
    ("GOLDILOCKS_T12", GRIFFIN_GOLDILOCKS_T12),
]

# ---------------------------------------------------------------------------
# Known-answer test vectors (self-derived: input = [0, 1, ..., t-1], constants
# are deterministically derived via GriffinParams._init_constants, so this KAT
# is reproducible from (p, t, R, alpha) alone)
# ---------------------------------------------------------------------------

KATS = {
    "BN254_T3": [
        {
            "input": [0, 1, 2],
            "output": [
                15862405785128810275837435502653224425290071258167230490599117376332100235254,
                13220756517509979517684528785753328587257706928708746278499548208567338458968,
                15550532036911446928426039913328049280239190234626561457568755196858003615133,
            ],
        },
        {
            "input": [0, 1, 2],
            "output": [
                15862405785128810275837435502653224425290071258167230490599117376332100235254,
                13220756517509979517684528785753328587257706928708746278499548208567338458968,
                15550532036911446928426039913328049280239190234626561457568755196858003615133,
            ],
        },
    ],
    "BLS12_T3": [
        {
            "input": [0, 1, 2],
            "output": [
                27379052990992335868007513827616442821891910747512806453448496790052721925738,
                24772506163846602410384726533562269659751910620421046453035480040071869301700,
                49516412382145609386411188123247611096923270094753177293676945877936768369921,
            ],
        },
    ],
    "ST_T3": [
        {
            "input": [0, 1, 2],
            "output": [
                842514964983803238368419627199749344096196210222292575243178149295586607630,
                978342616368383987780518867705880330172775772459103616733689669395344717067,
                1441212399909815083075619210710598400179959939489702004227985936886943578183,
            ],
        },
    ],
    "GOLDILOCKS_T8": [
        {
            "input": list(range(8)),
            "output": [
                11044291317764846690, 8392807820624905171, 11998765736882583965, 1120635896287166011,
                14047258533844332447, 10831574347988506351, 1971816747908971053, 3591530767801178415,
            ],
        },
    ],
    "GOLDILOCKS_T12": [
        {
            "input": list(range(12)),
            "output": [
                12636498849595024313, 6381010898939669123, 13498744415857791704, 17175163947875671109,
                17404550387514679938, 10298122775399632625, 12740681290952013714, 15333201480421367508,
                13768631036699908211, 6225520087020383210, 12114399014673009031, 300798477373108186,
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


@pytest.mark.parametrize("name,params,kat", KAT_CASES, ids=KAT_IDS)
def test_permutation_kat(name, params, kat):
    g = Griffin(params)
    inp = [g.to_field(x) for x in kat["input"]]
    out = g.permutation(inp)
    assert [g.from_field(x) for x in out] == kat["output"]


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    g = Griffin(params)
    inp = [g.F.random_element() for _ in range(g.t)]
    assert g.permutation(inp) == g.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    g = Griffin(params)
    inp1 = [g.F.random_element() for _ in range(g.t)]
    inp2 = [g.F.random_element() for _ in range(g.t)]
    while inp1 == inp2:
        inp2 = [g.F.random_element() for _ in range(g.t)]
    assert g.permutation(inp1) != g.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    g = Griffin(params)
    inp = [g.F.random_element() for _ in range(g.t)]
    assert g.permutation_inv(g.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_sponge_output_size(name, params):
    g = Griffin(params)
    data = [g.F.random_element() for _ in range(g.r * 3)]
    assert len(g.hash_sponge(data)) == g.d


# Some additional tests covering differrent state sizes and alphas

AFFINE_FIELDS = [
    ("BN254", BN254_SCALAR, 5),
    ("BLS12", BLS12_381_SCALAR, 5),
    ("ST", ST, 3),
    ("GOLDILOCKS", GOLDILOCKS, 7),
]

AFFINE_CASES = [
    (field_name, field, alpha, t) for field_name, field, alpha in AFFINE_FIELDS for t in (3, 4, 8)
]

@pytest.mark.parametrize(
    "field_name,field,alpha,t", AFFINE_CASES,
    ids=[f"{field_name} with t={t}, alpha={a}" for field_name, _, a, t in AFFINE_CASES],
)
def test_affine(field_name, field, alpha, t):
    params = GriffinParams(p=field.p, t=t, alpha=alpha, R=1)
    g = Griffin(params)

    inp = [g.F.random_element() for _ in range(t)]

    expected = matvecmul(g.M, inp)
    actual = g.AffineLayer(inp, params.R - 1)  # last round: round constant is zero
    assert actual == expected

import pytest

from arion.hash import Arion
from arion.instances import ARION_BLS12_T3

# ---------------------------------------------------------------------------
# Known-answer test vector (computed with this implementation; constants are
# deterministically derived via ArionParams._init_rcons, so this KAT is
# self-reproducible from (p, t, R, alpha1, alpha2) alone)
# ---------------------------------------------------------------------------

KATS = {
    "BLS12_T3": {
        "input": [1, 2, 3],
        "output": [
            8659761558258982444581154623385831633595691755172963733105476337475047514700,
            4791269626370762720123406288230192995551974039613211374308744809811237991818,
            52352966164464576463303472127080635169313939454784230048314196193104401844767,
        ],
    },
}

INSTANCES = [
    ("BLS12_T3", ARION_BLS12_T3),
]


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_kat(name, params):
    a = Arion(params)
    kat = KATS[name]
    inp = [a.to_field(x) for x in kat["input"]]
    out = a.permutation(inp)
    assert [a.from_field(x) for x in out] == kat["output"]


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_deterministic(name, params):
    a = Arion(params)
    inp = [a.F.random_element() for _ in range(a.t)]
    assert a.permutation(inp) == a.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_distinct_inputs(name, params):
    a = Arion(params)
    inp1 = [a.F.random_element() for _ in range(a.t)]
    inp2 = [a.F.random_element() for _ in range(a.t)]
    assert a.permutation(inp1) != a.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_permutation_roundtrip(name, params):
    a = Arion(params)
    inp = [a.F.random_element() for _ in range(a.t)]
    assert a.permutation_inv(a.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=[name for name, _ in INSTANCES])
def test_sponge_output_size(name, params):
    a = Arion(params)
    data = [a.F.random_element() for _ in range(a.r * 3)]
    assert len(a.hash_sponge(data)) == a.r

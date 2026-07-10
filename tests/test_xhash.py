# test_xhash.py
# ---------------------------------------------------------------------------
# Test suite for XHash, parametrized over the named instances in instances.py.
#
# Groups:
#   4.1 KATs         -- fixed input/output vectors (permutation + sponge)
#   4.2 Roundtrip    -- permutation_inv / layer inverses (extension S-box inverse is
#                       not implemented yet, so those raise NotImplementedError)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- derivation of the MDS matrix M and the round constants
#   4.5 Misc         -- validation/errors
#
# The KATs are self-generated regression vectors (XHash has no published test
# vectors, and hash_sponge uses this repo's rate-first state ordering). Goldilocks
# outputs are written in hex, Mersenne-31 outputs in decimal.
# ---------------------------------------------------------------------------

import pytest

from marvellous.hash import XHash
from marvellous.params import XHashParams
from marvellous.instances import (
    XHASH12_GOLDILOCKS_T12,
    XHASH8_GOLDILOCKS_T12,
    XHASH24_M31_T24,
    XHASH16_M31_T24,
    XHASH_CPOLYS_GOLDILOCKS_POW7,
    XHASH_CPOLYS_M31_POW5,
)

INSTANCES = [
    ("XHASH12_GOLDILOCKS_T12", XHASH12_GOLDILOCKS_T12),
    ("XHASH8_GOLDILOCKS_T12",  XHASH8_GOLDILOCKS_T12),
    ("XHASH24_M31_T24",        XHASH24_M31_T24),
    ("XHASH16_M31_T24",        XHASH16_M31_T24),
]
IDS = [name for name, _ in INSTANCES]

# (name, instance, raw cpolys, skipbox) -- used to rebuild an instance with M/rcons derived.
DERIVATION = [
    ("XHASH12_GOLDILOCKS_T12", XHASH12_GOLDILOCKS_T12, XHASH_CPOLYS_GOLDILOCKS_POW7, None),
    ("XHASH8_GOLDILOCKS_T12",  XHASH8_GOLDILOCKS_T12,  XHASH_CPOLYS_GOLDILOCKS_POW7, [3, 1]),
    ("XHASH24_M31_T24",        XHASH24_M31_T24,        XHASH_CPOLYS_M31_POW5,        None),
    ("XHASH16_M31_T24",        XHASH16_M31_T24,        XHASH_CPOLYS_M31_POW5,        [3, 1]),
]


def _derive(inst, cpolys, skipbox, **override):
    """Rebuild an XHashParams with the same parameters as `inst` but with `M`/`rcons`
    left for the derivation helpers (unless overridden)."""
    kw = dict(
        p=inst.p, t=inst.t, alpha=int(inst.alpha), alpha_inv=inst.alpha_inv,
        kappa=inst.kappa, r=inst.r, c=inst.c, R=inst.R, d=inst.d,
        cpolys=cpolys, skipbox=skipbox, M=None, rcons=None,
    )
    kw.update(override)
    return XHashParams(**kw)


# ---------------------------------------------------------------------------
# 4.1 Known-answer test vectors (self-generated regression vectors)
# ---------------------------------------------------------------------------

# permutation([0, 1, ..., t-1]) -> full output state
PERM_KATS = {
    "XHASH12_GOLDILOCKS_T12": [0x2af4f867d3ffe8e1, 0x16629d1e965f5865, 0x860d48b70fce7087, 0xde50059a8faec976, 0x5c23cf03951f18b, 0x9beb7b07e2694731, 0x8f14827ad01b5fc5, 0x2ee3baec7d041aca, 0x69572584d4ccfe16, 0x112fbe99485e442f, 0x2f459f1ddde9ed5a, 0xf2470425b50754f5],
    "XHASH8_GOLDILOCKS_T12": [0xfff04d2bccc40e94, 0xa3eb23c24815e29f, 0xfa0e90ffe6772892, 0xab9fc60a61d622e1, 0xc48396a94aba53ad, 0x92f7a05395c23f3d, 0x3361edea44fd2df0, 0xef966cde993b9b52, 0x23ba0d9cd336774e, 0xf3b879364c20a929, 0x3f0ad962e49d06bd, 0x871a1a2afe3c216d],
    "XHASH24_M31_T24": [222795846, 1503524025, 1221917599, 461702812, 719378039, 399580604, 1660023501, 307728312, 1100187307, 1936701771, 2024694627, 1420838141, 1654726963, 1525510163, 1563714974, 385578630, 6017514, 2104527862, 483224599, 2037268755, 516854790, 1717175209, 1701286044, 1824850620],
    "XHASH16_M31_T24": [1313309307, 1588591996, 1469616168, 2089606387, 921879129, 441213447, 1358478787, 1031185399, 1175036763, 1374607520, 690478873, 1728176968, 351959532, 16378273, 1649288182, 829070792, 2063466465, 657725845, 513121124, 432327380, 945427053, 181858884, 1598832537, 1809757222],
}

# hash_sponge([0, 1, ..., i-1]) -> digest, for i = 1 .. rate  (single block only)
SPONGE_KATS = {
    "XHASH12_GOLDILOCKS_T12": [
        (1, [0xa1e0bdd46cc07fdc, 0x2c20667c4f2c9627, 0xfe1d8ee9a99d65a6, 0x40ecc102469b57e1]),
        (2, [0xfe883b55f1366c84, 0x607130159d4d1cb1, 0x2dc9a0ea8f480d3d, 0x31ed39dbfd9e898c]),
        (3, [0xd68020b7befcde54, 0xe51061b34e43a874, 0x2844447467a43e6a, 0xc9a293c5369f9425]),
        (4, [0x143e65125645618, 0xd78f3df9c6b634d0, 0x7e33f2ff1326c045, 0xd5103469782bca35]),
        (5, [0x6e11474853ad9c02, 0xc0b184d55e27359a, 0x49bed2b8886610f6, 0xbe94ae4d59e0c9ef]),
        (6, [0x7a6c14abf43253db, 0x3b52e71a3d9a754a, 0x1762c5e671295e36, 0x8c40a4242d1d6a7f]),
        (7, [0xa782175ee99c16a1, 0xc59bbd38c4f44186, 0xc646384307a1bbca, 0x39c179c4ea66934]),
        (8, [0xe1ff5b4ad049b99d, 0x2c43b71e380fbe9a, 0xdfad05a61fcd184d, 0x31081275c2bdba69]),
    ],
    "XHASH8_GOLDILOCKS_T12": [
        (1, [0x845f21151c00104d, 0xfcaa54f63c83468a, 0x2fa02fa0efffd05d, 0xba0a141dcb7301f2]),
        (2, [0x47fe2010e70907a0, 0x7c5367d66068b85c, 0xb788dbc0e2e59e84, 0x5f11e5aebe05d7c0]),
        (3, [0x807c7e1ee6549194, 0x135da74ec66612e0, 0x52f48ccba8b6e5e8, 0x4a52eb8ffa1d9585]),
        (4, [0x4b63ad39dd9d79fe, 0x7e38f463f99a9044, 0x573623c3d09f31dd, 0xf0cd775f6a4ca95a]),
        (5, [0xf52f8f51e451023d, 0x2022c9a012fd0a79, 0xe27e5ea183274fac, 0xd7d1b889425376f2]),
        (6, [0x54a1d66b8dfdf55b, 0xee10cb5c19e757f9, 0xb3eb862cb28e77e6, 0x9b7964bda96e4db5]),
        (7, [0x9959127d0f63e601, 0x248ba8e181e176a5, 0x856d16ba569ec9ec, 0x286d2ed35d103846]),
        (8, [0xac1cd54a1ab81c04, 0x8a77cc57381f763a, 0x57d3864c902dfa61, 0xc72eda989d5c67a0]),
    ],
    "XHASH24_M31_T24": [
        (1, [605999748, 380756602, 232224832, 1684964641, 742856011, 758280869, 109716000, 2137340187]),
        (2, [1079145670, 1349121682, 530411325, 485992877, 1029827770, 2128285303, 1435187034, 1434893930]),
        (3, [255106810, 1002206478, 1689539896, 841984784, 964711341, 2068442309, 587612479, 2058656524]),
        (4, [1392566123, 301652415, 1504698918, 672248147, 911846052, 2113126053, 896658978, 2013076329]),
        (5, [1851649250, 1344226513, 1856365756, 1649424420, 1469690151, 1348467122, 1050229178, 1208124012]),
        (6, [562259650, 634614640, 1615054501, 745189324, 1673262860, 1895617232, 2085018795, 1256976706]),
        (7, [782993482, 754814785, 2036925805, 202554793, 871611951, 541328877, 727066022, 2137420117]),
        (8, [484805560, 293202153, 1871899967, 1302895167, 365189902, 2074402366, 1517370991, 1383043312]),
        (9, [1919401322, 1808618075, 38730685, 1840926488, 589076232, 836105467, 892055103, 439572105]),
        (10, [28394222, 1385798508, 719873410, 755906428, 1733606128, 621981261, 1074649534, 1789351469]),
        (11, [1779854516, 984364144, 1485757473, 944384202, 114088162, 1955560956, 1519930308, 1507062807]),
        (12, [1587962237, 63241281, 570129796, 1986270350, 1847277163, 2068444704, 1998936125, 153045908]),
        (13, [891338345, 1128732889, 225169812, 1631948920, 1779359501, 752797629, 610864268, 99763807]),
        (14, [1594553955, 1047940902, 194259061, 880809277, 1243183907, 816061485, 1934462823, 807738715]),
        (15, [563931818, 981788669, 602107915, 2128754867, 223672473, 1494699989, 2127806482, 1846093919]),
        (16, [299019951, 1346099474, 1560833134, 1103114536, 81250129, 538145139, 1572528154, 870787339]),
    ],
    "XHASH16_M31_T24": [
        (1, [1416754606, 2118796529, 1448269208, 1566203477, 1793609592, 1565254117, 1473728463, 451905639]),
        (2, [1288487873, 598166317, 38094407, 1474000742, 1215787938, 1357583757, 1817110645, 1754415908]),
        (3, [519111910, 1598947309, 1110159375, 138001350, 1778334594, 638136602, 593746715, 1334335562]),
        (4, [694862037, 857978532, 1113357260, 974607780, 2115494875, 526436059, 581858335, 88850]),
        (5, [1565261057, 726283889, 1238844247, 411355286, 1927176025, 1815232680, 164804270, 1596322995]),
        (6, [295786953, 111046980, 225751751, 1935330725, 360729682, 294743964, 1011052682, 334804826]),
        (7, [1348123204, 354985650, 1378275482, 279042323, 1138584405, 396671491, 921385809, 1050031082]),
        (8, [1767709873, 1442503436, 147170416, 333211652, 181778994, 743475204, 1459278830, 1722410783]),
        (9, [722328462, 1005457868, 810478081, 932643644, 685789066, 1816965430, 1812998723, 1147045466]),
        (10, [2032515028, 1914480933, 677089504, 725739089, 1712586260, 352709020, 1043562580, 624981127]),
        (11, [1198988704, 777109582, 519469404, 609280150, 991301006, 848838974, 1873420314, 1435121321]),
        (12, [392520260, 916035211, 2109988006, 812066512, 563759807, 1981817278, 474338286, 1334170769]),
        (13, [1034168561, 985486067, 1694448241, 894443727, 758242145, 1779842254, 2024064940, 58820170]),
        (14, [447195275, 1547797992, 921401914, 1610029279, 1263291172, 152185556, 576925755, 222250672]),
        (15, [1174298841, 1534813422, 1921408453, 1648154391, 1637908778, 1019732877, 1443694876, 1043325262]),
        (16, [120440825, 823946875, 2133266454, 1151776606, 514400999, 229070901, 691001609, 1701605700]),
    ],
}


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_kat(name, params):
    prim = XHash(params)
    inp = [prim.to_field(i) for i in range(prim.t)]
    out = [int(prim.from_field(x)) for x in prim.permutation(inp)]
    assert out == PERM_KATS[name]


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_sponge_kat(name, params):
    prim = XHash(params)
    for n, expected in SPONGE_KATS[name]:
        inp = [prim.to_field(j) for j in range(n)]
        out = [int(prim.from_field(x)) for x in prim.hash_sponge(inp)]
        assert out == expected, f"sponge KAT mismatch for input length {n}"


# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
#
# The XHash extension S-box inverse (_sbox_P3_inv) is not implemented yet, so the
# full permutation inverse and the P3 non-linear layer inverse raise
# NotImplementedError. The intended invertibility assertions are written in full
# but skip-marked until the inverse power map lands; the active tests document
# the current raising behavior.
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="XHash _sbox_P3_inv not implemented; permutation_inv raises NotImplementedError")
@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_roundtrip(name, params):
    prim = XHash(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_inv_raises_not_implemented(name, params):
    # Documents the current behavior until _sbox_P3_inv is implemented.
    prim = XHash(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    with pytest.raises(NotImplementedError):
        prim.permutation_inv(prim.permutation(inp))


@pytest.mark.skip(reason="XHash _sbox_P3_inv not implemented; the P3 rounds (r % 3 == 2) raise NotImplementedError")
@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_p3_layer_roundtrip(name, params):
    prim = XHash(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    for r in range(int(1.5 * prim.R)):
        if r % 3 == 2:
            assert prim.nonlinear_layer_inv(prim.nonlinear_layer(inp, r), r) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_layer_roundtrip(name, params):
    # Constant addition and the linear layer are genuine inverses for every round.
    # The non-linear layer is only invertible for non-P3 rounds; the P3 rounds
    # (r % 3 == 2) hit the unimplemented extension S-box inverse (see the
    # skip-marked test_p3_layer_roundtrip above).
    prim = XHash(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    for r in range(int(1.5 * prim.R)):
        assert prim.constant_addition_inv(prim.constant_addition(inp, r), r) == inp
        assert prim.linear_layer_inv(prim.linear_layer(inp, r), r) == inp
        if r % 3 == 2:
            with pytest.raises(NotImplementedError):
                prim.nonlinear_layer_inv(prim.nonlinear_layer(inp, r), r)
        else:
            assert prim.nonlinear_layer_inv(prim.nonlinear_layer(inp, r), r) == inp
    assert prim._pre_rounds_inv(prim._pre_rounds(inp)) == inp
    assert prim._post_rounds_inv(prim._post_rounds(inp)) == inp


# ---------------------------------------------------------------------------
# 4.3 Consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_digest_size(name, params):
    prim = XHash(params)
    out = prim.hash_sponge([prim.to_field(0)])
    assert len(out) == prim.r // 2 == prim.d


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_deterministic(name, params):
    prim = XHash(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_distinct_inputs(name, params):
    prim = XHash(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


# ---------------------------------------------------------------------------
# 4.4 Algebraic: derivation of M and the round constants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,inst,cpolys,skipbox", DERIVATION, ids=IDS)
def test_mds_matrix_derivation(name, inst, cpolys, skipbox):
    # _init_mat must reproduce the committed matrix (t=12 via RPO_MDS_ROWS, t=24 via
    # the truncated Reed-Solomon circulant).
    derived = _derive(inst, cpolys, skipbox)
    assert derived.M == inst.M


@pytest.mark.parametrize("name,inst,cpolys,skipbox", DERIVATION, ids=IDS)
def test_rcons_count_and_derivation(name, inst, cpolys, skipbox):
    # n_rcons is the number of round-constant rows the permutation consumes.
    assert inst.n_rcons == int(1.5 * inst.R) + 1
    derived = _derive(inst, cpolys, skipbox)
    # Derivation produces exactly n_rcons rows, each of length t.
    assert len(derived.rcons) == inst.n_rcons
    assert all(len(row) == inst.t for row in derived.rcons)
    # Derivation is deterministic.
    assert _derive(inst, cpolys, skipbox).rcons == derived.rcons


@pytest.mark.parametrize("name,inst,cpolys,skipbox", DERIVATION, ids=IDS)
def test_too_few_rcons_rejected(name, inst, cpolys, skipbox):
    # Supplying fewer than n_rcons rows must be rejected by parameter sanitization.
    too_few = [[0] * inst.t for _ in range(inst.n_rcons - 1)]
    with pytest.raises(ValueError):
        _derive(inst, cpolys, skipbox, rcons=too_few)


# ---------------------------------------------------------------------------
# 4.5 Misc: validation
# ---------------------------------------------------------------------------

def test_invalid_state_size():
    prim = XHash(XHASH12_GOLDILOCKS_T12)
    with pytest.raises(ValueError):
        prim.permutation([prim.F.zero()] * (prim.t + 1))

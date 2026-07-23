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
        kappa=inst.kappa, r=inst.sponge.r, c=inst.sponge.c, R=inst.R, d=inst.sponge.d,
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
        (1, [0x41dc3e753451ff23, 0xd9a0f3abaa511947, 0xe12c14623e66a425, 0x38970e6aa19b56c2]),
        (2, [0xfe61703c8add2886, 0x7e49d37636e64a18, 0x966757af166396db, 0xa4bb1702a2d58cfa]),
        (3, [0x961a6983f124a53e, 0xcd1130dcfba123b0, 0xf500e7d5cd03ece6, 0xdb1f0858318afe75]),
        (4, [0x12c5a7059fb6ce2c, 0xfee9fbbdb9b09f51, 0x873d4a24925dec67, 0xd5aae13789147dc5]),
        (5, [0x59ccd0a2de26265a, 0xe39f1e175e385e11, 0x96c3a14cae97796d, 0x10a41258ec171431]),
        (6, [0x566e6bed458513db, 0xbefdbb07b287c16d, 0xcea29504827ce171, 0x32973a44da77d31]),
        (7, [0x46863bb61ffea36c, 0xdb00b0a8754620f4, 0x750aba00a1ff5e0e, 0x3c83db67bb6779d0]),
        (8, [0xe1ff5b4ad049b99d, 0x2c43b71e380fbe9a, 0xdfad05a61fcd184d, 0x31081275c2bdba69]),
    ],
    "XHASH8_GOLDILOCKS_T12": [
        (1, [0x55f006b09eaad9ff, 0x5817230e66da6c49, 0xd1b539cac96d7e45, 0x4537c1af634dc3fc]),
        (2, [0x2c97c98132e1048f, 0x8d89ac319918f218, 0x2f7f18ebec57927e, 0xed90a8cea4984591]),
        (3, [0xd1e1fc00a8422473, 0x968c7db9cbd86c99, 0x67889484cf5b6125, 0xd5b515667ff8743f]),
        (4, [0x62c9b5e52eeafd53, 0xde8b574b7a314cd8, 0x511e5fab917e0236, 0xba0cd7344ccfca7e]),
        (5, [0x2507a71321095d80, 0x49f6e199013fc07a, 0xfb976f3ac6379861, 0x32049f09ed5193cc]),
        (6, [0x86d575559337f10f, 0x1335181aa84cd0b6, 0x8654597e25e959e4, 0x57aa8094291573f4]),
        (7, [0xcdec629611104ac4, 0xeb6b7c236e08d869, 0xcb55f4ec3c067a24, 0x94bdd31a22b73118]),
        (8, [0xac1cd54a1ab81c04, 0x8a77cc57381f763a, 0x57d3864c902dfa61, 0xc72eda989d5c67a0]),
    ],
    "XHASH24_M31_T24": [
        (1, [464997319, 283983415, 916386028, 413900914, 612044207, 2064548918, 882835715, 1043642470]),
        (2, [620602516, 1263128016, 411474467, 1705535020, 1978720964, 596186781, 387186873, 1725667604]),
        (3, [383096523, 1547425630, 612598721, 624527334, 171050006, 1244192119, 1511341329, 1822586540]),
        (4, [822791297, 87142835, 521718546, 2087560482, 63954909, 201518505, 177667015, 796410295]),
        (5, [1494690473, 375467793, 2115945278, 1064106545, 1654158373, 1575385341, 1410114145, 1375021722]),
        (6, [76080232, 437709791, 1518422139, 1011498858, 1439229749, 1675305105, 799163658, 1011876748]),
        (7, [1430873493, 1611422532, 1773270210, 624111006, 442544573, 1196384894, 1684685715, 462244800]),
        (8, [1491748175, 902746492, 1671787454, 527124404, 1033646068, 420826474, 1569016579, 1508470983]),
        (9, [1234232857, 1687876209, 2130994944, 1062392189, 1506662166, 1912005922, 539077940, 1602035875]),
        (10, [34350, 1697243146, 721357851, 10955531, 474774910, 166916482, 1020613624, 905135049]),
        (11, [663385632, 1744289477, 804506002, 852767703, 1382637470, 1848839884, 1116318909, 1187440372]),
        (12, [175826384, 1371536135, 812132796, 1460773046, 504150903, 1521275058, 1776561137, 350397933]),
        (13, [1264261205, 74826811, 2146908921, 1692909318, 1972444900, 1663389805, 898488337, 1011854456]),
        (14, [1751506797, 2121265294, 991329598, 241241833, 180901640, 1510599148, 1308713293, 147931460]),
        (15, [551638395, 1593119707, 2592709, 1227946725, 978939244, 2132986915, 312120361, 673480181]),
        (16, [299019951, 1346099474, 1560833134, 1103114536, 81250129, 538145139, 1572528154, 870787339]),
    ],
    "XHASH16_M31_T24": [
        (1, [300570971, 2068609971, 2109280869, 1845825732, 699051467, 950036770, 1926121558, 1326562368]),
        (2, [49177583, 98437222, 236028700, 128132030, 2082958489, 259866475, 1212540194, 61638308]),
        (3, [899266456, 369690887, 174982763, 849991107, 599652439, 427725658, 574377405, 2087734407]),
        (4, [898797421, 131702979, 2025301418, 77120463, 1034438291, 1561256862, 2105041431, 1500488083]),
        (5, [1603489899, 997170463, 11268659, 1932733412, 187789581, 1817128050, 926949186, 1422011062]),
        (6, [944147428, 246011122, 923959545, 1805584888, 428026952, 444405315, 924936236, 1218662469]),
        (7, [1808221095, 658239314, 1096454288, 69522144, 1943855392, 1003227176, 1579380189, 1888762749]),
        (8, [314574539, 1305801668, 637151440, 1403268064, 1455083347, 2076168830, 1755407679, 1198722599]),
        (9, [259394712, 1493356412, 2033921000, 893556566, 120863267, 141576766, 767056129, 953426125]),
        (10, [895469244, 268705368, 1501807789, 44403763, 291087115, 1285692119, 1865038948, 1705812828]),
        (11, [1036349069, 770102193, 639434912, 1567505007, 1117655023, 1805425748, 201064341, 544857828]),
        (12, [1686267822, 1411535098, 1155284700, 1137151987, 104918054, 919202902, 922537988, 165507314]),
        (13, [301925030, 955439510, 692915950, 761616454, 212638859, 1886567661, 133879017, 1292627531]),
        (14, [1059188574, 1275938866, 709213819, 769624242, 1918961548, 1438988550, 281983225, 1045487204]),
        (15, [563621380, 102869711, 791609749, 236398998, 2105792399, 239046012, 162035300, 1384589098]),
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
    assert len(out) == prim.sponge.r // 2 == prim.sponge.d


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

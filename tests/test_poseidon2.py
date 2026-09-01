# test_poseidon2.py
# ---------------------------------------------------------------------------
# Test suite for Poseidon2, parametrized over the named instances in
# hades/instances.py so every recommended instance is covered by the same checks.
#
# Groups:
#   4.1 KATs         -- fixed input/output vectors (HorizenLabs reference)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Algebraic    -- generated constants/matrices match the reference
#   4.5 Misc         -- validation/errors, warnings, reproducibility
# ---------------------------------------------------------------------------

import warnings

import pytest

from hades.hash import Poseidon2Perm, Poseidon2Hash
from hades.params import Poseidon2Params
from recommendations import ParamRecommendationWarning
from hades.instances import (
    POSEIDON2_BLS12_T2,
    POSEIDON2_BLS12_T3,
    POSEIDON2_BLS12_T4,
    POSEIDON2_BLS12_T8,
    POSEIDON2_BN254_T3,
    POSEIDON2_GOLDILOCKS_T8,
    POSEIDON2_GOLDILOCKS_T12,
    POSEIDON2_GOLDILOCKS_T16,
    POSEIDON2_GOLDILOCKS_T20,
    POSEIDON2_MERSENNE_T16,
    POSEIDON2_MERSENNE_T24,
)

INSTANCES = [
    ("BLS12_T2",       POSEIDON2_BLS12_T2),
    ("BLS12_T3",       POSEIDON2_BLS12_T3),
    ("BLS12_T4",       POSEIDON2_BLS12_T4),
    ("BLS12_T8",       POSEIDON2_BLS12_T8),
    ("BN254_T3",       POSEIDON2_BN254_T3),
    ("GOLDILOCKS_T8",  POSEIDON2_GOLDILOCKS_T8),
    ("GOLDILOCKS_T12", POSEIDON2_GOLDILOCKS_T12),
    ("GOLDILOCKS_T16", POSEIDON2_GOLDILOCKS_T16),
    ("GOLDILOCKS_T20", POSEIDON2_GOLDILOCKS_T20),
    ("MERSENNE_T16",   POSEIDON2_MERSENNE_T16),
    ("MERSENNE_T24",   POSEIDON2_MERSENNE_T24),
]
IDS = [name for name, _ in INSTANCES]
PARAMS = dict(INSTANCES)

# ---------------------------------------------------------------------------
# 4.1 Known-answer tests (from the upstream HorizenLabs reference permutation()
# tests, https://github.com/HorizenLabs/poseidon2)
# ---------------------------------------------------------------------------

KATS = {
    "GOLDILOCKS_T12": [
        {"input": list(range(12)),
         "output": [
             0xed3dbcc4ff1e8d33, 0xfb85eac6ac91a150, 0xd41e1e237ed3e2ef, 0x5e289bf0a4c11897,
             0x4398b20f93e3ba6b, 0x5659a48ffaf2901d, 0xe44d81e89a88f8ae, 0x08efdb285f8c3dbc,
             0x294ab7503297850e, 0xa11c61f4870b9904, 0xa6855c112cc08968, 0x17c6d53d2fb3e8c1,
         ]},
    ],
    "BLS12_T2": [
        {"input": [0, 1],
         "output": [
             0x50f38c87fbf14be6e91d0d911b52dc8c1b19fe439348c427514a8b59bdf92f62,
             0x3222c2d9d80f8be5aff518685e66ae4648cc76243d1ca077101bebb2ee245d30,
         ]},
    ],
    "BLS12_T3": [
        {"input": [0, 1, 2],
         "output": [
             0x562af4b3710cdba6cea53e1f73325b21bb97ac810943b74d863d87163ee8042e,
             0x4674eba4cef166510c0d7a9ddf08cf813637bc2081e2c40c5047dce7ecdf2b95,
             0x0cf55ec35287dca6195eb6dd43e9ac1aba8857b4d3e4501be8bd8e9946a8dc54,
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
    P = Poseidon2Perm(params)
    inp = [P.to_field(x) for x in kat["input"]]
    out = P.permute(inp)
    assert [int(P.from_field(x)) for x in out] == kat["output"]


# Poseidon2: flat round constants and flat internal matrix M_I (= J + diag(MAT_DIAG_M_1)),
# copied verbatim from the HorizenLabs reference implementation
# (https://github.com/HorizenLabs/poseidon2).
REF = {
    "GOLDILOCKS_T12": {
        "rc": [
    0xe034a8785fd284a7, 0xe2463f1ea42e1b80, 0x48742e681ae290a, 0xe4af50ade990154c, 0x8b13ffaaf4f78f8a, 0xe3fbead7dccd8d63, 0x631a47705eb92bf8, 0x88fbbb8698548659,
    0x74cd2003b0f349c9, 0xe16a3df6764a3f5d, 0x57ce63971a71aaa2, 0xdc1f7fd3e7823051, 0xbb8423be34c18d7a, 0xf8bc5a2a0c1b3d6d, 0xf1a01bbd6f7123e5, 0xed960a080f5e348b,
    0x1b9c0c1e87e2390e, 0x18c83caf729a613e, 0x671ab9fe037a72c4, 0x508565f67d4c276a, 0x4d2cd8827a482590, 0xa48e11e84dd3500b, 0x825a8c955fc2442b, 0xf573a6ee07cddc68,
    0x7dd3f19c73a39e0b, 0xcc0f13537a796fa6, 0x1d9006bfaedac57f, 0x4705f69b68b0b7de, 0x5b62bfb718bcc57f, 0x879d821770563827, 0x3da5ccb7f8dff0e3, 0xb49d6a706923fc5b,
    0xb6a0babe883a969d, 0x2984f9b055401960, 0xcd3496f05511d79d, 0x4791da5d63854fc5, 0xdb7344d0580a39d4, 0x5aedc4dad1de120a, 0x5e1bdc1fb8e1abf0, 0x3904c09a0e46747c,
    0xb54a0e23ab85ddcd, 0xc0c3cf05bccbdb3a, 0xb362076a73baf7e9, 0x212c953d81a5d5ba, 0x212d4cc965d898bd, 0xdd44ddd0f41509b9, 0x8931329fa67823c0, 0xc65510f4d2a873be,
    0xe3ecbb6ba1e16211, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x70f5b3266792bbb6, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0xe7560e690634757e, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0xafd0202bc7eaf66e, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x349f4c5871f220fd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x3697eb3e31529e0d, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x7735d5b0622d9900, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x5f5b58b9cf997668, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x645534b6548af9d9, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x4232d29d91a426a8, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0xb987278aed485d35, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x6dabeef669bb406e, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x35ee78288b749d40, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x6dcd560f14af0fc3, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x71ed3dc007ea6383, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x8b6b51caab7f5b6f, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0xcf2e8cc4181dbfa8, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0xa01d3f1c306f825a, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0xccee646a5d8ddb87, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x70df6f277cbaffeb, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x64ec0a6556b8f45c, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x6f68c9664fda6e37, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
    0x387356e4516fab6f, 0x35310dce33903e67, 0x45f3e5251d30f912, 0x7c97f480ca428f45, 0x74d5874c20b50de2, 0xff1d5b7cee3dc67f, 0xa04d5d5ac0ff3de9, 0x1cefb5eb7d24580e,
    0xf685e1bfcc0104ad, 0x6204dd95db22ead4, 0x8265c6c57c73c440, 0x4f708ab0b4e1e382, 0xcfc60c7a52fbffa7, 0x9c0c1951d8910306, 0x4d06df27c89819f2, 0x621bdb0e75eca660,
    0x343adffd079cee57, 0xa760f0e5debde398, 0xe3110fefd97b188a, 0xed6584e6b150297, 0x2b10e625d0d079c0, 0xefa493442057264f, 0xebcfaa7b3f26a2b6, 0xf36bcda28e343e2a,
    0xa1183cb63b67aa9e, 0x40f3e415d5e5b0ba, 0xc51fc2367eff7b15, 0xe07fe5f3aebc649f, 0xc9cb2be56968e8aa, 0x648600db69078a0e, 0x4e9135ab1256edb9, 0x382c73435556c2,
    0x1d78cafac9150ddf, 0xb8df60ab6215a233, 0xa7a65ba31f8fcd9a, 0x907d436dd964006b, 0x3bdf7fd528633b97, 0x265adb359c0cc0f8, 0xf16cfc4034b39614, 0x71f0751b08fa0947,
    0x3165eda4b5403a37, 0xca30fc5680467e46, 0x4c743354d37777c5, 0x3d1f0a4e6bba4a09, 0xc0c2e289afa75181, 0x1e4fa2ad948978b7, 0x2a226a127a0bb26a, 0xe61738a70357ce76,
        ],
        "mat_internal": [
    0xcf6f77ac16722afa, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1,
    0x1, 0x1, 0x1, 0x1, 0x1, 0x3fd4c0d74672aebd, 0x1, 0x1,
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1,
    0x1, 0x1, 0x9b72bf1c1c3d08a9, 0x1, 0x1, 0x1, 0x1, 0x1,
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0xe4940f84b71e4ac3,
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1,
    0x1, 0x1, 0x1, 0x1, 0x61b27b077118bc73, 0x1, 0x1, 0x1,
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1,
    0x1, 0x2efd8379b8e661e3, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1,
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x858edcf353df0342, 0x1,
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1,
    0x1, 0x1, 0x1, 0x2d9c20affb5c4517, 0x1, 0x1, 0x1, 0x1,
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1,
    0x5120143f0695defc, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1,
    0x1, 0x1, 0x1, 0x1, 0x1, 0x62fc898ae34a5c5c, 0x1, 0x1,
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1,
    0x1, 0x1, 0xa3d9560c99123ed3, 0x1, 0x1, 0x1, 0x1, 0x1,
    0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x98fd739d8e7fc934,
        ],
    },
    "BLS12_T3": {
        "rc": [
    0x452088f7ec90c80818a1b5665f38ea30116456becd6709977cd8a2e0a2b38b62, 0x21512346b8ece60d5951c1505089c2b4220707ca56373bb9d828fa33bbfd2a31, 0x2b3a40252c69e83e92c548e199bbbeba4291e0d7fc3b4810193606753da588c8, 0x2a1a778e3f303c4187c082ea4475734596fb10bd2954843e12be80e8c1c0d464,
    0x400df3a9dd4631e354222e1b1c2ab7092feb19a96eef425270e2016e2348fb96, 0x667e2e3deaf0278725697f5acc7222fd572e22944d04d7fd2f86c5c1c0cea988, 0x224d2355fc17ee0b5e46455d2ef3a85cfaa88b08689b0d0e4c111094fd780093, 0x6d82b530a685ea27a3f7d47626bb7997b59cf2c6ec2ac14ff28ff0562a95e22a,
    0x2241de16388cdd7ffda42e0838b5d59bc2182f14bbef622fa633d8b87250a740, 0x6b9fc67a95a7a01f9034b58ac9130de47ec903c21c77cbd05258b011e904b80e, 0x325f11e96905193f6836e6fa2c727dd0261ea083fedde9873f1e7b9d90419833, 0x6e39fe041e18592c01cb0ae2e7cf51f57797c498be313f3485de24974129f48f,
    0x51693257ab2fa82eb0e8040a6866fe4d121c59b8a22ff9b17c72ad78e4dcb42a, 0x0, 0x0, 0x48805d48952999e8f52884e543f775f6bdac8da75cf37bad19754415183c6516,
    0x0, 0x0, 0x1cc541b9ed19280c216f6b90876cbe83d07ba14fcc6f2af068e1dda739f5acb9, 0x0,
    0x0, 0xf6f3f6703c0dcd136b24ddb8766fbcd69bbc9cb3bb20a1da2f7130c4ba62664, 0x0, 0x0,
    0x352376600a75802c6e6c6da69001e0376328848bc7ada465176d571ba029a20b, 0x0, 0x0, 0x2b1708d59adc4ba04a6bcb2dd264c1b014e0b7bae9bb3af916eb276ee3a34565,
    0x0, 0x0, 0x66c5bbb78c64b8369b845ed0a25af2d05e3a4ce053ff4fd056a5c8a059e2cb3c, 0x0,
    0x0, 0x5e80c1b3b2e5b0b2cc4385a07f85777bf8d59ee2358ce694376f35c844d300f0, 0x0, 0x0,
    0x6c2ae56c365577b154b04b7683b8910dd66808b89858e03cf627b50057425c6c, 0x0, 0x0, 0x65720f82218ce2bd09c504216f1fe44967596b1720ab267749537206c0ddf03f,
    0x0, 0x0, 0x71a7333c7e07b20015f2c1f030adc0137e4d7453d795616eeda38b3a25e709b8, 0x0,
    0x0, 0x161c8a77adcd1a5f8dad71c3b044ce64bab9de792195da91a0b0acca4f8b4568, 0x0, 0x0,
    0x5b7f6f2e59eecbf4802a456ac633c38ba8895b41cdde35c1caa5dd1ff4bdef7e, 0x0, 0x0, 0x5a2569754178df0732e843339b2b7d55519590d8fe6f4f4bf9f35bcd8c589b4d,
    0x0, 0x0, 0x5bf5026511b12bbfbde38a5394aa7dd9a8bc9a5055fadbcbaa541f3b523925e9, 0x0,
    0x0, 0x1d53e37d6ddf6dd88beb25c0870b2d0af2a51efb6ada05c4ac7e5099a71499e0, 0x0, 0x0,
    0x65e488523b0b3430d5f91e25b4b96ee6d628fbe0529c2868749657d7f7aa3f95, 0x0, 0x0, 0xb6d7adb7b72cfaee0184354accbff821a14efb48b46405b397c037a5e15f095,
    0x0, 0x0, 0x2444c70bc898765b95c5438156c28671cfd20569a8d31b3f08cfa60d2bb18d6e, 0x0,
    0x0, 0x6979d5cf3da00fcb59a9832188456fbe515261a9be5cf052ab8e50c874a07ff8, 0x0, 0x0,
    0x1eca31224b0d4ae965b179fd952d958de48a5de147348ca5dd00790d5c76fb2f, 0x0, 0x0, 0x5e1c8f87eccb7e8cd338500800ca62b6bed6fc390597c5a7f21eb7e80cecad80,
    0x0, 0x0, 0xc3f882f7a3bd8ae1eb328e026f6419db30a5026c279df1219499333ef8caa06, 0x0,
    0x0, 0x38011264a16e7cf3e96f029dbfe344e778314b1e2e9d8a2f8f8f76ff5795430d, 0x0, 0x0,
    0x3119da354a6f450bf8f700b89b8319a6f57d6278bfb0bbf9d8e37d55c9f3133d, 0x0, 0x0, 0xbbe0649314a68a31d5e8222bfec7b1298fc5bc1e6ea098675c94695aa3aa221,
    0x0, 0x0, 0x4dfb208e602b401ba661e37ff5a5fd971874a1082cfb00ebefc6405c2d0a73d3, 0x0,
    0x0, 0x2238971cffd7a12e565e591c0b28c8e76a4582d57892d3db5c8be394a60ba3e9, 0x0, 0x0,
    0x4c466e3153daea54ee62f719c6f947207e7234dbd7257489b4a8a385be7645ba, 0x0, 0x0, 0x1badbb1e2e9734afa09ff6c92e98038b9e329e0c5d0bac9a7d7996392f5caf78,
    0x0, 0x0, 0x37dbb6c5059651ae1362eba3195c08716dab6a61ce8476b5cebd09274a53413c, 0x0,
    0x0, 0x7154aa15ad1736497d7902f9e74828b3e65f43295cc20bf69ba2c622592c224e, 0x0, 0x0,
    0x3ca22f38e795e433ae9a8d2d5f1d535bbe1a3c3fdadcd549a718e72cd257bb09, 0x0, 0x0, 0x854ff209558742b8d9b0cbe9db7767f33668f1f8888458bc954025ffe84b7da,
    0x0, 0x0, 0x377699a38c21b41939a5098a8202ed55b4d3472e01f185f4336f4c6c879051ad, 0x0,
    0x0, 0x3492b7170200764635922920ad5e3c8761fbbcbaeaa2fc08a59f9ddac49a59ed, 0x0, 0x0,
    0x372c88a6b45c1593f353789be4cdce85dd2ea1adfca9d8444c83b7a990921a25, 0x0, 0x0, 0x3d0996734441d7d414e14d72e46eb6cb3d403a822ce642357e1319d169f4ce6f,
    0x0, 0x0, 0x26362f8c8a86b5afadc22b1cc0aa53e9da137607812778e47d5f86740722a52, 0x0,
    0x0, 0x195b6294bdf65bb0c07569f6c6193de7e5e9ea5b17b5adc39179ff15280fda29, 0x0, 0x0,
    0x440a155e09c263aa0c3fdfc62d5b6f6c759735d1b73efd60c5afd4deddbd7bf8, 0x0, 0x0, 0x5f0135b9b6a8ba9cc5d83221ba344778dfc00b1f4ac578455dd5d71dca2bb64e,
    0x0, 0x0, 0x5d108be88e31ef2e03e46662075177281cdb28e77c120d158c89ce3850fd93d7, 0x0,
    0x0, 0x11dfbb5f5e48ea973c6ef2ece89463c5316bc767896b67b88be18a1d858d6f52, 0x0, 0x0,
    0x71180314ab2e242cc0552b728495b97e0c2e073970a264c43397356d6ef6c99, 0x0, 0x0, 0x13c1cc0a221c29fde3183f7dc644004d3f4dd341fe7626996ce68c69d73204c5,
    0x0, 0x0, 0x6fc0f5f038d0ab20f4815ba721b366824bf534980265836b224fecfccb6fcaf, 0x0,
    0x0, 0x122e97658c701fcb7b25d8ce0629f2942dd8f07d6ac06a91320a6f1f4421fd59, 0x0, 0x0,
    0x3a19573d57741adba1942d72016391d115a1971af158cfe2a776cb506d714272, 0x0, 0x0, 0x4037de356b6cb97e4b73d9604342a7b3e5bf44dd44e4318cdc116edde917fffe,
    0x0, 0x0, 0x3b34dfdced639990194ad27c8d3bebd9e6657350c0c522a2b65f2ace44dd16cf, 0x0,
    0x0, 0x546b1eed6e680434955e6259f20ba428460181982eed9832898c945ce343f9cb, 0x0, 0x0,
    0x4d144b9647ff9822cb76f838802bd249bc58c2782587d470f5cd906f860a79d6, 0x0, 0x0, 0x2d39cfe678f0816cf3bc7a0476517c070d9db6b0f20aa849fc9746e4be5bdf80,
    0x0, 0x0, 0x69f92e5361b806c4e082661a39cf8ead1431c49536eaf8b8450804c99d9c8899, 0x0,
    0x0, 0x1c2e919d0e061629fd6b5416898dc1d5a5cd0e130531151d18480906ea3d9cc5, 0x0, 0x0,
    0x3dae30c784fd66c4a551a6b0a9551747fc1cae54522bb25238f06a7a3e4490cb, 0x264c23f67c44aa792f1c731655e1c9eefc4b4b808913f6bb3806ee56caf9c8b, 0xfa6d7c32c55e7621d72604c5abc1d970e7569dbf1475f989816be1ac248f889, 0x2516928d25d3fe4ca89ff71d5958f4f256d86457b58215dc8c1b02454314ff19,
    0x10842e1683519bc44c3b3de92cf860e9185c5ed67b20662ae8c4f50008de4780, 0x6b7e2c013a40a2c5f5a992a0cd4bf198a64171617af313332c244edffcdf45be, 0x36c65d30abc46a63c4b26ea1e17c5325181354f800fa4c4f207ed1849bb8b3c, 0x568d2f7f0f8fb4fc28ef1bc673f2db513427dbab162b7c444a1070cd85f22397,
    0x53a9fd8aefc965366ddcf373f85ce62ee77f1ef0bdd18243842c2eb313db9989, 0x1577e14026128fcbe30d7fe646e0cfcf5a91052f2cadc41553e10aa4ea94eb81, 0x38814490cf1681f17c23adf62ea2988d48fdce37b3a2fc259b090391d72be770, 0x3c24dd5b9460893f28e95b9cbaaba0e1b6af9c00d8182b66ec771ca957b4cb8d,
        ],
        "mat_internal": [
    0x2, 0x1, 0x1, 0x1,
    0x2, 0x1, 0x1, 0x1,
    0x3,
        ],
    },
}


@pytest.mark.parametrize("name", list(REF), ids=list(REF))
def test_generated_matches_reference(name):
    """Generated round constants (external rounds full width, internal rounds branch 0 only)
    and internal matrix M_I = J + diag(mat_diag) reproduce the upstream reference constants."""
    params = PARAMS[name]
    rc = [int(params.from_field(x)) for row in params.rcons for x in row]
    mi = [int(params.from_field(x)) for row in params.M_int for x in row]
    assert rc == REF[name]["rc"]
    assert mi == REF[name]["mat_internal"]


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_roundtrip(name, params):
    P = Poseidon2Perm(params)
    inp = [P.F.random_element() for _ in range(P.t)]
    assert P.permute_inv(P.permute(inp)) == inp
    assert P.permute(P.permute_inv(inp)) == inp

@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_layer_roundtrip(name, params):
    # Each component layer must be undone by its _inv partner. Use a representative
    # external and internal round index.
    P = Poseidon2Perm(params)
    inp = [P.F.random_element() for _ in range(P.t)]
    ext_idx, int_idx = 0, P.R_ext_beg
    for r in [ext_idx, int_idx]:
        assert P.nonlinear_layer_inv(P.nonlinear_layer(inp, r), r) == inp
        assert P.linear_layer_inv(P.linear_layer(inp, r), r) == inp
        assert P.constant_addition_inv(P.constant_addition(inp, r), r) == inp
    assert P._pre_rounds_inv(P._pre_rounds(inp)) == inp
    assert P._post_rounds_inv(P._post_rounds(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_deterministic(name, params):
    P = Poseidon2Perm(params)
    inp = [P.F.random_element() for _ in range(P.t)]
    assert P.permute(inp) == P.permute(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_distinct_inputs(name, params):
    P = Poseidon2Perm(params)
    inp1 = [P.F.random_element() for _ in range(P.t)]
    inp2 = [P.F.random_element() for _ in range(P.t)]
    while inp1 == inp2:
        inp2 = [P.F.random_element() for _ in range(P.t)]
    assert P.permute(inp1) != P.permute(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_sponge_output_size(name, params):
    P = Poseidon2Perm(params)
    H = Poseidon2Hash(P, params.sponge)
    data = [P.F.random_element() for _ in range(params.sponge["r"])]
    assert len(H.hash(data, input_len_fixed=True)) == H.sponge.d
    data = [P.F.random_element() for _ in range(params.sponge["r"] * 3)]
    assert len(H.hash(data, input_len_fixed=True)) == H.sponge.d


# ---------------------------------------------------------------------------
# 4.5 Misc: validation, warnings, reproducibility
# ---------------------------------------------------------------------------

def test_invalid_state_size():
    P = Poseidon2Perm(POSEIDON2_BN254_T3)
    with pytest.raises(ValueError):
        P.permute([P.F.zero()] * (P.t + 1))


def test_alpha_must_be_permutation():
    # An explicit alpha not coprime with p-1 must be rejected by params.
    with pytest.raises(ValueError):
        Poseidon2Params(p=POSEIDON2_BN254_T3.p, t=3, alpha=2, R_ext=2, R_int=2, sponge=dict(r=2, c=1, d=1))


def test_toy_field_warns():
    with pytest.warns(ParamRecommendationWarning):
        Poseidon2Params(p=101, t=3, alpha=3, R_ext=2, R_int=2, sponge=dict(r=2, c=1, d=1), toy=True)  # tiny field


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_recommended_instance_no_warning(name, params):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParamRecommendationWarning)
        Poseidon2Params(p=params.p, t=params.t, alpha=params.alpha,
                        R_ext=params.R_ext, R_int=params.R_int,
                        sponge=dict(r=params.sponge["r"], c=params.sponge["c"], d=params.sponge["d"]),
                        version=params.version, mat_diag=params.mat_diag)


def test_constants_reproducible():
    # Same parameters -> identical derived constants and matrices.
    kwargs = dict(p=POSEIDON2_BN254_T3.p, t=3, alpha=5, R_ext=8, R_int=56, sponge=dict(r=2, c=1, d=1))
    a = Poseidon2Params(**kwargs)
    b = Poseidon2Params(**kwargs)
    assert a.rcons == b.rcons
    assert a.M_ext == b.M_ext
    assert a.M_int == b.M_int

"""Field definitions and predefined field instances.

The `Field` type plus the concrete fields used throughout the repo (Goldilocks,
Mersenne-31, the BLS/BN scalar fields, ...), so every consumer of an instance shares the
exact same field parameters. Also includes helpers to derive field parameters (alpha,
generator). Add a new `Field` here if the one you need is missing.

For verification of field parameters, run with sage's Python: sage --python field.py
"""

from dataclasses import dataclass
from math import gcd

@dataclass(frozen=True)
class Field:
    name: str
    p: int                   # characteristic (prime)
    n: int                   # extension degree; |F| = p^n
    bits: int                # ceil(log2(p^n))
    factors: dict[int, int]  # prime factorization of p^n - 1: {prime: exponent}                              -> sage_factor(p - 1)
    generator: int           # (smallest) primitive element of F^*                                            -> find_smallest_generator(p)
    alpha: int               # smallest alpha > 1 with gcd(alpha, p^n - 1) = 1; x^alpha is a permutation on F -> find_alpha(p, n)
    alpha_inv: int           # alpha^(-1) mod (p^n - 1)                                                       -> pow(alpha, -1, p - 1)


# ---------------------------------------------------------------------------
# Helpers  (verify or compute field parameters)
# ---------------------------------------------------------------------------

def find_alpha(p: int, n: int = 1) -> int:
    """Return the smallest alpha > 1 such that x^alpha is a permutation on GF(p^n)."""
    group_order = p**n - 1
    for a in range(2, group_order):
        if gcd(a, group_order) == 1:
            return a
    raise ValueError(f"no valid alpha found for p={p}, n={n}")


def sage_factor(n: int) -> dict[int, int]:
    """Return the prime factorisation of n as {prime: exponent}, using Sage."""
    from sage.all import factor
    return {int(p): int(e) for p, e in factor(n)}


def find_smallest_generator(p: int, n: int = 1) -> int:
    """Return the smallest primitive root modulo p (prime field, n=1 only)."""
    from sage.all import GF
    if n != 1:
        raise NotImplementedError("primitive element search for n>1 not implemented")
    return int(GF(p).multiplicative_generator())


# ---------------------------------------------------------------------------
# Fields used in symmetric primitives and ZK proof systems
# ---------------------------------------------------------------------------


# --- ~32-bit fields ----------------------------------------------------------

# p = 2^31 - 2^24 + 1
KOALABEAR = Field(
    name="KoalaBear",
    p=(1 << 31) - (1 << 24) + 1,
    n=1,
    bits=31,
    factors={2: 24, 127: 1},
    generator=3,
    alpha=3,
    alpha_inv=1420470955,
)

# p = 2^31 - 1  (Mersenne prime)
MERSENNE31 = Field(
    name="Mersenne31",
    p=(1 << 31) - 1,
    n=1,
    bits=31,
    factors={2: 1, 3: 2, 7: 1, 11: 1, 31: 1, 151: 1, 331: 1},
    generator=7,
    alpha=5,
    alpha_inv=1717986917,
)

# p = 15 * 2^27 + 1
BABYBEAR = Field(
    name="BabyBear",
    p=15 * (1 << 27) + 1,
    n=1,
    bits=31,
    factors={2: 27, 3: 1, 5: 1},
    generator=31,
    alpha=7,
    alpha_inv=1725656503,
)

# --- ~64-bit fields ----------------------------------------------------------

# p = 2^64 - 2^32 + 1
GOLDILOCKS = Field(
    name="Goldilocks",
    p=(1 << 64) - (1 << 32) + 1,
    n=1,
    bits=64,
    factors={2: 32, 3: 1, 5: 1, 17: 1, 257: 1, 65537: 1},
    generator=7,
    alpha=7,
    alpha_inv=10540996611094048183,
)

# p = 2^61 + 20*2^32 + 1  (StarkWare's original STARK-friendly prime)
STARKWARE = Field(
    name="Starkware",
    p=(1 << 61) + 20 * (1 << 32) + 1,
    n=1,
    bits=62,
    factors={2: 34, 13: 1, 167: 1, 211: 1, 293: 1},
    generator=3,
    alpha=3,
    alpha_inv=1537228730075359915,
)

# --- ~256-bit fields -----------------------------------------------------

# p = 509 * 2^241 + 1  (NTT-friendly Proth prime k*2^n + 1 with k < 2^n)
ST = Field(
    name="ST",
    p=509 * (1 << 241) + 1,
    n=1,
    bits=250,
    factors={2: 241, 509: 1},
    generator=3,
    alpha=3,
    alpha_inv=1199100207962930165010531237170860699408252404679586732911863408610591287979,
)

# p = 2^251 + 17 * 2^192 + 1  (StarkNet / Cairo native field element "felt")
FELT252 = Field(
    name="Felt252",
    p=(1 << 251) + 17 * (1 << 192) + 1,
    n=1,
    bits=252,
    factors={2: 192, 5: 1, 7: 1, 98714381: 1, 166848103: 1},
    generator=3,
    alpha=3,
    alpha_inv=2412335192444087475798215188730046737082071476887731133315394704090581346987,
)

# Ed25519 scalar field (group order of the Ed25519 curve)
ED25519_SCALAR = Field(
    name="Ed25519-scalar",
    p=(1 << 252) + 27742317777372353535851937790883648493,
    n=1,
    bits=253,
    factors={2: 2, 3: 1, 11: 1, 198211423230930754013084525763697: 1, 276602624281642239937218680557139826668747: 1},
    generator=2,
    alpha=5,
    alpha_inv=4342203346399357328383911937825796544514269815627944563601170562971272550593,
)

# BLS12-377 scalar field (= base field of the Ed-on-BLS12-377 curve)
BLS12_377_SCALAR = Field(
    name="BLS12-377-scalar",
    p=0x12AB655E9A2CA55660B44D1E5C37B00159AA76FED00000010A11800000000001,
    n=1,
    bits=253,
    factors={2: 47, 3: 1, 5: 1, 7: 1, 13: 1, 499: 1, 958612291309063373: 1, 9586122913090633729: 2},
    generator=22,
    alpha=11,
    alpha_inv=6909105067714121256203584040821265343853008546944234041037918282114243922851,
)

# BLS12-381 scalar field (= base field of the Jubjub curve)
BLS12_381_SCALAR = Field(
    name="BLS12-381-scalar",
    p=0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001,
    n=1,
    bits=255,
    factors={2: 32, 3: 1, 11: 1, 19: 1, 10177: 1, 125527: 1, 859267: 1, 906349: 2, 2508409: 1, 2529403: 1, 52437899: 1, 254760293: 2},
    generator=7,
    alpha=5,
    alpha_inv=20974350070050476191779096203274386335076221000211055129041463479975432473805,
)

# BN254 (alt-bn128) base field
BN254_BASE = Field(
    name="BN254-base",
    p=21888242871839275222246405745257275088696311157297823662689037894645226208583,
    n=1,
    bits=254,
    factors={2: 1, 3: 2, 13: 1, 29: 1, 67: 1, 229: 1, 311: 1, 983: 1, 11003: 1, 405928799: 1, 11465965001: 1, 13427688667394608761327070753331941386769: 1},
    generator=3,
    alpha=5,
    alpha_inv=8755297148735710088898562298102910035478524462919129465075615157858090483433,
)

# BN254 (alt-bn128) scalar field
BN254_SCALAR = Field(
    name="BN254-scalar",
    p=21888242871839275222246405745257275088548364400416034343698204186575808495617,
    n=1,
    bits=254,
    factors={2: 28, 3: 2, 13: 1, 29: 1, 983: 1, 11003: 1, 237073: 1, 405928799: 1, 1670836401704629: 1, 13818364434197438864469338081: 1},
    generator=5,
    alpha=5,
    alpha_inv=17510594297471420177797124596205820070838691520332827474958563349260646796493,
)

# Pallas base field (= Vesta scalar field)
PALLAS = Field(
    name="Pallas",
    p=28948022309329048855892746252171976963363056481941560715954676764349967630337,
    n=1,
    bits=255,
    factors={2: 32, 3: 1, 463: 1, 539204044132271846773: 1, 8999194758858563409123804352480028797519453: 1},
    generator=5,
    alpha=5,
    alpha_inv=23158417847463239084714197001737581570690445185553248572763741411479974104269,
)

# Vesta base field (= Pallas scalar field)
VESTA = Field(
    name="Vesta",
    p=28948022309329048855892746252171976963363056481941647379679742748393362948097,
    n=1,
    bits=255,
    factors={2: 32, 3: 2, 1709: 1, 24859: 1, 1690502597179744445941507: 1, 10427374428728808478656897599072717: 1},
    generator=5,
    alpha=5,
    alpha_inv=23158417847463239084714197001737581570690445185553317903743794198714690358477,
)

# BLS12-381 scalar field
BLS12_381_SCALAR = Field(
    name="BLS12-381-scalar",
    p=52435875175126190479447740508185965837690552500527637822603658699938581184513,
    n=1,
    bits=255,
    factors={2: 32, 3: 1, 11: 1, 19: 1, 10177: 1, 125527: 1, 859267: 1, 906349: 2, 2508409: 1, 2529403: 1, 52437899: 1, 254760293: 2},
    generator=7,
    alpha=5,
    alpha_inv=20974350070050476191779096203274386335076221000211055129041463479975432473805,
)

# --- larger fields ---------------------------------------------------------

# BLS12-377 base field (= scalar field of the BW6-761 curve)
BLS12_377_BASE = Field(
    name="BLS12-377-base",
    p=0x1AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001,
    n=1,
    bits=377,
    factors={2: 46, 3: 1, 7: 1, 13: 1, 53: 1, 409: 1, 499: 1, 2557: 1, 6633514200929891813: 1, 73387170334035996766247648424745786170238574695861388454532790956181: 1},
    generator=15,
    alpha=5,
    alpha_inv=206931540810375275208522186955914826829114810203931728431907410133376374678672658219975110511658688099552257166541,
)

# BLS12-381 base field
BLS12_381_BASE = Field(
    name="BLS12-381-base",
    p=0x1A0111EA397FE69A4B1BA7B6434BACD764774B84F38512BF6730D2A0F6B0F6241EABFFFEB153FFFFB9FEFFFFFFFFAAAB,
    n=1,
    bits=381,
    factors={2: 1, 3: 2, 11: 1, 23: 1, 47: 1, 10177: 1, 859267: 1, 52437899: 1, 2584487767265781317813: 1, 15778400344354997994418419698270088123916926905054652752758194827714659: 1},
    generator=2,
    alpha=5,
    alpha_inv=3201927644177333914734231860588723325245506255951206308265646508899225320392670291554150103303212531230315418047829,
)

# Ed448 scalar field (group order of the Ed448 curve)
ED448_SCALAR = Field(
    name="Ed448-scalar",
    p=(1 << 446) - 13818066809895115352007386748515426880336692474882178609894547503885,
    n=1,
    bits=446,
    factors={2: 1, 3: 1, 19: 2, 97: 1, 227393: 1, 3009341: 1, 342682509629: 1, 6730519843040614479184435237013: 1, 547972593843380542316719287015009101629889568888367769396279985548530313239: 1},
    generator=2,
    alpha=5,
    alpha_inv=109025808644341033582398571183200680153046204103097709042223529877087602376923751429717453175015425577986176225454997382864253515789867,
)

FIELDS: dict[str, Field] = {
    f.name: f for f in (
        KOALABEAR, MERSENNE31, BABYBEAR,
        GOLDILOCKS, STARKWARE,
        ST, FELT252, ED25519_SCALAR, BLS12_377_SCALAR, BN254_BASE, BN254_SCALAR, PALLAS, VESTA, BLS12_381_SCALAR,
        BLS12_377_BASE, BLS12_381_BASE, ED448_SCALAR,
    )
}

# For verification of field parameters, run with sage's Python: sage --python fields.py

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
    if n != 1:
        raise NotImplementedError("primitive element search for n>1 not implemented")
    factors = sage_factor(p**n - 1)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    raise ValueError(f"no primitive root found for p={p}")


# ---------------------------------------------------------------------------
# Fields used in symmetric primitives and ZK proof systems
# ---------------------------------------------------------------------------


# --- 31-bit fields ----------------------------------------------------------

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

# --- 64-bit fields ----------------------------------------------------------

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

# --- 250-255-bit fields -----------------------------------------------------

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
    p=2**252 + 27742317777372353535851937790883648493,
    n=1,
    bits=253,
    factors={2: 2, 3: 1, 11: 1, 198211423230930754013084525763697: 1, 276602624281642239937218680557139826668747: 1},
    generator=2,
    alpha=5,
    alpha_inv=4342203346399357328383911937825796544514269815627944563601170562971272550593,
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

FIELDS: dict[str, Field] = {
    f.name: f for f in (
        KOALABEAR, MERSENNE31, BABYBEAR,
        GOLDILOCKS, STARKWARE,
        ST, FELT252, ED25519_SCALAR, BN254_SCALAR, PALLAS, VESTA, BLS12_381_SCALAR,
    )
}

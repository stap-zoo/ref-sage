from fields import BLS12_381_SCALAR, BN254_SCALAR, ST, GOLDILOCKS
from rescue_prime.params import RescuePrimeParams

# ---------------------------------------------------------------------------
# t=3 instances (kappa=128, c=1)
# ---------------------------------------------------------------------------

RESCUE_PRIME_BLS12_T3 = RescuePrimeParams(
    p=BLS12_381_SCALAR.p,
    t=3,
    alpha=BLS12_381_SCALAR.alpha,
    alpha_inv=BLS12_381_SCALAR.alpha_inv,
    kappa=128,
    c=1,
    g=BLS12_381_SCALAR.generator,
    d=2,
)

RESCUE_PRIME_BN254_T3 = RescuePrimeParams(
    p=BN254_SCALAR.p,
    t=3,
    alpha=BN254_SCALAR.alpha,
    alpha_inv=BN254_SCALAR.alpha_inv,
    kappa=128,
    c=1,
    g=BN254_SCALAR.generator,
    d=2,
)

RESCUE_PRIME_ST_T3 = RescuePrimeParams(
    p=ST.p,
    t=3,
    alpha=ST.alpha,
    alpha_inv=ST.alpha_inv,
    kappa=128,
    c=1,
    g=ST.generator,
    d=2,
)

# ---------------------------------------------------------------------------
# Goldilocks instances (kappa=128, c=4)
# ---------------------------------------------------------------------------

RESCUE_PRIME_GOLDILOCKS_T8 = RescuePrimeParams(
    p=GOLDILOCKS.p,
    t=8,
    alpha=GOLDILOCKS.alpha,
    alpha_inv=GOLDILOCKS.alpha_inv,
    kappa=128,
    c=4,
    g=GOLDILOCKS.generator,
    d=4,
)

RESCUE_PRIME_GOLDILOCKS_T12 = RescuePrimeParams(
    p=GOLDILOCKS.p,
    t=12,
    alpha=GOLDILOCKS.alpha,
    alpha_inv=GOLDILOCKS.alpha_inv,
    kappa=128,
    c=4,
    g=GOLDILOCKS.generator,
    d=8,
)

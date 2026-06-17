# instances.py
# ---------------------------------------------------------------------------
# Concrete, named parameter sets for the Marvellous family (Rescue, Rescue
# Prime, Rescue Prime Optimized).
#
# Each entry is a ready-to-use params instance pinned to a specific field from
# fields.py, so every consumer agrees on the exact same parameters. Naming
# convention: RESCUE_<FIELD>_<VARIANT> / RESCUE_PRIME_<FIELD>_<VARIANT> /
# RPO_<FIELD>_<VARIANT>. The sponge rate is r = t - c.
# ---------------------------------------------------------------------------

from fields import BLS12_381_SCALAR, BN254_SCALAR, ST, GOLDILOCKS, STARKWARE, ED25519_SCALAR, ED448_SCALAR
from marvellous.params import RescueParams, RescuePrimeParams, RescuePrimeOptimizedParams

# ---------------------------------------------------------------------------
# Rescue
# ---------------------------------------------------------------------------

# t=3 instances (kappa=128, c=1)

RESCUE_BLS12_T3 = RescueParams(
    p=BLS12_381_SCALAR.p,
    t=3,
    alpha=BLS12_381_SCALAR.alpha,
    alpha_inv=BLS12_381_SCALAR.alpha_inv,
    kappa=128,
    r=2,
    c=1,
    g=BLS12_381_SCALAR.generator,
    d=2,
)

RESCUE_BN254_T3 = RescueParams(
    p=BN254_SCALAR.p,
    t=3,
    alpha=BN254_SCALAR.alpha,
    alpha_inv=BN254_SCALAR.alpha_inv,
    kappa=128,
    r=2,
    c=1,
    g=BN254_SCALAR.generator,
    d=2,
)

RESCUE_ST_T3 = RescueParams(
    p=ST.p,
    t=3,
    alpha=ST.alpha,
    alpha_inv=ST.alpha_inv,
    kappa=128,
    r=2,
    c=1,
    g=ST.generator,
    d=2,
)

# Goldilocks t=12 instance (kappa=128, c=1)

RESCUE_GOLDILOCKS_T12 = RescueParams(
    p=GOLDILOCKS.p,
    t=12,
    alpha=GOLDILOCKS.alpha,
    alpha_inv=GOLDILOCKS.alpha_inv,
    kappa=128,
    r=11,
    c=1,
    g=GOLDILOCKS.generator,
    d=11,
)

# StarkWare t=12 instance (kappa=122, c=4, R=10, as defined in the Rescue paper)

RESCUE_STARKWARE_T12 = RescueParams(
    p=STARKWARE.p,
    t=12,
    alpha=STARKWARE.alpha,
    alpha_inv=STARKWARE.alpha_inv,
    kappa=122,
    r=8,
    c=4,
    R=10,
    g=STARKWARE.generator,
    d=8,
)

# Ed25519 t=6 instance (kappa=128, c=1)

RESCUE_ED25519_T6 = RescueParams(
    p=ED25519_SCALAR.p,
    t=6,
    alpha=ED25519_SCALAR.alpha,
    alpha_inv=ED25519_SCALAR.alpha_inv,
    kappa=128,
    r=5,
    c=1,
    g=ED25519_SCALAR.generator,
    d=5,
)

# Ed448 t=10 instance (kappa=224, c=2, R=10)

RESCUE_ED448_T10 = RescueParams(
    p=ED448_SCALAR.p,
    t=10,
    alpha=ED448_SCALAR.alpha,
    alpha_inv=ED448_SCALAR.alpha_inv,
    kappa=224,
    r=8,
    c=2,
    R=10,
    g=ED448_SCALAR.generator,
    d=8,
)

# ---------------------------------------------------------------------------
# Rescue Prime
# ---------------------------------------------------------------------------

# t=3 instances (kappa=128, c=1)

RESCUE_PRIME_BLS12_T3 = RescuePrimeParams(
    p=BLS12_381_SCALAR.p,
    t=3,
    alpha=BLS12_381_SCALAR.alpha,
    alpha_inv=BLS12_381_SCALAR.alpha_inv,
    kappa=128,
    r=2,
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
    r=2,
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
    r=2,
    c=1,
    g=ST.generator,
    d=2,
)

# Goldilocks instances (kappa=128, c=4)

RESCUE_PRIME_GOLDILOCKS_T8 = RescuePrimeParams(
    p=GOLDILOCKS.p,
    t=8,
    alpha=GOLDILOCKS.alpha,
    alpha_inv=GOLDILOCKS.alpha_inv,
    kappa=128,
    r=4,
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
    r=8,
    c=4,
    g=GOLDILOCKS.generator,
    d=8,
)

# ---------------------------------------------------------------------------
# Rescue Prime Optimized (RPO)
# ---------------------------------------------------------------------------

RPO_GOLDILOCKS_T12 = RescuePrimeOptimizedParams(
    p=GOLDILOCKS.p,
    t=12,
    alpha=GOLDILOCKS.alpha,
    alpha_inv=GOLDILOCKS.alpha_inv,
    kappa=128,
    r=8,
    c=4,
    R=7,
    d=4,
)

RPO_GOLDILOCKS_T16 = RescuePrimeOptimizedParams(
    p=GOLDILOCKS.p,
    t=16,
    alpha=GOLDILOCKS.alpha,
    alpha_inv=GOLDILOCKS.alpha_inv,
    kappa=160,
    r=10,
    c=6,
    R=7,
    d=5,
)

# ---------------------------------------------------------------------------
# XHASH
# ---------------------------------------------------------------------------

# TODO
from fields import BLS12_381_SCALAR, BN254_SCALAR, ST, GOLDILOCKS, STARKWARE, ED25519_SCALAR
from rescue.params import RescueParams

# ---------------------------------------------------------------------------
# t=3 instances (kappa=128, c=1)
# ---------------------------------------------------------------------------

RESCUE_BLS12_T3 = RescueParams(
    p=BLS12_381_SCALAR.p,
    t=3,
    alpha=BLS12_381_SCALAR.alpha,
    alpha_inv=BLS12_381_SCALAR.alpha_inv,
    kappa=128,
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
    c=1,
    g=ST.generator,
    d=2,
)

# ---------------------------------------------------------------------------
# Goldilocks t=12 instance (kappa=128, c=1)
# ---------------------------------------------------------------------------

RESCUE_GOLDILOCKS_T12 = RescueParams(
    p=GOLDILOCKS.p,
    t=12,
    alpha=GOLDILOCKS.alpha,
    alpha_inv=GOLDILOCKS.alpha_inv,
    kappa=128,
    c=1,
    g=GOLDILOCKS.generator,
    d=11,
)

# ---------------------------------------------------------------------------
# StarkWare t=12 instance (kappa=122, c=4, R=10, as defined in the Rescue paper)
# ---------------------------------------------------------------------------

RESCUE_STARKWARE_T12 = RescueParams(
    p=STARKWARE.p,
    t=12,
    alpha=STARKWARE.alpha,
    alpha_inv=STARKWARE.alpha_inv,
    kappa=122,
    c=4,
    R=10,
    g=STARKWARE.generator,
    d=8,
)

# ---------------------------------------------------------------------------
# Ed25519 t=6 instance (kappa=128, c=1)
# ---------------------------------------------------------------------------

RESCUE_ED25519_T6 = RescueParams(
    p=ED25519_SCALAR.p,
    t=6,
    alpha=ED25519_SCALAR.alpha,
    alpha_inv=ED25519_SCALAR.alpha_inv,
    kappa=128,
    c=1,
    g=ED25519_SCALAR.generator,
    d=5,
)

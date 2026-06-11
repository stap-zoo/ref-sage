from fields import BLS12_381_SCALAR, BN254_SCALAR, ST, GOLDILOCKS
from griffin.params import GriffinParams

# ---------------------------------------------------------------------------
# t=3 instances (kappa=128, c=1)
# ---------------------------------------------------------------------------

GRIFFIN_BN254_T3 = GriffinParams(
    p=BN254_SCALAR.p,
    t=3,
    alpha=5,
    R=12,
    r=2,
    c=1,
    d=1,
)

GRIFFIN_BLS12_T3 = GriffinParams(
    p=BLS12_381_SCALAR.p,
    t=3,
    alpha=5,
    R=12,
    r=2,
    c=1,
    d=1,
)

GRIFFIN_ST_T3 = GriffinParams(
    p=ST.p,
    t=3,
    alpha=3,
    R=16,
    r=2,
    c=1,
    d=1,
)

# ---------------------------------------------------------------------------
# Goldilocks instances (kappa=128, c=4)
# ---------------------------------------------------------------------------

GRIFFIN_GOLDILOCKS_T8 = GriffinParams(
    p=GOLDILOCKS.p,
    t=8,
    alpha=7,
    R=8,
    r=4,
    c=4,
    d=4,
)

GRIFFIN_GOLDILOCKS_T12 = GriffinParams(
    p=GOLDILOCKS.p,
    t=12,
    alpha=7,
    R=8,
    r=8,
    c=4,
    d=8,
)

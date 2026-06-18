# instances.py
# ---------------------------------------------------------------------------
# Concrete, named parameter sets for Anemoi.
#
# Each entry is a ready-to-use AnemoiParams instance pinned to a specific field
# from utils/field.py, so every consumer agrees on the exact same parameters.
# Naming convention: ANEMOI_<FIELD>_<VARIANT> (T<state size>).
# ---------------------------------------------------------------------------

from utils.field import (
    BLS12_381_BASE, BLS12_381_SCALAR,
    BLS12_377_BASE, BLS12_377_SCALAR,
    BN254_BASE, BN254_SCALAR,
    PALLAS, VESTA, GOLDILOCKS,
)
from anemoi.params import AnemoiParams

# All instances target kappa=128. Round numbers match the reference derivation
# (AnemoiParams._init_rounds) and were recorded from the upstream sage implementation.
#
# Sponge parameters: for ~256-bit fields a single capacity/digest element suffices
# (r = t-1, c = 1, d = 1); for Goldilocks four 64-bit elements are needed (c = d = 4).

# ---------------------------------------------------------------------------
# BLS12-381 base field instances
# ---------------------------------------------------------------------------

ANEMOI_BLS12_381_BASE_T2 = AnemoiParams(
    p=BLS12_381_BASE.p,
    l=1,
    alpha=BLS12_381_BASE.alpha,
    g=BLS12_381_BASE.generator,
    R=21,
    r=1,
    c=1,
    d=1,
)

ANEMOI_BLS12_381_BASE_T4 = AnemoiParams(
    p=BLS12_381_BASE.p,
    l=2,
    alpha=BLS12_381_BASE.alpha,
    g=BLS12_381_BASE.generator,
    R=14,
    r=3,
    c=1,
    d=1,
)

ANEMOI_BLS12_381_BASE_T6 = AnemoiParams(
    p=BLS12_381_BASE.p,
    l=3,
    alpha=BLS12_381_BASE.alpha,
    g=BLS12_381_BASE.generator,
    R=12,
    r=5,
    c=1,
    d=1,
)

# ---------------------------------------------------------------------------
# BLS12-381 scalar field instances (base field of the Jubjub curve)
# ---------------------------------------------------------------------------

ANEMOI_BLS12_381_SCALAR_T2 = AnemoiParams(
    p=BLS12_381_SCALAR.p,
    l=1,
    alpha=BLS12_381_SCALAR.alpha,
    g=BLS12_381_SCALAR.generator,
    R=21,
    r=1,
    c=1,
    d=1,
)

ANEMOI_BLS12_381_SCALAR_T4 = AnemoiParams(
    p=BLS12_381_SCALAR.p,
    l=2,
    alpha=BLS12_381_SCALAR.alpha,
    g=BLS12_381_SCALAR.generator,
    R=14,
    r=3,
    c=1,
    d=1,
)

ANEMOI_BLS12_381_SCALAR_T6 = AnemoiParams(
    p=BLS12_381_SCALAR.p,
    l=3,
    alpha=BLS12_381_SCALAR.alpha,
    g=BLS12_381_SCALAR.generator,
    R=12,
    r=5,
    c=1,
    d=1,
)

# ---------------------------------------------------------------------------
# BLS12-377 base field instances (scalar field of the BW6-761 curve)
# ---------------------------------------------------------------------------

ANEMOI_BLS12_377_BASE_T2 = AnemoiParams(
    p=BLS12_377_BASE.p,
    l=1,
    alpha=BLS12_377_BASE.alpha,
    g=BLS12_377_BASE.generator,
    R=21,
    r=1,
    c=1,
    d=1,
)

ANEMOI_BLS12_377_BASE_T4 = AnemoiParams(
    p=BLS12_377_BASE.p,
    l=2,
    alpha=BLS12_377_BASE.alpha,
    g=BLS12_377_BASE.generator,
    R=14,
    r=3,
    c=1,
    d=1,
)

ANEMOI_BLS12_377_BASE_T6 = AnemoiParams(
    p=BLS12_377_BASE.p,
    l=3,
    alpha=BLS12_377_BASE.alpha,
    g=BLS12_377_BASE.generator,
    R=12,
    r=5,
    c=1,
    d=1,
)

# ---------------------------------------------------------------------------
# BLS12-377 scalar field instances (base field of the Ed-on-BLS12-377 curve)
# ---------------------------------------------------------------------------

ANEMOI_BLS12_377_SCALAR_T2 = AnemoiParams(
    p=BLS12_377_SCALAR.p,
    l=1,
    alpha=BLS12_377_SCALAR.alpha,
    g=BLS12_377_SCALAR.generator,
    R=19,
    r=1,
    c=1,
    d=1,
)

ANEMOI_BLS12_377_SCALAR_T4 = AnemoiParams(
    p=BLS12_377_SCALAR.p,
    l=2,
    alpha=BLS12_377_SCALAR.alpha,
    g=BLS12_377_SCALAR.generator,
    R=13,
    r=3,
    c=1,
    d=1,
)

ANEMOI_BLS12_377_SCALAR_T6 = AnemoiParams(
    p=BLS12_377_SCALAR.p,
    l=3,
    alpha=BLS12_377_SCALAR.alpha,
    g=BLS12_377_SCALAR.generator,
    R=11,
    r=5,
    c=1,
    d=1,
)

# ---------------------------------------------------------------------------
# BN254 base field instances
# ---------------------------------------------------------------------------

ANEMOI_BN254_BASE_T2 = AnemoiParams(
    p=BN254_BASE.p,
    l=1,
    alpha=BN254_BASE.alpha,
    g=BN254_BASE.generator,
    R=21,
    r=1,
    c=1,
    d=1,
)

ANEMOI_BN254_BASE_T4 = AnemoiParams(
    p=BN254_BASE.p,
    l=2,
    alpha=BN254_BASE.alpha,
    g=BN254_BASE.generator,
    R=14,
    r=3,
    c=1,
    d=1,
)

ANEMOI_BN254_BASE_T6 = AnemoiParams(
    p=BN254_BASE.p,
    l=3,
    alpha=BN254_BASE.alpha,
    g=BN254_BASE.generator,
    R=12,
    r=5,
    c=1,
    d=1,
)

# ---------------------------------------------------------------------------
# BN254 scalar field instances
# ---------------------------------------------------------------------------

ANEMOI_BN254_SCALAR_T2 = AnemoiParams(
    p=BN254_SCALAR.p,
    l=1,
    alpha=BN254_SCALAR.alpha,
    g=BN254_SCALAR.generator,
    R=21,
    r=1,
    c=1,
    d=1,
)

ANEMOI_BN254_SCALAR_T4 = AnemoiParams(
    p=BN254_SCALAR.p,
    l=2,
    alpha=BN254_SCALAR.alpha,
    g=BN254_SCALAR.generator,
    R=14,
    r=3,
    c=1,
    d=1,
)

ANEMOI_BN254_SCALAR_T6 = AnemoiParams(
    p=BN254_SCALAR.p,
    l=3,
    alpha=BN254_SCALAR.alpha,
    g=BN254_SCALAR.generator,
    R=12,
    r=5,
    c=1,
    d=1,
)

# ---------------------------------------------------------------------------
# Pallas base field instances
# ---------------------------------------------------------------------------

ANEMOI_PALLAS_T2 = AnemoiParams(
    p=PALLAS.p,
    l=1,
    alpha=PALLAS.alpha,
    g=PALLAS.generator,
    R=21,
    r=1,
    c=1,
    d=1,
)

ANEMOI_PALLAS_T4 = AnemoiParams(
    p=PALLAS.p,
    l=2,
    alpha=PALLAS.alpha,
    g=PALLAS.generator,
    R=14,
    r=3,
    c=1,
    d=1,
)

ANEMOI_PALLAS_T6 = AnemoiParams(
    p=PALLAS.p,
    l=3,
    alpha=PALLAS.alpha,
    g=PALLAS.generator,
    R=12,
    r=5,
    c=1,
    d=1,
)

# ---------------------------------------------------------------------------
# Vesta base field instances
# ---------------------------------------------------------------------------

ANEMOI_VESTA_T2 = AnemoiParams(
    p=VESTA.p,
    l=1,
    alpha=VESTA.alpha,
    g=VESTA.generator,
    R=21,
    r=1,
    c=1,
    d=1,
)

ANEMOI_VESTA_T4 = AnemoiParams(
    p=VESTA.p,
    l=2,
    alpha=VESTA.alpha,
    g=VESTA.generator,
    R=14,
    r=3,
    c=1,
    d=1,
)

ANEMOI_VESTA_T6 = AnemoiParams(
    p=VESTA.p,
    l=3,
    alpha=VESTA.alpha,
    g=VESTA.generator,
    R=12,
    r=5,
    c=1,
    d=1,
)

# ---------------------------------------------------------------------------
# Goldilocks instances (c = d = 4 for 128-bit security over a 64-bit field)
# ---------------------------------------------------------------------------

ANEMOI_GOLDILOCKS_T8 = AnemoiParams(
    p=GOLDILOCKS.p,
    l=4,
    alpha=GOLDILOCKS.alpha,
    g=GOLDILOCKS.generator,
    R=11,
    r=4,
    c=4,
    d=4,
)

ANEMOI_GOLDILOCKS_T10 = AnemoiParams(
    p=GOLDILOCKS.p,
    l=5,
    alpha=GOLDILOCKS.alpha,
    g=GOLDILOCKS.generator,
    R=11,
    r=6,
    c=4,
    d=4,
)

ANEMOI_GOLDILOCKS_T12 = AnemoiParams(
    p=GOLDILOCKS.p,
    l=6,
    alpha=GOLDILOCKS.alpha,
    g=GOLDILOCKS.generator,
    R=10,
    r=8,
    c=4,
    d=4,
)

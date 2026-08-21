# instances.py
# ---------------------------------------------------------------------------
# Concrete, named parameter sets for GMiMC and GMiMC2.
#
# Each entry is a ready-to-use GMiMCParams instance pinned to a specific field
# from utils/field.py, so every consumer agrees on the exact same parameters.
# Naming convention: GMIMC_<FIELD>_<VARIANT>.
# ---------------------------------------------------------------------------

from utils.field import BABYBEAR, BN254_SCALAR, BLS12_381_SCALAR, KOALABEAR, ST, GOLDILOCKS, MERSENNE31
from gmimc.params import GMiMCParams, GMiMC2Params

# ---------------------------------------------------------------------------
# GMiMC
# ---------------------------------------------------------------------------

GMIMC_BN254_T3 = GMiMCParams(p=BN254_SCALAR.p,      t=3,    R=228,  r=2,    c=1,    d=1,)
GMIMC_BN254_T4 = GMiMCParams(p=BN254_SCALAR.p,      t=4,    R=231,  r=3,    c=1,    d=1,)

GMIMC_BLS12_T3 = GMiMCParams(p=BLS12_381_SCALAR.p,  t=3,    R=228,  r=2,    c=1,    d=1,)
GMIMC_BLS12_T4 = GMiMCParams(p=BLS12_381_SCALAR.p,  t=4,    R=231,  r=3,    c=1,    d=1,)

GMIMC_GOLDILOCKS_T8  = GMiMCParams(p=GOLDILOCKS.p,   t=8,    R=68,   r=4,    c=4,    d=4,)
GMIMC_GOLDILOCKS_T12 = GMiMCParams(p=GOLDILOCKS.p,   t=12,   R=93,   r=8,    c=4,    d=4,)

GMIMC_BABYBEAR_T16 = GMiMCParams(p=BABYBEAR.p,  t=16,   R=158,  r=8,    c=8,    d=8,)
GMIMC_BABYBEAR_T24 = GMiMCParams(p=BABYBEAR.p,  t=24,   R=335,  r=16,   c=8,    d=8,)

GMIMC_KOALABEAR_T16 = GMiMCParams(p=KOALABEAR.p,  t=16,   R=158,  r=8,    c=8,    d=8,)
GMIMC_KOALABEAR_T24 = GMiMCParams(p=KOALABEAR.p,  t=24,   R=335,  r=16,   c=8,    d=8,)

GMIMC_MERSENNE_T16 = GMiMCParams(p=MERSENNE31.p,  t=16,   R=158,  r=8,    c=8,    d=8,)
GMIMC_MERSENNE_T24 = GMiMCParams(p=MERSENNE31.p,  t=24,   R=335,  r=16,   c=8,    d=8,)

# ---------------------------------------------------------------------------
# GMiMC2
# ---------------------------------------------------------------------------

GMIMC2_BN254_T4 = GMiMC2Params(p=BN254_SCALAR.p,        t=4,   alpha=8,    R=64,  r=3,    c=1,    d=1,)

GMIMC2_BLS12_T4 = GMiMC2Params(p=BLS12_381_SCALAR.p,    t=4,   alpha=8,   R=64,  r=3,    c=1,    d=1,)

GMIMC2_GOLDILOCKS_T8  = GMiMC2Params(p=GOLDILOCKS.p,    t=8,   alpha=4,    R=88,   r=4,    c=4,    d=4,)
GMIMC2_GOLDILOCKS_T12 = GMiMC2Params(p=GOLDILOCKS.p,    t=12,  alpha=4,    R=96,   r=8,    c=4,    d=4,)

GMIMC2_BABYBEAR_T16 = GMiMC2Params(p=BABYBEAR.p,    t=16,   alpha=2,    R=176,  r=8,    c=8,    d=8,)
GMIMC2_BABYBEAR_T24 = GMiMC2Params(p=BABYBEAR.p,    t=24,   alpha=2,    R=264,  r=16,   c=8,    d=8,)

GMIMC2_KOALABEAR_T16 = GMiMC2Params(p=KOALABEAR.p,  t=16, alpha=2,    R=176,  r=8,    c=8,    d=8,)
GMIMC2_KOALABEAR_T24 = GMiMC2Params(p=KOALABEAR.p,  t=24, alpha=2,    R=264,  r=16,   c=8,    d=8,)

GMIMC2_MERSENNE_T16 = GMiMC2Params(p=MERSENNE31.p,  t=16, alpha=2,    R=176,  r=8,    c=8,    d=8,)
GMIMC2_MERSENNE_T24 = GMiMC2Params(p=MERSENNE31.p,  t=24, alpha=2,    R=264,  r=16,   c=8,    d=8,)

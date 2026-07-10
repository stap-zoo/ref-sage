# instances.py
# ---------------------------------------------------------------------------
# Concrete, named parameter sets for GMiMC.
#
# Each entry is a ready-to-use GMiMCParams instance pinned to a specific field
# from utils/field.py, so every consumer agrees on the exact same parameters.
# Naming convention: GMIMC_<FIELD>_<VARIANT>.
# ---------------------------------------------------------------------------

from utils.field import BLS12_381_SCALAR, BN254_SCALAR
from gmimc.params import GMiMCParams

# ---------------------------------------------------------------------------
# t=3 instances (kappa=128, c=1)
# ---------------------------------------------------------------------------

GMIMC_BN254_T3 = GMiMCParams(
    p=BN254_SCALAR.p,
    t=3,
    R=228,
    r=2,
    c=1,
    d=1,
)

GMIMC_BLS12_T3 = GMiMCParams(
    p=BLS12_381_SCALAR.p,
    t=3,
    R=228,
    r=2,
    c=1,
    d=1,
)

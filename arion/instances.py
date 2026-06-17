# instances.py
# ---------------------------------------------------------------------------
# Concrete, named parameter sets for Arion.
#
# Each entry is a ready-to-use ArionParams instance pinned to a specific field
# from fields.py, so every consumer agrees on the exact same parameters.
# Naming convention: ARION_<FIELD>_<VARIANT>.
# ---------------------------------------------------------------------------

from fields import BLS12_381_SCALAR
from arion.params import ArionParams

# ---------------------------------------------------------------------------
# BLS12-381 t=3 instance (branches=3, rounds=6 -- Arion.sage defaults)
# ---------------------------------------------------------------------------

ARION_BLS12_T3 = ArionParams(
    p=BLS12_381_SCALAR.p,
    t=3,
    R=6,
    alpha1=5,
    alpha2=257,
    r=2,
    c=1,
    d=2,
)

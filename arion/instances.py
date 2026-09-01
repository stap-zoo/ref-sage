# instances.py
# Named parameter sets for Arion (<PREFIX>_<FIELD>_<VARIANT>).

from utils.field import BLS12_381_SCALAR
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
    sponge=dict(r=2, c=1, d=2),
)

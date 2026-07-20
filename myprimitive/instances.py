# instances.py
# ---------------------------------------------------------------------------
# Concrete, named parameter sets for MyPrimitive.
#
# Each entry is a ready-to-use MyPrimitiveParams instance pinned to a specific
# field from utils/field.py. To goal is that for a specific instance, every consumer 
# agrees on the exact same parameters.
#
# Naming convention: <PRIMITIVE>_<FIELD>_<VARIANT>, e.g. MYPRIMITIVE_GOLDILOCKS_T2.
# ---------------------------------------------------------------------------

from utils.field import GOLDILOCKS         # predefined field; add a new Field to utils/field.py if yours is missing
from myprimitive.params import MyPrimitiveParams
from utils.matrix import circulant

# ---------------------------------------------------------------------------
# GOLDILOCKS field instances  (p = 2^64 - 2^32 + 1)
# ---------------------------------------------------------------------------
#
# NOTE: the values below are illustrative placeholders to show the call shape;
# they MAY NOT satisfy any real constraints (M must be t x t, rcons must match
# the round schedule, ...). Replace them with valid parameters for your design.
MYPRIMITIVE_GOLDILOCKS_T3 = MyPrimitiveParams(
    p=GOLDILOCKS.p,                   # field characteristic, taken from the Field entry
    t=3,                              # state size (the sponge mode needs t >= 3)
    alpha=GOLDILOCKS.alpha,           # S-box exponent recommended for this field
    R=5,                              # number of rounds (omit to derive via _init_rounds)
    rcons=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],  # R x t round constants (omit to derive)
    M=circulant([1, 2, 3]),           # t x t MDS matrix (omit to derive via _init_mat)
    r=2,                              # sponge rate (set to None if you do not specify Sponge mode)
    c=1,                              # sponge capacity (note r + c == t)
    d=1,                              # digest size
    kappa=32,                         # security parameter (bits)
    toy=True                          # toy mode (if true, security constraints only warn but do not raise an exception)
)
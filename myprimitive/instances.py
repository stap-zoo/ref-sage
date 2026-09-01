# instances.py
# Named parameter sets for MyPrimitive (<PREFIX>_<FIELD>_<VARIANT>).

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
# All share the same permutation; only the mode dicts differ.
_COMMON = dict(
    p=GOLDILOCKS.p,                   # field characteristic, taken from the Field entry
    t=3,                              # state size (the sponge mode needs t >= 3)
    alpha=GOLDILOCKS.alpha,           # S-box exponent recommended for this field
    R=5,                              # number of rounds (omit to derive via _init_rounds)
    rcons=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],  # R x t round constants (omit to derive)
    M=circulant([1, 2, 3]),           # t x t MDS matrix (omit to derive via _init_mat)
    toy=True,                         # toy PERMUTATION (placeholders); security constraints only warn
)

# --- Sponge instance: defines only the sponge hash mode (comp defaults to None) ---
MYPRIMITIVE_GOLDILOCKS_T3_SPONGE = MyPrimitiveParams(
    **_COMMON,
    sponge=dict(r=2, c=1, d=1),       # rate / capacity / digest
    kappa=32,                         # security parameter (bits); c=1,d=1 meet the 32-bit floor
)

# --- Compression instance: defines only the feed-forward compression mode (sponge=None) ---
MYPRIMITIVE_GOLDILOCKS_T3_COMPRESS = MyPrimitiveParams(
    **_COMMON,
    sponge=None,                      # no sponge mode
    comp=dict(d=1),                   # truncation to digest d=1 (t=3 -> 1)
    kappa=32,
)

# --- Toy-mode variants: same permutation, but the ACTIVELY USED mode dict is
# overwritten with an explicitly toy configuration (toy=True). Here kappa=128 puts
# the mode below its security floor, so building the mode WARNS instead of raising
# (the mode's toy flag is independent of the permutation's -- see the tests). The
# dict may likewise override the mode's default kind via a "kind" key. ---
MYPRIMITIVE_GOLDILOCKS_T3_SPONGE_TOY = MyPrimitiveParams(
    **_COMMON,
    sponge=dict(r=2, c=1, d=1, toy=True),   # below the kappa=128 floor -> toy so it only warns
    kappa=128,
)

MYPRIMITIVE_GOLDILOCKS_T3_COMPRESS_TOY = MyPrimitiveParams(
    **_COMMON,
    sponge=None,
    comp=dict(d=1, toy=True),               # below the kappa=128 floor -> toy so it only warns
    kappa=128,
)

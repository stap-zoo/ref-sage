# instances.py
# ---------------------------------------------------------------------------
# Concrete, named parameter sets for Polocolo.
#
# Each entry is a ready-to-use PolocoloParams instance pinned to one of the two
# official fields (the BLS12-381 and BN254 scalar fields from utils/field.py),
# so that every consumer agrees on the exact same parameters. m and R are spelled
# out (Table 1; Table 7 for the tight variants) even though _init_m/_init_rounds would
# derive the same values -- the test suite asserts that they agree. All remaining
# constants (sigma and the lookup tables, the MDS matrix, the round constants)
# are derived deterministically inside PolocoloParams. The sponge uses one
# capacity element and a one-element digest (r = t-1, c = 1, d = 1), which gives
# 128-bit security on these ~255-bit fields (Section 2.2 of the paper).
#
# The tight instances (Section 6.1) drop the security margin: the round number
# is chosen so the best attack costs just 2^128 instead of 2^160. The paper
# defines them only for t in {3, 4, 6, 8}.
#
# NOTE on the reference implementation (polocolo/Polocolo-main/plain/): of its
# eight BLS12 instances only T4_TIGHT matches the paper exactly. Its T3/T3_TIGHT
# use a mistyped (non-MDS) M3, its T4 uses m = 1024 instead of 512, its
# T6/T6_TIGHT use m = 128, and its T6/T8/T8_TIGHT even mix the m of the round
# constants with a different m in the S-box (see params.py and the test suite,
# which reconstructs those variants via explicit overrides and checks them
# against reference test vectors). The instances below follow the paper.
#
# Naming convention: <PRIMITIVE>_<FIELD>_<VARIANT>, e.g. POLOCOLO_BLS12_381_SCALAR_T3.
# ---------------------------------------------------------------------------

from utils.field import BLS12_381_SCALAR, BN254_SCALAR
from polocolo.params import PolocoloParams

# ---------------------------------------------------------------------------
# BLS12-381 scalar field instances  (p = 0x73eda753...ffff00000001, 255 bits)
# ---------------------------------------------------------------------------

# Recommended parameters (Table 1)
POLOCOLO_BLS12_381_SCALAR_T3 = PolocoloParams(p=BLS12_381_SCALAR.p, t=3, m=1024, R=6, g = BLS12_381_SCALAR.generator, field_label="BLS12", r=2, c=1, d=1)
POLOCOLO_BLS12_381_SCALAR_T4 = PolocoloParams(p=BLS12_381_SCALAR.p, t=4, m=512,  R=5, g = BLS12_381_SCALAR.generator, field_label="BLS12", r=3, c=1, d=1)
POLOCOLO_BLS12_381_SCALAR_T5 = PolocoloParams(p=BLS12_381_SCALAR.p, t=5, m=128,  R=5, g = BLS12_381_SCALAR.generator, field_label="BLS12", r=4, c=1, d=1)
POLOCOLO_BLS12_381_SCALAR_T6 = PolocoloParams(p=BLS12_381_SCALAR.p, t=6, m=64,   R=5, g = BLS12_381_SCALAR.generator, field_label="BLS12", r=5, c=1, d=1)
POLOCOLO_BLS12_381_SCALAR_T7 = PolocoloParams(p=BLS12_381_SCALAR.p, t=7, m=32,   R=5, g = BLS12_381_SCALAR.generator, field_label="BLS12", r=6, c=1, d=1)
POLOCOLO_BLS12_381_SCALAR_T8 = PolocoloParams(p=BLS12_381_SCALAR.p, t=8, m=32,   R=5, g = BLS12_381_SCALAR.generator, field_label="BLS12", r=7, c=1, d=1)

# Tight parameters, no security margin (Table 7)
POLOCOLO_BLS12_381_SCALAR_T3_TIGHT = PolocoloParams(p=BLS12_381_SCALAR.p, t=3, m=1024, R=5, tight=True, g = BLS12_381_SCALAR.generator, field_label="BLS12", r=2, c=1, d=1)
POLOCOLO_BLS12_381_SCALAR_T4_TIGHT = PolocoloParams(p=BLS12_381_SCALAR.p, t=4, m=1024, R=4, tight=True, g = BLS12_381_SCALAR.generator, field_label="BLS12", r=3, c=1, d=1)
POLOCOLO_BLS12_381_SCALAR_T6_TIGHT = PolocoloParams(p=BLS12_381_SCALAR.p, t=6, m=64,   R=4, tight=True, g = BLS12_381_SCALAR.generator, field_label="BLS12", r=5, c=1, d=1)
POLOCOLO_BLS12_381_SCALAR_T8_TIGHT = PolocoloParams(p=BLS12_381_SCALAR.p, t=8, m=32,   R=4, tight=True, g = BLS12_381_SCALAR.generator, field_label="BLS12", r=7, c=1, d=1)

# ---------------------------------------------------------------------------
# BN254 scalar field instances  (p = 0x30644e72...f593f0000001, 254 bits)
# ---------------------------------------------------------------------------

# Recommended parameters (Table 1)
POLOCOLO_BN254_SCALAR_T3 = PolocoloParams(p=BN254_SCALAR.p, t=3, m=1024, R=6, g = BN254_SCALAR.generator, field_label="BN254", r=2, c=1, d=1)
POLOCOLO_BN254_SCALAR_T4 = PolocoloParams(p=BN254_SCALAR.p, t=4, m=512,  R=5, g = BN254_SCALAR.generator, field_label="BN254", r=3, c=1, d=1)
POLOCOLO_BN254_SCALAR_T5 = PolocoloParams(p=BN254_SCALAR.p, t=5, m=128,  R=5, g = BN254_SCALAR.generator, field_label="BN254", r=4, c=1, d=1)
POLOCOLO_BN254_SCALAR_T6 = PolocoloParams(p=BN254_SCALAR.p, t=6, m=64,   R=5, g = BN254_SCALAR.generator, field_label="BN254", r=5, c=1, d=1)
POLOCOLO_BN254_SCALAR_T7 = PolocoloParams(p=BN254_SCALAR.p, t=7, m=32,   R=5, g = BN254_SCALAR.generator, field_label="BN254", r=6, c=1, d=1)
POLOCOLO_BN254_SCALAR_T8 = PolocoloParams(p=BN254_SCALAR.p, t=8, m=32,   R=5, g = BN254_SCALAR.generator, field_label="BN254", r=7, c=1, d=1)

# Tight parameters, no security margin (Table 7)
POLOCOLO_BN254_SCALAR_T3_TIGHT = PolocoloParams(p=BN254_SCALAR.p, t=3, m=1024, R=5, tight=True, g = BN254_SCALAR.generator, field_label="BN254", r=2, c=1, d=1)
POLOCOLO_BN254_SCALAR_T4_TIGHT = PolocoloParams(p=BN254_SCALAR.p, t=4, m=1024, R=4, tight=True, g = BN254_SCALAR.generator, field_label="BN254", r=3, c=1, d=1)
POLOCOLO_BN254_SCALAR_T6_TIGHT = PolocoloParams(p=BN254_SCALAR.p, t=6, m=64,   R=4, tight=True, g = BN254_SCALAR.generator, field_label="BN254", r=5, c=1, d=1)
POLOCOLO_BN254_SCALAR_T8_TIGHT = PolocoloParams(p=BN254_SCALAR.p, t=8, m=32,   R=4, tight=True, g = BN254_SCALAR.generator, field_label="BN254", r=7, c=1, d=1)

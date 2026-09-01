# instances.py
# Named parameter sets for Polocolo (<PREFIX>_<FIELD>_<VARIANT>).

from utils.field import BLS12_381_SCALAR, BN254_SCALAR
from polocolo.params import PolocoloParams

# ---------------------------------------------------------------------------
# BLS12-381 scalar field instances  (p = 0x73eda753...ffff00000001, 255 bits)
# ---------------------------------------------------------------------------

# Recommended parameters (Table 1)
POLOCOLO_BLS12_381_SCALAR_T3 = PolocoloParams(p=BLS12_381_SCALAR.p, t=3, m=1024, R=6, g = BLS12_381_SCALAR.generator, field_label="BLS12", sponge=dict(r=2, c=1, d=1))
POLOCOLO_BLS12_381_SCALAR_T4 = PolocoloParams(p=BLS12_381_SCALAR.p, t=4, m=512,  R=5, g = BLS12_381_SCALAR.generator, field_label="BLS12", sponge=dict(r=3, c=1, d=1))
POLOCOLO_BLS12_381_SCALAR_T5 = PolocoloParams(p=BLS12_381_SCALAR.p, t=5, m=128,  R=5, g = BLS12_381_SCALAR.generator, field_label="BLS12", sponge=dict(r=4, c=1, d=1))
POLOCOLO_BLS12_381_SCALAR_T6 = PolocoloParams(p=BLS12_381_SCALAR.p, t=6, m=64,   R=5, g = BLS12_381_SCALAR.generator, field_label="BLS12", sponge=dict(r=5, c=1, d=1))
POLOCOLO_BLS12_381_SCALAR_T7 = PolocoloParams(p=BLS12_381_SCALAR.p, t=7, m=32,   R=5, g = BLS12_381_SCALAR.generator, field_label="BLS12", sponge=dict(r=6, c=1, d=1))
POLOCOLO_BLS12_381_SCALAR_T8 = PolocoloParams(p=BLS12_381_SCALAR.p, t=8, m=32,   R=5, g = BLS12_381_SCALAR.generator, field_label="BLS12", sponge=dict(r=7, c=1, d=1))

# Tight parameters, no security margin (Table 7)
POLOCOLO_BLS12_381_SCALAR_T3_TIGHT = PolocoloParams(p=BLS12_381_SCALAR.p, t=3, m=1024, R=5, tight=True, g = BLS12_381_SCALAR.generator, field_label="BLS12", sponge=dict(r=2, c=1, d=1))
POLOCOLO_BLS12_381_SCALAR_T4_TIGHT = PolocoloParams(p=BLS12_381_SCALAR.p, t=4, m=1024, R=4, tight=True, g = BLS12_381_SCALAR.generator, field_label="BLS12", sponge=dict(r=3, c=1, d=1))
POLOCOLO_BLS12_381_SCALAR_T6_TIGHT = PolocoloParams(p=BLS12_381_SCALAR.p, t=6, m=64,   R=4, tight=True, g = BLS12_381_SCALAR.generator, field_label="BLS12", sponge=dict(r=5, c=1, d=1))
POLOCOLO_BLS12_381_SCALAR_T8_TIGHT = PolocoloParams(p=BLS12_381_SCALAR.p, t=8, m=32,   R=4, tight=True, g = BLS12_381_SCALAR.generator, field_label="BLS12", sponge=dict(r=7, c=1, d=1))

# ---------------------------------------------------------------------------
# BN254 scalar field instances  (p = 0x30644e72...f593f0000001, 254 bits)
# ---------------------------------------------------------------------------

# Recommended parameters (Table 1)
POLOCOLO_BN254_SCALAR_T3 = PolocoloParams(p=BN254_SCALAR.p, t=3, m=1024, R=6, g = BN254_SCALAR.generator, field_label="BN254", sponge=dict(r=2, c=1, d=1))
POLOCOLO_BN254_SCALAR_T4 = PolocoloParams(p=BN254_SCALAR.p, t=4, m=512,  R=5, g = BN254_SCALAR.generator, field_label="BN254", sponge=dict(r=3, c=1, d=1))
POLOCOLO_BN254_SCALAR_T5 = PolocoloParams(p=BN254_SCALAR.p, t=5, m=128,  R=5, g = BN254_SCALAR.generator, field_label="BN254", sponge=dict(r=4, c=1, d=1))
POLOCOLO_BN254_SCALAR_T6 = PolocoloParams(p=BN254_SCALAR.p, t=6, m=64,   R=5, g = BN254_SCALAR.generator, field_label="BN254", sponge=dict(r=5, c=1, d=1))
POLOCOLO_BN254_SCALAR_T7 = PolocoloParams(p=BN254_SCALAR.p, t=7, m=32,   R=5, g = BN254_SCALAR.generator, field_label="BN254", sponge=dict(r=6, c=1, d=1))
POLOCOLO_BN254_SCALAR_T8 = PolocoloParams(p=BN254_SCALAR.p, t=8, m=32,   R=5, g = BN254_SCALAR.generator, field_label="BN254", sponge=dict(r=7, c=1, d=1))

# Tight parameters, no security margin (Table 7)
POLOCOLO_BN254_SCALAR_T3_TIGHT = PolocoloParams(p=BN254_SCALAR.p, t=3, m=1024, R=5, tight=True, g = BN254_SCALAR.generator, field_label="BN254", sponge=dict(r=2, c=1, d=1))
POLOCOLO_BN254_SCALAR_T4_TIGHT = PolocoloParams(p=BN254_SCALAR.p, t=4, m=1024, R=4, tight=True, g = BN254_SCALAR.generator, field_label="BN254", sponge=dict(r=3, c=1, d=1))
POLOCOLO_BN254_SCALAR_T6_TIGHT = PolocoloParams(p=BN254_SCALAR.p, t=6, m=64,   R=4, tight=True, g = BN254_SCALAR.generator, field_label="BN254", sponge=dict(r=5, c=1, d=1))
POLOCOLO_BN254_SCALAR_T8_TIGHT = PolocoloParams(p=BN254_SCALAR.p, t=8, m=32,   R=4, tight=True, g = BN254_SCALAR.generator, field_label="BN254", sponge=dict(r=7, c=1, d=1))

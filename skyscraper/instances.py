# instances.py
# ---------------------------------------------------------------------------
# Concrete, named parameter sets for Skyscraper.
#
# Each entry is a ready-to-use params instance pinned to a specific scalar field
# from utils/field.py, so every consumer agrees on the exact same parameters.
# Naming convention: SKYSCRAPER_<FIELD>_N<n>, where n is the extension degree of
# GF(p^n) the 2-branch Feistel operates over.
#
# All recommended instances use the 8-bit Bar decomposition (si = [256]*32, since
# every field here is 254-255 bits), R = 18 Feistel rounds, the Montgomery
# squaring, and Bar at rounds {6,7,10,11}. The extension modulus is x^n + beta,
# passed as the coefficient list fmod (low->high); the per-instance beta matches
# the Skyscraper reference. 
# 
# Sponge params are measured in terms of number of base field elements. 
# In particular, the always set it to one extension field element, thus r = c = d = n.
# ---------------------------------------------------------------------------

from utils.field import BLS12_381_SCALAR, BN254_SCALAR, PALLAS, VESTA
from skyscraper.params import SkyscraperParams

# Shared Bar decomposition: 32 bytes (covers any <=256-bit prime), rotation by 16.
_SI = [256] * 32

# ---------------------------------------------------------------------------
# BLS12-381 scalar field
# ---------------------------------------------------------------------------

SKYSCRAPER_BLS12_381_N1 = SkyscraperParams(p=BLS12_381_SCALAR.p, si=_SI, r=1, c=1, d=1)
SKYSCRAPER_BLS12_381_N2 = SkyscraperParams(p=BLS12_381_SCALAR.p, si=_SI, fmod=[5, 0, 1], n=2, r=2, c=2, d=2)
SKYSCRAPER_BLS12_381_N3 = SkyscraperParams(p=BLS12_381_SCALAR.p, si=_SI, fmod=[2, 0, 0, 1], n=3, r=3, c=3, d=3)

# ---------------------------------------------------------------------------
# BN254 scalar field
# ---------------------------------------------------------------------------

SKYSCRAPER_BN254_N1 = SkyscraperParams(p=BN254_SCALAR.p, si=_SI, r=1, c=1, d=1)
SKYSCRAPER_BN254_N2 = SkyscraperParams(p=BN254_SCALAR.p, si=_SI, fmod=[5, 0, 1], n=2, r=2, c=2, d=2)
SKYSCRAPER_BN254_N3 = SkyscraperParams(p=BN254_SCALAR.p, si=_SI, fmod=[3, 0, 0, 1], n=3, r=3, c=3, d=3)

# ---------------------------------------------------------------------------
# Pallas base field (= Vesta scalar field)
# ---------------------------------------------------------------------------

SKYSCRAPER_PALLAS_N1 = SkyscraperParams(p=PALLAS.p, si=_SI, r=1, c=1, d=1)
SKYSCRAPER_PALLAS_N2 = SkyscraperParams(p=PALLAS.p, si=_SI, fmod=[5, 0, 1], n=2, r=2, c=2, d=2)
SKYSCRAPER_PALLAS_N3 = SkyscraperParams(p=PALLAS.p, si=_SI, fmod=[2, 0, 0, 1], n=3, r=3, c=3, d=3)

# ---------------------------------------------------------------------------
# Vesta base field (= Pallas scalar field)
# ---------------------------------------------------------------------------

SKYSCRAPER_VESTA_N1 = SkyscraperParams(p=VESTA.p, si=_SI, r=1, c=1, d=1)
SKYSCRAPER_VESTA_N2 = SkyscraperParams(p=VESTA.p, si=_SI, fmod=[5, 0, 1], n=2, r=2, c=2, d=2)
SKYSCRAPER_VESTA_N3 = SkyscraperParams(p=VESTA.p, si=_SI, fmod=[2, 0, 0, 1], n=3, r=3, c=3, d=3)

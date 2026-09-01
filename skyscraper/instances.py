# instances.py
# Named parameter sets for Skyscraper (<PREFIX>_<FIELD>_<VARIANT>).

from utils.field import BLS12_381_SCALAR, BN254_SCALAR, PALLAS, VESTA
from skyscraper.params import SkyscraperParams

# Shared Bar decomposition: 32 bytes (covers any <=256-bit prime), rotation by 16.
_SI = [256] * 32

# ---------------------------------------------------------------------------
# BLS12-381 scalar field
# ---------------------------------------------------------------------------

SKYSCRAPER_BLS12_381_N1 = SkyscraperParams(p=BLS12_381_SCALAR.p, si=_SI, sponge=dict(r=1, c=1, d=1), comp=dict(a=2))
SKYSCRAPER_BLS12_381_N2 = SkyscraperParams(p=BLS12_381_SCALAR.p, si=_SI, fmod=[5, 0, 1], n=2, sponge=dict(r=2, c=2, d=2), comp=dict(a=2))
SKYSCRAPER_BLS12_381_N3 = SkyscraperParams(p=BLS12_381_SCALAR.p, si=_SI, fmod=[2, 0, 0, 1], n=3, sponge=dict(r=3, c=3, d=3), comp=dict(a=2))

# ---------------------------------------------------------------------------
# BN254 scalar field
# ---------------------------------------------------------------------------

SKYSCRAPER_BN254_N1 = SkyscraperParams(p=BN254_SCALAR.p, si=_SI, sponge=dict(r=1, c=1, d=1), comp=dict(a=2))
SKYSCRAPER_BN254_N2 = SkyscraperParams(p=BN254_SCALAR.p, si=_SI, fmod=[5, 0, 1], n=2, sponge=dict(r=2, c=2, d=2), comp=dict(a=2))
SKYSCRAPER_BN254_N3 = SkyscraperParams(p=BN254_SCALAR.p, si=_SI, fmod=[3, 0, 0, 1], n=3, sponge=dict(r=3, c=3, d=3), comp=dict(a=2))

# ---------------------------------------------------------------------------
# Pallas base field (= Vesta scalar field)
# ---------------------------------------------------------------------------

SKYSCRAPER_PALLAS_N1 = SkyscraperParams(p=PALLAS.p, si=_SI, sponge=dict(r=1, c=1, d=1), comp=dict(a=2))
SKYSCRAPER_PALLAS_N2 = SkyscraperParams(p=PALLAS.p, si=_SI, fmod=[5, 0, 1], n=2, sponge=dict(r=2, c=2, d=2), comp=dict(a=2))
SKYSCRAPER_PALLAS_N3 = SkyscraperParams(p=PALLAS.p, si=_SI, fmod=[2, 0, 0, 1], n=3, sponge=dict(r=3, c=3, d=3), comp=dict(a=2))

# ---------------------------------------------------------------------------
# Vesta base field (= Pallas scalar field)
# ---------------------------------------------------------------------------

SKYSCRAPER_VESTA_N1 = SkyscraperParams(p=VESTA.p, si=_SI, sponge=dict(r=1, c=1, d=1), comp=dict(a=2))
SKYSCRAPER_VESTA_N2 = SkyscraperParams(p=VESTA.p, si=_SI, fmod=[5, 0, 1], n=2, sponge=dict(r=2, c=2, d=2), comp=dict(a=2))
SKYSCRAPER_VESTA_N3 = SkyscraperParams(p=VESTA.p, si=_SI, fmod=[2, 0, 0, 1], n=3, sponge=dict(r=3, c=3, d=3), comp=dict(a=2))

# instances.py
# ---------------------------------------------------------------------------
# Concrete, named parameter sets for Grendel.
#
# The Grendel paper (https://eprint.iacr.org/2021/984) defines no table of
# recommended instances. 
# 
# The only concrete parameters it mentions are the two 16-bit toy fields 
# p = 65519 (p = 3 mod 4) and p = 65393 (p = 1 mod 4) with t = 2, r = c = 1,
# used for the Groebner-basis experiments. Those two are registered
# here, clearly marked TOY_: they are for cryptanalysis / regression testing
# and are NOT secure (constructing them raises ParamRecommendationWarning,
# which is the expected signal).
#
# Naming convention: <PRIMITIVE>_<FIELD>_<VARIANT>, with the TOY_ prefix for
# non-recommended sets.
# ---------------------------------------------------------------------------

from grendel.params import GrendelParams

# ---------------------------------------------------------------------------
# Toy instances of the paper's Groebner-basis experiments
# ---------------------------------------------------------------------------

# p = 3 mod 4, so alpha derives to 2; R derives to 7 at kappa = 16.
TOY_GRENDEL_65519_T2 = GrendelParams(
    p=65519,
    t=2,
    g=11,
    r=1,
    c=1,
    d=1,
    kappa=16, # sponge constraint min(r,c)*log2(p) >= kappa, for r=c=1 and log2(p)=16
)

# p = 1 mod 4, so alpha derives to 3; R derives to 5 at kappa = 16.
TOY_GRENDEL_65393_T2 = GrendelParams(
    p=65393,
    t=2,
    g=3,
    r=1,
    c=1,
    d=1,
    kappa=16, # sponge constraint min(r,c)*log2(p) >= kappa, for r=c=1 and log2(p)=16
)

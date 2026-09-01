# instances.py
# Named parameter sets for Grendel (<PREFIX>_<FIELD>_<VARIANT>).

from grendel.params import GrendelParams

# ---------------------------------------------------------------------------
# Toy instances of the paper's Groebner-basis experiments
# ---------------------------------------------------------------------------

# p = 3 mod 4, so alpha derives to 2; R derives to 7 at kappa = 16. kappa also seeds
# _init_cons's SHAKE256 stream (f"grendel-{p}-{t}-{kappa}"), so it can't be tuned to
# match the sponge floor exactly without changing the round constants and breaking the
# KATs below -- toy=True is what silences the (accurate) capacity/digest/root-finding
# shortfall warnings for c=d=1 over this 16-bit field, kappa stays at its original value.
TOY_GRENDEL_65519_T2 = GrendelParams(
    p=65519,
    t=2,
    g=11,
    sponge=dict(r=1, c=1, d=1, toy=True),
    kappa=16,
    toy=True,
)

# p = 1 mod 4, so alpha derives to 3; R derives to 5 at kappa = 16. Same reasoning as above.
TOY_GRENDEL_65393_T2 = GrendelParams(
    p=65393,
    t=2,
    g=3,
    sponge=dict(r=1, c=1, d=1, toy=True),
    kappa=16,
    toy=True,
)

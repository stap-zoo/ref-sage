"""Export known-answer test vectors for an arithmetization-oriented hash.

Usage:
    sage -python export_kat.py <construction>

where <construction> is one of:
    gmimc, neptune, rescue-prime, anemoi, arion

"""

import argparse
import importlib
import json
import warnings

from utils.field import BN254_SCALAR, BN254_BASE, BLS12_381_SCALAR, BLS12_381_BASE

# ---------------------------------------------------------------------------
# Construction registry: key -> (hash module, hash class, instances module,
#                                 instance-name prefix)
# The prefix selects the relevant instances from the module's namespace; the
# prime filter below then keeps only the BN254 / BLS12-381 ones.
# ---------------------------------------------------------------------------
CONSTRUCTIONS = {
    "gmimc":        ("gmimc.hash",      "GMiMC",       "gmimc.instances",      "GMIMC_"),
    "neptune":      ("hades.hash",      "Neptune",     "hades.instances",      "NEPTUNE_"),
    "rescue-prime": ("marvellous.hash", "RescuePrime", "marvellous.instances", "RESCUE_PRIME_"),
    "anemoi":       ("anemoi.hash",     "Anemoi",      "anemoi.instances",     "ANEMOI_"),
    "arion":        ("arion.hash",      "Arion",       "arion.instances",      "ARION_"),
}

# Aliases accepted on the command line (all normalised to the keys above).
ALIASES = {
    "rescue": "rescue-prime",
    "rescueprime": "rescue-prime",
    "rescue_prime": "rescue-prime",
}

# Primes we emit vectors for: the scalar and base fields of BN254 and BLS12-381.
ALLOWED_PRIMES = {BN254_SCALAR.p, BN254_BASE.p, BLS12_381_SCALAR.p, BLS12_381_BASE.p}


def emit(vectors, modes_seen, label, mode, h, inp_ints, produce):
    """Run `produce` on the field-encoded input and append a vector.

    `produce` receives the input as a list of field elements and returns the
    output as a list of field elements. If the mode is not defined for this
    instance (e.g. Davies-Meyer compression when t != 2*d), the underlying call
    raises and the vector is silently skipped.
    """
    fe = [h.to_field(x) for x in inp_ints]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = produce(fe)
    except (ValueError, NotImplementedError):
        return
    hexify = lambda xs: [hex(int(h.from_field(x))) for x in xs]
    vectors.append({
        "instance": label,
        "mode": mode,
        "input": hexify(fe),
        "output": hexify(out),
    })
    modes_seen.add(mode)


def build_vectors(key):
    mod_name, cls_name, inst_name, prefix = CONSTRUCTIONS[key]
    Hash = getattr(importlib.import_module(mod_name), cls_name)
    instances_mod = importlib.import_module(inst_name)

    # Select the instances of this construction living over an allowed prime.
    selected = []
    for name in dir(instances_mod):
        if not name.startswith(prefix):
            continue
        params = getattr(instances_mod, name)
        if getattr(params, "p", None) in ALLOWED_PRIMES:
            selected.append((name, params))
    selected.sort()

    vectors = []
    modes_seen = set()
    for name, params in selected:
        h = Hash(params)
        label = name.lower().replace("_", "-")  # e.g. GMIMC_BN254_T3 -> gmimc-bn254-t3

        # Permutation: full t-element state.
        perm_in = list(range(1, h.t + 1))
        emit(vectors, modes_seen, label, "permutation", h, perm_in, h.permutation)

        # Compression: 2-to-1 over two equal lanes. Anemoi's Jive works on
        # l-element lanes; Davies-Meyer works on d-element halves. The full
        # input is stored concatenated (first half = x1, second half = x2).
        lane = getattr(h, "l", None) or getattr(h, "d", None)
        if lane and hasattr(h, "compress_2_to_1"):
            comp_in = list(range(1, 2 * lane + 1))
            emit(vectors, modes_seen, label, "compression", h, comp_in,
                 lambda fe, n=lane: h.compress_2_to_1(fe[:n], fe[n:]))

        # Sponge: a single-block message and a longer, padding-exercising one.
        if hasattr(h, "hash_sponge"):
            for data in (list(range(1, h.r + 1)), list(range(1, 2 * h.r + 2))):
                emit(vectors, modes_seen, label, "sponge", h, data, h.hash_sponge)

    labels = sorted({v["instance"] for v in vectors})
    comment = (
        f"{key} known-answer test vectors. "
        f"modes present: {', '.join(sorted(modes_seen)) or 'none'}. "
        f"instances present: {', '.join(labels) or 'none'}."
    )
    return {"_comment": comment, "vectors": vectors}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "construction",
        help="one of: " + ", ".join(CONSTRUCTIONS)
             + " (polocolo is not implemented in this repo yet)",
    )
    args = ap.parse_args()

    key = args.construction.strip().lower().replace(" ", "-")
    key = ALIASES.get(key, key)
    if key not in CONSTRUCTIONS:
        ap.error(
            f"unknown/unavailable construction {args.construction!r}. "
            f"available: {', '.join(CONSTRUCTIONS)}"
        )

    print(json.dumps(build_vectors(key), indent=2))


if __name__ == "__main__":
    main()

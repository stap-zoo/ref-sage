# primitive.py
# ---------------------------------------------------------------------------
# The object model shared by every primitive, split by COMPOSITION:
#
#   * Permutation        -- the bare round function P: F_p^t -> F_p^t. Exposes permute /
#                           permute_inv and nothing else. Instantiable on its own, no mode of
#                           operation. Each concrete primitive subclasses it (e.g. AnemoiPerm).
#
#   * HashFunction       -- a permutation PLUS a sponge. Exposes hash (NOT permute).
#   * CompressionFunction-- a permutation PLUS a compression. Exposes compress (NOT permute).
#
# A hash and a compression are DIFFERENT functions on the same permutation, so they are
# separate classes, loaded independently, each wrapping a Permutation. A mode function is a
# hash/compression, not a permutation: it does not expose permute -- the wrapped permutation
# is reachable as `.permutation` if genuinely needed. Their initializer takes a Permutation
# instance and that mode's parameter dict -- nothing else: kappa, p, t, to_field come off the
# permutation. The class pins the default mode KIND (SPONGE_KIND / COMP_KIND), which an
# instance can override with a "kind" key in its mode dict; several sponge variants over one
# permutation are just several HashFunction subclasses:
#     from anemoi.hash import AnemoiPerm, AnemoiHash, AnemoiCompress
#     P = AnemoiPerm(params)
#     H = AnemoiHash(P, params.sponge); C = AnemoiCompress(P, params.comp)
# ---------------------------------------------------------------------------

from utils.mode import make_sponge, make_compression


class Permutation:
    """Base class for a concrete permutation P: F_p^t -> F_p^t built from a *Params object.

    Copies the instance-global fields (field + conversions, state size, prime, security level)
    out of params; subclasses call super().__init__(params) and then copy their own matrices /
    constants / round schedule, and implement permute / permute_inv. A pure consumer of params.
    `p` and `kappa` are here so the mode functions can build their mode from the permutation
    alone; `toy` records THIS permutation's toy status (a mode carries its own toy flag in its
    parameter dict, independent of the permutation's).
    """

    def __init__(self, params):
        self.F = params.F
        self.to_field = params.to_field
        self.from_field = params.from_field
        self.t = params.t
        self.p = params.p
        self.kappa = params.kappa
        self.toy = params.toy

    def permute(self, state):
        """Apply the permutation to a length-t state. Implemented by the subclass."""
        raise NotImplementedError("Permutation.permute must be defined by the concrete primitive")

    def permute_inv(self, state):
        """Invert the permutation. Implemented by the subclass."""
        raise NotImplementedError("Permutation.permute_inv must be defined by the concrete primitive")


class HashFunction:
    """A permutation plus a sponge. Built from a Permutation and the sponge parameter dict
    (dict(r, c, d), optionally toy=True); the class pins the default SPONGE_KIND, which an
    instance may override with a "kind" key in the dict. Exposes hash.

    This is a hash function, NOT a permutation: it does not expose permute. The wrapped
    permutation is available as `.permutation` if genuinely needed.

    The mode's toy flag is its OWN dict field, separate from the permutation's toy: a mode may
    be a toy version (below the security floors) independently of the permutation. Absent =>
    not a toy mode (floor violations raise)."""

    SPONGE_KIND = None   # default sponge kind; a "kind" key in the sponge dict overrides it

    def __init__(self, permutation, sponge):
        if sponge is None:
            raise ValueError(f"{type(self).__name__}: no sponge params given (sponge is None)")
        self.permutation = permutation
        P = permutation
        sponge = dict(sponge)                       # copy: we may pop "kind" (and don't mutate the caller's dict)
        kind = sponge.pop("kind", self.SPONGE_KIND)  # instance may override the class default
        # toy (if present) travels inside `sponge` and is consumed by make_sponge's toy arg.
        self.sponge = make_sponge(kind, P.kappa, P.p, t=P.t, to_field=P.to_field, **sponge)

    def hash(self, data, **kwargs):
        """Sponge hash of variable-length input to the sponge digest."""
        return self.sponge.hash(self.permutation.permute, data, **kwargs)


class CompressionFunction:
    """A permutation plus a compression. Built from a Permutation and the compression parameter
    dict (digest d and/or arity a, optional matrix M, optionally toy=True); the class pins the
    default COMP_KIND, which an instance may override with a "kind" key in the dict. Exposes
    compress.

    This is a compression function, NOT a permutation: it does not expose permute. The wrapped
    permutation is available as `.permutation` if genuinely needed.

    As for HashFunction, the mode's toy flag is its own dict field, separate from the
    permutation's toy; absent => not a toy mode (floor violations raise)."""

    COMP_KIND = None   # default compression kind; a "kind" key in the comp dict overrides it

    def __init__(self, permutation, comp):
        if comp is None:
            raise ValueError(f"{type(self).__name__}: no compression params given (comp is None)")
        self.permutation = permutation
        P = permutation
        comp = dict(comp)                          # copy: we may pop "kind" (and don't mutate the caller's dict)
        kind = comp.pop("kind", self.COMP_KIND)    # instance may override the class default
        # toy (if present) travels inside `comp` and is consumed by make_compression's toy arg.
        self.comp = make_compression(kind, P.kappa, P.p, t=P.t, to_field=P.to_field, **comp)

    def compress(self, data):
        """Compression M*(P(x)+x) on the full a*d = t state -> d elements."""
        return self.comp.compress(self.permutation.permute, data)

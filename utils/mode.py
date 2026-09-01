"""Modes of operation built on top of a permutation.

Turns a fixed-width permutation into a hash/compression function: sponge constructions, 
compression functions, padding rules, and rate/capacity/digest derivation.
"""

# Structural imports
from recommendations import recommend

# Math specific imports
from math import ceil, log2

# Custom imports
from utils.matrix import add_at, replace_at, matvecmul, vecadd

# ===========================================================================
# Small helpers, like universally used padding rules
# ===========================================================================

def bitsof(p: int) -> int:
    """Per-element bit budget, ceil(log2 p) -- the honest bits/element (real log2 p can
    leave an exact count a sub-bit short and force a spurious extra element)."""
    return ceil(log2(p))

def id_(x):
    return x

def pad_simple(data, rate, to_field=lambda x: x):
    """10*: append a single 1, then zeros to the next multiple of rate. Always applied.
    Injective on its own -- the safe default for variable-length input."""
    num_zeros = (rate - (len(data) + 1) % rate) % rate
    return data + [to_field(1)] + [to_field(0)] * num_zeros

def pad_zero(data, rate, to_field=lambda x: x):
    """0*: zero-pad to the next multiple of rate; no-op if already aligned. NOT injective
    on its own -- only safe when the input length is bound elsewhere (e.g. in the IV)."""
    num_zeros = (rate - len(data) % rate) % rate
    return data + [to_field(0)] * num_zeros

# ===========================================================================
# Sponge modes
# ===========================================================================

class Sponge:
    """Base sponge. Resolves + validates (t, r, c, d) at construction; owns the
    absorb -> permute -> squeeze skeleton. The permutation is supplied per hash call.

    Three hooks distinguish variants:
      pad(data, input_len_fixed)                      -> padded list    (padding rule)
      make_iv(output_len, input_len, input_len_fixed) -> list[c]        (IV)
      before_final_absorb(state)                      -> state          (domain sep for non-injective paddings)

    Domain separation lives in TWO places, both writing the capacity where a later
    permutation carries it into the squeezed rate:
      * init time (make_iv): output length, instance tag, fixed input length
      * end time  (before_final_absorb): the message-boundary / mode separator
    """

    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------
    
    def __init__(self, kappa, p, *, t=None, r=None, c=None, d=None, absorb=add_at, to_field=lambda x: x, capacity_first=False, toy=False):
        """Construct a sponge over F_p at security level kappa.
 
        The permutation is NOT passed here -- it is supplied per hash call, so a Sponge can
        be built when only kappa, p and the resolved sizes are known, and bound to a concrete
        permutation later over F_p^t later.
 
        Sizes are resolved from any sufficient subset of (t, r, c, d) via
        resolve_params; omitted values come from the security floors. With none of
        t/r given the default is a 2-to-1 tree hash (r = 2c, t = 3c).

        The default state layout is RATE-FIRST: state = [rate (r elements) | capacity (c elements)].
        Absorb adds into state[:r]; the IV occupies the capacity state[r:]; squeeze reads
        state[:r]; the last capacity element (for separators) is state[-1].
 
        Parameters
        ----------
        kappa      : target security level in bits (drives floors; kept for the output check).
        p          : prime of the field F_p; bits/element = ceil(log2 p).
        t, r, c, d : state / rate / capacity / digest size; any may be derived.
        absorb     : (state, block, off) -> state. add_at = additive | replace_at = overwrite.
        to_field   : int -> field element (default identity); used for IVs, separators, padding.
        toy        : if True, security-floor violations warn instead of raising.
        """
        self.kappa_target, self.p, self.to_field, self.toy = kappa, p, to_field, toy
        self.resolve_params(kappa, p, t, r, c, d, toy)

        self.rate_off = self.c if capacity_first else 0
        self.cap_off = 0 if capacity_first else self.r
        self._absorb_fn = lambda state, block: absorb(state=state, block=block, off=self.rate_off)
        
    # ---------------------------------------------------------------------------
    # Security bound helpers and parameter resolution
    # ---------------------------------------------------------------------------

    def get_min_capacity(self, kappa: int, pbits: int) -> int:
        """Capacity elements for a kappa-bit target: c * pbits >= 2 * kappa (birthday).
        round, not ceil -> snap to the nearest achievable element count (might arrive slightly below kappa)"""
        return max(1, round(2 * kappa / pbits))

    def get_min_digest(self, kappa: int, pbits: int) -> int:
        """Smallest collision-resistant output: d * pbits >= 2 * kappa. 
        round, not ceil -> snap to the nearest achievable element count (might arrive slightly below kappa)"""
        return max(1, round(2 * kappa / pbits))

    def resolve_params(self, kappa, p, t=None, r=None, c=None, d=None, toy=False):
        """Resolve (t, r, c, d) from any sufficient subset under t = r + c; omitted values
        come from the floors. Default (no t/r): 2-to-1 tree hash, r = 2c, t = 3c."""

        pbits = bitsof(p)
        c_min = self.get_min_capacity(kappa, pbits)
        d_min = self.get_min_digest(kappa, pbits)

        c = c_min if c is None else c
        d = d_min if d is None else d

        if t is not None and r is not None:
            if r + c != t:
                raise ValueError(f"sponge invariant violated: r + c = {r + c} != t = {t}")
        elif t is not None:
            r = t - c
        elif r is not None:
            t = r + c
        else:
            r, t = 2 * c, 3 * c  # default: sponge-based 2-to-1 tree hash

        if min(r, c, d) < 1:
            raise ValueError(f"need r, c, d >= 1; got r={r}, c={c}, d={d}")
        if c < c_min:
            recommend(f"capacity c={c} ({c * pbits} bits) below {kappa}-bit floor c_min={c_min}", toy)
        if d < d_min:
            recommend(f"digest d={d} ({d * pbits} bits) below {kappa}-bit floor d_min={d_min}", toy)
        if r < c_min:
            # rate below capacity does not break the capacity bound, but min(c,d,r) then caps
            # effective security at the rate; the floor to preserve target is r >= c_min, not r >= c.
            recommend(f"rate r={r} below {kappa}-bit floor {c_min}: rate becomes the binding count", toy)
        
        self.t, self.r, self.c, self.d, self.c_min, self.d_min = t, r, c, d, c_min, d_min
        self.kappa_achieved = min(c, d, r) * pbits // 2  # collision resistance actually delivered (assuming ideal permutation)

    # ---------------------------------------------------------------------------
    # Hooks (to be overridden by derived classes)
    # ---------------------------------------------------------------------------
    """
    Can hash() accept arbitrary-length (unknown-ahead-of-time) input without encoding collisions? 

    True iff the map
        variable-length input -> (block sequence, IV, end separators)
    is injective, so two different-length messages can never yield the same digest by an
    encoding accident.
    
    A subclass sets True when it supplies SOME injectivity source:
      * injective padding applied always (pad10*)                         -- SpongePlain
      * input length bound into the IV                                    -- Poseidon-style
      * an aligned/padded separator repairing conditional padding
        (a 1-bit IV flag, or an alignment-dependent mu)                   -- RPO / SpongePI
    
    Leave False when the padding is non-injective and nothing recovers the lost message
    boundary: the variant is then only safe for fixed-length input (caller passes
    input_len_fixed=True, which pins the length externally and makes the ambiguity moot).
    """
    variable_input_safe = None

    """
    Can ONE instance be squeezed to DIFFERENT output lengths without prefix confusion? 
    
    A plain sponge squeezed for d and for d' (d < d') gives outputs where the shorter is a prefix of the 
    longer, so a value produced under one declared length can be reused under another. 

    True iff the output length is encoded (e.g. in a capacity slot of the IV), making each length 
    a distinct function so no prefix relationship exists -- SpongePI / SAFE. 

    If False, this hash function has the XOF property. Fine for a single fixed d per instance, 
    else consider using an output-length-encoding variant.
    """
    variable_output_safe = False  # zero-IV, output length not encoded

    def pad(self, data: list, input_len_fixed: bool, rate_aligned: bool) -> list:
        """Pad `data` to a multiple of the rate; return the padded list and a boolean indicating 
        whether the data was already rate-aligned (i.e., len(data) is a multiple or the rate).

        The padding is the primary source of INPUT injectivity. Common choices:
        * pad_simple (M||10*; append 1, then zeros): injective on its own, so it needs
            no IV/separator help. Costs one extra block when the input is already rate-aligned.
            Might be conditionally applied, i.e., only if input_len_fixed=False.
        * pad_zero (M||0*;  fill with zeros): NOT injective alone; only sound when
            the input length is bound elsewhere (make_iv with input_len, or input_len_fixed).

        `input_len_fixed` is True when the caller guarantees a fixed input length, `rate_aligned` 
        indicates whether the original (unpadded) data was rate-aligned.
        """
        raise NotImplementedError("Abstract method pad(): must be defined by derived class.")

    def make_iv(self, output_len: int, input_len: int, input_len_fixed: bool, rate_aligned: bool, c_val):
        """Build the capacity initial value; return a list of exactly self.c field elements.
        Defaults to all-zero.

        The IV is the init-time domain-separation slot -> whatever is known AFTER padding but 
        BEFORE absorption and needs to distinguish this computation from another. 
        
        Common choices:
        * all zeros: nothing to separate (safe when padding is injective, e.g. SpongePlain).
        * input length in the capacity: lets non-injective padding be used at fixed length.
        * output length in the capacity: separates different digest lengths of one instance.
        * an instance/use-case tag: separates distinct protocols sharing one permutation.

        `output_len` is the requested digest length, `input_len` the unpadded input length,
        `input_len_fixed` whether the input length is fixed/known. `rate_aligned` indicates 
        whether the original (unpadded) data was rate-aligned. `c_val` is a constant value
        the IV defaults to.
        """
        return [self.to_field(c_val)] * self.c

    def before_final_absorb(self, state: list, rate_aligned: bool):
        """End-time domain separation: adjust `state` BEFORE the last permutation call in the 
        absorption phase, so the change diffuses into the squeezed rate. Return the state.

        Typically used by variants whose padding is not injective on its own, to inject the message-
        boundary / mode separator that the padding omitted. Default: no-op.
        """
        return state

    # ---------------------------------------------------------------------------
    # Sponge skeleton
    # ---------------------------------------------------------------------------

    def _run(self, perm, data, output_len, input_len_fixed, c_val):
        # Indicated whether the original (unpadded) message is rate-aligned. 
        # Might be used to apply a conditional padding rule or for domain separation.
        rate_aligned = (len(data) % self.r == 0)

        padded_data = self.pad(data, input_len_fixed, rate_aligned)
        if len(padded_data) % self.r != 0:
            raise ValueError("padding did not align data to the rate")
        
        iv = self.make_iv(output_len, len(data), input_len_fixed, rate_aligned, c_val)
        if len(iv) != self.c:
            raise ValueError(f"IV has {len(iv)} elements, expected c={self.c}")

        # (Padded) message blocks
        blocks = [padded_data[i:i + self.r] for i in range(0, len(padded_data), self.r)]

        # Initialize state (0-init with IV in capacity part)
        state = [self.to_field(0)] * self.t
        state = replace_at(state, iv, self.cap_off)

        # Absorption phase
        last = len(blocks) - 1
        for k, block in enumerate(blocks):
            state = self._absorb_fn(state, block)
            if k == last:
                state = self.before_final_absorb(state, rate_aligned) # inject BEFORE the last permute
            state = perm(state)
        
        # Squeezing phase
        out = []
        while len(out) < output_len:
            out.extend(state[self.rate_off:self.rate_off + self.r])
            if len(out) < output_len:
                state = perm(state)

        return out[:output_len]

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------

    def hash(self, perm, data, output_len=None, input_len_fixed=False, c_val=0):
        """Optional output_len defaults to the instance digest size d;
        # if given it must meet the kappa-bit collision floor (get_min_digest)."""
        o = self.d if output_len is None else output_len
        if not input_len_fixed and not self.variable_input_safe:
            recommend(f"{type(self).__name__}: variable-length input not injectivity-safe with this variant", self.toy)
        if o < self.d_min:
            recommend(f"output_len={o} below ~{self.kappa_target}-bit collision floor {self.d_min}", self.toy)
        if o != self.d and not self.variable_output_safe:
            recommend(f"{type(self).__name__}: output length not encoded; squeezing {o} != "
                    f"instance d={self.d} risks prefix confusion across lengths", self.toy)
                    
        return self._run(perm, data, o, input_len_fixed, c_val)
    
    def compress(self, perm, data: list) -> list:
        """Sponge-based compression  a*d -> d: absorb the fixed-length input `data`.
        The arity a = len(data)/d is computed here and must be a plain integer; raises otherwise."""
        if len(data) % self.d != 0:
            raise ValueError(f"sponge compression: input length {len(data)} is not a multiple "
                             f"of the digest size d={self.d} (no integer arity)")
        a = len(data) // self.d
        if a < 2:
            raise ValueError(f"sponge compression needs arity a >= 2 (a*d input), got a={a}")
        return self.hash(perm, data, input_len_fixed=True)   # output_len defaults to self.d



class SpongePlain(Sponge):
    """Bertoni et al. sponge."""

    variable_input_safe  = True    # injective pad_simple (no need for domain separation)
    variable_output_safe = False   # output length not encoded

    def pad(self, data, input_len_fixed, rate_aligned):
        # pad10* -- injective regardless of fixed/variable input length. 
        return pad_simple(data=data, rate=self.r, to_field=self.to_field)

class SpongeLE(Sponge):
    """Sponge with lenth encoding (LE): pad_zero (fixed-length input) / pad_simple (variable-length input) + 
    input length in IV.
    """

    variable_input_safe  = True    # length in the IV disambiguates lengths
    variable_output_safe = False   # output length not encoded

    def pad(self, data, input_len_fixed, rate_aligned):
        if input_len_fixed:
            return pad_zero(data, self.r, self.to_field)
        return pad_simple(data, self.r, self.to_field)

    def make_iv(self, output_len, input_len, input_len_fixed, rate_aligned, c_val):
        return [self.to_field(input_len)] + [self.to_field(c_val)] * (self.c - 1)

class SpongeCLE(Sponge):
    """Sponge with conditional length encoding (CLE): zero-pad, and encode the input length in the first
    capacity element ONLY when the input is not rate-aligned (aligned -> zero IV).
    Used by Arion and ReinforcedConcrete.

    Injective: unaligned inputs carry the length tag; distinct aligned lengths are separated
    by block count (an extra zero block forces an extra permutation)."""

    variable_input_safe  = True    # injective
    variable_output_safe = False   # output length not encoded

    def pad(self, data, input_len_fixed, rate_aligned):
        return pad_zero(data, self.r, self.to_field)

    def make_iv(self, output_len, input_len, input_len_fixed, rate_aligned, c_val):
        if rate_aligned:
            return [self.to_field(c_val)] * self.c
        return [self.to_field(input_len)] + [self.to_field(c_val)] * (self.c - 1)

class Sponge2(Sponge):
    """Specific instance of GSponge (https://eprint.iacr.org/2024/911.pdf), here without adding M0
    with |M0| = r0 < c into the initial capacity part.
    """

    variable_input_safe  = True    # alignment separator repairs the zero padding
    variable_output_safe = False   # output length not encoded

    def __init__(self, *a, absorb=replace_at, **k):
        super().__init__(*a, absorb=absorb, **k) 

    def pad(self, data, input_len_fixed, rate_aligned):
        return pad_zero(data, self.r, self.to_field)
        
    def make_iv(self, output_len, input_len, input_len_fixed, rate_aligned, c_val):
        """Domain separator (rate-aligned/full = 0, padded = >0) encoded in IV
        See https://eprint.iacr.org/2023/1045.pdf, Sec 4.6."""
        domain = input_len % self.r
        return [self.to_field(domain)] + [self.to_field(c_val)] * (self.c - 1)

class SpongeRescue(Sponge):
    """Rescue sponge: Fixed-length input uses no padding. 

    NOTE: Fixed-length input relies on the input length being a fixed constant of the instance, 
    so different fixed lengths must NOT share one instance (nothing in the zero IV disambiguates them)."""

    variable_input_safe  = True    # injective pad_simple (variable-length input path)
    variable_output_safe = False   # output length not encoded

    def pad(self, data, input_len_fixed, rate_aligned):
        if input_len_fixed:
            if len(data) % self.r != 0:
                raise ValueError(f"fixed-length input must be rate-aligned: len={len(data)}, r={self.r}")
            return data  # no padding; injectivity from the fixed length
        return pad_simple(data, self.r, self.to_field)

class SpongeRPO(Sponge):
    """RPO sponge: conditional padding + alignment seperator encoded in IV. Fixes problems of SpongeRescue.
    Note that RPO uses overwrite and capacity-first layout.
    """

    variable_input_safe  = True    # alignment separator repairs the conditional padding
    variable_output_safe = False   # output length not encoded

    def __init__(self, *a, absorb=replace_at, capacity_first=True, **k):
        super().__init__(*a, absorb=absorb, capacity_first=capacity_first, **k) 

    def pad(self, data, input_len_fixed, rate_aligned):
        """pad_simple (10*) only when unaligned; no padding when the final block is already full 
        (avoids the extra block/permutation)."""
        return data if rate_aligned else pad_simple(data, self.r, self.to_field)

    def make_iv(self, output_len, input_len, input_len_fixed, rate_aligned, c_val):
        """Domain separator (rate-aligned/full = 0, padded = 1) encoded in IV"""
        domain = 0 if rate_aligned else 1
        return [self.to_field(domain)] + [self.to_field(c_val)] * (self.c - 1)

class SpongeHirose(Sponge):
    """Hirose-mode sponge: conditional padding + alignment separator absorbed before last permutation call. 
    See https://www.mdpi.com/2410-387X/2/2/11."""

    variable_input_safe  = True    # alignment separator repairs the conditional padding
    variable_output_safe = False   # output length not encoded

    def pad(self, data, input_len_fixed, rate_aligned):
        """pad_simple (10*) only when unaligned; no padding when the final block is already full 
        (avoids the extra block/permutation)."""
        return data if rate_aligned else pad_simple(data, self.r, self.to_field)

    def before_final_absorb(self, state, rate_aligned):
        """Domain separator (rate-aligned/full = 0, padded = 1) added to last capacity element"""
        domain = 0 if rate_aligned else 1
        state = add_at(state=state, block=[self.to_field(domain)], off=self.cap_off + self.c - 1)
        return state

class SpongePI(SpongeHirose):
    """Sponge-pi: Hirose with output length encoded in capacity.
    See https://tosc.iacr.org/index.php/ToSC/article/view/12073/11914."""

    variable_input_safe  = True    # alignment separator repairs the conditional padding
    variable_output_safe = True    # output length encoded in the IV

    def make_iv(self, output_len, input_len, input_len_fixed, rate_aligned, c_val):
        return [self.to_field(c_val)] * (self.c - 1) + [self.to_field(output_len)]

# TODO implement
class SpongeSAFE(Sponge):
    """Sponge API for Field Elements (SAFE). See https://eprint.iacr.org/2023/522"""
    
    variable_input_safe  = True
    variable_output_safe = False

    def pad(self, data, input_len_fixed, rate_aligned):
        return pad_zero(data, self.r, self.to_field)

# ===========================================================================
# Compression modes
# ===========================================================================


# ---------------------------------------------------------------------------
# Compression modes (class hierarchy, mirroring Sponge)
# ---------------------------------------------------------------------------

class Compression:
    """Feed-forward compression  x in F_p^t  |->  M*(P(x) + x)  in F_p^d,  d < t.

    The unifying frame from compression.md: given a permutation P and a right-invertible
    matrix M in F_p^{d x t}, the compression function is M*(P(x) + x). Truncation (the
    'compression mode', M = I_{d x t}, keeping the first d elements) is the default; Jive_b
    is the same construction with a different M (see CompressionJive).

    Like Sponge, this resolves + validates its sizes (t, d, a) at construction and takes the
    permutation per compress() call, so it can be built as soon as t is known and bound to a
    concrete permutation over F_p^t later. The state is viewed as a = t/d blocks of size d
    (the arity) when d divides t.
    """

    def __init__(self, kappa, p, *, t, d=None, a=None, M=None, to_field=lambda x: x, toy=False):
        """Construct a compression over F_p at security level kappa.

        Parameters
        ----------
        kappa    : target security level in bits (drives the collision/guessing floors).
        p        : prime of the field F_p; bits/element = ceil(log2 p).
        t        : state size (= permutation width). Required.
        d, a     : digest size and arity under t = a*d; either may be derived from the other,
                   or both omitted to take the collision floor for d.
        M        : d x t matrix over F_p (right-invertible). Default: the mode's _default_M
                   (identity truncation here).
        to_field : int -> field element (default identity); used to build the default matrix.
        toy      : if True, security-floor violations warn instead of raising.
        """
        self.kappa_target, self.p, self.to_field, self.toy = kappa, p, to_field, toy
        self.resolve_params(kappa, p, t, d, a, toy)
        self.M = self._default_M() if M is None else M
        self._check_matrix(toy)

    # ---------------------------------------------------------------------------
    # Security bound helpers and parameter resolution
    # ---------------------------------------------------------------------------

    def get_min_digest(self, kappa: int, pbits: int) -> int:
        """Smallest collision-resistant digest: d * pbits >= 2 * kappa, i.e. the birthday
        bound on the output clears the target. round, not ceil -> snap to the nearest
        achievable element count (might arrive slightly below kappa). Mirrors Sponge."""
        return max(1, round(2 * kappa / pbits))

    def get_min_trunc(self, kappa: int, pbits: int) -> int:
        """Minimum number of state elements a truncation compression must DISCARD.

        trunc_d(P(x) + x) keeps d of the t state elements and drops t - d; the dropped tail
        is what makes the map non-invertible, but an attacker can guess it for p^(t-d) =
        2^((t-d)*pbits) work. To keep inversion/preimage above kappa the tail must satisfy
        t - d >= ceil(kappa / pbits)."""
        return max(1, ceil(kappa / pbits))

    def resolve_params(self, kappa, p, t, d, a, toy):
        """Resolve (t, d, a) under t = a*d from any sufficient subset; then apply the
        collision floor on d and the truncation-guessing floor on t - d."""
        pbits = bitsof(p)
        d_min = self.get_min_digest(kappa, pbits)
        trunc_min = self.get_min_trunc(kappa, pbits)

        if a is not None and d is not None:
            if a * d != t:
                raise ValueError(f"compression invariant violated: a*d = {a*d} != t = {t}")
        elif a is not None:
            if t % a != 0:
                raise ValueError(f"arity a={a} must divide t={t}")
            d = t // a
        elif d is not None:
            pass
        else:
            d = d_min

        # arity is only well-defined as an integer when d divides t (Jive needs this)
        a = t // d if t % d == 0 else None

        # --- the defining constraint: a compression must shrink (hard) ---
        if not (1 <= d < t):
            raise ValueError(f"digest d={d} must satisfy 1 <= d < t={t} (must shrink)")

        # --- security floors (recommend(): raise unless toy) ---
        if d < d_min:
            recommend(f"digest d={d} ({d * pbits} bits) below {kappa}-bit collision floor {d_min}", toy)
        if t - d < trunc_min:
            recommend(f"truncated tail t-d={t-d} below {kappa}-bit guessing floor {trunc_min}", toy)

        self.t, self.d, self.a, self.d_min = t, d, a, d_min
        self.kappa_achieved = d * pbits // 2  # collision resistance up to p^{d/2}

    def _default_M(self):
        """Truncation matrix I_{d x t}: the first d rows of the t x t identity (keep the
        first d elements of P(x) + x). Overridden by named modes (e.g. Jive)."""
        return [[self.to_field(1 if j == i else 0) for j in range(self.t)] for i in range(self.d)]

    def _check_matrix(self, toy):
        """M must be d x t (hard) and right-invertible, i.e. full row rank d (recommend()).
        The rank is checked over the field of M's entries when they live in one (sage
        field elements); for plain-int toy matrices the check is skipped."""
        if len(self.M) != self.d or any(len(row) != self.t for row in self.M):
            got = f"{len(self.M)}x{len(self.M[0]) if self.M else 0}"
            raise ValueError(f"M must be a {self.d}x{self.t} matrix; got {got}")
        try:
            from sage.all import Matrix
            rank = Matrix(self.M).rank()
        except Exception:
            rank = None  # entries not in a field sage can build a matrix over; skip
        if rank is not None and rank != self.d:
            recommend(f"M is not right-invertible: rank {rank} != d={self.d}", toy)

    # ---------------------------------------------------------------------------
    # Compression
    # ---------------------------------------------------------------------------

    def compress(self, perm, state):
        """M*(P(x) + x): apply the permutation, feed-forward, then the d x t matrix."""
        if len(state) != self.t:
            raise ValueError(f"compress expects t={self.t} elements, got {len(state)}")
        return matvecmul(self.M, vecadd(perm(state), state))


class CompressionJive(Compression):
    """Anemoi's Jive_b compression (https://eprint.iacr.org/2022/840.pdf, Sec. 3.2):
    the same M*(P(x)+x) frame with M = [I_d | I_d | ... | I_d] (a = t/d horizontal blocks).
    Block i of the output is the field-sum of block i across P(x) + x -- i.e.
        Jive_a(x_1, ..., x_a) = sum_j (x_j + P(x_1 || ... || x_a)_j).
    Requires d | t (equal-size blocks)."""

    def _default_M(self):
        if self.t % self.d != 0:
            raise ValueError(f"Jive needs d | t (equal blocks): t={self.t}, d={self.d}")
        return [[self.to_field(1 if (j % self.d) == i else 0) for j in range(self.t)] for i in range(self.d)]


# Sponge-based compression is NOT a Compression class: it is the Sponge.compress method (an
# a*d -> d node using the sponge's own capacity/squeeze for one-wayness). A primitive that wants
# it calls H.sponge.compress(perm, state) directly and defines no Compress class.


# ---------------------------------------------------------------------------
# Factory initializers: build a Sponge / Compression from a (kind, info) spec
# ---------------------------------------------------------------------------

_SPONGE_KINDS = {
    "plain": SpongePlain, "le": SpongeLE, "cle": SpongeCLE, "sponge2": Sponge2,
    "rescue": SpongeRescue, "rpo": SpongeRPO, "hirose": SpongeHirose, "pi": SpongePI,
    "safe": SpongeSAFE,
}

def make_sponge(kind, kappa, p, *, t, to_field=lambda x: x, toy=False, **info):
    """Build a Sponge subclass by name. `info` carries the sponge sizes (r, c, d, ...)."""
    if kind not in _SPONGE_KINDS:
        raise ValueError(f"unknown sponge kind {kind}; known: {sorted(_SPONGE_KINDS)}")
    return _SPONGE_KINDS[kind](kappa, p, t=t, to_field=to_field, toy=toy, **info)

def make_compression(kind, kappa, p, *, t, to_field=lambda x: x, toy=False, **info):
    """Build a feed-forward Compression M*(P(x)+x) by name. `info` carries d/a/M. (Sponge-based
    compression is not built here -- it is the Sponge.compress method.)"""
    if kind in ("trunc", "mode"):
        return Compression(kappa, p, t=t, to_field=to_field, toy=toy, **info)
    if kind == "jive":
        return CompressionJive(kappa, p, t=t, to_field=to_field, toy=toy, **info)
    raise ValueError(f"unknown compression kind {kind}")
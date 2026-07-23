"""Modes of operation built on top of a permutation.

Turns a fixed-width permutation into a hash/compression function: sponge constructions, 
compression functions, padding rules, and rate/capacity/digest derivation.
"""

# Structural imports
from recommendations import recommend

# Math specific imports
from math import ceil, log2

# Custom imports
from utils.matrix import add_at, replace_at

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
    
    def tree_hash(self, perm, children: list[list], arity: int = None) -> list:
        """a-to-1 Merkle node: absorb `arity` d-element children, squeeze one d-element parent."""
        a = len(children) if arity is None else arity
        if len(children) != a:
            raise ValueError(f"expected {a} children, got {len(children)}")
        if any(len(ch) != self.d for ch in children):
            raise ValueError(f"each child must be {self.d} elements; got {[len(ch) for ch in children]}")

        data = [x for ch in children for x in ch] # a*d elements
        if len(data) % self.r != 0:
            raise ValueError(f"arity*d = {len(data)} must be a multiple of rate r={self.r} for a padding-free tree node")
        return self.hash(perm, data, input_len_fixed=True)



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
# Mode parameters: derivation and security checks
# ---------------------------------------------------------------------------
def get_min_digest(kappa: int, p: int) -> int:
    """Smallest collision-resistant output: d*ceil(log2 p)/2 >= kappa, i.e. the birthday
    bound on the output itself must clear the target. Same 2*kappa budget and rounding as
    get_min_capacity."""
    return max(1, round(2 * kappa / bitsof(p)))

def get_min_trunc(kappa: int, p: int) -> int:
    """Minimum number of state elements a truncation compression must DISCARD.
    
    In a Davies-Meyer / truncation compression, trunc_d(P(x) + x) keeps d of the t state
    elements and drops the other t - d. Those dropped elements are what make the map
    non-invertible -- but an attacker can simply guess them. Guessing the truncated part
    costs p^(t-d) = 2^((t-d)*log2 p) work, so to keep inversion/preimage attacks above a
    kappa-bit target the discarded tail must satisfy t - d >= ceil(kappa / bitsof(p)).
    """
    return max(1, ceil(kappa / bitsof(p)))

def resolve_compression_params(kappa: int, p: int, t: int, mode: str | None, d: int = None, a: int = None) -> tuple[str, int, int] | None:
    """Resolve/validate the compression parameters for a state of size t.

    Returns (mode,d,a) with d the digest length and a = t/d the arity, or None if the
    primitive defines no compression (mode is None). Raises if a compression IS requested
    but can't be defined.

    Independent of the sponge digest on purpose: a sponge may squeeze d >= t (XOF), but a
    compression is only defined for d < t. Resolution order:
      * arity given  -> d = t // arity
      * digest given -> d = digest, b = t // d
      * neither      -> d = get_min_digest(kappa, p) (the security floor, = 2-to-1 when t=2c)
    """
    if mode is None:
        return None # primitive has no compression

    d_min = get_min_digest(kappa, p)
    if arity is not None:
        if t % arity != 0:
            raise ValueError(f"arity {arity} must divide t={t}")
        d = t // arity
    else:
        d = d_min if digest is None else digest

    # --- the defining constraint: a compression must shrink ---
    if d >= t:
        raise ValueError(f"no compression defined: digest d={d} >= state t={t} "
                         f"(nothing is compressed; use the sponge for d >= t)")
    if d < 1:
        raise ValueError(f"digest d={d} must be >= 1")

    # --- security floors (normally implied by the sponge's r >= c; kept as a guard) ---
    if d < d_min:
        raise ValueError(f"digest d={d} below {kappa}-bit collision floor {d_min}")

    b = t // d
    if mode == "jive":
        if t % d != 0:
            raise ValueError(f"Jive_b needs d | t: t={t}, d={d} (capacity/digest diverge, "
                             f"e.g. Tip5 t=16,d=5 -- use a sponge-style compression instead)")
    elif mode == "trunc":
        trunc_min = get_min_trunc(kappa, p)
        if t - d < trunc_min:
            raise ValueError(f"truncated tail t-d={t-d} below {kappa}-bit guessing floor {trunc_min}")
    else:
        raise ValueError(f"unknown compression mode {mode}")

    return mode, d, b

# ---------------------------------------------------------------------------
# Compression modes
# ---------------------------------------------------------------------------

def compress(perm, data: list, kappa: int, p: int, mode: str, digest: int, to_field=lambda x: x) -> list:
    """State-to-digest compression. Validates per `mode`, then delegates.

    perm    : the permutation (state -> state)
    data    : full state, len == t
    kappa   : target security level
    p       : prime
    mode    : "jive"  (Anemoi Jive_b: sum of b=t/d blocks of input and permuted output)
              "trunc" (Davies-Meyer: trunc_d(P(x) + x))
    digest  : output length d
    """
    t = len(data)
    d_min = get_min_digest(kappa, p)

    # --- shared validation ---
    if not (1 <= d < t):
        raise ValueError(f"digest d={d} must satisfy 1 <= d < t={t} (must shrink)")
    if digest < d_min:
        raise ValueError(f"digest d={d} below {kappa}-bit collision floor {d_min}")

    # --- mode-specific validation + dispatch ---
    if mode == "jive":
        
        return compress_jive(perm, data, d, to_field)

    if mode == "trunc":
        trunc_min = get_min_trunc(kappa, p)
        if t - d < trunc_min:
            raise ValueError(f"truncated part t-d={t-d} below {kappa}-bit guessing floor {trunc_min}")
        return compress_trunc(perm, data, d, to_field)

    raise ValueError(f"unknown compression mode {mode}")

def compress_trunc(perm, state: list, d: int, to_field=lambda x: x) -> list:
    """Davies-Meyer / truncation compression: trunc_d(P(x) + x), keeping the first d (= digest size) 
    of the t (= state size) elements. The truncated part is what makes it one-way."""
    out = perm(state)
    return [out[i] + state[i] for i in range(d)] # left-truncate to digest size

# TODO update
def compress_davies_meyer(perm, x_m: list, x_c: list, digest_size: int, to_field=lambda x: x) -> list:
    """Davies-Meyer compression: trunc(perm(x_m || x_c) + (x_m || x_c))."""
    if x_c is None:
        x_c = [to_field(0)] * digest_size
    return compress_trunc(perm, x_m + x_c, digest_size, to_field)

def compress_jive(perm, state: list, d: int, to_field=lambda x: x) -> list:
    """Anemoi's Jive_b (b*d -> d; b-to-1) compression mode (https://eprint.iacr.org/2022/840.pdf, Sec. 3.2):
        Jive_b(x_1, ..., x_b) = sum_j x_j + sum_j P(x_1 || ... || x_b)_j
    where P is the permutation and the state is viewed as b = t/d blocks of equal size d (= digest size). 
    Block i of the output is the field-sum of block i across both the input state and P(state)."""
    t = len(state)
    if t % d != 0:
        raise ValueError(f"Jive_b arity b=t/d must be integer: t={t}, d={d}")
    b = t//d
    out = perm(state)
    return [sum((state[i + d * j] + out[i + d * j] for j in range(b)), to_field(0)) for i in range(d)]
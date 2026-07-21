"""Modes of operation built on top of a permutation.

Turns a fixed-width permutation into a hash/compression function: sponge constructions
(plain, SAFE, padding-injected, Hirose), compression functions (Davies-Meyer, Jive),
the padding rules they use, and rate/capacity/digest derivation.
"""

# Structural imports
from recommendations import recommend

# Math specific imports
from math import ceil, log2

# Custom imports
from utils.matrix import add_to_start, replace_start

# ---------------------------------------------------------------------------
# Mode parameters: derivation and checks
# ---------------------------------------------------------------------------

def get_min_capacity(kappa: int, p: int) -> int:
    """Capacity elements for a kappa-bit target, from the sponge bound c*log2(p) >= 2*kappa
    (2x = birthday half). Uses ceil(log2 p) as bits/element (the honest per-element budget;
    real log2p can leave the exact count a sub-bit short and force an extra element) and
    round not ceil (snap to the nearest element count rather than always inflating past target)."""
    pbits = ceil(log2(p))
    return max(1, round(2 * kappa / pbits))

def get_min_digest(kappa: int, p: int) -> int:
    """Smallest collision-resistant output: d*ceil(log2 p)/2 >= kappa, i.e. the birthday
    bound on the output itself must clear the target. Same 2*kappa budget and rounding as
    get_min_capacity."""
    pbits = ceil(log2(p))
    return max(1, round(2 * kappa / pbits))

def resolve_sponge_params(kappa: int, p: int, t: int, r: int = None, c: int = None, d: int = None, toy: bool = False) -> tuple[int, int, int]:
    """Resolve the sponge parameters (rate r, capacity c, digest d) for a state of size t at
    preimage- and collision-security level kappa over Fp. Any omitted value is derived from the sponge
    bound (get_min_capacity / get_min_digest); any provided value is validated. toy=True downgrades
    the capacity/digest floor checks below from a hard error to a warning."""
    c_min = get_min_capacity(kappa, p)
    d_min = get_min_digest(kappa, p)
    pbits = ceil(log2(p))

    # capacity and rate are tied by r + c = t: fix one, the other follows
    c = c_min if c is None else c
    r = t - c if r is None else r
    d = d_min if d is None else d

    # structural invariant
    if r + c != t:
        raise ValueError(f"sponge invariant violated: r + c = {r + c} != t = {t}")
    if min(r, c, d) < 1:
        raise ValueError(f"need r, c, d >= 1; got r={r}, c={c}, d={d}")

    # collision-security floors at kappa bits
    if c < c_min:
        recommend(f"capacity c={c} ({c * pbits} bits) below {kappa}-bit floor c_min={c_min}", toy)
    if d < d_min:
        recommend(f"digest d={d} ({d * pbits} bits) below {kappa}-bit floor d_min={d_min}", toy)

    # rate must not undercut the capacity
    if r < c:
        raise ValueError(f"rate r={r} < capacity c={c}; state too small (need t >= c + c = {2 * c}, got t={t})")

    return r, c, d



# ---------------------------------------------------------------------------
# Compression modes
# ---------------------------------------------------------------------------

def compress_davies_meyer(perm, x_m: list, x_c: list, digest_size: int, to_field=lambda x: x) -> list:
    """Davies-Meyer compression: trunc(perm(x_m || x_c) + (x_m || x_c))."""
    if x_c is None:
        x_c = [to_field(0)] * digest_size
    x = x_m + x_c
    y = perm(x)
    return [y[i] + x[i] for i in range(digest_size)] # left-truncate to digest_size

def compress_jive(perm, inputs: list[list], b: int = None, to_field=lambda x: x) -> list:
    """Anemoi's Jive_b compression mode (https://eprint.iacr.org/2022/840.pdf, Sec. 3.2):
        Jive_b(x_1, ..., x_b) = sum_j x_j + sum_j P(x_1 || ... || x_b)_j
    where P is the permutation and the state is viewed as b blocks of equal size m = t / b. 
    Compresses b*m field elements to m, effectively giving a b-to-1 compression.
    `inputs` is the list of b blocks (each a list of m elements)."""
    if b is None:
        b = len(inputs)
    if len(inputs) != b:
        raise ValueError(f"expected {b} input blocks, got {len(inputs)}")
    m = len(inputs[0])
    if any(len(blk) != m for blk in inputs):
        raise ValueError("all input blocks must have equal length")

    state = [w for blk in inputs for w in blk]     # x_1 || ... || x_b
    out = perm(state)
    if len(out) != b * m:
        raise ValueError(f"permutation output length {len(out)} != b*m = {b * m}")

    return [sum((state[i + m * j] + out[i + m * j] for j in range(b)), to_field(0)) for i in range(m)]

# ---------------------------------------------------------------------------
# Padding rules
# ---------------------------------------------------------------------------

def pad_zero(data: list, rate: int, to_field=lambda x: x) -> tuple[list, bool]:
    """Zero-pad data to the next multiple of rate. No-op if already aligned. Returns (padded_data, was_aligned)."""
    was_aligned = len(data) % rate == 0
    rem = len(data) % rate
    num_zeros = (rate - rem) % rate
    return data + [to_field(0)] * num_zeros, was_aligned

def pad_one(data: list, rate: int, to_field=lambda x: x) -> tuple[list, bool]:
    """Append a single 1, then zero-pad to the next multiple of rate. Always applied,
    even if data is already aligned. Returns (padded_data, was_aligned)"""
    was_aligned = len(data) % rate == 0
    rem = (len(data) + 1) % rate
    num_zeros = (rate - rem) % rate
    return data + [to_field(1)] + [to_field(0)] * num_zeros, was_aligned

def pad_one_conditional(data: list, rate: int, to_field=lambda x: x) -> tuple[list, bool]:
    """Apply pad_one rule only if len(data) is not a multiple of the rate."""
    if len(data) % rate == 0:
        return list(data), True
    return pad_one(data, rate, to_field)

def pad_fixed_length(data: list, rate: int, to_field=lambda x: x) -> tuple[list, bool]:
    """Padding omitted; only valid when all inputs have known, aligned length."""
    if len(data) % rate != 0:
        raise ValueError(f"Input length must be a multiple of {rate}. Got {len(data)}")
    return list(data), True

def pad_pi(data: list, rate: int, to_field=lambda x: x) -> tuple[list, bool]:
    """Pad with a single 1 followed by zeros to the next multiple of rate. No-op if already aligned. Returns (padded_data, was_aligned)."""
    was_aligned = len(data) % rate == 0
    rem = len(data) % rate
    num_zeros = (rate - rem - 1) % rate
    padding = [to_field(1)] + [to_field(0)] * num_zeros if rem != 0 else []
    return data + padding, was_aligned

# ---------------------------------------------------------------------------
# Sponge modes
# ---------------------------------------------------------------------------

def hash_sponge(perm, data: list, state_size: int, rate: int, capacity: int, digest_size: int, IV: list, absorb=add_to_start, to_field=lambda x: x) -> list:
    """Sponge hash: absorb data in rate-sized blocks, squeeze digest_size elements.
    See https://link.springer.com/chapter/10.1007/978-3-540-78967-3_11 for details."""

    if state_size != rate + capacity:
        raise ValueError("state_size must equal rate + capacity")
    if len(IV) != capacity:
        raise ValueError("IV must have exactly `capacity` elements")
    if len(data) % rate != 0:
        raise ValueError("data must be padded to a multiple of the rate")

    blocks = [data[i:i + rate] for i in range(0, len(data), rate)]

    # Initialize state
    state = [to_field(0)] * rate + list(IV)

    # Sponge - absorption phase (handles any number of rate-sized blocks)
    for block in blocks:
        state = absorb(state, block)
        state = perm(state)

    # Sponge - squeezing phase: read rate-sized chunks of the outer state,
    # re-permuting between chunks until digest_size elements are collected.
    digest = []
    while True:
        digest.extend(state[:min(rate, digest_size - len(digest))])
        if len(digest) == digest_size:
            return digest
        state = perm(state)


def hash_sponge_pi(perm, data: list, state_size: int, rate: int, capacity: int, digest_size: int, IV: list, mu: int, absorb=add_to_start, to_field=lambda x: x) -> list:
    """Sponge-pi hash with domain separation mu on the last absorbed block.
    See https://tosc.iacr.org/index.php/ToSC/article/view/12073 for details."""

    if state_size != rate + capacity:
        raise ValueError("state_size must equal rate + capacity")
    if len(IV) != capacity:
        raise ValueError("IV must have exactly `capacity` elements")
    if len(data) % rate != 0:
        raise ValueError("data must be padded to a multiple of the rate")
    if digest_size > rate:
        raise NotImplementedError("Error: Not implemented -- sponge-pi squeezing over more than one block (digest_size > rate)")
    if IV[-1] != to_field(digest_size):
        raise ValueError("Invalid domain separation: digest_size must be encoded as last element in IV")

    blocks = [data[i:i + rate] for i in range(0, len(data), rate)]

    # Initialize state
    state = [to_field(0)] * rate + list(IV) 

    # Absorption phase
    for k, block in enumerate(blocks):
        state = absorb(state, block)
        if k == len(blocks) - 1:
            state[-1] += to_field(mu)
        state = perm(state)

    # Squeezing phase
    output = []
    while len(output) < digest_size:
        output.extend(state[:rate])
        if len(output) < digest_size:
            state = perm(state)
    return output[:digest_size]


def hash_sponge_hirose(perm, data: list, state_size: int, rate: int, capacity: int, digest_size: int, sigma: int, absorb=add_to_start, to_field=lambda x: x) -> list:
    """Hirose variant of the sponge: zero IV, domain separator sigma added to the last
    state element after the final absorb permutation, per-element squeezing with a
    re-permutation every `rate` elements. Used by Anemoi (https://eprint.iacr.org/2022/840)."""

    if state_size != rate + capacity:
        raise ValueError("state_size must equal rate + capacity")
    if len(data) % rate != 0:
        raise ValueError("data must be padded to a multiple of the rate")

    blocks = [data[i:i + rate] for i in range(0, len(data), rate)]

    # Initialize state
    state = [to_field(0)] * state_size

    # Absorption phase
    for block in blocks:
        state = absorb(state, block)
        state = perm(state)
    state[-1] += to_field(sigma)

    # Squeezing phase
    digest = []
    while True:
        digest.extend(state[:min(rate, digest_size - len(digest))])
        if len(digest) == digest_size:
            return digest
        state = perm(state)


def hash_sponge_safe(perm, data: list, state_size: int, rate: int, capacity: int, digest_size: int, IV: list = None, absorb=add_to_start, to_field=lambda x: x) -> list:
    # SAFE: Sponge API for Field Elements (https://eprint.iacr.org/2023/522)
    # TODO: implement the SAFE IV/tag schedule; this is currently a plain-sponge
    # passthrough, NOT the SAFE construction.
    return hash_sponge(perm, data, state_size, rate, capacity, digest_size, IV, absorb, to_field)

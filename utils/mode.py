"""Modes of operation built on top of a permutation.

Turns a fixed-width permutation into a hash/compression function: sponge constructions
(plain, SAFE, padding-injected, Hirose), compression functions (Davies-Meyer, Jive),
the padding rules they use, and rate/capacity/digest derivation.
"""

from utils.matrix import add_to_start, replace_start

# ---------------------------------------------------------------------------
# Rate / capacity / digest derivation
# ---------------------------------------------------------------------------

def derive_rate_capacity_digest(kappa: int, t: int, r: int = None, c: int = None, d: int = None) -> tuple[int, int, int]:
    """Resolve the sponge/compression parameters (rate r, capacity c, digest size d) for a state of
    size t at security level kappa. The derivation of any omitted value from kappa, t (and the values
    that ARE given) is the same sponge/compression security relation for every primitive, so it lives
    here once rather than in each params class.

    If all three are provided they are returned unchanged (and the t == r + c sponge invariant is
    checked); if any is omitted, it must be derived -- which is not yet implemented.
    """
    if r is not None and c is not None and d is not None:
        if r + c != t:
            raise ValueError(f"sponge invariant violated: r + c = {r + c} != t = {t}")
        return r, c, d
    # TODO: derive the missing rate/capacity/digest from kappa, t (and any provided r/c/d) via the
    # sponge/compression security bound (capacity ~ 2*kappa bits, rate r = t - c, digest d from kappa).
    raise NotImplementedError(
        "automatic rate/capacity/digest derivation not implemented; pass r, c, d explicitly"
    )

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
    if digest_size > rate:
        raise NotImplementedError(...)
    if len(data) > rate:
        raise NotImplementedError(...)

    blocks = [data[i:i + rate] for i in range(0, len(data), rate)]

    # Initialize state
    state = [to_field(0)] * rate + list(IV)

    # Sponge - absorption phase
    for block in blocks:
        state = absorb(state, block)
        state = perm(state)

    # Sponge - squeezing phase
    return state[:digest_size]


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
        raise NotImplementedError(...)
    if IV[-1] != to_field(digest_size):
        raise ValueError("Invalid domain separation: digest_size must be encoded as last element in IV")
    if len(data) > rate:
        raise NotImplementedError(...)

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
    # TODO implement
    return hash_sponge(perm, data, state_size, rate, capacity, digest_size, IV, absorb, to_field)

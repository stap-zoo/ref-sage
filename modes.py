from utils import add_to_start, replace_start

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


def hash_sponge_safe(perm, data: list, state_size: int, rate: int, capacity: int, digest_size: int, IV: list = None, absorb=add_to_start, to_field=lambda x: x) -> list:
    # SAFE: Sponge API for Field Elements (https://eprint.iacr.org/2023/522)
    # TODO implement
    return hash_sponge(perm, data, state_size, rate, capacity, digest_size, IV, absorb, to_field)

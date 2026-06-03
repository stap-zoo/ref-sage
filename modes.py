from utils import add_in_place

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

def pad_zero(data: list, rate: int, to_field=lambda x: x) -> tuple[list, int]:
    """Zero-pad data to the next multiple of rate. No-op if already aligned. Returns (padded_data, was_aligned)."""
    rem = len(data) % rate
    num_zeros = (rate - rem) % rate
    return data + [to_field(0)] * num_zeros, int(rem == 0)

def pad_pi(data: list, rate: int, to_field=lambda x: x) -> tuple[list, int]:
    """Pad with a single 1 followed by zeros to the next multiple of rate. No-op if already aligned. Returns (padded_data, was_aligned)."""
    rem = len(data) % rate
    num_zeros = (rate - rem - 1) % rate
    padding = [to_field(1)] + [to_field(0)] * num_zeros if rem != 0 else []
    return data + padding, int(rem == 0)

# ---------------------------------------------------------------------------
# Sponge modes
# ---------------------------------------------------------------------------

def hash_sponge(perm, data: list, state_size: int, rate: int, capacity: int, digest_size: int, IV: list = None, pad=pad_zero, to_field=lambda x: x) -> list:
    """Sponge hash: absorb data in rate-sized blocks, squeeze digest_size elements.
    See https://link.springer.com/chapter/10.1007/978-3-540-78967-3_11 for details."""

    if state_size != rate + capacity:
        raise ValueError("state_size must equal rate + capacity")

    if digest_size > rate:
        raise NotImplementedError(f"Digest size must be at most rate. Got digest_size={digest_size}, rate={rate}")

    # Apply padding
    data, _ = pad(data, rate, to_field)
    blocks = [data[i:i + rate] for i in range(0, len(data), rate)]
    assert all(len(block) == rate for block in blocks)

    # Initialize state
    if IV is None:
        IV = [to_field(0)] * capacity
    state = [to_field(0)] * rate + list(IV)

    # Sponge - absorption phase
    for block in blocks:
        add_in_place(state, block)
        state = perm(state)

    # Sponge - squeezing phase
    return state[:digest_size]


def hash_sponge_pi(perm, data: list, state_size: int, rate: int, capacity: int, digest_size: int, IV: list = None, pad=pad_pi, to_field=lambda x: x) -> list:
    """Sponge-pi hash with domain separation on the last absorbed block.
    See https://tosc.iacr.org/index.php/ToSC/article/view/12073 for details."""

    if state_size != rate + capacity:
        raise ValueError("state_size must equal rate + capacity")

    if digest_size > rate:
        raise NotImplementedError(f"Digest size must be at most rate. Got digest_size={digest_size}, rate={rate}")

    # Apply padding
    data, mu = pad(data, rate, to_field)
    blocks = [data[i:i + rate] for i in range(0, len(data), rate)]
    assert all(len(block) == rate for block in blocks)

    # Initialize state
    if IV is None:
        IV = [to_field(0)] * (capacity - 1)
    if len(IV) != capacity - 1:
        raise ValueError(f"IV must have length capacity - 1. Got IV of length {len(IV)}, expected {capacity - 1}")
    state = [to_field(0)] * rate + list(IV) + [to_field(digest_size)]  # domain separation: digest_size in last element

    # Absorption phase
    for k, block in enumerate(blocks):
        add_in_place(state, block)
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

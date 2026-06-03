# ref
Python/Sage reference implementations of STAP primitives, intended for correctness verification of optimized implementations and reuse in cryptanalytic research.

## Structure

Shared modules at the root level:

- **`fields.py`** frozen dataclass definitions for the prime fields used across primitives (BLS12-381, BN254, ST, Goldilocks, ...). Each entry stores the characteristic `p`, the smallest permutation exponent `alpha` with `gcd(alpha, p-1) = 1`, its modular inverse `alpha_inv`, and auxiliary parameters.
- **`modes.py`** field-agnostic hash construction modes that can be instantiated by any permutation:
  - `compress_davies_meyer` Davies-Meyer compression: `trunc(perm(x_m ∥ x_c) + (x_m ∥ x_c))`.
  - `hash_sponge` standard sponge (absorb rate-sized blocks with zero-padding, squeeze `digest_size` elements).
  - `hash_sponge_pi` variant of arithmetization-Oriented sponges sponge-pi, see [Lefevre et al., ToSC 2025](https://tosc.iacr.org/index.php/ToSC/article/view/12073).
  - `pad_zero` / `pad_pi` padding rules used by the sponge variants.
- **`utils.py`** low-level linear-algebra helpers

Each primitive lives in its own folder and follows a common layout:

- **`hash.py`** implements the permutation and higher-level hash modes (compression, sponge). Takes a concrete parameter instance as its constructor argument.
- **`instances.py`** defines concrete instances (e.g. BLS12-381, BN254) by constructing a params object with the appropriate constants.
- **`params.py`** defines the params class whose constructor validates and stores all numerical parameters. Also contains helper routines for derived values such as the MDS matrix and round constants.

## How to run

The code requires SageMath. How to invoke it depends on the installation:

- **Standalone SageMath** (SageMath ships its own Python):
  ```sh
  sage --python script.py
  sage --python -m pytest tests/
  ```
- **SageMath as a Python package** (installed via `pip` into an existing Python environment):
  ```sh
  python script.py
  pytest tests/
  ```

Developed with `SageMath 10.6` using `Python 3.12.5`.

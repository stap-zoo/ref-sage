# ref
Python/Sage reference implementations of STAP primitives, intended for correctness verification of optimized implementations and reuse in cryptanalytic research.

## Structure

Shared modules at the root level:

- **`fields.py`** frozen dataclass definitions for the prime fields used across primitives (BLS12-381, BN254, ST, Goldilocks, ...). Each entry stores the characteristic `p`, the smallest permutation exponent `alpha` with `gcd(alpha, p-1) = 1`, its modular inverse `alpha_inv`, and auxiliary parameters.
- **`modes.py`** field-agnostic hash construction modes that can be instantiated by any permutation:
  - `compress_davies_meyer` Davies-Meyer compression: `trunc(perm(x_m ∥ x_c) + (x_m ∥ x_c))`.
  - `hash_sponge` standard sponge (absorb rate-sized blocks with zero-padding, squeeze `digest_size` elements).
  - `hash_sponge_pi` variant *sponge-pi* of arithmetization-oriented sponges, see [Lefevre et al., ToSC 2025](https://tosc.iacr.org/index.php/ToSC/article/view/12073).
  - `hash_sponge_safe` Sponge API *SAFE* for Field Elements, see [Aumasson et al., ePrint](https://eprint.iacr.org/2023/522).
  - `pad_zero` / `pad_pi` padding rules used by the sponge variants.
- **`utils.py`** shared helpers

Each primitive lives in its own folder and follows a common layout:

- **`hash.py`** implements the permutation and higher-level hash modes (compression, sponge). Takes a concrete parameter instance as its constructor argument.
- **`instances.py`** defines concrete instances (e.g. BLS12-381, BN254) by constructing a params object with the appropriate constants.
- **`params.py`** defines the params class whose constructor validates and stores all numerical parameters. Also contains helper routines for derived values such as the MDS matrix and round constants.

## Primitives

<table>
  <thead>
    <tr>
      <th>Primitive</th>
      <th>Instance</th>
      <th>Field</th>
      <th>State size</th>
      <th>Rounds</th>
      <th>Hash modes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><a href="https://eprint.iacr.org/2021/1038">Reinforced Concrete</a></td>
      <td><code>RC_BLS12_T3</code></td>
      <td>BLS12-381 scalar (255 bit)</td>
      <td>3</td>
      <td>3+1+3</td>
      <td>2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>RC_BN254_T3</code></td>
      <td>BN254 scalar (254 bit)</td>
      <td>3</td>
      <td>3+1+3</td>
      <td>2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>RC_ST_T3</code></td>
      <td>ST (250 bit)</td>
      <td>3</td>
      <td>3+1+3</td>
      <td>2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://eprint.iacr.org/2023/1025">Monolith</a></td>
      <td><code>MONOLITH_M31_T16</code></td>
      <td>Mersenne-31 (31 bit)</td>
      <td>16</td>
      <td>6</td>
      <td>2-to-1 compression</td>
    </tr>
    <tr>
      <td><code>MONOLITH_M31_T24</code></td>
      <td>Mersenne-31 (31 bit)</td>
      <td>24</td>
      <td>6</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>MONOLITH_GOLDILOCKS_T8</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>8</td>
      <td>6</td>
      <td>2-to-1 compression</td>
    </tr>
    <tr>
      <td><code>MONOLITH_GOLDILOCKS_T12</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>12</td>
      <td>6</td>
      <td>sponge</td>
    </tr>
  </tbody>
</table>

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

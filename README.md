# ref
Python/Sage reference implementations of STAP primitives, intended for correctness verification of optimized implementations and reuse in cryptanalytic research.

## Structure

Shared code lives in the `utils/` package plus one root-level module, `recommendations.py`:

- **`utils/field.py`** the `Field` frozen dataclass and the predefined prime fields used across primitives (BLS12-381, BN254, ST, Goldilocks, Mersenne-31, Pallas/Vesta, ...). Each entry stores the characteristic `p`, extension degree `n` and bit size, the factorization of `p-1`, a multiplicative `generator`, and the smallest permutation exponent `alpha` with `gcd(alpha, p-1) = 1` together with its modular inverse `alpha_inv`.
- **`utils/mode.py`** field-agnostic hash construction modes that can be instantiated by any permutation:
  - The `Sponge` base class and its variants, each fixing an absorb/squeeze convention and padding rule: `SpongePlain`, `SpongeLE` (little-endian rate ordering), `SpongeCLE`, `Sponge2`, `SpongeRescue`, `SpongeRPO`, `SpongeHirose` (Hirose variant with a domain separator after the final absorption, used by Anemoi), `SpongePI` (*sponge-pi*, see [Lefevre et al., ToSC 2025](https://tosc.iacr.org/index.php/ToSC/article/view/12073)), and `SpongeSAFE` (Sponge API *SAFE* for Field Elements, see [Aumasson et al., ePrint](https://eprint.iacr.org/2023/522)).
  - Compression functions: `compress_davies_meyer` Davies-Meyer compression `trunc(perm(x_m ∥ x_c) + (x_m ∥ x_c))`; `compress_jive` *Jive_b* compression `out[i] = sum_j (x[i+c*j] + perm(x)[i+c*j])`, see [Bouvier et al., CRYPTO 2023](https://eprint.iacr.org/2022/840) (Anemoi paper); `compress_trunc` truncation; and the generic `compress` dispatcher with `resolve_compression_params`.
  - `pad_zero` / `pad_simple` padding rules used by the sponge and compression variants.
- **`utils/matrix.py`** matrix/vector arithmetic, nested-list mapping, and MDS/diffusion-matrix constructions (circulant, Cauchy/Vandermonde, M4 block-circulant, ...).
- **`utils/sampler.py`** deterministic field-element samplers (XOF-seeded) used to derive round constants reproducibly.
- **`utils/lut.py`** lookup-table helpers for LUT-based constructions (Reinforced Concrete, Monolith, Skyscraper).
- **`utils/complexities.py`** attack-complexity estimators (Gröbner-basis, differential, ...) used by the round-number derivations.
- **`utils/poly.py`** multivariate-polynomial representations and coordinate polynomials of power maps, for algebraic cryptanalysis.
- **`recommendations.py`** the `ParamRecommendationWarning` / `ModeRecommendationWarning` categories and the `recommend` helper: recommendation-level checks warn for toy instances and raise otherwise.

Each primitive lives in its own folder and follows a common layout:

- **`hash.py`** implements the permutation and higher-level hash modes (compression, sponge). Takes a concrete parameter instance as its constructor argument.
- **`instances.py`** defines concrete instances (e.g. BLS12-381, BN254) by constructing a params object with the appropriate constants.
- **`params.py`** defines the params class whose constructor validates and stores all numerical parameters. Every params class implements the same five-function contract (see `myprimitive/README.md` for details):
  - `_input_sanitization` validates the raw constructor arguments (hard checks raise, deviations from the recommended settings warn with `ParamRecommendationWarning`);
  - `_parameter_sanitization` validates the fully-constructed object (shapes and counts of the stored/derived values) as the constructor's last step;
  - `_init_rounds` derives the round number(s), `_init_cons` the (round) constants, and `_init_mat` the matrix, whenever the user does not pass them explicitly. Derivations that are not worked out yet exist as stubs raising `NotImplementedError("Error: Not implemented -- ...")` with a `TODO` docstring.

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
      <td rowspan="9"><a href="https://eprint.iacr.org/2022/840">Anemoi</a></td>
      <td><code>ANEMOI_BLS12_381_BASE_T2/T4/T6</code></td>
      <td>BLS12-381 base (381 bit)</td>
      <td>2 / 4 / 6</td>
      <td>21 / 14 / 12</td>
      <td>Jive 2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>ANEMOI_BLS12_381_SCALAR_T2/T4/T6</code></td>
      <td>BLS12-381 scalar (255 bit)</td>
      <td>2 / 4 / 6</td>
      <td>21 / 14 / 12</td>
      <td>Jive 2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>ANEMOI_BLS12_377_BASE_T2/T4/T6</code></td>
      <td>BLS12-377 base (377 bit)</td>
      <td>2 / 4 / 6</td>
      <td>21 / 14 / 12</td>
      <td>Jive 2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>ANEMOI_BLS12_377_SCALAR_T2/T4/T6</code></td>
      <td>BLS12-377 scalar (253 bit)</td>
      <td>2 / 4 / 6</td>
      <td>19 / 13 / 11</td>
      <td>Jive 2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>ANEMOI_BN254_BASE_T2/T4/T6</code></td>
      <td>BN254 base (254 bit)</td>
      <td>2 / 4 / 6</td>
      <td>21 / 14 / 12</td>
      <td>Jive 2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>ANEMOI_BN254_SCALAR_T2/T4/T6</code></td>
      <td>BN254 scalar (254 bit)</td>
      <td>2 / 4 / 6</td>
      <td>21 / 14 / 12</td>
      <td>Jive 2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>ANEMOI_PALLAS_T2/T4/T6</code></td>
      <td>Pallas (255 bit)</td>
      <td>2 / 4 / 6</td>
      <td>21 / 14 / 12</td>
      <td>Jive 2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>ANEMOI_VESTA_T2/T4/T6</code></td>
      <td>Vesta (255 bit)</td>
      <td>2 / 4 / 6</td>
      <td>21 / 14 / 12</td>
      <td>Jive 2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>ANEMOI_GOLDILOCKS_T8/T10/T12</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>8 / 10 / 12</td>
      <td>11 / 11 / 10</td>
      <td>Jive 2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><a href="https://eprint.iacr.org/2023/588">Arion</a></td>
      <td><code>ARION_BLS12_T3</code></td>
      <td>BLS12-381 scalar (255 bit)</td>
      <td>3</td>
      <td>6</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td rowspan="5"><a href="https://eprint.iacr.org/2022/403">Griffin</a></td>
      <td><code>GRIFFIN_GOLDILOCKS_T8</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>8</td>
      <td>8</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>GRIFFIN_GOLDILOCKS_T12</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>12</td>
      <td>8</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>GRIFFIN_ST_T3</code></td>
      <td>ST (250 bit)</td>
      <td>3</td>
      <td>16</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>GRIFFIN_BN254_T3</code></td>
      <td>BN254 scalar (254 bit)</td>
      <td>3</td>
      <td>12</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>GRIFFIN_BLS12_T3</code></td>
      <td>BLS12-381 scalar (255 bit)</td>
      <td>3</td>
      <td>12</td>
      <td>sponge</td>
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
    <tr>
      <td rowspan="4"><a href="https://eprint.iacr.org/2025/926">Polocolo</a></td>
      <td><code>POLOCOLO_BLS12_381_SCALAR_T3/T4/T5/T6/T7/T8</code></td>
      <td>BLS12-381 scalar (255 bit)</td>
      <td>3 / 4 / 5 / 6 / 7 / 8</td>
      <td>6 / 5 / 5 / 5 / 5 / 5</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>POLOCOLO_BLS12_381_SCALAR_T3/T4/T6/T8_TIGHT</code></td>
      <td>BLS12-381 scalar (255 bit)</td>
      <td>3 / 4 / 6 / 8</td>
      <td>5 / 4 / 4 / 4</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>POLOCOLO_BN254_SCALAR_T3/T4/T5/T6/T7/T8</code></td>
      <td>BN254 scalar (254 bit)</td>
      <td>3 / 4 / 5 / 6 / 7 / 8</td>
      <td>6 / 5 / 5 / 5 / 5 / 5</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>POLOCOLO_BN254_SCALAR_T3/T4/T6/T8_TIGHT</code></td>
      <td>BN254 scalar (254 bit)</td>
      <td>3 / 4 / 6 / 8</td>
      <td>5 / 4 / 4 / 4</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://eprint.iacr.org/2021/1038">Reinforced Concrete</a></td>
      <td><code>RC_ST_T3</code></td>
      <td>ST (250 bit)</td>
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
      <td><code>RC_BLS12_T3</code></td>
      <td>BLS12-381 scalar (255 bit)</td>
      <td>3</td>
      <td>3+1+3</td>
      <td>2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td rowspan="7"><a href="https://eprint.iacr.org/2019/426">Rescue</a></td>
      <td><code>RESCUE_STARKWARE_T12</code></td>
      <td>StarkWare (62 bit)</td>
      <td>12</td>
      <td>10</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>RESCUE_GOLDILOCKS_T12</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>12</td>
      <td>10</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>RESCUE_ST_T3</code></td>
      <td>ST (250 bit)</td>
      <td>3</td>
      <td>22</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>RESCUE_ED25519_T6</code></td>
      <td>Ed25519 scalar (253 bit)</td>
      <td>6</td>
      <td>10</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>RESCUE_BN254_T3</code></td>
      <td>BN254 scalar (254 bit)</td>
      <td>3</td>
      <td>16</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>RESCUE_BLS12_T3</code></td>
      <td>BLS12-381 scalar (255 bit)</td>
      <td>3</td>
      <td>16</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>RESCUE_ED448_T10</code></td>
      <td>Ed448 scalar (446 bit)</td>
      <td>10</td>
      <td>10</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td rowspan="5"><a href="https://eprint.iacr.org/2020/1143">Rescue Prime</a></td>
      <td><code>RESCUE_PRIME_GOLDILOCKS_T8</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>8</td>
      <td>8</td>
      <td>sponge (pad-one)</td>
    </tr>
    <tr>
      <td><code>RESCUE_PRIME_GOLDILOCKS_T12</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>12</td>
      <td>8</td>
      <td>sponge (pad-one)</td>
    </tr>
    <tr>
      <td><code>RESCUE_PRIME_ST_T3</code></td>
      <td>ST (250 bit)</td>
      <td>3</td>
      <td>18</td>
      <td>sponge (pad-one)</td>
    </tr>
    <tr>
      <td><code>RESCUE_PRIME_BN254_T3</code></td>
      <td>BN254 scalar (254 bit)</td>
      <td>3</td>
      <td>14</td>
      <td>sponge (pad-one)</td>
    </tr>
    <tr>
      <td><code>RESCUE_PRIME_BLS12_T3</code></td>
      <td>BLS12-381 scalar (255 bit)</td>
      <td>3</td>
      <td>14</td>
      <td>sponge (pad-one)</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://eprint.iacr.org/2022/1577">Rescue Prime Optimized</a></td>
      <td><code>RPO_GOLDILOCKS_T12</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>12</td>
      <td>7</td>
      <td>fixed-output sponge (2-to-1 compression)</td>
    </tr>
    <tr>
      <td><code>RPO_GOLDILOCKS_T16</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>16</td>
      <td>7</td>
      <td>fixed-output sponge (2-to-1 compression)</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://eprint.iacr.org/2023/107">Tip5</a></td>
      <td><code>TIP5</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>16</td>
      <td>5</td>
      <td>fixed-length sponge</td>
    </tr>
    <tr>
      <td><code>TIP4</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>16</td>
      <td>5</td>
      <td>fixed-length sponge</td>
    </tr>
    <tr>
      <td><code>TIP4_PRIME</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>12</td>
      <td>5</td>
      <td>fixed-length sponge</td>
    </tr>
    <tr>
      <td rowspan="5"><a href="https://eprint.iacr.org/2019/458">Poseidon</a></td>
      <td><code>POSEIDON_BN254_T3</code></td>
      <td>BN254 scalar (254 bit)</td>
      <td>3</td>
      <td>8 + 57</td>
      <td>sponge, Davies–Meyer 2-to-1</td>
    </tr>
    <tr>
      <td><code>POSEIDON_BLS12_T2/T3</code></td>
      <td>BLS12-381 scalar (255 bit)</td>
      <td>2 / 3</td>
      <td>8 + 56 / 8 + 57</td>
      <td>sponge, Davies–Meyer 2-to-1</td>
    </tr>
    <tr>
      <td><code>POSEIDON_ST_T3</code></td>
      <td>ST (250 bit)</td>
      <td>3</td>
      <td>8 + 84</td>
      <td>sponge, Davies–Meyer 2-to-1</td>
    </tr>
    <tr>
      <td><code>POSEIDON_GOLDILOCKS_T8/T12</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>8 / 12</td>
      <td>8 + 22</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>POSEIDON_MERSENNE_T16/T24</code></td>
      <td>Mersenne31 (31 bit)</td>
      <td>16 / 24</td>
      <td>8 + 14 / 8 + 22</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://eprint.iacr.org/2023/323">Poseidon2</a></td>
      <td><code>POSEIDON2_BLS12_T2/T3/T4/T8</code></td>
      <td>BLS12-381 scalar (255 bit)</td>
      <td>2 / 3 / 4 / 8</td>
      <td>8 + 56 (t=8: 8 + 57)</td>
      <td>sponge, Davies–Meyer 2-to-1</td>
    </tr>
    <tr>
      <td><code>POSEIDON2_BN254_T3</code></td>
      <td>BN254 scalar (254 bit)</td>
      <td>3</td>
      <td>8 + 56</td>
      <td>sponge, Davies–Meyer 2-to-1</td>
    </tr>
    <tr>
      <td><code>POSEIDON2_GOLDILOCKS_T8/T12/T16/T20</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>8 / 12 / 16 / 20</td>
      <td>8 + 22</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td><code>POSEIDON2_MERSENNE_T16/T24</code></td>
      <td>Mersenne31 (31 bit)</td>
      <td>16 / 24</td>
      <td>8 + 14 / 8 + 22</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://eprint.iacr.org/2021/1695">Neptune</a></td>
      <td><code>NEPTUNE_BN254_T4</code> / <code>NEPTUNE_BLS12_T2/T4</code></td>
      <td>BN254 (254 bit) / BLS12-381 (255 bit)</td>
      <td>4 / 2 / 4</td>
      <td>6 + 68 / 8 + 56 / 6 + 68</td>
      <td>sponge, Davies–Meyer 2-to-1</td>
    </tr>
    <tr>
      <td><code>NEPTUNE_ST_T4</code></td>
      <td>ST (250 bit)</td>
      <td>4</td>
      <td>6 + 96</td>
      <td>sponge, Davies–Meyer 2-to-1</td>
    </tr>
    <tr>
      <td><code>NEPTUNE_GOLDILOCKS_T8/T12</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>8 / 12</td>
      <td>6 + 38 / 6 + 42</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://eprint.iacr.org/2026/1129">pSquare-hash</a></td>
      <td><code>PSQUAREHASH_MERSENNE_T16/T24</code></td>
      <td>Mersenne31 (31 bit)</td>
      <td>16 / 24</td>
      <td>52</td>
      <td>sponge, Davies–Meyer 2-to-1</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="https://eprint.iacr.org/2019/397">GMiMC</a></td>
      <td><code>GMIMC_BN254_T3/T4</code> / <code>GMIMC_BLS12_T3/T4</code></td>
      <td>BN254 (254 bit) / BLS12-381 (255 bit)</td>
      <td>3 / 4</td>
      <td>228 / 231</td>
      <td>sponge</td>
      <tr>
      <td><code>GMIMC_GOLDILOCKS_T8/T12</code>
      <td>Goldilocks (64 bit)</td>
      <td>8 / 12</td>
      <td>68 / 93</td>
      <td>sponge</td>
      </tr>
      <td><code>GMIMC_MERSENNE_T12/T24</code>
      <td>Mersenne 31 (31 bits)</td>
      <td>16 / 24</td>
      <td>158 / 335</td>
      <td>sponge</td>
      </tr>
    </tr>
    <tr>
      <td rowspan="3">GMiMCHash2</td>
      <td><code>GMIMC2_BN254_T4</code> / <code>GMIMC2_BLS12T4</code></td>
      <td>BN254 (254 bit) / BLS12-381 (255 bit)</td>
      <td>4</td>
      <td>64</td>
      <td>sponge</td>
      <tr>
      <td><code>GMIMC2_GOLDILOCKS_T8/T12</code>
      <td>Goldilocks (64 bit)</td>
      <td>8 / 12</td>
      <td>88 / 96</td>
      <td>(2-to-1 compression), sponge</td>
      </tr>
      <td><code>GMIMC2_MERSENNE_T12/T24</code>
      <td>Mersenne 31 (31 bits)</td>
      <td>16 / 24</td>
      <td>176 / 264</td>
      <td>(2-to-1 compression), sponge</td>
      </tr>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://eprint.iacr.org/2021/984">Grendel</a></td>
      <td><code>TOY_GRENDEL_65519</code> / <code>TOY_GRENDEL_65393</code></td>
      <td>p65519 / p65393 (16 bit, toy)</td>
      <td>2</td>
      <td>7 / 5</td>
      <td>sponge</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://eprint.iacr.org/2025/058">Skyscraper</a></td>
      <td><code>SKYSCRAPER_BLS12_381_N1/N2/N3</code></td>
      <td>BLS12-381 scalar (255 bit), GF(p^n) for n = 1/2/3</td>
      <td>2 branches (2n elements)</td>
      <td>18</td>
      <td>2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>SKYSCRAPER_BN254_N1/N2/N3</code></td>
      <td>BN254 scalar (254 bit), GF(p^n) for n = 1/2/3</td>
      <td>2 branches (2n elements)</td>
      <td>18</td>
      <td>2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>SKYSCRAPER_PALLAS_N1/N2/N3</code></td>
      <td>Pallas (255 bit), GF(p^n) for n = 1/2/3</td>
      <td>2 branches (2n elements)</td>
      <td>18</td>
      <td>2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td><code>SKYSCRAPER_VESTA_N1/N2/N3</code></td>
      <td>Vesta (255 bit), GF(p^n) for n = 1/2/3</td>
      <td>2 branches (2n elements)</td>
      <td>18</td>
      <td>2-to-1 compression, sponge</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://eprint.iacr.org/2023/1045">XHash</a></td>
      <td><code>XHASH12_GOLDILOCKS_T12</code> / <code>XHASH8_GOLDILOCKS_T12</code></td>
      <td>Goldilocks (64 bit)</td>
      <td>12</td>
      <td>6</td>
      <td>fixed-length sponge</td>
    </tr>
    <tr>
      <td><code>XHASH24_M31_T24</code> / <code>XHASH16_M31_T24</code></td>
      <td>Mersenne-31 (31 bit)</td>
      <td>24</td>
      <td>6</td>
      <td>fixed-length sponge</td>
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

### Requirements

Besides SageMath itself, the following packages must be installed into the Python environment Sage uses:

- **`blake3`** — XOF used to derive the Tip5 round constants (`utils.FieldElementSampler`).
- **`pytest`** — only needed to run the test suite.

```sh
sage --pip install blake3 pytest   # standalone SageMath
pip install blake3 pytest          # SageMath as a Python package
```

Developed with `SageMath 10.6` using `Python 3.12.5`, `blake3 1.0.8`, and `pytest 8.3.2`.

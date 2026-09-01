# ref-sage
Python/Sage reference implementations of STAP primitives, intended for correctness verification of optimized implementations and reuse in cryptanalytic research.

## Structure

The whole library is built on one split: a hash is a **permutation** plus a **mode of operation**. A permutation knows nothing about hashing, a mode works on any permutation, and the two are combined by *composition*. Everything below follows from that.

### The object model (`utils/primitive.py`)

Three base classes. A concrete primitive subclasses `Permutation`; the two mode classes are used as-is, each *wrapping* a permutation:

| Class | What it is | Exposes | Constructed from |
|---|---|---|---|
| `Permutation` | the bare round function `P: F_p^t → F_p^t` | `permute`, `permute_inv` | a `<Name>Params` object |
| `HashFunction` | a permutation **+ a sponge** | `hash` | a `Permutation` + its `sponge` dict |
| `CompressionFunction` | a permutation **+ a compression** | `compress` | a `Permutation` + its `comp` dict |

A `HashFunction` / `CompressionFunction` is *not* a permutation and does not expose `permute` (the wrapped permutation is reachable as `.permutation` if needed). Each mode class pins only its **kind** as a class attribute (`SPONGE_KIND` / `COMP_KIND`); the actual sizes come from the instance's mode dict. Putting it together:

```python
from anemoi.hash import AnemoiPerm, AnemoiHash, AnemoiCompress
from anemoi.instances import ANEMOI_BLS12_381_SCALAR_T2 as params

P = AnemoiPerm(params)                 # the permutation
H = AnemoiHash(P, params.sponge)       # sponge hash on top of it
C = AnemoiCompress(P, params.comp)     # Jive compression on top of it
digest = H.hash(data)                  # variable-length input -> d elements
node   = C.compress(state)             # full t-element state   -> d elements
```

### Per-primitive files

Each primitive lives in its own folder; all three files consume a single `<Name>Params` object:

- **`params.py`**: `<Name>Params`, the single source of truth for an instance. The constructor sanitizes the user-facing arguments and fills in everything left unspecified, following the shared **five-function contract**: `_input_sanitization` (raw-argument checks: hard checks raise, deviations from the recommended settings warn with `ParamRecommendationWarning`), `_parameter_sanitization` (whole-object checks, run last), and `_init_rounds` / `_init_cons` / `_init_mat` (derive the round number, round constants, and matrix when not given). Unfinished derivations are stubs raising `NotImplementedError`. The mode parameters are stored as dicts, `sponge=dict(r, c, d)` and `comp=dict(...)`, each `None` when that mode is undefined for the instance. *(Full authoring walkthrough: `myprimitive/README.md`.)*
- **`hash.py`**: the classes from the object model, specialized: `<Name>Perm(Permutation)` (the round function), `<Name>Hash(HashFunction)`, and, where the primitive defines one, `<Name>Compress(CompressionFunction)`. A pure consumer of params: it only *applies* the parameters, never derives or validates them.
- **`instances.py`**: named `<Name>Params` instances, `<PREFIX>_<FIELD>_<VARIANT>` (e.g. `ANEMOI_BLS12_381_SCALAR_T2`). Most instances are sponge-only (`comp=None`); a primitive that is itself a compression function (e.g. Skyscraper) sets `comp` too.

### Shared building blocks (`utils/` + `recommendations.py`)

- **`utils/mode.py`**: the field-agnostic **mode catalog** that any permutation can be plugged into, plus the factories that build a mode from a *kind* string:
  - `Sponge` and its variants, each fixing an absorb/squeeze convention and padding rule: `SpongePlain`, `SpongeLE` (little-endian rate ordering), `SpongeCLE`, `Sponge2`, `SpongeRescue`, `SpongeRPO`, `SpongeHirose` (Hirose variant with a domain separator after the final absorption, used by Anemoi), `SpongePI` (*sponge-pi*, see [Lefevre et al., ToSC 2025](https://tosc.iacr.org/index.php/ToSC/article/view/12073)), and `SpongeSAFE` (*SAFE* API for field elements, see [Aumasson et al., ePrint](https://eprint.iacr.org/2023/522)).
  - `Compression` and its variants, all of the form `M·(P(x) + x)`: `Compression` (default `M = I_{d×t}` = truncation / Davies-Meyer) and `CompressionJive` (*Jive_b*, `M = [I_d | … | I_d]`, see [Bouvier et al., CRYPTO 2023](https://eprint.iacr.org/2022/840)). Compression *through a sponge* is not one of these, it is the `Sponge.compress` method (an `a·d → d` node using the sponge's capacity/squeeze), called directly as `H.sponge.compress(perm, state)`.
  - The security floors are methods on the mode classes (`Sponge.get_min_capacity` / `get_min_digest`; `Compression.get_min_digest` / `get_min_trunc`): they warn on a toy mode and raise otherwise.
  - The factories `make_sponge(kind, …)` / `make_compression(kind, …)` map a kind string to a built mode object; these are what `HashFunction` / `CompressionFunction` call internally.
- **`utils/field.py`**: the `Field` frozen dataclass and the predefined prime fields (BLS12-381, BN254, ST, Goldilocks, Mersenne-31, Pallas/Vesta, ...). Each stores `p`, extension degree `n` and bit size, the factorization of `p-1`, a `generator`, and the smallest permutation exponent `alpha` with `gcd(alpha, p-1) = 1` plus its inverse `alpha_inv`.
- **`utils/matrix.py`**: matrix/vector arithmetic and MDS/diffusion-matrix constructions (circulant, Cauchy/Vandermonde, M4 block-circulant, ...).
- **`utils/sampler.py`**: deterministic (XOF-seeded) field-element samplers used to derive round constants reproducibly.
- **`utils/lut.py`**: lookup-table helpers for LUT-based constructions (Reinforced Concrete, Monolith, Skyscraper).
- **`utils/complexities.py`**: attack-complexity estimators (Gröbner-basis, differential, ...) used by the round-number derivations.
- **`utils/poly.py`**: multivariate-polynomial representations and coordinate polynomials of power maps, for algebraic cryptanalysis.
- **`recommendations.py`**: the `ParamRecommendationWarning` / `ModeRecommendationWarning` categories and the `recommend` helper: recommendation-level checks warn for toy instances and raise otherwise.

### How to add a mode

A mode is generic: implement it once in `utils/mode.py` and every primitive can use it. To add a **sponge** variant:

1. **Subclass `Sponge`** in `utils/mode.py`. The base provides `hash`, size resolution, and the security floors; a variant overrides only what differs, the padding rule `pad(data, input_len_fixed, rate_aligned)` and, if its IV/domain-separation differs, `make_iv(...)`.
2. **Register its kind** in the `_SPONGE_KINDS` dict (e.g. `"myhash": SpongeMyHash`).
3. **Point a primitive at it** by setting `SPONGE_KIND = "myhash"` on that primitive's `<Name>Hash` class, or, per instance, with a `"kind"` key in the `sponge` dict (`sponge=dict(r=…, c=…, d=…, kind="myhash")`).

Adding a **compression** variant is the same shape: subclass `Compression`, override `_default_M()` to return the `d×t` matrix that defines the mode (the base's `compress` computes `M·(P(x)+x)`), add a branch to `make_compression`, and set `COMP_KIND` on the primitive's `<Name>Compress` class. A compression realized *through a sponge* rather than a matrix is not a `Compression` mode at all, it is the `Sponge.compress` method (`a·d → d`), called directly as `H.sponge.compress(perm, state)` with no `Compress` class.

In both cases the security floors come for free from the base class; a deliberately weak (toy) instance sets `toy=True` inside its mode dict, which turns floor violations from errors into warnings, independent of whether the permutation itself is a toy instance.

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

- **`blake3`**: XOF used to derive the Tip5 round constants (`utils.FieldElementSampler`).
- **`pytest`**: only needed to run the test suite.

```sh
sage --pip install blake3 pytest   # standalone SageMath
pip install blake3 pytest          # SageMath as a Python package
```

Developed with `SageMath 10.6` using `Python 3.12.5`, `blake3 1.0.8`, and `pytest 8.3.2`.

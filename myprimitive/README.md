# Adding a New Primitive to the Framework

This guide walks through what you need to implement when adding a new arithmetization-oriented primitive (a hash / permutation such as Anemoi, Poseidon, Rescue, ...). Each primitive (or family of primitives) lives in its own module and exposes the same handful of files. We use **MyPrimitive** as the running example throughout.

> **Convention.** Every primitive starts from a `<Name>Params` class defined in its `params.py`. This object is the single source of truth: it sets the field, matrices, round constants, round numbers etc., eventually deriving them if not given. Every other file (the permutation, the sponge/hash wrapper, the test vectors) consumes a `<Name>Params` instance.

---

## Contents

- [1. `params.py`](#1-paramspy)
  - [1.1 Module-level constants](#11-module-level-constants)
  - [1.2 The `<Name>Params` constructor](#12-the-nameparams-constructor)
  - [1.3 Input and parameter sanitization](#13-input-and-parameter-sanitization)
  - [1.4 Field-conversion helpers](#14-field-conversion-helpers)
  - [1.5 The `_init_*` derivation helpers](#15-the-_init_-derivation-helpers)
  - [1.6 Checklist for a new `params.py`](#16-checklist-for-a-new-paramspy)
- [2. `instances.py`](#2-instancespy)
  - [2.1 Fields come from `utils/field.py`](#21-fields-come-from-utilsfieldpy)
  - [2.2 Adding a new field](#22-adding-a-new-field)
  - [2.3 Defining an instance](#23-defining-an-instance)
- [3. `hash.py`](#3-hashpy)
  - [3.1 Component layers and the round index](#31-component-layers-and-the-round-index)
  - [3.2 Layer ordering vs. layer behaviour](#32-layer-ordering-vs-layer-behaviour)
  - [3.3 Pre-/post-round steps](#33-pre-post-round-steps)
  - [3.4 The permutation and its inverse](#34-the-permutation-and-its-inverse)
  - [3.5 Hash modes](#35-hash-modes)
  - [3.6 Running components over polynomials (cryptanalysis)](#36-running-components-over-polynomials-cryptanalysis)
- [4. Tests](#4-tests)
  - [4.1 Known-answer tests (KATs)](#41-known-answer-tests-kats)
  - [4.2 Roundtrip (invertibility) tests](#42-roundtrip-invertibility-tests)
  - [4.3 Consistency tests](#43-consistency-tests)
  - [4.4 Algebraic tests](#44-algebraic-tests)
  - [4.5 Anything else worth adding](#45-anything-else-worth-adding)

---

## 1. `params.py`

The job of `params.py` is to take a small set of *user-facing* parameters and expand them into a *fully-specified* instance, filling in every value the primitive needs with sensible, design-driven defaults whenever the user does not provide them. The `params.py` template (`MyPrimitiveParams`) is the starting point: copy it and fill in the pieces marked with `TODO`.

It has two parts:

1. **Module-level constants**: fixed, field-independent design data specific to the primitive.
2. **The `<Name>Params` class**: the constructor, the two sanitization methods (`_input_sanitization` before construction, `_parameter_sanitization` after), a couple of field-conversion helpers, and a set of `_init_*` helpers that derive the missing values. Three of the `_init_*` helpers have **canonical names shared by every primitive**: `_init_rounds` (round numbers), `_init_cons` (round constants and any other derived constants), and `_init_mat` (matrix generation).

### 1.1 Module-level constants

Put here anything that is a fixed *design choice* baked into the primitive and is independent of the chosen field (e.g., magic seed constants). Anything that can by *provided by* the user's parameters or can be *computed from* them does **not** belong here; in the latter case, it is derived inside the class instead.

In the `MyPrimitiveParams` template this section is empty except for a comment and a commented-out example (`PI_0`, a long run of pi digits that a primitive might use to seed its round constants). Replace it with whatever fixed, field-independent data your construction needs, and give each entry a short comment saying where it comes from and what it is used for.

Pinned design data (published matrices, lookup tables, circulant rows, ...) always lives here — as a module-level constant, typically a dict keyed by the state size `t` or the decomposition radix — never inline behind `if t == ...` branches inside an `_init_*` helper (which then just does a lookup and falls through to the generic construction or raises). Precedents: `polocolo.MDS`, `anemoi.CIRCULANT_MDS_ROWS`, `monolith.MONOLITH_LUTS`, `marvellous.RPO_MDS_ROWS`. When a primitive reuses another design's table (Skyscraper reuses Monolith's Bar LUT, Tip4' reuses RPO's MDS rows), it imports that primitive's constant instead of duplicating the data; only *generic* builders (circulant, Cauchy/Vandermonde constructions, the chi-landscape machinery, ...) belong in `utils/`.

### 1.2 The `<Name>Params` constructor

The constructor takes the user-facing parameters, runs them through input sanitization, and then fills in every unspecified value. Its first action is to hand the complete argument list to `_input_sanitization` (see 1.3); after that it populates the instance in labelled blocks:

- **Non-linear layer**: values tied to the S-box. `alpha` defaults to the smallest valid exponent, and `alpha_inv = alpha^{-1} mod (p-1)` is derived.
- **Linear layer**: the matrix `M`, provided or built by `_init_mat()`, lifted to
  field elements via `map_to_field`, with its inverse `M_inv` precomputed.
- **Round constants**: `rcons`, provided or built by `_init_cons()`, also
  lifted into the field via `map_to_field`.
- **Hash modes**: the per-mode parameter dicts `sponge` (rate/capacity/digest `r, c, d`) and `comp` (compression params) that `hash.py` reads — each either a dict or `None` when that mode is not defined for the instance. They are stored verbatim (`self.sponge, self.comp = sponge, comp`); the mode objects are built later, in `hash.py`.
- **Round number**: `R` is taken as given, or derived via `_init_rounds()`.

> **A note on block order.** Input sanitization must come first — every other block assumes the arguments are already valid. The remaining blocks, however, are *not* in a fixed order: arrange them to respect your primitive's dependencies. Two kinds of ordering constraints commonly appear.
>
> *Data dependencies.* A derivation may consume values produced by an earlier block. Round-number derivation, for example, often depends on the S-box exponent (`_init_rounds` needs `self.alpha`), so the non-linear block must run before the round-number block; if `_init_cons` needs `R` and `alpha`, it must run after both. Set each attribute before the block that reads it.
>
> *Reproducibility.* When the matrix and/or the round constants are drawn from a single deterministic stream (e.g. a `FieldElementSampler` / XOF seeded once) the *order in which you query that stream* fixes the output. To reproduce a reference implementation's test vectors you must query it in exactly the same order (say, all matrix entries before any round constant, or interleaved per round). Reordering the blocks then silently changes the generated values even though each block's own logic is untouched.

For `MyPrimitiveParams` the user-provided parameters are:

| Parameter | Required | Meaning |
|---|---|---|
| `p` | yes | Field characteristic (a prime). Defines `F = GF(p)`. |
| `t` | yes | State size (number of field elements). |
| `sponge` | no | Sponge params as a dict `dict(r, c, d)` (rate, capacity, digest), or `None` for no sponge mode. |
| `comp` | no | Compression params as a dict (digest `d` and/or arity `a`), or `None` for no compression mode. |
| `alpha` | no | S-box exponent, coprime with `p-1`. Smallest valid exponent if omitted. |
| `R` | no | Number of rounds. Derived from the attack complexity if omitted. |
| `M` | no | `t x t` MDS matrix for the linear layer. Generated by `_init_mat()` if omitted. |
| `rcons` | no | Round constants. Generated by `_init_cons()` if omitted. |
| `kappa` | no | Target security level in bits (default `128`). |

Matrices are generally accepted as `list[list[int]]` and round constants as `list[int]` (or similar); both are converted to field elements with `map_to_field` so the rest of the framework only ever sees field elements. Taking plain integers at the boundary keeps instance generation independent of any finite-field logic (the same matrix or constant table can be written down, diffed, and reused across fields) and makes the raw parameters easy to inspect, log, and compare against a reference, with `map_to_field` performing the single, explicit lift into `F`.


### 1.3 Input and parameter sanitization

Sanitization happens twice, bracketing the constructor: `_input_sanitization` validates the *raw arguments* before anything is stored, and `_parameter_sanitization` validates the *fully-constructed object* (stored and derived values) as the constructor's last step.

The constructor's first step delegates to `_input_sanitization`, which receives a single struct holding every constructor argument, accessed by attribute (`params.p`, `params.t`, `params.alpha`, ...). Bundling the arguments this way means the method's signature never has to change as you add checks. It performs two kinds of checks.

*Hard checks* are invariants that must always hold for the construction to make sense, and they `raise`. The template rejects characteristic 2, a non-positive state size, and an explicitly supplied `alpha` that is not coprime with `p-1` (which would stop the power map from being a permutation). Add whatever else your design genuinely requires.

*Warnings* flag settings that depart from the recommended ("official") parameters but that the primitive can still run with (e.g., a very small field, which is fine for a toy instance used in cryptanalysis but is not secure / not analyzed for real use). The template emits every such warning under a dedicated category, `ParamRecommendationWarning` (a subclass of `UserWarning`), rather than as a plain warning:

```python
class ParamRecommendationWarning(UserWarning):
    """A parameter deviates from the recommended ("official") settings."""

# at the warning site, e.g. the toy-field check:
warnings.warn(msg, ParamRecommendationWarning, stacklevel=2)
```

Going through `warnings.warn(...)` means the instance is **still built**; only the hard checks stop construction. Routing them through one category (instead of the default `UserWarning`) is what lets a user switch off exactly these recommendation warnings without silencing anything else (see below).

Be deliberate about which bucket each check goes in: raise only where correctness demands it, and prefer a warning wherever an analyst might legitimately want to override the recommendation.

**Disabling the warnings (as a user).** Because the recommendation warnings have their own category, a user can switch off exactly those (and nothing else) without touching the library:

```python
import warnings
from recommendations import ParamRecommendationWarning
warnings.filterwarnings("ignore", category=ParamRecommendationWarning)
```

To suppress them only around a single construction rather than for the whole session, scope it with a context manager:

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ParamRecommendationWarning)
    params = MyPrimitiveParams(...)
```

Note that by default Python shows each unique warning only once per call site, so looping over many toy instances won't spam the console unless `simplefilter("always")` is set.

**Parameter sanitization.** After every value has been stored or derived, the constructor ends with `self._parameter_sanitization()`. Where `_input_sanitization` can only see the raw arguments (some of which may be `None`), this method sees the finished object, so it checks the *structural invariants of the stored values*: the matrix is `t x t`, `rcons` has one row per round of the right width, a derived exponent still defines a permutation, and so on. It uses the same two buckets as `_input_sanitization`: hard checks `raise`, deviations from the recommended settings go through `warnings.warn(..., ParamRecommendationWarning)`. Checks on user *input* belong in `_input_sanitization`; checks on *derived or stored* values belong here — inline validation sprinkled through the constructor body belongs in neither.

### 1.4 Field-conversion helpers

`to_field(n)` lifts an `int` into `F = GF(p)`, and `from_field(el)` brings a field element back to an `Integer`. Every primitive shares these two helpers so that test vectors, round constants, and matrices move between the integer and field representations the same way across the whole framework.

### 1.5 The `_init_*` derivation helpers

This is where the actual design logic lives: each helper supplies the default for one parameter when the user did not pass it. These are the most important methods to get right, because they encode the security argument. The template ships one working helper and three stubs:

- **`_init_alpha`** *(implemented)*: picks the smallest exponent `>= 3` that is coprime with `p-1`, i.e. an invertible S-box exponent.
- **`_init_rounds`** *(stub)*: derive the round count needed to resist the relevant attacks (algebraic / Groebner-basis, differential, ...). This is the security core of the primitive; document the bound you use.
- **`_init_mat`** *(stub)*: return a `t x t` MDS matrix over `F`. If your construction is generic and reusable (e.g. a Cauchy or circulant search), search for it in `utils/matrix.py` or implement it there if not present.
- **`_init_cons`** *(stub)*: derive the round constants (from digits of pi, a fixed seed, a counter, ...). For reproducible "random" constants, many primitives use a `FieldElementSampler` from `utils/sampler.py` (recall the query-order caveat in 1.2).

The pattern to mirror for any new primitive: one `_init_<thing>` per parameter that has a non-trivial default, with the constructor choosing whether to call it based on whether the user supplied a value. Keep the three canonical names (`_init_rounds`, `_init_cons`, `_init_mat`) even when your primitive extends them (e.g. `_init_cons` may derive coefficient tables alongside the round constants from one XOF stream, and a family base class may split the matrix into `_init_mat_ext` / `_init_mat_int`).

**Stub idiom.** A helper whose derivation is not (yet) worked out must still exist, as a stub, so the contract is uniform and the gap is visible:

```python
def _init_rounds(self) -> int:
    """Derive round numbers to resist known attacks.
    TODO: replace with your primitive's round number derivation strategy."""
    raise NotImplementedError("Error: Not implemented -- round number derivation for MyPrimitive")
```

A docstring with a `TODO:` line saying what should happen (citing the paper's criterion where possible), plus a `NotImplementedError` whose message starts with `Error: Not implemented`. For primitives whose spec fixes a value outright (e.g. a fixed round count), the helper simply returns that value with a docstring noting it is spec-fixed — that is an implementation, not a stub.

---

## 2. `instances.py`

Where `params.py` defines *how* to build an instance, `instances.py` pins down *particular* ones: a registry of concrete, named `<Name>Params` objects over specific fields. Give each a stable module-level name so tests, benchmarks, and applications import the exact same parameters by name rather than re-typing them. Keep officially supported parameter sets here, i.e., the ones whose parameters have actually been verified (e.g., the matrix is confirmed MDS, the round number meets the security bound for the target kappa, the S-box exponent is a valid permutation, and so on). Do **not** place toy or experimental sets alongside them unannotated. Sometimes an instance that fails to meet the security requirements is still useful (for benchmarking, regression tests, or cryptanalysis) but it must be clearly marked as such (e.g. a TOY_/INSECURE_ name prefix and a comment stating why it is not secure) so it can never be mistaken for a recommended set.

### 2.1 Fields come from `utils/field.py`

`utils/field.py` holds a large collection of predefined finite fields, each a `Field(...)` that bundles the field's defining data in one place: its `name`, characteristic `p` (with extension degree `n` and bit size `bits`), the factorization of `p-1` (`factors`), a multiplicative `generator`, and a recommended S-box exponent `alpha` together with its inverse `alpha_inv`. For example, `GOLDILOCKS` is `p = 2^64 - 2^32 + 1`.

Import the field you need and reference its attributes (`GOLDILOCKS.p`, `GOLDILOCKS.alpha`, ...) instead of hard-coding numbers. **If your target field is not already in `utils/field.py`, add a new `Field` entry there** rather than inlining the constants in your instance. That keeps field data in one shared place, reused across every primitive, and means values like `alpha_inv` are computed once.

### 2.2 Adding a new field

If your target field is not in `utils/field.py`, add a `Field` entry for it. Each
field is described by:

| Field | Meaning | How to obtain it |
|---|---|---|
| `name` | Human-readable label used in instance names and messages. | Your choice. |
| `p` | The prime characteristic. | Given by your design; write the closed form in a comment. |
| `n` | Extension degree (`1` for a prime field `GF(p)`). | `1` unless you are over an extension field. |
| `bits` | Bit size of the field. | `p.bit_length()` for a prime field. |
| `factors` | Prime factorization of `p-1`. | `dict(factor(p - 1))` in Sage. |
| `generator` | A multiplicative generator of `GF(p)*` (order `p-1`). | `GF(p).multiplicative_generator()` (or `primitive_root(p)`). |
| `alpha` | Recommended S-box exponent, coprime with `p-1`. | Smallest `>= 3` with `gcd(alpha, p-1) == 1`, or a chosen efficient one. |
| `alpha_inv` | `alpha^{-1} mod (p-1)`. | `pow(alpha, -1, p - 1)`. |


### 2.3 Defining an instance

Instantiate `<Name>Params` with concrete values, drawing the field-derived ones
from the imported `Field`:

```python
from utils.field import GOLDILOCKS
from myprimitive.params import MyPrimitiveParams

MYPRIMITIVE_GOLDILOCKS_T2 = MyPrimitiveParams(
    p=GOLDILOCKS.p,                   # field characteristic, taken from the Field entry
    t=3,                              # state size
    alpha=GOLDILOCKS.alpha,           # S-box exponent recommended for this field
    R=5,                              # number of rounds (omit to derive via _init_rounds)
    rcons=[1, 2, 3, 4, 5],            # round constants (omit to derive via _init_cons)
    M=[[1, 2, 3, 4], [5, 6, 7, 8]],   # MDS matrix (omit to derive via _init_mat)
    sponge=dict(r=2, c=1, d=1),       # sponge params: rate / capacity (r + c == t) / digest; None for no sponge
    comp=None,                        # compression params dict (e.g. dict(a=2) / dict(d=1)); None for no compression
)
```

Each instance declares its modes of operation as per-mode dicts: `sponge=dict(r, c, d)` and `comp=dict(...)`, either being `None` when the instance does not define that mode. Reference parameter sets typically specify only the sponge, so most instances are sponge-only (`comp=None`); set `comp` only for a primitive that is itself a compression function. Use the `<PRIMITIVE>_<FIELD>_<VARIANT>` naming convention (`MYPRIMITIVE_GOLDILOCKS_T2`) so instances are unambiguous and easy to find. Remember that any value you omit is filled in by the corresponding `_init_*` helper. Non-recommended instances will trip the `ParamRecommendationWarning` from `_input_sanitization`, which is the expected signal that they are toy/experimental rather than secure.

---

## 3. `hash.py`

`hash.py` defines the `MyPrimitivePerm` class — the round function (the permutation) — together with the mode classes built on top of it (`MyPrimitiveHash`, and `MyPrimitiveCompress` where applicable; see 3.5). `MyPrimitivePerm` is a **pure consumer** of a `MyPrimitiveParams` instance: its `__init__` calls `super().__init__(params)` (which copies the instance-global fields `F`, `to_field`, `from_field`, `t`, `p`, `kappa`, `toy`) and then copies its own already-derived matrices/constants/rounds out of `params`; nothing is computed or validated here. The guiding principle is *fidelity to the specification over efficiency*: the code should read like the paper so it can be checked against it, and its components should be runnable symbolically for cryptanalysis (see 3.6), not just evaluated. Speed is explicitly a non-goal.

### 3.1 Component layers and the round index

Each cryptographic step is a *component layer* - one operation applied to the whole state - with the uniform signature `(self, state, r) -> state`, and a matching `_inv` partner that undoes it for the same round `r`. The template ships `constant_addition`, `linear_layer`, and `nonlinear_layer` (plus inverses).

**Every layer takes the round index `r`, even layers that do not use it.** For a primitive whose layers never vary by round this looks redundant. It is kept anyway so that any layer can *become* round-dependent without changing a single call site, which gives every SPN-based scheme in the framework one common interface. A round-dependent layer simply branches on `r`. 

> **Example.** In the template, `nonlinear_layer` applies `x**alpha` on even rounds and `x**alpha_inv` on odd ones. 

Where the round-dependence is data rather than control flow, prefer indexing a per-round table (as `constant_addition` does with `self.rcons[r]`) over a branch.

### 3.2 Layer ordering vs. layer behaviour

These are two different, orthogonal notions of "round-dependence", and the framework keeps them in two different places. Keep them separate when you adapt the template:

- **What a layer does in round `r` (behaviour)** lives *inside the layer*, keyed by `r` (e.g., which exponent `nonlinear_layer` uses).
- **In which order the layers run in round `r` (ordering)** lives *in the permutation loop* (e.g., `ARK -> S -> M`).

So a layer never decides where it sits in the round, and the loop never reaches
inside a layer to change what it computes. 

> **Example.** In the template both behaviour and ordering vary by parity to illustrate the distinction: the loop applies `linear -> nonlinear -> constant_addition` on even rounds and the reverse on odd rounds (ordering), while `nonlinear_layer` independently switches its exponent by parity (behaviour). 

Most primitives fix one order for all rounds and vary only behaviour, but the separation is what lets either change without disturbing the other.

### 3.3 Pre-/post-round steps

`_pre_rounds` / `_post_rounds` are transformations applied once before the first round and once after the last (e.g. an initial linear map). Because they are not per-round, they take no round index. Each must be the mutual inverse of its `_inv` counterpart.

### 3.4 The permutation and its inverse

`permute` checks the state size, applies `_pre_rounds`, runs the round loop (which fixes the per-round ordering, per 3.2), and applies `_post_rounds`.

`permute_inv` runs the whole thing backwards, and getting this reversal right is the easy thing to get subtly wrong, so follow the rule mechanically: undo `_post_rounds` first, iterate the rounds from `R-1` down to `0`, and **within each round apply the inverse layers in the reverse of the forward order** (the last layer applied is the first undone), then undo `_pre_rounds`. Always test it with a round-trip assertion (`permute_inv(permute(x)) == x` for random `x` over a few instances, see 4) because an ordering slip there fails silently.

### 3.5 Hash modes

The permutation and the modes built on it are **separate classes**, split by composition (see `utils/primitive.py`). `hash.py` therefore defines:

- `MyPrimitivePerm(Permutation)` — the bare permutation from 3.1–3.4, exposing `permute` / `permute_inv` and nothing else.
- `MyPrimitiveHash(HashFunction)` — a permutation **plus a sponge**, exposing `hash`. It pins only its sponge *kind* via a class attribute, `SPONGE_KIND = "..."`; the sizes come from the instance's `params.sponge` dict.
- `MyPrimitiveCompress(CompressionFunction)` — a permutation **plus a compression**, exposing `compress`, defined only if the primitive has a feed-forward compression. It pins `COMP_KIND = "..."` (`"trunc"` for a truncation/Davies-Meyer feed-forward, `"jive"` for Jive); the params come from `params.comp`. A 2-to-1 node realized *through the sponge* instead needs no `Compress` class — call `H.sponge.compress(perm, state)` directly (as Reinforced Concrete does).

A mode class is a hash/compression, **not** a permutation — it does not expose `permute`; the wrapped permutation is reachable as `.permutation`. Each is constructed from a permutation instance and that mode's parameter dict:

```python
P = MyPrimitivePerm(params)
H = MyPrimitiveHash(P, params.sponge)      # raises if params.sponge is None
C = MyPrimitiveCompress(P, params.comp)    # raises if params.comp is None
H.hash(data)                               # sponge hash -> d elements
C.compress(state)                          # compression on the full t-element state -> d
```

The class pins only the *kind*; all the actual mode logic (the sponge and compression constructions with their padding rules) lives in `utils/mode.py`, shared across all primitives, and is built by the `make_sponge` / `make_compression` factories the two base classes call. An instance may override the pinned kind with a `"kind"` key in its mode dict, and mark a mode as toy (below the security floors, so violations only warn) with `toy=True` in the dict. If the mode your scheme needs is not yet in `utils/mode.py` but is generic, implement it there — add a `Sponge` / `Compression` subclass and register its kind — so other primitives can reuse it; reserve primitive-specific mode code for this file only when it genuinely cannot be generalized. A primitive whose Merkle 2-to-1 is just its sponge over the concatenated children (e.g. Rescue, Reinforced Concrete) defines no `Compress` class at all — it calls `H.hash(x1 + x2)`.

### 3.6 Running components over polynomials (cryptanalysis)

A central reason the implementation favours spec-fidelity over speed is that the *same* code must be usable for algebraic cryptanalysis, not only for evaluation. This works as long as every component is written **generically**: it uses only the ring operations `+`, `-`, `*`, `**` and never inspects or branches on the *value* of a state element. (Branching on the round index `r` is fine since `r` is a known Python integer, not field data.)

Because field elements and polynomials over the field support the same ring operations, a generic component runs unchanged on either. To obtain the permutation output as polynomials in the inputs (e.g., to measure degree growth or to build the polynomial system for a Gröbner-basis attack) instantiate a polynomial ring over `F` and feed its generators in as the state:

```python
from sage.all import PolynomialRing
prim = MyPrimitivePerm(params)
R = PolynomialRing(prim.F, 'x', prim.t)
xs = list(R.gens())            # symbolic state [x0, x1, ...]
out = prim.permute(xs)         # list of polynomials in x0..x_{t-1}
```

Two rules keep this working when you implement your own layers:

- **Never coerce inside a component.** `to_field` / `from_field` are for the I/O boundary only; calling them on intermediate values would force a symbolic input back into `F` and defeat the purpose.
- **Express round-dependence as data or as `r`-branching, never as value-branching.** A test like `if x == 0` changes the function for symbolic inputs and usually errors on a polynomial.

One practical caveat concerns the inverse direction: an inverse S-box exponent such as `alpha_inv` is astronomically large, so `x ** alpha_inv` on a polynomial is not something you evaluate symbolically. Algebraic models instead keep the low-degree *forward* relation by introducing a fresh variable `y` constrained by `y**alpha - x == 0` rather than computing `x**alpha_inv`. The generic forward components are exactly what let you build those relations.

> Note that this generic approach only covers components expressible purely through ring operations. It does **not** apply to lookup-based constructions (S-boxes or other steps defined by a table indexed by the input value). A table lookup inherently branches on the *value* of a state element, which has no meaning for a polynomial, so such a component cannot be run symbolically this way; modelling it algebraically requires a separate, construction-specific treatment (e.g. an interpolating polynomial for the table, or dedicated lookup constraints) rather than the same evaluation code. The **Skyscraper** module is the reference example for handling lookup-based (and Feistel) constructions.

---

## 4. Tests

Every primitive ships a `pytest` suite that runs the same battery of checks across all of its named instances. Parametrize over the `instances.py` registry so each recommended instance is exercised identically and a failure names the exact instance:

```python
INSTANCES = [
    ("GOLDILOCKS_T2", MYPRIMITIVE_GOLDILOCKS_T2),
    # ... one entry per named instance
]
# @pytest.mark.parametrize("name,params", INSTANCES, ids=[n for n, _ in INSTANCES])
```

The suite has five groups.

### 4.1 Known-answer tests (KATs)

KATs pin the exact output of the permutation (and each hash mode) on fixed inputs, so any unintended change (e.g., to the matrix, the constants, the round count, or the layer ordering) is caught immediately.

**Format.** Store vectors as a dict keyed by instance name; each entry is a list of `{"input": [...], "output": [...]}` cases where **inputs and outputs are plain integers**, not field elements:

```python
KATS = {
    "GOLDILOCKS_T2": [
        {"input": [0, 1, 2], "output": [<int>, <int>, <int>]},
    ],
}
```

Plain integers keep the vectors field-agnostic, easy to read, and stable to store; the test lifts inputs through `to_field`, runs the permutation, and lowers the result through `from_field` before comparing:

```python
def test_permutation_kat(name, params, kat):
    f = MyPrimitivePerm(params)
    out = f.permute([f.to_field(x) for x in kat["input"]])
    assert [f.from_field(x) for x in out] == kat["output"]
```

Use two sources of vectors, ideally both:
- *Self-derived*: take, for example,  `input = [0, 1, ..., t-1]` or `input = [0, 0, ..., 0]`; because the constants and matrix are deterministically derived inside `params` from `(p, t, R, alpha)`, the output is fully reproducible. Generate it by running the permutation once and pasting the result. This guards against regressions. 
- *Reference*: vectors from the paper or a reference implementation (if you are not the author); these guard against a wrong-but-self-consistent implementation, which self-derived vectors cannot catch.

Hash-mode KATs use the same dict shape but build the mode object and call it: `H = MyPrimitiveHash(MyPrimitivePerm(params), params.sponge); H.hash(...)` (and `C = MyPrimitiveCompress(...); C.compress(...)`). For the sponge, choose inputs that exercise the padding rules: empty input, a length that is not a multiple of the rate `r`, and exactly one and two full blocks. The sponge output has length `d` (available as `H.sponge.d`).

### 4.2 Roundtrip (invertibility) tests

The permutation is a bijection, so `permute_inv ∘ permute` is the identity:

```python
def test_permutation_roundtrip(name, params):
    f = MyPrimitivePerm(params)
    inp = [f.F.random_element() for _ in range(f.t)]
    assert f.permute_inv(f.permute(inp)) == inp
```

Worth adding at finer granularity, because they localize a failure: each layer against its inverse (`layer_inv(layer(x, r), r) == x` for every component over a few rounds `r`) and the pre/post steps (`_pre_rounds_inv(_pre_rounds(x)) == x`). Hash modes are not invertible, so there is no roundtrip there.

### 4.3 Consistency tests

Cheap invariants that catch gross breakage: 
- **Determinism**: the same input twice gives the same output.
- **Injectivity** of `permute` and `permute_inv`: distinct inputs → distinct outputs (a permutation is injective, so two different inputs must not collide, which catches a degenerate/collapsing map)
- **Output sizes**: `permute` returns `t` elements, `H.hash(...)` returns `d`, etc.

### 4.4 Algebraic tests

These check that components match their mathematical definition and behave correctly as polynomials. 
- **Component identities**: pin a layer to its formula in a setting where it simplifies, e.g., with `R = 1` and a zero round constant, the affine/linear layer must equal the bare `matvecmul(M, x)` (as in the example's `test_affine`, which also sweeps several fields, state sizes, and exponents).
- **Symbolic degree**: run the permutation on the generators of `PolynomialRing(F, 'x', t)` (per 3.6) and check the output's total degree matches the expected growth (on the order of `alpha**R`); this suggests the components are generic and pins the algebraic complexity the security analysis relies on.
- **Satisfiability**: build a polynomial system used for algebraic attacks, then substitute a concrete input/output pair (together with the actual intermediate values from running the permutation on that input), and check that every equation vanishes. This confirms the algebraic model agrees with the concrete evaluation, which the whole Gröbner-basis analysis depends on.
- **Derivation vs. pinned instance** (`test_generated_matches_instance`): rebuild the params *without* passing `M` / `rcons` (and, where derivable, `R`) and check the `_init_*` helpers reproduce exactly the values stored in `instances.py`. This ties the generation code to the published constants.

**Deactivated tests.** When a check cannot run yet because the feature it exercises is a stub (e.g. `test_generated_matches_instance` while `_init_mat` raises `NotImplementedError`), still write the test in full and deactivate it with `@pytest.mark.skip(reason="...")` naming the blocker. Never comment tests out: a skip shows up in every pytest run (`-rs` lists the reasons), so the gap stays visible until the stub is implemented.

### 4.5 Anything else worth adding

- **Validation / error tests:** a wrong-sized state to `permute` raises `ValueError`; the hard params checks raise (e.g. an `alpha` not coprime with `p-1`); building a mode whose params dict is `None` (`MyPrimitiveHash(P, None)`) raises.
- **Warning tests:** a toy field raises `ParamRecommendationWarning` (`with pytest.warns(ParamRecommendationWarning): ...`), and a valid recommended instance raises none.
- **Reproducibility:** constructing the same params twice yields identical `rcons` and `M`; if constants come from a seeded sampler, a fixed seed
  reproduces them.
- **Mode toy flag:** a mode below its security floor raises when its dict has no `toy`, and only warns with `toy=True` in the dict — independent of the permutation's own `toy` (see `test_mode_toy_is_its_own_flag`).
- **Field boundary:** `from_field(to_field(n)) == n % p`.
- **Edge inputs:** all-zero state, all-equal elements, and the field's additive / multiplicative identities.

A handy way to bootstrap the self-derived KATs and the symbolic-degree expectations is a small generator script that builds each instance, runs the permutation once on `[0, ..., t-1]`, and prints the vectors and degree. Then paste its output into the suite.
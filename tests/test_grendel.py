# test_grendel.py
# ---------------------------------------------------------------------------
# Test suite for Grendel, parametrized over the named instances in instances.py.
#
# Groups:
#   4.1 KATs         -- self-derived vectors (permutation + sponge; the paper
#                       ships no reference vectors)
#   4.2 Roundtrip    -- permutation_inv undoes permutation (and per-layer)
#   4.3 Consistency  -- determinism, distinct inputs -> distinct outputs, sizes
#   4.4 Derivation   -- alpha/R derivation, round constants vs. a raw SHAKE256
#                       reimplementation of Algorithm 5, MDS matrix vs. a direct
#                       Sage computation of Algorithm 4, Legendre symbol vs. Sage
#   4.5 Algebraic    -- the known-Legendre-symbol round relations (Eq. 33) vanish
#                       on a concrete permutation trace
#   4.6 Misc         -- validation/errors, warnings, reproducibility
# ---------------------------------------------------------------------------

import warnings
from hashlib import shake_256
from math import ceil, log2

import pytest
from sage.all import GF, matrix, legendre_symbol

from recommendations import ParamRecommendationWarning
from grendel.hash import Grendel
from grendel.params import GrendelParams
from grendel.instances import TOY_GRENDEL_65519_T2, TOY_GRENDEL_65393_T2
from utils.field import BLS12_381_SCALAR
from utils.matrix import is_mds

INSTANCES = [
    ("65519", TOY_GRENDEL_65519_T2),
    ("65393", TOY_GRENDEL_65393_T2),
]
IDS = [name for name, _ in INSTANCES]

# Self-derived known-answer vectors: generated once by running this
# implementation (inputs and outputs are plain integers).
SELF_KATS = {
    "65519": {
        "permutation": [
            {"input": [0, 1],     "output": [9355, 53255]},
            {"input": [0, 0],     "output": [64103, 21183]},
            {"input": [123, 456], "output": [43243, 6845]},
        ],
        # rate r = 1, so these inputs exercise the pad_one rule (the appended 1
        # always adds a block) and multi-block absorption
        "sponge": [
            {"input": [],        "output": [61024]},
            {"input": [5],       "output": [22319]},
            {"input": [5, 6],    "output": [58234]},
            {"input": [5, 6, 7], "output": [58286]},
        ],
    },
    "65393": {
        "permutation": [
            {"input": [0, 1],     "output": [27840, 51872]},
            {"input": [0, 0],     "output": [64948, 9360]},
            {"input": [123, 456], "output": [45184, 43939]},
        ],
        "sponge": [
            {"input": [],        "output": [38104]},
            {"input": [5],       "output": [5875]},
            {"input": [5, 6],    "output": [58189]},
            {"input": [5, 6, 7], "output": [17470]},
        ],
    },
}

# Expected derived parameters: alpha per Sec. 4.1 (2 iff p = 3 mod 4), the
# smallest primitive element g, and the round number R of the Sec. 5.7 rule at
# kappa = 16 (verified against an independent evaluation of the Table 1 bounds).
EXPECTED_DERIVATION = {
    "65519": {"alpha": 2, "g": 11, "R": 7},
    "65393": {"alpha": 3, "g": 3, "R": 5},
}


# ---------------------------------------------------------------------------
# 4.1 Known-answer tests
# ---------------------------------------------------------------------------

PERMUTATION_KAT_CASES = [(name, params, kat) for name, params in INSTANCES
                         for kat in SELF_KATS[name]["permutation"]]
PERMUTATION_KAT_IDS = [f"{name}-{i}" for name, _ in INSTANCES
                       for i, _ in enumerate(SELF_KATS[name]["permutation"])]


@pytest.mark.parametrize("name,params,kat", PERMUTATION_KAT_CASES, ids=PERMUTATION_KAT_IDS)
def test_permutation_kat_self(name, params, kat):
    prim = Grendel(params)
    out = prim.permutation([prim.to_field(x) for x in kat["input"]])
    assert [prim.from_field(x) for x in out] == kat["output"]


SPONGE_KAT_CASES = [(name, params, kat) for name, params in INSTANCES
                    for kat in SELF_KATS[name]["sponge"]]
SPONGE_KAT_IDS = [f"{name}-{i}" for name, _ in INSTANCES
                  for i, _ in enumerate(SELF_KATS[name]["sponge"])]


@pytest.mark.parametrize("name,params,kat", SPONGE_KAT_CASES, ids=SPONGE_KAT_IDS)
def test_sponge_kat_self(name, params, kat):
    prim = Grendel(params)
    out = prim.hash_sponge([prim.to_field(x) for x in kat["input"]])
    assert [prim.from_field(x) for x in out] == kat["output"]


# ---------------------------------------------------------------------------
# 4.2 Roundtrip (invertibility)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_roundtrip(name, params):
    prim = Grendel(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation_inv(prim.permutation(inp)) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_layer_roundtrip(name, params):
    prim = Grendel(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    for r in range(prim.R):
        assert prim.nonlinear_layer_inv(prim.nonlinear_layer(inp, r), r) == inp
        assert prim.linear_layer_inv(prim.linear_layer(inp, r), r) == inp
        assert prim.constant_addition_inv(prim.constant_addition(inp, r), r) == inp


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_sbox_roundtrip_and_zero(name, params):
    prim = Grendel(params)
    for x in [prim.F.zero(), prim.F.one(), prim.to_field(-1), prim.F.random_element()]:
        assert prim._sbox_inv(prim._sbox(x)) == x
    assert prim._sbox(prim.F.zero()) == prim.F.zero()   # f(0) = 0 (legendre(0) = 0)


# ---------------------------------------------------------------------------
# 4.3 Consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_deterministic(name, params):
    prim = Grendel(params)
    inp = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp) == prim.permutation(inp)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_permutation_distinct_inputs(name, params):
    prim = Grendel(params)
    inp1 = [prim.F.random_element() for _ in range(prim.t)]
    inp2 = [prim.F.random_element() for _ in range(prim.t)]
    while inp1 == inp2:
        inp2 = [prim.F.random_element() for _ in range(prim.t)]
    assert prim.permutation(inp1) != prim.permutation(inp2)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_output_sizes(name, params):
    prim = Grendel(params)
    assert len(prim.permutation([prim.F.zero()] * prim.t)) == prim.t
    assert len(prim.hash_sponge([prim.F.random_element() for _ in range(prim.r)])) == prim.d


# ---------------------------------------------------------------------------
# 4.4 Derivation and constants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_parameter_derivation(name, params):
    expected = EXPECTED_DERIVATION[name]
    assert params.alpha == expected["alpha"]
    assert params.g == expected["g"]
    assert params.R == expected["R"]
    # the S-box exponent e = alpha + (p-1)/2 must define a permutation
    assert params.e == params.alpha + (params.p - 1) // 2
    assert (params.e * params.e_inv) % (params.p - 1) == 1


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_round_constants_reproduce_algorithm5(name, params):
    # Independent, literal reimplementation of Algorithm 5: expand the seed
    # phrase with SHAKE256 to R*t chunks of w = 1 + ceil(log2(p)/8) bytes, parse
    # each chunk most-significant-byte-first, and reduce mod p.
    p, t, R = params.p, params.t, params.R
    w = 1 + ceil(int(p).bit_length() / 8)
    seed = f"grendel-{p}-{t}-{params.kappa}".encode()
    buffer = shake_256(seed).digest(R * t * w)
    constants = [int.from_bytes(buffer[i * w:(i + 1) * w], "big") % p for i in range(R * t)]
    assert [[params.from_field(x) for x in row] for row in params.rcons] \
        == [constants[i * t:(i + 1) * t] for i in range(R)]


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_mds_matrix_reproduces_algorithm4(name, params):
    # Independent, literal reimplementation of Algorithm 4: row-reduce the
    # t x 2t Reed-Solomon generator matrix G[i][j] = g^(i*j) (g the smallest
    # primitive element) to systematic form (I | M^T) and transpose the right half.
    F = GF(params.p)
    g = next(g for g in range(2, params.p) if F(g).multiplicative_order() == params.p - 1)
    assert g == params.g
    G = matrix(F, [[F(g) ** (i * j) for j in range(2 * params.t)] for i in range(params.t)])
    M = G.rref()[:, params.t:].transpose()
    assert [list(row) for row in M] == params.M
    assert is_mds(params.M)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_legendre_agrees_with_sage(name, params):
    prim = Grendel(params)
    for x in [0, 1, 2, 3, 5, 7, 1234, params.p - 1]:
        assert prim.from_field(prim._legendre(prim.to_field(x))) % params.p \
            == legendre_symbol(x, params.p) % params.p


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_sbox_closed_form(name, params):
    # x^alpha * legendre(x) equals the single power map x^(alpha + (p-1)/2)
    prim = Grendel(params)
    for _ in range(10):
        x = prim.F.random_element()
        assert prim._sbox(x) == x ** prim.e


# ---------------------------------------------------------------------------
# 4.5 Algebraic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_known_legendre_model_satisfiability(name, params):
    # The known-Legendre-symbol round relations (Eq. 33): with S_{ti+j} the
    # Legendre symbol of round i's input element j, every round satisfies
    #     sum_j M[k][j] * x_{ti+j}^alpha * S_{ti+j} + C_{ti+k} - x_{t(i+1)+k} = 0.
    # Substituting the intermediate states of a concrete permutation run must
    # make every equation vanish, confirming the algebraic model matches the
    # evaluation code.
    prim = Grendel(params)
    states = [[prim.F.random_element() for _ in range(prim.t)]]
    for r in range(prim.R):
        state = prim.constant_addition(prim.linear_layer(prim.nonlinear_layer(states[-1], r), r), r)
        states.append(state)
    assert states[-1] == prim.permutation(states[0])

    for i in range(prim.R):
        symbols = [prim._legendre(x) for x in states[i]]
        for k in range(prim.t):
            eq = sum(prim.M[k][j] * states[i][j] ** prim.alpha * symbols[j]
                     for j in range(prim.t)) + prim.rcons[i][k] - states[i + 1][k]
            assert eq == 0


# ---------------------------------------------------------------------------
# 4.6 Misc: validation, warnings, reproducibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_invalid_state_size_raises(name, params):
    prim = Grendel(params)
    with pytest.raises(ValueError):
        prim.permutation([prim.F.zero()] * (prim.t + 1))
    with pytest.raises(ValueError):
        prim.permutation_inv([prim.F.zero()] * (prim.t + 1))


def test_invalid_alpha_raises():
    # alpha = 3 over p = 65519 gives e = 3 + (p-1)/2 even, sharing the factor 2
    # with p-1: the S-box would not be a permutation
    with pytest.raises(ValueError):
        GrendelParams(p=65519, t=2, r=1, c=1, d=1, alpha=3, kappa=16)


def test_too_small_state_raises():
    with pytest.raises(ValueError):
        GrendelParams(p=65519, t=1, r=1, c=0, d=1, kappa=16)


def test_toy_instance_warns():
    with pytest.warns(ParamRecommendationWarning):
        GrendelParams(p=65519, t=2, r=1, c=1, d=1, kappa=16, toy=True)


def test_recommended_instance_no_warning():
    # A 255-bit field clears every round-independent bound at kappa = 128:
    # root finding needs p >= 2^160, the digest needs d * log2(p) >= 256.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParamRecommendationWarning)
        GrendelParams(p=BLS12_381_SCALAR.p, t=3, r=2, c=1, d=2, kappa=128)


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_params_reproducible(name, params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ParamRecommendationWarning)
        again = GrendelParams(p=params.p, t=params.t, r=params.r, c=params.c,
                              d=params.d, kappa=params.kappa, toy=params.toy)
    assert again.alpha == params.alpha and again.R == params.R and again.g == params.g
    assert again.M == params.M
    assert again.rcons == params.rcons


@pytest.mark.parametrize("name,params", INSTANCES, ids=IDS)
def test_field_boundary(name, params):
    for n in [0, 1, params.p - 1, params.p, params.p + 5]:
        assert params.from_field(params.to_field(n)) == n % params.p

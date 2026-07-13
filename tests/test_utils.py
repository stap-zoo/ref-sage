# test_utils.py
# ---------------------------------------------------------------------------
# Test suite for the shared utils/ helpers, exercised against real primitive
# instances (Reinforced Concrete and Monolith) rather than synthetic data:
# mixed-radix decompose/compose roundtrips and digit-range checks.
# ---------------------------------------------------------------------------

import pytest
from sage.all import GF, Integer

from utils.lut import mixed_radix_decompose, mixed_radix_compose
from reinforced_concrete.instances import RC_BLS12_T3, RC_BN254_T3, RC_ST_T3
from monolith.instances import (
    MONOLITH_M31_T16,
    MONOLITH_M31_T24,
    MONOLITH_GOLDILOCKS_T8,
    MONOLITH_GOLDILOCKS_T12,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_field_helpers(p):
    F = GF(p)
    return F, lambda el: Integer(el), lambda n: F(n)

# ---------------------------------------------------------------------------
# Roundtrip tests
# ---------------------------------------------------------------------------

RC_INSTANCES = [
    ("BLS12", RC_BLS12_T3),
    ("BN254",  RC_BN254_T3),
    ("ST",     RC_ST_T3),
]

MONOLITH_INSTANCES = [
    ("M31_T16",        MONOLITH_M31_T16),
    ("M31_T24",        MONOLITH_M31_T24),
    ("Goldilocks_T8",  MONOLITH_GOLDILOCKS_T8),
    ("Goldilocks_T12", MONOLITH_GOLDILOCKS_T12),
]


@pytest.mark.parametrize("name,params", RC_INSTANCES)
def test_rc_compose_roundtrip(name, params):
    F, from_field, to_field = _make_field_helpers(params.p)
    for _ in range(5):
        x = F.random_element()
        digits = mixed_radix_decompose(x, params.si, from_field)
        assert mixed_radix_compose(digits, params.si, to_field) == x


@pytest.mark.parametrize("name,params", MONOLITH_INSTANCES)
def test_monolith_compose_roundtrip(name, params):
    F, from_field, to_field = _make_field_helpers(params.p)
    for _ in range(5):
        x = F.random_element()
        digits = mixed_radix_decompose(x, params.si, from_field)
        assert mixed_radix_compose(digits, params.si, to_field) == x


@pytest.mark.parametrize("name,params", RC_INSTANCES + MONOLITH_INSTANCES)
def test_digits_in_range(name, params):
    F, from_field, to_field = _make_field_helpers(params.p)
    for _ in range(5):
        x = F.random_element()
        digits = mixed_radix_decompose(x, params.si, from_field)
        assert len(digits) == len(params.si)
        for d, s in zip(digits, params.si):
            assert 0 <= d < s

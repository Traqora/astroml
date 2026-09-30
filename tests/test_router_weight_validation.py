"""Regression tests for router weight validation (issue #980)."""

import math

import pytest

from astroml.db.schema import validate_router_weights


def test_normalises_weights():
    result = validate_router_weights({"a": 1, "b": 3})
    assert result == {"a": 0.25, "b": 0.75}
    assert math.isclose(sum(result.values()), 1.0)


def test_allows_zero_weight_route():
    assert validate_router_weights({"a": 0, "b": 2.0}) == {"a": 0.0, "b": 1.0}


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"a": 0, "b": 0},
        {"a": -1, "b": 2},
        {"a": float("nan")},
        {"a": float("inf")},
        {"a": "1"},
        {"a": True},
        {"": 1},
        {"  ": 1},
    ],
)
def test_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        validate_router_weights(weights)

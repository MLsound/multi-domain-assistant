"""
Unit tests for MLPRouter.

These tests use random (untrained) weights — they validate structure and
interface, not classification accuracy.
"""

from __future__ import annotations

import pytest

from src.router.mlp_router import MLPRouter


@pytest.fixture(scope="module")
def router() -> MLPRouter:
    """Shared router with random weights (no model path)."""
    return MLPRouter()


def test_route_returns_correct_keys(router: MLPRouter) -> None:
    """Output must contain exactly the three domain keys."""
    result = router.route("What is photovoltaic efficiency?")
    assert set(result.keys()) == {"Software", "User", "Science"}


def test_route_probabilities_sum_to_one(router: MLPRouter) -> None:
    """Softmax output must sum to ~1.0."""
    result = router.route("HEMS thermostat control")
    total = sum(result.values())
    assert abs(total - 1.0) < 1e-5, f"Probabilities sum to {total}, expected 1.0"


def test_route_empty_string_no_crash(router: MLPRouter) -> None:
    """Empty string must not raise; output must still be a valid dict."""
    result = router.route("")
    assert isinstance(result, dict)
    assert len(result) == 3


def test_route_output_types(router: MLPRouter) -> None:
    """All probability values must be Python floats."""
    result = router.route("battery safety NFPA clearance requirements")
    assert all(isinstance(v, float) for v in result.values())


def test_route_probabilities_non_negative(router: MLPRouter) -> None:
    """All probability values must be >= 0."""
    result = router.route("Nest thermostat API command")
    assert all(v >= 0.0 for v in result.values())


def test_route_long_query_no_crash(router: MLPRouter) -> None:
    """A 1 000-character query must not raise."""
    long_query = "photovoltaic efficiency environmental variable " * 22
    result = router.route(long_query)
    assert isinstance(result, dict)
    assert len(result) == 3

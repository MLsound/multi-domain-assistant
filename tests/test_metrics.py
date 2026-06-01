"""
Unit tests for evaluation metrics.

Tests the deterministic tool-call metrics and the error-handling
wrapper around async RAGAS metrics.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.metrics import (
    METRIC_UNAVAILABLE,
    compute_tool_metrics,
    _robust_ascore,
)


class TestComputeToolMetrics:
    """Deterministic tool-call success flags (no LLM judge needed)."""

    def test_all_success(self) -> None:
        result = {
            "dominant_category": "Science",
            "retrieved_chunks": [{"content": "chunk1"}, {"content": "chunk2"}],
            "retrieval_time_ms": 42.5,
            "retry_count": 0,
            "action_result": {"audit_success": True},
        }
        flags = compute_tool_metrics(result)
        assert flags["tool_router_success"] == 1.0
        assert flags["tool_retrieval_success"] == 1.0
        assert flags["tool_critic_success"] == 1.0
        assert flags["tool_action_success"] == 1.0

    def test_router_failure(self) -> None:
        result = {
            "dominant_category": "",
            "retrieved_chunks": [{"content": "chunk1"}],
            "retrieval_time_ms": 10.0,
            "retry_count": 0,
            "action_result": {"audit_success": True},
        }
        flags = compute_tool_metrics(result)
        assert flags["tool_router_success"] == 0.0

    def test_retrieval_failure_empty_chunks(self) -> None:
        result = {
            "dominant_category": "Software",
            "retrieved_chunks": [],
            "retrieval_time_ms": 0.0,
            "retry_count": 0,
            "action_result": {"audit_success": True},
        }
        flags = compute_tool_metrics(result)
        assert flags["tool_retrieval_success"] == 0.0

    def test_retrieval_failure_zero_time(self) -> None:
        result = {
            "dominant_category": "Software",
            "retrieved_chunks": [{"content": "chunk1"}],
            "retrieval_time_ms": 0.0,
            "retry_count": 0,
            "action_result": {"audit_success": True},
        }
        flags = compute_tool_metrics(result)
        assert flags["tool_retrieval_success"] == 0.0

    def test_critic_partial_credit(self) -> None:
        result = {
            "dominant_category": "User",
            "retrieved_chunks": [{"content": "chunk1"}],
            "retrieval_time_ms": 15.0,
            "retry_count": 1,
            "action_result": {"audit_success": True},
        }
        flags = compute_tool_metrics(result)
        assert flags["tool_critic_success"] == 0.5

    def test_action_failure(self) -> None:
        result = {
            "dominant_category": "User",
            "retrieved_chunks": [{"content": "chunk1"}],
            "retrieval_time_ms": 15.0,
            "retry_count": 0,
            "action_result": {"audit_success": False},
        }
        flags = compute_tool_metrics(result)
        assert flags["tool_action_success"] == 0.0

    def test_missing_action_result(self) -> None:
        result = {
            "dominant_category": "Science",
            "retrieved_chunks": [{"content": "chunk1"}],
            "retrieval_time_ms": 8.0,
            "retry_count": 0,
            "action_result": None,
        }
        flags = compute_tool_metrics(result)
        assert flags["tool_action_success"] == 0.0

    def test_no_retry_count(self) -> None:
        result = {
            "dominant_category": "Science",
            "retrieved_chunks": [{"content": "chunk1"}],
            "retrieval_time_ms": 8.0,
            "retry_count": -1,
            "action_result": {"audit_success": True},
        }
        flags = compute_tool_metrics(result)
        assert flags["tool_critic_success"] == 0.0


@pytest.mark.asyncio
async def test_robust_ascore_success() -> None:
    """Validates that a successful ascore() returns the expected value."""
    mock_scorer = AsyncMock()
    mock_scorer.ascore.return_value = MagicMock(value=0.85)

    score = await _robust_ascore(mock_scorer, user_input="q", reference="a", retrieved_contexts=["c"])
    assert score == 0.85
    mock_scorer.ascore.assert_awaited_once()


@pytest.mark.asyncio
async def test_robust_ascore_retries_on_timeout() -> None:
    """Validates exponential-backoff retry on timeout."""
    mock_scorer = AsyncMock()
    # First 2 calls timeout, 3rd succeeds
    mock_scorer.ascore.side_effect = [
        TimeoutError("timeout"),
        TimeoutError("timeout"),
        MagicMock(value=0.72),
    ]

    with patch("asyncio.sleep", AsyncMock()):
        score = await _robust_ascore(mock_scorer, max_retries=2, user_input="q", reference="a", retrieved_contexts=["c"])
    assert score == 0.72
    assert mock_scorer.ascore.await_count == 3


@pytest.mark.asyncio
async def test_robust_ascore_sentinel_on_exhaustion() -> None:
    """Validates that exhausted retries return METRIC_UNAVAILABLE."""
    mock_scorer = AsyncMock()
    mock_scorer.ascore.side_effect = TimeoutError("always fails")

    with patch("asyncio.sleep", AsyncMock()):
        score = await _robust_ascore(mock_scorer, max_retries=2, user_input="q", reference="a", retrieved_contexts=["c"])
    assert score == METRIC_UNAVAILABLE
    assert mock_scorer.ascore.await_count == 3  # initial + 2 retries

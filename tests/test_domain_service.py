"""
Unit tests for the domain/service layer (issue #4).

These exercise the business rules of ``QueryService`` with in-memory fakes for
every port — no FastAPI TestClient, no database, no MLflow server. This is the
payoff of the hexagonal separation: the rules are testable in isolation.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from src.domain.exceptions import (
    QueryProcessingError,
    QueryTimeoutError,
    QuotaExceededError,
    RateLimitExceededError,
)
from src.domain.metrics import MetricsCollector
from src.domain.models import QueryCommand
from src.domain.services import QueryService


class FakeRateLimiter:
    def __init__(self, allowed: bool = True, retry_after: int = 0) -> None:
        self._allowed = allowed
        self._retry_after = retry_after

    def allow(self, user_id: int):
        return self._allowed, self._retry_after


class FakeRagEngine:
    provider_name = "fake"

    def __init__(self, result: dict | None = None, raises: Exception | None = None) -> None:
        self._result = result or {}
        self._raises = raises

    def invoke(self, inputs: dict) -> dict:
        if self._raises:
            raise self._raises
        return self._result


class FakeTracker:
    def __init__(self) -> None:
        self.params: dict = {}
        self.metrics: dict = {}

    @contextmanager
    def start_run(self, run_name=None, tags=None):
        yield None

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics, step=None):
        self.metrics.update(metrics)


class FakeRepository:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def record_query(self, **kwargs):
        self.records.append(kwargs)

    def list_user_queries(self, user_id, limit):
        return []


def _service(rag=None, limiter=None, metrics=None, tracker=None):
    return QueryService(
        rag_engine=rag or FakeRagEngine({"response": "ok"}),
        rate_limiter=limiter or FakeRateLimiter(),
        tracker=tracker or FakeTracker(),
        metrics=metrics or MetricsCollector(),
    )


_CMD = QueryCommand(user_id=1, query="hello", session_id="s1")


@pytest.mark.asyncio
async def test_rate_limited_raises_and_counts():
    metrics = MetricsCollector()
    svc = _service(limiter=FakeRateLimiter(allowed=False, retry_after=7), metrics=metrics)
    with pytest.raises(RateLimitExceededError) as exc:
        await svc.handle(_CMD, quota_remaining_ok=True, repository=FakeRepository())
    assert exc.value.retry_after == 7
    assert metrics.snapshot()["rate_limited_count"] == 1


@pytest.mark.asyncio
async def test_quota_exceeded_raises():
    svc = _service()
    with pytest.raises(QuotaExceededError):
        await svc.handle(_CMD, quota_remaining_ok=False, repository=FakeRepository())


@pytest.mark.asyncio
async def test_successful_query_persists_and_maps_outcome():
    repo = FakeRepository()
    rag = FakeRagEngine({
        "response": "The answer.",
        "sources_cited": ["a.txt"],
        "dominant_category": "Science",
        "confidence": 0.9,
        "guard_input_result": {"is_safe": True, "injection_score": 0.1, "pii_detections": ["x"]},
        "guard_output_result": {"pii_redacted_on_output": []},
    })
    metrics = MetricsCollector()
    svc = _service(rag=rag, metrics=metrics)

    outcome = await svc.handle(_CMD, quota_remaining_ok=True, repository=repo)

    assert outcome.response == "The answer."
    assert outcome.dominant_category == "Science"
    assert outcome.pii_redacted_count == 1
    assert outcome.scoped_session == "u1:s1"
    # Persistence happened via the repository port.
    assert len(repo.records) == 1
    assert repo.records[0]["user_id"] == 1
    # Metrics recorded a success.
    snap = metrics.snapshot()
    assert snap["total_requests"] == 1
    assert snap["pii_redacted_count"] == 1


@pytest.mark.asyncio
async def test_blocked_query_counts_guard_block():
    rag = FakeRagEngine({
        "response": "blocked",
        "guard_input_result": {"is_safe": False},
        "guard_output_result": {},
    })
    metrics = MetricsCollector()
    svc = _service(rag=rag, metrics=metrics)
    outcome = await svc.handle(_CMD, quota_remaining_ok=True, repository=FakeRepository())
    assert outcome.blocked_by_guard is True
    assert metrics.snapshot()["blocked_by_guard_rate"] == 1.0


@pytest.mark.asyncio
async def test_engine_failure_becomes_processing_error():
    rag = FakeRagEngine(raises=RuntimeError("boom"))
    metrics = MetricsCollector()
    svc = _service(rag=rag, metrics=metrics)
    with pytest.raises(QueryProcessingError):
        await svc.handle(_CMD, quota_remaining_ok=True, repository=FakeRepository())
    assert metrics.snapshot()["total_errors"] == 1

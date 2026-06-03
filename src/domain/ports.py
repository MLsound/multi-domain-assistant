"""
Ports — the abstract boundaries the service depends on.

Following the dependency-inversion principle, the application service
(:mod:`src.domain.services`) is written against these ``Protocol`` interfaces,
never against concrete infrastructure. Adapters in :mod:`src.domain.adapters`
(and the existing rate limiter) satisfy them.

Using ``typing.Protocol`` means the existing classes (``SlidingWindowLimiter``,
``MLflowManager``) already conform structurally — no inheritance required.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable


@runtime_checkable
class RateLimiter(Protocol):
    """Per-user burst limiter (OWASP LLM04 mitigation)."""

    def allow(self, user_id: int) -> Tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``."""
        ...


@runtime_checkable
class RagEngine(Protocol):
    """Synchronous entry point into the multi-agent RAG pipeline."""

    @property
    def provider_name(self) -> str:
        ...

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the pipeline to completion and return the final state dict."""
        ...


@runtime_checkable
class ExperimentTracker(Protocol):
    """Observability sink (MLflow today, swappable tomorrow)."""

    def start_run(
        self, run_name: str | None = None, tags: Dict[str, str] | None = None
    ) -> AbstractContextManager[Any]:
        ...

    def log_params(self, params: Dict[str, Any]) -> None:
        ...

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        ...


@runtime_checkable
class QueryRepository(Protocol):
    """Persistence boundary for per-user query state.

    Implementations encapsulate all database access so the service never
    touches SQLAlchemy directly.
    """

    def record_query(
        self,
        *,
        user_id: int,
        session_id: str,
        query: str,
        response_preview: str,
        dominant_category: str,
        confidence: float,
        latency_ms: float,
        blocked_by_guard: bool,
    ) -> None:
        """Persist one query record and increment the user's daily counter."""
        ...

    def list_user_queries(self, user_id: int, limit: int) -> List[Any]:
        """Return the user's most recent query records (newest first)."""
        ...

"""
Application service for the query use case.

This is where the business logic that previously lived inside the FastAPI
``/query`` handler now lives: rate limiting, quota enforcement, RAG invocation
with a timeout budget, metrics aggregation, experiment tracking and
persistence. The service depends only on the ports in :mod:`src.domain.ports`
and raises domain exceptions — it has no knowledge of HTTP or SQLAlchemy.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

from src.domain.exceptions import (
    QueryProcessingError,
    QueryTimeoutError,
    QuotaExceededError,
    RateLimitExceededError,
)
from src.domain.metrics import MetricsCollector
from src.domain.models import QueryCommand, QueryOutcome
from src.domain.ports import (
    ExperimentTracker,
    QueryRepository,
    RagEngine,
    RateLimiter,
)

logger = logging.getLogger(__name__)

# Time budget for the full RAG pipeline. A business policy, not a transport
# detail, so it belongs in the service.
QUERY_TIMEOUT_SEC = 30.0


def _build_rag_inputs(query: str, scoped_session: str, is_help_section: bool) -> Dict[str, Any]:
    """Construct the initial LangGraph state for a query.

    Centralising the (large) input contract here keeps the API adapter free of
    pipeline internals.
    """
    return {
        "query": query,
        "session_id": scoped_session,
        "is_help_section": is_help_section,
        "history": [],
        "retry_count": 0,
        "from_cache": False,
        "sanitized_query": "",
        "guard_input_result": {},
        "guard_output_result": {},
        "category_probs": {},
        "dominant_category": "",
        "confidence": 0.0,
        "retrieved_chunks": [],
        "context_metadata": {},
        "retrieval_time_ms": 0.0,
        "sources_cited": [],
        "response": "",
        "token_count": 0,
        "critic_verdict": {},
        "action_result": {},
    }


class QueryService:
    """Orchestrates a single authenticated query end-to-end."""

    def __init__(
        self,
        *,
        rag_engine: RagEngine,
        rate_limiter: RateLimiter,
        tracker: ExperimentTracker,
        metrics: MetricsCollector,
        timeout_sec: float = QUERY_TIMEOUT_SEC,
    ) -> None:
        self._rag = rag_engine
        self._limiter = rate_limiter
        self._tracker = tracker
        self._metrics = metrics
        self._timeout_sec = timeout_sec

    async def handle(
        self,
        command: QueryCommand,
        *,
        quota_remaining_ok: bool,
        repository: QueryRepository,
    ) -> QueryOutcome:
        """Run the query use case.

        ``quota_remaining_ok`` is supplied by the caller (it depends on the
        live ``User`` row) so the service stays decoupled from the ORM.

        Raises one of the domain exceptions in :mod:`src.domain.exceptions`.
        """
        # 1. Burst rate limit (OWASP LLM04).
        allowed, retry_after = self._limiter.allow(command.user_id)
        if not allowed:
            self._metrics.record_rate_limited()
            raise RateLimitExceededError(retry_after)

        # 2. Daily quota.
        if not quota_remaining_ok:
            raise QuotaExceededError()

        scoped_session = f"u{command.user_id}:{command.session_id or 'default'}"

        with self._tracker.start_run(
            run_name=f"query-u{command.user_id}-{int(time.time())}",
            tags={"user_id": str(command.user_id), "session_id": command.session_id or "default"},
        ):
            self._tracker.log_params({
                "query_length": len(command.query),
                "session_id": command.session_id or "default",
                "is_help_override": command.is_help_override,
            })

            t0 = time.perf_counter()
            try:
                inputs = _build_rag_inputs(
                    command.query, scoped_session, command.is_help_override
                )
                # Task: Replace invoke wrapped in to_thread with native async ainvoke.
                result = await asyncio.wait_for(
                    self._rag.ainvoke(inputs),
                    timeout=self._timeout_sec,
                )
            except asyncio.TimeoutError as exc:
                self._metrics.record_error()
                raise QueryTimeoutError(self._timeout_sec) from exc
            except Exception as exc:  # noqa: BLE001 — translated to a domain error
                self._metrics.record_error()
                logger.exception("Unhandled error processing query")
                raise QueryProcessingError(str(exc)) from exc

            latency_ms = (time.perf_counter() - t0) * 1000
            outcome = self._assemble_outcome(result, latency_ms, scoped_session)

            self._metrics.record_success(
                latency_ms=latency_ms,
                from_cache=outcome.from_cache,
                blocked_by_guard=outcome.blocked_by_guard,
                pii_redacted=outcome.pii_redacted_count > 0,
            )

            self._tracker.log_metrics({
                "latency_ms": round(latency_ms, 2),
                "token_count": outcome.token_count,
                "confidence": outcome.confidence,
                "retry_count": outcome.retry_count,
                "retrieval_time_ms": outcome.retrieval_time_ms,
                "from_cache": 1 if outcome.from_cache else 0,
                "blocked_by_guard": 1 if outcome.blocked_by_guard else 0,
            })

            repository.record_query(
                user_id=command.user_id,
                session_id=scoped_session,
                query=command.query[:1000],
                response_preview=outcome.response[:500],
                dominant_category=outcome.dominant_category,
                confidence=outcome.confidence,
                latency_ms=latency_ms,
                blocked_by_guard=outcome.blocked_by_guard,
            )

            return outcome

    @staticmethod
    def _assemble_outcome(
        result: Dict[str, Any], latency_ms: float, scoped_session: str
    ) -> QueryOutcome:
        """Map the raw pipeline state into a domain outcome."""
        guard_in = result.get("guard_input_result")
        guard_out = result.get("guard_output_result")

        # Handle both dicts (from mocks) and Pydantic objects (from real graph)
        def _get_val(obj, key, default):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        blocked = not _get_val(guard_in, "is_safe", True)
        pii_in = _get_val(guard_in, "pii_detections", [])
        pii_out = _get_val(guard_out, "pii_redacted_on_output", [])
        inj_score = _get_val(guard_in, "injection_score", 0.0)
        inj_decision = _get_val(guard_in, "injection_decision", "allow")

        return QueryOutcome(
            response=result.get("response", "") or "",
            sources_cited=result.get("sources_cited", []),
            category_probs=result.get("category_probs", {}),
            dominant_category=result.get("dominant_category", ""),
            confidence=result.get("confidence", 0.0),
            retrieval_time_ms=result.get("retrieval_time_ms", 0.0),
            token_count=result.get("token_count", 0),
            retry_count=result.get("retry_count", 0),
            from_cache=result.get("from_cache", False),
            injection_score=inj_score,
            injection_decision=inj_decision,
            pii_redacted_count=len(pii_in) + len(pii_out),
            latency_ms=latency_ms,
            blocked_by_guard=blocked,
            scoped_session=scoped_session,
        )

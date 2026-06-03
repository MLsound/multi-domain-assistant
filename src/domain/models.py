"""
Framework-agnostic domain models.

These dataclasses are the contract between the API adapter and the service.
They deliberately avoid Pydantic / FastAPI / SQLAlchemy imports so the service
layer can be tested and reused without a web stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class QueryCommand:
    """An authenticated request to run the RAG pipeline.

    ``user_id`` and the quota fields are passed in so the service never needs
    to know about the ORM ``User`` object — only the data it acts on.
    """

    user_id: int
    query: str
    session_id: str | None = None
    is_help_override: bool = False


@dataclass
class QueryOutcome:
    """The result of a successfully processed query.

    Carries everything the API adapter needs to build its HTTP response, plus
    persistence-facing fields (``latency_ms``, ``blocked_by_guard``). It is the
    single return type of :meth:`QueryService.handle`.
    """

    response: str = ""
    sources_cited: List[str] = field(default_factory=list)
    category_probs: Dict[str, float] = field(default_factory=dict)
    dominant_category: str = ""
    confidence: float = 0.0
    retrieval_time_ms: float = 0.0
    token_count: int = 0
    retry_count: int = 0
    from_cache: bool = False
    injection_score: float = 0.0
    injection_decision: str = "allow"
    pii_redacted_count: int = 0

    # Telemetry used for persistence / metrics, not necessarily surfaced 1:1.
    latency_ms: float = 0.0
    blocked_by_guard: bool = False
    scoped_session: str = ""

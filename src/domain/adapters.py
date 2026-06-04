"""
Adapters — concrete implementations of the domain ports.

These are the only classes in the domain package that touch infrastructure
(the LangGraph ``RAGGraph`` and SQLAlchemy). They translate between the
framework world and the port contracts the service expects, keeping
infrastructure leakage out of :mod:`src.domain.services`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy.orm import Session

from src.auth.models import QueryRecord, User


class RagGraphEngine:
    """Adapts ``src.agents.graph.RAGGraph`` to the ``RagEngine`` port.

    ``RAGGraph`` exposes a compiled graph at ``.app``; the port wants a flat
    ``invoke(inputs)`` plus ``provider_name``.
    """

    def __init__(self, rag_graph: Any) -> None:
        self._graph = rag_graph

    @property
    def provider_name(self) -> str:
        return self._graph.provider_name

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._graph.app.invoke(inputs)


class SqlAlchemyQueryRepository:
    """Adapts a SQLAlchemy ``Session`` to the ``QueryRepository`` port.

    Constructed per-request with the request-scoped session yielded by the
    FastAPI ``get_db`` dependency.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

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
        # Increment the user's daily counter atomically with the insert.
        user = self._db.get(User, user_id)
        if user is not None:
            user.queries_today += 1

        record = QueryRecord(
            user_id=user_id,
            session_id=session_id,
            query=query,
            response_preview=response_preview,
            dominant_category=dominant_category,
            confidence=confidence,
            latency_ms=latency_ms,
            blocked_by_guard=blocked_by_guard,
        )
        self._db.add(record)
        self._db.commit()

    def list_user_queries(self, user_id: int, limit: int) -> List[QueryRecord]:
        return (
            self._db.query(QueryRecord)
            .filter(QueryRecord.user_id == user_id)
            .order_by(QueryRecord.created_at.desc())
            .limit(min(limit, 200))
            .all()
        )

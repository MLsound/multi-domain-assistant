"""
Domain / service layer.

This package isolates business logic from transport (FastAPI) and
infrastructure (SQLAlchemy, MLflow, the LangGraph RAG engine) following a
hexagonal (ports & adapters) architecture:

  * ``models``       — framework-agnostic data structures (commands / outcomes).
  * ``exceptions``   — domain errors, translated to HTTP status codes by the
                       API adapter. The domain never imports FastAPI.
  * ``ports``        — Protocol interfaces the service depends on. Concrete
                       infrastructure implements these (dependency inversion).
  * ``metrics``      — in-process metrics collector (was inlined in the API).
  * ``services``     — the application service that orchestrates a query.
  * ``adapters``     — concrete implementations of the ports backed by the
                       existing infrastructure (RAGGraph, SQLAlchemy, MLflow).

The dependency rule points inwards: ``services`` and ``models`` know nothing
about FastAPI or SQLAlchemy; the API layer wires concrete adapters into the
service at startup.
"""

from src.domain.exceptions import (
    DomainError,
    QueryProcessingError,
    QueryTimeoutError,
    QuotaExceededError,
    RateLimitExceededError,
)
from src.domain.models import QueryCommand, QueryOutcome
from src.domain.services import QueryService

__all__ = [
    "DomainError",
    "QueryCommand",
    "QueryOutcome",
    "QueryProcessingError",
    "QueryService",
    "QueryTimeoutError",
    "QuotaExceededError",
    "RateLimitExceededError",
]

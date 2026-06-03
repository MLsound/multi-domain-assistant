"""
FastAPI application for the Knowledge Assistant.

Endpoints:
  POST /auth/register, /auth/login, GET /auth/me  — authentication
  POST /query                                     — protected, JWT required
  GET  /health                                    — public health check
  GET  /metrics                                   — public aggregate metrics
  GET  /me/queries                                — per-user query history
"""

from __future__ import annotations

import logging
import sys

# Windows console defaults to cp1252; MLflow writes 🏃 to stdout when ending a
# run, which raises UnicodeEncodeError mid-request. Reconfigure stdio to UTF-8
# at import time so background uvicorn workers don't crash on log emojis.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.api.schemas import HealthResponse, QueryRecordOut, QueryRequest, QueryResponse
from src.auth.database import get_db, init_db
from src.auth.deps import get_current_user
from src.auth.models import User
from src.auth.router import router as auth_router
from src.config.mlflow_config import manager as mlflow_manager
from src.config.settings import settings
from src.domain.adapters import RagGraphEngine, SqlAlchemyQueryRepository
from src.domain.exceptions import (
    QueryProcessingError,
    QueryTimeoutError,
    QuotaExceededError,
    RateLimitExceededError,
)
from src.domain.metrics import metrics_collector
from src.domain.models import QueryCommand
from src.domain.services import QueryService
from src.security.rate_limiter import limiter

load_dotenv()
logger = logging.getLogger(__name__)

# Application service + RAG engine, wired once at startup (see lifespan).
# The API layer holds only these references; all business logic lives in the
# domain/service layer (src.domain).
_rag_system = None
_query_service: QueryService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources once at startup and wire the service."""
    global _rag_system, _query_service

    # CRITICAL: Set MLflow tracking URI BEFORE any @mlflow.trace decorated
    # methods are called. The decorators capture mlflow module state at call
    # time, not at import time, so this must run before RAGGraph initialisation.
    mlflow_manager.initialise()
    logger.info("MLflow tracking URI set to %s", settings.mlflow_tracking_uri)

    init_db()
    logger.info("Auth DB ready")

    logger.info("Initialising RAGGraph...")
    from src.agents.graph import RAGGraph

    _rag_system = RAGGraph()
    logger.info("RAGGraph ready — provider=%s", _rag_system.provider_name)

    # Compose the application service from concrete adapters (ports & adapters).
    _query_service = QueryService(
        rag_engine=RagGraphEngine(_rag_system),
        rate_limiter=limiter,
        tracker=mlflow_manager,
        metrics=metrics_collector,
    )
    logger.info("QueryService ready")
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title="Knowledge Assistant API",
    version="0.3.0",
    description=(
        "Multi-agent Agentic RAG for sustainable energy and smart building "
        "knowledge. Hardened against OWASP Top-10 LLM threats; user-aware."
    ),
    lifespan=lifespan,
)

app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "Knowledge Assistant API",
        "version": "0.3.0",
        "docs": "/docs",
        "health": "/health",
        "auth": ["/auth/register", "/auth/login", "/auth/me"],
        "query": "POST /query (Bearer token required)",
    }


# ---------------------------------------------------------------------------
# Protected query endpoint
# ---------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Thin transport adapter: build a command, delegate to the service, map
    domain outcomes/errors back to HTTP. No business logic lives here."""
    assert _query_service is not None, "QueryService not initialised"

    command = QueryCommand(
        user_id=user.id,
        query=req.query,
        session_id=req.session_id,
        is_help_override=req.is_help_override,
    )

    try:
        outcome = await _query_service.handle(
            command,
            quota_remaining_ok=user.queries_today < user.quota_queries_per_day,
            repository=SqlAlchemyQueryRepository(db),
        )
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QueryTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except QueryProcessingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        response=outcome.response,
        sources_cited=outcome.sources_cited,
        category_probs=outcome.category_probs,
        dominant_category=outcome.dominant_category,
        confidence=outcome.confidence,
        retrieval_time_ms=outcome.retrieval_time_ms,
        token_count=outcome.token_count,
        retry_count=outcome.retry_count,
        from_cache=outcome.from_cache,
        injection_score=outcome.injection_score,
        injection_decision=outcome.injection_decision,
        pii_redacted_count=outcome.pii_redacted_count,
    )


@app.get("/me/queries", response_model=List[QueryRecordOut])
def my_queries(
    limit: int = 50,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the authenticated user's own query history."""
    rows = SqlAlchemyQueryRepository(db).list_user_queries(user.id, limit)
    return [QueryRecordOut.model_validate(r) for r in rows]


@app.get("/health", response_model=HealthResponse)
async def health():
    qdrant_ok = False
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, timeout=3)
        client.get_collections()
        qdrant_ok = True
    except Exception:
        logger.warning("Qdrant health check failed")

    status = "ok" if qdrant_ok else "degraded"
    provider = _rag_system.provider_name if _rag_system else "not_initialised"
    return HealthResponse(
        status=status,
        qdrant_connected=qdrant_ok,
        active_provider=provider,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/metrics")
async def metrics():
    result = metrics_collector.snapshot()
    # Log aggregate metrics to MLflow (best-effort)
    try:
        mlflow_manager.log_metrics({
            "api.total_requests": result["total_requests"],
            "api.avg_latency_ms": result["avg_latency_ms"],
            "api.error_rate": result["error_rate"],
            "api.cache_hit_rate": result["cache_hit_rate"],
        })
    except Exception:
        pass
    return result

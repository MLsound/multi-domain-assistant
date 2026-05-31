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

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.api.schemas import HealthResponse, QueryRecordOut, QueryRequest, QueryResponse
from src.auth.database import get_db, init_db
from src.auth.deps import get_current_user
from src.auth.models import QueryRecord, User
from src.auth.router import router as auth_router
from src.config.mlflow_config import manager as mlflow_manager
from src.config.settings import settings
from src.security.rate_limiter import limiter

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory metrics counters (reset on process restart)
# ---------------------------------------------------------------------------
_metrics: Dict[str, Any] = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "errors": 0,
    "cache_hits": 0,
    "blocked_by_guard": 0,
    "rate_limited": 0,
    "pii_redacted": 0,
}

_rag_system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources once at startup."""
    global _rag_system

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
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title="Knowledge Assistant API",
    version="0.3.0",
    description=(
        "Multi-agent Agentic RAG for sustainable energy and smart building "
        "knowledge. Hardened against OWASP Top-10 LLM threats; user-aware. "
        "See `docs/CONTEXT.md` for details on the specific application context "
        "used in this deployment."
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
    global _metrics

    # Rate limit (LLM04 mitigation)
    allowed, retry_after = limiter.allow(user.id)
    if not allowed:
        _metrics["rate_limited"] += 1
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    # Daily quota
    if user.queries_today >= user.quota_queries_per_day:
        raise HTTPException(status_code=429, detail="Daily query quota exceeded")

    with mlflow_manager.start_run(
        run_name=f"query-u{user.id}-{int(time.time())}",
        tags={"user_id": str(user.id), "session_id": req.session_id or "default"},
    ):
        mlflow_manager.log_params({
            "query_length": len(req.query),
            "session_id": req.session_id or "default",
            "is_help_override": req.is_help_override,
        })

        t0 = time.perf_counter()
        try:
            # Session id is namespaced by user so two users cannot share state.
            scoped_session = f"u{user.id}:{req.session_id or 'default'}"

            inputs = {
                "query": req.query,
                "session_id": scoped_session,
                "is_help_section": req.is_help_override,
                "history": [],
                "retry_count": 0,
                "from_cache": False,
                "sanitized_query": "",
                "category_probs": {},
                "dominant_category": "",
                "confidence": 0.0,
                "retrieved_chunks": [],
                "context_metadata": {},
                "retrieval_time_ms": 0.0,
                "sources_cited": [],
                "response": "",
                "token_count": 0,
            }

            result = await asyncio.wait_for(
                _rag_system.app.ainvoke(inputs),
                timeout=30.0,
            )

            latency_ms = (time.perf_counter() - t0) * 1000
            _metrics["total_requests"] += 1
            _metrics["total_latency_ms"] += latency_ms
            if result.get("from_cache"):
                _metrics["cache_hits"] += 1

            guard_in = result.get("guard_input_result")
            guard_out = result.get("guard_output_result")
            blocked = not guard_in.is_safe if guard_in else False
            if blocked:
                _metrics["blocked_by_guard"] += 1
            pii_in = guard_in.pii_detections if guard_in else []
            pii_out = guard_out.pii_redacted_on_output if guard_out else []
            if pii_in or pii_out:
                _metrics["pii_redacted"] += 1

            # Log response metrics to MLflow
            mlflow_manager.log_metrics({
                "latency_ms": round(latency_ms, 2),
                "token_count": result.get("token_count", 0),
                "confidence": result.get("confidence", 0.0),
                "retry_count": result.get("retry_count", 0),
                "retrieval_time_ms": result.get("retrieval_time_ms", 0.0),
                "from_cache": 1 if result.get("from_cache") else 0,
                "blocked_by_guard": 1 if blocked else 0,
            })

            # Per-user persistent log
            user.queries_today += 1
            rec = QueryRecord(
                user_id=user.id,
                session_id=scoped_session,
                query=req.query[:1000],
                response_preview=(result.get("response", "") or "")[:500],
                dominant_category=result.get("dominant_category", ""),
                confidence=result.get("confidence", 0.0),
                latency_ms=latency_ms,
                blocked_by_guard=blocked,
            )
            db.add(rec)
            db.commit()

            return QueryResponse(
                response=result.get("response", ""),
                sources_cited=result.get("sources_cited", []),
                category_probs=result.get("category_probs", {}),
                dominant_category=result.get("dominant_category", ""),
                confidence=result.get("confidence", 0.0),
                retrieval_time_ms=result.get("retrieval_time_ms", 0.0),
                token_count=result.get("token_count", 0),
                retry_count=result.get("retry_count", 0),
                from_cache=result.get("from_cache", False),
                injection_score=guard_in.injection_score if guard_in else 0.0,
                injection_decision=guard_in.injection_decision if guard_in else "allow",
                pii_redacted_count=len(pii_in) + len(pii_out),
            )

        except asyncio.TimeoutError:
            _metrics["errors"] += 1
            raise HTTPException(status_code=504, detail="Request timed out after 30 seconds")
        except HTTPException:
            raise
        except Exception as exc:
            _metrics["errors"] += 1
            logger.exception("Unhandled error processing query")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/me/queries", response_model=List[QueryRecordOut])
def my_queries(
    limit: int = 50,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the authenticated user's own query history."""
    rows = (
        db.query(QueryRecord)
        .filter(QueryRecord.user_id == user.id)
        .order_by(QueryRecord.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )
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
    n = _metrics["total_requests"]
    result = {
        "total_requests": n,
        "avg_latency_ms": round(_metrics["total_latency_ms"] / n, 2) if n else 0.0,
        "error_rate": round(_metrics["errors"] / n, 4) if n else 0.0,
        "cache_hit_rate": round(_metrics["cache_hits"] / n, 4) if n else 0.0,
        "blocked_by_guard_rate": round(_metrics["blocked_by_guard"] / n, 4) if n else 0.0,
        "rate_limited_count": _metrics["rate_limited"],
        "pii_redacted_count": _metrics["pii_redacted"],
        "total_errors": _metrics["errors"],
        "total_cache_hits": _metrics["cache_hits"],
    }
    # Log aggregate metrics to MLflow (best-effort)
    try:
        mlflow_manager.log_metrics({
            "api.total_requests": n,
            "api.avg_latency_ms": result["avg_latency_ms"],
            "api.error_rate": result["error_rate"],
            "api.cache_hit_rate": result["cache_hit_rate"],
        })
    except Exception:
        pass
    return result

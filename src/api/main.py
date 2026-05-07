"""
FastAPI application for the Knowledge Assistant.

Endpoints:
  POST /query   — Submit a query and receive a grounded response.
  GET  /health  — Qdrant connectivity + active provider status.
  GET  /metrics — Aggregated request metrics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import HealthResponse, QueryRequest, QueryResponse
from src.config.settings import settings

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
}

# Lazy-initialised graph (populated in lifespan)
_rag_system = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources once at startup."""
    global _rag_system
    logger.info("Initialising RAGGraph...")
    from src.agents.graph import RAGGraph

    _rag_system = RAGGraph()
    logger.info("RAGGraph ready — provider=%s", _rag_system.provider_name)
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title="Knowledge Assistant API",
    version="0.2.0",
    description="Multi-agent Agentic RAG for sustainable energy and smart building knowledge.",
    lifespan=lifespan,
)

# CORS — allow localhost origins for development / demo
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint — returns API name and links to docs and health check."""
    return {
        "name": "Knowledge Assistant API",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "usage": "POST /query with JSON body: {\"query\": \"your question here\"}",
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Submit a natural-language query and receive a grounded response."""
    global _metrics

    t0 = time.perf_counter()
    try:
        inputs = {
            "query": req.query,
            "session_id": req.session_id or "anonymous",
            "is_help_section": req.is_help_override,
            "history": [],
            "retry_count": 0,
            "from_cache": False,
            # Provide sensible defaults for all optional state fields
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

        # Run graph in a thread to avoid blocking the asyncio event loop
        result = await asyncio.wait_for(
            asyncio.to_thread(_rag_system.app.invoke, inputs),
            timeout=30.0,
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        _metrics["total_requests"] += 1
        _metrics["total_latency_ms"] += latency_ms
        if result.get("from_cache"):
            _metrics["cache_hits"] += 1

        logger.info(
            "Query completed — latency=%.0fms provider=%s cache=%s",
            latency_ms,
            _rag_system.provider_name,
            result.get("from_cache"),
        )

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
        )

    except asyncio.TimeoutError:
        _metrics["errors"] += 1
        logger.error("Query timed out after 30s")
        raise HTTPException(status_code=504, detail="Request timed out after 30 seconds")
    except Exception as exc:
        _metrics["errors"] += 1
        logger.exception("Unhandled error processing query")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check system health: Qdrant connectivity and active LLM provider."""
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
    )


@app.get("/metrics")
async def metrics():
    """Return aggregated request metrics since last process start."""
    n = _metrics["total_requests"]
    return {
        "total_requests": n,
        "avg_latency_ms": round(_metrics["total_latency_ms"] / n, 2) if n else 0.0,
        "error_rate": round(_metrics["errors"] / n, 4) if n else 0.0,
        "cache_hit_rate": round(_metrics["cache_hits"] / n, 4) if n else 0.0,
        "total_errors": _metrics["errors"],
        "total_cache_hits": _metrics["cache_hits"],
    }

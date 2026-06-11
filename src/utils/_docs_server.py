"""
Lightweight docs server for taking screenshots of the API surface.

Why this script exists:
1. When generating API documentation or taking automated screenshots of Swagger UI/ReDoc,
   we don't want to spin up the entire multi-agent RAG system, Qdrant vector database,
   or MLflow tracking servers, as these require active API keys and external infrastructure.
2. This script imports the FastAPI application object `app` from `src.api.main` but 
   dynamically intercepts and replaces its `lifespan` handler with a no-op context manager.
3. As a result, the server boots in under 1 second, renders the OpenAPI schema perfectly,
   but does not initialize heavy graph, database, or telemetry components.

Usage:
  poetry run python -m src.utils.docs_server [port]
"""

from __future__ import annotations

import contextlib
import sys

# Import the main FastAPI application module
from src.api import main as m


@contextlib.asynccontextmanager
async def _noop_lifespan(app):  # noqa: ANN001
    """
    A no-op lifespan context manager.
    
    Replaces the heavy database migrations, MLflow start-up hooks, and LangGraph 
    resource allocation with a silent pass, setting the internal RAG system reference to None.
    """
    m._rag_system = None
    yield


# Swap the startup lifespan context on the FastAPI app before starting uvicorn.
# This prevents the async context manager inside `src.api.main` from executing.
m.app.router.lifespan_context = _noop_lifespan


if __name__ == "__main__":
    import uvicorn

    # Allow custom port overriding via command line argument (defaults to 8077)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
    
    # Run uvicorn on localhost with warning logs only to keep the CLI clean
    uvicorn.run(m.app, host="127.0.0.1", port=port, log_level="warning")

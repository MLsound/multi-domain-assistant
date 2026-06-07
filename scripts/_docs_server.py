"""
Lightweight docs server for taking screenshots of the API surface.

Imports the real FastAPI ``app`` but replaces the heavy startup ``lifespan``
(which would initialise MLflow + the RAGGraph / Qdrant stack) with a no-op so
the Swagger UI (/docs) and ReDoc (/redoc) render without external services.

Run:  python scripts/_docs_server.py [port]
"""

from __future__ import annotations

import contextlib
import sys

from src.api import main as m


@contextlib.asynccontextmanager
async def _noop_lifespan(app):  # noqa: ANN001
    # Provide a placeholder so any incidental access doesn't explode.
    m._rag_system = None
    yield


# Swap the startup context before uvicorn drives it.
m.app.router.lifespan_context = _noop_lifespan


if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
    uvicorn.run(m.app, host="127.0.0.1", port=port, log_level="warning")

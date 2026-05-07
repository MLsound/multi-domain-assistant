"""
Agent tracer — decorator that logs per-invocation traces to a JSONL file.

Controlled by settings.enable_tracing (default False).
When enabled, each agent run() call appends one JSON record to
settings.agent_traces_path (default: logs/agent_traces.jsonl).

Record schema:
  {
    "timestamp": ISO-8601 string,
    "agent": str,
    "latency_ms": float,
    "success": bool,
    "error": str | null,
    "input_query": str (first 100 chars),
    "output_keys": [str]
  }
"""

from __future__ import annotations

import functools
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.config.settings import settings

logger = logging.getLogger(__name__)


def trace_agent(agent_name: str) -> Callable:
    """
    Decorator factory for agent run()/execute() methods.

    If settings.enable_tracing is False, the decorator is a transparent no-op
    with negligible overhead.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self_or_state, state_arg=None):
            # Support both bound methods (self, state) and plain functions (state)
            actual_state: dict = state_arg if state_arg is not None else self_or_state

            t0 = time.perf_counter()
            error_msg: str | None = None
            result: Any = {}

            try:
                if state_arg is not None:
                    result = func(self_or_state, state_arg)
                else:
                    result = func(self_or_state)
                return result
            except Exception as exc:
                error_msg = str(exc)
                raise
            finally:
                if settings.enable_tracing:
                    latency_ms = (time.perf_counter() - t0) * 1000
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "agent": agent_name,
                        "latency_ms": round(latency_ms, 2),
                        "success": error_msg is None,
                        "error": error_msg,
                        "input_query": (
                            str(actual_state.get("query", ""))[:100]
                            if isinstance(actual_state, dict)
                            else ""
                        ),
                        "output_keys": list(result.keys()) if isinstance(result, dict) else [],
                    }
                    _write_trace(record)

        return wrapper

    return decorator


def _write_trace(record: dict) -> None:
    """Append a trace record to the JSONL file."""
    path = Path(settings.agent_traces_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logger.warning("Failed to write trace record for agent=%s", record.get("agent"))

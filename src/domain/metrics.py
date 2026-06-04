"""
In-process metrics collector.

Previously this lived as a bare ``_metrics`` dict inside ``src.api.main``,
mixing aggregation logic into the transport layer. It now lives in the domain
so the service owns counter updates and the API only reads a computed snapshot.

Counters reset on process restart (same semantics as before). For a
multi-worker deployment this would move behind a shared store; the interface
stays the same.
"""

from __future__ import annotations

import threading
from typing import Any, Dict


class MetricsCollector:
    """Thread-safe aggregate request counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c: Dict[str, Any] = {
            "total_requests": 0,
            "total_latency_ms": 0.0,
            "errors": 0,
            "cache_hits": 0,
            "blocked_by_guard": 0,
            "rate_limited": 0,
            "pii_redacted": 0,
        }

    def record_success(
        self,
        *,
        latency_ms: float,
        from_cache: bool,
        blocked_by_guard: bool,
        pii_redacted: bool,
    ) -> None:
        with self._lock:
            self._c["total_requests"] += 1
            self._c["total_latency_ms"] += latency_ms
            if from_cache:
                self._c["cache_hits"] += 1
            if blocked_by_guard:
                self._c["blocked_by_guard"] += 1
            if pii_redacted:
                self._c["pii_redacted"] += 1

    def record_error(self) -> None:
        with self._lock:
            self._c["errors"] += 1

    def record_rate_limited(self) -> None:
        with self._lock:
            self._c["rate_limited"] += 1

    def snapshot(self) -> Dict[str, Any]:
        """Return the computed aggregate metrics (rates + totals)."""
        with self._lock:
            n = self._c["total_requests"]
            return {
                "total_requests": n,
                "avg_latency_ms": round(self._c["total_latency_ms"] / n, 2) if n else 0.0,
                "error_rate": round(self._c["errors"] / n, 4) if n else 0.0,
                "cache_hit_rate": round(self._c["cache_hits"] / n, 4) if n else 0.0,
                "blocked_by_guard_rate": (
                    round(self._c["blocked_by_guard"] / n, 4) if n else 0.0
                ),
                "rate_limited_count": self._c["rate_limited"],
                "pii_redacted_count": self._c["pii_redacted"],
                "total_errors": self._c["errors"],
                "total_cache_hits": self._c["cache_hits"],
            }


# Module-level singleton — single-process FastAPI deployment.
metrics_collector = MetricsCollector()

"""
Per-user token-bucket rate limiter — addresses OWASP LLM04
(Model Denial of Service).

In-memory implementation (sufficient for a single-process FastAPI
deployment). For a multi-worker production rollout this would move to
Redis; the public interface stays the same.

Two limits are enforced concurrently:
  1. Burst: `burst` requests per `window_sec` seconds (sliding window).
  2. Daily quota: persisted on the User row in the database.

Both are needed: the burst stops scraping spikes; the daily quota stops
slow long-horizon scraping (slide 31 — "Model leeching").
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self, *, burst: int = 30, window_sec: float = 60.0) -> None:
        self.burst = burst
        self.window_sec = window_sec
        self._events: dict[int, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, user_id: int) -> tuple[bool, int]:
        """
        Returns (allowed, retry_after_seconds).

        retry_after is 0 when the request is allowed.
        """
        now = time.time()
        cutoff = now - self.window_sec
        with self._lock:
            q = self._events[user_id]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.burst:
                wait = max(1, int(self.window_sec - (now - q[0])))
                return False, wait
            q.append(now)
            return True, 0


# Module-level singleton — safe because FastAPI runs in a single process.
limiter = SlidingWindowLimiter()

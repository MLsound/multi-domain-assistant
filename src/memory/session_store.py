"""
In-memory session history store with TTL-based expiration.

Stores the last N conversation turns per session_id. Sessions expire after
settings.session_ttl_minutes of inactivity. Designed to be replaced with a
Redis backend in production (the interface is identical).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List

from src.agents.state import ConversationTurn
from src.config.settings import settings

logger = logging.getLogger(__name__)


class SessionStore:
    """Thread-unsafe in-memory session store (sufficient for single-worker dev)."""

    def __init__(self) -> None:
        # {session_id: {"history": [...], "last_access": float}}
        self._store: Dict[str, Dict] = {}

    def get_history(self, session_id: str) -> List[ConversationTurn]:
        """Return conversation history for session_id, or [] if expired/absent."""
        entry = self._store.get(session_id)
        if entry is None:
            return []

        age_minutes = (time.time() - entry["last_access"]) / 60
        if age_minutes > settings.session_ttl_minutes:
            del self._store[session_id]
            logger.debug(
                "Session '%s' expired after %.1f minutes", session_id, age_minutes
            )
            return []

        entry["last_access"] = time.time()
        return list(entry["history"])

    def append_turn(
        self, session_id: str, query: str, response: str
    ) -> None:
        """Append one conversation turn to the session history."""
        if session_id not in self._store:
            self._store[session_id] = {"history": [], "last_access": time.time()}

        self._store[session_id]["history"].append(
            ConversationTurn(query=query, response=response)
        )
        self._store[session_id]["last_access"] = time.time()

        logger.debug(
            "Session '%s': %d turns stored",
            session_id,
            len(self._store[session_id]["history"]),
        )

    def clear(self, session_id: str) -> None:
        """Remove a session entirely."""
        self._store.pop(session_id, None)

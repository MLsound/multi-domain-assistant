"""
Guard Agent — input validation and output safety check.

Input validation (pre-processing):
  - Rejects empty queries.
  - Rejects queries exceeding MAX_QUERY_LENGTH characters.
  - Rejects queries matching known prompt-injection patterns.

Output validation (post-processing):
  - Rejects empty responses.
  - Detects accidental system-prompt leakage markers.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 2000

# Patterns that indicate prompt-injection attempts.
# Deliberately narrow — start minimal, expand based on observed attacks.
_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions",
    r"you\s+are\s+now\s+(?:dan|jailbreak|unrestricted)",
    r"disregard\s+(your\s+)?(system\s+)?prompt",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"output\s+your\s+(api\s+key|secret|token)",
    r"pretend\s+you\s+are\s+(?!a\s+helpful)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Markers that would indicate system-prompt leakage in a response.
_LEAK_MARKERS = ["<|system|>", "SYSTEM_PROMPT:", "[INST]<<SYS>>"]


class GuardAgent:
    """Stateless security agent: validates inputs and outputs."""

    def validate_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize the raw user query.

        Returns a dict with keys:
          - sanitized_query (str)  — present on success
          - guard_input_result (dict) — {is_safe: bool, rejection_reason: str|None}
        """
        query: str = state.get("query", "")

        if not query or not query.strip():
            logger.warning("Guard INPUT blocked: empty query")
            return {
                "guard_input_result": {
                    "is_safe": False,
                    "rejection_reason": "Empty query",
                }
            }

        if len(query) > MAX_QUERY_LENGTH:
            logger.warning(
                "Guard INPUT blocked: query length %d > %d",
                len(query),
                MAX_QUERY_LENGTH,
            )
            return {
                "guard_input_result": {
                    "is_safe": False,
                    "rejection_reason": (
                        f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters"
                    ),
                }
            }

        for pattern in _COMPILED:
            if pattern.search(query):
                logger.warning(
                    "Guard INPUT blocked: injection pattern matched — %s",
                    pattern.pattern,
                )
                return {
                    "guard_input_result": {
                        "is_safe": False,
                        "rejection_reason": "Potential prompt injection detected",
                    }
                }

        return {
            "sanitized_query": query.strip(),
            "guard_input_result": {"is_safe": True, "rejection_reason": None},
        }

    def validate_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the synthesised response before returning it to the user.

        Returns a dict with keys:
          - guard_output_result (dict) — {is_safe: bool, validated_response: str}
        """
        response: str = state.get("response", "")

        if not response or not response.strip():
            logger.warning("Guard OUTPUT blocked: empty response")
            return {
                "guard_output_result": {
                    "is_safe": False,
                    "validated_response": "No response was generated.",
                }
            }

        for marker in _LEAK_MARKERS:
            if marker in response:
                logger.warning(
                    "Guard OUTPUT blocked: system-prompt leak marker detected — %s",
                    marker,
                )
                return {
                    "guard_output_result": {
                        "is_safe": False,
                        "validated_response": "Response could not be delivered safely.",
                    }
                }

        return {
            "guard_output_result": {
                "is_safe": True,
                "validated_response": response,
            }
        }

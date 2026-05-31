"""
Guard Agent — input validation and output safety check.

Layered defence (defence in depth, slide 38):
  Input phase:
    1. Empty / over-length rejection.
    2. Heuristic injection scorer (own implementation).
    3. PII redaction (own implementation) — never propagate raw PII to LLM.

  Output phase:
    1. Empty rejection.
    2. Canary token leak detection (signed-prompt, slide 25).
    3. System-prompt-marker leak detection.
    4. Output PII scrubbing (final fail-safe).

This module addresses OWASP Top-10 for LLM 2025: LLM01 (prompt injection),
LLM02 (insecure output), LLM06 (sensitive info disclosure).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import mlflow

from src.agents.state import GuardResult
from src.security.canary import output_leaks_canary
from src.security.injection_scorer import score as score_injection
from src.security.pii_redactor import redact

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 2000

# Markers that would indicate raw system-prompt leakage in a response.
_LEAK_MARKERS = ["<|system|>", "SYSTEM_PROMPT:", "[INST]<<SYS>>", "[INTERNAL]"]


class GuardAgent:
    """Stateless security agent: validates inputs and outputs."""

    # ------------------------------------------------------------------
    # INPUT phase
    # ------------------------------------------------------------------
    @mlflow.trace(name="guard_input")
    async def validate_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query: str = state.get("query", "")

        if not query or not query.strip():
            return self._reject_input("Empty query")

        if len(query) > MAX_QUERY_LENGTH:
            return self._reject_input(
                f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters"
            )

        # 2. Heuristic injection scoring (own rules)
        verdict = score_injection(query)
        if verdict.decision == "block":
            logger.warning(
                "Guard INPUT blocked: injection score=%.2f rules=%s",
                verdict.score, verdict.matched_rules,
            )
            return self._reject_input(
                f"Potential prompt injection detected "
                f"(score={verdict.score:.2f}, rules={','.join(verdict.matched_rules)})"
            )

        # 3. PII redaction — strip before passing to retrieval / synthesis.
        redaction = redact(query)
        sanitized_query = redaction.text.strip()

        return {
            "sanitized_query": sanitized_query,
            "guard_input_result": GuardResult(
                is_safe=True,
                rejection_reason=None,
                injection_score=verdict.score,
                injection_decision=verdict.decision,
                pii_detections=redaction.detections,
            ),
        }

    # ------------------------------------------------------------------
    # OUTPUT phase
    # ------------------------------------------------------------------
    @mlflow.trace(name="guard_output")
    async def validate_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response: str = state.get("response", "")

        if not response or not response.strip():
            return self._reject_output("No response was generated.")

        # 1. Canary leak (signed-prompt, slide 25)
        if output_leaks_canary(response):
            logger.error("Guard OUTPUT blocked: canary token leaked — system prompt was exposed")
            return self._reject_output("Response could not be delivered safely.")

        # 2. Generic system-prompt markers
        for marker in _LEAK_MARKERS:
            if marker in response:
                logger.warning("Guard OUTPUT blocked: leak marker detected — %s", marker)
                return self._reject_output("Response could not be delivered safely.")

        # 3. Final PII scrub on the way out (defensive — synthesis should
        #    not produce PII because input was redacted, but if any leaks
        #    in via retrieved chunks we redact here).
        redaction = redact(response)

        return {
            "guard_output_result": GuardResult(
                is_safe=True,
                validated_response=redaction.text,
                pii_redacted_on_output=redaction.detections,
            )
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _reject_input(reason: str) -> Dict[str, Any]:
        return {
            "guard_input_result": GuardResult(
                is_safe=False,
                rejection_reason=reason,
            )
        }

    @staticmethod
    def _reject_output(safe_message: str) -> Dict[str, Any]:
        return {
            "guard_output_result": GuardResult(
                is_safe=False,
                validated_response=safe_message,
            )
        }

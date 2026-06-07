"""
Action Agent — automated output actions.

Fires after every successful response:
  1. Always: writes a structured JSON record to logs/audit.jsonl.
  2. Optional: POSTs the same record to WEBHOOK_URL if configured.

Action failures are logged but never block response delivery (fire-and-forget).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import src.config.mlflow_setup
import mlflow

from src.agents.state import ActionResult
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ActionAgent:
    """Executes automated post-response output actions."""

    @mlflow.trace(name="action")
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write audit log and optionally fire a webhook.

        Reads:  session_id, query, dominant_category, confidence,
                category_probs, retrieval_time_ms, token_count,
                sources_cited, critic_verdict, retry_count,
                from_cache, response
        Writes: action_result
        """
        critic_verdict = state.get("critic_verdict")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": state.get("session_id", "unknown"),
            "query": state.get("query", ""),
            "dominant_category": state.get("dominant_category", ""),
            "confidence": state.get("confidence", 0.0),
            "category_probs": state.get("category_probs", {}),
            "retrieval_time_ms": state.get("retrieval_time_ms", 0.0),
            "token_count": state.get("token_count", 0),
            "sources_cited": state.get("sources_cited", []),
            "critic_score": critic_verdict.score if critic_verdict else None,
            "retry_count": state.get("retry_count", 0),
            "from_cache": state.get("from_cache", False),
            # Store only a preview to keep log files manageable
            "response_preview": state.get("response", "")[:200],
        }

        action_type = "json_log"
        success = False
        details = ""

        # --- 1. JSON audit log (always active) ---
        log_path = Path(settings.audit_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            success = True
            details = str(log_path)
            logger.debug("Audit log written: %s", log_path)
        except Exception:
            logger.exception("Audit log write failed")
            details = "log write error"

        # --- 2. Webhook POST (optional) ---
        if settings.webhook_url:
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        settings.webhook_url,
                        json=record,
                        timeout=10.0,
                    )
                    r.raise_for_status()
                action_type = "json_log+webhook"
                logger.info("Webhook delivered to %s", settings.webhook_url)
            except Exception:
                logger.exception(
                    "Webhook delivery failed — response still returned to user"
                )

        return {
            "action_result": ActionResult(
                action_type=action_type,
                success=success,
                details=details,
            )
        }

"""
Router Agent — domain classification.

Wraps MLPRouter and applies the help-section probability override when
state["is_help_section"] is True.  The dominant category and scalar
confidence value are derived here so downstream agents don't need to
recompute them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import mlflow

from src.config.settings import settings
from src.router.mlp_router import MLPRouter

logger = logging.getLogger(__name__)


class RouterAgent:
    """Classifies queries into domain categories using a pre-trained MLP."""

    def __init__(self, mlp_router: MLPRouter) -> None:
        self.mlp_router = mlp_router

    @mlflow.trace(name="router")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify the sanitized query.

        Reads:  state["sanitized_query"] or state["query"]
                state["is_help_section"]
        Writes: category_probs, dominant_category, confidence
        """
        query: str = state.get("sanitized_query") or state.get("query", "")

        probs = self.mlp_router.route(query)
        dominant = max(probs, key=probs.get)
        confidence = probs[dominant]

        logger.info(
            "Router: dominant=%s confidence=%.3f probs=%s",
            dominant,
            confidence,
            {k: round(v, 3) for k, v in probs.items()},
        )

        # Help-section override: shift weight toward Software documentation
        if state.get("is_help_section"):
            probs = dict(settings.help_override_probs)
            dominant = max(probs, key=probs.get)
            confidence = probs[dominant]
            logger.info("Help-section override applied: %s", probs)

        return {
            "category_probs": probs,
            "dominant_category": dominant,
            "confidence": confidence,
        }

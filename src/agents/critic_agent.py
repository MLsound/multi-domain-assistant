"""
Critic Agent — faithfulness evaluator and dynamic retry trigger.

This is the system's inter-agent dynamic communication mechanism:
  - Evaluates whether every factual claim in the synthesis response is
    directly supported by the retrieved context chunks.
  - Returns a structured verdict that drives the LangGraph conditional edge:
      approved=True  → proceed to output guard
      approved=False → loop back to Retrieval with a refined query
  - If the LLM call fails, defaults to approved=True (fail-open) so that
    a single critic failure never blocks the user from receiving a response.
  - If max_retries has been reached, also approves unconditionally.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

import mlflow

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.agents.state import CriticVerdict
from src.config.model_registry import ModelRegistry
from src.config.settings import settings

logger = logging.getLogger(__name__)

_CRITIC_SYSTEM_PROMPT = (
    "You are a strict faithfulness evaluator for a retrieval-augmented generation system. "
    "You receive a query, context chunks retrieved from a knowledge base, "
    "and a generated response. "
    "Evaluate whether every factual claim in the response is directly supported "
    "by the provided context. "
    "Return ONLY a valid JSON object with these fields:\n"
    '  "approved": boolean — true if all claims are supported, false otherwise\n'
    '  "score": float 0.0–1.0 — fraction of claims that are supported\n'
    '  "issues": list of strings — unsupported claims (empty list if approved)\n'
    '  "suggested_refinement": string — a refined query hint if not approved, '
    "otherwise null\n"
    "Return ONLY the JSON object. No preamble, no explanation, no markdown fences."
)

_DEFAULT_VERDICT = CriticVerdict(
    approved=True,
    score=1.0,
    issues=[],
    suggested_refinement=None,
)


def _parse_verdict(raw: str) -> Dict[str, Any]:
    """Parse JSON verdict from LLM output, stripping markdown fences if present."""
    raw = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first and last fence lines
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        raw = "\n".join(inner).strip()
    return json.loads(raw)


class CriticAgent:
    """Evaluates synthesis output and drives the conditional retry loop."""

    def __init__(self, model_registry: ModelRegistry) -> None:
        self.registry = model_registry

    @mlflow.trace(name="critic")
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the synthesised response for faithfulness.

        Reads:  state["sanitized_query"] / state["query"]
                state["response"]
                state["retrieved_chunks"]
                state["retry_count"]
        Writes: critic_verdict
        """
        query: str = state.get("sanitized_query") or state.get("query", "")
        response: str = state.get("response", "")
        chunks = state.get("retrieved_chunks", [])
        retry_count: int = state.get("retry_count", 0)

        # Force-approve if max retries already reached (prevents infinite loops)
        if retry_count >= settings.max_retries:
            logger.info(
                "Critic: max_retries=%d reached — force-approving",
                settings.max_retries,
            )
            return {"critic_verdict": _DEFAULT_VERDICT}

        context_str = "\n\n".join(
            f"[source:{c.metadata.get('source_id', '?')}] {c.content}"
            for c in chunks
        )

        llm = self.registry.get_llm(complexity="simple")
        messages = [
            SystemMessage(content=_CRITIC_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Query: {query}\n\n"
                    f"Context chunks:\n{context_str}\n\n"
                    f"Generated response:\n{response}"
                )
            ),
        ]

        @retry(
            wait=wait_exponential(min=2, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        async def _invoke():
            return await llm.ainvoke(messages)

        try:
            resp = await _invoke()
            verdict_dict = _parse_verdict(resp.content)
            verdict = CriticVerdict(**verdict_dict)

            logger.info(
                "Critic verdict: approved=%s score=%.2f issues=%s",
                verdict.approved,
                verdict.score,
                verdict.issues,
            )
            return {"critic_verdict": verdict}

        except Exception:
            logger.exception(
                "Critic LLM call failed — defaulting to approve (fail-open)"
            )
            return {"critic_verdict": _DEFAULT_VERDICT}

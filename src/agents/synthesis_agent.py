"""
Synthesis Agent — grounded LLM response generation.

Uses ModelRegistry to select the appropriate model tier:
  - confidence > threshold → simple (faster/cheaper)
  - confidence <= threshold → complex (more capable)

The system prompt instructs the model to:
  1. Respond only from provided context.
  2. Append a SOURCES: [...] line listing source_id values used.
  3. Admit ignorance rather than hallucinate.

Token count is extracted from the response's usage_metadata when available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.model_registry import ModelRegistry
from src.config.settings import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a Knowledge Assistant specialised in sustainable energy, "
    "photovoltaic systems, smart building technologies, and home energy management. "
    "Synthesise a response based ONLY on the provided context chunks. "
    "After your answer, on a new line write exactly: "
    "SOURCES: [source_id_1, source_id_2, ...] "
    "listing every source_id you drew information from. "
    "If the context does not contain enough information, respond with: "
    "'I don't have enough information in my knowledge base to answer this question.' "
    "Never fabricate facts or cite sources not present in the context."
)


def _extract_token_count(response_obj: Any) -> int:
    """Safely extract total token count from a LangChain response object."""
    try:
        meta = getattr(response_obj, "usage_metadata", None)
        if meta is None:
            return 0
        if isinstance(meta, dict):
            return int(meta.get("total_tokens", 0))
        return int(getattr(meta, "total_tokens", 0))
    except Exception:
        return 0


class SynthesisAgent:
    """Generates a grounded, source-cited response from retrieved context."""

    def __init__(self, model_registry: ModelRegistry) -> None:
        self.registry = model_registry

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response from retrieved chunks.

        Reads:  state["sanitized_query"] / state["query"]
                state["retrieved_chunks"]
                state["context_metadata"]
                state["history"]
                state["confidence"]
        Writes: response, sources_cited, token_count
        """
        chunks = state.get("retrieved_chunks", [])
        context_metadata = state.get("context_metadata", {})
        query: str = state.get("sanitized_query") or state.get("query", "")
        history = state.get("history", [])

        # Build context string with source tags
        context_parts = [
            f"[{c.get('category', '?')}]"
            f"[source:{c.get('metadata', {}).get('source_id', 'unknown')}] "
            f"{c.get('content', '')}"
            for c in chunks
        ]
        context_str = "\n\n".join(context_parts)

        if context_metadata:
            context_str += f"\n\n[Environmental Data]: {context_metadata}"

        if not context_str.strip():
            logger.warning("Synthesis: no context available — returning fallback")
            return {
                "response": (
                    "I don't have enough information in my knowledge base "
                    "to answer this question."
                ),
                "sources_cited": [],
                "token_count": 0,
            }

        # Include last 3 conversation turns for continuity
        history_str = ""
        if history:
            recent = history[-3:]
            history_str = "\n\nConversation history:\n" + "\n".join(
                f"User: {h.get('query', '')}\nAssistant: {h.get('response', '')}"
                for h in recent
            )

        confidence: float = state.get("confidence", 0.0)
        complexity = (
            "simple"
            if confidence > settings.router_confidence_threshold
            else "complex"
        )
        llm = self.registry.get_llm(complexity=complexity)
        logger.debug(
            "Synthesis: confidence=%.3f → complexity=%s", confidence, complexity
        )

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(
                content=f"Context:\n{context_str}{history_str}\n\nQuery: {query}"
            ),
        ]

        @retry(
            wait=wait_exponential(min=1, max=30),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def _invoke():
            return llm.invoke(messages)

        try:
            resp = _invoke()
            response_text: str = resp.content
            token_count = _extract_token_count(resp)

            # Parse "SOURCES: [...]" from the end of the response
            sources_cited: list[str] = []
            if "SOURCES:" in response_text:
                parts = response_text.split("SOURCES:")
                response_text = parts[0].strip()
                raw_sources = parts[1].strip().strip("[]")
                sources_cited = [
                    s.strip().strip("'\"")
                    for s in raw_sources.split(",")
                    if s.strip()
                ]

            logger.info(
                "Synthesis complete — tokens=%d sources=%s",
                token_count,
                sources_cited,
            )
            return {
                "response": response_text,
                "sources_cited": sources_cited,
                "token_count": token_count,
            }

        except Exception:
            logger.exception("Synthesis LLM call failed")
            return {
                "response": "Service temporarily unavailable. Please try again.",
                "sources_cited": [],
                "token_count": 0,
            }

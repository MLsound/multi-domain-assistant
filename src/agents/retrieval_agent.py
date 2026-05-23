"""
Retrieval Agent — weighted semantic search with environmental enrichment.

Stage 1: Weighted Qdrant search (Score = P(c|q) * sim(q, d)).
Stage 2: Cross-encoder reranking (added in Phase 5 inside WeightedRetriever).
Stage 3: Science-domain MCP enrichment when P(Science) > threshold.

On retry (retry_count > 0), appends the Critic's suggested_refinement to
the query so that re-retrieval targets the specific gap identified.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

import mlflow

from src.config.settings import settings
from src.retrieval.weighted_retriever import WeightedRetriever
from src.tools.weather_mcp import get_environmental_data

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """Executes the two-stage retriever-ranker pipeline."""

    def __init__(self, retriever: WeightedRetriever) -> None:
        self.retriever = retriever

    @mlflow.trace(name="retrieval")
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve context chunks for the current query.

        Reads:  state["sanitized_query"] / state["query"]
                state["category_probs"]
                state["retry_count"]
                state["critic_verdict"] (on retry)
        Writes: retrieved_chunks, context_metadata, retrieval_time_ms, sources_cited
        """
        query: str = state.get("sanitized_query") or state.get("query", "")

        # On retry, append critic's refinement hint to the query
        retry_count: int = state.get("retry_count", 0)
        if retry_count > 0:
            verdict = state.get("critic_verdict") or {}
            refinement = verdict.get("suggested_refinement")
            if refinement:
                query = f"{query} {refinement}"
                logger.info(
                    "Retry %d — refined query: %.100s", retry_count, query
                )

        probs: Dict[str, float] = state.get("category_probs", {})

        t0 = time.perf_counter()
        chunks = self.retriever.search(
            query=query,
            category_probs=probs,
            top_k=settings.retrieval_top_k,
        )
        retrieval_time_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Retrieved %d chunks in %.1f ms", len(chunks), retrieval_time_ms
        )

        # Science-domain enrichment via MCP tool
        context_metadata: Dict[str, Any] = {}
        if probs.get("Science", 0) > settings.science_threshold:
            try:
                env_data = get_environmental_data.invoke({})
                context_metadata = {"environmental_conditions": env_data}
                logger.debug("Environmental metadata injected: %s", env_data)
            except Exception:
                logger.warning("MCP tool failed — proceeding without env data")

        sources = list({
            c.get("metadata", {}).get("source_id", "unknown")
            for c in chunks
        })

        return {
            "retrieved_chunks": chunks,
            "context_metadata": context_metadata,
            "retrieval_time_ms": retrieval_time_ms,
            "sources_cited": sources,
        }

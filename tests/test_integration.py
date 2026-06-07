"""
End-to-end integration tests for the Knowledge Assistant graph.

The LLM (ModelRegistry) and Qdrant (WeightedRetriever + SemanticCache) are
mocked so these tests run without external services.  The full LangGraph
pipeline executes with real agent logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.state import RetrievalChunk


def _mock_llm_response(text: str) -> MagicMock:
    """Build a mock LLM response object."""
    resp = MagicMock()
    resp.content = text
    resp.usage_metadata = None
    return resp


def _mock_chunks(category: str = "Science", source: str = "test.txt") -> list[RetrievalChunk]:
    return [
        RetrievalChunk(
            content="Crystalline silicon PV efficiency degrades above 25°C.",
            category=category,
            metadata={"source_id": source, "category": category},
            score=0.9,
            original_score=0.9,
        )
    ]


def _base_inputs(query: str, is_help: bool = False) -> dict:
    return {
        "query": query,
        "session_id": "integration-test",
        "is_help_section": is_help,
        "history": [],
        "retry_count": 0,
        "from_cache": False,
        "sanitized_query": "",
        "guard_input_result": {},
        "guard_output_result": {},
        "category_probs": {},
        "dominant_category": "",
        "confidence": 0.0,
        "retrieved_chunks": [],
        "context_metadata": {},
        "retrieval_time_ms": 0.0,
        "sources_cited": [],
        "response": "",
        "token_count": 0,
        "critic_verdict": {},
        "action_result": {},
    }


@pytest.fixture
def rag_graph():
    """Build a RAGGraph with all external dependencies mocked."""
    with patch("src.agents.graph.ModelRegistry") as mock_registry_cls, \
         patch("src.agents.graph.MLPRouter") as mock_mlp_cls, \
         patch("src.agents.graph.WeightedRetriever") as mock_retriever_cls, \
         patch("src.agents.graph.SemanticCache") as mock_cache_cls:

        # ModelRegistry
        mock_registry = MagicMock()
        mock_registry.provider_name = "test-mock"
        mock_registry_cls.return_value = mock_registry

        # MLPRouter — Science-dominant routing by default
        mock_mlp = MagicMock()
        mock_mlp.route.return_value = {
            "Science": 0.80, "Software": 0.12, "User": 0.08
        }
        mock_mlp_cls.return_value = mock_mlp

        # WeightedRetriever
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = _mock_chunks("Science", "pv_yield_physics.txt")
        mock_retriever_cls.return_value = mock_retriever

        # SemanticCache — always miss
        mock_cache = MagicMock()
        mock_cache.check.return_value = None
        mock_cache.store.return_value = None
        mock_cache_cls.return_value = mock_cache

        # LLM — returns a faithful synthesis response
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=_mock_llm_response(
            "PV efficiency degrades at 0.4-0.5% per °C above 25°C.\n"
            "SOURCES: [pv_yield_physics.txt]"
        ))
        mock_registry.get_llm.return_value = mock_llm

        from src.agents.graph import RAGGraph

        graph = RAGGraph()
        # Expose mocks for per-test override
        graph._mock_mlp = mock_mlp
        graph._mock_retriever = mock_retriever
        graph._mock_llm = mock_llm
        graph._mock_cache = mock_cache
        yield graph


@pytest.mark.asyncio
async def test_science_query_routes_to_science_domain(rag_graph) -> None:
    """A science-domain query must have Science as dominant_category in the result."""
    result = await rag_graph.app.ainvoke(
        _base_inputs("How does temperature affect photovoltaic efficiency?")
    )
    assert result["dominant_category"] == "Science"
    assert result["confidence"] >= 0.5


@pytest.mark.asyncio
async def test_science_query_returns_non_empty_response(rag_graph) -> None:
    """Response must be non-empty and not the fallback 'not enough information' message."""
    result = await rag_graph.app.ainvoke(
        _base_inputs("What is the MPPT efficiency formula?")
    )
    response = result.get("response", "")
    assert len(response) > 0
    assert "don't have enough information" not in response.lower()


@pytest.mark.asyncio
async def test_help_override_applies_correct_probability_weights(rag_graph) -> None:
    """
    With is_help_section=True, category_probs must match the help override
    configuration (Software=0.85 by default).
    """
    result = await rag_graph.app.ainvoke(
        _base_inputs("Explain scientific optimization.", is_help=True)
    )
    probs = result.get("category_probs", {})
    assert probs.get("Software", 0) == pytest.approx(0.85, abs=1e-6)
    assert probs.get("User", 0) == pytest.approx(0.0, abs=1e-6)

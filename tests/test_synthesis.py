"""
Unit tests for SynthesisAgent.

Gemini is mocked via unittest.mock.patch so these tests run without an API key.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.agents.synthesis_agent import SynthesisAgent


def _make_registry(response_text: str) -> MagicMock:
    """Build a ModelRegistry mock whose get_llm().invoke() returns response_text."""
    mock_resp = MagicMock()
    mock_resp.content = response_text
    mock_resp.usage_metadata = None

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_resp

    mock_registry = MagicMock()
    mock_registry.get_llm.return_value = mock_llm
    return mock_registry


def _make_chunk(content: str, source_id: str = "test_doc.txt", category: str = "Science") -> dict:
    return {
        "content": content,
        "category": category,
        "metadata": {"source_id": source_id},
        "score": 0.9,
    }


def test_synthesis_with_chunks_returns_response() -> None:
    registry = _make_registry("PV efficiency degrades with heat.\nSOURCES: [pv_yield_physics.txt]")
    agent = SynthesisAgent(registry)

    state = {
        "query": "How does heat affect PV efficiency?",
        "sanitized_query": "How does heat affect PV efficiency?",
        "retrieved_chunks": [_make_chunk("Crystalline silicon degrades above 25°C.")],
        "context_metadata": {},
        "history": [],
        "confidence": 0.9,
    }
    result = agent.run(state)

    assert "PV efficiency" in result["response"]
    assert "pv_yield_physics.txt" in result["sources_cited"]


def test_synthesis_empty_context_returns_fallback() -> None:
    registry = _make_registry("This should never be returned.")
    agent = SynthesisAgent(registry)

    state = {
        "query": "Tell me about batteries.",
        "sanitized_query": "Tell me about batteries.",
        "retrieved_chunks": [],
        "context_metadata": {},
        "history": [],
        "confidence": 0.5,
    }
    result = agent.run(state)

    assert "don't have enough information" in result["response"]
    assert result["sources_cited"] == []
    assert result["token_count"] == 0


def test_synthesis_extracts_multiple_sources() -> None:
    response_text = (
        "The HEMS uses MPPT and WI-SUN standards.\n"
        "SOURCES: [hems_mode_strategies.txt, wi_sun_outlet_logic.txt]"
    )
    registry = _make_registry(response_text)
    agent = SynthesisAgent(registry)

    state = {
        "query": "How does HEMS work?",
        "sanitized_query": "How does HEMS work?",
        "retrieved_chunks": [
            _make_chunk("HEMS uses AI scheduling.", "hems_mode_strategies.txt", "Software"),
            _make_chunk("WI-SUN HAN spec.", "wi_sun_outlet_logic.txt", "Software"),
        ],
        "context_metadata": {},
        "history": [],
        "confidence": 0.8,
    }
    result = agent.run(state)

    assert "hems_mode_strategies.txt" in result["sources_cited"]
    assert "wi_sun_outlet_logic.txt" in result["sources_cited"]

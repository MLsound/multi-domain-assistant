"""
Unit tests for CriticAgent.

Gemini is mocked so tests run without an API key.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.critic_agent import CriticAgent
from src.agents.state import RetrievalChunk


def _make_registry(json_response: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.content = json_response
    mock_llm = MagicMock()
    # Use AsyncMock for ainvoke
    mock_llm.ainvoke = AsyncMock(return_value=mock_resp)
    mock_registry = MagicMock()
    mock_registry.get_llm.return_value = mock_llm
    return mock_registry


def _base_state() -> dict:
    return {
        "query": "What is the Weibull distribution?",
        "sanitized_query": "What is the Weibull distribution?",
        "response": "The two-parameter Weibull distribution models PV module lifespan.",
        "retrieved_chunks": [
            RetrievalChunk(
                content="PV lifespan is modelled by the two-parameter Weibull distribution.",
                metadata={"source_id": "pv_reliability_math.txt"},
                score=0.9,
                original_score=0.9,
                category="Science",
            )
        ],
        "retry_count": 0,
    }


@pytest.mark.asyncio
async def test_critic_approves_faithful_response() -> None:
    json_out = '{"approved": true, "score": 0.95, "issues": [], "suggested_refinement": null}'
    agent = CriticAgent(_make_registry(json_out))
    result = await agent.run(_base_state())
    assert result["critic_verdict"].approved is True
    assert result["critic_verdict"].score >= 0.9


@pytest.mark.asyncio
async def test_critic_rejects_unfaithful_response() -> None:
    json_out = (
        '{"approved": false, "score": 0.3, '
        '"issues": ["Response mentions beta parameter not in context"], '
        '"suggested_refinement": "Include Weibull shape and scale parameters"}'
    )
    agent = CriticAgent(_make_registry(json_out))
    result = await agent.run(_base_state())
    assert result["critic_verdict"].approved is False
    assert len(result["critic_verdict"].issues) > 0
    assert result["critic_verdict"].suggested_refinement is not None


@pytest.mark.asyncio
async def test_critic_defaults_to_approve_on_llm_failure() -> None:
    mock_llm = MagicMock()
    # Mock ainvoke with an error
    mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API unreachable"))
    mock_registry = MagicMock()
    mock_registry.get_llm.return_value = mock_llm

    agent = CriticAgent(mock_registry)
    result = await agent.run(_base_state())

    # Must default to approved (fail-open) and not raise
    assert result["critic_verdict"].approved is True
    assert result["critic_verdict"].score == 1.0

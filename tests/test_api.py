"""
Integration tests for the FastAPI endpoints.

The LLM and Qdrant are mocked so these tests run without external services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _make_mock_result(response: str = "Test response.", from_cache: bool = False) -> dict:
    return {
        "response": response,
        "sources_cited": ["test_doc.txt"],
        "category_probs": {"Science": 0.8, "Software": 0.1, "User": 0.1},
        "dominant_category": "Science",
        "confidence": 0.8,
        "retrieval_time_ms": 50.0,
        "token_count": 100,
        "retry_count": 0,
        "from_cache": from_cache,
        "guard_output_result": {"is_safe": True, "validated_response": response},
        "guard_input_result": {"is_safe": True, "rejection_reason": None},
        "critic_verdict": {"approved": True, "score": 0.9},
        "action_result": {"action_type": "json_log", "success": True, "details": "logs/audit.jsonl"},
    }


@pytest.fixture
def client():
    """Build a TestClient with mocked RAGGraph and SemanticCache."""
    with patch("src.agents.graph.ModelRegistry") as mock_registry_cls, \
         patch("src.agents.graph.MLPRouter"), \
         patch("src.agents.graph.WeightedRetriever"), \
         patch("src.agents.graph.SemanticCache") as mock_cache_cls:

        # Registry mock
        mock_registry = MagicMock()
        mock_registry.provider_name = "gemini-mock"
        mock_registry_cls.return_value = mock_registry

        # Cache mock — always miss so graph runs fully
        mock_cache = MagicMock()
        mock_cache.check.return_value = None
        mock_cache.store.return_value = None
        mock_cache_cls.return_value = mock_cache

        # LLM mock
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Mocked synthesis response.\nSOURCES: [test_doc.txt]"
        mock_resp.usage_metadata = None
        mock_llm.invoke.return_value = mock_resp
        mock_registry.get_llm.return_value = mock_llm

        from src.api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_health_endpoint_returns_200(client: TestClient) -> None:
    with patch("src.api.main._rag_system") as mock_system, \
         patch("qdrant_client.QdrantClient") as mock_qdrant:
        mock_system.provider_name = "gemini-mock"
        mock_qdrant.return_value.get_collections.return_value = MagicMock(collections=[])

        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "qdrant_connected" in data
    assert "active_provider" in data


def test_metrics_endpoint_returns_valid_json(client: TestClient) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "avg_latency_ms" in data
    assert "cache_hit_rate" in data


def test_empty_query_rejected_by_pydantic(client: TestClient) -> None:
    """FastAPI/Pydantic enforces min_length=1 before the guard even runs."""
    resp = client.post("/query", json={"query": ""})
    assert resp.status_code == 422


def test_query_too_long_rejected(client: TestClient) -> None:
    long_query = "x" * 2001
    resp = client.post("/query", json={"query": long_query})
    assert resp.status_code == 422

"""
Unit tests for ActionAgent.

Tests file writing (always active) and webhook failure graceful handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.action_agent import ActionAgent
from src.agents.state import CriticVerdict


def _base_state() -> dict:
    return {
        "session_id": "test-session-001",
        "query": "What is MPPT?",
        "dominant_category": "Science",
        "confidence": 0.85,
        "category_probs": {"Science": 0.85, "Software": 0.10, "User": 0.05},
        "retrieval_time_ms": 120.5,
        "token_count": 350,
        "sources_cited": ["mppt_efficiency_physics.txt"],
        "critic_verdict": CriticVerdict(score=0.9, approved=True, issues=[], suggested_refinement=None),
        "retry_count": 0,
        "from_cache": False,
        "response": "MPPT stands for Maximum Power Point Tracking.",
    }


@pytest.mark.asyncio
async def test_action_writes_audit_log(tmp_path: Path) -> None:
    """Audit log file must be created and contain valid JSON."""
    log_file = tmp_path / "audit.jsonl"

    with patch("src.agents.action_agent.settings") as mock_settings:
        mock_settings.audit_log_path = str(log_file)
        mock_settings.webhook_url = None

        agent = ActionAgent()
        result = await agent.execute(_base_state())

    assert result["action_result"].success is True
    assert log_file.exists()
    with open(log_file, encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert record["query"] == "What is MPPT?"
    assert record["dominant_category"] == "Science"


@pytest.mark.asyncio
async def test_action_succeeds_without_webhook(tmp_path: Path) -> None:
    """When WEBHOOK_URL is not configured, action still succeeds via log."""
    log_file = tmp_path / "audit.jsonl"

    with patch("src.agents.action_agent.settings") as mock_settings:
        mock_settings.audit_log_path = str(log_file)
        mock_settings.webhook_url = None

        agent = ActionAgent()
        result = await agent.execute(_base_state())

    assert result["action_result"].success is True
    assert log_file.exists()


@pytest.mark.asyncio
async def test_action_handles_webhook_failure_gracefully(tmp_path: Path) -> None:
    """A failing webhook must not prevent successful log write or raise."""
    log_file = tmp_path / "audit.jsonl"

    with patch("src.agents.action_agent.settings") as mock_settings:
        mock_settings.audit_log_path = str(log_file)
        mock_settings.webhook_url = "http://localhost:0/nonexistent"

        agent = ActionAgent()
        # Should not raise even though the webhook will fail
        result = await agent.execute(_base_state())

    # Log write still succeeded
    assert result["action_result"].success is True
    assert log_file.exists()

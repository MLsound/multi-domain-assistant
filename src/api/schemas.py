"""
Pydantic schemas for the FastAPI endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query text")
    session_id: Optional[str] = Field(None, description="Conversation session identifier")
    is_help_override: bool = Field(
        False,
        description="Force Software/Science domain weighting (help-section mode)",
    )


class QueryResponse(BaseModel):
    response: str
    sources_cited: List[str] = Field(default_factory=list)
    category_probs: Dict[str, float] = Field(default_factory=dict)
    dominant_category: str = ""
    confidence: float = 0.0
    retrieval_time_ms: float = 0.0
    token_count: int = 0
    retry_count: int = 0
    from_cache: bool = False
    # Security telemetry surfaced to the client (helps the demo presentation)
    injection_score: float = 0.0
    injection_decision: str = "allow"
    pii_redacted_count: int = 0


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    active_provider: str
    timestamp: str


class QueryRecordOut(BaseModel):
    id: int
    session_id: str
    query: str
    response_preview: str
    dominant_category: str
    confidence: float
    latency_ms: float
    blocked_by_guard: bool
    created_at: datetime

    model_config = {"from_attributes": True}

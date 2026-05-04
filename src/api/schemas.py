"""
Pydantic schemas for the FastAPI endpoints.
"""

from __future__ import annotations

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
    response: str = Field(..., description="Generated answer")
    sources_cited: List[str] = Field(default_factory=list, description="Source document IDs used")
    category_probs: Dict[str, float] = Field(
        default_factory=dict, description="Domain probability distribution"
    )
    dominant_category: str = Field("", description="Highest-probability domain")
    confidence: float = Field(0.0, description="Router confidence score")
    retrieval_time_ms: float = Field(0.0, description="Qdrant retrieval latency in ms")
    token_count: int = Field(0, description="LLM tokens consumed")
    retry_count: int = Field(0, description="Number of critic-triggered retries")
    from_cache: bool = Field(False, description="True if response was served from semantic cache")


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' or 'degraded'")
    qdrant_connected: bool
    active_provider: str = Field(..., description="Active LLM provider name")

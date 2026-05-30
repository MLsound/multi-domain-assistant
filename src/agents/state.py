"""
LangGraph state schema for the Knowledge Assistant graph.

Every field written by any agent or graph node is declared here.
LangGraph requires all state keys to be declared upfront.

The `history` field uses operator.add so that each turn's messages are
appended rather than overwritten when multiple nodes update it.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ConversationTurn(BaseModel):
    query: str
    response: str


class GuardResult(BaseModel):
    is_safe: bool
    rejection_reason: Optional[str] = None
    validated_response: Optional[str] = None
    injection_score: Optional[float] = None
    injection_decision: Optional[str] = None
    pii_detections: List[str] = Field(default_factory=list)
    pii_redacted_on_output: List[str] = Field(default_factory=list)


class RetrievalChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float
    original_score: float
    category: str
    rerank_score: Optional[float] = None


class CriticVerdict(BaseModel):
    approved: bool
    score: float
    issues: List[str] = Field(default_factory=list)
    suggested_refinement: Optional[str] = None


class ActionResult(BaseModel):
    action_type: str
    success: bool
    details: str


class GraphState(TypedDict):
    # ------------------------------------------------------------------
    # Input fields (populated before graph entry)
    # ------------------------------------------------------------------
    query: str
    is_help_section: bool
    session_id: Optional[str]
    history: Annotated[List[ConversationTurn], operator.add]

    # ------------------------------------------------------------------
    # Guard (input phase) output
    # ------------------------------------------------------------------
    sanitized_query: str
    guard_input_result: GuardResult

    # ------------------------------------------------------------------
    # Guard (output phase) output
    # ------------------------------------------------------------------
    guard_output_result: GuardResult

    # ------------------------------------------------------------------
    # Router Agent output
    # ------------------------------------------------------------------
    category_probs: Dict[str, float]
    dominant_category: str
    confidence: float

    # ------------------------------------------------------------------
    # Retrieval Agent output
    # ------------------------------------------------------------------
    retrieved_chunks: List[RetrievalChunk]
    context_metadata: Dict[str, Any]
    retrieval_time_ms: float
    sources_cited: List[str]

    # ------------------------------------------------------------------
    # Synthesis Agent output
    # ------------------------------------------------------------------
    response: str
    token_count: int

    # ------------------------------------------------------------------
    # Critic Agent output
    # ------------------------------------------------------------------
    critic_verdict: CriticVerdict
    retry_count: int

    # ------------------------------------------------------------------
    # Semantic cache flag
    # ------------------------------------------------------------------
    from_cache: bool

    # ------------------------------------------------------------------
    # Action Agent output
    # ------------------------------------------------------------------
    action_result: ActionResult

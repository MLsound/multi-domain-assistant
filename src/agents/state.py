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

from typing_extensions import TypedDict


class GraphState(TypedDict):
    # ------------------------------------------------------------------
    # Input fields (populated before graph entry)
    # ------------------------------------------------------------------
    query: str
    is_help_section: bool
    session_id: Optional[str]
    history: Annotated[List[Dict[str, str]], operator.add]

    # ------------------------------------------------------------------
    # Guard (input phase) output
    # ------------------------------------------------------------------
    sanitized_query: str
    # {is_safe: bool, rejection_reason: str | None}
    guard_input_result: Dict[str, Any]

    # ------------------------------------------------------------------
    # Guard (output phase) output
    # ------------------------------------------------------------------
    # {is_safe: bool, validated_response: str}
    guard_output_result: Dict[str, Any]

    # ------------------------------------------------------------------
    # Router Agent output
    # ------------------------------------------------------------------
    category_probs: Dict[str, float]
    dominant_category: str
    confidence: float

    # ------------------------------------------------------------------
    # Retrieval Agent output
    # ------------------------------------------------------------------
    retrieved_chunks: List[Dict[str, Any]]
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
    # {approved: bool, score: float, issues: list, suggested_refinement: str|None}
    critic_verdict: Dict[str, Any]
    retry_count: int

    # ------------------------------------------------------------------
    # Semantic cache flag
    # ------------------------------------------------------------------
    from_cache: bool

    # ------------------------------------------------------------------
    # Action Agent output
    # ------------------------------------------------------------------
    # {action_type: str, success: bool, details: str}
    action_result: Dict[str, Any]

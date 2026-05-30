"""
RAGGraph — Full LangGraph StateGraph with conditional edges and retry loop.

Topology:
  cache_check → (HIT: END | MISS: guard_input)
  guard_input → (safe: router | blocked: END)
  router → retrieval → synthesis → critic
  critic → (approved/force: guard_output | retry: retrieval)
  guard_output → action → END

The critic feedback loop is the dynamic inter-agent communication mechanism
satisfying FR-02.  The Critic's verdict (approved + suggested_refinement)
drives whether Retrieval re-executes with a refined query.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from src.agents.action_agent import ActionAgent
from src.agents.critic_agent import CriticAgent
from src.agents.guard_agent import GuardAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.router_agent import RouterAgent
from src.agents.state import GraphState, CriticVerdict, GuardResult
from src.agents.synthesis_agent import SynthesisAgent
from src.cache.semantic_cache import SemanticCache
from src.config.model_registry import ModelRegistry
from src.config.settings import settings
from src.retrieval.weighted_retriever import WeightedRetriever
from src.router.mlp_router import MLPRouter

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def _cache_decision(state: GraphState) -> Literal["hit", "miss"]:
    """Short-circuit to END if a cache hit was recorded."""
    return "hit" if state.get("from_cache") else "miss"


def _guard_decision(state: GraphState) -> Literal["proceed", "blocked"]:
    """Stop the pipeline early if the input guard rejected the query."""
    result = state.get("guard_input_result")
    return "proceed" if result and result.is_safe else "blocked"


def _critic_decision(
    state: GraphState,
) -> Literal["approve", "retry"]:
    """Drive the retry loop: retry retrieval or proceed to output guard."""
    verdict = state.get("critic_verdict")
    retry_count: int = state.get("retry_count", 0)

    if retry_count >= settings.max_retries:
        logger.info("Critic: max_retries reached — force-approving")
        return "approve"

    return "approve" if (verdict and verdict.approved) else "retry"


# ---------------------------------------------------------------------------
# Node wrappers
# ---------------------------------------------------------------------------

def _make_cache_check_node(cache: SemanticCache):
    """Return a node function that checks the semantic cache."""

    def cache_check(state: GraphState) -> Dict[str, Any]:
        query = state.get("query", "")
        hit = cache.check(query)
        if hit:
            logger.info("Cache HIT — returning cached response")
            return {
                "response": hit.get("response", ""),
                "sources_cited": hit.get("sources_cited", []),
                "from_cache": True,
                "category_probs": {},
                "dominant_category": "",
                "confidence": 0.0,
                "retrieval_time_ms": 0.0,
                "token_count": 0,
                "retry_count": 0,
                "critic_verdict": CriticVerdict(
                    approved=True,
                    score=1.0,
                    issues=[],
                    suggested_refinement=None
                ),
                "guard_output_result": GuardResult(
                    is_safe=True,
                    validated_response=hit.get("response", "")
                ),
            }
        return {"from_cache": False}

    return cache_check


def _make_guard_input_node(guard: GuardAgent):
    def node(state: GraphState) -> Dict[str, Any]:
        result = guard.validate_input(state)
        guard_result = result.get("guard_input_result")
        if guard_result and not guard_result.is_safe:
            reason = guard_result.rejection_reason or "Query rejected"
            result["response"] = f"Request blocked: {reason}"
        return result
    return node


_UNCACHEABLE_RESPONSES = {
    "service temporarily unavailable. please try again.",
    "i don't have enough information in my knowledge base to answer this question.",
    "no response was generated.",
    "response could not be delivered safely.",
}


def _make_guard_output_node(guard: GuardAgent, cache: SemanticCache):
    def node(state: GraphState) -> Dict[str, Any]:
        result = guard.validate_output(state)
        guard_result = result.get("guard_output_result")
        validated = guard_result.validated_response if guard_result else ""

        # Only cache safe, substantive responses with at least one cited source.
        # Never cache fallback / error messages — doing so poisons the cache and
        # causes every repeated query to return the error instead of retrying.
        should_cache = (
            guard_result and guard_result.is_safe
            and not state.get("from_cache")
            and bool(state.get("sources_cited"))
            and validated.strip().lower() not in _UNCACHEABLE_RESPONSES
            and len(validated.strip()) > 50
        )
        if should_cache:
            try:
                cache.store(
                    state.get("query", ""),
                    validated,
                    state.get("sources_cited", []),
                )
            except Exception:
                logger.warning("Cache store failed after output validation")
        return result
    return node


def _make_critic_node(critic: CriticAgent):
    """Wrap CriticAgent.run() and increment retry_count on rejection."""

    def node(state: GraphState) -> Dict[str, Any]:
        verdict_result = critic.run(state)
        verdict = verdict_result.get("critic_verdict")
        retry_count = state.get("retry_count", 0)
        if verdict and not verdict.approved and retry_count < settings.max_retries:
            return {**verdict_result, "retry_count": retry_count + 1}
        return verdict_result

    return node


# ---------------------------------------------------------------------------
# RAGGraph
# ---------------------------------------------------------------------------

class RAGGraph:
    """
    Compiled LangGraph StateGraph implementing the full Knowledge Assistant pipeline.

    Graph topology (Phase 3):
      cache_check → guard_input → router → retrieval → synthesis → critic
                 → guard_output → action → END
    with conditional short-circuits and the critic retry loop.
    """

    def __init__(self, mlp_model_path: str | None = None) -> None:
        path = mlp_model_path or settings.mlp_model_path

        # Shared infrastructure
        self._registry = ModelRegistry()
        mlp_router = MLPRouter(path)
        retriever = WeightedRetriever()
        cache = SemanticCache()

        # Agents
        guard = GuardAgent()
        router_agent = RouterAgent(mlp_router)
        retrieval_agent = RetrievalAgent(retriever)
        synthesis_agent = SynthesisAgent(self._registry)
        critic_agent = CriticAgent(self._registry)
        action_agent = ActionAgent()

        # Build graph
        workflow = StateGraph(GraphState)

        workflow.add_node("cache_check", _make_cache_check_node(cache))
        workflow.add_node("guard_input", _make_guard_input_node(guard))
        workflow.add_node("router", router_agent.run)
        workflow.add_node("retrieval", retrieval_agent.run)
        workflow.add_node("synthesis", synthesis_agent.run)
        workflow.add_node("critic", _make_critic_node(critic_agent))
        workflow.add_node("guard_output", _make_guard_output_node(guard, cache))
        workflow.add_node("action", action_agent.execute)

        # Entry point
        workflow.set_entry_point("cache_check")

        # Conditional: cache hit → END, miss → guard_input
        workflow.add_conditional_edges(
            "cache_check",
            _cache_decision,
            {"hit": END, "miss": "guard_input"},
        )

        # Conditional: guard passes → router, blocked → END
        workflow.add_conditional_edges(
            "guard_input",
            _guard_decision,
            {"proceed": "router", "blocked": END},
        )

        # Linear: router → retrieval → synthesis → critic
        workflow.add_edge("router", "retrieval")
        workflow.add_edge("retrieval", "synthesis")
        workflow.add_edge("synthesis", "critic")

        # Conditional: critic approves → guard_output, retry → retrieval
        workflow.add_conditional_edges(
            "critic",
            _critic_decision,
            {"approve": "guard_output", "retry": "retrieval"},
        )

        # Linear: guard_output → action → END
        workflow.add_edge("guard_output", "action")
        workflow.add_edge("action", END)

        self.app = workflow.compile()
        logger.info("RAGGraph compiled (Phase 3 full topology with conditional edges)")

    @property
    def provider_name(self) -> str:
        return self._registry.provider_name

"""
Evaluation metrics utilities.

Provides:
  - timing_decorator: wraps agent run() methods to record latency
  - count_tokens: safely extracts total token count from LLM responses
  - semantic_similarity: cosine similarity between two text strings
  - compute_rouge_l: ROUGE-L F1 score between hypothesis and reference
  - compute_context_precision: async RAGAS Context Precision via LLM judge
  - compute_answer_relevancy:  async RAGAS Answer Relevancy via LLM judge
  - compute_tool_metrics:      deterministic per-agent tool-call success flags
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Sentinel value returned when a metric cannot be computed
METRIC_UNAVAILABLE = -1.0


def timing_decorator(agent_name: str):
    """
    Decorator factory — records execution latency of any agent run() method.

    Usage:
        @timing_decorator("router")
        def run(self, state): ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.debug("[timing] %s: %.1f ms", agent_name, latency_ms)
            # Inject latency into result dict if it is a dict
            if isinstance(result, dict):
                result[f"_{agent_name}_latency_ms"] = round(latency_ms, 2)
            return result

        return wrapper

    return decorator


def count_tokens(response_obj: Any) -> int:
    """Safely extract total token count from a LangChain response object."""
    try:
        meta = getattr(response_obj, "usage_metadata", None)
        if meta is None:
            return 0
        if isinstance(meta, dict):
            return int(meta.get("total_tokens", 0))
        return int(getattr(meta, "total_tokens", 0))
    except Exception:
        return 0


def semantic_similarity(text_a: str, text_b: str, model) -> float:
    """
    Compute cosine similarity between two texts using a sentence-transformer model.

    Args:
        text_a: First text (e.g. generated response).
        text_b: Second text (e.g. ground truth).
        model:  A SentenceTransformer instance.

    Returns:
        Cosine similarity score in [0, 1].
    """
    import numpy as np

    try:
        emb_a = model.encode(text_a, convert_to_numpy=True)
        emb_b = model.encode(text_b, convert_to_numpy=True)

        # Normalise and compute dot product
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
    except Exception:
        logger.exception("semantic_similarity failed")
        return 0.0


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between hypothesis (generated) and reference (ground truth).

    Returns 0.0 on failure.
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return float(scores["rougeL"].fmeasure)
    except Exception:
        logger.exception("compute_rouge_l failed")
        return 0.0


# ---------------------------------------------------------------------------
# P1 — New RAGAS-based metrics (Context Precision, Answer Relevancy)
# ---------------------------------------------------------------------------

async def _robust_ascore(scorer, *, max_retries: int = 2, timeout: float = 15.0, **kwargs) -> float:
    """
    Call scorer.ascore(**kwargs) with exponential-backoff retry and timeout.

    Returns METRIC_UNAVAILABLE (-1.0) if all attempts fail.
    """
    for attempt in range(max_retries + 1):
        try:
            result = await asyncio.wait_for(scorer.ascore(**kwargs), timeout=timeout)
            return float(result.value) if hasattr(result, "value") else float(result)
        except asyncio.TimeoutError:
            logger.warning(
                "Metric %s timed out after %ss (attempt %d/%d)",
                scorer.__class__.__name__, timeout, attempt + 1, max_retries + 1,
            )
        except Exception:
            logger.exception(
                "Metric %s failed (attempt %d/%d)",
                scorer.__class__.__name__, attempt + 1, max_retries + 1,
            )
        if attempt < max_retries:
            await asyncio.sleep(2 ** attempt)
    return METRIC_UNAVAILABLE


# Module-level semaphore to limit concurrent LLM judge calls (Groq rate-limit safety)
# Created once with the first max_concurrent value seen; subsequent calls with a
# different max_concurrent will reuse the existing semaphore (single-creation).
_metric_semaphore: asyncio.Semaphore | None = None


def _get_metric_semaphore(max_concurrent: int = 5) -> asyncio.Semaphore:
    global _metric_semaphore
    if _metric_semaphore is None:
        _metric_semaphore = asyncio.Semaphore(max_concurrent)
    return _metric_semaphore


async def compute_context_precision(
    evaluator_llm,
    *,
    user_input: str,
    reference: str,
    retrieved_contexts: list[str],
    max_concurrent: int = 5,
) -> float:
    """
    Compute RAGAS Context Precision: are relevant contexts ranked higher?

    Delegates to ``ragas.metrics.collections.ContextPrecision.ascore()``.

    Returns METRIC_UNAVAILABLE (-1.0) on failure.
    """
    try:
        from ragas.metrics.collections import ContextPrecision
        from ragas.llms import LangchainLLMWrapper

        wrapped = LangchainLLMWrapper(evaluator_llm)
        scorer = ContextPrecision(llm=wrapped)
        sem = _get_metric_semaphore(max_concurrent)
        async with sem:
            return await _robust_ascore(
                scorer,
                user_input=user_input,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
    except ImportError:
        logger.warning("ragas.metrics.collections not available — context_precision skipped")
        return METRIC_UNAVAILABLE


async def compute_answer_relevancy(
    evaluator_llm,
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
    max_concurrent: int = 5,
) -> float:
    """
    Compute RAGAS Answer Relevancy: does the response address the question?

    Delegates to ``ragas.metrics.collections.AnswerRelevancy.ascore()``.

    Returns METRIC_UNAVAILABLE (-1.0) on failure.
    """
    try:
        from ragas.metrics.collections import AnswerRelevancy
        from ragas.llms import LangchainLLMWrapper

        wrapped = LangchainLLMWrapper(evaluator_llm)
        scorer = AnswerRelevancy(llm=wrapped)
        sem = _get_metric_semaphore(max_concurrent)
        async with sem:
            return await _robust_ascore(
                scorer,
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )
    except ImportError:
        logger.warning("ragas.metrics.collections not available — answer_relevancy skipped")
        return METRIC_UNAVAILABLE


# ---------------------------------------------------------------------------
# P1 — Deterministic tool-call success metrics (no LLM judge needed)
# ---------------------------------------------------------------------------

def compute_tool_metrics(result: dict) -> dict[str, float]:
    """
    Compute four deterministic tool-call success flags from the graph result.

    Returns
    -------
    dict with keys:
        tool_router_success     — pipeline produced a non-empty dominant_category  (0/1)
        tool_retrieval_success  — non-empty retrieved_chunks + positive time       (0/1)
        tool_critic_success     — no retries needed, or critic approved            (0/1)
        tool_action_success     — action_result.audit_success / webhook_success    (0/1)
    """
    flags: dict[str, float] = {}

    # 1. Router success: did the pipeline produce a dominant category?
    dominant = result.get("dominant_category", "")
    flags["tool_router_success"] = 1.0 if dominant else 0.0

    # 2. Retrieval success: non-empty chunks + positive retrieval time
    chunks = result.get("retrieved_chunks", [])
    ret_time = result.get("retrieval_time_ms", 0.0)
    flags["tool_retrieval_success"] = 1.0 if (len(chunks) > 0 and ret_time > 0.0) else 0.0

    # 3. Critic success: retry_count == 0 means first-pass approval
    retry_count = result.get("retry_count", -1)
    if retry_count == 0:
        flags["tool_critic_success"] = 1.0
    elif retry_count == -1:
        flags["tool_critic_success"] = 0.0  # no critic data
    else:
        # Retries were used but a response was produced -> partial credit
        flags["tool_critic_success"] = 0.5

    # 4. Action success: audit write succeeded
    action = result.get("action_result", {}) or {}
    audit_ok = action.get("audit_success", False)
    flags["tool_action_success"] = 1.0 if audit_ok else 0.0

    return flags

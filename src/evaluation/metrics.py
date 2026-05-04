"""
Evaluation metrics utilities.

Provides:
  - timing_decorator: wraps agent run() methods to record latency
  - count_tokens: safely extracts total token count from LLM responses
  - semantic_similarity: cosine similarity between two text strings
  - compute_rouge_l: ROUGE-L F1 score between hypothesis and reference
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


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

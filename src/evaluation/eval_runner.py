"""
Evaluation runner for the Knowledge Assistant.

Loads test cases from data/eval/test_suite.json, runs inference via RAGGraph,
and computes per-question metrics:
  - faithfulness     (Ragas 0.2 — LLM-based)
  - context_recall   (Ragas 0.2 — LLM-based)
  - precision_at_k   (fraction of top-k chunks from expected domain)
  - retrieval_time_ms
  - total_latency_ms
  - token_count
  - semantic_similarity (cosine via bge-large-en-v1.5)
  - rouge_l

Compatible with ragas>=0.2.15. The 0.1 wrappers (BaseRagasLLM,
BaseRagasEmbeddings) have been removed; ragas 0.2 accepts any LangChain
BaseChatModel directly via LangchainLLMWrapper.

The evaluator LLM is selected from the project's ModelRegistry so the same
provider used for synthesis (Groq, Gemini, etc.) is also used for evaluation.
GOOGLE_API_KEY is no longer required — any configured provider works.

MLflow integration: each evaluation run is tracked as an MLflow run with
parameters, per-question metrics, aggregate metrics, and the results JSON
logged as an artifact.

Output: reports/evaluation_results.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def _load_test_suite(path: str) -> List[Dict[str, Any]]:
    suite_path = Path(path)
    if not suite_path.exists():
        raise FileNotFoundError(f"Test suite not found: {suite_path}")
    with open(suite_path, encoding="utf-8") as f:
        return json.load(f)


def _precision_at_k(chunks: List[Any], expected_domain: str, k: int = 5) -> float:
    """Fraction of top-k chunks whose category matches the expected domain."""
    top_k = chunks[:k]
    if not top_k:
        return 0.0
    matches = sum(1 for c in top_k if c.category == expected_domain)
    return matches / len(top_k)


async def run_evaluation(
    test_suite_path: str = "data/eval/test_suite.json",
    output_path: str = "reports/evaluation_results.json",
) -> Dict[str, Any]:
    """Run the full evaluation suite and write results to output_path."""

    # --- Ragas 0.2 imports ---
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import context_recall, faithfulness
    from ragas.run_config import RunConfig
    from sentence_transformers import SentenceTransformer

    from src.agents.graph import RAGGraph
    from src.config.mlflow_config import manager as mlflow_manager
    from src.config.model_registry import ModelRegistry
    from src.config.settings import settings
    from src.evaluation.metrics import compute_rouge_l, semantic_similarity

    # --- Initialise the RAG system and evaluator LLM ---
    # Use the same provider as the main pipeline (set by .env).
    registry = ModelRegistry()
    evaluator_langchain_llm = registry.get_llm(complexity="simple")

    # Ragas 0.2 wraps any LangChain BaseChatModel directly.
    wrapped_llm = LangchainLLMWrapper(evaluator_langchain_llm)

    sim_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    rag_system = RAGGraph()

    # --- Load test suite ---
    def _load_test_suite(path: str) -> List[Dict[str, Any]]:
        suite_path = Path(path)
        if not suite_path.exists():
            raise FileNotFoundError(f"Test suite not found: {suite_path}")
        with open(suite_path, encoding="utf-8") as f:
            return json.load(f)

    test_cases = _load_test_suite(test_suite_path)
    logger.info("Loaded %d test cases from %s", len(test_cases), test_suite_path)

    # --- Start MLflow run for this evaluation ---
    with mlflow_manager.start_run(run_name=f"eval-{int(time.time())}") as run:
        mlflow_manager.log_params({
            "provider": registry.provider_name,
            "test_suite_path": test_suite_path,
            "n_questions": len(test_cases),
            "max_retries": settings.max_retries,
            "cache_similarity_threshold": settings.cache_similarity_threshold,
            "router_confidence_threshold": settings.router_confidence_threshold,
            "science_threshold": settings.science_threshold,
        })

        print(f"--- Running inference on {len(test_cases)} questions ---")

        per_question_metrics: List[Dict] = []
        ragas_samples: List[SingleTurnSample] = []

        for i, case in enumerate(test_cases):
            question = case["question"]
            expected_domain = case.get("expected_domain", "")
            ground_truth = case.get("ground_truth", "")
            is_help = case.get("is_help", False)

            print(f"[{i + 1}/{len(test_cases)}] {question[:80]}")

            inputs = {
                "query": question,
                "session_id": "eval",
                "is_help_section": is_help,
                "history": [],
                "retry_count": 0,
                "from_cache": False,
                "sanitized_query": "",
                "category_probs": {},
                "dominant_category": "",
                "confidence": 0.0,
                "retrieved_chunks": [],
                "context_metadata": {},
                "retrieval_time_ms": 0.0,
                "sources_cited": [],
                "response": "",
                "token_count": 0,
            }

            t0 = time.perf_counter()
            result = await rag_system.app.ainvoke(inputs)
            total_latency_ms = (time.perf_counter() - t0) * 1000

            answer = result.get("response", "")
            chunks = result.get("retrieved_chunks", [])
            context_texts = [c.content for c in chunks]

            sim_score = semantic_similarity(answer, ground_truth, sim_model) if ground_truth else None
            rouge = compute_rouge_l(answer, ground_truth) if ground_truth else None
            prec_k = _precision_at_k(chunks, expected_domain) if expected_domain else None

            per_question_metrics.append({
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "expected_domain": expected_domain,
                "dominant_category": result.get("dominant_category", ""),
                "confidence": result.get("confidence", 0.0),
                "retrieval_time_ms": result.get("retrieval_time_ms", 0.0),
                "total_latency_ms": round(total_latency_ms, 2),
                "token_count": result.get("token_count", 0),
                "semantic_similarity": round(sim_score, 4) if sim_score is not None else None,
                "rouge_l": round(rouge, 4) if rouge is not None else None,
                "precision_at_k": round(prec_k, 4) if prec_k is not None else None,
                "sources_cited": result.get("sources_cited", []),
                "retry_count": result.get("retry_count", 0),
            })

            # Build ragas 0.2 SingleTurnSample
            ragas_samples.append(SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=context_texts if context_texts else [""],
                reference=ground_truth if ground_truth else "",
            ))

            if i < len(test_cases) - 1:
                logger.debug("Sleeping 3s between queries...")
                await asyncio.sleep(3)

        # --- Ragas 0.2 evaluation ---
        eval_dataset = EvaluationDataset(samples=ragas_samples)
        run_cfg = RunConfig(max_workers=1, timeout=300)

        ragas_scores_per_q: List[Dict] = []
        try:
            print("--- Running Ragas faithfulness + context_recall ---")
            ragas_result = evaluate(
                dataset=eval_dataset,
                metrics=[faithfulness, context_recall],
                llm=wrapped_llm,
                run_config=run_cfg,
            )
            # ragas 0.2 returns a dict-like EvaluationResult; convert to list of dicts
            ragas_df = ragas_result.to_pandas()
            ragas_scores_per_q = ragas_df.to_dict(orient="records")
        except Exception:
            logger.exception(
                "Ragas evaluation failed — continuing with non-LLM metrics only"
            )

        # Merge Ragas scores into per-question metrics
        for i, row in enumerate(ragas_scores_per_q):
            if i < len(per_question_metrics):
                per_question_metrics[i]["faithfulness"] = row.get("faithfulness")
                per_question_metrics[i]["context_recall"] = row.get("context_recall")

        # --- Aggregate ---
        def _avg(key: str) -> float | None:
            vals = [r[key] for r in per_question_metrics if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        summary = {
            "n_questions": len(per_question_metrics),
            "provider": registry.provider_name,
            "avg_faithfulness": _avg("faithfulness"),
            "avg_context_recall": _avg("context_recall"),
            "avg_semantic_similarity": _avg("semantic_similarity"),
            "avg_rouge_l": _avg("rouge_l"),
            "avg_precision_at_k": _avg("precision_at_k"),
            "avg_retrieval_time_ms": _avg("retrieval_time_ms"),
            "avg_total_latency_ms": _avg("total_latency_ms"),
            "avg_token_count": _avg("token_count"),
        }

        output = {"summary": summary, "per_question": per_question_metrics}

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info("Evaluation results written to %s", out_path)

        # --- Log to MLflow ---
        mlflow_manager.log_metrics({
            k: v for k, v in summary.items()
            if v is not None and isinstance(v, (int, float))
        })

        for i, pm in enumerate(per_question_metrics):
            mlflow_manager.log_metrics({
                f"q{i}.{k}": v for k, v in pm.items()
                if v is not None and isinstance(v, (int, float))
            })

        mlflow_manager.log_artifact(str(out_path))

        print("\n--- Evaluation Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    return output

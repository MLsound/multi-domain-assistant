"""
Evaluation runner for the Knowledge Assistant.

Loads test cases from data/eval/test_suite.json, runs inference via RAGGraph,
and computes per-question metrics:
  - faithfulness       (Ragas 0.2 — LLM-based)
  - context_recall     (Ragas 0.2 — LLM-based)
  - context_precision  (Ragas 0.2 — LLM-based, async)
  - answer_relevancy   (Ragas 0.2 — LLM-based, async)
  - precision_at_k     (fraction of top-k chunks from expected domain)
  - retrieval_time_ms
  - total_latency_ms
  - token_count
  - semantic_similarity (cosine via bge-large-en-v1.5)
  - rouge_l
  - tool_*_success     (4 deterministic per-agent flags)

Compatible with ragas>=0.2.15. The evaluator LLM is selected from the project's
ModelRegistry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from src.config.settings import settings

load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

METRIC_UNAVAILABLE = -1.0


def _precision_at_k(chunks: List[Any], expected_domain: str, k: int = 5) -> float:
    """Fraction of top-k chunks whose category matches the expected domain."""
    top_k = chunks[:k]
    if not top_k:
        return 0.0
    # chunks might be dicts (from result) or objects (from state)
    def get_cat(c):
        return c.get("category") if isinstance(c, dict) else getattr(c, "category", "")

    matches = sum(1 for c in top_k if get_cat(c) == expected_domain)
    return matches / len(top_k)


def _avg(key: str, records: List[Dict]) -> float | None:
    vals = [r[key] for r in records if r.get(key) is not None and r[key] != METRIC_UNAVAILABLE]
    return round(sum(vals) / len(vals), 4) if vals else None


def _baseline_summary(summary: Dict[str, Any]) -> list[str]:
    baseline_path = Path("reports/eval_baseline.json")
    recommendations: list[str] = []

    if not baseline_path.exists():
        recommendations.append("No baseline found — this run will become the new baseline.")
        return recommendations

    try:
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)
    except Exception:
        recommendations.append("Could not read baseline file — skipping comparison.")
        return recommendations

    DROPS = {
        "avg_context_precision": (0.05, "Context Precision dropped >5% → increase retrieval_top_k or review embedding model."),
        "avg_answer_relevancy": (0.05, "Answer Relevancy dropped >5% → review prompt template or increase retrieval_final_k."),
        "avg_tool_retrieval_success": (0.05, "Retrieval success dropped >5% → check Qdrant connectivity / chunk overlap."),
        "avg_tool_router_success": (0.05, "Router success dropped >5% → review MLP training data or lower router_confidence_threshold."),
        "avg_tool_critic_success": (0.05, "Critic success dropped >5% → increase max_retries or lower critic_approval_threshold."),
    }

    for key, (threshold, msg) in DROPS.items():
        curr = summary.get(key)
        prev = baseline.get(key)
        if curr is not None and prev is not None and prev != METRIC_UNAVAILABLE:
            if prev > 0 and (prev - curr) / prev > threshold:
                recommendations.append(msg)

    if not recommendations:
        recommendations.append("No significant regressions detected against baseline.")
    return recommendations


async def _score_new_metrics_async(
    evaluator_llm,
    per_question_data: list[dict],
    max_concurrent: int = 5,
) -> list[dict]:
    from src.evaluation.metrics import (
        compute_answer_relevancy,
        compute_context_precision,
    )

    async def _score_one(item: dict) -> dict:
        cp = await compute_context_precision(
            evaluator_llm,
            user_input=item["user_input"],
            reference=item["reference"],
            retrieved_contexts=item["retrieved_contexts"],
            max_concurrent=max_concurrent,
        )
        ar = await compute_answer_relevancy(
            evaluator_llm,
            user_input=item["user_input"],
            response=item["response"],
            retrieved_contexts=item["retrieved_contexts"],
            max_concurrent=max_concurrent,
        )
        item["context_precision"] = cp
        item["answer_relevancy"] = ar
        return item

    tasks = [_score_one(item) for item in per_question_data]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final: list[dict] = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error("Async metric scoring failed for question %d: %s", i, res)
            item = per_question_data[i].copy()
            item["context_precision"] = METRIC_UNAVAILABLE
            item["answer_relevancy"] = METRIC_UNAVAILABLE
            final.append(item)
        else:
            final.append(res)
    return final


async def run_evaluation(
    test_suite_path: str = "data/eval/test_suite.json",
    output_path: str = "reports/evaluation_results.json",
) -> Dict[str, Any]:
    """Run the full evaluation suite and write results to output_path."""

    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import context_recall, faithfulness
    from ragas.run_config import RunConfig
    from sentence_transformers import SentenceTransformer

    from src.agents.graph import RAGGraph
    from src.config.mlflow_config import manager as mlflow_manager
    from src.config.model_registry import ModelRegistry
    from src.evaluation.metrics import (
        compute_rouge_l,
        compute_tool_metrics,
        semantic_similarity,
    )

    mlflow_manager.initialise()

    registry = ModelRegistry()
    evaluator_langchain_llm = registry.get_llm(complexity="simple")
    wrapped_llm = LangchainLLMWrapper(evaluator_langchain_llm)

    sim_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    rag_system = RAGGraph()

    suite_path = Path(test_suite_path)
    if not suite_path.exists():
        raise FileNotFoundError(f"Test suite not found: {suite_path}")
    with open(suite_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    logger.info("Loaded %d test cases from %s", len(test_cases), test_suite_path)

    with mlflow_manager.start_run(run_name=f"eval-{int(time.time())}"):
        mlflow_manager.log_params({
            "provider": registry.provider_name,
            "test_suite_path": test_suite_path,
            "n_questions": len(test_cases),
            "max_retries": settings.max_retries,
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
            }

            t0 = time.perf_counter()
            # Task: Use native async ainvoke.
            result = await rag_system.app.ainvoke(inputs)
            total_latency_ms = (time.perf_counter() - t0) * 1000

            # Throttle to avoid 429
            await asyncio.sleep(0.5)

            answer = result.get("response", "")
            chunks = result.get("retrieved_chunks", [])
            context_texts = [c.content for c in chunks]

            sim_score = semantic_similarity(answer, ground_truth, sim_model) if ground_truth else None
            rouge = compute_rouge_l(answer, ground_truth) if ground_truth else None
            prec_k = _precision_at_k(chunks, expected_domain) if expected_domain else None
            tool_flags = compute_tool_metrics(result)

            metric_record = {
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
                "retrieved_context_texts": context_texts,
                "context_precision": None,
                "answer_relevancy": None,
                **tool_flags,
            }
            per_question_metrics.append(metric_record)

            ragas_samples.append(SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=context_texts if context_texts else [""],
                reference=ground_truth if ground_truth else "",
            ))

            if i < len(test_cases) - 1:
                await asyncio.sleep(3)

        # --- Ragas evaluation ---
        eval_dataset = EvaluationDataset(samples=ragas_samples)
        run_cfg = RunConfig(max_workers=1, timeout=300)

        try:
            print("--- Running Ragas faithfulness + context_recall ---")
            ragas_result = evaluate(
                dataset=eval_dataset,
                metrics=[faithfulness, context_recall],
                llm=wrapped_llm,
                run_config=run_cfg,
            )
            ragas_df = ragas_result.to_pandas()
            for i, row in enumerate(ragas_df.to_dict(orient="records")):
                if i < len(per_question_metrics):
                    per_question_metrics[i]["faithfulness"] = row.get("faithfulness")
                    per_question_metrics[i]["context_recall"] = row.get("context_recall")
        except Exception:
            logger.exception("Ragas eval failed")

        print("--- Running Ragas context_precision + answer_relevancy ---")
        async_data = [
            {
                "user_input": m["question"],
                "reference": m.get("ground_truth", ""),
                "retrieved_contexts": m.get("retrieved_context_texts") or [""],
                "response": m.get("answer", ""),
            }
            for m in per_question_metrics
        ]
        try:
            scored = await _score_new_metrics_async(evaluator_langchain_llm, async_data)
            for i, item in enumerate(scored):
                if i < len(per_question_metrics):
                    per_question_metrics[i]["context_precision"] = item.get("context_precision")
                    per_question_metrics[i]["answer_relevancy"] = item.get("answer_relevancy")
        except Exception:
            logger.exception("Async metrics failed")

        # --- Aggregate ---
        summary = {
            "n_questions": len(per_question_metrics),
            "provider": registry.provider_name,
            "avg_faithfulness": _avg("faithfulness", per_question_metrics),
            "avg_context_recall": _avg("context_recall", per_question_metrics),
            "avg_context_precision": _avg("context_precision", per_question_metrics),
            "avg_answer_relevancy": _avg("answer_relevancy", per_question_metrics),
            "avg_semantic_similarity": _avg("semantic_similarity", per_question_metrics),
            "avg_rouge_l": _avg("rouge_l", per_question_metrics),
            "avg_precision_at_k": _avg("precision_at_k", per_question_metrics),
            "avg_retrieval_time_ms": _avg("retrieval_time_ms", per_question_metrics),
            "avg_total_latency_ms": _avg("total_latency_ms", per_question_metrics),
            "avg_token_count": _avg("token_count", per_question_metrics),
            "avg_tool_router_success": _avg("tool_router_success", per_question_metrics),
            "avg_tool_retrieval_success": _avg("tool_retrieval_success", per_question_metrics),
            "avg_tool_critic_success": _avg("tool_critic_success", per_question_metrics),
            "avg_tool_action_success": _avg("tool_action_success", per_question_metrics),
        }

        output = {"summary": summary, "per_question": per_question_metrics}
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # Log artifact
        art_path = Path("reports/eval_mlflow_artifact.json")
        with open(art_path, "w", encoding="utf-8") as f:
            json.dump(per_question_metrics, f, indent=2)
        mlflow_manager.log_artifact(str(art_path))
        loggable = {k: v for k, v in summary.items() if isinstance(v, (int, float)) and v is not None}
        mlflow_manager.log_metrics(loggable)

        recommendations = _baseline_summary(summary)
        output["recommendations"] = recommendations

        # Persist baseline
        with open("reports/eval_baseline.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n--- Evaluation Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        return output

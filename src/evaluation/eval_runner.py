"""
Evaluation runner for the Knowledge Assistant.

Loads test cases from data/eval/test_suite.json, runs inference via RAGGraph,
and computes per-question metrics:
  - faithfulness (Ragas)
  - context_recall (Ragas)
  - precision_at_k (fraction of top-k chunks from expected domain)
  - retrieval_time_ms
  - total_latency_ms
  - token_count
  - semantic_similarity (cosine, bge-large)
  - rouge_l

Replaces the brute-force asyncio.sleep(15) in evaluate.py with tenacity
exponential backoff on rate-limit errors.

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


def _precision_at_k(chunks: List[Dict], expected_domain: str, k: int = 5) -> float:
    """Fraction of top-k chunks whose category matches the expected domain."""
    top_k = chunks[:k]
    if not top_k:
        return 0.0
    matches = sum(1 for c in top_k if c.get("category") == expected_domain)
    return matches / len(top_k)


def run_evaluation(
    test_suite_path: str = "data/eval/test_suite.json",
    output_path: str = "reports/evaluation_results.json",
) -> Dict[str, Any]:
    """Run the full evaluation suite and write results to output_path."""

    if "GOOGLE_API_KEY" not in os.environ:
        logger.error("GOOGLE_API_KEY not set — Ragas evaluation requires Gemini API access.")
        return {}

    # Import heavy dependencies only when evaluation actually runs
    from datasets import Dataset
    from langchain_core.outputs import Generation, LLMResult
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import BaseRagasEmbeddings
    from ragas.llms import BaseRagasLLM
    from ragas.metrics import context_recall, faithfulness
    from ragas.run_config import RunConfig
    from sentence_transformers import SentenceTransformer
    from tenacity import retry, stop_after_attempt, wait_exponential

    from src.agents.graph import RAGGraph
    from src.evaluation.metrics import compute_rouge_l, semantic_similarity

    # --- Rate-limit safe Ragas wrappers (tenacity replaces asyncio.sleep) ---
    gemini_lock = asyncio.Lock()

    class SafeRagasLLM(BaseRagasLLM):
        def __init__(self, llm):
            self.langchain_llm = llm

        def get_temperature(self, temperature=None):
            return 0

        def generate_text(self, prompt, n=1, temperature=1e-8, callbacks=None, **kwargs):
            return asyncio.run(
                self.agenerate_text(prompt, n, temperature, callbacks, **kwargs)
            )

        async def generate(self, prompts, n=1, temperature=1e-8, callbacks=None, **kwargs):
            generations = []
            for prompt in prompts:
                text = await self.agenerate_text(prompt, n, temperature, callbacks, **kwargs)
                generations.append([Generation(text=text)])
            return LLMResult(generations=generations)

        @retry(
            wait=wait_exponential(min=5, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        async def agenerate_text(self, prompt, n=1, temperature=1e-8, callbacks=None, **kwargs):
            async with gemini_lock:
                prompt_text = (
                    prompt.to_string()
                    if hasattr(prompt, "to_string")
                    else (prompt.text if hasattr(prompt, "text") else str(prompt))
                )
                res = await self.langchain_llm.ainvoke(
                    prompt_text, stop=kwargs.get("stop")
                )
                return res.content

    class SafeRagasEmbeddings(BaseRagasEmbeddings):
        def __init__(self, embeddings):
            self.embeddings = embeddings

        def embed_query(self, text):
            return asyncio.run(self._embed([text], is_query=True))[0]

        def embed_documents(self, texts):
            return asyncio.run(self._embed(texts, is_query=False))

        @retry(
            wait=wait_exponential(min=5, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        async def _embed(self, texts, is_query=False):
            results = []
            for text in texts:
                async with gemini_lock:
                    if is_query:
                        res = await self.embeddings.aembed_query(text)
                    else:
                        res = await self.embeddings.aembed_documents([text])
                        res = res[0]
                    results.append(res)
            return results

    # --- Load test suite ---
    test_cases = _load_test_suite(test_suite_path)
    logger.info("Loaded %d test cases from %s", len(test_cases), test_suite_path)

    # --- Initialise systems ---
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
    gemini_emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    evaluator_llm = SafeRagasLLM(gemini_llm)
    evaluator_emb = SafeRagasEmbeddings(gemini_emb)

    sim_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    rag_system = RAGGraph()

    # --- Run inference ---
    ragas_data: Dict[str, List] = {
        "question": [], "answer": [], "contexts": [], "ground_truth": []
    }
    per_question_metrics: List[Dict] = []

    for i, case in enumerate(test_cases):
        question = case["question"]
        expected_domain = case.get("expected_domain", "")
        ground_truth = case.get("ground_truth", "")
        is_help = case.get("is_help", False)

        logger.info("[%d/%d] Querying: %s", i + 1, len(test_cases), question[:80])

        inputs = {
            "query": question,
            "session_id": "eval",
            "is_help_section": is_help,
            "history": [],
            "retry_count": 0,
            "from_cache": False,
            "sanitized_query": "",
            "guard_input_result": {},
            "guard_output_result": {},
            "category_probs": {},
            "dominant_category": "",
            "confidence": 0.0,
            "retrieved_chunks": [],
            "context_metadata": {},
            "retrieval_time_ms": 0.0,
            "sources_cited": [],
            "response": "",
            "token_count": 0,
            "critic_verdict": {},
            "action_result": {},
        }

        t0 = time.perf_counter()
        result = rag_system.app.invoke(inputs)
        total_latency_ms = (time.perf_counter() - t0) * 1000

        answer = result.get("response", "")
        chunks = result.get("retrieved_chunks", [])
        context_texts = [c.get("content", "") for c in chunks]

        sim_score = semantic_similarity(answer, ground_truth, sim_model) if ground_truth else None
        rouge = compute_rouge_l(answer, ground_truth) if ground_truth else None
        prec_k = _precision_at_k(chunks, expected_domain) if expected_domain else None

        per_question_metrics.append({
            "question": question,
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

        ragas_data["question"].append(question)
        ragas_data["answer"].append(answer)
        ragas_data["contexts"].append(context_texts)
        ragas_data["ground_truth"].append(ground_truth)

        if i < len(test_cases) - 1:
            logger.info("Sleeping 10s for rate-limit safety...")
            time.sleep(10)

    # --- Ragas evaluation ---
    dataset = Dataset.from_dict(ragas_data)
    run_cfg = RunConfig(max_workers=1, timeout=900)

    logger.info("Running Ragas faithfulness + context_recall...")
    ragas_result = evaluate(
        dataset,
        metrics=[faithfulness, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_emb,
        run_config=run_cfg,
    )
    ragas_scores = ragas_result.to_pandas().to_dict(orient="records")

    # Merge Ragas scores into per-question metrics
    for i, row in enumerate(ragas_scores):
        if i < len(per_question_metrics):
            per_question_metrics[i]["faithfulness"] = row.get("faithfulness")
            per_question_metrics[i]["context_recall"] = row.get("context_recall")

    # --- Aggregate ---
    def _avg(key):
        vals = [r[key] for r in per_question_metrics if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "n_questions": len(per_question_metrics),
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
    print("\n--- Evaluation Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return output

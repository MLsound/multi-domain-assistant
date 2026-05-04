"""
Weighted semantic retriever with two-stage retriever-ranker pipeline.

Stage 1 — Weighted Qdrant search:
  Score(d) = P(c|q) * sim(q, d)

Stage 2 — Cross-encoder reranking:
  Applies ms-marco-MiniLM-L6-v2 to rerank top-K candidates by
  query-chunk relevance, then returns the final top-N.

External API calls (Qdrant) are wrapped with tenacity for resilience.

Changes from v0.1:
  - print() replaced with structured logging.
  - Constructor parameters from settings.
  - Cross-encoder reranking added (Phase 5).
  - Tenacity retry on Qdrant queries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings

logger = logging.getLogger(__name__)


class WeightedRetriever:
    """Two-stage retriever-ranker: weighted Qdrant search + cross-encoder reranking."""

    def __init__(
        self,
        qdrant_url: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.client = QdrantClient(url=qdrant_url or settings.qdrant_url)
        self.collection_name = collection_name or settings.collection_name
        self.embedding_model = SentenceTransformer(settings.retrieval_embedding_model)
        self.reranker = CrossEncoder(settings.reranker_model)
        self.categories = ["Software", "User", "Science"]
        logger.info(
            "WeightedRetriever ready — collection=%s embedding=%s reranker=%s",
            self.collection_name,
            settings.retrieval_embedding_model,
            settings.reranker_model,
        )

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _qdrant_query(
        self,
        query_vector: list,
        category: str,
        limit: int,
    ) -> list:
        """Execute a single Qdrant category query with retry on failure."""
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=category),
                    )
                ]
            ),
            limit=limit,
        ).points

    def search(
        self,
        query: str,
        category_probs: Dict[str, float],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank context chunks.

        Stage 1: Weighted cosine similarity search per domain.
        Stage 2: Cross-encoder reranking on the combined candidate pool.

        Args:
            query:          Raw (or refined) query string.
            category_probs: Domain probability distribution from Router.
            top_k:          Number of candidates to fetch per category.
                            Defaults to settings.retrieval_top_k.

        Returns:
            Top settings.retrieval_final_k chunks sorted by rerank score.
        """
        if top_k is None:
            top_k = settings.retrieval_top_k

        query_vector = self.embedding_model.encode(query).tolist()
        candidates: List[Dict[str, Any]] = []

        for category, prob in category_probs.items():
            if prob <= 0:
                continue

            try:
                results = self._qdrant_query(query_vector, category, top_k)
            except Exception:
                logger.exception("Qdrant query failed for category=%s", category)
                continue

            for res in results:
                weighted_score = prob * res.score
                candidates.append({
                    "content": res.payload.get("text", ""),
                    "metadata": res.payload,
                    "score": weighted_score,
                    "original_score": res.score,
                    "category": category,
                })

        if not candidates:
            logger.warning("No candidates retrieved from Qdrant")
            return []

        # --- Stage 2: Cross-encoder reranking ---
        final_k = settings.retrieval_final_k
        if len(candidates) > final_k:
            pairs = [(query, c["content"]) for c in candidates]
            try:
                rerank_scores = self.reranker.predict(pairs)
                for i, score in enumerate(rerank_scores):
                    candidates[i]["rerank_score"] = float(score)
                candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                logger.debug(
                    "Reranked %d candidates → top %d",
                    len(candidates),
                    final_k,
                )
            except Exception:
                logger.exception(
                    "Cross-encoder reranking failed — falling back to weighted scores"
                )
                candidates.sort(key=lambda x: x["score"], reverse=True)
        else:
            candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:final_k]

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Create Qdrant collection if absent, then upsert all document chunks."""
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024, distance=models.Distance.COSINE
                ),
            )
            logger.info("Created Qdrant collection: %s", self.collection_name)
        else:
            logger.info("Collection '%s' exists — upserting.", self.collection_name)

        points = []
        for i, doc in enumerate(documents):
            vector = self.embedding_model.encode(doc["text"]).tolist()
            points.append(
                models.PointStruct(id=i, vector=vector, payload=doc)
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Indexed %d chunks into '%s'.", len(documents), self.collection_name)

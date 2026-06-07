"""
Semantic query cache backed by a dedicated Qdrant collection.

On each request, the cache is checked before the Guard agent runs.
A hit short-circuits the entire pipeline and returns the cached response.

Similarity threshold: settings.cache_similarity_threshold (default 0.95).
TTL: settings.cache_ttl_seconds (default 3600 s = 1 hour).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.config.settings import settings

logger = logging.getLogger(__name__)


class SemanticCache:
    """Query-level semantic cache using Qdrant as the vector backend."""

    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.cache_collection_name
        self.model = SentenceTransformer(settings.retrieval_embedding_model)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=1024, distance=models.Distance.COSINE
                ),
            )
            logger.info("Cache collection '%s' created", self.collection)

    def check(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Look up query in the cache.

        Returns the cached payload dict if a sufficiently similar query is found
        within TTL, otherwise None.
        """
        vec = self.model.encode(query).tolist()

        try:
            results = self.client.query_points(
                collection_name=self.collection,
                query=vec,
                limit=1,
            ).points
        except Exception:
            logger.warning("Cache check failed — proceeding without cache")
            return None

        if not results:
            return None

        hit = results[0]
        if hit.score < settings.cache_similarity_threshold:
            logger.debug("Cache MISS (score=%.3f < threshold=%.2f)", hit.score, settings.cache_similarity_threshold)
            return None

        payload = hit.payload or {}
        age = time.time() - payload.get("timestamp", 0)
        if age > settings.cache_ttl_seconds:
            logger.debug("Cache MISS (entry expired, age=%.0fs)", age)
            return None

        logger.info(
            "Cache HIT — score=%.3f age=%.0fs query=%.60s",
            hit.score,
            age,
            query,
        )
        return payload

    def store(self, query: str, response: str, sources: list) -> None:
        """Store a query/response pair in the cache."""
        vec = self.model.encode(query).tolist()
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload={
                            "query": query,
                            "response": response,
                            "sources_cited": sources,
                            "timestamp": time.time(),
                        },
                    )
                ],
            )
            logger.debug("Cache STORE: %.60s", query)
        except Exception:
            logger.warning("Cache store failed — continuing without caching")

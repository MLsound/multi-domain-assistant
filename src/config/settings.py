"""
Centralised configuration for the Knowledge Assistant.

All values that were previously hardcoded across graph.py, mlp_router.py,
and weighted_retriever.py are now sourced from environment variables or
.env file via Pydantic BaseSettings.
"""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # LLM provider API keys
    # ------------------------------------------------------------------
    google_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    moonshot_api_key: Optional[str] = None

    # Provider selection:
    #   "auto"       – check keys in priority order (Gemini→Claude→Groq→OpenRouter→Kimi→Ollama)
    #   "gemini"     – force Google Gemini
    #   "claude"     – force Anthropic Claude
    #   "groq"       – force Groq
    #   "openrouter" – force OpenRouter
    #   "kimi"       – force Moonshot/Kimi
    #   "ollama"     – force local Ollama
    llm_provider: str = "auto"

    # ------------------------------------------------------------------
    # Model identifiers per provider
    # ------------------------------------------------------------------
    # Gemini
    gemini_model_simple: str = "gemini-2.0-flash-lite"
    gemini_model_complex: str = "gemini-2.0-flash"

    # Claude
    claude_model: str = "claude-haiku-4-5-20251001"

    # Groq
    groq_model_simple: str = "llama-3.1-8b-instant"
    groq_model_complex: str = "llama-3.3-70b-versatile"

    # OpenRouter (free :free models)
    openrouter_model_simple: str = "google/gemma-4-31b-it:free"
    openrouter_model_complex: str = "meta-llama/llama-3.3-70b-instruct:free"

    # Moonshot / Kimi
    kimi_model: str = "kimi-k2.5"

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    # ------------------------------------------------------------------
    # Qdrant vector store
    # ------------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "rag_collection"
    cache_collection_name: str = "query_cache"

    # ------------------------------------------------------------------
    # Embedding & routing models
    # ------------------------------------------------------------------
    router_embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # ------------------------------------------------------------------
    # Retrieval behaviour
    # ------------------------------------------------------------------
    # Retrieve top-K from Qdrant before reranking, then keep final_k
    retrieval_top_k: int = 10
    retrieval_final_k: int = 5

    # Science domain MCP enrichment threshold
    science_threshold: float = 0.4

    # Help-section probability override (loaded as JSON string in env if needed)
    help_override_probs: Dict[str, float] = Field(
        default={"Software": 0.85, "User": 0.0, "Science": 0.15}
    )

    # ------------------------------------------------------------------
    # Router / MLP
    # ------------------------------------------------------------------
    mlp_model_path: str = "models/router_mlp.pth"
    router_confidence_threshold: float = 0.8

    # ------------------------------------------------------------------
    # Critic / retry loop
    # ------------------------------------------------------------------
    max_retries: int = 2
    critic_approval_threshold: float = 0.7

    # ------------------------------------------------------------------
    # Semantic cache
    # ------------------------------------------------------------------
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 3600

    # ------------------------------------------------------------------
    # Session memory
    # ------------------------------------------------------------------
    session_ttl_minutes: int = 30

    # ------------------------------------------------------------------
    # Automated action (Action Agent)
    # ------------------------------------------------------------------
    webhook_url: Optional[str] = None
    audit_log_path: str = "logs/audit.jsonl"

    # ------------------------------------------------------------------
    # Observability / MLflow
    # ------------------------------------------------------------------
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "knowledge-assistant"
    enable_tracing: bool = True  # MLflow tracing enabled by default

    # ------------------------------------------------------------------
    # MLflow tracking
    # ------------------------------------------------------------------
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "knowledge-assistant"

    # ------------------------------------------------------------------
    # Document ingestion / chunking
    # ------------------------------------------------------------------
    chunk_size: int = 512
    chunk_overlap: int = 50

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Module-level singleton — import this everywhere
settings = Settings()

"""
ModelRegistry — multi-provider LLM abstraction.

Priority order when LLM_PROVIDER=auto:
  Gemini → Claude → Groq → OpenRouter → Kimi → Ollama (local)

Option C behaviour: on startup, always prints the active provider to stdout
regardless of log level, so developers can always see which LLM is in use.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel

from src.config.settings import settings

logger = logging.getLogger(__name__)

_PRIORITY = ["gemini", "claude", "groq", "openrouter", "kimi", "ollama"]


def _detect_provider() -> str:
    """Return the first available provider based on environment API keys."""
    if settings.llm_provider != "auto":
        logger.info("LLM_PROVIDER forced to: %s", settings.llm_provider)
        return settings.llm_provider

    if settings.google_api_key:
        return "gemini"
    if settings.anthropic_api_key:
        return "claude"
    if settings.groq_api_key:
        return "groq"
    if settings.openrouter_api_key:
        return "openrouter"
    if settings.moonshot_api_key:
        return "kimi"
    # Default to Ollama — will fail at invocation time if Ollama is not running
    return "ollama"


def _build_llm(
    provider: str,
    complexity: Literal["simple", "complex"],
) -> BaseChatModel:
    """Instantiate and return the appropriate BaseChatModel."""

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = (
            settings.gemini_model_simple
            if complexity == "simple"
            else settings.gemini_model_complex
        )
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
        )

    if provider == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic provider requires: poetry install --extras anthropic"
            ) from exc
        return ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
        )

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise ImportError(
                "Groq provider requires: poetry install --extras groq"
            ) from exc
        model = (
            settings.groq_model_simple
            if complexity == "simple"
            else settings.groq_model_complex
        )
        return ChatGroq(model=model, groq_api_key=settings.groq_api_key)

    if provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenRouter provider requires: poetry install --extras openrouter"
            ) from exc
        model = (
            settings.openrouter_model_simple
            if complexity == "simple"
            else settings.openrouter_model_complex
        )
        return ChatOpenAI(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )

    if provider == "kimi":
        try:
            from langchain_moonshot import ChatMoonshot
        except ImportError as exc:
            raise ImportError(
                "Kimi provider requires: poetry install --extras moonshot"
            ) from exc
        return ChatMoonshot(
            model=settings.kimi_model,
            moonshot_api_key=settings.moonshot_api_key,
        )

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "Ollama provider requires: poetry install --extras ollama"
            ) from exc
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. "
        f"Valid options: {_PRIORITY}"
    )


class ModelRegistry:
    """
    Resolves the active LLM provider once at startup and vends BaseChatModel
    instances on demand.

    Option C: always logs the active provider to stdout at startup so that
    developers can see at a glance which LLM backend is in use.
    """

    def __init__(self) -> None:
        self._provider = _detect_provider()
        self._announce_provider()

    def _announce_provider(self) -> None:
        _labels = {
            "gemini": (
                f"Google Gemini "
                f"({settings.gemini_model_simple} / {settings.gemini_model_complex})"
            ),
            "claude": f"Anthropic Claude ({settings.claude_model})",
            "groq": (
                f"Groq "
                f"({settings.groq_model_simple} / {settings.groq_model_complex})"
            ),
            "openrouter": (
                f"OpenRouter "
                f"({settings.openrouter_model_simple} / {settings.openrouter_model_complex})"
            ),
            "kimi": f"Moonshot Kimi ({settings.kimi_model})",
            "ollama": (
                f"Ollama local "
                f"({settings.ollama_model} @ {settings.ollama_base_url})"
            ),
        }
        label = _labels.get(self._provider, self._provider)
        # Intentional unconditional stdout print (Option C)
        print(f"[ModelRegistry] Active provider: {label}")
        logger.info("ModelRegistry initialised — provider=%s", self._provider)

    def get_llm(
        self,
        complexity: Literal["simple", "complex"] = "simple",
    ) -> BaseChatModel:
        """Return a configured BaseChatModel for the active provider.

        Args:
            complexity: "simple" selects a faster/cheaper model tier;
                        "complex" selects a more capable model tier.
        """
        return _build_llm(self._provider, complexity)

    @property
    def provider_name(self) -> str:
        """Name of the active provider (e.g. 'gemini', 'groq')."""
        return self._provider

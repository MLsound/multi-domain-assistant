"""
Unit tests for ModelRegistry provider detection.

API calls are never made — we only verify that the correct provider name
is selected based on environment variables.
"""

from __future__ import annotations

from unittest.mock import patch


def _registry_with_env(**env_overrides):
    """Helper: patch settings attributes and return a fresh ModelRegistry."""
    from src.config import model_registry as mr_module

    # Reload to get a fresh _detect_provider call
    with patch.multiple("src.config.model_registry.settings", **env_overrides):
        # Patch print so Option C startup log doesn't clutter test output
        with patch("builtins.print"):
            from importlib import reload

            reload(mr_module)
            registry = mr_module.ModelRegistry()
    return registry


def test_gemini_selected_when_key_present() -> None:
    with patch("src.config.model_registry.settings") as mock_settings:
        mock_settings.llm_provider = "auto"
        mock_settings.google_api_key = "fake-google-key"
        mock_settings.anthropic_api_key = None
        mock_settings.groq_api_key = None
        mock_settings.openrouter_api_key = None
        mock_settings.moonshot_api_key = None
        mock_settings.gemini_model_simple = "gemini-2.0-flash-lite"
        mock_settings.gemini_model_complex = "gemini-2.0-flash"
        mock_settings.claude_model = "claude-haiku"
        mock_settings.groq_model_simple = "llama-3.1-8b-instant"
        mock_settings.groq_model_complex = "llama-3.3-70b-versatile"
        mock_settings.openrouter_model_simple = "gemma:free"
        mock_settings.openrouter_model_complex = "llama:free"
        mock_settings.kimi_model = "kimi-k2.5"
        mock_settings.ollama_model = "phi3:mini"
        mock_settings.ollama_base_url = "http://localhost:11434"

        with patch("builtins.print"):
            from src.config.model_registry import _detect_provider

            provider = _detect_provider()

    assert provider == "gemini"


def test_groq_selected_when_only_groq_key() -> None:
    with patch("src.config.model_registry.settings") as mock_settings:
        mock_settings.llm_provider = "auto"
        mock_settings.google_api_key = None
        mock_settings.anthropic_api_key = None
        mock_settings.groq_api_key = "fake-groq-key"
        mock_settings.openrouter_api_key = None
        mock_settings.moonshot_api_key = None

        with patch("builtins.print"):
            from src.config.model_registry import _detect_provider

            provider = _detect_provider()

    assert provider == "groq"


def test_ollama_selected_when_no_api_keys() -> None:
    with patch("src.config.model_registry.settings") as mock_settings:
        mock_settings.llm_provider = "auto"
        mock_settings.google_api_key = None
        mock_settings.anthropic_api_key = None
        mock_settings.groq_api_key = None
        mock_settings.openrouter_api_key = None
        mock_settings.moonshot_api_key = None

        with patch("builtins.print"):
            from src.config.model_registry import _detect_provider

            provider = _detect_provider()

    assert provider == "ollama"


def test_forced_provider_overrides_keys() -> None:
    with patch("src.config.model_registry.settings") as mock_settings:
        mock_settings.llm_provider = "groq"
        mock_settings.google_api_key = "fake-google-key"  # would normally win
        mock_settings.groq_api_key = "fake-groq-key"

        with patch("builtins.print"):
            from src.config.model_registry import _detect_provider

            provider = _detect_provider()

    assert provider == "groq"

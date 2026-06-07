#!/usr/bin/env python3
"""
Local model setup helper for the Ollama fallback provider.

Ollama manages its own model storage — there is no GGUF file to download
manually.  This script verifies that Ollama is running and pulls phi3:mini
if it is not already present.

Usage:
    python scripts/download_local_model.py

Prerequisites:
    1. Install Ollama from https://ollama.ai/download
    2. Start the Ollama service:
         Windows: Ollama starts automatically after installation.
         Linux/macOS: ollama serve
    3. Run this script.

After setup, set in .env:
    LLM_PROVIDER=ollama
    OLLAMA_MODEL=phi3:mini
"""

from __future__ import annotations

import subprocess
import sys


def check_ollama_running() -> bool:
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def pull_model(model: str = "phi3:mini") -> bool:
    print(f"Pulling model '{model}' via Ollama...")
    result = subprocess.run(
        ["ollama", "pull", model],
        capture_output=False,
    )
    return result.returncode == 0


def main() -> None:
    print("=" * 60)
    print("  Knowledge Assistant — Local Model Setup (Ollama)")
    print("=" * 60)

    if not check_ollama_running():
        print(
            "\nOllama is not running or not installed.\n"
            "Please install Ollama from: https://ollama.ai/download\n"
            "Then start it and re-run this script."
        )
        sys.exit(1)

    print("Ollama is running.")
    model = "phi3:mini"

    if pull_model(model):
        print(f"\nModel '{model}' is ready.")
        print("\nTo use Ollama as the LLM provider, add to your .env file:")
        print("  LLM_PROVIDER=ollama")
        print(f"  OLLAMA_MODEL={model}")
    else:
        print(f"\nFailed to pull '{model}'. Check Ollama logs.")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Interactive CLI chat interface for the Knowledge Assistant.

Reads settings from .env (loaded automatically). Displays router confidence,
sources cited, and active provider on each turn.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from src.agents.graph import RAGGraph  # noqa: E402 — after load_dotenv
from src.config.settings import settings  # noqa: E402


def chat() -> None:
    rag_system = RAGGraph()

    print("\n" + "=" * 60)
    print("   Knowledge Assistant — Agentic RAG System")
    print("=" * 60)
    print("Type 'exit' or 'quit' to end the session.\n")

    session_id = "cli-session"
    is_help = input("[Help section override? (y/n)] > ").strip().lower() == "y"

    while True:
        try:
            query = input("[Query] > ").strip()
            if query.lower() in {"exit", "quit", ""}:
                print("\nEnding session. Goodbye!")
                break

            inputs = {
                "query": query,
                "session_id": session_id,
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

            print("\n--- Processing ---")
            result = rag_system.app.invoke(inputs)

            print(f"\n[Provider] {rag_system.provider_name}")
            print(f"[Domain]   {result.get('dominant_category', '?')}  "
                  f"(confidence={result.get('confidence', 0):.2f})")
            if result.get("sources_cited"):
                print(f"[Sources]  {', '.join(result['sources_cited'])}")
            if result.get("from_cache"):
                print("[Cache]    HIT — served from semantic cache")

            print("\n[Assistant]")
            print("-" * 60)
            print(result.get("response", "No response generated."))
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    chat()

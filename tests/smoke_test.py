"""
Smoke test script for the RAG pipeline.

Runs a few hardcoded queries to verify the basic flow.
See `docs/CONTEXT.md` for the domain context of these tests.
"""

import asyncio
import os
import time
from dotenv import load_dotenv

from src.agents.graph import RAGGraph

async def main():
    load_dotenv()
    # Ensure GOOGLE_API_KEY is set or provide a warning
    if "GOOGLE_API_KEY" not in os.environ:
        print("Warning: GOOGLE_API_KEY not set. LLM synthesis will fail.")
        # os.environ["GOOGLE_API_KEY"] = "AIza..." 

    mlp_model_path = "models/router_mlp.pth"
    rag_system = RAGGraph(mlp_model_path)

    # Test Query 1: Normal Science query
    print("\n--- Testing Science Query ---")
    inputs = {"query": "How do Environmental systems use Science variables?", "history": [], "is_help_section": False}
    result = await rag_system.app.ainvoke(inputs)
    print(f"Response: {result['response']}")

    print("\n--- Waiting 5 seconds (Rate Limit Safety) ---")
    await asyncio.sleep(5)

    # Test Query 2: Help section override
    print("\n--- Testing Help Section Override ---")
    inputs = {"query": "Explain scientific optimization.", "history": [], "is_help_section": True}
    result = await rag_system.app.ainvoke(inputs)
    print(f"Response: {result['response']}")
    print(f"Category Probabilities used: {result['category_probs']}")

if __name__ == "__main__":
    asyncio.run(main())

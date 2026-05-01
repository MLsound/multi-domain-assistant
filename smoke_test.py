from src.agents.graph import RAGGraph
from dotenv import load_dotenv
import os
import time

def main():
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
    result = rag_system.app.invoke(inputs)
    print(f"Response: {result['response']}")

    print("\n--- Waiting 10 seconds (Rate Limit Safety) ---")
    time.sleep(10)

    # Test Query 2: Help section override
    print("\n--- Testing Help Section Override ---")
    inputs = {"query": "Explain scientific optimization.", "history": [], "is_help_section": True}
    result = rag_system.app.invoke(inputs)
    print(f"Response: {result['response']}")
    print(f"Category Probabilities used: {result['category_probs']}")

if __name__ == "__main__":
    main()

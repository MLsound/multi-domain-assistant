from src.agents.graph import RAGGraph
from dotenv import load_dotenv
import os

def chat():
    load_dotenv()
    mlp_model_path = "models/router_mlp.pth"
    rag_system = RAGGraph(mlp_model_path)

    print("\n" + "="*50)
    print("      Knowledge Assistant: Agentic RAG System")
    print("="*50)
    print("(type 'exit' or 'quit' to end session)")
    
    while True:
        try:
            query = input("\n[Query] > ")
            if query.lower() in ["exit", "quit"]:
                print("\nEnding session. Goodbye!")
                break
                
            is_help = input("[Help Section Override? (y/n)] > ").lower() == 'y'
            
            inputs = {
                "query": query, 
                "history": [], 
                "is_help_section": is_help
            }
            
            print("\n--- Processing ---")
            result = rag_system.app.invoke(inputs)
            
            print(f"Router Confidence: {result['category_probs']}")
            if result.get('context_metadata'):
                print(f"Domain Metadata: {result['context_metadata']}")
            
            print("\n[Assistant]:")
            print("-" * 20)
            print(result['response'])
            print("-" * 20)
            
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    chat()

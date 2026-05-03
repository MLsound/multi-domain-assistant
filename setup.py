from src.router.mlp_router import MLPRouter
from src.retrieval.weighted_retriever import WeightedRetriever
import os
import glob

def setup():
    # 1. Initialize mapping of directories to categories
    category_map = {
        "data/software": "Software",
        "data/user": "User",
        "data/scientific": "Science"
    }
    
    documents = []
    
    print("--- Starting Document Ingestion ---")
    # Support multiple file extensions
    extensions = ["*.txt", "*.json", "*.yaml", "*.yml", "*.py"]
    
    for directory, category in category_map.items():
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} not found.")
            continue
            
        category_files = []
        for ext in extensions:
            category_files.extend(glob.glob(os.path.join(directory, ext)))
            
        print(f"Scanning {directory} (Category: {category}): Found {len(category_files)} files.")
        
        for file_path in category_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append({
                        "text": content,
                        "category": category,
                        "source_id": os.path.basename(file_path)
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    if not documents:
        print("No documents found to index or train on.")
        return

    # 2. Train and Save MLP Router
    print("\n--- Training MLP Router ---")
    os.makedirs("models", exist_ok=True)
    model_path = "models/router_mlp.pth"
    
    router = MLPRouter()
    # Using the same documents for training as for indexing
    # In a real scenario, you'd want specialized training queries, 
    # but using document chunks/content is a good heuristic for bootstrapping.
    router.train_model(documents, epochs=50)
    router.save(model_path)
    
    # 3. Initialize and Index Retriever
    print("\n--- Indexing Documents in Vector DB ---")
    try:
        retriever = WeightedRetriever()
        retriever.index_documents(documents)
        print(f"Successfully indexed {len(documents)} documents.")
    except Exception as e:
        print(f"Error indexing documents: {e}")

if __name__ == "__main__":
    setup()

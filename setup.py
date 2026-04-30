from src.router.mlp_router import create_dummy_model
from src.retrieval.weighted_retriever import WeightedRetriever
import os
import glob

def setup():
    # 1. Create model directory and dummy MLP
    os.makedirs("models", exist_ok=True)
    model_path = "models/router_mlp.pth"
    if not os.path.exists(model_path):
        create_dummy_model(model_path)
    
    # 2. Initialize Retriever
    retriever = WeightedRetriever()
    
    # 3. Define mapping of directories to categories
    category_map = {
        "data/software": "Software",
        "data/user_manuals": "User",
        "data/scientific": "Science"
    }
    
    documents = []
    
    print("--- Starting Document Ingestion ---")
    for directory, category in category_map.items():
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} not found.")
            continue
            
        files = glob.glob(os.path.join(directory, "*.txt"))
        print(f"Scanning {directory} (Category: {category}): Found {len(files)} files.")
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "text": content,
                    "category": category,
                    "source_id": os.path.basename(file_path)
                })
    
    if documents:
        try:
            retriever.index_documents(documents)
            print(f"Successfully indexed {len(documents)} documents from local directories.")
        except Exception as e:
            print(f"Error indexing documents: {e}")
    else:
        print("No documents found to index.")

if __name__ == "__main__":
    setup()

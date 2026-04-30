from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class WeightedRetriever:
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "rag_collection"):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.categories = ["Software", "User", "Science"]

    def search(self, query: str, category_probs: Dict[str, float], top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = self.embedding_model.encode(query).tolist()
        
        all_results = []
        
        for category, prob in category_probs.items():
            if prob <= 0:
                continue
            
            # Use 'query_points' method for modern qdrant-client versions
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="category", match=models.MatchValue(value=category))
                    ]
                ),
                limit=top_k
            ).points
            
            for res in results:
                # Apply weighted scoring formula: Score(d) = P(c|q) * sim(q, d)
                weighted_score = prob * res.score
                all_results.append({
                    "content": res.payload.get("text", ""),
                    "metadata": res.payload,
                    "score": weighted_score,
                    "original_score": res.score,
                    "category": category
                })
        
        # Sort by weighted score and take top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def index_documents(self, documents: List[Dict[str, Any]]):
        # Helper to index documents if needed
        
        # Create collection if not exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
            )
        
        points = []
        for i, doc in enumerate(documents):
            vector = self.embedding_model.encode(doc["text"]).tolist()
            points.append(models.PointStruct(
                id=i,
                vector=vector,
                payload=doc
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Indexed {len(documents)} documents.")

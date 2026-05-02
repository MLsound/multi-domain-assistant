import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

class MLPRouter:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.input_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.categories = ["Software", "User", "Science"]
        self.num_classes = len(self.categories)
        self.mlp = MLP(self.input_dim, self.num_classes).to(self.device)
        
        if model_path:
            self.mlp.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # Initialize with some random weights for simulation if no model is provided
            print("Warning: No pre-trained MLP model provided. Using random weights.")
        
        self.mlp.eval()

    def route(self, query: str) -> Dict[str, float]:
        self.mlp.eval()
        with torch.no_grad():
            embedding = self.embedding_model.encode([query], convert_to_tensor=True).to(self.device)
            probs = self.mlp(embedding).cpu().numpy()[0]
            
        return {cat: float(prob) for cat, prob in zip(self.categories, probs)}

    def train_model(self, training_data: List[Dict[str, str]], epochs: int = 50):
        """
        Trains the MLP on the provided texts and their corresponding categories.
        """
        self.mlp.train()
        texts = [d["text"] for d in training_data]
        # Map category names to indices
        labels = torch.tensor([self.categories.index(d["category"]) for d in training_data]).to(self.device)
        
        print(f"Embedding {len(texts)} training samples...")
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True).to(self.device)
        
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.mlp(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        self.mlp.eval()
        print("Training complete.")

    def save(self, path: str):
        torch.save(self.mlp.state_dict(), path)
        print(f"Model saved to {path}")

# Helper to create a dummy model for testing
def create_dummy_model(save_path: str):
    router = MLPRouter()
    router.save(save_path)

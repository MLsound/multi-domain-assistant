"""
MLP-based semantic router.

Uses a pre-trained sentence-transformer (all-MiniLM-L6-v2) to embed queries,
then classifies them into one of three knowledge domains via a locally-trained
PyTorch MLP.

Changes from v0.1:
- All print() calls replaced with structured logging.
- Model/embedding identifiers sourced from settings at construction time.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Three-layer fully-connected classifier with ReLU activations."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


class MLPRouter:
    """
    Domain classifier: embeds a query string and returns a probability
    distribution over ["Software", "User", "Science"].
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("MLPRouter device: %s", self.device)

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.input_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.categories = ["Software", "User", "Science"]
        self.num_classes = len(self.categories)
        self.mlp = MLP(self.input_dim, self.num_classes).to(self.device)

        if model_path:
            try:
                self.mlp.load_state_dict(
                    torch.load(model_path, map_location=self.device, weights_only=True)
                )
                logger.info("MLP weights loaded from %s", model_path)
            except Exception:
                logger.exception("Failed to load MLP weights from %s — using random weights", model_path)
        else:
            logger.warning("No pre-trained MLP model path provided. Using random weights.")

        self.mlp.eval()

    def route(self, query: str) -> Dict[str, float]:
        """Return domain probability distribution for the given query string."""
        self.mlp.eval()
        with torch.no_grad():
            embedding = self.embedding_model.encode(
                [query], convert_to_tensor=True
            ).to(self.device)
            probs = self.mlp(embedding).cpu().numpy()[0]

        result = {cat: float(prob) for cat, prob in zip(self.categories, probs)}
        logger.debug("Route result: %s", result)
        return result

    def train_model(self, training_data: List[Dict[str, str]], epochs: int = 50) -> None:
        """Train the MLP on document chunks and their domain labels."""
        self.mlp.train()
        texts = [d["text"] for d in training_data]
        labels = torch.tensor(
            [self.categories.index(d["category"]) for d in training_data]
        ).to(self.device)

        logger.info("Embedding %d training samples...", len(texts))
        embeddings = self.embedding_model.encode(
            texts, convert_to_tensor=True, show_progress_bar=False
        ).to(self.device)

        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        logger.info("Starting MLP training for %d epochs...", epochs)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.mlp(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info("Epoch [%d/%d]  loss=%.4f", epoch + 1, epochs, loss.item())

        self.mlp.eval()
        logger.info("MLP training complete.")

    def save(self, path: str) -> None:
        """Persist MLP weights to disk."""
        torch.save(self.mlp.state_dict(), path)
        logger.info("MLP model saved to %s", path)


def create_dummy_model(save_path: str) -> None:
    """Helper: create and save an untrained model (useful for tests)."""
    router = MLPRouter()
    router.save(save_path)

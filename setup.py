"""
System bootstrap: trains the MLP router and indexes documents into Qdrant.

Changes from v0.1:
- Documents are now split into overlapping chunks before indexing
  (RecursiveCharacterTextSplitter, chunk_size and chunk_overlap from settings).
- All print() calls replaced with structured logging.
- Hardcoded values sourced from src.config.settings.
"""

from __future__ import annotations

import glob
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings
from src.retrieval.weighted_retriever import WeightedRetriever
from src.router.mlp_router import MLPRouter


def setup() -> None:
    """Train the MLP router on document embeddings and index chunked documents into Qdrant."""

    category_map = {
        "data/software": "Software",
        "data/user": "User",
        "data/scientific": "Science",
    }

    extensions = ["*.txt", "*.json", "*.yaml", "*.yml", "*.py", "*.md"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    documents: list[dict] = []

    logger.info("Starting document ingestion (chunk_size=%d, overlap=%d)",
                settings.chunk_size, settings.chunk_overlap)

    for directory, category in category_map.items():
        if not os.path.exists(directory):
            logger.warning("Directory not found, skipping: %s", directory)
            continue

        category_files: list[str] = []
        for ext in extensions:
            category_files.extend(glob.glob(os.path.join(directory, ext)))

        logger.info("Scanning %s (category=%s): %d files found",
                    directory, category, len(category_files))

        for file_path in category_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                source_id = os.path.basename(file_path)
                chunks = splitter.split_text(content)

                for chunk_idx, chunk_text in enumerate(chunks):
                    documents.append({
                        "text": chunk_text,
                        "category": category,
                        "source_id": source_id,
                        "chunk_index": chunk_idx,
                    })

            except Exception:
                logger.exception("Error reading file: %s", file_path)

    if not documents:
        logger.error("No documents found. Aborting.")
        return

    logger.info("Total chunks prepared for indexing: %d", len(documents))

    # --- Train and save MLP router ---
    logger.info("Training MLP router...")
    os.makedirs("models", exist_ok=True)
    router = MLPRouter()
    router.train_model(documents, epochs=50)
    router.save(settings.mlp_model_path)

    # --- Index chunks into Qdrant ---
    logger.info("Indexing chunks into Qdrant...")
    try:
        retriever = WeightedRetriever()
        retriever.index_documents(documents)
        logger.info("Successfully indexed %d chunks.", len(documents))
    except Exception:
        logger.exception("Error indexing documents into Qdrant")


if __name__ == "__main__":
    setup()

#!/bin/bash

# Exit on error
set -e

echo "--- 1. Starting Qdrant in background ---"
if [ "$(docker ps -q -f name=qdrant_rag)" ]; then
    echo "Qdrant container is already running."
else
    if [ "$(docker ps -aq -f name=qdrant_rag)" ]; then
        echo "Starting existing Qdrant container..."
        docker start qdrant_rag
    else
        echo "Creating and starting new Qdrant container..."
        docker run -d -p 6333:6333 --name qdrant_rag qdrant/qdrant
    fi
fi

echo "--- 2. Waiting for Qdrant to be ready ---"
until curl -s http://localhost:6333/health > /dev/null; do
  echo "Qdrant is unavailable - sleeping..."
  sleep 2
done
echo "Qdrant is up!"

echo "--- 3. Setting up environment and indexing data ---"
poetry run python setup.py

echo "--- 4. Running Smoke Tests ---"
poetry run python smoke_test.py

echo "--- Done ---"
echo "Note: The Qdrant container (qdrant_rag) is still running in the background."
echo "To stop it, run: docker stop qdrant_rag"

# =============================================================================
# Knowledge Assistant — Dockerfile
# Multi-stage build using CPU-only PyTorch to keep image size ~1.5 GB.
#
# To install optional LLM provider extras, append to the poetry install line:
#   --extras "groq anthropic"
# =============================================================================

# --- Stage 1: dependency builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copy dependency files first (cache layer)
COPY pyproject.toml poetry.lock ./

# Install CPU-only torch first (avoids pulling ~2 GB CUDA wheel)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (no venv inside container)
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --without dev --no-interaction --no-ansi

# --- Stage 2: runtime ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY . .

# Create writable log/report directories
RUN mkdir -p logs reports models

EXPOSE 8000

# Single worker is safe for a course project demo
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

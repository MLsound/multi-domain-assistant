#!/bin/bash
# =============================================================================
# Knowledge Assistant — All-in-one setup and launch script
#
# This script uses the standalone Qdrant Docker container approach (port 6333).
# For Docker Compose mode (full stack including the API in a container), use:
#   docker-compose up --build
#   poetry run python setup.py   # index documents into the containerised Qdrant
#
# Usage:
#   chmod +x run.sh
#   ./run.sh
# =============================================================================

set -e

# ─── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "============================================================"
echo "   Knowledge Assistant — Setup & Launch"
echo "============================================================"
echo ""

# ─── Step 1: Prerequisite checks ─────────────────────────────────────────────
info "Step 1: Checking prerequisites..."

if [ ! -f ".env" ]; then
    warn ".env file not found."
    warn "Copy .env.example to .env and set at least one API key:"
    warn "  cp .env.example .env"
    warn ""
    warn "Supported providers (set in priority order):"
    warn "  GOOGLE_API_KEY     — Google Gemini (free tier available)"
    warn "  ANTHROPIC_API_KEY  — Anthropic Claude (paid)"
    warn "  GROQ_API_KEY       — Groq (free ongoing tier)"
    warn "  OPENROUTER_API_KEY — OpenRouter (free :free models)"
    warn "  MOONSHOT_API_KEY   — Moonshot / Kimi"
    warn "  LLM_PROVIDER=ollama — Local Ollama (no API key needed)"
    error "Aborting: no .env file present."
fi

if ! command -v docker &> /dev/null; then
    error "Docker is not installed or not in PATH. Install from https://docs.docker.com/get-docker/"
fi

if ! command -v poetry &> /dev/null; then
    error "Poetry is not installed or not in PATH. Install from https://python-poetry.org/docs/"
fi

info "Prerequisites satisfied."

# ─── Step 2: Install / update Python dependencies ────────────────────────────
info "Step 2: Installing Python dependencies..."
poetry install --no-interaction
info "Dependencies ready."

# ─── Step 3: Start Qdrant ────────────────────────────────────────────────────
info "Step 3: Starting Qdrant vector database..."

if [ "$(docker ps -q -f name=qdrant_rag)" ]; then
    info "Qdrant container 'qdrant_rag' is already running."
else
    if [ "$(docker ps -aq -f name=qdrant_rag)" ]; then
        info "Starting existing Qdrant container..."
        docker start qdrant_rag
    else
        info "Creating and starting new Qdrant container..."
        docker run -d -p 6333:6333 --name qdrant_rag qdrant/qdrant
    fi
fi

info "Waiting for Qdrant to be ready..."
until curl -s http://localhost:6333/health > /dev/null 2>&1; do
    echo "  Qdrant not yet ready — retrying in 2s..."
    sleep 2
done
info "Qdrant is up at http://localhost:6333"

# ─── Step 4: Index documents and train MLP router ────────────────────────────
info "Step 4: Indexing documents and training MLP router..."
info "  (chunk_size=512, overlap=50 — this may take a few minutes)"
poetry run python setup.py
info "Indexing complete."

# ─── Step 5: Run unit and integration tests ───────────────────────────────────
info "Step 5: Running test suite..."
poetry run pytest tests/ -q --tb=short
info "All tests passed."

# ─── Step 6: Start the FastAPI server ────────────────────────────────────────
info "Step 6: Starting Knowledge Assistant API..."

# Kill any process already using port 8000
if lsof -Pi :8000 -sTCP:LISTEN -t &> /dev/null 2>&1; then
    warn "Port 8000 is already in use. Attempting to stop the existing process..."
    kill "$(lsof -Pi :8000 -sTCP:LISTEN -t)" 2>/dev/null || true
    sleep 1
fi

# Start uvicorn in background, redirect output to a log file
mkdir -p logs
poetry run uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    > logs/api.log 2>&1 &
API_PID=$!
echo "$API_PID" > logs/api.pid

info "API server started (PID $API_PID). Waiting for it to become ready..."
MAX_WAIT=30
WAITED=0
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        error "API server did not start within ${MAX_WAIT}s. Check logs/api.log for details."
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done

# Read and display the active provider from the health endpoint
PROVIDER=$(curl -s http://localhost:8000/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('active_provider','unknown'))" 2>/dev/null || echo "unknown")
info "API is ready — Active LLM provider: ${PROVIDER}"

# ─── Done — Usage summary ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo -e "   ${GREEN}Knowledge Assistant is running!${NC}"
echo "============================================================"
echo ""
echo "  API base URL   : http://localhost:8000"
echo "  Qdrant console : http://localhost:6333/dashboard"
echo "  API docs       : http://localhost:8000/docs"
echo ""
echo "  Quick query (curl):"
echo '    curl -s -X POST http://localhost:8000/query \'
echo '      -H "Content-Type: application/json" \'
echo '      -d '"'"'{"query": "What are the NFPA 855 clearance requirements for BESS?"}'"'"' | python3 -m json.tool'
echo ""
echo "  Health check   : curl http://localhost:8000/health"
echo "  Metrics        : curl http://localhost:8000/metrics"
echo ""
echo "  Interactive CLI:"
echo "    poetry run python chat.py"
echo ""
echo "  Run evaluation:"
echo "    poetry run python scripts/run_evaluation.py"
echo ""
echo "  Stop everything:"
echo "    kill \$(cat logs/api.pid) 2>/dev/null; docker stop qdrant_rag"
echo ""
echo "  API server log : logs/api.log"
echo "  Audit log      : logs/audit.jsonl"
echo "============================================================"

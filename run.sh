#!/bin/bash
# =============================================================================
# Knowledge Assistant - All-in-one setup and launch script
#
# On Windows (PowerShell / Git Bash / WSL):
#   This script detects the environment and delegates to run.ps1 when
#   running on Windows where Docker and Poetry are Windows-native tools.
#
# On Linux / macOS:
#   Runs the full setup pipeline directly (Qdrant, setup.py, tests, API).
#
# Usage:
#   chmod +x run.sh
#   ./run.sh
#
# To run directly on Windows without bash:
#   .\run.ps1
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Detect platform and delegate to PowerShell on Windows
# ---------------------------------------------------------------------------
OS="$(uname -s 2>/dev/null || echo Unknown)"

if [[ "$OS" == MINGW* ]] || [[ "$OS" == MSYS* ]] || [[ "$OS" == CYGWIN* ]]; then
    # Git Bash / MSYS2 / Cygwin on Windows
    echo "[INFO] Detected Windows (Git Bash / MSYS2). Launching run.ps1 via PowerShell..."
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(pwd -W)/run.ps1"
    exit $?
fi

if grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
    # WSL — Docker and Poetry are Windows executables, not available in WSL PATH
    echo "[INFO] Detected WSL environment. Launching run.ps1 via PowerShell..."
    WIN_PATH=$(wslpath -w "$(pwd)/run.ps1")
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$WIN_PATH"
    exit $?
fi

# ---------------------------------------------------------------------------
# Native Linux / macOS path
# ---------------------------------------------------------------------------

# Colours
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fatal() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "============================================================"
echo "   Knowledge Assistant - Setup & Launch"
echo "============================================================"
echo ""

# ── Step 1: Prerequisites ──────────────────────────────────────────────────
info "Step 1: Checking prerequisites..."

if [ ! -f ".env" ]; then
    warn ".env file not found."
    warn "Copy .env.example to .env and set at least one provider key:"
    warn "  cp .env.example .env"
    warn ""
    warn "Supported providers (priority order):"
    warn "  GOOGLE_API_KEY     - Google Gemini (free tier)"
    warn "  ANTHROPIC_API_KEY  - Anthropic Claude (paid)"
    warn "  GROQ_API_KEY       - Groq (free ongoing tier)"
    warn "  OPENROUTER_API_KEY - OpenRouter (free :free models)"
    warn "  MOONSHOT_API_KEY   - Moonshot / Kimi"
    warn "  LLM_PROVIDER=ollama - Local Ollama (no API key needed)"
    fatal "Aborting: no .env file present."
fi

command -v docker  >/dev/null 2>&1 || fatal "Docker not found. Install from https://docs.docker.com/get-docker/"
command -v poetry  >/dev/null 2>&1 || fatal "Poetry not found. Install from https://python-poetry.org/docs/"
command -v curl    >/dev/null 2>&1 || fatal "curl not found. Install curl and retry."

info "Prerequisites satisfied."

# ── Step 2: Install / update Python dependencies ───────────────────────────
info "Step 2: Installing Python dependencies..."
poetry install --no-interaction
info "Dependencies ready."

# ── Step 3: Start Qdrant ───────────────────────────────────────────────────
info "Step 3: Starting Qdrant vector database..."

if [ "$(docker ps -q -f name=qdrant_rag)" ]; then
    info "Qdrant container 'qdrant_rag' is already running."
elif [ "$(docker ps -aq -f name=qdrant_rag)" ]; then
    info "Starting existing Qdrant container..."
    docker start qdrant_rag
else
    info "Creating and starting new Qdrant container..."
    docker run -d -p 6333:6333 --name qdrant_rag qdrant/qdrant
fi

info "Waiting for Qdrant to be ready..."
MAX_WAIT=30
WAITED=0
until curl -sf http://localhost:6333/readyz > /dev/null; do
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        fatal "Qdrant did not become ready within ${MAX_WAIT}s."
    fi
    echo "  Qdrant not yet ready - retrying in 2s..."
    sleep 2
    WAITED=$((WAITED + 2))
done
info "Qdrant is up at http://localhost:6333"

# ── Step 4: Index documents and train MLP router ───────────────────────────
info "Step 4: Indexing documents and training MLP router..."
info "  (chunk_size=512, overlap=50 - this takes 3-8 minutes)"
poetry run python setup.py
info "Indexing complete."

# ── Step 5: Run test suite ─────────────────────────────────────────────────
info "Step 5: Running test suite (32 tests)..."
poetry run pytest tests/ -q --tb=short
info "All tests passed."

# ── Step 6: Start FastAPI server ───────────────────────────────────────────
info "Step 6: Starting Knowledge Assistant API..."
mkdir -p logs

# Release port 8000 if already bound
if command -v lsof >/dev/null 2>&1; then
    EXISTING_PID=$(lsof -ti :8000 -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "$EXISTING_PID" ]; then
        warn "Port 8000 is in use (PID $EXISTING_PID). Stopping..."
        kill "$EXISTING_PID" 2>/dev/null || true
        sleep 1
    fi
elif command -v ss >/dev/null 2>&1; then
    EXISTING_PID=$(ss -tlnp 'sport = :8000' 2>/dev/null | awk 'NR>1{print $NF}' | grep -oP 'pid=\K[0-9]+' | head -1 || true)
    if [ -n "$EXISTING_PID" ]; then
        warn "Port 8000 is in use (PID $EXISTING_PID). Stopping..."
        kill "$EXISTING_PID" 2>/dev/null || true
        sleep 1
    fi
fi

# Launch uvicorn in background
poetry run uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    > logs/api.log 2>&1 &
API_PID=$!
echo "$API_PID" > logs/api.pid

info "API server starting (PID $API_PID). Polling /health..."
MAX_WAIT=45
WAITED=0
API_READY=false
until $API_READY; do
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        warn "API did not respond within ${MAX_WAIT}s."
        warn "Check logs/api.log for details."
        warn "You can start the API manually with:"
        warn "  poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        API_READY=true
    else
        echo "  API not yet ready - retrying..."
    fi
done

if $API_READY; then
    PROVIDER=$(curl -sf http://localhost:8000/health \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('active_provider','unknown'))" 2>/dev/null \
        || echo "unknown")
    info "API is ready - Active LLM provider: ${PROVIDER}"
fi

# ── Usage summary ──────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "   Knowledge Assistant is running!"
echo "============================================================"
echo ""
echo "  API base URL   : http://localhost:8000"
echo "  Qdrant console : http://localhost:6333/dashboard"
echo "  API docs       : http://localhost:8000/docs"
echo ""
echo "  Quick query (curl):"
echo '    curl -s -X POST http://localhost:8000/query \'
echo '      -H "Content-Type: application/json" \'
echo "      -d '{\"query\": \"What are the NFPA 855 clearance requirements for BESS?\"}' | python3 -m json.tool"
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

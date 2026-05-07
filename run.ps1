# =============================================================================
# Knowledge Assistant - All-in-one setup and launch script (PowerShell)
#
# Usage:
#   .\run.ps1
#
# For Docker Compose mode (API + Qdrant in containers):
#   docker-compose up --build
#   poetry run python setup.py
# =============================================================================

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Info  { param($msg) Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Warn  { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Fatal { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Knowledge Assistant - Setup & Launch" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Step 1: Prerequisite checks
# ---------------------------------------------------------------------------
Info "Step 1: Checking prerequisites..."

if (-not (Test-Path ".env")) {
    Warn ".env file not found."
    Warn "Copy .env.example to .env and configure at least one provider key:"
    Warn "  Copy-Item .env.example .env"
    Warn ""
    Warn "Supported providers (priority order):"
    Warn "  GOOGLE_API_KEY     - Google Gemini (free tier available)"
    Warn "  ANTHROPIC_API_KEY  - Anthropic Claude (paid, min USD5 deposit)"
    Warn "  GROQ_API_KEY       - Groq (free ongoing tier, no credit card)"
    Warn "  OPENROUTER_API_KEY - OpenRouter (free :free models)"
    Warn "  MOONSHOT_API_KEY   - Moonshot / Kimi (verify regional access)"
    Warn "  LLM_PROVIDER=ollama - Local Ollama (no API key needed)"
    Fatal "Aborting: no .env file found."
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fatal "Docker is not installed or not in PATH. Install from https://docs.docker.com/get-docker/"
}

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Fatal "Poetry is not installed or not in PATH. Install from https://python-poetry.org/docs/"
}

# Check Docker daemon is actually running (not just installed)
$ErrorActionPreference = "SilentlyContinue"
$dockerPing = docker info 2>&1
$ErrorActionPreference = "Stop"
if ($LASTEXITCODE -ne 0) {
    Warn "Docker is installed but the Docker Desktop engine is not running."
    Warn ""
    Warn "Start Docker Desktop:"
    Warn "  1. Open Docker Desktop from the Start Menu, OR"
    Warn "  2. Run: Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'"
    Warn ""
    Warn "Wait for the whale icon in the system tray to stop animating, then re-run this script."
    Fatal "Aborting: Docker daemon is not reachable (//./pipe/dockerDesktopLinuxEngine not found)."
}

Info "Prerequisites satisfied."

# ---------------------------------------------------------------------------
# Step 1b: Validate .env - catch placeholder keys before wasting time
# ---------------------------------------------------------------------------
$placeholders = @(
    "your_gemini_api_key_here",
    "your_claude_api_key_here",
    "your_groq_api_key_here",
    "your_openrouter_api_key_here",
    "your_kimi_api_key_here"
)

$envContent  = Get-Content ".env" -ErrorAction SilentlyContinue
$activeLines = $envContent | Where-Object { $_ -notmatch "^#" -and $_.Trim() -ne "" }

# Build a dict of key=value pairs from active lines
$envVars = @{}
foreach ($line in $activeLines) {
    if ($line -match "^([^=]+)=(.*)$") {
        $envVars[$Matches[1].Trim()] = $Matches[2].Trim()
    }
}

$apiKeyNames = @("GOOGLE_API_KEY","ANTHROPIC_API_KEY","GROQ_API_KEY","OPENROUTER_API_KEY","MOONSHOT_API_KEY")
$llmProvider = $envVars["LLM_PROVIDER"]
$hasRealKey  = $false

foreach ($keyName in $apiKeyNames) {
    $val = $envVars[$keyName]
    if ($val -and $val -ne "" -and $placeholders -notcontains $val) {
        $hasRealKey = $true
        break
    }
}

if (-not $hasRealKey -and $llmProvider -ne "ollama") {
    Warn "No valid API key found in .env and LLM_PROVIDER is not set to 'ollama'."
    Warn ""
    Warn "All API key fields are either empty or still contain placeholder values."
    Warn "The system cannot generate responses without a working LLM provider."
    Warn ""
    Warn "Choose one of these options and edit .env:"
    Warn ""
    Warn "  Option A - Groq (FREE, no credit card, recommended):"
    Warn "    1. Register at https://console.groq.com"
    Warn "    2. Create an API key"
    Warn "    3. In .env, add:  GROQ_API_KEY=gsk_your_real_key_here"
    Warn "    4. Run:  poetry install --extras groq"
    Warn ""
    Warn "  Option B - Google Gemini (free tier):"
    Warn "    1. Get a key at https://aistudio.google.com/apikey"
    Warn "    2. In .env, replace: GOOGLE_API_KEY=your_actual_key_here"
    Warn ""
    Warn "  Option C - Ollama local (no API key, requires Ollama installed):"
    Warn "    1. Install from https://ollama.ai/download"
    Warn "    2. Run: ollama pull phi3:mini"
    Warn "    3. In .env, add: LLM_PROVIDER=ollama"
    Warn "    4. Run: poetry install --extras ollama"
    Fatal "Aborting: no usable LLM provider configured."
}

if ($hasRealKey) {
    Info "LLM provider key detected in .env."
} else {
    Info "LLM_PROVIDER=ollama detected - local inference mode."
}

# ---------------------------------------------------------------------------
# Step 2: Install / update Python dependencies
# ---------------------------------------------------------------------------
Info "Step 2: Installing Python dependencies..."
poetry install --no-interaction
if ($LASTEXITCODE -ne 0) { Fatal "poetry install failed." }
Info "Dependencies ready."

# ---------------------------------------------------------------------------
# Step 3: Start Qdrant
# ---------------------------------------------------------------------------
Info "Step 3: Starting Qdrant vector database..."

$qdrantRunning = docker ps -q -f name=qdrant_rag
if ($qdrantRunning) {
    Info "Qdrant container 'qdrant_rag' is already running."
} else {
    $qdrantExists = docker ps -aq -f name=qdrant_rag
    if ($qdrantExists) {
        Info "Starting existing Qdrant container..."
        docker start qdrant_rag
    } else {
        Info "Creating and starting new Qdrant container..."
        docker run -d -p 6333:6333 --name qdrant_rag qdrant/qdrant
    }
    if ($LASTEXITCODE -ne 0) { Fatal "Failed to start Qdrant container." }
}

Info "Waiting for Qdrant to be ready..."
$maxWait = 30
$waited  = 0
do {
    Start-Sleep -Seconds 2
    $waited += 2
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:6333/readyz" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        $qdrantReady = ($resp.StatusCode -eq 200)
    } catch {
        $qdrantReady = $false
    }
    if (-not $qdrantReady) { Write-Host "  Qdrant not yet ready - retrying..." }
} while (-not $qdrantReady -and $waited -lt $maxWait)

if (-not $qdrantReady) { Fatal "Qdrant did not become ready within ${maxWait}s." }
Info "Qdrant is up at http://127.0.0.1:6333"

# ---------------------------------------------------------------------------
# Step 4: Index documents and train MLP router
# ---------------------------------------------------------------------------
Info "Step 4: Indexing documents and training MLP router..."
Info "  (chunk_size=512, overlap=50 - this takes 3-8 minutes)"
poetry run python setup.py
if ($LASTEXITCODE -ne 0) { Fatal "setup.py failed. Check Qdrant connectivity and logs." }
Info "Indexing complete."

# ---------------------------------------------------------------------------
# Step 5: Run test suite
# ---------------------------------------------------------------------------
Info "Step 5: Running test suite (32 tests)..."
poetry run pytest tests/ -q --tb=short
if ($LASTEXITCODE -ne 0) { Fatal "Tests failed. Fix the failures before starting the API." }
Info "All tests passed."

# ---------------------------------------------------------------------------
# Step 6: Start FastAPI server
# ---------------------------------------------------------------------------
Info "Step 6: Starting Knowledge Assistant API..."

New-Item -ItemType Directory -Force -Path logs | Out-Null

# Stop any existing process on port 8000
$existingPort = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($existingPort) {
    Warn "Port 8000 is in use. Stopping the existing process..."
    $existingPort | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

# Resolve the absolute path to the log file (Start-Process needs a full path for redirect)
$logFile = (Resolve-Path -Path "." ).Path + "\logs\api.log"

# Start-Process creates a truly independent process that survives the script session.
# Start-Job runs inside the current PS session and dies with it.
$uvicornProc = Start-Process `
    -FilePath "poetry" `
    -ArgumentList "run", "uvicorn", "src.api.main:app",
                  "--host", "0.0.0.0",
                  "--port", "8000",
                  "--log-level", "info" `
    -WorkingDirectory (Get-Location).Path `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError  ($logFile -replace '\.log$', '_err.log') `
    -PassThru `
    -NoNewWindow

$uvicornProc.Id | Out-File -FilePath "logs\api.pid" -Encoding ascii
Info "API server starting (PID $($uvicornProc.Id)). Polling /health..."

$maxWait = 60
$waited  = 0
$apiReady = $false
do {
    Start-Sleep -Seconds 2
    $waited += 2
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        $apiReady = ($resp.StatusCode -eq 200)
    } catch {
        $apiReady = $false
    }
    if (-not $apiReady) { Write-Host "  API not yet ready - retrying (${waited}s)..." }
} while (-not $apiReady -and $waited -lt $maxWait)

if (-not $apiReady) {
    Warn "API did not respond within ${maxWait}s."
    Warn "Check logs\api.log and logs\api_err.log for details."
    Warn "You can start the API manually with:"
    Warn "  poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"
} else {
    # Read active provider from health endpoint
    try {
        $health   = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
        $provider = $health.active_provider
    } catch {
        $provider = "unknown"
    }
    Info "API is ready - Active LLM provider: $provider"
}

# ---------------------------------------------------------------------------
# Usage summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Knowledge Assistant is running!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  API base URL   : http://localhost:8000"
Write-Host "  Qdrant console : http://localhost:6333/dashboard"
Write-Host "  API docs       : http://localhost:8000/docs"
Write-Host ""
Write-Host "  Quick query (PowerShell):"
Write-Host '    $body = ''{"query": "What are the NFPA 855 clearance requirements for BESS?"}'''
Write-Host '    Invoke-RestMethod -Uri http://localhost:8000/query -Method Post -ContentType "application/json" -Body $body'
Write-Host ""
Write-Host "  Health check   : Invoke-RestMethod http://localhost:8000/health"
Write-Host "  Metrics        : Invoke-RestMethod http://localhost:8000/metrics"
Write-Host ""
Write-Host "  Interactive CLI:"
Write-Host "    poetry run python chat.py"
Write-Host ""
Write-Host "  Run evaluation:"
Write-Host "    poetry run python scripts/run_evaluation.py"
Write-Host ""
Write-Host "  Stop everything:"
if ($uvicornProc -and -not $uvicornProc.HasExited) {
    Write-Host "    Stop-Process -Id $($uvicornProc.Id)   # stop API server"
} else {
    Write-Host "    Stop-Process -Id (Get-Content logs\api.pid)   # stop API server"
}
Write-Host "    docker stop qdrant_rag"
Write-Host ""
Write-Host "  API server log : logs\api.log"
Write-Host "  API error log  : logs\api_err.log"
Write-Host "  Audit log      : logs\audit.jsonl"
Write-Host "============================================================" -ForegroundColor Cyan

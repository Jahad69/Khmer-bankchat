#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_all.sh
# Launches both the FastAPI backend and the Streamlit frontend
# in background processes. Logs are written to logs/ directory.
#
# Usage:
#   chmod +x start_all.sh
#   ./start_all.sh
#
# Stop everything:
#   ./stop_all.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  🏦  Khmer BankChat – Starting all services"
echo "════════════════════════════════════════════════════════════"

# ── 1. FastAPI Backend ──────────────────────────────────────────────────────
echo "  [1/2] Starting FastAPI backend on :8000 …"
uvicorn backend.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    > logs/api.log 2>&1 &
API_PID=$!
echo "        PID $API_PID  →  logs/api.log"

# Wait briefly to make sure the API is up before starting the UI
sleep 4

# ── 2. Streamlit Frontend ───────────────────────────────────────────────────
echo "  [2/2] Starting Streamlit UI on :8501 …"
streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    > logs/ui.log 2>&1 &
UI_PID=$!
echo "        PID $UI_PID  →  logs/ui.log"

# ── Save PIDs for stop_all.sh ───────────────────────────────────────────────
echo "$API_PID" > logs/api.pid
echo "$UI_PID"  > logs/ui.pid

echo ""
echo "────────────────────────────────────────────────────────────"
echo "  ✅  Both services running!"
echo "  🔗  API  : http://localhost:8000"
echo "  📖  Docs : http://localhost:8000/docs"
echo "  💬  UI   : http://localhost:8501"
echo "────────────────────────────────────────────────────────────"
echo "  Run  ./stop_all.sh  to shut everything down."
echo ""

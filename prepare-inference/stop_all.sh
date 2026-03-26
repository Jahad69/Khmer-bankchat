#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# stop_all.sh  –  Gracefully stop all BankChat services
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

stop_pid() {
    local label="$1"
    local pidfile="$2"
    if [[ -f "$pidfile" ]]; then
        PID=$(cat "$pidfile")
        if kill -0 "$PID" 2>/dev/null; then
            echo "  Stopping $label (PID $PID) …"
            kill "$PID"
        else
            echo "  $label (PID $PID) already stopped."
        fi
        rm -f "$pidfile"
    else
        echo "  No PID file found for $label"
    fi
}

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  🛑  Khmer BankChat – Stopping all services"
echo "════════════════════════════════════════════════════════════"

stop_pid "FastAPI backend" "logs/api.pid"
stop_pid "Streamlit UI"    "logs/ui.pid"

echo "  ✅  Done."
echo ""

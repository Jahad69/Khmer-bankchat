#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_ui.sh
# Launch the Streamlit frontend on port 8501.
# Make sure start_api.sh is already running in another terminal first.
#
# Usage:
#   chmod +x start_ui.sh
#   ./start_ui.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "────────────────────────────────────────────────────────────"
echo "  🏦  Khmer BankChat  ·  Streamlit UI"
echo "  URL : http://localhost:8501"
echo "────────────────────────────────────────────────────────────"

# Optional: activate a conda / venv environment
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate bankchat

streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false

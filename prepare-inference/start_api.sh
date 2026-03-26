#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_api.sh
# Launch the FastAPI inference backend on port 8000.
#
# Usage:
#   chmod +x start_api.sh
#   ./start_api.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Loaded environment from .env"
fi

echo "────────────────────────────────────────────────────────────"
echo "  🏦  Khmer BankChat  ·  FastAPI Backend"
echo "  URL : http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"
echo "  Model Path: ${MODEL_ROOT_PATH:-<default>}"
echo "────────────────────────────────────────────────────────────"

# Optional: activate a conda / venv environment
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate bankchat

uvicorn backend.api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info

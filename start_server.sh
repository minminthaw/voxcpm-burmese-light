#!/usr/bin/env bash
# ==============================================================================
# Headless Server Launcher for Production Background Services & Cloud Deployments
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "../VoxCPM/.venv/bin/activate" ]; then
    source ../VoxCPM/.venv/bin/activate
fi

PORT="${1:-7860}"

echo "============================================================="
echo "Starting VoxCPM2 Headless OpenAI Speech API Server..."
echo "  Port:      $PORT"
echo "  Endpoint:  POST /v1/audio/speech"
echo "  Health:    GET  /health"
echo "============================================================="

exec python webui.py --port "$PORT"

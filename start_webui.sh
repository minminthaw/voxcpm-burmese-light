#!/usr/bin/env bash
# ==============================================================================
# Launcher for VoxCPM2 Lightweight Web UI & Speech API Server
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
echo "Starting VoxCPM2 Lightweight Studio & Speech API Server..."
echo "  👉 Web UI:    http://localhost:$PORT"
echo "  👉 TTS API:   POST http://localhost:$PORT/v1/audio/speech"
echo "  👉 Memory:    ~3.5 GB (INT8 Dynamic Mode Active)"
echo "============================================================="

# If on macOS, attempt to open browser after short delay
if [[ "$OSTYPE" == "darwin"* ]]; then
    (sleep 2 && open "http://localhost:$PORT") >/dev/null 2>&1 &
fi

python webui.py --port "$PORT"

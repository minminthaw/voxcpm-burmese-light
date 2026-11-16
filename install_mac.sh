#!/usr/bin/env bash
# ==============================================================================
# 1-Command Installer for macOS (Apple Silicon M1/M2/M3/M4 & Intel)
# VoxCPM2 Lightweight (Burmese + English)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================="
echo "  VoxCPM2 Lightweight - macOS 1-Command Installer"
echo "  Target: Burmese (my-MM) & English (en) Speech Synthesis"
echo "============================================================="

# 1. Check Python version
PYTHON_CMD=""
for cmd in python3.11 python3.12 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PY_VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ] && [ "$MINOR" -le 12 ]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Python 3.10, 3.11, or 3.12 is required."
    echo "Please install it via Homebrew: brew install python@3.11"
    exit 1
fi
echo "✓ Found compatible Python: $PYTHON_CMD ($PY_VER)"

# 2. Check ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Installing ffmpeg via Homebrew..."
    if command -v brew >/dev/null 2>&1; then
        brew install ffmpeg
    else
        echo "⚠️ Homebrew not found. Please install ffmpeg manually: https://ffmpeg.org"
    fi
else
    echo "✓ ffmpeg is installed"
fi

# 3. Setup Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    "$PYTHON_CMD" -m venv .venv
fi

source .venv/bin/activate
echo "✓ Virtual environment activated"

# 4. Install Dependencies
echo "Installing Python dependencies (with Apple Silicon Metal MPS support)..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download & Prune Model Weights
echo "Setting up VoxCPM2 Lightweight Burmese + English weights..."
python download_models.py

echo ""
echo "============================================================="
echo "🎉 Installation completed successfully!"
echo "To start the Web UI & Speech API, run:"
echo "    ./start_webui.sh"
echo "Then open: http://localhost:7860"
echo "============================================================="

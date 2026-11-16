#!/usr/bin/env bash
# ==============================================================================
# 1-Command Installer for Ubuntu Server (CPU & NVIDIA GPU)
# VoxCPM2 Lightweight (Burmese + English)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================="
echo "  VoxCPM2 Lightweight - Ubuntu Server 1-Command Installer"
echo "  Target: Burmese (my-MM) & English (en) Speech Synthesis"
echo "============================================================="

# 1. Install System Dependencies
echo "[1/4] Installing system dependencies (ffmpeg, libsndfile)..."
if command -v sudo >/dev/null 2>&1; then
    sudo apt update && sudo apt install -y python3 python3-pip python3-venv ffmpeg libsndfile1 git curl
else
    apt update && apt install -y python3 python3-pip python3-venv ffmpeg libsndfile1 git curl
fi

# 2. Setup Virtual Environment
echo "[2/4] Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip

# 3. Detect Hardware Backend (CUDA vs CPU)
echo "[3/4] Detecting compute hardware..."
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "✓ NVIDIA GPU detected! Installing PyTorch with CUDA acceleration..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
else
    echo "ℹ No NVIDIA GPU detected. Installing lightweight CPU-only PyTorch (optimized for Intel/AMD)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
fi

# 4. Download & Prune Model Weights
echo "[4/4] Setting up VoxCPM2 Lightweight Burmese + English weights..."
python download_models.py

echo ""
echo "============================================================="
echo "🎉 Ubuntu Server Setup Completed Successfully!"
echo ""
echo "To run Web UI & API interactively:"
echo "    ./start_webui.sh 7860"
echo ""
echo "To run headless as a background service:"
echo "    ./start_server.sh 7860"
echo ""
echo "To configure systemd service daemon:"
echo "    sudo cp voxcpm.service /etc/systemd/system/"
echo "    sudo systemctl enable --now voxcpm"
echo "============================================================="

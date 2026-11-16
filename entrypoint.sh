#!/usr/bin/env bash
set -e

# Check if model files exist; if not, auto-download from Hugging Face
if [ ! -f "pretrained_models/VoxCPM2_MM_Light/model.safetensors" ]; then
    echo "============================================================="
    echo "VoxCPM2-Burmese-Light: Downloading weights from Hugging Face..."
    echo "Source: https://huggingface.co/minthura635/voxmmlite"
    echo "============================================================="
    python download_models.py
fi

echo "Starting VoxCPM2 WebUI & OpenAI Speech API Server on port ${PORT:-7860}..."
exec python webui.py --port "${PORT:-7860}"

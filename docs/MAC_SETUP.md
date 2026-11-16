# macOS Setup & User Guide

This guide covers setting up and running **VoxCPM2 Lightweight (Burmese + English)** on Apple Silicon (M1 / M2 / M3 / M4) and Intel Macs.

---

## 1. Quick Start (1-Command Installation)

Open Terminal in the cloned repository directory and run:

```bash
./install_mac.sh
```

### What this script automatically does:
1. Detects your Python version (requires Python 3.10, 3.11, or 3.12).
2. Verifies and installs `ffmpeg` via Homebrew if needed.
3. Creates an isolated `.venv` virtual environment.
4. Installs PyTorch with native **Apple Silicon Metal (MPS)** acceleration.
5. Downloads the foundation weights and applies Burmese + English vocabulary pruning (-28,930 multilingual tokens).
6. Configures optimized KV-cache context.

---

## 2. Starting the Web UI & API

Run the startup launcher:

```bash
./start_webui.sh
```

* **Web Dashboard:** Automatically opens at `http://localhost:7860`
* **OpenAI Compatible Endpoint:** `POST http://localhost:7860/v1/audio/speech`
* **Memory Mode:** Runs in **~3.5 GB RAM mode** by default (slashing the original 10.5 GB footprint by 65%).

---

## 3. Command Line Interface (CLI) Usage

For batch processing, long story synthesis, or scripting without a browser:

```bash
# Basic synthesis (WAV):
python run_tts.py --text "မင်္ဂလာပါခင်ဗျာ။" --output my_speech.wav

# Synthesize a multi-sentence short story with Auto Voice-Lock:
python run_tts.py --text-file stories/tortoise_and_hare.txt --output story.wav

# Zero-Shot Voice Cloning (using a reference audio):
python run_tts.py --text "မင်္ဂလာပါခင်ဗျာ" --ref-audio examples/reference_speaker.wav --output cloned.wav
```

---

## 4. Hardware Resource & Performance on Mac

| Hardware | RAM Consumption | Speed (CFM 5-Step) | Audio Quality |
|---|---|---|---|
| **Apple M1 / M2 / M3 / M4** | **~3.5 GB Unified Memory** | **~1.8x Speedup** (~1.4s/token) | **48,000 Hz Studio** |
| Original Model Baseline | ~10.5 GB Unified Memory | Baseline (~2.8s/token) | 48,000 Hz Studio |

---

## 5. Troubleshooting on macOS

* **Issue: `Command not found: brew`**  
  Install Homebrew from [brew.sh](https://brew.sh):
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
* **Issue: Python version mismatch**  
  Install Python 3.11 via Homebrew:
  ```bash
  brew install python@3.11
  ```

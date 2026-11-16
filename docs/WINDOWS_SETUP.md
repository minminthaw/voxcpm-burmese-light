# Windows 10 / 11 Installation & Usage Guide

VoxCPM2 Lightweight (Burmese + English) can run natively on Windows 10/11 using Python and standard batch scripts, supporting both **NVIDIA CUDA GPUs** (RTX 3060/4060/4070/4080/4090 or laptop GPUs) and **CPU-only mode**.

---

## ⚡ Prerequisites

1. **Python 3.10 or 3.11** (Recommended: [Python 3.11 from python.org](https://www.python.org/downloads/)):
   > ⚠️ **Important:** During installation, check the box **"Add python.exe to PATH"**.
2. **Git for Windows** (Optional but recommended): [gitforwindows.org](https://gitforwindows.org/)
3. **FFmpeg**: Handled automatically by the installer via `winget`, or install manually from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/).

---

## 🚀 1-Click Automated Setup

1. Open PowerShell or Command Prompt, then clone the repository:
   ```cmd
   git clone https://github.com/minminthaw/voxcpm-burmese-light.git
   cd voxcpm-burmese-light
   ```

2. Run the 1-click installer:
   ```cmd
   install_windows.bat
   ```
   *(Or simply double-click `install_windows.bat` in Windows File Explorer)*

### What the installer handles automatically:
- ✅ Verifies Python 3.10/3.11.
- ✅ Installs `ffmpeg` via Windows Package Manager (`winget`) if missing.
- ✅ Creates a dedicated Python virtual environment (`.venv`).
- ✅ **Hardware auto-detection**:
  - If an NVIDIA GPU is detected (`nvidia-smi`), installs PyTorch with **CUDA 12.1 acceleration**.
  - If no GPU is found, installs CPU-optimized PyTorch.
- ✅ Automatically downloads pre-pruned model weights from Hugging Face (`minthura635/voxmmlite`).

---

## 🎙️ Launching the Web UI & API

Once installed, you never need to run `install_windows.bat` again.

Simply double-click:
```cmd
start_webui.bat
```
*(Or run `.\start_webui.bat` in Command Prompt/PowerShell)*

This will:
1. Activate `.venv`.
2. Automatically launch your default web browser to `http://localhost:7860`.
3. Start the Web UI & OpenAI-compatible Speech API (`POST http://localhost:7860/v1/audio/speech`).

---

## 💻 Command Line (CLI) Usage

Generate audio directly from Command Prompt:

```cmd
.venv\Scripts\activate.bat

python run_tts.py --text "မင်္ဂလာပါခင်ဗျာ၊ Windows မှာ အောင်မြင်စွာ အလုပ်လုပ်နေပါပြီ။" --output outputs/windows_test.wav
```

### Zero-Shot Voice Cloning on Windows:
```cmd
python run_tts.py --text "မင်္ဂလာပါခင်ဗျာ၊ ဒါကတော့ အသံတုပခြင်း စနစ်ဖြစ်ပါတယ်။" --ref_audio "examples/reference_speaker.wav" --output outputs/windows_clone.wav
```

---

## 🐧 Alternative: Windows Subsystem for Linux (WSL2)

If you prefer running in an Ubuntu Linux environment inside Windows with direct NVIDIA GPU passthrough:

1. Enable WSL2 in PowerShell:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
2. Open Ubuntu terminal, then follow [UBUNTU_SERVER.md](file:///Users/macbookprom1pro/My%20Projects/CreatorPlatform/voxcpm-burmese-light/docs/UBUNTU_SERVER.md):
   ```bash
   git clone https://github.com/minminthaw/voxcpm-burmese-light.git
   cd voxcpm-burmese-light
   chmod +x install_ubuntu.sh start_webui.sh
   ./install_ubuntu.sh
   ./start_webui.sh
   ```

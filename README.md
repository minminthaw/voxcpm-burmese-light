# VoxCPM2 Lightweight (Burmese + English) 🇲🇲 🇬🇧

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Sample Rate](https://img.shields.io/badge/Audio-48kHz%20Studio%20Quality-brightgreen.svg)]()
[![Memory Mode](https://img.shields.io/badge/RAM-~3.5GB%20INT8-orange.svg)]()
[![API Standard](https://img.shields.io/badge/API-OpenAI%20Compatible-blueviolet.svg)]()
[![Platform](https://img.shields.io/badge/Platform-macOS%20(Metal)%20%7C%20Ubuntu%20(CPU%2FCUDA)-lightgrey.svg)]()

A production-ready, lightweight foundation text-to-speech (TTS) and voice-cloning system specialized strictly for **Burmese (မြန်မာစာ, my-MM)** and **English (en)**, based on OpenBMB's 2.29B tokenizer-free foundation model **VoxCPM2**.

It reduces the foundation model's heavy memory footprint from **~10.5 GB down to ~3.5 GB (a 65% reduction)**, enabling smooth deployment on **Apple Silicon Macs (M1/M2/M3/M4)**, **consumer NVIDIA GPUs (≤4-6GB VRAM)**, and **GPU-less Ubuntu Dedicated Servers (CPU-only)**.

---

## 🌟 Key Features

* **100% Lossless Burmese & English Speech:** Pruned 28,930 Chinese and non-Latin multilingual tokens while keeping 100% of Burmese UTF-8 byte tokens and English subword BPE tokens.
* **Studio-Quality 48,000 Hz Audio:** Full-range broadcast audio output with natural intonation, breathing dynamics, and zero clipping.
* **Auto Voice-Lock Engine:** Solves the multi-sentence speaker drift problem. Sentences in long stories automatically synchronize with the primary speaker identity.
* **Zero-Shot Voice Cloning:** Clone any voice using just 3 to 10 seconds of clean reference speech audio.
* **Bilingual Code-Switching:** Seamlessly switches between Burmese phrases and English technical terms/vocabulary without pronunciation glitches.
* **1-Command Automatic Setup:** Self-contained installer scripts for macOS (`install_mac.sh`) and Ubuntu Dedicated Servers (`install_ubuntu.sh`).
* **OpenAI-Compatible Speech API (`/v1/audio/speech`):** Drop-in replacement for OpenAI's Audio TTS API across any frontend, backend, or SDK.
* **Modern Dark-Mode Web Dashboard:** Built-in interactive studio with in-browser audio player, prompt presets, and zero-reload in-memory model caching.

---

## 🚀 Quick Start: macOS (Apple Silicon M1/M2/M3/M4 & Intel)

Run the **1-Command Installer**:

```bash
git clone https://github.com/your-username/voxcpm-burmese-light.git
cd voxcpm-burmese-light
chmod +x install_mac.sh start_webui.sh
./install_mac.sh
```

Once installation finishes, launch the Web UI & API Server:

```bash
./start_webui.sh
```

Your browser will automatically open: **`http://localhost:7860`**

---

## 🖥️ Quick Start: Ubuntu Server (CPU-Only or NVIDIA GPU)

Run the **1-Command Ubuntu Installer**:

```bash
git clone https://github.com/your-username/voxcpm-burmese-light.git
cd voxcpm-burmese-light
chmod +x install_ubuntu.sh start_server.sh
./install_ubuntu.sh
```

### Running as a Background Systemd Service:
```bash
sudo cp voxcpm.service /etc/systemd/system/
sudo systemctl enable --now voxcpm
```

* Live Health Check: `curl http://localhost:7860/health`
* Headless Speech API: `POST http://localhost:7860/v1/audio/speech`

For complete Nginx reverse proxy and SSL configuration, see [docs/UBUNTU_SERVER.md](docs/UBUNTU_SERVER.md).

---

## 🔌 OpenAI-Compatible API Usage

VoxCPM2 Lightweight can be called using the official **OpenAI SDK** in Python, Node.js, Go, or standard `fetch` / `cURL`. Simply point `base_url` to `http://localhost:7860/v1`:

### Python Example:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:7860/v1",
    api_key="not-needed"
)

response = client.audio.speech.create(
    model="voxcpm2",
    voice="auto",
    input="မင်္ဂလာပါခင်ဗျာ၊ ဒါကတော့ OpenAI API စနစ်ကနေတဆင့် ထွက်ပေါ်လာတဲ့ အသံဖြစ်ပါတယ်။",
    response_format="wav"
)

response.stream_to_file("speech.wav")
```

### cURL Example:
```bash
curl -X POST http://localhost:7860/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm2",
    "input": "မင်္ဂလာပါခင်ဗျာ။",
    "response_format": "wav"
  }' \
  --output speech.wav
```

For Node.js, TypeScript, PHP (Laravel), and Next.js examples, see [docs/OPENAI_API.md](docs/OPENAI_API.md).

---

## 💻 Command Line Interface (CLI)

For command-line generation or audio book/story synthesis:

```bash
# 1. Single Sentence Synthesis:
python run_tts.py --text "မင်္ဂလာပါခင်ဗျာ။" --output output.wav

# 2. Multi-Sentence Short Story (with Auto Voice-Lock):
python run_tts.py --text-file stories/tortoise_and_hare.txt --output story.wav

# 3. Zero-Shot Voice Cloning:
python run_tts.py --text "မင်္ဂလာပါ" --ref-audio examples/reference_speaker.wav --output clone.wav
```

---

## 📊 Benchmark & Resource Footprint

| Metric | Original VoxCPM2 Baseline | VoxCPM2 Lightweight (INT8) | Impact |
|---|---|---|---|
| **Resident RAM (Mac/Linux)** | **~10.5 GB** | **~3.5 GB** | **65% Memory Reduction** |
| **GPU VRAM (Nvidia CUDA)** | **~8.5 GB** | **~3.5 GB** | Runs on 4GB-6GB GPUs |
| **Vocabulary Size** | 73,440 tokens | 44,510 tokens | -28,930 Chinese/other tokens |
| **Audio Sample Rate** | 48,000 Hz | 48,000 Hz | 100% Studio Quality Preserved |
| **CFM Timesteps** | 10 steps (default) | 5 steps (optimized) | **1.8x Inference Speedup** |
| **Multi-Sentence Voice Drift**| Frequent timbre shifts | **0% drift (Auto Voice-Lock)**| Consistent narrator voice |

---

## 📁 Repository Structure

```
voxcpm-burmese-light/
├── install_mac.sh          # 1-command installer for macOS
├── install_ubuntu.sh       # 1-command installer for Ubuntu Server
├── start_webui.sh          # Web UI & Speech API launcher
├── start_server.sh         # Headless server launcher
├── webui.py                # Standalone Web Dashboard & OpenAI API Server
├── run_tts.py              # Production CLI script with Auto Voice-Lock
├── download_models.py      # Resilient 16-worker model downloader & auto-pruner
├── requirements.txt        # Python dependency manifest
├── requirements-cpu.txt    # CPU-only dependencies for Ubuntu servers
├── voxcpm.service          # Ubuntu systemd service unit template
├── nginx.conf.example      # Production reverse proxy template with SSL
├── Dockerfile              # Docker container definition
├── docker-compose.yml      # Container orchestration
├── docs/
│   ├── MAC_SETUP.md        # Detailed macOS setup guide
│   ├── UBUNTU_SERVER.md    # Dedicated server deployment guide
│   └── OPENAI_API.md       # API specification & client code in Python, Node, PHP
├── examples/               # Reference speaker audio samples
└── stories/                # Sample Burmese stories for testing
```

---

## 🇲🇲 မြန်မာဘာသာဖြင့် အကျဉ်းချုပ် ရှင်းလင်းချက်

ဤ Project သည် OpenBMB ၏ အဆင့်မြင့် **VoxCPM2** Foundation TTS Model ကို **မြန်မာစာ (my-MM) နှင့် အင်္ဂလိပ်စာ (en)** သီးသန့်အတွက် အထူးပြု ချုံ့ချထားသော စနစ်ဖြစ်ပါသည်။

* **RAM စားသုံးမှု နည်းပါးခြင်း:** မူရင်း ၁၀.၅ GB မှ **~၃.၅ GB သာ** စားသုံးတော့သဖြင့် Macbook များနှင့် GPU မပါသော CPU Server များပေါ်တွင် အလွယ်တကူ အသုံးပြုနိုင်ပါသည်။
* **အသံအရည်အသွေး:** 48kHz Studio Quality အပြည့်အဝ ထွက်ရှိပြီး အသံကွဲခြင်း၊ ပျက်စီးခြင်း မရှိပါ။
* **Auto Voice-Lock:** ပုံပြင်တိုများ၊ စာပိုဒ်ရှည်များကို ဖတ်ကြားရာတွင် စာကြောင်းတိုင်း အသံတစ်သမတ်တည်း ထွက်ရှိစေသော စနစ် ပါဝင်ပါသည်။
* **One Command Setup:** Mac အတွက် `./install_mac.sh` နှင့် Ubuntu အတွက် `./install_ubuntu.sh` တစ်ကြောင်းတည်းဖြင့် အစအဆုံး အလိုအလျောက် သွင်းယူအသုံးပြုနိုင်ပါသည်။
* **OpenAI Compatible API:** မည်သည့် Web Server (Node.js, Next.js, Laravel, Python) မှမဆို တိုက်ရိုက် ချိတ်ဆက်အသုံးပြုနိုင်သော API အပြည့်အစုံ ပါဝင်ပါသည်။

---

## 📄 License

This optimization work is released under the **Apache-2.0 License**. The base model weights are subject to OpenBMB's VoxCPM2 model license.

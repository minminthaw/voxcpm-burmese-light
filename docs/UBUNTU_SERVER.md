# Ubuntu Dedicated Server Setup & Production Deployment

This guide covers deploying **VoxCPM2 Lightweight (Burmese + English)** on Ubuntu Dedicated Servers (20.04, 22.04, 24.04 LTS), both on **CPU-only servers** (Intel Xeon / AMD EPYC) and **NVIDIA GPU instances**.

---

## 1. Quick Start (1-Command Installation)

SSH into your Ubuntu server, clone the repository, and run:

```bash
git clone https://github.com/your-username/voxcpm-burmese-light.git
cd voxcpm-burmese-light
chmod +x install_ubuntu.sh
./install_ubuntu.sh
```

### What the installer handles:
1. Installs OS packages: `ffmpeg`, `libsndfile1`, `python3-venv`, `git`, `curl`.
2. Hardware detection:
   - **NVIDIA GPU detected:** Automatically installs PyTorch with CUDA 12.1.
   - **CPU-Only Server:** Automatically installs lightweight CPU-only PyTorch with Intel/AMD AVX2 multi-threading.
3. Sets up an isolated `.venv` environment and installs all dependencies.
4. Downloads the foundation model with a 16-worker parallel chunk downloader and applies Burmese + English vocabulary pruning.

---

## 2. Running Headless in Production

### Option A: Direct Headless Launcher
```bash
./start_server.sh 7860
```

### Option B: Systemd Background Service Daemon (Recommended)

To run VoxCPM2 automatically on system boot and auto-restart on failures:

1. Copy the systemd service template:
```bash
sudo cp voxcpm.service /etc/systemd/system/voxcpm.service
```

2. Edit the path and username if your directory is not `/opt/voxcpm-burmese-light`:
```ini
[Service]
WorkingDirectory=/path/to/voxcpm-burmese-light
ExecStart=/path/to/voxcpm-burmese-light/.venv/bin/python webui.py --port 7860
User=ubuntu
```

3. Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now voxcpm
```

4. Check service status and live logs:
```bash
sudo systemctl status voxcpm
sudo journalctl -u voxcpm -f
```

---

## 3. Nginx Reverse Proxy with HTTPS

To make the OpenAI-compatible speech endpoint accessible over public domain with SSL (`https://tts.yourdomain.com`):

1. Copy the Nginx configuration:
```bash
sudo cp nginx.conf.example /etc/nginx/sites-available/voxcpm.conf
sudo ln -s /etc/nginx/sites-available/voxcpm.conf /etc/nginx/sites-enabled/
```

2. Test and reload Nginx:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

3. Enable free SSL certificates with Certbot:
```bash
sudo certbot --nginx -d tts.yourdomain.com
```

Now any web application across the internet can call:
```
POST https://tts.yourdomain.com/v1/audio/speech
```

---

## 4. Hardware Sizing & Capacity Planning

| Server Tier | Target Architecture | Worker Setup | RAM Footprint | Performance (RTF) |
|---|---|---|---|---|
| **Entry CPU Server** (4-8 vCPUs) | Intel Xeon / AMD EPYC | 1 Worker (FIFO Queue) | ~3.5 GB - 4.2 GB | ~8x - 12x RTF |
| **Mid-Tier Server** (16-32 vCPUs) | AMD EPYC / Ryzen | 2-3 Parallel Workers | ~7.0 GB - 10.5 GB | ~5x - 8x RTF |
| **GPU Server** (1x RTX 3060/4060/T4) | NVIDIA CUDA (8-16 GB VRAM) | 1-2 Workers | ~3.5 GB VRAM | **~0.8x - 1.5x RTF (Near Real-time)** |

---

## 5. Web Server Architecture Integration

```
[ User / Browser ] 
        │
        ▼ (HTTPS)
[ Web Server (Next.js / Laravel / Express / FastAPI) ]
        │
        ▼ (Internal Network HTTP: POST /v1/audio/speech)
[ VoxCPM2 TTS API Server (:7860) ]
        │
        ▼ (48kHz WAV / MP3 Stream)
[ Audio Player / File Storage ]
```

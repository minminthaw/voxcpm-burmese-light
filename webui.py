#!/usr/bin/env python3
"""
Production-Ready Web UI & OpenAI-Compatible REST API Server for VoxCPM2 (Burmese + English).
Zero external Web framework dependencies (Pure Python http.server).

Endpoints:
  - GET  /                     : Interactive Modern Dark-Mode Web Dashboard
  - POST /v1/audio/speech      : OpenAI Official TTS API Drop-in Replacement (WAV / MP3)
  - GET  /v1/models            : OpenAI Models Listing
  - GET  /health               : System & Model Health Status
  - POST /api/generate         : Web Dashboard Speech Generation
  - GET  /outputs/<filename>   : Audio Static Stream
"""

import os
import sys
import json
import time
import subprocess
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import threading
import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure voxcpm package is accessible
sys.path.insert(0, SCRIPT_DIR)

import torch
from voxcpm.model.voxcpm2 import VoxCPM2Model

DEFAULT_PORT = 7860
DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", "VoxCPM2_MM_Light")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ModelManager:
    """Manages loaded model instance and inference cache with thread safety."""
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.current_model_path = None
        self.current_mode = None
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    @classmethod
    def get(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ModelManager()
        return cls._instance

    def load_model(self, model_path=DEFAULT_MODEL_DIR, mode="int8"):
        if self.model is not None and self.current_model_path == model_path and self.current_mode == mode:
            return self.model

        with self._lock:
            if self.model is not None and self.current_model_path == model_path and self.current_mode == mode:
                return self.model
            print(f"[ModelManager] Loading model from: {model_path} (mode: {mode}, device: {self.device})...")
            t0 = time.time()
            self.model = VoxCPM2Model.from_local(model_path, optimize=False, device=self.device)
            if mode == "int8":
                from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
                quantize_(self.model.base_lm, Int8DynamicActivationInt8WeightConfig())
                quantize_(self.model.residual_lm, Int8DynamicActivationInt8WeightConfig())
                print(f"[ModelManager] Applied INT8 Quantization (~3.5GB RAM mode) in {time.time() - t0:.2f}s")
            self.current_model_path = model_path
            self.current_mode = mode
            return self.model

    def synthesize(self, text, ref_audio="", prompt_text="", timesteps=5, cfg=2.0, seed=42, pause=0.3, min_len=0, model_path=DEFAULT_MODEL_DIR, mode="int8"):
        with self._lock:  # Serialized lock guarantees zero KV-cache data races
            model = self.load_model(model_path, mode)
            
            # Split text into sentences for long-form narrative
            raw_lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
            sentences = []
            for line in raw_lines:
                parts = [p.strip() for p in line.split("။") if p.strip()]
                for p in parts:
                    if not p.endswith("။") and not p.endswith("."):
                        sentences.append(p + "။")
                    else:
                        sentences.append(p)
            if not sentences:
                sentences = [text]

            all_chunks = []
            pause_samples = int(model.sample_rate * pause)
            silence = np.zeros(pause_samples, dtype=np.float32)

            auto_ref_path = os.path.join(OUTPUT_DIR, f".auto_ref_{int(time.time() * 1000)}.wav")

            for i, sent in enumerate(sentences):
                current_ref = ref_audio
                # Auto Voice-Lock: use Sentence 1 as reference for subsequent sentences
                if not current_ref and i > 0 and os.path.exists(auto_ref_path):
                    current_ref = auto_ref_path

                gen_kwargs = {
                    "target_text": sent,
                    "reference_wav_path": current_ref,
                    "prompt_text": prompt_text,
                    "inference_timesteps": timesteps,
                    "cfg_value": cfg,
                    "seed": seed,
                }
                if min_len > 0:
                    gen_kwargs["min_len"] = min_len

                wav = model.generate(**gen_kwargs)
                chunk = wav.squeeze().cpu().numpy()

                if i == 0 and not ref_audio and len(sentences) > 1:
                    sf.write(auto_ref_path, chunk, model.sample_rate)

                all_chunks.append(chunk)
                if i < len(sentences) - 1 and pause > 0:
                    all_chunks.append(silence)

            if os.path.exists(auto_ref_path):
                try:
                    os.remove(auto_ref_path)
                except OSError:
                    pass

            final_wav = np.concatenate(all_chunks, axis=0)
            return final_wav, model.sample_rate

HTML_PAGE = """<!DOCTYPE html>
<html lang="my">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VoxCPM2 Lightweight Studio (Burmese & English)</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card: #151d30;
      --border: #263554;
      --primary: #38bdf8;
      --primary-hover: #0284c7;
      --accent: #a855f7;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #22c55e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 24px;
      line-height: 1.5;
    }
    .container { max-width: 920px; margin: 0 auto; }
    header {
      text-align: center;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
    }
    .badge {
      display: inline-block;
      padding: 4px 14px;
      background: rgba(56, 189, 248, 0.12);
      color: var(--primary);
      border: 1px solid var(--primary);
      border-radius: 999px;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 12px;
    }
    h1 { font-size: 28px; font-weight: 700; margin-bottom: 6px; letter-spacing: -0.5px; }
    .subtitle { color: var(--text-muted); font-size: 14px; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px;
      margin-bottom: 20px;
      box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.3);
    }
    .label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    textarea {
      width: 100%;
      height: 120px;
      background: #080c16;
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      padding: 12px;
      font-size: 15px;
      resize: vertical;
      font-family: inherit;
    }
    textarea:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2); }
    .presets {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .preset-btn {
      background: #1e293b;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .preset-btn:hover { background: #334155; border-color: var(--primary); }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
    select, input[type="number"], input[type="text"] {
      width: 100%;
      background: #080c16;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 10px;
      border-radius: 6px;
      font-size: 14px;
    }
    .range-header { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
    input[type="range"] { width: 100%; accent-color: var(--primary); }
    .btn-generate {
      width: 100%;
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color: #fff;
      font-weight: 700;
      font-size: 16px;
      padding: 14px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      box-shadow: 0 4px 16px rgba(56, 189, 248, 0.35);
      transition: transform 0.1s, opacity 0.2s;
    }
    .btn-generate:hover { opacity: 0.95; }
    .btn-generate:active { transform: scale(0.99); }
    .btn-generate:disabled { opacity: 0.5; cursor: not-allowed; }
    #status-bar {
      margin-top: 14px;
      font-size: 14px;
      color: var(--primary);
      text-align: center;
      min-height: 20px;
    }
    .player-card { display: none; margin-top: 20px; border-color: var(--primary); }
    audio { width: 100%; margin-top: 12px; border-radius: 8px; }
    .download-btn {
      display: inline-block;
      margin-top: 12px;
      padding: 8px 18px;
      background: var(--success);
      color: #fff;
      text-decoration: none;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      transition: opacity 0.2s;
    }
    .download-btn:hover { opacity: 0.9; }
    .api-hint {
      margin-top: 24px;
      font-size: 13px;
      color: var(--text-muted);
      text-align: center;
      border-top: 1px solid var(--border);
      padding-top: 16px;
    }
    code {
      background: #1e293b;
      padding: 2px 6px;
      border-radius: 4px;
      color: var(--primary);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">OpenBMB VoxCPM2 • Burmese (my-MM) & English (en)</div>
      <h1>Lightweight TTS Studio & API</h1>
      <p class="subtitle">48kHz Studio Quality • RAM ~3.5GB Mode • Auto Voice-Lock • OpenAI Compatible</p>
    </header>

    <div class="card">
      <label class="label">Text to Synthesize (စာသားရိုက်ထည့်ပါ)</label>
      <textarea id="text-input" placeholder="မင်္ဂလာပါခင်ဗျာ... (သို့မဟုတ် စာအုပ်/ပုံပြင် စာပိုဒ်ရှည်များကို ထည့်သွင်းပါ)"></textarea>
      <div class="presets">
        <button class="preset-btn" onclick="setPreset('greet')">🇲🇲 Greeting (နှုတ်ခွန်းဆက်)</button>
        <button class="preset-btn" onclick="setPreset('mixed')">🔤 Mixed (မြန်မာ + English)</button>
        <button class="preset-btn" onclick="setPreset('story')">📖 Short Story (ပုံပြင်တို ၅ ကြောင်း)</button>
        <button class="preset-btn" onclick="setPreset('english')">🇬🇧 English Style</button>
      </div>
    </div>

    <div class="card">
      <div class="grid">
        <div>
          <label class="label">Model Checkpoint</label>
          <select id="model-select">
            <option value="pretrained_models/VoxCPM2_MM_Light">VoxCPM2_MM_Light (⭐ Recommended)</option>
            <option value="pretrained_models/VoxCPM2_MM_Light_14L">VoxCPM2_MM_Light_14L (⚡ Ultra Fast 1.6x)</option>
            <option value="pretrained_models/VoxCPM2">VoxCPM2 Original Baseline (Full)</option>
          </select>
        </div>
        <div>
          <label class="label">Memory Mode</label>
          <select id="mode-select">
            <option value="int8">INT8 Quantization (~3.5GB RAM)</option>
            <option value="fp16">Full Precision FP16 (~10GB RAM)</option>
          </select>
        </div>
      </div>

      <div class="grid">
        <div>
          <div class="range-header">
            <span>CFM Timesteps (Speed vs Quality)</span>
            <span id="timesteps-val">5</span>
          </div>
          <input type="range" id="timesteps" min="3" max="15" value="5" oninput="document.getElementById('timesteps-val').innerText = this.value">
        </div>
        <div>
          <div class="range-header">
            <span>Pause Duration Between Sentences</span>
            <span id="pause-val">0.3s</span>
          </div>
          <input type="range" id="pause" min="0.1" max="1.0" step="0.1" value="0.3" oninput="document.getElementById('pause-val').innerText = this.value + 's'">
        </div>
      </div>

      <div style="margin-top: 10px;">
        <label class="label">Zero-Shot Voice Cloning (Reference Audio Path)</label>
        <input type="text" id="ref-audio" placeholder="examples/reference_speaker.wav (Optional)">
      </div>
    </div>

    <button class="btn-generate" id="btn-gen" onclick="generateSpeech()">Generate 48kHz Speech (အသံထုတ်ယူရန်)</button>
    <div id="status-bar"></div>

    <div class="card player-card" id="player-card">
      <label class="label" style="color: var(--success);">Generated Audio (48,000 Hz Studio Quality)</label>
      <audio id="audio-player" controls></audio>
      <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
        <a id="download-link" class="download-btn" href="#" download="speech.wav">⬇ Download 48kHz WAV</a>
        <span id="gen-stats" style="font-size: 13px; color: var(--text-muted);"></span>
      </div>
    </div>

    <div class="api-hint">
      OpenAI Compatible API Endpoint: <code>POST /v1/audio/speech</code> | Model Name: <code>voxcpm2</code>
    </div>
  </div>

  <script>
    const presets = {
      greet: "မင်္ဂလာပါခင်ဗျာ။ အခုဆိုရင် VoxCPM စနစ်က ပိုမို ပေါ့ပါးသွက်လက်စွာ အလုပ်လုပ်ဆောင်နိုင်ပါပြီ။",
      mixed: "မင်္ဂလာပါခင်ဗျာ၊ ကျွန်တော်ကတော့ AI Developer တစ်ယောက် ဖြစ်ပါတယ်။",
      story: "တခါတုန်းက သာယာတဲ့ တောအုပ်ကြီးတစ်ခုထဲမှာ ဉာဏ်ကောင်းတဲ့ ယုန်ကလေးတစ်ကောင်နဲ့ လိပ်ကလေးတစ်ကောင် ရှိကြပါတယ်။\\nတစ်နေ့မှာတော့ ယုန်ကလေးက သူ့ရဲ့အပြေးမြန်တာကို ဝါကြွားပြီး လိပ်ကလေးကို အပြေးပြိုင်ဖို့ စိန်ခေါ်ခဲ့ပါတယ်။\\nပြိုင်ပွဲစတင်ချိန်မှာ ယုန်ကလေးဟာ အလွန်မြန်မြန် ပြေးထွက်သွားပေမဲ့ လမ်းခုလတ် အပင်ရိပ်အောက်မှာ အိပ်ပျော်သွားခဲ့ပါတယ်။\\nလိပ်ကလေးကတော့ မနားတမ်း ဇွဲလုံ့လနဲ့ ဖြည်းဖြည်းချင်း လျှောက်လှမ်းခဲ့ရာ နောက်ဆုံးမှာတော့ ပန်းတိုင်ကို အရင်ဆုံး ရောက်ရှိသွားခဲ့ပါတယ်။\\nဒီပုံပြင်လေးက ပေးတဲ့သင်ခန်းစာကတော့ မဆုတ်မနစ်သော ဇွဲလုံ့လဟာ အောင်မြင်ခြင်းရဲ့ သော့ချက်ဖြစ်တယ် ဆိုတာပါပဲ။",
      english: "(A friendly young woman, warm and clear voice) Hello, this is VoxCPM running in lightweight mode with studio sound quality."
    };

    function setPreset(key) {
      document.getElementById('text-input').value = presets[key];
    }

    async function generateSpeech() {
      const text = document.getElementById('text-input').value.trim();
      if (!text) {
        alert('Please enter some text!');
        return;
      }
      const btn = document.getElementById('btn-gen');
      const status = document.getElementById('status-bar');
      const playerCard = document.getElementById('player-card');

      btn.disabled = true;
      status.innerText = "⏳ Synthesizing speech with VoxCPM2... Please wait.";

      const payload = {
        text: text,
        model_path: document.getElementById('model-select').value,
        mode: document.getElementById('mode-select').value,
        timesteps: parseInt(document.getElementById('timesteps').value),
        pause: parseFloat(document.getElementById('pause').value),
        ref_audio: document.getElementById('ref-audio').value.trim()
      };

      const t0 = performance.now();
      try {
        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.error) {
          status.innerText = "❌ Error: " + data.error;
        } else {
          const durSec = ((performance.now() - t0) / 1000).toFixed(1);
          status.innerText = "✅ Generated in " + durSec + "s! Audio Duration: " + data.duration.toFixed(2) + "s";
          document.getElementById('audio-player').src = data.audio_url + '?t=' + Date.now();
          document.getElementById('download-link').href = data.audio_url;
          document.getElementById('gen-stats').innerText = "Duration: " + data.duration.toFixed(2) + "s | Sample Rate: 48,000 Hz";
          playerCard.style.display = 'block';
          document.getElementById('audio-player').play();
        }
      } catch (err) {
        status.innerText = "❌ Connection failed: " + err;
      } finally {
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""

class SimpleTTSHandler(BaseHTTPRequestHandler):
    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
            return

        elif path == "/health":
            manager = ModelManager.get()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            resp = {
                "status": "healthy",
                "device": manager.device,
                "model_loaded": manager.model is not None,
                "active_mode": manager.current_mode,
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif path in ("/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            models_data = {
                "object": "list",
                "data": [
                    {"id": "voxcpm2", "object": "model", "created": 1700000000, "owned_by": "openbmb"},
                    {"id": "tts-1", "object": "model", "created": 1700000000, "owned_by": "openai-alias"},
                    {"id": "tts-1-hd", "object": "model", "created": 1700000000, "owned_by": "openai-alias"}
                ]
            }
            self.wfile.write(json.dumps(models_data).encode("utf-8"))
            return

        elif path.startswith("/outputs/"):
            filename = os.path.basename(path)
            filepath = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(filepath):
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(os.path.getsize(filepath)))
                self.send_cors_headers()
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "File not found")
                return

        elif path.startswith("/examples/"):
            filepath = os.path.join(SCRIPT_DIR, path.lstrip("/"))
            if os.path.isfile(filepath):
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(os.path.getsize(filepath)))
                self.send_cors_headers()
                self.end_headers()
                with open(filepath, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "File not found")
                return

        self.send_error(404, "Not Found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if path == "/api/generate":
            try:
                data = json.loads(body.decode("utf-8"))
                text = data.get("text", "")
                model_path = data.get("model_path", DEFAULT_MODEL_DIR)
                mode = data.get("mode", "int8")
                timesteps = int(data.get("timesteps", 5))
                pause = float(data.get("pause", 0.3))
                ref_audio = data.get("ref_audio", "")

                manager = ModelManager.get()
                wav_data, sample_rate = manager.synthesize(
                    text=text,
                    ref_audio=ref_audio,
                    timesteps=timesteps,
                    pause=pause,
                    model_path=model_path,
                    mode=mode
                )

                out_filename = f"speech_{int(time.time() * 1000)}.wav"
                out_path = os.path.join(OUTPUT_DIR, out_filename)
                sf.write(out_path, wav_data, sample_rate)
                duration = len(wav_data) / sample_rate

                response = {
                  "status": "success",
                  "audio_url": f"/outputs/{out_filename}",
                  "duration": duration,
                  "sample_rate": sample_rate
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif path == "/v1/audio/speech":
            # OpenAI-compatible TTS endpoint
            try:
                data = json.loads(body.decode("utf-8"))
                text = data.get("input", "")
                response_format = data.get("response_format", "wav").lower()
                voice = data.get("voice", "auto")

                ref_audio = ""
                if voice and voice != "auto" and os.path.exists(voice):
                    ref_audio = voice

                manager = ModelManager.get()
                wav_data, sample_rate = manager.synthesize(
                    text=text,
                    ref_audio=ref_audio,
                    timesteps=5,
                    mode="int8"
                )

                # Write to buffer
                import io
                buf = io.BytesIO()
                sf.write(buf, wav_data, sample_rate, format="WAV")
                buf.seek(0)
                wav_bytes = buf.read()

                if response_format == "mp3":
                    # Convert to MP3 using ffmpeg if available
                    proc = subprocess.Popen(
                        ["ffmpeg", "-y", "-i", "pipe:0", "-f", "mp3", "-b:a", "192k", "pipe:1"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    out_bytes, _ = proc.communicate(input=wav_bytes)
                    content_type = "audio/mpeg"
                    final_bytes = out_bytes if proc.returncode == 0 else wav_bytes
                else:
                    content_type = "audio/wav"
                    final_bytes = wav_bytes

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(final_bytes)))
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(final_bytes)
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_error(404, "Not Found")

def run_server(port=DEFAULT_PORT):
    server = ThreadingHTTPServer(("0.0.0.0", port), SimpleTTSHandler)
    print("=" * 65)
    print(f"VoxCPM2 Lightweight Web UI & Speech API Server running at:")
    print(f"  👉 Web Dashboard:  http://localhost:{port}")
    print(f"  👉 OpenAI TTS API:  POST http://localhost:{port}/v1/audio/speech")
    print(f"  👉 Health Check:   GET  http://localhost:{port}/health")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    run_server(args.port)

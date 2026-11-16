#!/usr/bin/env python3
"""
Automated Fast Model Downloader for VoxCPM2-Burmese-Light (Hosted on Hugging Face).
Downloads pre-pruned weights directly from minthura635/voxmmlite:
  - 16-worker multi-threaded chunk downloader with auto-resume.
  - Zero manual conversion or pruning needed.
"""

import os
import sys
import time
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", "VoxCPM2_MM_Light")

BASE_HF_URL = "https://huggingface.co/minthura635/voxmmlite/resolve/main"

FILES_TO_DOWNLOAD = [
    ("model.safetensors", 4461528640),
    ("audiovae.pth", 376829777),
    ("config.json", None),
    ("tokenizer.json", None),
    ("tokenizer_config.json", None),
    ("special_tokens_map.json", None),
    ("tokenization_voxcpm2.py", None),
]

CHUNK_SIZE = 16 * 1024 * 1024  # 16 MB
MAX_WORKERS = 16
RETRIES = 5

def download_chunk(url, chunk_idx, start, end, expected_len, chunk_path):
    if os.path.exists(chunk_path) and os.path.getsize(chunk_path) == expected_len:
        return chunk_idx, expected_len, True

    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
                if len(data) == expected_len:
                    tmp = chunk_path + ".tmp"
                    with open(tmp, "wb") as f:
                        f.write(data)
                    os.rename(tmp, chunk_path)
                    return chunk_idx, expected_len, False
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download chunk {chunk_idx} after {RETRIES} retries")

def parallel_download_file(filename, expected_size):
    url = f"{BASE_HF_URL}/{filename}"
    dest_path = os.path.join(TARGET_DIR, filename)
    chunk_dir = os.path.join(TARGET_DIR, f".chunks_{filename}")
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(TARGET_DIR, exist_ok=True)

    if os.path.exists(dest_path):
        actual_size = os.path.getsize(dest_path)
        if expected_size is None or actual_size == expected_size:
            print(f"  ✓ {filename} already downloaded ({actual_size / (1024**2):.1f} MB)")
            return dest_path

    # Get size
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get("Content-Length", expected_size or 0))

    if total_size < CHUNK_SIZE:
        print(f"  Downloading {filename} ({total_size / 1024:.1f} KB)...")
        urllib.request.urlretrieve(url, dest_path)
        return dest_path

    num_chunks = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  Downloading {filename} ({total_size / (1024**3):.2f} GB) in {num_chunks} parallel chunks...")

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for idx in range(num_chunks):
            start = idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE - 1, total_size - 1)
            clen = end - start + 1
            cpath = os.path.join(chunk_dir, f"chunk_{idx:04d}.bin")
            futures.append(pool.submit(download_chunk, url, idx, start, end, clen, cpath))

        done_bytes = 0
        t0 = time.time()
        for f in as_completed(futures):
            _, clen, _ = f.result()
            done_bytes += clen
            mb = done_bytes / (1024**2)
            pct = (done_bytes / total_size) * 100
            speed = mb / max(time.time() - t0, 0.1)
            print(f"\r    Progress: {mb:.1f} MB / {total_size / (1024**2):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", end="", flush=True)

    print("\n  Assembling chunks...")
    with open(dest_path, "wb") as out_f:
        for idx in range(num_chunks):
            cpath = os.path.join(chunk_dir, f"chunk_{idx:04d}.bin")
            with open(cpath, "rb") as cf:
                shutil.copyfileobj(cf, out_f, 4 * 1024 * 1024)
    shutil.rmtree(chunk_dir, ignore_errors=True)
    print(f"  ✓ {filename} assembled successfully ({os.path.getsize(dest_path) / (1024**3):.2f} GB)")
    return dest_path

def main():
    print("=" * 65)
    print("VoxCPM2-Burmese-Light Model Setup (minthura635/voxmmlite)")
    print("=" * 65)
    target_safetensors = os.path.join(TARGET_DIR, "model.safetensors")
    target_vae = os.path.join(TARGET_DIR, "audiovae.pth")

    if os.path.exists(target_safetensors) and os.path.exists(target_vae):
        print(f"✓ Light model already exists in: {TARGET_DIR}")
        print("  Size:", f"{os.path.getsize(target_safetensors) / (1024**3):.2f} GB")
        print("Setup complete! You can run ./start_webui.sh or python run_tts.py.")
        return

    print("\nDownloading Pre-Pruned Burmese + English Model Weights from Hugging Face...")
    for fname, size in FILES_TO_DOWNLOAD:
        parallel_download_file(fname, size)

    print("\n" + "=" * 65)
    print("SUCCESS: VoxCPM2-Burmese-Light is fully installed and ready!")
    print("Run WebUI / Speech API Server: ./start_webui.sh")
    print("=" * 65)

if __name__ == "__main__":
    main()

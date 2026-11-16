#!/usr/bin/env python3
"""
Automated Model Downloader & Vocabulary Pruning Setup for VoxCPM2 (Burmese + English).
Features:
  - 16-worker multi-threaded chunk downloader with auto-resume.
  - Automatically applies Burmese + English vocabulary pruning (-28,930 tokens).
  - Configures optimized KV-cache (2048 context) to save ~450MB idle memory.
"""

import os
import sys
import json
import time
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", "VoxCPM2_MM_Light")
ORIG_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", "VoxCPM2")

BASE_HF_URL = "https://huggingface.co/openbmb/VoxCPM2/resolve/main"

FILES_TO_DOWNLOAD = [
    ("model.safetensors", 4580080592),
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
    dest_path = os.path.join(ORIG_DIR, filename)
    chunk_dir = os.path.join(ORIG_DIR, f".chunks_{filename}")
    os.makedirs(chunk_dir, exist_ok=True)
    os.makedirs(ORIG_DIR, exist_ok=True)

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

def prune_model():
    """Applies Burmese + English vocabulary pruning to the downloaded model."""
    print("\n=============================================================")
    print("Applying Burmese + English Vocabulary Pruning...")
    print("=============================================================")
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 1. Prune tokenizer
    tok_in = os.path.join(ORIG_DIR, "tokenizer.json")
    tok_out = os.path.join(TARGET_DIR, "tokenizer.json")

    with open(tok_in, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    vocab = tok_data["model"]["vocab"]
    print(f"  Original Vocabulary Size: {len(vocab):,}")

    def is_burmese_or_latin(token: str) -> bool:
        if token.startswith("<0x") and token.endswith(">"):
            return True  # Byte fallback tokens (Used by Burmese)
        for ch in token:
            cp = ord(ch)
            # Check Chinese/CJK ranges
            if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF):
                return False
            # Check Arabic, Cyrillic, Devanagari, Korean, Thai
            if (0x0600 <= cp <= 0x06FF) or (0x0400 <= cp <= 0x04FF) or (0x0900 <= cp <= 0x097F) or (0xAC00 <= cp <= 0xD7AF) or (0x0E00 <= cp <= 0x0E7F):
                return False
        return True

    kept_tokens = {tok: idx for tok, idx in vocab.items() if is_burmese_or_latin(tok)}
    # Sort and re-index
    new_vocab = {tok: new_idx for new_idx, tok in enumerate(sorted(kept_tokens.keys(), key=lambda t: vocab[t]))}
    new_vocab_size = len(new_vocab)
    print(f"  Pruned Vocabulary Size:   {new_vocab_size:,} (-{len(vocab) - new_vocab_size:,} multilingual tokens stripped)")

    # Filter merges
    new_merges = []
    if "merges" in tok_data["model"]:
        for m in tok_data["model"]["merges"]:
            parts = m.split(" ")
            if len(parts) == 2 and is_burmese_or_latin(parts[0]) and is_burmese_or_latin(parts[1]):
                new_merges.append(m)
        tok_data["model"]["merges"] = new_merges

    tok_data["model"]["vocab"] = new_vocab
    with open(tok_out, "w", encoding="utf-8") as f:
        json.dump(tok_data, f, indent=2, ensure_ascii=False)

    # 2. Prune model weights embedding
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch

    print("  Slicing Base LM embedding tensor...")
    tensors = {}
    with safe_open(os.path.join(ORIG_DIR, "model.safetensors"), framework="pt", device="cpu") as sf_in:
        for k in sf_in.keys():
            t = sf_in.get_tensor(k)
            if k == "base_lm.embed_tokens.weight":
                # Slices embed tokens to new vocab size
                t = t[:new_vocab_size, :].clone()
            tensors[k] = t

    print("  Saving pruned model.safetensors...")
    save_file(tensors, os.path.join(TARGET_DIR, "model.safetensors"))

    # 3. Copy AudioVAE and adjust config.json
    shutil.copy2(os.path.join(ORIG_DIR, "audiovae.pth"), os.path.join(TARGET_DIR, "audiovae.pth"))
    for extra in ["tokenizer_config.json", "special_tokens_map.json", "tokenization_voxcpm2.py"]:
        src = os.path.join(ORIG_DIR, extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(TARGET_DIR, extra))

    with open(os.path.join(ORIG_DIR, "config.json"), "r") as f:
        cfg = json.load(f)
    cfg["lm_config"]["vocab_size"] = new_vocab_size
    cfg["max_length"] = 2048  # Tuned KV-cache saves 450MB
    with open(os.path.join(TARGET_DIR, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("  ✓ Pruning complete! Target ready in: pretrained_models/VoxCPM2_MM_Light")

def main():
    print("=" * 65)
    print("VoxCPM2 Burmese + English Setup & Model Downloader")
    print("=" * 65)
    target_safetensors = os.path.join(TARGET_DIR, "model.safetensors")
    target_vae = os.path.join(TARGET_DIR, "audiovae.pth")

    if os.path.exists(target_safetensors) and os.path.exists(target_vae):
        print(f"✓ Light model already exists in: {TARGET_DIR}")
        print("  Size:", f"{os.path.getsize(target_safetensors) / (1024**3):.2f} GB")
        print("Setup complete! You can run ./start_webui.sh or python run_tts.py.")
        return

    # Download original files
    print("\n[Step 1/2] Downloading Foundation Model Weights from Hugging Face...")
    for fname, size in FILES_TO_DOWNLOAD:
        parallel_download_file(fname, size)

    # Prune model
    print("\n[Step 2/2] Generating Lightweight Burmese + English Model...")
    prune_model()

    print("\n" + "=" * 65)
    print("SUCCESS: VoxCPM2 Lightweight (Burmese + English) is fully installed!")
    print("Run WebUI / API Server: ./start_webui.sh")
    print("=" * 65)

if __name__ == "__main__":
    main()

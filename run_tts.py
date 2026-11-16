#!/usr/bin/env python3
"""
Production-Ready Easy CLI for Lightweight VoxCPM2 (Burmese + English).
Supports single sentences as well as long stories (automatic sentence chunking).

Usage:
    # Basic TTS (INT8 mode: ~3.5GB RAM/VRAM):
    python run_tts.py --text "မင်္ဂလာပါခင်ဗျာ။" --output my_speech.wav

    # Synthesize a Short Story from text file:
    python run_tts.py --text-file stories/tortoise_and_hare.txt --output outputs/story_tortoise_hare.wav

    # Voice Design (English style):
    python run_tts.py --text "(A warm female voice) Hello, welcome to our AI speech system." --output english.wav

    # Voice Cloning (Reference Audio):
    python run_tts.py --text "ဒီအသံကတော့ အသံတူ ကူးယူထားတာ ဖြစ်ပါတယ်။" --ref-audio sample.wav --output clone.wav
"""

import os
import sys
import argparse
import time
import re
import numpy as np

# Ensure VoxCPM source is available in path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "VoxCPM", "src"))

import torch
import soundfile as sf
from voxcpm.model.voxcpm2 import VoxCPM2Model
from voxcpm.watermark import embed_watermark

DEFAULT_MODEL_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", "VoxCPM2_MM_Light")

def parse_args():
    parser = argparse.ArgumentParser(description="VoxCPM2 Lightweight TTS (Burmese + English)")
    parser.add_argument("--text", "-t", type=str, default="", help="Text to synthesize (Burmese or English)")
    parser.add_argument("--text-file", "-f", type=str, default="", help="Path to text file to synthesize (for stories/articles)")
    parser.add_argument("--output", "-o", type=str, default="outputs/output.wav", help="Output WAV file path")
    parser.add_argument("--ref-audio", "-r", type=str, default="", help="Path to reference WAV for voice cloning")
    parser.add_argument("--prompt-text", type=str, default="", help="Transcript of reference audio (optional)")
    parser.add_argument("--mode", type=str, choices=["int8", "fp16"], default="int8", help="Inference mode: int8 (low memory, ~3.5GB) or fp16 (full precision)")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_DIR, help="Path to model directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Compute device")
    parser.add_argument("--timesteps", type=int, default=5, help="CFM inference timesteps (default: 5 for fast, 10 for max)")
    parser.add_argument("--cfg", type=float, default=2.0, help="CFG guidance scale (default: 2.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pause", type=float, default=0.3, help="Pause duration (seconds) between story sentences (default: 0.3s)")
    parser.add_argument("--min-len", type=int, default=0, help="Minimum generated token steps (useful for pruned models)")
    parser.add_argument("--no-watermark", action="store_true", help="Disable imperceptible acoustic watermark")
    return parser.parse_args()

def split_into_sentences(text: str) -> list[str]:
    """Split Burmese / English text into natural sentences."""
    raw_lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    sentences = []
    for line in raw_lines:
        # Split by Burmese sentence terminator '။'
        parts = [p.strip() for p in line.split("။") if p.strip()]
        for p in parts:
            if not p.endswith("။") and not p.endswith("."):
                sentences.append(p + "။")
            else:
                sentences.append(p)
    return sentences if sentences else [text]

def main():
    args = parse_args()

    # Load text from file if provided or if args.text is a valid path
    input_text = ""
    if args.text_file and os.path.isfile(args.text_file):
        with open(args.text_file, "r", encoding="utf-8") as f:
            input_text = f.read().strip()
    elif args.text and os.path.isfile(args.text):
        with open(args.text, "r", encoding="utf-8") as f:
            input_text = f.read().strip()
    elif args.text:
        input_text = args.text.strip()
    else:
        print("Error: Please provide --text \"...\" or --text-file \"path/to/story.txt\"")
        sys.exit(1)

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    sentences = split_into_sentences(input_text)

    print("=" * 65)
    print("VoxCPM2 Lightweight TTS (Burmese & English)")
    print("=" * 65)
    print(f"Device:          {device}")
    print(f"Mode:            {args.mode.upper()}")
    print(f"Model Path:      {args.model_path}")
    print(f"Total Sentences: {len(sentences)}")
    print(f"Output File:     {args.output}")
    print(f"CFM Timesteps:   {args.timesteps}")
    print("=" * 65)

    if not os.path.exists(args.model_path):
        print(f"Error: Model directory not found: {args.model_path}")
        sys.exit(1)

    # 1. Load Model
    print("\n[1/3] Loading model into memory...")
    t0 = time.time()
    model = VoxCPM2Model.from_local(
        args.model_path,
        optimize=False,
        device=device,
    )
    print(f"Model loaded in {time.time() - t0:.2f}s")

    # 2. Apply INT8 Quantization if requested
    if args.mode == "int8":
        print("\n[2/3] Applying INT8 Quantization to Base LM & Residual LM...")
        t_q = time.time()
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
        quantize_(model.base_lm, Int8DynamicActivationInt8WeightConfig())
        quantize_(model.residual_lm, Int8DynamicActivationInt8WeightConfig())
        print(f"Quantized to INT8 in {time.time() - t_q:.2f}s (~3.5GB RAM mode active)!")

    # 3. Generate Audio for each sentence
    print(f"\n[3/3] Generating 48kHz audio for {len(sentences)} sentence(s)...")
    t_start = time.time()
    all_audio_chunks = []
    pause_samples = int(model.sample_rate * args.pause)
    silence_padding = np.zeros(pause_samples, dtype=np.float32)

    speaker_ref_audio = args.ref_audio
    auto_ref_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), ".auto_speaker_ref.wav")

    for i, sent in enumerate(sentences):
        print(f"\n--- [Sentence {i+1}/{len(sentences)}] ---")
        print(f"Text: {sent}")
        t_sent = time.time()
        
        # Determine reference audio for voice consistency
        current_ref_audio = speaker_ref_audio
        # If user did not provide reference audio, lock onto Sentence 1's voice for remaining sentences!
        if not current_ref_audio and i > 0 and os.path.exists(auto_ref_path):
            current_ref_audio = auto_ref_path
            print(f"[Voice Lock] Synchronizing voice with Sentence 1 speaker identity")

        gen_kwargs = {
            "target_text": sent,
            "reference_wav_path": current_ref_audio,
            "prompt_text": args.prompt_text,
            "inference_timesteps": args.timesteps,
            "cfg_value": args.cfg,
            "seed": args.seed,  # Fixed seed maintains consistent acoustic timbre
        }
        if args.min_len > 0:
            gen_kwargs["min_len"] = args.min_len

        wav = model.generate(**gen_kwargs)
        audio_chunk = wav.squeeze().cpu().numpy()
        sent_dur = len(audio_chunk) / model.sample_rate
        print(f"Generated {sent_dur:.2f}s of audio in {time.time() - t_sent:.2f}s")

        # If sentence 1 and no external ref audio was provided, save sentence 1 as voice lock reference
        if i == 0 and not speaker_ref_audio and len(sentences) > 1:
            os.makedirs(os.path.dirname(auto_ref_path), exist_ok=True)
            sf.write(auto_ref_path, audio_chunk, model.sample_rate)
            print(f"[Voice Lock] Saved Sentence 1 voice signature for story consistency.")

        all_audio_chunks.append(audio_chunk)
        # Add silence between sentences (except after the last one)
        if i < len(sentences) - 1 and args.pause > 0:
            all_audio_chunks.append(silence_padding)

    # Clean up temporary voice lock reference if created
    if os.path.exists(auto_ref_path):
        try:
            os.remove(auto_ref_path)
        except OSError:
            pass

    # 4. Watermark & Save
    final_audio = np.concatenate(all_audio_chunks, axis=0)
    total_duration = len(final_audio) / model.sample_rate
    total_time = time.time() - t_start

    if not args.no_watermark:
        final_audio = embed_watermark(final_audio, sr=model.sample_rate)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, final_audio, model.sample_rate)

    print("\n" + "=" * 65)
    print(f"[Success] Story Audio saved to: {args.output}")
    print(f"  Total Audio Duration: {total_duration:.2f} seconds")
    print(f"  Sample Rate:          {model.sample_rate} Hz (48kHz Studio Quality)")
    print(f"  Watermark:            {'Acoustic DSSS (Inaudible)' if not args.no_watermark else 'Disabled'}")
    print(f"  Total Synthesis Time: {total_time:.2f}s")
    print(f"  Real-time Factor:     {total_time / total_duration:.2f}x")
    print("=" * 65)

if __name__ == "__main__":
    main()

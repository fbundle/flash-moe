#!/usr/bin/env python3
"""Compare _moe_infer_rs logits against transformers running the ORIGINAL
(BF16, unquantized) stripped model. This measures actual quantization
quality (engine + BQ4 pipeline vs ground truth), not just engine-vs-dequant
round-trip.

Usage:
    python helpers/verify_vs_original.py
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

HF_DIR  = os.path.join(ROOT, "hub", "models--Qwen--Qwen3.6-35B-A3B-Strip")
BQ4_DIR = os.path.join(ROOT, "data", "Qwen3.6-35B-A3B-Strip", "model_bq4")

TOKENS = [
    248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 593,
    26003, 248046, 198, 248045, 846, 198, 9419, 11, 1204, 513,
    488, 30, 248046, 198, 248045, 74455, 198, 248068, 198,
]


def run_transformers():
    import torch
    from transformers import AutoModelForCausalLM
    print("[v] Loading ORIGINAL BF16 stripped HF model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(HF_DIR, dtype=torch.bfloat16)
    model.eval()
    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        out = model(ids)
    logits = out.logits[0, -1].float().numpy()
    elapsed = time.time() - t0
    print(f"[v] transformers (BF16 original) : {elapsed*1000:5.0f} ms  "
          f"min={logits.min():.4f} max={logits.max():.4f} "
          f"mean={logits.mean():.4f} NaNs={np.isnan(logits).sum()}")
    return logits


def run_rust(mode="Qwen35MoEFusedExp2"):
    import _moe_infer_rs as _rs

    # Patch arch for stripped dispatch
    cfg_path = os.path.join(BQ4_DIR, "config.json")
    with open(cfg_path) as f:
        orig = json.load(f)
    patched = dict(orig)
    patched["architectures"] = ["Qwen3_5MoeForConditionalGeneration_Stripped"]
    with open(cfg_path, "w") as f:
        json.dump(patched, f)
    try:
        t0 = time.time()
        model = _rs.Model(BQ4_DIR)
        engine = _rs.Engine(model, mode, 0)
        cache = _rs.Cache(model)
        ids_arr = np.array(TOKENS, dtype=np.int64)
        logits_all = engine.forward(ids_arr, cache)
        elapsed = time.time() - t0
        logits = np.array(logits_all[-1], dtype=np.float32)
        print(f"[v] Rust {mode:<22}: {elapsed*1000:5.0f} ms  "
              f"min={logits.min():.4f} max={logits.max():.4f} "
              f"mean={logits.mean():.4f} NaNs={np.isnan(logits).sum()}")
        return logits
    finally:
        with open(cfg_path, "w") as f:
            json.dump(orig, f)


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def compare(label1, logits1, label2, logits2, eps=1e-3):
    n = min(len(logits1), len(logits2))
    a = logits1[:n].astype(np.float64)
    b = logits2[:n].astype(np.float64)
    diff = np.abs(a - b)
    idx = diff.argmax()
    cos = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

    pa = _softmax(a); pb = _softmax(b)
    a_top1 = int(np.argmax(a)); b_top1 = int(np.argmax(b))
    def top_k(p, k): return np.argpartition(p, -k)[-k:]
    o10 = len(set(top_k(pa, 10).tolist()) & set(top_k(pb, 10).tolist()))
    o50 = len(set(top_k(pa, 50).tolist()) & set(top_k(pb, 50).tolist()))
    eps_p = 1e-30
    kl_ab = float(np.sum(pa * (np.log(pa + eps_p) - np.log(pb + eps_p))))
    kl_ba = float(np.sum(pb * (np.log(pb + eps_p) - np.log(pa + eps_p))))
    tv = float(0.5 * np.sum(np.abs(pa - pb)))

    print(f"\n  {label1} vs {label2}:")
    print(f"    max_diff = {diff.max():.6f} at idx {idx} "
          f"({label1}={a[idx]:.4f}, {label2}={b[idx]:.4f})")
    print(f"    mean_diff = {diff.mean():.6f}")
    print(f"    cosine_sim = {cos:.8f}")
    print(f"    top-1 match: {'YES' if a_top1 == b_top1 else 'NO '} "
          f"({label1}_top1={a_top1}, {label2}_top1={b_top1})")
    print(f"    top-10 overlap: {o10}/10    top-50 overlap: {o50}/50")
    print(f"    KL({label1}||{label2}) = {kl_ab:.6f}    "
          f"KL({label2}||{label1}) = {kl_ba:.6f}")
    print(f"    total variation: {tv:.6f}")


def main():
    print("=" * 60)
    print("Verification: ORIGINAL BF16 stripped HF model vs _moe_infer_rs")
    print(f"Tokens: {len(TOKENS)}")
    print("=" * 60)
    hf = run_transformers()
    for mode in ("Qwen35MoEFusedExp2", "Qwen35MoEFusedExp1"):
        try:
            rust = run_rust(mode)
            compare("transformers", hf, mode, rust)
        except Exception as e:
            print(f"\n[v] Rust {mode}: FAILED — {e}")


if __name__ == "__main__":
    main()

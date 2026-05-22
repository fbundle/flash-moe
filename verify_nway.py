#!/usr/bin/env python3
"""3-way logit verification: mlx-lm vs Fused3 vs FusedExp on stripped model."""
import subprocess, sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MLX_DIR = os.path.join(ROOT, "hub", "models--mlx-community--Qwen3.5-35B-A3B-4bit-stripped")

TOKENS = [248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 593,
          26003, 248046, 198, 248045, 846, 198, 9419, 11, 1204, 513,
          488, 30, 248046, 198, 248045, 74455, 198, 248068, 198]


def run_rust(mode):
    """Run Rust engine with given pipeline mode, return last-position logits."""
    from moe_infer import Context, Cache

    print(f"[nway] Running Rust {mode}...")
    t0 = __import__('time').time()

    ctx = Context()
    ctx.load_model(MLX_DIR, pipeline_mode=mode)
    cache = ctx.new_cache()

    ids_arr = np.array(TOKENS, dtype=np.int64)
    logits_all = ctx.forward(ids_arr, cache)

    elapsed = __import__('time').time() - t0
    logits = np.array(logits_all[-1], dtype=np.float32)

    print(f"[nway] Rust {mode}: {elapsed*1000:.0f} ms")
    print(f"  min={logits.min():.4f} max={logits.max():.4f} mean={logits.mean():.4f} NaNs={np.isnan(logits).sum()}")
    ctx.unload_model()
    return logits


def run_mlx():
    """Run MLX-LM on stripped model, return last-position logits."""
    print("[nway] Running MLX-LM...")
    t0 = __import__('time').time()

    from pathlib import Path
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm import tokenizer_utils

    model_path = Path(MLX_DIR)
    model, _ = load(str(model_path))
    tokenizer = tokenizer_utils.load(model_path)

    input_ids = mx.array(TOKENS, dtype=mx.int32)[None, :]
    outputs = model(input_ids)
    logits = np.array(mx.array(outputs[0, -1, :]).astype(mx.float32))

    elapsed = __import__('time').time() - t0
    print(f"[nway] MLX-LM: {elapsed*1000:.0f} ms")
    print(f"  min={logits.min():.4f} max={logits.max():.4f} mean={logits.mean():.4f} NaNs={np.isnan(logits).sum()}")
    return logits


def compare(label1, logits1, label2, logits2, eps=1e-3):
    """Compare two logit arrays."""
    min_len = min(len(logits1), len(logits2))
    a = logits1[:min_len].astype(np.float64)
    b = logits2[:min_len].astype(np.float64)
    diff = np.abs(a - b)
    max_diff = diff.max()
    mean_diff = diff.mean()
    idx_max = diff.argmax()

    # Relative diff: |a-b| / max(|a|, |b|, 1e-8)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)
    rel_diff = diff / denom
    max_rel = rel_diff.max()
    idx_rel = rel_diff.argmax()

    # Cosine similarity
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos_sim = float(np.dot(a, b) / max(a_norm * b_norm, 1e-12))

    matching = (diff < eps).sum()
    pct = 100.0 * matching / len(diff)

    print(f"\n  {label1} vs {label2}:")
    print(f"    max_diff={max_diff:.6f} at idx {idx_max} ({label1}={a[idx_max]:.6f}, {label2}={b[idx_max]:.6f})")
    print(f"    mean_diff={mean_diff:.6f}")
    print(f"    max_rel_diff={max_rel:.6f} at idx {idx_rel} ({label1}={a[idx_rel]:.6f}, {label2}={b[idx_rel]:.6f})")
    print(f"    cosine_sim={cos_sim:.8f}")
    print(f"    within {eps}: {matching}/{len(diff)} ({pct:.2f}%)")
    return max_diff


def main():
    print("=" * 60)
    print("3-Way Verification: mlx-lm vs Fused3 vs FusedExp")
    print(f"Model: stripped  (4 layers, 4 experts)")
    print(f"Tokens: {len(TOKENS)}")
    print("=" * 60)

    results = {}
    for mode in ["Cpu", "Gpu", "Fused3", "FusedExp"]:
        results[mode] = run_rust(mode)

    results["mlx-lm"] = run_mlx()

    print("\n" + "=" * 60)
    print("Pairwise comparisons")
    print("=" * 60)

    engines = ["Cpu", "Gpu", "Fused3", "FusedExp", "mlx-lm"]
    max_diffs = {}
    for i, e1 in enumerate(engines):
        for e2 in engines[i+1:]:
            key = f"{e1}_vs_{e2}"
            max_diffs[key] = compare(e1, results[e1], e2, results[e2])

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for pair, md in max_diffs.items():
        status = "PASS" if md < 1e-3 else ("CLOSE" if md < 1e-1 else "FAIL")
        print(f"  {pair}: max_diff={md:.6f} [{status}]")

    # Cpu and Gpu must match (pure CPU vs GPU non-fused — same algorithms)
    if max_diffs.get("Cpu_vs_Gpu", 1.0) < 1e-3:
        print("\n[verify] Cpu and Gpu match — baseline is solid.")
    else:
        print("\n[verify] WARNING: Cpu vs Gpu diverge — baseline is broken!")

    # Cpu must match mlx-lm
    if max_diffs.get("Cpu_vs_mlx-lm", 1.0) < 1e-2:
        print("[verify] Cpu matches mlx-lm — numerical correctness confirmed.")
    else:
        print("[verify] WARNING: Cpu vs mlx-lm diverge — shared code has bugs!")


if __name__ == "__main__":
    main()

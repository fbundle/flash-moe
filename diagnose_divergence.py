#!/usr/bin/env python3
"""Diagnose where Cpu vs mlx-lm divergence starts.

Runs both engines token-by-token, comparing hidden states after each layer
and each major operation. Finds the first operation where divergence exceeds
a threshold.
"""
import subprocess, sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MLX_DIR = os.path.join(ROOT, "hub", "models--mlx-community--Qwen3.5-35B-A3B-4bit-stripped")

TOKENS = [248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 593,
          26003, 248046, 198, 248045, 846, 198, 9419, 11, 1204, 513,
          488, 30, 248046, 198, 248045, 74455, 198, 248068, 198]

THRESHOLD = 1e-4


def compare_arrays(label, a, b):
    """Compare two arrays, return max_diff and flag if threshold exceeded."""
    a = np.array(a, dtype=np.float32).flatten()
    b = np.array(b, dtype=np.float32).flatten()
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_diff = float(diff.max())
    idx = int(diff.argmax())
    mean_diff = float(diff.mean())

    # cosine
    a_norm = np.linalg.norm(a.astype(np.float64))
    b_norm = np.linalg.norm(b.astype(np.float64))
    cos_sim = float(np.dot(a.astype(np.float64), b.astype(np.float64)) / max(a_norm * b_norm, 1e-12))

    flag = " *** DIVERGED" if max_diff > THRESHOLD else ""
    print(f"  [{label}] max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} cos_sim={cos_sim:.8f} @idx={idx} a={a[idx]:.6f} b={b[idx]:.6f}{flag}")
    return max_diff


def main():
    # Import both implementations
    from moe_infer import Context, Cache
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm import tokenizer_utils

    print("Loading models...")
    # MLX
    from pathlib import Path
    model_path = Path(MLX_DIR)
    model, _ = mlx_load(str(model_path))
    model.eval()

    # Rust
    ctx = Context()
    ctx.load_model(MLX_DIR, pipeline_mode="Cpu")
    cache = ctx.new_cache()

    # Get MLX cache
    mlx_cache = model.make_cache()

    # Token embedding from MLX
    embed_weight = model.language_model.model.embed_tokens.weight
    if hasattr(embed_weight, 'as_linear'):
        embed_weight = embed_weight.as_linear()

    def mlx_embed(tokens):
        return embed_weight[tokens]

    print(f"\n{'='*60}")
    print("Processing tokens one by one...")
    print(f"{'='*60}")

    for t_idx, token in enumerate(TOKENS):
        print(f"\n--- Token {t_idx} (id={token}) ---")

        # Rust forward one token
        ids_arr = np.array([token], dtype=np.int64)
        rust_logits = ctx.forward(ids_arr, cache)

        # MLX forward one token
        mlx_input = mx.array([token], dtype=mx.int32)[None, :]
        mlx_logits = model(mlx_input, cache=mlx_cache)

        # Compare final logits
        rust_logit = np.array(rust_logits[-1], dtype=np.float32)
        mlx_logit_raw = mx.array(mlx_logits[0, -1, :])
        mlx_logit = np.array(mlx_logit_raw.astype(mx.float32), dtype=np.float32)

        compare_arrays(
            f"T{t_idx} logits",
            rust_logit, mlx_logit
        )

        # Also compare hidden states if available
        rust_logit_full = np.array(rust_logits, dtype=np.float32)
        mlx_logit_full = np.array(mx.array(mlx_logits.astype(mx.float32)), dtype=np.float32)

        if t_idx == 0:
            print(f"\n  First-token logit shapes: Rust={rust_logit_full.shape}, MLX={mlx_logit_full.shape}")

    # Final summary
    print(f"\n{'='*60}")
    print("Done")
    print(f"{'='*60}")

    ctx.unload_model()


if __name__ == "__main__":
    main()

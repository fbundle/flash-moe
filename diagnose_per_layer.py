#!/usr/bin/env python3
"""Per-layer divergence diagnosis: captures hidden states from both Rust Cpu and MLX.

Compares per-layer outputs to find where divergence first exceeds threshold.
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MLX_DIR = os.path.join(ROOT, "hub", "models--mlx-community--Qwen3.5-35B-A3B-4bit-stripped")

TOKENS = [248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 593,
          26003, 248046, 198, 248045, 846, 198, 9419, 11, 1204, 513,
          488, 30, 248046, 198, 248045, 74455, 198, 248068, 198]

THRESHOLD = 1e-4


def compare_arrays(label, a, b):
    a = np.array(a, dtype=np.float32).flatten()
    b = np.array(b, dtype=np.float32).flatten()
    min_len = min(len(a), len(b))
    a64, b64 = a[:min_len].astype(np.float64), b[:min_len].astype(np.float64)
    diff = np.abs(a64 - b64)
    max_diff = float(diff.max())
    idx = int(diff.argmax())
    mean_diff = float(diff.mean())

    a_norm = np.linalg.norm(a64)
    b_norm = np.linalg.norm(b64)
    cos_sim = float(np.dot(a64, b64) / max(a_norm * b_norm, 1e-12))

    flag = " *** DIVERGED" if max_diff > THRESHOLD else ""
    print(f"  [{label}] max_diff={max_diff:.6e} mean_diff={mean_diff:.6e} "
          f"cos_sim={cos_sim:.8f} @idx={idx} a={a64[idx]:.6f} b={b64[idx]:.6f}{flag}")
    return max_diff


def main():
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm.models.qwen3_5 import DecoderLayer

    # ── Load MLX model ──
    from pathlib import Path
    model_path = Path(MLX_DIR)
    model, _ = mlx_load(str(model_path))
    model.eval()

    # Patch DecoderLayer.__call__ at the CLASS level (instance patching doesn't work for __special__ methods)
    original_call = DecoderLayer.__call__
    mlx_layer_outputs = []

    def patched_call(self, x, mask=None, cache=None):
        out = original_call(self, x, mask=mask, cache=cache)
        mlx_layer_outputs.append(np.array(out.astype(mx.float32)))
        return out

    DecoderLayer.__call__ = patched_call

    print("Running MLX batch forward...")
    t0 = time.time()
    mlx_cache = model.make_cache()
    mlx_input = mx.array(TOKENS, dtype=mx.int32)[None, :]
    mlx_logits_out = model(mlx_input, cache=mlx_cache)
    mlx_logits = np.array(mlx_logits_out[0, -1, :].astype(mx.float32))
    mlx_elapsed = time.time() - t0
    print(f"MLX done in {mlx_elapsed*1000:.0f}ms")

    # Restore
    DecoderLayer.__call__ = original_call

    print(f"\nMLX logits: min={mlx_logits.min():.4f} max={mlx_logits.max():.4f} "
          f"mean={mlx_logits.mean():.4f} NaNs={np.isnan(mlx_logits).sum()}")
    print(f"Captured {len(mlx_layer_outputs)} per-layer outputs")

    # Extract last-token per-layer outputs from MLX
    mlx_per_layer = []
    for lo in mlx_layer_outputs:
        h = lo[0, -1, :]  # last token
        mlx_per_layer.append(h)

    # ── Run Rust Cpu batch forward with debug ──
    from moe_infer import Context, Cache

    print("\nRunning Rust Cpu forward_debug...")
    t0 = time.time()
    ctx = Context()
    ctx.load_model(MLX_DIR, pipeline_mode="Cpu")
    cache = ctx.new_cache()

    ids_arr = np.array(TOKENS, dtype=np.int64)
    rust_logits_arr, rust_layers_list = ctx.forward_debug(ids_arr, cache)
    rust_logits = np.array(rust_logits_arr[-1], dtype=np.float32)
    rust_elapsed = time.time() - t0
    print(f"Rust done in {rust_elapsed*1000:.0f}ms")

    rust_per_layer = [np.array(arr, dtype=np.float32) for arr in rust_layers_list]

    print(f"\nRust logits: min={rust_logits.min():.4f} max={rust_logits.max():.4f} "
          f"mean={rust_logits.mean():.4f} NaNs={np.isnan(rust_logits).sum()}")

    ctx.unload_model()

    # ── Compare per-layer outputs ──
    print(f"\n{'='*60}")
    print("Per-layer hidden state comparison")
    print(f"{'='*60}")
    num_layers = min(len(rust_per_layer), len(mlx_per_layer))
    for i in range(num_layers):
        # Print stats for both
        rh = rust_per_layer[i]
        mh = mlx_per_layer[i]
        print(f"\n  Layer {i}:")
        print(f"    Rust:  min={rh.min():.4f} max={rh.max():.4f} mean={rh.mean():.4f} norm={np.linalg.norm(rh):.4f}")
        print(f"    MLX:   min={mh.min():.4f} max={mh.max():.4f} mean={mh.mean():.4f} norm={np.linalg.norm(mh):.4f}")
        compare_arrays(f"Layer_{i}_hidden", rh, mh)

    # Compare final logits
    print(f"\n{'='*60}")
    print("Final logit comparison")
    print(f"{'='*60}")
    compare_arrays("Cpu_vs_MLX_logits", rust_logits, mlx_logits)

    # ── Also compare embedding ──
    print(f"\n{'='*60}")
    print("Embedding comparison")
    print(f"{'='*60}")
    # Get MLX embedding
    embed_weight = model.language_model.model.embed_tokens.weight
    if hasattr(embed_weight, 'as_linear'):
        embed_weight = embed_weight.as_linear()
    mlx_embed_all = np.array(embed_weight[mx.array(TOKENS, dtype=mx.int32)].astype(mx.float32))
    mlx_first_embed = mlx_embed_all[0, :]
    # Rust embedding for first token
    # We can't easily get Rust embedding separately, but we know layer 0 input should match
    # The difference between embedding and first MoE output tells us about attention contribution

    print(f"MLX embedding[0]: min={mlx_first_embed.min():.4f} max={mlx_first_embed.max():.4f} "
          f"mean={mlx_first_embed.mean():.4f} norm={np.linalg.norm(mlx_first_embed):.4f}")


if __name__ == "__main__":
    main()

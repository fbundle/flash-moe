#!/usr/bin/env python3
"""Per-operation comparison: MLX vs Rust Cpu for Layer 0, single token (id=248045).

Captures every intermediate tensor from both engines and compares them
using max_diff, mean_diff, and relative_diff = |a-b| / (|a| + eps).
"""
import numpy as np

ROOT = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
MLX_DIR = __import__('os').path.join(ROOT, "hub", "models--mlx-community--Qwen3.5-35B-A3B-4bit-stripped")

TOKEN_ID = 248045
EPS = 1e-8


def compare_op(name, rust_arr, mlx_arr):
    """Compare two float32 arrays, return dict of metrics."""
    a = np.array(rust_arr, dtype=np.float32).flatten()
    b = np.array(mlx_arr, dtype=np.float32).flatten()
    min_len = min(len(a), len(b))
    a64 = a[:min_len].astype(np.float64)
    b64 = b[:min_len].astype(np.float64)

    abs_diff = np.abs(a64 - b64)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    max_idx = int(abs_diff.argmax())

    # Relative diff: |a-b| / (|a| + eps)
    rel_diff = abs_diff / (np.abs(a64) + EPS)
    max_rel = float(rel_diff.max())
    mean_rel = float(rel_diff.mean())
    max_rel_idx = int(rel_diff.argmax())

    # Cosine similarity
    a_norm = np.linalg.norm(a64)
    b_norm = np.linalg.norm(b64)
    cos_sim = float(np.dot(a64, b64) / max(a_norm * b_norm, 1e-12))

    # Statistics
    a_min, a_max, a_mean = float(a64.min()), float(a64.max()), float(a64.mean())
    b_min, b_max, b_mean = float(b64.min()), float(b64.max()), float(b64.mean())

    status = "OK" if max_abs < 1e-4 else ("~" if max_abs < 1e-2 else "FAIL")

    print(f"  [{name:20s}] {status:4s} | max_abs={max_abs:.2e} mean_abs={mean_abs:.2e} "
          f"max_rel={max_rel:.2e} mean_rel={mean_rel:.2e} cos={cos_sim:.8f}")
    if max_abs > 1e-5:
        print(f"    max_abs @{max_idx}: rust={a64[max_idx]:.8f} mlx={b64[max_idx]:.8f}")
        print(f"    max_rel @{max_rel_idx}: rust={a64[max_rel_idx]:.8f} mlx={b64[max_rel_idx]:.8f} rel={max_rel:.2e}")
    print(f"    ranges: rust=[{a_min:.4f}, {a_max:.4f}] mlx=[{b_min:.4f}, {b_max:.4f}]")
    print(f"    means:  rust={a_mean:.4f} mlx={b_mean:.4f}")

    return {
        "name": name, "status": status,
        "max_abs": max_abs, "mean_abs": mean_abs,
        "max_rel": max_rel, "mean_rel": mean_rel,
        "cos_sim": cos_sim,
    }


def get_mlx_intermediates():
    """Patch MLX GatedDeltaNet.__call__ to capture all intermediate tensors for token 0."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load
    from pathlib import Path
    from mlx_lm.models.qwen3_5 import GatedDeltaNet

    model_path = Path(MLX_DIR)
    model, _ = mlx_load(str(model_path))
    model.eval()

    original_call = GatedDeltaNet.__call__
    captured = {}
    call_count = [0]  # mutable counter

    def patched_call(self, inputs, mask=None, cache=None):
        call_count[0] += 1
        # Only capture the FIRST call (layer 0), not layers 1, 2
        if call_count[0] != 1:
            return original_call(self, inputs, mask=mask, cache=cache)

        B, S, _ = inputs.shape
        captured["inputs"] = np.array(inputs.astype(mx.float32))[0, -1, :]

        qkv = self.in_proj_qkv(inputs)
        captured["qkv"] = np.array(qkv.astype(mx.float32))[0, -1, :]

        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        captured["z"] = np.array(z.astype(mx.float32))[0, -1, :, :]

        b = self.in_proj_b(inputs)
        captured["beta"] = np.array(b.astype(mx.float32))[0, -1, :]

        a = self.in_proj_a(inputs)
        captured["alpha"] = np.array(a.astype(mx.float32))[0, -1, :]

        # Conv state (zero for first token)
        conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        captured["conv_input"] = np.array(conv_input.astype(mx.float32))[0, :, :]

        conv_out = nn.silu(self.conv1d(conv_input))
        captured["conv_out"] = np.array(conv_out.astype(mx.float32))[0, -1, :]

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        captured["lin_q"] = np.array(q.astype(mx.float32))[0, -1, :, :]
        captured["lin_k"] = np.array(k.astype(mx.float32))[0, -1, :, :]
        captured["lin_v"] = np.array(v.astype(mx.float32))[0, -1, :, :]

        state = None  # first token
        inv_scale = k.shape[-1] ** -0.5
        q_normed = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k_normed = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        captured["q_normed"] = np.array(q_normed.astype(mx.float32))[0, -1, :, :]
        captured["k_normed"] = np.array(k_normed.astype(mx.float32))[0, -1, :, :]

        # Run SSM with state=None (zero init)
        from mlx_lm.models.gated_delta import gated_delta_update
        out, new_state = gated_delta_update(
            q_normed, k_normed, v, a, b, self.A_log, self.dt_bias, state, mask,
            use_kernel=not self.training,
        )
        captured["out_values"] = np.array(out.astype(mx.float32))[0, -1, :, :]

        out_normed = self.norm(out, z)
        captured["gated_out"] = np.array(out_normed.astype(mx.float32))[0, -1, :, :]

        out_proj = self.out_proj(out_normed.reshape(B, S, -1))
        captured["attn_out"] = np.array(out_proj.astype(mx.float32))[0, -1, :]

        return out_proj

    GatedDeltaNet.__call__ = patched_call

    # Run single token
    input_ids = mx.array([TOKEN_ID], dtype=mx.int32)[None, :]
    model(input_ids)

    GatedDeltaNet.__call__ = original_call
    return captured


def get_rust_intermediates():
    """Get Rust debug_layer0 intermediates."""
    from moe_infer import Context

    # Get the actual embedding from MLX by running a forward pass and capturing it
    # The embedding is the input to layer 0 (after embed_tokens lookup).
    # We'll use MLX to get this, being careful about the 4-bit packed format.
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from pathlib import Path

    model_path = Path(MLX_DIR)
    model, _ = mlx_load(str(model_path))

    # Run a single token through just the embedding
    input_ids = mx.array([TOKEN_ID], dtype=mx.int32)[None, :]
    hidden_states = model.language_model.model.embed_tokens(input_ids)
    token_embed = np.array(hidden_states.astype(mx.float32))[0, 0, :]
    print(f"  MLX embedding: shape={token_embed.shape} min={token_embed.min():.4f} max={token_embed.max():.4f}")

    ctx = Context()
    ctx.load_model(MLX_DIR, pipeline_mode="Cpu")
    result = ctx.debug_layer0(token_embed)
    ctx.unload_model()
    return result


def main():
    print("=" * 70)
    print(f"Per-Operation Comparison: Layer 0, Token {TOKEN_ID}")
    print("=" * 70)

    print("\nCapturing MLX intermediates...")
    mlx_ops = get_mlx_intermediates()

    print("Capturing Rust intermediates...")
    rust_ops = get_rust_intermediates()

    # Map MLX names to Rust names
    mapping = [
        ("inputs", "normed", "Input RMS Norm"),        # MLX input = Rust normed
        ("qkv", "qkv", "QKV projection"),
        ("z", "z", "Z projection"),
        ("beta", "beta", "Beta projection"),
        ("alpha", "alpha", "Alpha projection"),
        ("conv_out", "conv_out", "Conv1d + SiLU"),
        ("lin_q", "lin_q", "Q (from conv_out)"),
        ("lin_k", "lin_k", "K (from conv_out)"),
        ("lin_v", "lin_v", "V (from conv_out)"),
        ("q_normed", "q_normed", "Q RMS norm"),
        ("k_normed", "k_normed", "K RMS norm"),
        ("out_values", "out_values", "SSM state update"),
        ("gated_out", "gated_out", "Gated RMS norm"),
        ("attn_out", "attn_out", "Output projection"),
        ("hidden_out", "hidden_out", "Residual add (layer output)"),
    ]

    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")

    results = []
    for mlx_key, rust_key, desc in mapping:
        print(f"\n--- {desc} ---")
        if rust_key not in rust_ops:
            print(f"  [SKIP] Rust key '{rust_key}' not found")
            continue
        if mlx_key not in mlx_ops:
            print(f"  [SKIP] MLX key '{mlx_key}' not found")
            continue

        mlx_arr = mlx_ops[mlx_key]
        rust_arr = rust_ops[rust_key]
        r = compare_op(f"{mlx_key}/{rust_key}", rust_arr, mlx_arr)
        results.append(r)

    # Summary table
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Operation':<30s} {'Status':>5s} {'max_abs':>10s} {'max_rel':>10s} {'cos_sim':>12s}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30s} {r['status']:>5s} {r['max_abs']:>10.2e} {r['max_rel']:>10.2e} {r['cos_sim']:>12.8f}")

    # Find first failure
    first_fail = None
    for r in results:
        if r['status'] != 'OK':
            first_fail = r['name']
            break
    if first_fail:
        print(f"\nFirst divergence at: {first_fail}")
    else:
        print(f"\nAll operations match within 1e-4")


if __name__ == "__main__":
    main()

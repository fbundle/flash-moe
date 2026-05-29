#!/usr/bin/env python3
"""analyze_rotated_quant.py — Test whether random rotation + lower-bit
quantization (rotated INT3 / INT2) preserves expert matrix output quality
better than the current straight INT4.

For each expert weight matrix W [out, in]:
  baseline: per-row group min-max scalar quant at {2,3,4} bits, group=64
  rotated:  W' = W @ R^T  with R orthogonal, then quantize W' the same way
            (inference: y = W'q @ (R @ x) approximates W @ x)

Metric: per-sample cosine similarity of W·x vs Wq·x (and W'q·R·x rotated)
on isotropic Gaussian inputs. Same caveat as analyze_expert_lowrank.py:
Gaussian inputs are a proxy for real activations — treat numbers as a
rough upper bound on cos.

Usage:
    python helpers/analyze_rotated_quant.py \
        --model data/Qwen3.6-35B-A3B/model_bq4 --layer 10 --num-experts 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Reuse low-rank script's dequant helpers
import sys
sys.path.insert(0, str(Path(__file__).parent))
from analyze_expert_lowrank import (
    expert_layout, read_section, GROUP_SIZE,
)

# ─── BF16 source loader (from HF safetensors, no re-quant) ─────────────────

import struct

def _parse_safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        return header, 8 + hlen


def _bf16_to_f32(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32).reshape(shape)


def load_hf_expert_layer(hf_dir: str, layer: int, num_experts: int
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Load BF16 experts for a single MoE layer from HF safetensors.

    Returns (gate_up [E, 2*mi, hd] float32, down [E, hd, mi] float32).
    """
    idx_path = Path(hf_dir) / "model.safetensors.index.json"
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]

    gu_name = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
    dn_name = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
    if gu_name not in wm or dn_name not in wm:
        raise SystemExit(f"layer {layer} expert tensors not in safetensors index")

    def _read(name: str) -> np.ndarray:
        fname = wm[name]
        path = str(Path(hf_dir) / fname)
        header, data_start = _parse_safetensors_header(path)
        meta = header[name]
        off0, off1 = meta["data_offsets"]
        shape = tuple(meta["shape"])
        with open(path, "rb") as f:
            f.seek(data_start + off0)
            raw = f.read(off1 - off0)
        return _bf16_to_f32(raw, shape)

    gate_up = _read(gu_name)
    down    = _read(dn_name)
    return gate_up[:num_experts], down[:num_experts]


# ─── Group min-max scalar quantization (matches BQ4 scheme) ────────────────

def quant_dequant_group_minmax(W: np.ndarray, bits: int, group: int = GROUP_SIZE
                                ) -> tuple[np.ndarray, int]:
    """Per-row group-wise min-max quant to `bits`, then dequantize.

    Returns (W_recon, bytes_per_row_for_quantized_storage).
    Storage model matches BQ4: packed weights + BF16 scales + BF16 biases
    per group. Bytes = (in/8)*bits + 2 groups*2 bytes (scale+bias as BF16).
    Note: BQ4 currently uses bits=4. We compute the same per-group scheme
    for any `bits` and let the caller compare.
    """
    out_dim, in_dim = W.shape
    assert in_dim % group == 0
    num_groups = in_dim // group
    levels = (1 << bits) - 1  # 2^bits - 1 max code

    Wg = W.reshape(out_dim, num_groups, group)
    vmin = Wg.min(axis=2, keepdims=True)
    vmax = Wg.max(axis=2, keepdims=True)
    scale = (vmax - vmin) / max(levels, 1)
    scale_safe = np.where(scale == 0, 1.0, scale)

    codes = np.round((Wg - vmin) / scale_safe).clip(0, levels)
    W_recon = (codes * scale_safe + vmin).reshape(out_dim, in_dim)

    # Storage bytes: codes (bits/8) per element + 2 BF16 per group (scale, bias).
    # For sub-byte widths we just use bits/8 directly (info-theoretic).
    code_bytes = out_dim * in_dim * bits / 8.0
    sb_bytes   = out_dim * num_groups * 2 * 2  # scale + bias, BF16
    return W_recon, int(code_bytes + sb_bytes)


# ─── Random orthogonal rotation (Gaussian QR — simple, not fast Hadamard) ──

def make_rotation(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, n)).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    # Random sign flip to remove QR sign bias
    signs = (rng.integers(0, 2, size=n) * 2 - 1).astype(np.float32)
    return Q * signs[None, :]


# ─── Output cosine on probe activations ────────────────────────────────────

def output_cosine(W: np.ndarray, W_recon: np.ndarray, X: np.ndarray) -> np.ndarray:
    Y = W @ X
    Y_q = W_recon @ X
    num = (Y * Y_q).sum(axis=0)
    den = np.linalg.norm(Y, axis=0) * np.linalg.norm(Y_q, axis=0) + 1e-12
    return num / den


# ─── Analysis loop ─────────────────────────────────────────────────────────

def analyze_matrix(W: np.ndarray, bits_list: list[int], X: np.ndarray,
                   rotation_seed: int = 0) -> dict:
    """For one weight matrix, sweep bit widths with and without rotation."""
    out_dim, in_dim = W.shape
    R = make_rotation(in_dim, rotation_seed)
    # Rotated input axis: W' = W @ R^T  (so that W'·(R x) == W·x exactly)
    W_rot = W @ R.T
    # The rotated input the quant matrix will see:
    X_rot = R @ X

    res = {"shape": (out_dim, in_dim)}
    for bits in bits_list:
        W_recon, bytes_b = quant_dequant_group_minmax(W, bits)
        cos_b = output_cosine(W, W_recon, X)

        W_rot_recon, bytes_r = quant_dequant_group_minmax(W_rot, bits)
        # Inference path: rotated W' @ (R x) approximates W @ x
        # So we compare W·X (truth) vs W_rot_recon·X_rot (approx)
        Y_true = W @ X
        Y_rot  = W_rot_recon @ X_rot
        num = (Y_true * Y_rot).sum(axis=0)
        den = (np.linalg.norm(Y_true, axis=0) *
               np.linalg.norm(Y_rot, axis=0) + 1e-12)
        cos_r = num / den

        res[bits] = {
            "no_rot_cos_mean": float(cos_b.mean()),
            "no_rot_cos_p1":   float(np.percentile(cos_b, 1)),
            "no_rot_bytes":    bytes_b,
            "rot_cos_mean":    float(cos_r.mean()),
            "rot_cos_p1":      float(np.percentile(cos_r, 1)),
            "rot_bytes":       bytes_r,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="BQ4 model dir (used for layout/config). For --hf-bf16 mode "
                         "this is still consulted for hidden_size and moe_intermediate_size.")
    ap.add_argument("--hf-bf16", type=str, default=None,
                    help="HF safetensors dir for BF16 source weights (skips BQ4 re-quant)")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--bits", type=str, default="2,3,4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bits_list = [int(b) for b in args.bits.split(",")]

    model_path = Path(args.model)
    with open(model_path / "model_weights.json") as f:
        cfg = json.load(f)["config"]
    hd, mi = cfg["hidden_size"], cfg["moe_intermediate_size"]
    num_experts_total = cfg["num_experts"]
    num_experts = min(args.num_experts, num_experts_total)

    print(f"Model: {model_path}    Layer: {args.layer}")
    print(f"Hidden: {hd}    MoE inter: {mi}    Experts sampled: {num_experts}")
    print(f"Bits sweep: {bits_list}    Samples per cos probe: {args.num_samples}")
    print()

    rng = np.random.default_rng(args.seed)
    X_hd = rng.standard_normal((hd, args.num_samples)).astype(np.float32)
    X_mi = rng.standard_normal((mi, args.num_samples)).astype(np.float32)

    matrix_results: dict[str, list[dict]] = {"gate": [], "up": [], "down": []}
    t0 = time.time()

    if args.hf_bf16:
        print(f"Source: BF16 from {args.hf_bf16} (no re-quant)")
        gate_up, down = load_hf_expert_layer(args.hf_bf16, args.layer, num_experts)
        # gate_up: [E, 2*mi, hd]   down: [E, hd, mi]
        for e in range(num_experts):
            gate_W = gate_up[e, :mi, :]
            up_W   = gate_up[e, mi:, :]
            down_W = down[e]
            matrix_results["gate"].append(
                analyze_matrix(gate_W, bits_list, X_hd, rotation_seed=args.seed + e * 3 + 0))
            matrix_results["up"].append(
                analyze_matrix(up_W,   bits_list, X_hd, rotation_seed=args.seed + e * 3 + 1))
            matrix_results["down"].append(
                analyze_matrix(down_W, bits_list, X_mi, rotation_seed=args.seed + e * 3 + 2))
            print(f"  expert {e}: done [{(e+1)/(time.time()-t0):.2f} expert/s]")
    else:
        print(f"Source: BQ4 re-quantized from {model_path}")
        layout = expert_layout(hd, mi)
        layer_path = model_path / "packed_experts" / f"layer_{args.layer:02d}.bin"
        with open(layer_path, "rb") as f:
            buf = memoryview(f.read())

        for e in range(num_experts):
            base = e * layout["size"]
            gate_W = read_section(buf, base + layout["gate_w"], mi, hd)
            up_W   = read_section(buf, base + layout["up_w"],   mi, hd)
            down_W = read_section(buf, base + layout["down_w"], hd, mi)
            matrix_results["gate"].append(
                analyze_matrix(gate_W, bits_list, X_hd, rotation_seed=args.seed + e * 3 + 0))
            matrix_results["up"].append(
                analyze_matrix(up_W,   bits_list, X_hd, rotation_seed=args.seed + e * 3 + 1))
            matrix_results["down"].append(
                analyze_matrix(down_W, bits_list, X_mi, rotation_seed=args.seed + e * 3 + 2))
            print(f"  expert {e}: done [{(e+1)/(time.time()-t0):.2f} expert/s]")

    print()
    print("=" * 92)
    print(f"SUMMARY  layer={args.layer}, N={num_experts} experts, "
          f"Gaussian probes (proxy for real activations)")
    print("=" * 92)

    for mat in ("gate", "up", "down"):
        results = matrix_results[mat]
        shape = results[0]["shape"]
        print(f"\n  {mat}_proj  shape={shape[0]}x{shape[1]}")
        print(f"    {'bits':>4s}  {'no-rot cos':>12s}  {'no-rot p1':>11s}  "
              f"{'rot cos':>9s}  {'rot p1':>9s}  "
              f"{'no-rot KB':>10s}  {'rot KB':>8s}  {'ratio':>6s}")
        for bits in bits_list:
            cos_b = np.array([r[bits]["no_rot_cos_mean"] for r in results])
            p1_b  = np.array([r[bits]["no_rot_cos_p1"]   for r in results])
            cos_r = np.array([r[bits]["rot_cos_mean"]    for r in results])
            p1_r  = np.array([r[bits]["rot_cos_p1"]      for r in results])
            kb_b  = results[0][bits]["no_rot_bytes"] / 1024.0
            kb_r  = results[0][bits]["rot_bytes"]    / 1024.0
            print(f"    {bits:>4d}  "
                  f"{cos_b.mean():>12.4f}  {p1_b.mean():>11.4f}  "
                  f"{cos_r.mean():>9.4f}  {p1_r.mean():>9.4f}  "
                  f"{kb_b:>10.1f}  {kb_r:>8.1f}  {kb_r/kb_b:>6.3f}")

    print()
    print("Interpretation:")
    print("  - 'no-rot cos' at bits=4 reflects our CURRENT INT4 quality")
    print("  - 'rot cos' at bits=3 is the rotated-INT3 candidate")
    print("  - Storage shown is per matrix (codes + per-group BF16 scale/bias)")
    print("  - Bandwidth saving = 1 - bytes_at_target / bytes_at_current_INT4")
    print()
    print("Note: Gaussian probes are isotropic; real activations are not.")
    print("      Rotation often looks better here than on real activations")
    print("      because rotation flattens the *input* distribution toward")
    print("      Gaussian — which the probe already is. For more rigour, re-")
    print("      probe with hidden states captured from a transformers forward.")


if __name__ == "__main__":
    main()

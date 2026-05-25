#!/usr/bin/env python3
"""
quantize.py — Quantize HF BF16 Qwen3.5/3.6 MoE model using BQ4.

Applies the BQ4 block→format classification from quant/README.md:
  - FP16: attention projections, routers, lm_head, patch_embed.proj
  - wint4:  all other matrices (experts, linear attention, embeddings, etc.)
  - BF16: all metadata/auxiliary tensors (norms, scales, biases, etc.)
  - FP32: A_log scalars

Uses quant/name_mapping.json for HF→MLX tensor name mapping.

Output:
  - model_weights.bin      : all non-expert weights
  - model_weights.json     : manifest (name → offset, size, shape, dtype)
  - packed_experts/layer_XX.bin : per-layer expert weights

Usage:
    python quant/quantize.py \\
        --model hub/models--Qwen--Qwen3.6-35B-A3B \\
        --output data/my-model
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ─── Types ───────────────────────────────────────────────────────────────

class Quant(Enum):
    FP32 = "f32"
    BF16 = "bf16"
    FP16 = "f16"
    INT4 = "u32"

    def __str__(self) -> str:
        return self.value

# ─── Constants ───────────────────────────────────────────────────────────

GROUP_SIZE = 64
ALIGN = 64


# ─── BF16 / FP16 / FP32 conversion ───────────────────────────────────────

def bf16_bytes_to_f32(data: bytes, shape: list[int]) -> np.ndarray:
    u16 = np.frombuffer(data, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape)


def f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    u32 = arr.view(np.uint32)
    round_bit = (u32 >> 15) & 1
    sticky = u32 & 0x7FFF
    round_up = round_bit & (sticky | ((u32 >> 16) & 1))
    u32 = u32 + (round_up.astype(np.uint32) << 16)
    return (u32 >> 16).astype(np.uint16)


def f32_to_f16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 to float16 (uint16)."""
    return arr.astype(np.float16).view(np.uint16)


# ─── Wilkinson INT4 quantization ───────────────────────────────────────────
#
# weight = m × 2^E + B   where m ∈ {0..15}, B = bf16 bias, 2^E = fp16 scale.
#
# Storage per group (64 weights):
#   32 bytes  — packed 4-bit mantissas (8 per uint32, LSB-first)
#   2 bytes   — fp16 scale (2^E)
#   2 bytes   — bf16 bias (B)
#   ─────────
#   36 bytes  = 4.5 bits per weight
#
# Same wire format as standard INT4.  The kernel (nibble × scale + bias) is
# unchanged; the quantization algorithm constrains scale = 2^E to give constant
# relative error across the group.

def quant_f32_to_int4(values: np.ndarray, out_dim: int, in_dim: int
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize float32 [out_dim, in_dim] to Wilkinson INT4.

    Returns (packed_uint32, scales_f16, biases_bf16).
    scales stores 2^E as fp16, biases stores B as bf16.
    """
    num_groups = in_dim // GROUP_SIZE
    packed = np.zeros(out_dim * in_dim // 8, dtype=np.uint32)
    scales = np.zeros(out_dim * num_groups, dtype=np.uint16)
    biases = np.zeros(out_dim * num_groups, dtype=np.uint16)

    CHUNK_ROWS = max(1, min(256, 64 * 1024 * 1024 // in_dim))
    PACK_SHIFTS = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    B_OFFSETS = np.array([0.0, -0.25, 0.25, -0.5, 0.5], dtype=np.float32)

    for row_start in range(0, out_dim, CHUNK_ROWS):
        row_end = min(row_start + CHUNK_ROWS, out_dim)
        chunk_rows = row_end - row_start

        v = values[row_start:row_end].reshape(chunk_rows, num_groups, GROUP_SIZE)
        # v: [chunk_rows, num_groups, 64]

        wmin = v.min(axis=2).astype(np.float32)   # [chunk_rows, num_groups]
        wmax = v.max(axis=2).astype(np.float32)
        span = wmax - wmin                         # [chunk_rows, num_groups]

        # Find E range covering all groups in this chunk
        ideal_E = np.where(span > 1e-12,
                           np.log2(span / 15.0).astype(np.float32),
                           np.float32(-16.0))
        E_lo = max(-24, int(ideal_E.min()) - 1)
        E_hi = min(15, int(ideal_E.max()) + 1)   # 2^15 = 32768 fits in fp16

        best_max_err = np.full((chunk_rows, num_groups), np.float32(np.inf))
        best_E_arr = np.zeros((chunk_rows, num_groups), dtype=np.int32)
        best_B_arr = np.zeros((chunk_rows, num_groups), dtype=np.float32)
        best_m = np.zeros((chunk_rows, num_groups, GROUP_SIZE), dtype=np.uint8)

        # Search over E and B offsets, vectorized across all groups
        for E in range(E_lo, E_hi + 1):
            step = 2.0 ** E
            for dB in B_OFFSETS:
                B = (wmin + dB * step).astype(np.float32)

                m_cand = np.round((v - B[:, :, np.newaxis]) / step)
                m_clip = np.clip(m_cand, 0, 15).astype(np.uint8)
                recon = m_clip.astype(np.float32) * step + B[:, :, np.newaxis]
                max_err = np.abs(v - recon).max(axis=2)  # [chunk_rows, num_groups]

                better = max_err < best_max_err
                best_max_err[better] = max_err[better]
                best_E_arr[better] = E
                best_B_arr[better] = B[better]
                best_m[better] = m_clip[better]

        # Write scales and biases for this chunk
        s_chunk_slice = slice(row_start * num_groups, row_end * num_groups)
        steps = np.power(np.float32(2.0), best_E_arr.astype(np.float32)).ravel()
        scales[s_chunk_slice] = f32_to_f16(steps).ravel()
        biases[s_chunk_slice] = f32_to_bf16_u16(best_B_arr.ravel())

        # Pack mantissas: [chunk_rows, num_groups, 64] → [chunk_rows, num_groups, 8] uint32
        m_rs = best_m.reshape(chunk_rows, num_groups, GROUP_SIZE // 8, 8)
        words = ((m_rs.astype(np.uint32) & 0xF) << PACK_SHIFTS).sum(axis=3)
        packed_words = words.reshape(chunk_rows, -1)  # [chunk_rows, words_per_row]

        words_per_row = in_dim // 8
        for r in range(chunk_rows):
            row_idx = row_start + r
            w_start = row_idx * words_per_row
            packed[w_start:w_start + words_per_row] = packed_words[r]

    return packed, scales, biases


def int4_to_f32(packed: np.ndarray, scales: np.ndarray, biases: np.ndarray,
                out_dim: int, in_dim: int) -> np.ndarray:
    """Dequantize Wilkinson INT4 packed weights back to float32."""
    num_groups = in_dim // GROUP_SIZE
    result = np.zeros(out_dim * in_dim, dtype=np.float32)

    CHUNK_ROWS = max(1, min(4096, 64 * 1024 * 1024 // in_dim))

    for row_start in range(0, out_dim, CHUNK_ROWS):
        row_end = min(row_start + CHUNK_ROWS, out_dim)
        chunk_rows = row_end - row_start

        words_per_row = in_dim // 8
        w_start = row_start * words_per_row
        w_end = row_end * words_per_row
        w = packed[w_start:w_end].view(np.uint32) \
            .reshape(chunk_rows, num_groups, GROUP_SIZE // 8)

        s_start = row_start * num_groups
        s_end = row_end * num_groups
        s = scales[s_start:s_end].view(np.uint16) \
            .reshape(chunk_rows, num_groups)

        b_start = row_start * num_groups
        b_end = row_end * num_groups
        b = biases[b_start:b_end].view(np.uint16) \
            .reshape(chunk_rows, num_groups)

        # Unpack nibbles LSB-first
        nibbles = np.zeros((chunk_rows, num_groups, GROUP_SIZE), dtype=np.uint8)
        for j in range(8):
            nibbles[:, :, j::8] = ((w >> (j * 4)) & 0xF).astype(np.uint8)

        # Dequant: m × 2^E + B
        # fp16 scale → f32
        step_f32 = s.view(np.float16).astype(np.float32)[:, :, np.newaxis]
        # bf16 bias → f32
        bias_f32 = (b.astype(np.uint32) << 16).view(np.float32)[:, :, np.newaxis]

        vals = nibbles.astype(np.float32) * step_f32 + bias_f32

        r_start = row_start * in_dim
        result[r_start:r_start + vals.size] = vals.ravel()

    return result.reshape(out_dim, in_dim)


def verify(source: np.ndarray, recon: np.ndarray,
           orig_in_dim: int | None = None) -> dict:
    """Compare source F32 against dequantized F32. Returns metrics."""
    if orig_in_dim is not None and orig_in_dim < source.shape[1]:
        source = source[:, :orig_in_dim]
        recon = recon[:, :orig_in_dim]

    diff = np.abs(source - recon)
    return {
        "max_err": float(diff.max()),
        "mean_err": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean((source - recon) ** 2))),
        "snr_db": float(20 * np.log10(np.linalg.norm(source) / max(np.linalg.norm(source - recon), 1e-30))),
    }


# ─── Safetensors I/O ─────────────────────────────────────────────────────

def parse_safetensors_header(filepath: Path) -> tuple[dict, int]:
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def read_tensor_raw(filepath: Path, header: dict, name: str,
                    data_start: int) -> tuple[bytes, list[int], str]:
    meta = header[name]
    off = meta['data_offsets']
    length = off[1] - off[0]
    with open(filepath, 'rb') as f:
        f.seek(data_start + off[0])
        data = f.read(length)
    return data, meta['shape'], meta['dtype']


# ─── Name mapping ────────────────────────────────────────────────────────

def load_name_mapping(mapping_path: Path, num_layers: int,
                      num_vision_blocks: int) -> dict[str, str]:
    """Load name_mapping.json and expand {L}/{B} patterns into a flat dict."""
    with open(mapping_path) as f:
        mapping = json.load(f)

    flat = {}
    for hf_pat, mlx_pat in mapping.items():
        if '{L}' in hf_pat:
            for l in range(num_layers):
                flat[hf_pat.format(L=l)] = mlx_pat.format(L=l)
        elif '{B}' in hf_pat:
            for b in range(num_vision_blocks):
                flat[hf_pat.format(B=b)] = mlx_pat.format(B=b)
        else:
            flat[hf_pat] = mlx_pat

    return flat


# ─── BQ4 classification ──────────────────────────────────────────────────

# Blocks kept in FP16 (from quant/README.md matrixTable)
FP16_BLOCKS = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate",
    "lm_head",
    "attn.qkv",
    "attn.proj",
    "patch_embed.proj",
    "pos_embed",
}

# Prefixes stripped to get the relative block (from quant/README.md)
# Order matters: layered prefixes matched before bare namespace prefixes.
_PREFIX_RE = re.compile(
    r'^(language_model\.model\.layers\.\d+\.'
    r'|language_model\.'
    r'|vision_tower\.blocks\.\d+\.'
    r'|vision_tower\.'
    r'|mtp\.layers\.\d+\.'
    r'|mtp\.)'
)


def _strip_layer_prefix(full_block: str) -> str:
    return _PREFIX_RE.sub('', full_block)


def split_on_last_dot(name: str) -> tuple[str, str]:
    """Split on last dot → (prefix, kind)."""
    idx = name.rfind('.')
    if idx == -1:
        return name, ''
    return name[:idx], name[idx + 1:]


def bq4(mlx_name: str, shape: list[int]) -> Quant:
    """Classify a tensor by its MLX name and shape → Quant enum."""
    prefix, kind = split_on_last_dot(mlx_name)
    ndim = len(shape)

    if kind == 'A_log':
        assert ndim <= 1, f"A_log must be scalar/vector, got ndim={ndim}: {mlx_name}"
        return Quant.FP32

    if kind in ('scales', 'biases', 'bias', 'dt_bias'):
        assert ndim <= 2, f"{kind} must be vector, got ndim={ndim}: {mlx_name}"
        return Quant.BF16

    if kind == 'weight':
        if ndim != 2:               # matrix = exactly 2D
            return Quant.BF16
        block = _strip_layer_prefix(prefix)
        if block in FP16_BLOCKS:
            return Quant.FP16
        return Quant.INT4

    assert False, f"unknown kind: {kind!r} in {mlx_name}"


# ─── Tensor classification ───────────────────────────────────────────────

def is_expert_tensor(mlx_name: str) -> bool:
    return bool(re.search(r'\.switch_mlp\.(gate_proj|gate_up_proj|up_proj|down_proj)\.', mlx_name))


def is_vision_tensor(hf_name: str) -> bool:
    return bool(re.match(r'^(vision_tower|model\.visual)', hf_name))


# ─── Expert repacking ────────────────────────────────────────────────────

def repack_experts_layer(model_path: Path, header_cache: dict,
                         weight_map: dict, layer_idx: int,
                         hidden_dim: int, moe_intermediate: int,
                         num_experts: int,
                         expert_hf_names: set[str]) -> bytes:
    """Quantize and repack all experts for one layer.

    HF uses fused gate_up_proj [E, 2*I, H].  We split into gate [E, I, H]
    and up [E, I, H], quantize each expert with Wilkinson INT4, and pack.
    """
    # Find the gate_up_proj and down_proj keys for this layer
    gate_up_hf = None
    down_hf = None
    for name in expert_hf_names:
        if 'gate_up_proj' in name:
            gate_up_hf = name
        elif 'down_proj' in name:
            down_hf = name

    if gate_up_hf is None or down_hf is None:
        raise KeyError(f"Expert tensors not found for layer {layer_idx}")

    def load_bf16_tensor(key: str) -> np.ndarray:
        sf_name = weight_map[key]
        sf_path = model_path / sf_name
        header, data_start = header_cache[sf_name]
        data, shape, _ = read_tensor_raw(sf_path, header, key, data_start)
        return bf16_bytes_to_f32(data, shape)

    gate_up_f32 = load_bf16_tensor(gate_up_hf)   # [E, 2*I, H]
    down_f32 = load_bf16_tensor(down_hf)          # [E, H, I]

    inter = moe_intermediate
    hidden = hidden_dim
    gs = GROUP_SIZE

    # Wilkinson INT4 component sizes (weights + scales + biases)
    gate_w_bytes = inter * (hidden // 8) * 4  # uint32 packed weights
    gate_s_bytes = inter * (hidden // gs) * 2  # fp16 scales (2^E)
    gate_b_bytes = gate_s_bytes                # bf16 biases (B)
    up_w_bytes = gate_w_bytes
    up_s_bytes = gate_s_bytes
    up_b_bytes = gate_b_bytes
    down_w_bytes = hidden * (inter // 8) * 4
    down_s_bytes = hidden * (inter // gs) * 2
    down_b_bytes = down_s_bytes

    # Layout per expert: gate_w | gate_s | gate_b | up_w | up_s | up_b | down_w | down_s | down_b
    gate_w_off = 0
    gate_s_off = gate_w_off + gate_w_bytes
    gate_b_off = gate_s_off + gate_s_bytes
    up_w_off = gate_b_off + gate_b_bytes
    up_s_off = up_w_off + up_w_bytes
    up_b_off = up_s_off + up_s_bytes
    down_w_off = up_b_off + up_b_bytes
    down_s_off = down_w_off + down_w_bytes
    down_b_off = down_s_off + down_s_bytes
    expert_size = down_b_off + down_b_bytes

    buf = bytearray(num_experts * expert_size)

    for e in range(num_experts):
        gate_f32 = gate_up_f32[e, :inter, :]
        up_f32 = gate_up_f32[e, inter:2 * inter, :]
        down_f32_e = down_f32[e, :, :]

        gate_p, gate_s, gate_b = quant_f32_to_int4(gate_f32, inter, hidden)
        up_p, up_s, up_b = quant_f32_to_int4(up_f32, inter, hidden)
        down_p, down_s, down_b = quant_f32_to_int4(down_f32_e, hidden, inter)

        base = e * expert_size
        buf[base + gate_w_off:base + gate_w_off + gate_w_bytes] = gate_p.tobytes()
        buf[base + gate_s_off:base + gate_s_off + gate_s_bytes] = gate_s.tobytes()
        buf[base + gate_b_off:base + gate_b_off + gate_b_bytes] = gate_b.tobytes()
        buf[base + up_w_off:base + up_w_off + up_w_bytes] = up_p.tobytes()
        buf[base + up_s_off:base + up_s_off + up_s_bytes] = up_s.tobytes()
        buf[base + up_b_off:base + up_b_off + up_b_bytes] = up_b.tobytes()
        buf[base + down_w_off:base + down_w_off + down_w_bytes] = down_p.tobytes()
        buf[base + down_s_off:base + down_s_off + down_s_bytes] = down_s.tobytes()
        buf[base + down_b_off:base + down_b_off + down_b_bytes] = down_b.tobytes()

    return bytes(buf)


# ─── Verify-only pass ────────────────────────────────────────────────────

def _run_verify(model_path: Path, header_cache: dict,
                non_expert: dict[str, str], name_map: dict):
    """Quantize every non-expert tensor in memory, dequantize, compare, report.
    No files written."""
    sorted_tensors = sorted(non_expert.items(), key=lambda x: x[0])
    results: list[tuple[str, str, dict]] = []

    pbar = tqdm(sorted_tensors, desc="Verifying", unit="tensor")
    for orig_name, sf_name in pbar:
        sf_path = model_path / sf_name
        header, data_start = header_cache[sf_name]
        meta = header[orig_name]
        shape = meta['shape']
        off = meta['data_offsets']
        byte_len = off[1] - off[0]

        with open(sf_path, 'rb') as sf:
            sf.seek(data_start + off[0])
            raw_data = sf.read(byte_len)

        mlx_name = name_map[orig_name]
        q = bq4(mlx_name, shape)
        pbar.set_postfix_str(f"{q.value} {mlx_name.rsplit('.', 3)[-2][:50]}")

        source = bf16_bytes_to_f32(raw_data, shape)

        if q == Quant.INT4:
            out_dim, in_dim = shape
            padded_in = (in_dim + GROUP_SIZE - 1) // GROUP_SIZE * GROUP_SIZE
            if padded_in != in_dim:
                p = np.zeros((out_dim, padded_in), dtype=np.float32)
                p[:, :in_dim] = source
                source_vals = p
            else:
                source_vals = source

            packed, scales, biases = quant_f32_to_int4(source_vals, out_dim, padded_in)
            recon = int4_to_f32(packed, scales, biases, out_dim, padded_in)
            results.append((mlx_name, q.value,
                           verify(source_vals, recon,
                                  in_dim if padded_in != in_dim else None)))

        elif q == Quant.FP16:
            f16_data = f32_to_f16(source)
            recon = f16_data.view(np.float16).astype(np.float32).reshape(shape)
            results.append((mlx_name, q.value, verify(source, recon)))

        elif q == Quant.BF16:
            # Passthrough — no quantization error
            pass

        elif q == Quant.FP32:
            # Lossless conversion — no quantization error
            pass

    if not results:
        print("No quantized tensors to verify.")
        return

    by_dtype: dict[str, list[dict]] = defaultdict(list)
    for _, dt, m in results:
        by_dtype[dt].append(m)

    print(f"\n{'=' * 60}")
    print(f"Verification: {len(results)} quantized tensors")
    print(f"  {'dtype':>6s}  {'tensors':>7s}  {'max_err':>10s}  {'mean_err':>10s}  {'rmse':>10s}  {'snr_db':>8s}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for dt in sorted(by_dtype):
        metrics = by_dtype[dt]
        agg = {
            k: np.mean([m[k] for m in metrics])
            for k in ["max_err", "mean_err", "rmse", "snr_db"]
        }
        n = len(metrics)
        print(f"  {dt:>6s}  {n:>7d}  {agg['max_err']:10.6f}  {agg['mean_err']:10.6f}  "
              f"{agg['rmse']:10.6f}  {agg['snr_db']:8.1f}")

    worst = sorted(results, key=lambda x: -x[2]["max_err"])[:10]
    print(f"\n  Worst 10 by max_err:")
    for name, dt, m in worst:
        short = name.split('model.layers.', 1)[-1] if 'model.layers.' in name \
            else name.rsplit('.', 3)[-2]
        print(f"    {dt:>4s} max_err={m['max_err']:.6f} snr={m['snr_db']:.1f}dB  {short}")


# ─── Main ────────────────────────────────────────────────────────────────

def run(model_path_str: str, output_dir_str: str, *, strip: bool = False,
        verify_flag: bool = False):
    model_path = Path(model_path_str)
    output_dir = Path(output_dir_str)

    STRIP_LAYERS = 4
    STRIP_EXPERTS = 4

    if not verify_flag:
        output_dir.mkdir(parents=True, exist_ok=True)
        experts_dir = output_dir / "packed_experts"
        experts_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ──────────────────────────────────────────────────────
    config_path = model_path / "config.json"
    with open(config_path) as f:
        hf_config = json.load(f)
    tc = hf_config.get("text_config", hf_config)

    hidden_dim = tc["hidden_size"]
    num_layers = tc["num_hidden_layers"]
    moe_intermediate = tc["moe_intermediate_size"]
    num_experts = tc["num_experts"]
    num_experts_per_tok = tc["num_experts_per_tok"]
    shared_intermediate = tc["shared_expert_intermediate_size"]
    mtp_num_layers = tc.get("mtp_num_hidden_layers", 0)
    full_attn_interval = tc.get("full_attention_interval", 4)
    vocab_size = tc["vocab_size"]
    num_attn_heads = tc["num_attention_heads"]
    num_kv_heads = tc["num_key_value_heads"]
    head_dim = tc["head_dim"]

    num_main_layers = num_layers - mtp_num_layers
    num_vision_blocks = 27

    print(f"Model config:")
    print(f"  hidden_dim={hidden_dim}, vocab_size={vocab_size}")
    print(f"  num_layers={num_layers} (main={num_main_layers}, mtp={mtp_num_layers})")
    print(f"  num_experts={num_experts}, experts_per_tok={num_experts_per_tok}")
    print(f"  moe_intermediate={moe_intermediate}, shared_intermediate={shared_intermediate}")

    # ── Load name mapping ────────────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent
    mapping_path = script_dir / "name_mapping.json"
    if not mapping_path.exists():
        print(f"ERROR: {mapping_path} not found", file=sys.stderr)
        sys.exit(1)
    name_map = load_name_mapping(mapping_path, num_layers, num_vision_blocks)
    print(f"  Name mapping entries: {len(name_map)}")

    # ── Load weight map ──────────────────────────────────────────────────
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            idx = json.load(f)
        weight_map = idx["weight_map"]
    else:
        print("No safetensors index found, scanning shards...")
        weight_map = {}
        for sf_path in sorted(model_path.glob("model-*.safetensors")):
            header, _ = parse_safetensors_header(sf_path)
            for k in header:
                if k != "__metadata__":
                    weight_map[k] = sf_path.name

    print(f"  Total tensors: {len(weight_map)}")

    # ── Classify tensors ─────────────────────────────────────────────────
    non_expert = {}
    expert_names = {}
    unmapped = []

    for hf_name, filename in sorted(weight_map.items()):
        if hf_name not in name_map:
            unmapped.append(hf_name)
            continue
        mlx_name = name_map[hf_name]
        if is_expert_tensor(mlx_name):
            expert_names[hf_name] = filename
        else:
            non_expert[hf_name] = filename

    assert len(unmapped) == 0, \
        f"{len(unmapped)} tensors not in name_mapping.json:\n  " + \
        "\n  ".join(unmapped[:20])

    print(f"  Non-expert: {len(non_expert)}, Expert: {len(expert_names)}")

    # ── Strip mode ──────────────────────────────────────────────────────────
    if strip:
        num_layers = STRIP_LAYERS
        num_experts = STRIP_EXPERTS
        num_experts_per_tok = min(num_experts_per_tok, STRIP_EXPERTS)

        # Update architecturally-significant config fields
        hf_config["text_config" if "text_config" in hf_config else "_"]["num_hidden_layers"] = STRIP_LAYERS
        if "text_config" in hf_config:
            hf_config["text_config"]["num_hidden_layers"] = STRIP_LAYERS
        else:
            hf_config["num_hidden_layers"] = STRIP_LAYERS

        # Filter non-expert: keep non-layered tensors + layers 0..STRIP_LAYERS-1
        _layer_re = re.compile(r'layers\.(\d+)')
        stripped_non_expert = {}
        for hf_name, filename in non_expert.items():
            m = _layer_re.search(hf_name)
            if m and int(m.group(1)) >= STRIP_LAYERS:
                continue
            stripped_non_expert[hf_name] = filename
        stripped_expert_names = {}
        for hf_name, filename in expert_names.items():
            m = _layer_re.search(hf_name)
            if m and int(m.group(1)) >= STRIP_LAYERS:
                continue
            stripped_expert_names[hf_name] = filename

        print(f"  [strip] layers={STRIP_LAYERS}, experts={STRIP_EXPERTS}")
        print(f"  [strip] non-expert: {len(stripped_non_expert)} (was {len(non_expert)})")
        print(f"  [strip] expert: {len(stripped_expert_names)} (was {len(expert_names)})")
        non_expert = stripped_non_expert
        expert_names = stripped_expert_names

        # Update manifest config
        mtp_num_layers = 0  # strip mode has no MTP
        num_main_layers = STRIP_LAYERS

    # ── Cache safetensors headers ────────────────────────────────────────
    sf_files = set(non_expert.values()) | set(expert_names.values())
    header_cache = {}
    for sf_name in tqdm(sorted(sf_files), desc="Caching headers", unit="shard"):
        sf_path = model_path / sf_name
        if sf_path.exists():
            header_cache[sf_name] = parse_safetensors_header(sf_path)

    # ── Verify-only mode: quantize in memory, dequantize, compare, exit ──
    if verify_flag:
        _run_verify(model_path, header_cache, non_expert, name_map)
        return

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: Non-expert weights → model_weights.bin + .json
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("Quantizing non-expert weights (BQ4)...")
    print(f"{'=' * 60}")

    manifest = {
        "model": str(model_path),
        "config": {
            "hidden_size": hidden_dim,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_attn_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "moe_intermediate_size": moe_intermediate,
            "shared_expert_intermediate_size": shared_intermediate,
            "full_attention_interval": full_attn_interval,
            "linear_num_value_heads": tc.get("linear_num_value_heads", 32),
            "linear_num_key_heads": tc.get("linear_num_key_heads", 16),
            "linear_key_head_dim": tc.get("linear_key_head_dim", 128),
            "linear_value_head_dim": tc.get("linear_value_head_dim", 128),
            "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim", 4),
            "partial_rotary_factor": tc.get("partial_rotary_factor",
                tc.get("rope_parameters", {}).get("partial_rotary_factor", 0.25)),
            "rope_theta": tc.get("rope_theta",
                tc.get("rope_parameters", {}).get("rope_theta", 10000000.0)),
            "mtp_num_hidden_layers": mtp_num_layers,
        },
        "num_tensors": 0,
        "tensors": {},
    }

    # Layer type map
    layer_types = []
    for i in range(num_main_layers):
        if (i + 1) % full_attn_interval == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")
    manifest["config"]["layer_types"] = layer_types

    # ── Sanitization rules (from MLX qwen3_5.py) ─────────────────────────
    MLX_NORM_KEYS = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    MTP_NORM_KEYS = (
        ".hnorm.weight",
        ".enorm.weight",
        ".shared_head.norm.weight",
        ".norm1.weight",
        ".norm2.weight",
    )
    NORM_KEYS = MLX_NORM_KEYS + MTP_NORM_KEYS

    has_unsanitized_conv1d = any(
        "conv1d.weight" in k for k in non_expert
    )
    has_mtp = mtp_num_layers > 0
    should_shift_norm_weights = has_mtp or has_unsanitized_conv1d

    sorted_non_expert = sorted(non_expert.items(), key=lambda x: x[0])

    bin_path = output_dir / "model_weights.bin"
    t0 = time.time()
    offset = 0
    total_bytes = 0
    tensor_count = 0

    quant_summary = defaultdict(int)

    with open(bin_path, 'wb') as out_f:
        pbar = tqdm(sorted_non_expert, desc="Quantizing non-expert", unit="tensor")
        for orig_name, sf_name in pbar:
            sf_path = model_path / sf_name
            header, data_start = header_cache[sf_name]

            if orig_name not in header:
                print(f"  WARNING: {orig_name} not in {sf_name}, skipping")
                continue

            meta = header[orig_name]
            shape = meta['shape']
            dtype = meta['dtype']
            off = meta['data_offsets']
            byte_len = off[1] - off[0]

            mlx_name = name_map[orig_name]
            q = bq4(mlx_name, shape)
            pbar.set_postfix_str(f"{q.value} {mlx_name.rsplit('.', 3)[-2][:50]}")

            with open(sf_path, 'rb') as sf:
                sf.seek(data_start + off[0])
                raw_data = sf.read(byte_len)

            if offset % ALIGN != 0:
                pad = ALIGN - (offset % ALIGN)
                out_f.write(b'\x00' * pad)
                offset += pad

            if q == Quant.INT4:
                # Quantize BF16 → Wilkinson INT4 (weight + scales + biases)
                out_dim, in_dim = shape
                f32_vals = bf16_bytes_to_f32(raw_data, shape)

                # Pad inner dim to multiple of GROUP_SIZE
                padded_in = (in_dim + GROUP_SIZE - 1) // GROUP_SIZE * GROUP_SIZE
                if padded_in != in_dim:
                    p = np.zeros((out_dim, padded_in), dtype=np.float32)
                    p[:, :in_dim] = f32_vals
                    f32_vals = p

                packed, scales, biases = quant_f32_to_int4(f32_vals, out_dim, padded_in)

                base = mlx_name.removesuffix('.weight')
                for suffix, data, dt, pshape in [
                    (".weight", packed.tobytes(), "u32",
                     [out_dim, padded_in // 8]),
                    (".scales", scales.tobytes(), "f16",
                     [out_dim, padded_in // GROUP_SIZE]),
                    (".biases", biases.tobytes(), "bf16",
                     [out_dim, padded_in // GROUP_SIZE]),
                ]:
                    tensor_name = base + suffix
                    data_len = len(data)
                    out_f.write(data)
                    manifest["tensors"][tensor_name] = {
                        "offset": offset, "size": data_len,
                        "shape": pshape, "dtype": dt,
                    }
                    offset += data_len
                    total_bytes += data_len
                    tensor_count += 1

                quant_summary[q.value] += 1

            elif q == Quant.FP16:
                # Convert BF16 → FP16
                f32_vals = bf16_bytes_to_f32(raw_data, shape)
                f16_data = f32_to_f16(f32_vals)
                raw_out = f16_data.tobytes()

                out_f.write(raw_out)
                manifest["tensors"][mlx_name] = {
                    "offset": offset, "size": len(raw_out),
                    "shape": shape, "dtype": "f16",
                }
                offset += len(raw_out)
                total_bytes += len(raw_out)
                tensor_count += 1
                quant_summary[q.value] += 1

            elif q == Quant.BF16:
                # Pass through with sanitization
                out_dtype = "bf16"

                # Norm +1.0 shift
                if should_shift_norm_weights and any(
                        mlx_name.endswith(sk) for sk in NORM_KEYS):
                    f32_norm = bf16_bytes_to_f32(raw_data, shape)
                    f32_norm = f32_norm + 1.0
                    u16_norm = f32_to_bf16_u16(f32_norm)
                    raw_data = u16_norm.tobytes()

                # conv1d moveaxis (C,1,K) → (C,K,1)
                if "conv1d.weight" in mlx_name:
                    f32_c = bf16_bytes_to_f32(raw_data, shape)
                    f32_c = np.moveaxis(f32_c, 2, 1)
                    raw_data = f32_to_bf16_u16(f32_c).tobytes()
                    shape = list(f32_c.shape)

                out_f.write(raw_data)
                manifest["tensors"][mlx_name] = {
                    "offset": offset, "size": len(raw_data),
                    "shape": shape, "dtype": out_dtype,
                }
                offset += len(raw_data)
                total_bytes += len(raw_data)
                tensor_count += 1
                quant_summary[q.value] += 1

            elif q == Quant.FP32:
                # Convert BF16 → F32
                f32_data = bf16_bytes_to_f32(raw_data, shape)
                raw_out = f32_data.tobytes()

                out_f.write(raw_out)
                manifest["tensors"][mlx_name] = {
                    "offset": offset, "size": len(raw_out),
                    "shape": shape, "dtype": "f32",
                }
                offset += len(raw_out)
                total_bytes += len(raw_out)
                tensor_count += 1
                quant_summary[q.value] += 1

    manifest["num_tensors"] = tensor_count

    elapsed = time.time() - t0
    print(f"  {tensor_count} tensors, {total_bytes / 1e9:.2f} GB")
    print(f"  Written in {elapsed:.1f}s ({total_bytes / elapsed / 1e9:.1f} GB/s)")
    print(f"  By dtype: {dict(quant_summary)}")

    # Write manifest
    json_path = output_dir / "model_weights.json"
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {json_path}")

    # ── Category summary ─────────────────────────────────────────────────
    categories = defaultdict(lambda: {"count": 0, "bytes": 0})
    for tname, info in manifest["tensors"].items():
        if "embed_tokens" in tname:
            cat = "embedding"
        elif ".norm." in tname and "layers." not in tname:
            cat = "final_norm"
        elif "lm_head" in tname:
            cat = "lm_head"
        elif any(x in tname for x in ("input_layernorm", "post_attention_layernorm")):
            cat = "layer_norms"
        elif "linear_attn" in tname:
            cat = "linear_attention"
        elif "self_attn" in tname:
            cat = "full_attention"
        elif "mlp.gate." in tname:
            cat = "routing_gate"
        elif "shared_expert." in tname:
            cat = "shared_expert"
        elif "shared_expert_gate" in tname:
            cat = "shared_expert_gate"
        elif "switch_mlp" in tname:
            cat = "routed_experts"
        elif "eh_proj" in tname:
            cat = "mtp_eh_proj"
        elif any(x in tname for x in ("enorm", "hnorm")):
            cat = "mtp_norms"
        elif "shared_head" in tname:
            cat = "mtp_shared_head"
        elif ".norm" in tname and ".norm." in tname:
            cat = "layer_norms"
        elif "conv1d" in tname:
            cat = "conv1d"
        else:
            cat = "other"
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += info["size"]

    print("\nWeight categories:")
    for cat in sorted(categories):
        info = categories[cat]
        print(f"  {cat:25s}: {info['count']:4d} tensors, {info['bytes'] / 1e6:8.1f} MB")

    # Copy config.json
    src_config = model_path / "config.json"
    dst_config = output_dir / "config.json"
    if src_config.exists():
        shutil.copy2(src_config, dst_config)
        print(f"  Copied config.json")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: Expert weights → packed_experts/layer_XX.bin
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print("Quantizing expert weights (wint4)...")
    print(f"{'=' * 60}")

    t1 = time.time()

    # Group expert tensors by layer index
    expert_layers = defaultdict(set)
    for hf_name in expert_names:
        m = re.search(r'layers\.(\d+)', hf_name)
        if m:
            expert_layers[int(m.group(1))].add(hf_name)
        elif 'mtp' in hf_name:
            mm = re.search(r'mtp\.layers\.(\d+)', hf_name)
            if mm:
                expert_layers[num_main_layers + int(mm.group(1))].add(hf_name)

    expert_layers_done = 0
    for layer_idx in tqdm(sorted(expert_layers.keys()),
                          desc="Quantizing experts", unit="layer"):
        try:
            data = repack_experts_layer(
                model_path, header_cache, weight_map,
                layer_idx,
                hidden_dim, moe_intermediate, num_experts,
                expert_layers[layer_idx],
            )
        except KeyError as e:
            tqdm.write(f"  Layer {layer_idx:2d} SKIPPED ({e})")
            continue

        out_path = experts_dir / f"layer_{layer_idx:02d}.bin"
        with open(out_path, 'wb') as f:
            f.write(data)
        tqdm.write(f"  Layer {layer_idx:02d}: {len(data) / 1e6:.1f} MB → {out_path.name}")
        expert_layers_done += 1

    t2 = time.time()
    print(f"\n  {expert_layers_done} expert layers in {t2 - t1:.1f}s")

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"  model_weights.bin : {os.path.getsize(bin_path) / 1e9:.2f} GB")
    print(f"  model_weights.json: {json_path}")
    print(f"  packed_experts    : {expert_layers_done} layers")
    print(f"  Total time        : {t2 - t0:.1f}s")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize HF BF16 Qwen MoE model → BQ4 format")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to HuggingFace model directory (BF16 safetensors)')
    parser.add_argument('--output', type=str,
                        default='data/models--Qwen--Qwen3.6-35B-A3B-bq4',
                        help='Output directory')
    parser.add_argument('--strip', action='store_true',
                        help='Strip to 4 layers × 4 experts for verification')
    parser.add_argument('--verify', action='store_true',
                        help='Dequantize and compare against source after each tensor')
    args = parser.parse_args()
    run(args.model, args.output, strip=args.strip, verify_flag=args.verify)


if __name__ == '__main__':
    main()

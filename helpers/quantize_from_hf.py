#!/usr/bin/env python3
"""
quantize_from_hf.py — Quantize original HuggingFace BF16 Qwen3.5/3.6 MoE model
to flash-moe's 4-bit runtime format.

Combines the MLX-LM quantization pipeline with extract_weights.py binary layout:
  1. MLX-LM sanitize (norm +1.0 shift, conv1d moveaxis, gate_up split)
  2. BF16 → 4-bit affine quantization (matching MLX's mx.quantize)
  3. Binary layout + manifest (matching extract_weights.py)

Outputs:
  - model_weights.bin      : all non-expert weights (mmap'd by the engine)
  - model_weights.json     : manifest with tensor offsets, shapes, dtypes, config
  - packed_experts/layer_XX.bin : per-layer concatenated expert weights

Usage:
    python helpers/quantize_from_hf.py \
        --model hub/models--Qwen--Qwen3.6-35B-A3B \
        --output data/my-model
"""

import argparse
import json
import os
import re
import shutil
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ─── BF16 ↔ F32 ────────────────────────────────────────────────────────────────

# Source: extract_weights.py:40-42, vectorized
def bf16_bytes_to_f32(data: bytes, shape: list[int]) -> np.ndarray:
    """Convert BF16 raw bytes to float32 via vectorized left-shift."""
    u16 = np.frombuffer(data, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape)


# Source: extract_weights.py:45-54, vectorized
def f32_vec_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 array to BF16 uint16 (round-to-nearest-even)."""
    u32 = arr.view(np.uint32)
    round_bit = (u32 >> 15) & 1
    sticky = u32 & 0x7FFF
    round_up = round_bit & (sticky | ((u32 >> 16) & 1))
    u32 = u32 + (round_up.astype(np.uint32) << 16)
    return (u32 >> 16).astype(np.uint16)


# ─── 4-bit quantization ────────────────────────────────────────────────────────

GROUP_SIZE = 64
ALIGN = 64


# Source: extract_weights.py:82-123, vectorized for performance.
# Algorithm is identical: per-group min/max → affine quantize to 4-bit → pack LSB-first.
def quant_f32_to_4bit_packed(values: np.ndarray, out_dim: int, in_dim: int
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize float32 [out_dim, in_dim] to 4-bit packed.

    Returns (packed_uint32, scales_bf16, biases_bf16).
    """
    num_groups = in_dim // GROUP_SIZE
    total = out_dim * in_dim
    packed = np.zeros(total // 8, dtype=np.uint32)
    scales = np.zeros(out_dim * num_groups, dtype=np.uint16)
    biases = np.zeros(out_dim * num_groups, dtype=np.uint16)

    CHUNK_ROWS = max(1, min(4096, 64 * 1024 * 1024 // in_dim))

    for row_start in range(0, out_dim, CHUNK_ROWS):
        row_end = min(row_start + CHUNK_ROWS, out_dim)
        chunk_rows = row_end - row_start

        v = values[row_start:row_end].reshape(chunk_rows, num_groups, GROUP_SIZE)

        vmin = v.min(axis=2)
        vmax = v.max(axis=2)
        degenerate = (vmax == vmin)
        vmax[degenerate] = vmin[degenerate] + 1.0

        fscale = (vmax - vmin) / 15.0
        fbias = vmin

        s_idx_start = row_start * num_groups
        s_idx_end = row_end * num_groups
        scales[s_idx_start:s_idx_end] = f32_vec_to_bf16_u16(fscale).ravel()
        biases[s_idx_start:s_idx_end] = f32_vec_to_bf16_u16(fbias).ravel()

        v_centered = v - fbias[:, :, np.newaxis]
        q = np.round(v_centered / fscale[:, :, np.newaxis])
        q = q.clip(0, 15).astype(np.uint32)

        # Pack nibbles LSB-first (matching MLX's pack_bits)
        q_packed = q.reshape(chunk_rows, num_groups, GROUP_SIZE // 8, 8)
        shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
        packed_words = ((q_packed & 0xF) << shifts) \
            .sum(axis=3).astype(np.uint32) \
            .reshape(chunk_rows, -1)

        words_per_row = in_dim // 8
        for r in range(chunk_rows):
            row_idx = row_start + r
            w_start = row_idx * words_per_row
            packed[w_start:w_start + words_per_row] = packed_words[r]

    return packed, scales, biases


# ─── Safetensors I/O ───────────────────────────────────────────────────────────

# Source: extract_weights.py:31-37
def parse_safetensors_header(filepath: Path) -> tuple[dict, int]:
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


# Source: extract_weights.py:310-319 (read_tensor inner function)
def read_tensor_raw(filepath: Path, header: dict, name: str,
                    data_start: int) -> tuple[bytes, list[int], str]:
    meta = header[name]
    off = meta['data_offsets']
    length = off[1] - off[0]
    with open(filepath, 'rb') as f:
        f.seek(data_start + off[0])
        data = f.read(length)
    return data, meta['shape'], meta['dtype']


# ─── Tensor classification ─────────────────────────────────────────────────────

# Source: extract_weights.py:145
def is_expert_tensor(name: str) -> bool:
    """MoE expert weight (gate_up_proj or down_proj) — goes to packed_experts.

    Matches: model.language_model.layers.{i}.mlp.experts.{gate_up_proj,down_proj}
             mtp.layers.{b}.mlp.experts.{gate_up_proj,down_proj}
             *.switch_mlp.{gate_proj,up_proj,down_proj}
    """
    return bool(re.search(
        r'(?:mlp\.experts|switch_mlp)\.(gate_up_proj|gate_proj|up_proj|down_proj)',
        name))


# Source: extract_weights.py:147
def is_vision_tensor(name: str) -> bool:
    return bool(re.match(r'^(vision_tower|model\.visual)', name))


# Source: mlx.nn.quantize behavior — all modules with to_quantized()
# (nn.Linear, nn.Embedding, nn.BitLinear) get their 2D weight matrices quantized.
# At the raw tensor level this means: 2D tensors ending in .weight are quantized.
# The wrapped_predicate (vendor/mlx-lm/mlx_lm/utils.py:823-827) also checks
# weight.shape[-1] % group_size == 0, always true here (hidden=2048, inter=512, gs=64).
def should_quantize_4bit(name: str, shape: list[int]) -> bool:
    return len(shape) == 2 and name.endswith(".weight")


# ─── Name sanitization ─────────────────────────────────────────────────────────

# Source: extract_weights.py:184-191 (prefix stripping)
# Plus MTP remapping matching qwen3_5_moe.py:23-52 (gate_up split)
# and qwen3_5.py:307-331 (MTP filter + norm shift).
def sanitize_name(name: str, mtp_layer_idx: int | None = None) -> str | None:
    """Strip HF/model prefix, remap MTP tensors to flash-moe naming.

    HF uses:
      model.language_model.layers.{i}.*  -> model.layers.{i}.*
      mtp.fc.*                           -> model.layers.{mtp}.eh_proj.*
      mtp.pre_fc_norm_embedding.*        -> model.layers.{mtp}.enorm.*
      mtp.pre_fc_norm_hidden.*           -> model.layers.{mtp}.hnorm.*
      mtp.norm.*                         -> model.layers.{mtp}.shared_head.norm.*
      mtp.layers.{b}.*                   -> SKIP (duplicate of layers.{mtp+b}.*)

    Returns None for tensors that should be skipped.
    """
    # Source: extract_weights.py:187-189
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model."):]
    elif name.startswith("language_model."):
        name = "model." + name[len("language_model."):]

    if mtp_layer_idx is None:
        return name

    # Source: qwen3_5.py:313 — filter out mtp.* weights (they're duplicates).
    # But we WANT MTP weights (eh_proj, enorm, hnorm, shared_head.norm).
    # We only skip mtp.layers.* which are duplicates of model.language_model.layers.*
    if name.startswith("mtp."):
        # Source: qwen3_5_moe.py:36-48 — MTP layers share experts with main layers.
        # mtp.layers.{b}.* duplicates model.language_model.layers.{mtp+b}.* → SKIP
        if re.match(r'mtp\.layers\.(\d+)\.', name):
            return None

        # Map MTP-specific tensors to canonical names.
        # These have no analogue in the main layers, so MLX filters them out,
        # but we preserve them for flash-moe's MTP speculative decoding.
        for mtp_stem, mapped_stem in [
            ("mtp.fc",                    f"model.layers.{mtp_layer_idx}.eh_proj"),
            ("mtp.pre_fc_norm_embedding", f"model.layers.{mtp_layer_idx}.enorm"),
            ("mtp.pre_fc_norm_hidden",    f"model.layers.{mtp_layer_idx}.hnorm"),
            ("mtp.norm",                  f"model.layers.{mtp_layer_idx}.shared_head.norm"),
        ]:
            if name.startswith(mtp_stem):
                suffix = name[len(mtp_stem):]
                return mapped_stem + suffix

    return name


# ─── Per-layer expert repacking ────────────────────────────────────────────────

# Source: qwen3_5_moe.py:36-48 (gate_up split)
# Layout: config.rs ExpertLayout (gate_w|gate_s|gate_b|up_w|up_s|up_b|down_w|down_s|down_b)
def repack_experts_layer(model_path: Path, header_cache: dict,
                         weight_map: dict, layer_idx: int,
                         hidden_dim: int, moe_intermediate: int,
                         num_experts: int) -> bytes:
    """Quantize and repack all experts for one layer into binary layout.

    HF uses fused gate_up_proj [E, 2*I, H]. We split into gate [E, I, H]
    and up [E, I, H] (matching qwen3_5_moe.py:41-46), quantize each expert,
    and pack per ExpertLayout offsets.
    """
    gate_up_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
    down_key = f"model.language_model.layers.{layer_idx}.mlp.experts.down_proj"

    if gate_up_key not in weight_map or down_key not in weight_map:
        raise KeyError(f"Expert tensors not found for layer {layer_idx}")

    def load_bf16_tensor(key: str) -> np.ndarray:
        sf_name = weight_map[key]
        sf_path = model_path / sf_name
        header, data_start = header_cache[sf_name]
        data, shape, _ = read_tensor_raw(sf_path, header, key, data_start)
        return bf16_bytes_to_f32(data, shape)

    gate_up_f32 = load_bf16_tensor(gate_up_key)   # [E, 2*I, H]
    down_f32 = load_bf16_tensor(down_key)          # [E, H, I]

    inter = moe_intermediate
    hidden = hidden_dim
    gs = GROUP_SIZE

    # Component byte sizes (matching Rust ExpertLayout in config.rs)
    gate_w_bytes = inter * (hidden // 8) * 4
    gate_sb_bytes = inter * (hidden // gs) * 2
    up_w_bytes = inter * (hidden // 8) * 4
    up_sb_bytes = inter * (hidden // gs) * 2
    down_w_bytes = hidden * (inter // 8) * 4
    down_sb_bytes = hidden * (inter // gs) * 2

    # Offsets within one expert: gate_w | gate_s | gate_b | up_w | up_s | up_b | down_w | down_s | down_b
    gate_w_off = 0
    gate_s_off = gate_w_off + gate_w_bytes
    gate_b_off = gate_s_off + gate_sb_bytes
    up_w_off = gate_b_off + gate_sb_bytes
    up_s_off = up_w_off + up_w_bytes
    up_b_off = up_s_off + up_sb_bytes
    down_w_off = up_b_off + up_sb_bytes
    down_s_off = down_w_off + down_w_bytes
    down_b_off = down_s_off + down_sb_bytes
    expert_size = down_b_off + down_sb_bytes

    buf = bytearray(num_experts * expert_size)

    for e in range(num_experts):
        gate_f32 = gate_up_f32[e, :inter, :]
        up_f32   = gate_up_f32[e, inter:2*inter, :]
        down_f32_e = down_f32[e, :, :]

        gate_p, gate_s, gate_b = quant_f32_to_4bit_packed(gate_f32, inter, hidden)
        up_p,   up_s,   up_b   = quant_f32_to_4bit_packed(up_f32,   inter, hidden)
        down_p, down_s, down_b = quant_f32_to_4bit_packed(down_f32_e, hidden, inter)

        base = e * expert_size
        buf[base + gate_w_off:base + gate_w_off + gate_w_bytes] = gate_p.tobytes()
        buf[base + gate_s_off:base + gate_s_off + gate_sb_bytes] = gate_s.tobytes()
        buf[base + gate_b_off:base + gate_b_off + gate_sb_bytes] = gate_b.tobytes()
        buf[base + up_w_off:base + up_w_off + up_w_bytes] = up_p.tobytes()
        buf[base + up_s_off:base + up_s_off + up_sb_bytes] = up_s.tobytes()
        buf[base + up_b_off:base + up_b_off + up_sb_bytes] = up_b.tobytes()
        buf[base + down_w_off:base + down_w_off + down_w_bytes] = down_p.tobytes()
        buf[base + down_s_off:base + down_s_off + down_sb_bytes] = down_s.tobytes()
        buf[base + down_b_off:base + down_b_off + down_sb_bytes] = down_b.tobytes()

    return bytes(buf)


# ─── Main ──────────────────────────────────────────────────────────────────────

def run(model_path_str: str, output_dir_str: str, *, qwen36: bool = False):
    model_path = Path(model_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    experts_dir = output_dir / "packed_experts"
    experts_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config (source: extract_weights.py:203-236) ───────────────────
    config_path = model_path / "config.json"
    with open(config_path) as f:
        hf_config = json.load(f)
    tc = hf_config.get("text_config", hf_config)

    hidden_dim          = tc["hidden_size"]
    num_layers          = tc["num_hidden_layers"]
    moe_intermediate    = tc["moe_intermediate_size"]
    num_experts         = tc["num_experts"]
    num_experts_per_tok = tc["num_experts_per_tok"]
    shared_intermediate = tc["shared_expert_intermediate_size"]
    mtp_num_layers      = tc.get("mtp_num_hidden_layers", 0)
    full_attn_interval  = tc.get("full_attention_interval", 4)
    vocab_size          = tc["vocab_size"]
    num_attn_heads      = tc["num_attention_heads"]
    num_kv_heads        = tc["num_key_value_heads"]
    head_dim            = tc["head_dim"]

    num_main_layers = num_layers - mtp_num_layers
    mtp_layer_idx = num_main_layers if mtp_num_layers > 0 else None

    has_mtp = mtp_num_layers > 0

    print(f"Model config:")
    print(f"  hidden_dim={hidden_dim}, vocab_size={vocab_size}")
    print(f"  num_layers={num_layers} (main={num_main_layers}, mtp={mtp_num_layers})")
    print(f"  num_experts={num_experts}, experts_per_tok={num_experts_per_tok}")
    print(f"  moe_intermediate={moe_intermediate}, shared_intermediate={shared_intermediate}")

    # ── Load weight map (source: extract_weights.py:132-140) ───────────────
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

    # ── Classify tensors (source: extract_weights.py:149-160) ──────────────
    non_expert = {}     # name -> filename
    expert_names = {}   # name -> filename
    skipped_vision = 0

    for name, filename in sorted(weight_map.items()):
        if is_vision_tensor(name):
            skipped_vision += 1
            continue
        if is_expert_tensor(name):
            expert_names[name] = filename
        else:
            non_expert[name] = filename

    print(f"  Skipped vision: {skipped_vision}")
    print(f"  Non-expert: {len(non_expert)}, Expert: {len(expert_names)}")

    # ── Cache safetensors headers (source: extract_weights.py:174-178) ─────
    sf_files = set(non_expert.values()) | set(expert_names.values())
    header_cache = {}
    for sf_name in tqdm(sorted(sf_files), desc="Caching headers", unit="shard"):
        sf_path = model_path / sf_name
        if sf_path.exists():
            header_cache[sf_name] = parse_safetensors_header(sf_path)

    # ══════════════════════════════════════════════════════════════════════
    # PART 1: Non-expert weights -> model_weights.bin + .json
    # Source: extract_weights.py:359-424 (binary writing + manifest)
    # Plus MLX qwen3_5.py:307-331 (sanitize transforms)
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print("Quantizing non-expert weights...")
    print(f"{'='*60}")

    # Source: extract_weights.py:210-265 (manifest + config)
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

    # Source: extract_weights.py:269-275
    layer_types = []
    for i in range(num_main_layers):
        if (i + 1) % full_attn_interval == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")
    manifest["config"]["layer_types"] = layer_types

    sorted_non_expert = sorted(non_expert.items(), key=lambda x: x[0])

    bin_path = output_dir / "model_weights.bin"
    t0 = time.time()
    offset = 0
    total_bytes = 0
    tensor_count = 0

    # NOTE: Qwen3.6 norm weights are shifted by -1.0 vs Qwen3.5 convention.
    # Pass --qwen36 to normalize them (+1.0) so engines can treat them uniformly.
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
    )
    NORM_KEYS = MLX_NORM_KEYS + MTP_NORM_KEYS

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

            pbar.set_postfix_str(orig_name.rsplit('.', 2)[-2][:50] if '.' in orig_name else orig_name[:50])

            with open(sf_path, 'rb') as sf:
                sf.seek(data_start + off[0])
                raw_data = sf.read(byte_len)

            san_name = sanitize_name(orig_name, mtp_layer_idx)
            if san_name is None:
                continue  # skip mtp.layers.* duplicates (qwen3_5_moe.py:36-48)

            # Source: mlx_lm/utils.py quantize_model wrapped_predicate
            if should_quantize_4bit(san_name, shape):
                f32_vals = bf16_bytes_to_f32(raw_data, shape)
                out_dim, in_dim = shape

                # Source: extract_weights.py:82-123 (quant_f32_to_4bit_packed)
                packed, scales, biases = quant_f32_to_4bit_packed(
                    f32_vals, out_dim, in_dim)

                base = san_name.removesuffix(".weight")
                w_packed_shape = [out_dim, in_dim // 8]
                sb_shape = [out_dim, in_dim // GROUP_SIZE]
                for suffix, data, dt, pshape in [
                    (".weight", packed.tobytes(), "U32", w_packed_shape),
                    (".scales", scales.tobytes(), "BF16", sb_shape),
                    (".biases", biases.tobytes(), "BF16", sb_shape),
                ]:
                    tensor_name = base + suffix
                    data_len = len(data)
                    if offset % ALIGN != 0:
                        pad = ALIGN - (offset % ALIGN)
                        out_f.write(b'\x00' * pad)
                        offset += pad
                    out_f.write(data)
                    manifest["tensors"][tensor_name] = {
                        "offset": offset, "size": data_len,
                        "shape": pshape, "dtype": dt,
                    }
                    offset += data_len
                    total_bytes += data_len
                    tensor_count += 1
            else:
                out_dtype = dtype

                # Qwen3.6 norm +1.0 shift: normalize to Qwen3.5 convention
                if qwen36 and out_dtype == "BF16" \
                        and any(san_name.endswith(sk) for sk in NORM_KEYS):
                    f32_norm = bf16_bytes_to_f32(raw_data, shape)
                    f32_norm = f32_norm + 1.0
                    u16_norm = f32_vec_to_bf16_u16(f32_norm)
                    raw_data = u16_norm.tobytes()

                # Source: qwen3_5.py:348-349 (cast_predicate) — A_log stays F32
                if "A_log" in san_name and out_dtype != "F32":
                    raw_data = bf16_bytes_to_f32(raw_data, shape).tobytes()
                    out_dtype = "F32"

                # Source: qwen3_5.py:326-327 — conv1d moveaxis(2, 1)
                if "conv1d.weight" in san_name:
                    f32_c = bf16_bytes_to_f32(raw_data, shape)
                    f32_c = np.moveaxis(f32_c, 2, 1)
                    raw_data = f32_vec_to_bf16_u16(f32_c).tobytes()
                    shape = list(f32_c.shape)

                if offset % ALIGN != 0:
                    pad = ALIGN - (offset % ALIGN)
                    out_f.write(b'\x00' * pad)
                    offset += pad

                out_f.write(raw_data)
                manifest["tensors"][san_name] = {
                    "offset": offset, "size": len(raw_data),
                    "shape": shape, "dtype": out_dtype,
                }
                offset += len(raw_data)
                total_bytes += len(raw_data)
                tensor_count += 1

    manifest["num_tensors"] = tensor_count

    elapsed = time.time() - t0
    print(f"  {tensor_count} tensors, {total_bytes / 1e9:.2f} GB")
    print(f"  Written in {elapsed:.1f}s ({total_bytes / elapsed / 1e9:.1f} GB/s)")

    # Write manifest (source: extract_weights.py:421-423)
    json_path = output_dir / "model_weights.json"
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {json_path}")

    # Summary by category (source: extract_weights.py:427-457)
    categories = defaultdict(lambda: {"count": 0, "bytes": 0})
    for tname, info in manifest["tensors"].items():
        if "embed_tokens" in tname:
            cat = "embedding"
        elif "norm.weight" in tname and "layers." not in tname:
            cat = "final_norm"
        elif "lm_head" in tname:
            cat = "lm_head"
        elif any(x in tname for x in ("input_layernorm", "post_attention_layernorm")):
            cat = "layer_norms"
        elif "linear_attn" in tname:
            cat = "linear_attention"
        elif "self_attn" in tname:
            cat = "full_attention"
        elif "mlp.gate." in tname or "mlp.gate_" in tname:
            cat = "routing_gate"
        elif "shared_expert." in tname:
            cat = "shared_expert"
        elif "shared_expert_gate" in tname:
            cat = "shared_expert_gate"
        elif "eh_proj" in tname:
            cat = "mtp_eh_proj"
        elif any(x in tname for x in ("enorm", "hnorm")):
            cat = "mtp_norms"
        elif "shared_head" in tname:
            cat = "mtp_shared_head"
        else:
            cat = "other"
        categories[cat]["count"] += 1
        categories[cat]["bytes"] += info["size"]

    print("\nWeight categories:")
    for cat in sorted(categories):
        info = categories[cat]
        print(f"  {cat:25s}: {info['count']:4d} tensors, {info['bytes']/1e6:8.1f} MB")

    # Copy config.json (source: extract_weights.py behavior)
    src_config = model_path / "config.json"
    dst_config = output_dir / "config.json"
    if src_config.exists():
        shutil.copy2(src_config, dst_config)
        print(f"  Copied config.json")

    # ══════════════════════════════════════════════════════════════════════
    # PART 2: Expert weights -> packed_experts/layer_XX.bin
    # Source: qwen3_5_moe.py:36-48 (gate_up split) + config.rs (ExpertLayout)
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print("Quantizing expert weights...")
    print(f"{'='*60}")

    t1 = time.time()
    expert_layers_done = 0

    # Group expert tensors by layer index
    expert_layers = defaultdict(set)
    for name in expert_names:
        m = re.search(r'layers\.(\d+)', name)
        if m:
            expert_layers[int(m.group(1))].add(name)
        elif 'mtp' in name:
            mm = re.search(r'mtp\.layers\.(\d+)', name)
            if mm:
                expert_layers[num_main_layers + int(mm.group(1))].add(name)

    for layer_idx in tqdm(sorted(expert_layers.keys()), desc="Quantizing experts", unit="layer"):
        is_mtp_layer = layer_idx >= num_main_layers
        mtp_sub_idx = layer_idx - num_main_layers if is_mtp_layer else None

        if is_mtp_layer:
            desc = f"MTP head #{mtp_sub_idx}"
        else:
            desc = f"layer {layer_idx}/{num_main_layers - 1}"

        try:
            data = repack_experts_layer(
                model_path, header_cache, weight_map,
                layer_idx,
                hidden_dim, moe_intermediate, num_experts,
            )
        except KeyError as e:
            tqdm.write(f"  Layer {layer_idx:2d} SKIPPED ({e})")
            continue

        out_path = experts_dir / f"layer_{layer_idx:02d}.bin"
        with open(out_path, 'wb') as f:
            f.write(data)
        tqdm.write(f"  {desc}: {len(data) / 1e6:.1f} MB -> {out_path.name}")
        expert_layers_done += 1

    t2 = time.time()
    print(f"\n  {expert_layers_done} expert layers in {t2 - t1:.1f}s")

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Done!")
    print(f"  model_weights.bin : {os.path.getsize(bin_path) / 1e9:.2f} GB")
    print(f"  model_weights.json: {json_path}")
    print(f"  packed_experts    : {expert_layers_done} layers")
    print(f"  Total time        : {t2 - t0:.1f}s")
    print(f"\n  To run:")
    print(f"    python chat.py --model {output_dir} --tokenizer {model_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize HF BF16 Qwen MoE model -> flash-moe 4-bit format")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to HuggingFace model directory (with BF16 safetensors)')
    parser.add_argument('--output', type=str, default='data/models--Qwen--Qwen3.6-35B-A3B-4bit',
                        help='Output directory')
    parser.add_argument('--qwen36', action='store_true',
                        help='Apply +1.0 shift to norm weights (Qwen3.6 → 3.5 convention)')
    args = parser.parse_args()
    run(args.model, args.output, qwen36=args.qwen36)


if __name__ == '__main__':
    main()

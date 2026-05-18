#!/usr/bin/env python3
"""
gen_model_config.py — Generate model_config.h from a HuggingFace model's config.json.

Computes all #define constants needed by the Flash-MoE inference engine,
including derived values like expert sizes, offsets, and layer counts.

Usage:
    python gen_model_config.py --model ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B
    python gen_model_config.py --model /path/to/model --output model_config_qwen35_35b.h
"""

import argparse
import json
import os
import sys
from pathlib import Path


GROUP_SIZE = 64
BITS = 4
HEAD_DIM = 256
CONV_KERNEL_SIZE = 4
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10_000_000.0
PARTIAL_ROTARY = 0.25
FULL_ATTN_INTERVAL = 4
LINEAR_KEY_DIM = 128
LINEAR_VALUE_DIM = 128
MAX_SEQ_LEN = 1_048_576
GPU_KV_SEQ = 8192
MAX_K = 8

# Special tokens (same across Qwen3 family)
EOS_TOKEN_1 = 248046
EOS_TOKEN_2 = 248044
THINK_START_TOKEN = 248068
THINK_END_TOKEN = 248069


def load_hf_config(model_path):
    """Load model dimensions from HuggingFace config.json."""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)
    # Qwen3.5 MoE multimodal models nest text params under "text_config"
    if "text_config" in cfg:
        cfg = cfg["text_config"]
    return cfg


def compute_expert_layout(hidden_dim, moe_intermediate):
    """Compute 4-bit and 2-bit expert component sizes and offsets.

    Weights are packed as uint32 (4 bytes each), each holding 8 4-bit values.
    Scales and biases are bf16 (2 bytes each), one per group of GROUP_SIZE elements.
    """
    # Packed weight bytes: elements / 8 values_per_uint32 * 4 bytes_per_uint32 = elements / 2
    gate_w = moe_intermediate * hidden_dim // 2
    gate_sb = moe_intermediate * (hidden_dim // GROUP_SIZE) * 2
    up_w = gate_w
    up_sb = gate_sb
    down_w = hidden_dim * moe_intermediate // 2
    down_sb = hidden_dim * (moe_intermediate // GROUP_SIZE) * 2

    gate_w_off = 0
    gate_s_off = gate_w
    gate_b_off = gate_w + gate_sb
    up_w_off = gate_w + 2 * gate_sb
    up_s_off = up_w_off + up_w
    up_b_off = up_s_off + up_sb
    down_w_off = up_b_off + up_sb
    down_s_off = down_w_off + down_w
    down_b_off = down_s_off + down_sb
    expert_size = down_b_off + down_sb

    # 2-bit: 16 values per uint32 instead of 8, so packed bytes = elements / 4
    gate_w2 = moe_intermediate * hidden_dim // 4
    up_w2 = gate_w2
    down_w2 = hidden_dim * moe_intermediate // 4

    gate_w_off_2 = 0
    gate_s_off_2 = gate_w2
    gate_b_off_2 = gate_w2 + gate_sb
    up_w_off_2 = gate_w2 + 2 * gate_sb
    up_s_off_2 = up_w_off_2 + up_w2
    up_b_off_2 = up_s_off_2 + up_sb
    down_w_off_2 = up_b_off_2 + up_sb
    down_s_off_2 = down_w_off_2 + down_w2
    down_b_off_2 = down_s_off_2 + down_sb
    expert_size_2bit = down_b_off_2 + down_sb

    return {
        "expert_size": expert_size,
        "expert_size_2bit": expert_size_2bit,
        # 4-bit component sizes
        "gate_w_size": gate_w, "gate_s_size": gate_sb, "gate_b_size": gate_sb,
        "up_w_size": up_w, "up_s_size": up_sb, "up_b_size": up_sb,
        "down_w_size": down_w, "down_s_size": down_sb, "down_b_size": down_sb,
        "gate_w_off": gate_w_off, "gate_s_off": gate_s_off, "gate_b_off": gate_b_off,
        "up_w_off": up_w_off, "up_s_off": up_s_off, "up_b_off": up_b_off,
        "down_w_off": down_w_off, "down_s_off": down_s_off, "down_b_off": down_b_off,
        # 2-bit component sizes
        "gate_w_size_2": gate_w2, "gate_s_size_2": gate_sb, "gate_b_size_2": gate_sb,
        "up_w_size_2": up_w2, "up_s_size_2": up_sb, "up_b_size_2": up_sb,
        "down_w_size_2": down_w2, "down_s_size_2": down_sb, "down_b_size_2": down_sb,
        "gate_w_off_2": gate_w_off_2, "gate_s_off_2": gate_s_off_2, "gate_b_off_2": gate_b_off_2,
        "up_w_off_2": up_w_off_2, "up_s_off_2": up_s_off_2, "up_b_off_2": up_b_off_2,
        "down_w_off_2": down_w_off_2, "down_s_off_2": down_s_off_2, "down_b_off_2": down_b_off_2,
    }


def generate_header(cfg, model_name, output_path):
    """Generate model_config.h from parsed config."""
    hidden_dim = cfg.get("hidden_size", 4096)
    num_layers = cfg.get("num_hidden_layers", 60)
    num_attn_heads = cfg.get("num_attention_heads", 32)
    num_kv_heads = cfg.get("num_key_value_heads", 2)
    vocab_size = cfg.get("vocab_size", 248320)
    num_experts = cfg.get("num_experts", 512)
    num_experts_per_tok = cfg.get("num_experts_per_tok", 10)
    moe_intermediate = cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 1024))
    shared_intermediate = cfg.get("shared_expert_intermediate_size", cfg.get("intermediate_size", 1024))
    linear_num_v_heads = cfg.get("linear_num_value_heads", 64)
    linear_num_k_heads = cfg.get("linear_num_key_heads", 16)

    linear_total_key = linear_num_k_heads * LINEAR_KEY_DIM
    linear_total_value = linear_num_v_heads * LINEAR_VALUE_DIM
    linear_conv_dim = linear_total_key * 2 + linear_total_value
    rotary_dim = int(HEAD_DIM * PARTIAL_ROTARY)
    num_full_attn_layers = num_layers // FULL_ATTN_INTERVAL
    num_linear_layers = num_layers - num_full_attn_layers

    layout = compute_expert_layout(hidden_dim, moe_intermediate)

    lines = []
    lines.append("// Auto-generated by gen_model_config.py")
    lines.append(f"// Model: {model_name}")
    lines.append(f"// Generated from config.json in: (source model directory)")
    lines.append("")
    lines.append("#ifndef MODEL_CONFIG_H")
    lines.append("#define MODEL_CONFIG_H")
    lines.append("")
    lines.append("// ============================================================================")
    lines.append("// Model dimensions")
    lines.append("// ============================================================================")
    lines.append("")
    lines.append(f"#define HIDDEN_DIM          {hidden_dim}")
    lines.append(f"#define NUM_LAYERS          {num_layers}")
    lines.append(f"#define NUM_ATTN_HEADS      {num_attn_heads}")
    lines.append(f"#define NUM_KV_HEADS        {num_kv_heads}")
    lines.append(f"#define VOCAB_SIZE          {vocab_size}")
    lines.append(f"#define NUM_EXPERTS         {num_experts}")
    lines.append(f"#define NUM_EXPERTS_PER_TOK {num_experts_per_tok}")
    lines.append(f"#define MOE_INTERMEDIATE    {moe_intermediate}")
    lines.append(f"#define SHARED_INTERMEDIATE {shared_intermediate}")
    lines.append("")
    lines.append("// ============================================================================")
    lines.append("// Compile-time constants (same across all Qwen3.5 MoE models)")
    lines.append("// ============================================================================")
    lines.append("")
    lines.append(f"#define HEAD_DIM            {HEAD_DIM}")
    lines.append(f"#define GROUP_SIZE          {GROUP_SIZE}")
    lines.append(f"#define BITS                {BITS}")
    lines.append(f"#define RMS_NORM_EPS        {RMS_NORM_EPS}f")
    lines.append(f"#define ROPE_THETA          {ROPE_THETA}f")
    lines.append(f"#define PARTIAL_ROTARY      {PARTIAL_ROTARY}f")
    lines.append(f"#define ROTARY_DIM          {rotary_dim}")
    lines.append(f"#define FULL_ATTN_INTERVAL  {FULL_ATTN_INTERVAL}")
    lines.append(f"#define CONV_KERNEL_SIZE    {CONV_KERNEL_SIZE}")
    lines.append(f"#define MAX_SEQ_LEN         {MAX_SEQ_LEN}")
    lines.append(f"#define GPU_KV_SEQ          {GPU_KV_SEQ}")
    lines.append(f"#define MAX_K               {MAX_K}")
    lines.append("")
    lines.append("// ============================================================================")
    lines.append("// Linear attention (GatedDeltaNet)")
    lines.append("// ============================================================================")
    lines.append("")
    lines.append(f"#define LINEAR_KEY_DIM      {LINEAR_KEY_DIM}")
    lines.append(f"#define LINEAR_VALUE_DIM    {LINEAR_VALUE_DIM}")
    lines.append(f"#define LINEAR_NUM_V_HEADS  {linear_num_v_heads}")
    lines.append(f"#define LINEAR_NUM_K_HEADS  {linear_num_k_heads}")
    lines.append(f"#define LINEAR_TOTAL_KEY    {linear_total_key}")
    lines.append(f"#define LINEAR_TOTAL_VALUE  {linear_total_value}")
    lines.append(f"#define LINEAR_CONV_DIM     {linear_conv_dim}")
    lines.append(f"#define NUM_FULL_ATTN_LAYERS {num_full_attn_layers}")
    lines.append(f"#define NUM_LINEAR_LAYERS   {num_linear_layers}")
    lines.append("")
    lines.append("// ============================================================================")
    lines.append("// 4-bit expert layout")
    lines.append("// ============================================================================")
    lines.append("")
    lines.append(f"#define EXPERT_SIZE         {layout['expert_size']}")
    lines.append(f"#define GATE_W_SIZE         {layout['gate_w_size']}")
    lines.append(f"#define GATE_S_SIZE         {layout['gate_s_size']}")
    lines.append(f"#define GATE_B_SIZE         {layout['gate_b_size']}")
    lines.append(f"#define GATE_W_OFF          {layout['gate_w_off']}")
    lines.append(f"#define GATE_S_OFF          {layout['gate_s_off']}")
    lines.append(f"#define GATE_B_OFF          {layout['gate_b_off']}")
    lines.append(f"#define UP_W_SIZE           {layout['up_w_size']}")
    lines.append(f"#define UP_S_SIZE           {layout['up_s_size']}")
    lines.append(f"#define UP_B_SIZE           {layout['up_b_size']}")
    lines.append(f"#define UP_W_OFF            {layout['up_w_off']}")
    lines.append(f"#define UP_S_OFF            {layout['up_s_off']}")
    lines.append(f"#define UP_B_OFF            {layout['up_b_off']}")
    lines.append(f"#define DOWN_W_SIZE         {layout['down_w_size']}")
    lines.append(f"#define DOWN_S_SIZE         {layout['down_s_size']}")
    lines.append(f"#define DOWN_B_SIZE         {layout['down_b_size']}")
    lines.append(f"#define DOWN_W_OFF          {layout['down_w_off']}")
    lines.append(f"#define DOWN_S_OFF          {layout['down_s_off']}")
    lines.append(f"#define DOWN_B_OFF          {layout['down_b_off']}")
    lines.append("")
    lines.append("// ============================================================================")
    lines.append("// 2-bit expert layout")
    lines.append("// ============================================================================")
    lines.append("")
    lines.append(f"#define EXPERT_SIZE_2BIT    {layout['expert_size_2bit']}")
    lines.append(f"#define GATE_W_SIZE_2       {layout['gate_w_size_2']}")
    lines.append(f"#define GATE_S_SIZE_2       {layout['gate_s_size_2']}")
    lines.append(f"#define GATE_B_SIZE_2       {layout['gate_b_size_2']}")
    lines.append(f"#define GATE_W_OFF_2        {layout['gate_w_off_2']}")
    lines.append(f"#define GATE_S_OFF_2        {layout['gate_s_off_2']}")
    lines.append(f"#define GATE_B_OFF_2        {layout['gate_b_off_2']}")
    lines.append(f"#define UP_W_SIZE_2         {layout['up_w_size_2']}")
    lines.append(f"#define UP_S_SIZE_2         {layout['up_s_size_2']}")
    lines.append(f"#define UP_B_SIZE_2         {layout['up_b_size_2']}")
    lines.append(f"#define UP_W_OFF_2          {layout['up_w_off_2']}")
    lines.append(f"#define UP_S_OFF_2          {layout['up_s_off_2']}")
    lines.append(f"#define UP_B_OFF_2          {layout['up_b_off_2']}")
    lines.append(f"#define DOWN_W_SIZE_2       {layout['down_w_size_2']}")
    lines.append(f"#define DOWN_S_SIZE_2       {layout['down_s_size_2']}")
    lines.append(f"#define DOWN_B_SIZE_2       {layout['down_b_size_2']}")
    lines.append(f"#define DOWN_W_OFF_2        {layout['down_w_off_2']}")
    lines.append(f"#define DOWN_S_OFF_2        {layout['down_s_off_2']}")
    lines.append(f"#define DOWN_B_OFF_2        {layout['down_b_off_2']}")
    lines.append("")
    lines.append("// ============================================================================")
    lines.append("// Special tokens")
    lines.append("// ============================================================================")
    lines.append("")
    lines.append(f"#define EOS_TOKEN_1         {EOS_TOKEN_1}")
    lines.append(f"#define EOS_TOKEN_2         {EOS_TOKEN_2}")
    lines.append(f"#define THINK_START_TOKEN   {THINK_START_TOKEN}")
    lines.append(f"#define THINK_END_TOKEN     {THINK_END_TOKEN}")
    lines.append("")
    lines.append("#endif // MODEL_CONFIG_H")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Generated {output_path}:")
    print(f"  hidden_dim={hidden_dim}, layers={num_layers} ({num_full_attn_layers} full + {num_linear_layers} linear)")
    print(f"  experts={num_experts}, K={num_experts_per_tok}, moe_inter={moe_intermediate}")
    print(f"  attn_heads={num_attn_heads}, kv_heads={num_kv_heads}")
    print(f"  linear: v_heads={linear_num_v_heads}, k_heads={linear_num_k_heads}")
    print(f"  expert_size={layout['expert_size']:,} bytes (4-bit), {layout['expert_size_2bit']:,} bytes (2-bit)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate model_config.h from a HuggingFace model's config.json")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to HuggingFace model directory (containing config.json)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output header file (default: model_config.h in current dir)")
    parser.add_argument("--name", type=str, default=None,
                        help="Model name for header comment (default: derived from path)")
    args = parser.parse_args()

    model_path = Path(args.model)
    cfg = load_hf_config(str(model_path))

    model_name = args.name or model_path.name
    output = args.output or "src/model_config.h"

    generate_header(cfg, model_name, output)


if __name__ == "__main__":
    main()

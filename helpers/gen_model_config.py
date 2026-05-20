#!/usr/bin/env python3
"""
gen_model_config.py — Generate model_config.json from a HuggingFace config.json.

Computes all model dimensions and derived values (expert sizes, offsets,
linear attention dims, layer counts). Output written to the model directory.

Usage:
    python gen_model_config.py --model hub/models--mlx-community--Qwen3.6-35B-A3B-4bit
    python gen_model_config.py --model hub/models--Qwen3.6-35B-A3B-4bit --output output/my-model
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Architectural constants
GROUP_SIZE = 64
HEAD_DIM = 256
FULL_ATTN_INTERVAL = 4
LINEAR_KEY_DIM = 128
LINEAR_VALUE_DIM = 128
PARTIAL_ROTARY = 0.25


def load_hf_config(model_path):
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)
    if "text_config" in cfg:
        cfg = cfg["text_config"]
    return cfg


def compute_expert_layout(hidden_dim, moe_intermediate):
    """Compute 4-bit and 2-bit expert component sizes and offsets."""
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
    expert_size_4bit = down_b_off + down_sb

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
        "expert_size_4bit": expert_size_4bit,
        "expert_size_2bit": expert_size_2bit,
        "layout_4bit": {
            "gate_w_off": gate_w_off, "gate_s_off": gate_s_off, "gate_b_off": gate_b_off,
            "up_w_off": up_w_off, "up_s_off": up_s_off, "up_b_off": up_b_off,
            "down_w_off": down_w_off, "down_s_off": down_s_off, "down_b_off": down_b_off,
            "gate_w_size": gate_w, "gate_s_size": gate_sb, "gate_b_size": gate_sb,
            "up_w_size": up_w, "up_s_size": up_sb, "up_b_size": up_sb,
            "down_w_size": down_w, "down_s_size": down_sb, "down_b_size": down_sb,
        },
        "layout_2bit": {
            "gate_w_off": gate_w_off_2, "gate_s_off": gate_s_off_2, "gate_b_off": gate_b_off_2,
            "up_w_off": up_w_off_2, "up_s_off": up_s_off_2, "up_b_off": up_b_off_2,
            "down_w_off": down_w_off_2, "down_s_off": down_s_off_2, "down_b_off": down_b_off_2,
            "gate_w_size": gate_w2, "gate_s_size": gate_sb, "gate_b_size": gate_sb,
            "up_w_size": up_w2, "up_s_size": up_sb, "up_b_size": up_sb,
            "down_w_size": down_w2, "down_s_size": down_sb, "down_b_size": down_sb,
        },
    }


def generate_json(cfg, output_dir):
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

    head_dim = cfg.get("head_dim", HEAD_DIM)
    rp = cfg.get("rope_parameters", {})
    rope_theta = rp.get("rope_theta", 10000.0)
    partial_rotary = rp.get("partial_rotary_factor", PARTIAL_ROTARY)
    rotary_dim = int(head_dim * partial_rotary)
    linear_total_key = linear_num_k_heads * LINEAR_KEY_DIM
    linear_total_value = linear_num_v_heads * LINEAR_VALUE_DIM
    linear_conv_dim = linear_total_key * 2 + linear_total_value
    num_full_attn_layers = num_layers // FULL_ATTN_INTERVAL
    num_linear_layers = num_layers - num_full_attn_layers

    layout = compute_expert_layout(hidden_dim, moe_intermediate)

    config = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_attn_heads": num_attn_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "moe_intermediate": moe_intermediate,
        "shared_intermediate": shared_intermediate,
        "linear_num_v_heads": linear_num_v_heads,
        "linear_num_k_heads": linear_num_k_heads,
        "rotary_dim": rotary_dim,
        "rope_theta": rope_theta,
        "linear_total_key": linear_total_key,
        "linear_total_value": linear_total_value,
        "linear_conv_dim": linear_conv_dim,
        "num_full_attn_layers": num_full_attn_layers,
        "num_linear_layers": num_linear_layers,
        "expert_size_4bit": layout["expert_size_4bit"],
        "expert_size_2bit": layout["expert_size_2bit"],
        "expert_layout_4bit": layout["layout_4bit"],
        "expert_layout_2bit": layout["layout_2bit"],
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model_config.json")
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate model_config.json from a HuggingFace config.json")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to HuggingFace model directory (containing config.json)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: same as --model)")
    args = parser.parse_args()

    model_path = Path(args.model)
    cfg = load_hf_config(str(model_path))
    output_dir = args.output or str(model_path)
    generate_json(cfg, output_dir)


if __name__ == "__main__":
    main()

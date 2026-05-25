#!/usr/bin/env python3
"""
quantize.py — Quantize HF BF16 Qwen3.5/3.6 MoE model using BQ4.

Quantization is performed entirely in Rust via ``moe_infer.quantize()``.
The Python side only handles argument parsing and path resolution.

Usage:
    python quant/quantize.py \
        --model hub/models--Qwen--Qwen3.6-35B-A3B \
        --output data/my-model
"""

import argparse
import os
import sys

import moe_infer


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
    parser.add_argument('--qwen36', action='store_true',
                        help='Apply +1.0 shift to norm weights (Qwen3.6 → 3.5 convention)')
    args = parser.parse_args()

    # Resolve name_mapping.json relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_path = os.path.join(script_dir, "name_mapping.json")
    if not os.path.exists(mapping_path):
        print(f"ERROR: {mapping_path} not found", file=sys.stderr)
        sys.exit(1)

    strip_layers = 4 if args.strip else 0
    strip_experts = 4 if args.strip else 0

    moe_infer.quantize(
        args.model,
        args.output,
        mapping_path,
        qwen36=args.qwen36,
        strip_layers=strip_layers,
        strip_experts=strip_experts,
    )


if __name__ == '__main__':
    main()

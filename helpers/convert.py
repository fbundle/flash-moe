#!/usr/bin/env python3
"""
convert.py — Convert a HuggingFace Qwen3 MoE model to Flash-MoE format.

Thin orchestrator that delegates to existing helper modules:
  1. tokenizer.bin → export_tokenizer.py
  2. model_config.json → gen_model_config.py
  3. model_weights.bin + .json → extract_weights.py
  4. packed_experts/ → repack_experts_4bit.py

Usage:
    python helpers/convert.py --model hub/models--mlx-community--Qwen3.5-35B-A3B-4bit --output data

    Or step-by-step:
    python helpers/convert.py --model ... --step tokenizer
    python helpers/convert.py --model ... --step config
    python helpers/convert.py --model ... --step weights
    python helpers/convert.py --model ... --step experts
"""

import argparse
import os
import sys
import time
from pathlib import Path

import helpers.export_tokenizer as export_tokenizer
import helpers.extract_weights as extract_weights
import helpers.gen_model_config as gen_model_config
import helpers.repack_experts_4bit as repack_experts_4bit


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert HF Qwen3 MoE model to Flash-MoE format"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: <model>/../flash-moe-data)",
    )
    parser.add_argument(
        "--step", type=str, default=None,
        choices=["tokenizer", "config", "weights", "experts"],
        help="Run a single step (default: all)",
    )
    args = parser.parse_args()

    model_dir = str(Path(args.model).resolve())
    output_dir = os.path.join(args.output or "data", Path(model_dir).name)
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Flash-MoE Converter")
    print(f"  Model:  {model_dir}")
    print(f"  Output: {output_dir}")
    print()

    steps = ["tokenizer", "config", "weights", "experts"]
    if args.step:
        steps = [args.step]

    t0 = time.time()

    for i, step in enumerate(steps):
        print(f"{'=' * 50}")
        print(f"Step {i + 1}/{len(steps)}: {step}")
        print(f"{'=' * 50}")

        if step == "tokenizer":
            import shutil
            hf_tok = os.path.join(model_dir, "tokenizer.json")
            shutil.copy2(hf_tok, os.path.join(output_dir, "tokenizer.json"))
            print(f"  Copied {hf_tok} → {output_dir}/tokenizer.json")

        elif step == "config":
            import shutil
            hf_config = os.path.join(model_dir, "config.json")
            shutil.copy2(hf_config, os.path.join(output_dir, "config.json"))
            print(f"  Copied {hf_config} → {output_dir}/config.json")

        elif step == "weights":
            extract_weights.run(model_dir, output_dir, include_experts=False)

        elif step == "experts":
            packed_dir = os.path.join(output_dir, "packed_experts")
            repack_experts_4bit.run(model_dir, packed_dir)

        print()

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Model ready in: {output_dir}/")
    print()
    print("Next steps:")
    print(f"  cd moe_infer_rs")
    print(f"  cargo run --release -- --serve 8000 --model {output_dir}")


if __name__ == "__main__":
    main()

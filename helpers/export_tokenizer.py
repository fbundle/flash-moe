#!/usr/bin/env python3
"""Export HuggingFace tokenizer.json to a compact binary format.

Can be used standalone or imported via run(model_dir, output_dir).

Usage:
    python export_tokenizer.py [tokenizer.json] [output.bin]
    python export_tokenizer.py --model <dir> --output <dir>

Binary format:
  Header:
    magic: "BPET" (4 bytes)
    version: uint32 (1)
    vocab_size: uint32
    num_merges: uint32
    num_added: uint32
  Vocab section (sorted by token_id):
    For each entry: uint32 token_id, uint16 str_len, char[str_len]
  Merges section (ordered by priority, index 0 = highest priority):
    For each entry: uint16 len_a, char[len_a], uint16 len_b, char[len_b]
  Added tokens section:
    For each entry: uint32 token_id, uint16 str_len, char[str_len]
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path


def run(model_dir: str, output_dir: str):
    """Export tokenizer.json → tokenizer.bin in output_dir."""
    tok_path = Path(model_dir) / "tokenizer.json"
    if not tok_path.exists():
        print(f"ERROR: {tok_path} not found", file=sys.stderr)
        sys.exit(1)

    out_path = Path(output_dir) / "tokenizer.bin"

    with open(tok_path, "r", encoding="utf-8") as f:
        t = json.load(f)

    model = t["model"]
    vocab = model["vocab"]
    merges = model["merges"]
    added = t["added_tokens"]

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    with open(out_path, "wb") as f:
        f.write(b"BPET")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", len(sorted_vocab)))
        f.write(struct.pack("<I", len(merges)))
        f.write(struct.pack("<I", len(added)))

        for token_str, token_id in sorted_vocab:
            b = token_str.encode("utf-8")
            f.write(struct.pack("<I", token_id))
            f.write(struct.pack("<H", len(b)))
            f.write(b)

        for pair in merges:
            a, b = pair[0], pair[1]
            ab = a.encode("utf-8")
            bb = b.encode("utf-8")
            f.write(struct.pack("<H", len(ab)))
            f.write(ab)
            f.write(struct.pack("<H", len(bb)))
            f.write(bb)

        for tok in added:
            b = tok["content"].encode("utf-8")
            f.write(struct.pack("<I", tok["id"]))
            f.write(struct.pack("<H", len(b)))
            f.write(b)

    sz = os.path.getsize(out_path)
    print(f"  tokenizer.bin: {len(sorted_vocab)} vocab, {len(merges)} merges ({sz / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace tokenizer.json to compact binary format"
    )
    parser.add_argument(
        "tok_path", nargs="?", default=None,
        help="Path to tokenizer.json (positional, legacy)",
    )
    parser.add_argument(
        "out_path", nargs="?", default=None,
        help="Output path for tokenizer.bin (positional, legacy)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model directory containing tokenizer.json",
    )
    parser.add_argument(
        "--output", type=str, default=".",
        help="Output directory for tokenizer.bin",
    )
    args = parser.parse_args()

    if args.model:
        run(args.model, args.output)
    elif args.tok_path:
        tok_dir = str(Path(args.tok_path).parent)
        out_path = args.out_path or "tokenizer.bin"
        out_dir = str(Path(out_path).parent)
        out_name = Path(out_path).name
        # For legacy interface, just run with the parent directories
        import shutil
        run_temp = str(Path(out_path))
        run(tok_dir, out_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

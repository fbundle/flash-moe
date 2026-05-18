#!/usr/bin/env python3
"""
export_tokenizer.py — Extract vocab.bin and tokenizer.bin from HuggingFace model.

Reads vocab.json and tokenizer.json, writes binary files the C engine expects.

vocab.bin (for decode_token in infer.m):
  uint32 num_entries, uint32 max_id
  For each entry (sorted by id): uint16 byte_len, char[byte_len]

tokenizer.bin (for bpe_load in tokenizer.h):
  Magic "BPET", uint32 version=1
  uint32 vocab_size, uint32 num_merges, uint32 num_added
  Vocab (sorted by id): uint32 id, uint16 len, char[len]
  Merges: uint16 len_a, char[len_a], uint16 len_b, char[len_b]
  Added tokens: uint32 id, uint16 len, char[len]

Usage:
    python helpers/export_tokenizer.py --model hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
"""

import argparse
import json
import struct
from pathlib import Path


def write_vocab_bin(model_path, out_dir):
    """Generate vocab.bin from vocab.json (token_str → token_id)."""
    vocab_path = Path(model_path) / "vocab.json"
    if not vocab_path.exists():
        print(f"ERROR: {vocab_path} not found")
        return

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    max_id = max(v for _, v in sorted_vocab)

    out_path = Path(out_dir) / "vocab.bin"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", len(sorted_vocab)))
        f.write(struct.pack("<I", max_id))
        for token_str, token_id in sorted_vocab:
            b = token_str.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)

    size_mb = out_path.stat().st_size / 1e6
    print(f"vocab.bin: {len(sorted_vocab)} tokens, {size_mb:.1f} MB")


def write_tokenizer_bin(model_path, out_dir):
    """Generate tokenizer.bin from tokenizer.json."""
    tok_path = Path(model_path) / "tokenizer.json"
    if not tok_path.exists():
        print(f"ERROR: {tok_path} not found")
        return

    with open(tok_path, "r", encoding="utf-8") as f:
        t = json.load(f)

    model = t["model"]
    vocab = model["vocab"]
    merges = model["merges"]
    added = t.get("added_tokens", [])

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    out_path = Path(out_dir) / "tokenizer.bin"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"BPET")
        f.write(struct.pack("<I", 1))  # version
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

    size_mb = out_path.stat().st_size / 1e6
    print(f"tokenizer.bin: {len(sorted_vocab)} vocab, {len(merges)} merges, "
          f"{len(added)} added, {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export vocab.bin and tokenizer.bin from HuggingFace model"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory (containing vocab.json, tokenizer.json)")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory [default: data]")
    args = parser.parse_args()

    write_vocab_bin(args.model, args.output)
    write_tokenizer_bin(args.model, args.output)


if __name__ == "__main__":
    main()

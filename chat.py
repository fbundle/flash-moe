#!/usr/bin/env python3
"""Simple interactive chat using Flash-MoE with the FusedExp pipeline.

Usage:
    python chat.py

Requires the model at data/models--mlx-community--Qwen3.6-35B-A3B-4bit/
(relative to this script's directory).
"""

import os
import sys
import numpy as np
from tokenizers import Tokenizer
from moe_infer import Context, Cache

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "data", "models--mlx-community--Qwen3.6-35B-A3B-4bit")
EOS_IDS = [248046, 248044]  # <|im_end|>, <|endoftext|>

# Sampling parameters with "no effect" defaults (matching Rust sample() logic):
#   top_k=0    -> disabled (only active when top_k > 0 && top_k < n)
#   top_p=1.0  -> disabled (only active when top_p < 1.0)
#   min_p=0.0  -> disabled (only active when min_p > 0.0)
TEMPERATURE = 0.6
TOP_K = 0
TOP_P = 1.0
MIN_P = 0.0
MAX_TOKENS = 512


def main():
    if not os.path.isdir(MODEL_DIR):
        print(f"Error: model directory not found at {MODEL_DIR}", file=sys.stderr)
        print("Make sure the model has been downloaded and converted.", file=sys.stderr)
        sys.exit(1)

    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
    if not os.path.isfile(tokenizer_path):
        print(f"Error: tokenizer.json not found at {tokenizer_path}", file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer.from_file(tokenizer_path)

    print("Loading model...", end=" ", flush=True)
    ctx = Context()
    ctx.load_model(MODEL_DIR, pipeline_mode="FusedExp")
    cache = ctx.new_cache()
    print("ready.\n")

    # Accumulated token IDs for the full conversation.
    # This grows across turns and is always passed as input_ids to
    # stream_generate. The cache skips already-processed positions, so only
    # new tokens incur a forward pass.
    history_token_ids: list[int] = []

    while True:
        try:
            prompt = input(">>> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt.lower() == "exit":
            break

        # Qwen chat-format: user message + assistant header.
        new_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        new_tokens = tokenizer.encode(new_text).ids

        full_ids = np.array(history_token_ids + new_tokens, dtype=np.int64)

        response_tokens: list[int] = []
        for token_id, _logits in ctx.stream_generate(
            full_ids,
            cache,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            min_p=MIN_P,
        ):
            if token_id in EOS_IDS:
                break
            text = tokenizer.decode([token_id], skip_special_tokens=True)
            print(text, end="", flush=True)
            response_tokens.append(token_id)

        print()
        history_token_ids = list(full_ids) + response_tokens


if __name__ == "__main__":
    main()

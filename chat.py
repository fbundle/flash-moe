#!/usr/bin/env python3
"""Interactive chat using moe-infer-rs (Rust backend) with streaming output.

Usage:
  uv run chat.py --model data --tokenizer Qwen/Qwen3-8B
"""

import argparse
import readline
import time

from transformers import AutoTokenizer

import moe_infer_rs as _core


def generate_rust(model, cache, prompt_ids, max_tokens, eos_token_id,
                   temperature, top_k, top_p, min_p):
    """Run prefill then token-by-token autoregressive generation.
    Yields each token_id immediately after it's sampled — true streaming."""
    pos = cache.position
    new_ids = prompt_ids[pos:]

    if new_ids:
        logits, _ = model.forward(new_ids, cache)
    else:
        logits, _ = model.forward([prompt_ids[-1]], cache)

    # Get the last vocab-sized slice and find argmax for the first token
    vocab = model.vocab_size
    last_logits = logits[-vocab:] if len(logits) >= vocab else logits
    next_id = max(range(len(last_logits)), key=lambda i: last_logits[i])

    # Loop in Python — each call to model.sample does one forward + sample
    # and returns immediately, so we yield tokens as they're produced
    for _ in range(max_tokens):
        next_id = model.sample(next_id, cache, temperature, top_k, top_p, min_p)
        if next_id == eos_token_id:
            break
        yield next_id


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with Flash-MoE (Rust)")
    parser.add_argument("--model", "-m", default="data")
    parser.add_argument("--tokenizer", "-t", default=None)
    parser.add_argument("--max-tokens", "-n", type=int, default=512)
    parser.add_argument("--temperature", "-T", type=float, default=0.6)
    parser.add_argument("--top-k", "-k", type=int, default=0)
    parser.add_argument("--top-p", "-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    args = parser.parse_args()

    # Default to the Qwen3.6 tokenizer bundled with the converted model
    tok_path = args.tokenizer or "hub/models--mlx-community--Qwen3.6-35B-A3B-4bit"
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    print(f"Loading model from {args.model} ...")
    model = _core.Model(args.model)
    messages = []

    print(f"Ready. {model.num_layers} layers, "
          f"{model.hidden_dim} dim.\n")

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        result = tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = [int(t) for t in result["input_ids"]]

        cache = _core.Cache(model)
        t0 = time.monotonic()
        ttft = 0.0
        response_ids = []

        for token_id in generate_rust(model, cache, prompt_ids,
                                       args.max_tokens, tok.eos_token_id,
                                       args.temperature, args.top_k,
                                       args.top_p, args.min_p):
            if not response_ids:
                ttft = time.monotonic() - t0
            response_ids.append(token_id)
            print(tok.decode([token_id]), end="", flush=True)

        elapsed = time.monotonic() - t0
        n_tok = len(response_ids)
        tok_s = n_tok / elapsed if elapsed > 0 else 0

        response_text = tok.decode(response_ids, skip_special_tokens=False)
        messages.append({"role": "assistant", "content": response_text})

        print(f"\n[{n_tok} tokens, {tok_s:.1f} tok/s, "
              f"TTFT {ttft:.2f}s]\n")


if __name__ == "__main__":
    main()

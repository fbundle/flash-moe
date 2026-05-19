#!/usr/bin/env python3
"""Interactive chat using moe-infer-mlx with greedy decoding.

Usage:
  uv run chat.py --model data --tokenizer Qwen/Qwen3-8B
"""

import argparse
import readline
import time

from transformers import AutoTokenizer

import moe_infer_mlx as fm


def generate_c(model, cache, prompt_ids, max_tokens, eos_token_id, temperature):
    """Run prefill then C-side autoregressive generation. Yields token_ids one at a time."""
    pos = cache.position
    new_ids = prompt_ids[pos:]
    if new_ids:
        logits, cache = model.forward(new_ids, cache)
    else:
        logits, cache = model.forward([prompt_ids[-1]], cache)

    first_id = int(logits[-1].argmax())
    token_ids, cache = model.generate(
        first_id, cache, eos_token_id,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    yield from token_ids


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with Flash-MoE")
    parser.add_argument("--model", "-m", default="data")
    parser.add_argument("--tokenizer", "-t", default=None)
    parser.add_argument("--max-tokens", "-n", type=int, default=512)
    parser.add_argument("--temperature", "-T", type=float, default=0.6)
    args = parser.parse_args()

    tok_path = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    model = fm.Model(args.model)
    messages = []

    with model:
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

            cache = fm.Cache(model)
            t0 = time.monotonic()
            response_ids = []

            for token_id in generate_c(model, cache, prompt_ids,
                                       args.max_tokens, tok.eos_token_id,
                                       args.temperature):
                response_ids.append(token_id)
                print(tok.decode([token_id]), end="", flush=True)

            elapsed = time.monotonic() - t0
            n_tok = len(response_ids)
            tok_s = n_tok / elapsed if elapsed > 0 else 0

            response_text = tok.decode(response_ids, skip_special_tokens=False)
            messages.append({"role": "assistant", "content": response_text})

            print(f"\n[{n_tok} tokens, {tok_s:.1f} tok/s, "
                  f"TTFT {t0:.2f}s]\n")


if __name__ == "__main__":
    main()

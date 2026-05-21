#!/usr/bin/env python3
"""Example: Flash-MoE inference via Rust PyO3 bindings.

Requires:
  pip install maturin transformers numpy torch
  cd moe_infer_rs && maturin develop --release --features python-bindings
"""
import time
import numpy as np
from transformers import AutoTokenizer
from moe_infer import Context, Cache

MODEL_PATH = "/Volumes/Hippopotamus/vault/code/flash-moe/data/models--mlx-community--Qwen3.5-35B-A3B-4bit"
HF_MODEL_PATH = "/Volumes/Hippopotamus/vault/code/flash-moe/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit"
PROMPT = "Explain quantum computing in one sentence."
MAX_TOKENS = 100
EOS_IDS = [248046, 248044]


def print_telemetry(t, label=""):
    """Print performance metrics from a telemetry dict."""
    prefix = f"[{label}] " if label else ""
    parts = [f"ttft: {t['ttft_ms']:.0f} ms"]
    if t['tokens_generated'] > 0:
        parts.append(f"total: {t['total_ms']:.0f} ms")
        parts.append(f"tokens: {t['tokens_generated']}")
        parts.append(f"speed: {t['tokens_per_sec']:.1f} tok/s")
    print(f"[telemetry] {prefix}" + ", ".join(parts))


def main():
    # Load tokenizer from HF hub (has tokenizer_config.json with chat template)
    print(f"[example] Loading tokenizer from {HF_MODEL_PATH}")
    tok = AutoTokenizer.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)

    # Format prompt with chat template
    messages = [{"role": "user", "content": PROMPT}]
    prompt_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(prompt_str, add_special_tokens=False)
    print(f"[example] Prompt: {len(input_ids)} tokens")

    # Load model via Rust binding
    print(f"[example] Loading model...")
    t0 = time.time()
    ctx = Context()
    ctx.load_model(MODEL_PATH, pipeline_mode="FusedExp")
    cache = ctx.new_cache()
    print(f"[example] Model loaded in {(time.time() - t0) * 1000:.0f} ms")

    # Forward pass (prefill): get logits for all positions
    print(f"[example] Forward (prefill)...")
    ids_arr = np.array(input_ids, dtype=np.int64)
    logits = ctx.forward(ids_arr, cache)
    print_telemetry(ctx.telemetry(), "prefill")

    # First token
    first_token = int(np.argmax(logits[-1]))
    print(f"[example] First token: {first_token} ('{tok.decode([first_token])}')")

    # Generate
    print(f"[example] Generating up to {MAX_TOKENS} tokens...")
    new_ids = ctx.generate(ids_arr, cache,
                           max_tokens=MAX_TOKENS,
                           temperature=0.0,  # greedy
                           top_k=1,
                           eos_token_ids=np.array(EOS_IDS, dtype=np.int64))
    print_telemetry(ctx.telemetry(), "generate")

    # Decode
    output = tok.decode(new_ids.tolist())
    print(f"\n[example] Output:\n{output}")

    # Multi-turn: add user follow-up
    follow_up = "Can you explain it differently?"
    follow_msg = messages + [
        {"role": "assistant", "content": output},
        {"role": "user", "content": follow_up},
    ]
    follow_str = tok.apply_chat_template(follow_msg, tokenize=False, add_generation_prompt=True)
    follow_ids = tok.encode(follow_str, add_special_tokens=False)

    print(f"\n[example] Multi-turn: {len(follow_ids)} total tokens (cache.pos={cache.pos})")
    # Only new tokens will be processed (cache already has previous tokens)
    follow_arr = np.array(follow_ids, dtype=np.int64)
    new_reply = ctx.generate(follow_arr, cache,
                             max_tokens=MAX_TOKENS,
                             temperature=0.7,  # creative
                             top_k=40, top_p=0.9,
                             eos_token_ids=np.array(EOS_IDS, dtype=np.int64))
    print_telemetry(ctx.telemetry(), "multi-turn")
    print(f"[example] Follow-up reply: {tok.decode(new_reply.tolist())}")

    # Cleanup
    ctx.unload_model()
    print("[example] Done.")


if __name__ == "__main__":
    main()

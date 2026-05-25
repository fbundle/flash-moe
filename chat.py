#!/usr/bin/env python3
"""Interactive chat demo for MoE-Infer."""

import argparse
import time

import numpy as np
from transformers import AutoTokenizer
from moe_infer import Model, Engine, Cache, record_engine_telemetry  # type: ignore


# Maps model family names to response-extraction functions. Each function
# receives the raw decoded completion string and returns the final response.
# Add entries here when supporting new model families.
_RESPONSE_EXTRACTORS = {
    "qwen3": lambda c: c.removesuffix("<|im_end|>").split("</think>")[-1],
    "raw": lambda c: c,
}


def _get_response_extractor(model_name: str):
    """Pick the right response stripper based on a lowercase model-name hint."""
    name = model_name.lower()
    if "qwen" in name:
        return _RESPONSE_EXTRACTORS["qwen3"]
    return _RESPONSE_EXTRACTORS["raw"]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _sample(logits: np.ndarray, temperature: float,
            top_k: int, top_p: float, min_p: float) -> int:
    """Sample a token from logits. Modifies logits in-place."""
    n = len(logits)
    if abs(temperature - 1.0) > 1e-7:
        logits /= max(temperature, 1e-8)
    if temperature < 0.01:
        return int(np.argmax(logits))
    probs = _softmax(logits)

    if top_k > 0 and top_k < n:
        indices = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.ones(n, dtype=bool)
        mask[indices] = False
        probs[mask] = 0.0
    if top_p < 1.0:
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff_idx = np.searchsorted(cumsum, top_p)
        if cutoff_idx < n:
            probs[sorted_idx[cutoff_idx + 1:]] = 0.0
    if min_p > 0.0:
        threshold = probs.max() * min_p
        probs[probs < threshold] = 0.0

    total = probs.sum()
    if total <= 0:
        return 0
    probs /= total
    return int(np.random.choice(n, p=probs))


class Conversation:
    def __init__(self, tokenizer_path: str, model_path: str,
                 response_extractor, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = Model(model_path)
        self.engine = Engine(self.model, **kwargs)
        self.cache = Cache(self.model)
        self.extract_response = response_extractor
        self.messages: list[dict] = []
        self._last_chat_tlm: dict = {}

    @property
    def telemetry(self) -> dict:
        """Return combined chat-level + engine-level telemetry for the last chat."""
        return {**self._last_chat_tlm, "engine": self.engine.telemetry()}

    def chat(self, message: str,
             max_tokens: int = 256, temperature: float = 0.0,
             top_k: int = 0, top_p: float = 1.0, min_p: float = 0.0,
             eos_token_ids: list[int] | None = None) -> str:
        if eos_token_ids is None:
            eos_token_ids = [248046, 248044]
        self.messages.append({"role": "user", "content": message})

        input_ids = np.array(
            self.tokenizer.apply_chat_template(
                self.messages, add_generation_prompt=True,
                enable_thinking=False,
            ).input_ids,
            dtype=np.int64,
        )[self.cache.pos:]

        t0 = time.time()
        logits = self.engine.forward(input_ids, self.cache)
        prefill_ms = (time.time() - t0) * 1000.0

        last_logits = np.asarray(logits[-1])
        completion_ids: list[int] = []

        for _ in range(max_tokens):
            token = _sample(last_logits, temperature, top_k, top_p, min_p)
            if token in eos_token_ids:
                break
            completion_ids.append(token)
            print(self.tokenizer.decode([token]), end="", flush=True)
            logits = self.engine.forward(
                np.array([token], dtype=np.int64), self.cache)
            last_logits = np.asarray(logits[0])

        print()
        total_ms = (time.time() - t0) * 1000.0
        n_tokens = len(completion_ids) + 1
        gen_ms = total_ms - prefill_ms
        tps = (n_tokens - 1) / (gen_ms / 1000.0) if gen_ms > 0 else 0.0

        response = self.extract_response(self.tokenizer.decode(completion_ids))
        self.messages.append({"role": "assistant", "content": response})

        self._last_chat_tlm = {
            "prefill_ms": prefill_ms,
            "total_ms": total_ms,
            "tokens_generated": n_tokens,
            "tokens_per_sec": tps,
        }
        return response


def main():
    default_tokenizer = "hub/models--Qwen--Qwen3.6-35B-A3B"
    default_model = "data/models--Qwen--Qwen3.6-35B-A3B-bq4"

    parser = argparse.ArgumentParser(description="MoE-Infer interactive chat")
    parser.add_argument(
        "--tokenizer", default=default_tokenizer,
        help=f"Path to HF tokenizer (default: {default_tokenizer})",
    )
    parser.add_argument(
        "--model", default=default_model,
        help=f"Path to MoE-Infer model directory (default: {default_model})",
    )
    parser.add_argument(
        "--mode", default="Qwen35MoEBq4Exp2",
        choices=["Cpu", "Qwen35MoEBq4Exp2"],
        help="Pipeline mode",
    )
    parser.add_argument(
        "--k", type=int, default=4,
        help="Active experts per token (0 = model default)",
    )
    parser.add_argument(
        "--telemetry", action="store_true",
        help="Enable per-layer engine timing telemetry",
    )
    args = parser.parse_args()

    record_engine_telemetry(args.k != 0 and args.telemetry)
    extractor = _get_response_extractor(args.model)
    k = 0 if args.k == 0 else args.k
    conv = Conversation(args.tokenizer, args.model, extractor,
                        pipeline_mode=args.mode, k=k)

    while True:
        try:
            message = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        conv.chat(message)
        print()
        print(conv.telemetry)


if __name__ == "__main__":
    main()

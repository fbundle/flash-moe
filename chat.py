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


from moe_infer.generate import generate_from


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

        completion_text, gen_stats = generate_from(
            logits[-1], self.engine, self.cache, self.tokenizer,
            max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, min_p=min_p,
            eos_ids=tuple(eos_token_ids),
            on_token=lambda tok: print(self.tokenizer.decode([tok]), end="", flush=True),
        )
        print()

        total_ms = (time.time() - t0) * 1000.0
        n_tokens = gen_stats["tokens"] + 1  # +1 for prefill

        response = self.extract_response(completion_text)
        self.messages.append({"role": "assistant", "content": response})

        self._last_chat_tlm = {
            "prefill_ms": prefill_ms,
            "total_ms": total_ms,
            "tokens_generated": n_tokens,
            "tokens_per_sec": gen_stats["tok_per_s"],
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
        choices=["Cpu", "Qwen35MoEBq4Exp1", "Qwen35MoEBq4Exp2"],
        help="Pipeline mode",
    )
    parser.add_argument(
        "--k", type=int, default=0,
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

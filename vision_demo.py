#!/usr/bin/env python3
"""Vision demo for Qwen3.6-35B-A3B. Uses transformers for vision encoder, Rust for LM."""

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer, AutoImageProcessor
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeVisionConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel

from moe_infer import Model, Engine, Cache # type: ignore

# ─── Constants ───────────────────────────────────────────────────────────────

HUB = "hub/models--Qwen--Qwen3.6-35B-A3B"
IMAGE_PAD = 248055


# ─── Vision encoder loader ───────────────────────────────────────────────────

def load_vision_encoder(hub: str) -> Qwen3_5MoeVisionModel:
    """Load Qwen3_5MoeVisionModel with only vision weights from safetensors."""
    print("[vision] Loading weights...", flush=True)
    t0 = time.time()

    with open(f"{hub}/model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]

    vis_shards = sorted(set(sn for k, sn in wm.items() if k.startswith("model.visual.")))

    state = {}
    for sn in vis_shards:
        for k, v in load_safetensors(f"{hub}/{sn}").items():
            if k.startswith("model.visual."):
                state[k.removeprefix("model.visual.")] = v

    cfg = Qwen3_5MoeVisionConfig.from_pretrained(hub)
    vis = Qwen3_5MoeVisionModel(cfg)
    vis.load_state_dict(state, strict=True)
    vis.eval()

    print(f"[vision] Loaded {len(state)} tensors in {time.time()-t0:.1f}s", flush=True)
    return vis


# ─── Demo ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen3.6-35B-A3B vision demo")
    parser.add_argument("--image", default="data/crycat-crying-cat.gif")
    parser.add_argument("--question", default="What is in this image?")
    parser.add_argument("--model", default="data/models--Qwen--Qwen3.6-35B-A3B-bq4")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--hub", default=HUB)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-pixels", type=int, default=0,
                        help="Lower bound on total image pixels (0 = default: 65536)")
    parser.add_argument("--max-pixels", type=int, default=0,
                        help="Upper bound on total image pixels (0 = default: 16777216)")
    args = parser.parse_args()

    hub = args.hub

    # 1. Preprocess image
    print(f"[demo] Loading image: {args.image}")
    proc = AutoImageProcessor.from_pretrained(hub)
    img = Image.open(args.image).convert("RGB")
    proc_kwargs = {"images": img, "return_tensors": "pt"}
    if args.min_pixels > 0 or args.max_pixels > 0:
        proc_kwargs["size"] = {
            "shortest_edge": args.min_pixels or 65536,
            "longest_edge": args.max_pixels or 16777216,
        }
    inputs = proc(**proc_kwargs)
    pixel_values = inputs["pixel_values"]                            # [N, 1536]
    grid_thw = inputs["image_grid_thw"]                              # [[1, H, W]]
    n_merged = int((grid_thw[0, 1] // 2) * (grid_thw[0, 2] // 2))
    print(f"[demo] Patches: {pixel_values.shape[0]}, merged: {n_merged}", flush=True)

    # 2. Vision encoder forward (transformers)
    vision = load_vision_encoder(hub)
    with torch.no_grad():
        out = vision(pixel_values, grid_thw)
    vis_feats = out.pooler_output.numpy().astype(np.float32)         # [N_merged, 2048]
    print(f"[vision] Output: {vis_feats.shape}", flush=True)

    # 3. Load language model
    print(f"[demo] Loading model: {args.model}")
    model = Model(args.model)
    engine = Engine(model, pipeline_mode="Qwen35MoEBq4Exp2", k=0)
    cache = Cache(model)
    tokenizer = AutoTokenizer.from_pretrained(hub)

    # 4. Build input with vision tokens: embed text parts, splice in vision features
    before = tokenizer.encode(f"<|im_start|>user\n<|vision_start|>", add_special_tokens=False)
    after  = tokenizer.encode(f"<|vision_end|>\n{args.question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)

    print(f"[demo] Building embeddings ({len(before)} + {n_merged} + {len(after)} tokens)...")
    embeds = np.concatenate([
        engine.embed_lookup(np.array(before, dtype=np.int64)),
        vis_feats,
        engine.embed_lookup(np.array(after, dtype=np.int64)),
    ])

    # 5. LM forward
    print(f"[demo] Running LM forward...")
    t0 = time.time()
    logits = engine.forward_hidden(embeds, cache)                    # [N, vocab]
    print(f"[demo] Prefill: {time.time()-t0:.1f}s", flush=True)

    # 6. Generate
    from moe_infer.generate import generate_from

    print(f"[demo] Generating (max {args.max_tokens} tokens)...")
    text, stats = generate_from(
        logits[-1], engine, cache, tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        on_token=lambda tok: print(tokenizer.decode([tok]), end="", flush=True),
    )
    n = stats["tokens"]
    if n > 0:
        print(f"\n\n[demo] {n} tokens in {stats['seconds']:.1f}s ({stats['tok_per_s']:.1f} tok/s)")
    else:
        print(f"\n\n[demo] 0 tokens generated")


if __name__ == "__main__":
    main()

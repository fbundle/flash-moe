#!/usr/bin/env python3
"""OpenAI-compatible completion API server for MoE-Infer."""

import argparse
import json
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from moe_infer import Model, Engine, Cache, record_engine_telemetry  # type: ignore

app = FastAPI(title="MoE-Infer Server")


# ─── Sampling ────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _sample(logits: np.ndarray, temperature: float,
            top_k: int, top_p: float, min_p: float) -> int:
    n = len(logits)
    if abs(temperature - 1.0) > 1e-7:
        logits = logits / max(temperature, 1e-8)
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
        cutoff = np.searchsorted(cumsum, top_p)
        if cutoff < n:
            probs[sorted_idx[cutoff + 1:]] = 0.0
    if min_p > 0.0:
        threshold = probs.max() * min_p
        probs[probs < threshold] = 0.0
    total = probs.sum()
    if total <= 0:
        return 0
    probs /= total
    return int(np.random.choice(n, p=probs))


# ─── Global state ────────────────────────────────────────────────────────────

_engine: Optional[Engine] = None
_model: Optional[Model] = None
_tokenizer = None


def load_engine(model_path: str, tokenizer_path: str, pipeline_mode: str = "Fused4bit", k: int = 0):
    global _model, _engine, _tokenizer
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _model = Model(model_path)
    _engine = Engine(_model, pipeline_mode=pipeline_mode, k=k)


# ─── Request/Response models ─────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str = "moe-infer"
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    stream: bool = False
    stop: Optional[list[str]] = None
    seed: Optional[int] = None


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: Optional[dict] = None
    finish_reason: str = "stop"


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if _engine is None:
        raise HTTPException(503, "Engine not loaded")

    if req.seed is not None:
        np.random.seed(req.seed)

    eos_ids = [248046, 248044]
    stop_strs = req.stop or []

    prompt_ids = _tokenizer.encode(req.prompt)
    cache = Cache(_model)

    input_ids = np.array(prompt_ids, dtype=np.int64)
    logits = _engine.forward(input_ids, cache)
    last_logits = np.asarray(logits[-1])

    completion_ids: list[int] = []
    finish_reason = "length"

    def generate():
        nonlocal last_logits, finish_reason, completion_ids
        for i in range(req.max_tokens):
            token = _sample(last_logits, req.temperature, req.top_k, req.top_p, req.min_p)
            if token in eos_ids:
                finish_reason = "stop"
                break

            completion_ids.append(token)
            text = _tokenizer.decode([token])

            # Check stop strings
            for s in stop_strs:
                full = _tokenizer.decode(completion_ids)
                if s in full:
                    finish_reason = "stop"
                    break
            if finish_reason == "stop":
                break

            logits = _engine.forward(np.array([token], dtype=np.int64), cache)
            last_logits = np.asarray(logits[0])
            yield text

    if req.stream:
        def event_stream():
            created = int(time.time())
            req_id = f"cmpl-{uuid.uuid4().hex[:24]}"
            for text in generate():
                chunk = {
                    "id": req_id,
                    "object": "text_completion",
                    "created": created,
                    "model": req.model,
                    "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            final = {
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": req.model,
                "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": finish_reason}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    t0 = time.time()
    generated = "".join(list(generate()))
    created = int(t0)

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:24]}",
        created=created,
        model=req.model,
        choices=[CompletionChoice(text=generated, finish_reason=finish_reason)],
        usage=CompletionUsage(
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(completion_ids),
            total_tokens=len(prompt_ids) + len(completion_ids),
        ),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _engine is not None}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    default_tokenizer = "hub/models--mlx-community--Qwen3.6-35B-A3B-4bit"
    default_model = "data/models--mlx-community--Qwen3.6-35B-A3B-4bit"

    parser = argparse.ArgumentParser(description="MoE-Infer OpenAI-compatible server")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--tokenizer", default=default_tokenizer)
    parser.add_argument("--mode", default="Fused4bit", choices=["Fused4bit", "Fused4bitStripped"])
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--telemetry", action="store_true")
    args = parser.parse_args()

    if args.telemetry:
        record_engine_telemetry(True)

    print(f"Loading model from {args.model}...")
    print(f"Tokenizer from {args.tokenizer}...")
    load_engine(args.model, args.tokenizer, pipeline_mode=args.mode, k=args.k)
    print(f"Ready. Listening on {args.host}:{args.port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

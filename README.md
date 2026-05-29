# MoE-Infer

Run the Qwen3.6-35B-A3B MoE model on your Mac.  It chats, streams
word-by-word, and understands images — all on-device, no internet needed.

## What you need

- A Mac with Apple Silicon (M1 or newer) and at least 32 GB of unified memory
  (the BQ4 model is ~19 GB on disk and the engine mmaps it plus expert
  scratch buffers; expect roughly model size + a few GB of headroom)
- About 100 GB of free disk space (70 GB download + ~20 GB per quantized model)
- The original HuggingFace model downloaded to `hub/models--Qwen--Qwen3.6-35B-A3B`

## Setup (do this once)

Open Terminal and run these commands from the project folder:

```bash
# 1. Build and install
uv add moe_infer[all]  # or `uv sync --reinstall --extra vision` if you live inside this project

# 2. Download the model (about 70 GB)
hf download Qwen/Qwen3.6-35B-A3B \
  --local-dir hub/models--Qwen--Qwen3.6-35B-A3B

```

If you live in a cave, you can install with pip `pip install moe_infer[all]`

```python
# 3. Convert the model (takes ~5 minutes per scheme)
from moe_infer.qwen35_moe import convert


# BQ4 only (block-aware quantization)
convert('hub/models--Qwen--Qwen3.6-35B-A3B', 'data/Qwen3.6-35B-A3B', version='3.6')

# INT4 only (all weights quantized)
convert('hub/models--Qwen--Qwen3.6-35B-A3B', 'data/Qwen3.6-35B-A3B', version='3.6', scheme='int4')

# Both at once
convert('hub/models--Qwen--Qwen3.6-35B-A3B', 'data/Qwen3.6-35B-A3B', version='3.6', scheme=['bq4', 'int4'])
```

When it's done you'll see:

```
data/Qwen3.6-35B-A3B/
├── model_bq4/        ← BQ4 compressed model (~19 GB)
├── model_int4/       ← INT4 compressed model (~18 GB)
├── tokenizer/        ← turns words into numbers (and back)
└── vision_encoder/   ← lets it "see" images
```

## Chat

Start Python and load the model:

```python
from moe_infer import Qwen35MoEPipeline

# Default: uses model_bq4/
pipe = Qwen35MoEPipeline("data/Qwen3.6-35B-A3B")

# INT4 quantization:
pipe = Qwen35MoEPipeline("data/Qwen3.6-35B-A3B", quantize_mode="int4")
```

### Text

```python
pipe.chat("Hello!")
# → 'Hello! How can I help you today?'

pipe.chat("What is the capital of France?")
# → 'The capital of France is **Paris**.'
```

The model remembers your conversation.  Call `pipe.reset()` to start fresh.

### Streaming

See words appear as the model generates them:

```python
for word in pipe.chat("Write a haiku about cats", stream=True):
    print(word, end="", flush=True)
```

### Images

Point it at a photo and ask about it:

```python
pipe.chat("What is in this photo?", images=["data/crycat-crying-cat.gif"], max_image_pixels=65536)
```

### Expert LRU cache

By default the engine `pread`s routed expert weights from the mmap'd
weight file every MoE layer, every token, and relies on the macOS page
cache to absorb repeats.  That's usually enough on M2/M3/M4 + a fast
SSD.  On older hardware (M1, M1 Pro) or slower disks, expert I/O can
dominate and a GPU-resident LRU helps.

Pass `expert_cache=N` to allocate an N-entry shared LRU:

```python
pipe = Qwen35MoEPipeline("data/Qwen3.6-35B-A3B", expert_cache=32)
```

The cache is a single LRU shared across all MoE layers, keyed by
`(layer, expert_idx)`.  Routers structurally re-use hot experts at
multiple layers, so a small shared pool catches that overlap with
minimal memory.  `0` disables it.

Each entry is ~1.7 MB on Qwen3.6-35B, so:

| `expert_cache=` | Footprint | Notes |
|---|---|---|
| `0` (default) | 0 MB | OS page cache only — best on M2/M3/M4 + fast SSD |
| `32` | ~54 MB | Sweet spot on M1 / M1 Pro |
| `128` | ~218 MB | More headroom for long contexts; diminishing returns |

Empirical measurements on the "write a 200 word essay" prompt:

| Hardware | `expert_cache=0` | `expert_cache=32` |
|---|---|---|
| M4 + fast SSD | 6.5 tok/s | 6.8 tok/s |
| M1 Pro + internal SSD | ~7 tok/s | ~9.5 tok/s |

The break-even depends on how often your prompts route to the same
experts across layers; the LRU pays off when cross-layer expert reuse
is high.

### Multi-Token Prediction (MTP)

Qwen3.6 includes an MTP draft head that predicts future tokens.  The
quantized model bundles the MTP weights automatically — no extra config
needed.  The engine loads them at startup:

```python
pipe = Qwen35MoEPipeline("data/Qwen3.6-35B-A3B")
print(pipe._has_mtp)  # True for Qwen3.6, False for Qwen3.5

# Off by default; opt in to capture last_h_pre_norm for a custom draft loop.
pipe.chat("Hello", mtp=True)
```

The Rust MTP forward pass (`engine.mtp_forward(token_id)`) and KV cache
plumbing (`engine.mtp_reset()`, `engine.mtp_rollback(pos)`) are functional.
The Python-side speculative decoding loop (`generate_from_mtp`) currently
delegates to the standard autoregressive loop.

**Why is it default-off?**  Speculative decoding needs the main engine to
verify K draft tokens in roughly the cost of one main forward.  This
engine's `forward_hidden` processes input tokens sequentially (1-token
forward ≈ 100 ms, 2-token forward ≈ 210 ms on the M4 / 35B BQ4), so
spec decoding `(1+α) tokens / 2× main cost` is strictly worse than
baseline `1 token / 1× main cost`.  A future batched-attention engine
would make MTP a real win — until then, the flag stays for
researchers wiring up alternative draft loops.

## Verification

To verify that the BQ4 engine computes the same logits as the original
model, the pipeline is:

1.  **Strip** the HF model to 4 layers / 4 experts (fast test).
2.  **Quantize** the stripped model to BQ4.
3.  **Dequantize** BQ4 back to HF safetensors.
4.  **Compare** logits: dequantized HF (transformers) vs BQ4 engine (Rust).

All four steps run locally — no network needed after the initial download.

```bash
# 1. Strip: 40 layers → 4 layers, 256 experts → 4 experts (~2.4 GB)
python -m moe_infer.strip hub/models--Qwen--Qwen3.6-35B-A3B \
  --out hub/models--Qwen--Qwen3.6-35B-A3B-Strip

# 2. Quantize to BQ4 (~0.9 GB)
python -c "
from moe_infer.qwen35_moe import convert
convert('hub/models--Qwen--Qwen3.6-35B-A3B-Strip',
        'data/Qwen3.6-35B-A3B-Strip', version='3.6')
"

# 3. Dequantize back to HF safetensors (~2.4 GB)
python -m moe_infer.dequantize data/Qwen3.6-35B-A3B-Strip/model_bq4 \
  --ref hub/models--Qwen--Qwen3.6-35B-A3B-Strip \
  --out hub/models--Qwen--Qwen3.6-35B-A3B-Strip-Dequant

# 4. Compare logits
python verify_nway.py
```

Expected metrics on the 29-token test sequence (vocab_size=248,320):

| Comparison | Cosine | Max Diff | Mean Diff | Notes |
|---|---|---|---|---|
| Original HF vs BQ4 engine | ~0.902 | ~4.6 | ~0.79 | INT4 quantization loss |
| **Dequantized HF vs BQ4 engine** | **~0.925** | **~4.5** | **~0.72** | Same weights, GPU vs CPU path |
| FusedExp1 vs FusedExp2 | 1.000 | 0.000 | 0.000 | Bit-exact between Rust pipelines |

The dequantized model reproduces the engine more closely than the original
because both use the same quantized weights.  The remaining gap (~0.075) is
numerical precision differences between Metal GPU (on-the-fly INT4 matvec)
and PyTorch CPU (pre-dequantized BF16 matmul).

## Tips

| Tip | What to do |
|---|---|
| Text-only chat | `Qwen35MoEPipeline("data/...")` — it finds everything automatically |
| Reset conversation | `pipe.reset()` |
| Conversation history | `pipe.messages` — see what was said |
| Engine timing | `pipe.telemetry` — how long each step took |
| Switch quantization | `quantize_mode="int4"` or `quantize_mode="bq4"` (default) |
| Expert LRU cache | `Qwen35MoEPipeline(..., expert_cache=32)` — try it on M1/M1 Pro |
| MTP support | Qwen3.6 models load MTP automatically; `pipe._has_mtp` reports status |

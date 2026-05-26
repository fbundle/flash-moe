# MoE-Infer

Fast Mixture-of-Experts inference on Apple Silicon.  Pure Rust engine with
hand-tuned Metal shaders — no Python ML frameworks at runtime.  Expert
weights stream from SSD on demand via mmap.

## Quick Start

### 1. Build

```bash
maturin develop --release -m moe_infer_rs/Cargo.toml
```

### 2. Quantize

Download the HF model to `hub/models--Qwen--Qwen3.6-35B-A3B`, then:

```bash
python quantize.py \
    --model hub/models--Qwen--Qwen3.6-35B-A3B \
    --output data/models--Qwen--Qwen3.6-35B-A3B-bq4 \
    --qwen36
```

The `--qwen36` flag corrects Qwen3.6 norm weights to the Qwen3.5 convention
used by the engine.  Quantization takes ~10 minutes and produces:

```
data/models--Qwen--Qwen3.6-35B-A3B-bq4/
├── config.json
├── model_weights.bin       # ~2 GB (mmap'd)
├── model_weights.json      # tensor manifest
└── packed_experts/          # 40 × layer_NN.bin (~4.5 GB total)
```

### 3. Chat

```bash
python chat.py --model data/models--Qwen--Qwen3.6-35B-A3B-bq4
```

## Python API

```python
from moe_infer import Model, Engine, Cache
```

### Model

```python
model = Model("data/models--Qwen--Qwen3.6-35B-A3B-bq4")
```

### Engine

```python
engine = Engine(model, pipeline_mode="Qwen35MoEBq4Exp2", k=0)
```

`k=0` uses the model default (8 experts per token).  Set `k=4` for lower
expert I/O at a small quality cost.

| Method | Description |
|--------|-------------|
| `engine.embed_lookup(token_ids)` | Token IDs → embeddings `[N, hidden]` |
| `engine.forward_hidden(embeddings, cache)` | Embeddings → logits `[N, vocab]` |
| `engine.forward(token_ids, cache)` | Convenience: `embed_lookup` + `forward_hidden` |
| `engine.upload_cache(cache)` | CPU cache → GPU buffers |
| `engine.download_cache(cache)` | GPU buffers → CPU cache |
| `engine.telemetry()` | Per-layer timing dict |

### Cache

```python
cache = Cache(model)
```

| Method | Description |
|--------|-------------|
| `cache.pos` | Current sequence position |
| `cache.reset()` | Clear KV cache and linear attention state |
| `cache.save(bin, json)` | Persist to disk |
| `Cache.load(bin, json)` | Restore from disk |

### Pipeline modes

| Mode | Description |
|------|-------------|
| `Qwen35MoEBq4Exp1` | 40 layers, 256 experts |
| `Qwen35MoEBq4Exp2` | 40 layers, 256 experts (newer kernel) |

Architecture suffix `_Stripped` (4 layers, 4 experts) auto-detected from
`config.json` for verification models built with `--strip`.

## Benchmarks

```bash
python bench.py
```

M1 Pro 14 GPU, Qwen3.6-35B-A3B-BQ4, 32-token prompt, 100-token greedy decode:

| Mode | tok/s |
|------|-------|
| Qwen35MoEBq4Exp2 | ~10 |
| CPU (reference) | ~0.15 |

Expert I/O dominates at ~70% of per-layer time.  LZ4 compression cuts SSD
reads by 40–55% with negligible decompression overhead:

```bash
python helpers/compress_experts_lz4.py data/models--Qwen--Qwen3.6-35B-A3B-bq4
```

The engine auto-detects `packed_experts_lz4/` at load time.

## Verification

```bash
python verify_nway.py
```

Cross-checks logits across CPU, Metal, and reference implementations on the
stripped 4-layer model.

## Quantization

See [`quant/README.md`](quant/README.md) for the BQ4 scheme.  Quick summary:

| Block | Format | Why |
|-------|--------|-----|
| `self_attn.{q,k,v,o}_proj` | BF16 | Q·Kᵀ amplifies noise quadratically |
| `mlp.gate` | BF16 | Router error misroutes tokens |
| `attn.qkv`, `attn.proj` | BF16 | Attention projections |
| `patch_embed.proj`, `pos_embed` | BF16 | Vision encoder sensitivity |
| `lm_head` | INT8 | Per-channel symmetric, 49% size reduction |
| Everything else | INT4 | 64-group affine, 4.5 bits/weight |
| Norm weights, biases | BF16 | Vectors, sensitive to error |

Vision encoder weights (`vision_tower.*`) are excluded from the main pipeline
and loaded directly from HF safetensors at runtime by `vision_demo.py`.

## Model format

```
model_dir/
├── config.json
├── model_weights.bin           # mmap'd non-expert weights
├── model_weights.json          # tensor manifest
└── packed_experts/             # layer_00.bin … layer_39.bin
    (packed_experts_lz4/)       # optional LZ4-compressed experts
```

Tensor names use the MLX convention: `language_model.model.layers.{L}.{block}.{kind}`.
HF → MLX mapping: [`quant/name_mapping.json`](quant/name_mapping.json).

## Project layout

```
moe_infer_rs/src/
  engine.rs                   Engine trait + DynEngine
  engine/qwen35_moe/
    constants.rs              ModelConfig trait + FullModel/StrippedModel
    fused_bq4_exp1.rs         Metal GPU engine (variant 1)
    fused_bq4_exp2.rs         Metal GPU engine (variant 2, current)
    metal_context.rs          Metal device, pipelines, expert LRU cache
    metal_kernels.rs          Kernel dispatch helpers
    shaders.metal             Compute shaders
    cpu.rs                    Pure-CPU reference engine
  model.rs                    Model loading (config + weights + experts)
  model/config.rs             Runtime config store
  model/weights.rs            Mmap'd weight file + manifest
  model/expert.rs             Per-layer expert file I/O
  cache.rs                    KV cache + linear attention state
  quant.rs                    Quant enum + BF16/INT4/INT8 encode/decode
  quantize/qwen35_moe/bq4.rs  HF → BQ4 quantization pipeline
  python_bindings.rs          PyO3 bindings

quant/
  README.md                   BQ4 quantization reference
  name_mapping.json           HF → MLX tensor name mapping

helpers/
  quantize_from_hf.py         HF BF16 → MoE-Infer conversion (calls Rust)
  convert.py                  MLX 4-bit → MoE-Infer conversion
  compress_experts_lz4.py     LZ4-compress packed experts
  strip_model.py              Build stripped model for verification
```

## License

MIT

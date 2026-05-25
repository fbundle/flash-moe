# MoE-Infer

High-performance inference engine for Mixture-of-Experts models on Apple Silicon. Streams expert weights from SSD on demand — no Python ML frameworks at runtime, just Rust and hand-tuned Metal shaders.

Supports `mlx-community/Qwen3.5-35B-A3B-4bit` and `mlx-community/Qwen3.6-35B-A3B-4bit`.

## Hardware Requirements

- Mac with Apple Silicon (M1/M2/M3/M4)
- ~20 GB free SSD space for model weights
- macOS 14+ (for Metal 3)

## Quick Start

### 1. Quantize the model

```bash
# From HuggingFace unquantized BF16 model:
python helpers/quantize_from_hf.py \
    --model hub/models--Qwen--Qwen3.6-35B-A3B \
    --output data/my-model
```

Or convert from an MLX-format 4-bit model:

```bash
python helpers/convert.py --model hub/models--mlx-community--Qwen3.6-35B-A3B-4bit
```

### 2. Build and run

```bash
maturin develop --release -m moe_infer_rs/Cargo.toml
python chat.py
```

## Python API

```python
from moe_infer import Model, Engine, Cache, record_engine_telemetry
```

### Model

| Method | Description |
|--------|-------------|
| `model = Model(model_path)` | Load model weights, config, and expert file handles |

### Engine

| Method | Description |
|--------|-------------|
| `engine = Engine(model, pipeline_mode="Fused4bit", k=0)` | Initialize Metal GPU resources. `k` selects experts per token (0 = model default, 8) |
| `engine.forward(input_ids, cache)` | Forward pass, returns `[n_tokens, vocab_size]` float32 logits |
| `engine.upload_cache(cache)` | Sync CPU cache → GPU buffers |
| `engine.download_cache(cache)` | Sync GPU buffers → CPU cache |
| `engine.telemetry()` | Returns dict of per-engine timing metrics |

### Cache

| Method | Description |
|--------|-------------|
| `cache = Cache(model)` | Create KV caches + linear attention state |
| `cache.reset()` | Reset position, KV caches, and linear attention states |
| `cache.save(bin_path, json_path)` | Save cache to flat binary + JSON manifest |
| `cache.load(bin_path, json_path)` | Load cache from flat binary + JSON manifest |

### Pipeline Modes

| Mode | Description |
|------|-------------|
| `Fused4bit` | Full model: 40 layers, 256 experts, K=8 |
| `Fused4bitStripped` | Stripped model: 4 layers, 4 experts, K=4 (for verification) |

### CPU Engine (Rust only)

A pure-CPU reference engine using `ndarray` that mirrors the GPU pipeline. Not exposed via Python bindings yet.

```rust
use moe_infer::engine::qwen35_moe::CpuEngine;
let mut engine = CpuEngine::<FullModel>::new(&model, k)?;
let logits = engine.forward(&[1, 2, 3], &mut |_| false)?;
```

## Quantization

See [`quant/README.md`](quant/README.md) for the BQ4 quantization scheme. Tensor names follow the MLX convention described in [`quant/name_mapping.json`](quant/name_mapping.json).

## Model Format

MoE-Infer expects a model directory with:

```
model_dir/
├── config.json                 # HF config (read directly by Rust engine)
├── model_weights.bin           # Mmap'd: all non-expert weights (MLX naming)
├── model_weights.json          # Tensor manifest (name → offset, size, shape, dtype)
├── packed_experts/             # Per-layer expert files
│   ├── layer_00.bin
│   ├── layer_01.bin
│   └── ...
├── packed_experts_lz4/         # LZ4-compressed experts (optional, smaller SSD footprint)
│   ├── layer_00.bin
│   └── ...
├── tokenizer.json
└── vocab.json
```

Tensor names use the MLX convention: `language_model.model.layers.{L}.{block}.{kind}`.
An HF→MLX name mapping is provided in [`quant/name_mapping.json`](quant/name_mapping.json).

The engine auto-detects `packed_experts_lz4/` at load time and falls back to `packed_experts/`.

### Cache Format

Cache files use the same flat binary + JSON manifest format as model_weights:

```
cache.bin       # Flat concatenation of all cache tensors (f32 + u32 scalars)
cache.json      # Manifest: name → {offset, size, shape, dtype}
```

## Verification

`verify_nway.py` checks numerical correctness:

```bash
python verify_nway.py
```

Compares `Cpu`, `Fused4bit` (Rust), `C` (C bench), and `mlx-lm` on the stripped 4-layer model. Outputs an N×N max_diff matrix.

Expected: all non-mlx engines agree within ULP-level tolerance.

## Benchmarking

```bash
python bench.py
```

Tests forward passes across engine variants. Requires the full 40-layer model.

## Performance

Apple M1 Pro 14 GPUs, Qwen3.5-35B-A3B-4bit (40 layers, 256 experts, K=8), 32-token prompt, 100-token greedy generation:

| Mode | tok/s |
|------|-------|
| Fused4bit (Rust) | ~10 |
| Cpu (reference) | ~0.15 |

Expert I/O (SSD reads) dominates at ~70% of per-layer time.

## Project Structure

```
moe_infer_rs/                 Rust engine + Python bindings
  src/
    lib.rs                    Module declarations + Python module init
    engine.rs                 Engine trait, DynEngine, EngineEnum, telemetry
    model.rs                  Model struct (loads config + weight/expert files)
    cache.rs                  KV cache + linear attention state (binary I/O format)
    math_util.rs              CPU math: RMS norm, softmax, RoPE, dequant, SwiGLU, SSM, attention, conv1d
    constants.rs              Shared constants + backward-compat re-exports
    error.rs                  Error types
    python_bindings.rs        PyO3 bindings (Model, Cache, Engine)
    engine/
      qwen35_moe.rs           Module declarations for qwen35_moe submodules
      qwen35_moe/
        constants.rs          ModelConfig trait + FullModel/StrippedModel impls
        cpu.rs                CPU reference engine (ndarray, pure f32)
        fused_4bit.rs           Fused4bit GPU pipeline (3-CMD, Metal)
        metal_context.rs      Metal device/pipelines, ExpertCache LRU, scratch bufs
        metal_kernels.rs      Metal kernel dispatch (matvec, SwiGLU, conv1d, SSM, attention)
        shaders.metal         Metal compute shaders (embedded via include_str!)
  Cargo.toml
  pyproject.toml

quant/                        BQ4 quantization scheme and name mapping
  README.md                   Block→format quantization policy
  name_mapping.json           HF → MLX tensor name mapping (177 patterns)
  verify_name_mapping.py      Check mapping covers all HF tensors

helpers/                      Model conversion scripts
  quantize_from_hf.py         HF unquantized → MoE-Infer 4-bit format
  convert.py                  One-step MLX → MoE-Infer conversion
  extract_weights.py          Non-expert weights → model_weights.bin + .json
  repack_experts_4bit.py      MLX 4-bit experts → packed_experts/
  compress_experts_lz4.py     packed_experts/ → packed_experts_lz4/ (~40-55% compression)
  repack_experts_2bit.py      packed_experts/ → packed_experts_2bit/ (experimental)
  strip_model.py              Build 4-layer stripped model for verification

bench.py                      Multi-engine performance benchmark
verify_nway.py                N-way logit comparison
chat.py                       Interactive chat demo
```

## License

MIT

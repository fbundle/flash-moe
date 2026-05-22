# MoE-Infer

High-performance inference engine for Mixture-of-Experts models on Apple Silicon. Streams expert weights from SSD on demand — no Python ML frameworks at runtime, just Rust and hand-tuned Metal shaders.

Supports `mlx-community/Qwen3.5-35B-A3B-4bit` and `mlx-community/Qwen3.6-35B-A3B-4bit`.

The `FusedWoods` pipeline mode is named after Dan Woods, author of the original C/Metal inference engine this project builds upon.

## Hardware Requirements

- Mac with Apple Silicon (M1/M2/M3/M4)
- ~20 GB free SSD space for model weights
- macOS 14+ (for Metal 3)

## Quick Start

### 1. Convert the model

```bash
python helpers/convert.py --model hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
```

### 2. Build and install Python bindings and Run inference

```bash
maturin develop --release -m moe_infer_rs/Cargo.toml
python chat.py
```


## Python API

```python
from moe_infer import Model, Engine, Cache
```

### Model

| Method | Description |
|--------|-------------|
| `model = Model(model_path)` | Load model weights, config, and expert file handles (data only) |

### Engine

| Method | Description |
|--------|-------------|
| `engine = Engine(model, pipeline_mode="FusedExp")` | Initialize Metal GPU resources and load the model |
| `engine.forward(input_ids, cache)` | Forward pass, returns `[n_tokens, vocab_size]` float32 logits |
| `engine.generate(input_ids, cache, max_tokens, temperature, top_k, top_p, min_p, eos_token_ids)` | Autoregressive generation, returns `[n_tokens]` int64 token ids |
| `engine.stream_generate(input_ids, cache, ...)` | Like generate but yields `(token_id, logits)` tuples incrementally |
| `engine.telemetry()` | Returns dict: `prefill_ms`, `total_ms`, `tokens_generated`, `tokens_per_sec` |

### Cache

| Method | Description |
|--------|-------------|
| `cache = Cache(model)` | Create KV caches + linear attention state for the given model |
| `cache.reset()` | Reset position, KV caches, and linear attention states |
| `cache.pos` | Current sequence position (read-only) |

### Pipeline Modes

| Mode | Description |
|------|-------------|
| `Cpu` | Pure CPU reference. All operations on CPU. Slow but useful for debugging. |
| `Gpu` | GPU kernels with individual dispatch. No command buffer fusion. |
| `FusedExp` | Linear attention fused into one command buffer. MoE experts dispatched individually. |
| `FusedWoods` | Full 3-command-buffer pipeline (CMD1 + CMD2 + async CMD3). **Recommended.** |

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 256 | Maximum new tokens to generate |
| `temperature` | 0.0 | < 0.01 for greedy, > 0 for sampling |
| `top_k` | 0 | Keep top-k logits (0 = disabled) |
| `top_p` | 1.0 | Nucleus sampling threshold |
| `min_p` | 0.0 | Minimum probability relative to max |
| `eos_token_ids` | [248046, 248044] | Stop tokens |

## Verification

`verify_nway.py` checks numerical correctness across all engines on a stripped 4-layer model:

```bash
python verify_nway.py
```

Runs `Cpu`, `Gpu`, `FusedWoods`, `FusedExp` (Rust), `C` (C bench), and `mlx-lm` on the same token sequence, then compares logits pairwise. Outputs an N×N `max_diff` matrix with per-engine status (IDENTICAL / MATCH / CLOSE / DIVERGE).

Expected results: all non-mlx engines agree within `2.6e-05` (ULP-level). mlx-lm diverges at `~0.11` due to bf16 precision. C bench and Rust FusedWoods are byte-for-byte identical.

## Benchmarking

`bench.py` benchmarks C and Rust GPU pipelines on the full 40-layer model:

```bash
python bench.py
```

Tests forward passes at 100, 200, and 300 tokens across `Gpu`, `FusedWoods`, `FusedExp` (Rust) and the C bench. Prints per-engine latency and throughput, plus speedup vs C.

Requires the full model in `data/models--mlx-community--Qwen3.5-35B-A3B-4bit`.

## Model Format

MoE-Infer expects a model directory with:

```
model_dir/
├── model_weights.bin        # Mmap'd: all non-expert weights (embeddings, norms, projections, shared experts)
├── model_weights.json       # Tensor manifest (name → offset, size, shape, dtype)
├── model_config.json        # Model hyperparameters
├── packed_experts/          # Per-layer expert files
│   ├── layer_00.bin
│   ├── layer_01.bin
│   └── ...
├── tokenizer.json           # HF tokenizer (used by Python bindings)
└── vocab.json
```

Helper scripts in `helpers/` convert from HuggingFace/MLX format:
- `convert.py` — One-step conversion: config, weights, and expert repacking
- `extract_weights.py` — Non-expert weights → `model_weights.bin` + `model_weights.json`
- `repack_experts_4bit.py` — MLX 4-bit experts → `packed_experts/` per-layer files

## Performance

Apple M4, Qwen3.5-35B-A3B-4bit (40 layers, 256 experts, K=8), 32-token prompt, 100-token greedy generation:

| Mode | tok/s |
|------|-------|
| FusedWoods | 2.69 |
| FusedExp | 2.14 |
| Gpu | 1.70 |
| Cpu | 0.15 |

Expert I/O (SSD reads) dominates at ~72% of per-layer time.

## Project Structure

```
moe_infer_rs/           Rust engine + Python bindings
  src/
    gpu_forward.rs      Layer forward: linear/full attention, MoE routing
    pipeline_common.rs  Shared types, CPU helpers, DeferredExperts
    pipeline_cpu.rs     Cpu pipeline mode
    pipeline_fusedwoods.rs  FusedWoods pipeline mode (CMD1+CMD2+CMD3)
    pipeline_fusedexp.rs FusedExp pipeline mode
    python_bindings.rs  PyO3 bindings (Model, Engine, Cache)
    metal_context.rs    Metal device init, pipeline creation
    kernels.rs          GPU kernel dispatch
    weights.rs          Mmap'd weight file + tensor lookup
    config.rs           JSON model config
    quant.rs            bf16↔f32, CPU dequant matvec, SwiGLU, RMS norm
    lib.rs              Module declarations
  shaders/
    shaders.metal       Metal compute shaders (embedded at compile time)
  Cargo.toml

helpers/                Model conversion scripts
  convert.py            One-step convert: config + weights + experts
  extract_weights.py    Non-expert weights → model_weights.bin
  repack_experts_4bit.py MLX experts → packed_experts/
  gen_model_config.py   Config generation

verify_nway.py          Multi-engine logit verification
```

## License

MIT

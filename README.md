# MoE-Infer

High-performance inference engine for Mixture-of-Experts models on Apple Silicon. Streams expert weights from SSD on demand — no Python ML frameworks at runtime, just Rust and hand-tuned Metal shaders.

Supports `mlx-community/Qwen3.5-35B-A3B-4bit` and `mlx-community/Qwen3.6-35B-A3B-4bit`.

The `FusedWoods` pipeline mode is named after Dan Woods, author of the original C/Metal inference engine this project builds upon.

## Hardware Requirements

- Mac with Apple Silicon (M1/M2/M3/M4)
- ~20 GB free SSD space for model weights
- macOS 14+ (for Metal 3)

## Quick Start

### 1. Download and convert the model

```bash
# Download from HuggingFace
pip install huggingface_hub
hf download mlx-community/Qwen3.5-35B-A3B-4bit \
  --local-dir hub/models--mlx-community--Qwen3.5-35B-A3B-4bit

# Convert to MoE-Infer format
python helpers/extract_weights.py \
  --model hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --output hub/models--mlx-community--Qwen3.5-35B-A3B-4bit

python helpers/repack_experts_4bit.py \
  --model hub/models--mlx-community--Qwen3.5-35B-A3B-4bit

python helpers/gen_model_config.py \
  --model hub/models--mlx-community--Qwen3.5-35B-A3B-4bit
```

### 2. Build and install Python bindings

```bash
cd moe_infer_rs
maturin develop --release
```

### 3. Run inference

```python
from moe_infer import Context, Cache
import numpy as np

ctx = Context()
ctx.load_model("hub/models--mlx-community--Qwen3.5-35B-A3B-4bit",
               pipeline_mode="FusedWoods")
cache = ctx.new_cache()

# Forward pass
input_ids = np.array([248045, 8678, 198], dtype=np.int64)
logits = ctx.forward(input_ids, cache)

# Generate
new_ids = ctx.generate(input_ids, cache,
    max_tokens=256, temperature=0.7, top_k=50, top_p=0.9)

# Streaming
for token_id, logits in ctx.stream_generate(input_ids, cache, max_tokens=256):
    print(token_id)
```

## Python API

```python
from moe_infer import Context, Cache
```

### Context

| Method | Description |
|--------|-------------|
| `ctx.load_model(path, pipeline_mode="FusedWoods")` | Load a model. Modes: `Cpu`, `Gpu`, `FusedExp`, `FusedWoods` |
| `ctx.unload_model()` | Free Metal resources and close expert files |
| `ctx.new_cache()` | Create a new KV cache + linear attention state |
| `ctx.forward(input_ids, cache)` | Forward pass, returns `[n_tokens, vocab_size]` float32 logits |
| `ctx.generate(input_ids, cache, max_tokens, temperature, top_k, top_p, min_p, eos_token_ids)` | Autoregressive generation, returns `[n_tokens]` int64 token ids |
| `ctx.stream_generate(input_ids, cache, ...)` | Like generate but yields `(token_id, logits)` tuples |
| `ctx.telemetry()` | Returns dict: `prefill_ms`, `total_ms`, `tokens_generated`, `tokens_per_sec` |

### Cache

| Method | Description |
|--------|-------------|
| `cache.reset()` | Reset position, KV caches, and linear attention states for a new conversation |
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
| `temperature` | 1.0 | < 0.01 for greedy, > 0 for sampling |
| `top_k` | 50 | Keep top-k logits (0 = disabled) |
| `top_p` | 0.9 | Nucleus sampling threshold |
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
- `extract_weights.py` — Non-expert weights → `model_weights.bin` + `model_weights.json`
- `repack_experts_4bit.py` — MLX 4-bit experts → `packed_experts/` per-layer files
- `gen_model_config.py` — HF `config.json` → `model_config.json`

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
    python_bindings.rs  PyO3 bindings (Context, Cache)
    metal_context.rs    Metal device init, pipeline creation
    kernels.rs          GPU kernel dispatch
    weights.rs          Mmap'd weight file + tensor lookup
    config.rs           JSON model config
    quant.rs            bf16↔f32, CPU dequant matvec, SwiGLU, RMS norm
    tokenizer.rs        BPE tokenizer
    error.rs            Error types
    lib.rs              Module declarations
  shaders/
    shaders.metal       Metal compute shaders (embedded at compile time)
  Cargo.toml

helpers/                Model conversion scripts
  extract_weights.py    Non-expert weights → model_weights.bin
  repack_experts_4bit.py MLX experts → packed_experts/
  gen_model_config.py   Config generation
  export_tokenizer.py   Tokenizer export (C path)

verify_nway.py          Multi-engine logit verification
```

## License

MIT

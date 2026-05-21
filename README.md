# Flash-MoE

Pure C/Metal and Rust/Metal inference engine for Mixture-of-Experts models on Apple Silicon. Streams expert weights from SSD on demand — no Python ML frameworks, just C, Rust, and hand-tuned Metal shaders.

Supports both `mlx-community/Qwen3.5-35B-A3B-4bit` and `mlx-community/Qwen3.6-35B-A3B-4bit`.

## Quick Start (Python)

```bash
# Build Rust + Python bindings # Run the example
cd moe_infer_rs
maturin develop --release --features python-bindings
cd ..
python example.py
```

## Rust Build

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust toolchain (via [rustup](https://rustup.rs))
- Xcode Command Line Tools (for Metal framework)

### Prepare model data

```bash
# Download model into hub/
pip install huggingface_hub
hf download mlx-community/Qwen3.5-35B-A3B-4bit \
  --local-dir hub/models--mlx-community--Qwen3.5-35B-A3B-4bit

# Convert to Flash-MoE format → data/
python helpers/convert.py \
  --model hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --output data
```

### Build and run

```bash
# Benchmark binary (pure Rust, no Python)
cd moe_infer_rs
cargo run --release --bin bench -- \
  --model ../data/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --tokens 500
```

### Python bindings

```bash
cd moe_infer_rs

# Build and install into current Python environment
maturin develop --release --features python-bindings

# Python usage
python -c "
from moe_infer import Context, Cache
ctx = Context()
ctx.load_model('data/models--mlx-community--Qwen3.5-35B-A3B-4bit')
cache = ctx.new_cache()
# ... feed input_ids via ctx.forward(...) or ctx.generate(...)
"
```

### Rust server

```bash
cd moe_infer_rs
cargo run --release -- --model ../data/models--mlx-community--Qwen3.5-35B-A3B-4bit --serve 8080
```

## C Build (baseline)

```bash
cd moe_infer_c
python3 patch_bench.py
clang -O2 -Wall -fobjc-arc -framework Metal -framework Foundation \
      -framework Accelerate bench.m -lpthread -lcompression -o bench
./bench --prompt "bench" --tokens 500 --k 8
```

## Python API

```python
from moe_infer import Context, Cache
import numpy as np

ctx = Context()

# Load model (pipeline_mode: "CpuOnly", "Gpu", "FusedExp")
ctx.load_model("/path/to/model", pipeline_mode="FusedExp")

# Create cache (holds KV cache + linear attention states)
cache = ctx.new_cache()

# Forward pass: input_ids is the FULL conversation
# Only new tokens (cache.pos onwards) are processed
input_ids = np.array([1, 2, 3, ...], dtype=np.int64)
logits = ctx.forward(input_ids, cache)  # shape: [n_tokens, vocab_size]

# Generate with sampling
new_ids = ctx.generate(input_ids, cache,
    max_tokens=256,
    temperature=0.7, top_k=50, top_p=0.9,
    eos_token_ids=np.array([248046, 248044], dtype=np.int64))

# Streaming generate — returns list of (token_id, logits) tuples
results = ctx.stream_generate(input_ids, cache, max_tokens=256)

# Telemetry from last call
info = ctx.telemetry()
# {"prefill_ms": 123.4, "total_ms": 567.8, "tokens_generated": 50, "tokens_per_sec": 88.0}

# Reset for new conversation
cache.reset()

# Cleanup
ctx.unload_model()
```

## Architecture

35B-A3B model: 40 layers (30 linear attention + 10 full attention), 256 experts, K=8 active. Hidden dim 2048, head dim 256.

### Key Techniques

1. **SSD Expert Streaming** — Expert weights (~19GB 4-bit) read from SSD on demand via parallel `pread()`. Only K active experts per layer are loaded (~1.77MB each).

2. **Metal Compute Shaders** — 4-bit dequantized matvec, fused SwiGLU, RMS norm, batched attention, GPU RoPE, MoE combine + residual.

3. **Fused GPU Pipeline** — FusedExp mode fuses linear attention (qkv/z/b/a + conv1d + SSM) into a single CMD1. MoE experts are dispatched individually with GPU dequant. Fused3 (full C pipeline with async CMD3 + GPU combine) is planned but not yet implemented.

4. **FMA-Optimized Dequant** — Rearranges `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)`, using GPU fused multiply-add in one instruction.

## Project Structure

```
moe_infer_c/          Original C vendor code (performance baseline)
  infer.m             ~7000 line inference engine
  bench.m             Generated benchmark binary
  shaders.metal       Metal compute shaders
  patch_bench.py      Generate bench.m from infer.m

moe_infer_rs/         Rust port
  src/
    main.rs           CLI + HTTP server entry point
    bin/bench.rs      Pure Rust benchmark
    gpu_forward.rs    Fused layer forward, linear/full attention, MoE routing
    metal_context.rs  Metal init, pipeline creation, GPU weight buffer
    kernels.rs        GPU kernel dispatch wrappers
    expert.rs         Expert forward (CPU dequant matvec)
    moe.rs            MoE routing + expert dispatch
    full_forward.rs   Full model forward pass
    weights.rs        Weight file mmap + tensor lookup
    config.rs         JSON model config loading
    quant.rs          bf16→f32, CPU dequant matvec, SwiGLU
    tokenizer.rs      BPE tokenizer
    server.rs         HTTP server + SSE streaming
    python_bindings.rs PyO3 bindings (Context, Cache)
    timer.rs          Wall-clock timing
    lib.rs            Module declarations + re-exports
  shaders/
    shaders.metal     Metal compute shaders (embedded at compile time)
  Cargo.toml

helpers/
  extract_weights.py       Non-expert weights → model_weights.bin
  repack_experts_4bit.py   MLX 4-bit experts → packed_experts/
  repack_experts_2bit.py   4-bit → 2-bit requantization
  gen_model_config.py      Generate model_config.json from HF config.json
  export_tokenizer.py      Generate vocab.bin and tokenizer.bin (C path)

example.py            Python example using PyO3 bindings
pyproject.toml        Maturin build configuration
```

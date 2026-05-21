# Rust Port Report: Flash-MoE → Qwen3.5-35B-A3B-4bit

## Status

The Rust port (`moe_infer_rs/`) builds, runs, and generates coherent text on Apple M4 via PyO3 Python bindings. The original vendor C code lives in `moe_infer_c/` (patched for the 35B model) and serves as the performance baseline.

All interaction is through Python — `main.rs`, `bench.rs`, `tokenizer.rs`, and `server.rs` have been removed.

## Architecture Comparison

| Aspect | C (`moe_infer_c/infer.m`) | Rust (`moe_infer_rs/`) |
|--------|--------------------------|------------------------|
| Model config | `#define` compile-time constants | JSON-driven at runtime |
| Weight loading | mmap + zero-copy Metal buffer | Same (`newBufferWithBytesNoCopy`) |
| Shader compilation | Runtime (`newLibraryWithSource`) | Same (embedded via `include_str!`) |
| Expert I/O | `pread` from per-layer files | Same (`libc::pread`) |
| Linear attention | Fused CMD1: qkv/z/b/a + conv1d + SSM | Fused CMD1 (same architecture) |
| Full attention | GPU batched (scores, softmax, values, sigmoid) | GPU batched (same kernels) |
| MoE routing | CMD2: o_proj + residual + norm + gate | Separate CMD: gate + shared projs |
| Expert dispatch | CMD3: async commit + GPU combine | **Async CMD3 + GPU combine** ✓ |
| KV cache | GPU bf16 buffers | CPU f32 buffers |
| Python bindings | None (C-only) | PyO3 + Maturin (Context, Cache classes) |

## Pipeline Mode: FusedExp → Fused3 Progress

The goal is incrementally evolving `FusedExp` into `Fused3` (the C engine's 3-CMD architecture).

### What's implemented (FusedExp)

| Feature | Status |
|---------|--------|
| Fused CMD1 (linear attention: qkv/z/b/a + conv1d + SSM) | Done |
| GPU batched full attention (scores + softmax + values + sigmoid) | Done |
| GPU moe_combine_residual (expert weighted sum + shared expert + residual in one kernel) | Done |
| Async CMD3 (commit without wait, complete on next layer) | Done |
| PyO3 bindings (Context, Cache, telemetry, stream_generate) | Done |
| Ctrl-C interrupt handling (`py.check_signals()`) | Done |

### Remaining gaps to reach Fused3

| Gap | Description |
|-----|-------------|
| CMD2 fusion | Full-attention layers: fuse o_proj with router+shared projections into one CMD (currently 2 separate CMDs) |
| CMD1 attention fusion | Full-attention layers: QKV proj + batched attention are separate CMDs (Q/K norms and RoPE are CPU-side; need GPU kernels) |
| GPU KV cache | KV cache currently stored as CPU f32 buffers, uploaded per layer. Store on GPU persistently |
| GPU SSM state | Linear attention SSM state uploaded/downloaded per layer in non-fused path |

## Current sync points per layer

- **Linear attention** (30/40 layers): 2 sync points — CMD1 (fused linear) + async CMD3
- **Full attention** (10/40 layers): 4 sync points — QKV proj + batched attn + o_proj + router → async CMD3

C engine: 2 sync points for all layers. Closing the full-attention gap requires GPU RoPE and head-wise Q/K norm kernels.

## Output Coherence

The Rust engine produces coherent, sensible output verified against the same prompt. Earlier output divergence between C and Rust engines has been resolved — the remaining architectural differences (GPU vs CPU KV cache format, non-fused full-attention CMD2) don't affect correctness.

## Python API

```python
from moe_infer import Context, Cache

ctx = Context()
ctx.load_model("/path/to/model", pipeline_mode="FusedExp")
cache = ctx.new_cache()

# Forward / generate / stream
logits = ctx.forward(input_ids, cache)
new_ids = ctx.generate(input_ids, cache, max_tokens=256, temperature=0.7)
results = ctx.stream_generate(input_ids, cache, max_tokens=256)

# Telemetry
info = ctx.telemetry()
# {"ttft_ms": ..., "tokens_per_sec": ..., "tokens_generated": ...}
```

## Next Steps

1. **GPU RoPE kernel** — port C `apply_rope` shader to enable fusing QKV proj + batched attention in full-attention layers
2. **GPU KV cache** — store K/V caches persistently on GPU instead of uploading per layer
3. **CMD2 fusion** — fuse o_proj with router+shared gate projections for full-attention layers

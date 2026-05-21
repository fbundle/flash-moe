# Rust Port Report: Flash-MoE → Qwen3.5-35B-A3B-4bit

## Status

The Rust port (`moe_infer_rs/`) builds, runs, and generates coherent text on Apple M4. The Cython-wrapped C library (`moe_infer/`) has been deleted — it was too slow due to Python overhead and malloc/free per token.

The original vendor C code lives in `moe_infer_c/` (patched for the 35B model) and serves as the performance baseline.

## Architecture Comparison

| Aspect | C (`moe_infer_c/infer.m`) | Rust (`moe_infer_rs/`) |
|--------|--------------------------|------------------------|
| Model config | `#define` compile-time constants | JSON-driven at runtime (`config.json`) |
| Weight loading | mmap + zero-copy Metal buffer | Same (`newBufferWithBytesNoCopy`) |
| Shader compilation | Runtime (`newLibraryWithSource`) | Same (embedded via `include_str!`) |
| Expert I/O | `pread` from per-layer files | Same (`libc::pread`) |
| Linear attention | Fused CMD1: qkv/z/b/a + conv1d + SSM | Individual dispatches (bench.rs) / fused path exists (gpu_forward.rs) |
| MoE routing | CMD2: o_proj + residual + norm + gate | Individual dispatches |
| Expert dispatch | CMD3: async + deferred + GPU combine | Synchronous (wait_until_completed) |
| Full attention | GPU batched (scores, softmax, values) | CPU scalar (bench.rs) / GPU path exists |
| KV cache | GPU bf16 buffers | CPU f32 buffers (bench.rs) |
| Memory management | malloc/free per token for final_norm | Pre-allocated Vec<f32> |
| Deferred experts | Implemented (CMD3 commit without wait) | Placeholder (DeferredExperts struct, always sync) |
| Tokenizer | C BPE (binary `vocab.bin`) | HF `tokenizers` crate |

## Gaps vs C Code

### 1. bench.rs doesn't use the fused GPU path
`gpu_forward.rs` has `linear_attention_forward()` and `moe_layer_forward()` with the fused CMD1/CMD2/CMD3 architecture. But `src/bin/bench.rs` has inline copies of CPU attention functions that dispatch individual GPU matvecs per projection (one command buffer per matmul). Wiring the fused path would:
- Reduce prefill time (fewer command buffer commits)
- Fix output divergence (C and Rust produce different token sequences)
- Potentially increase generation speed

### 2. No CMD3 async expert dispatch
In C, CMD3 dispatches all K experts and commits without waiting — results are collected in the *next* layer. In Rust, experts are dispatched synchronously. The `DeferredExperts` struct is a placeholder. To implement:
- Store `&CommandBufferRef` in `DeferredExperts` using `unsafe` ObjC retain/release
- Commit CMD3 without waiting
- Wait and read back in the next layer's forward

### 3. No GPU-side combine (`moe_combine_residual`)
The kernel exists in shaders and the pipeline is created, but expert outputs are accumulated on CPU. Using GPU-side combine eliminates CPU↔GPU round-trips.

### 4. No `PipelineMode` enum
The forward functions always try GPU paths and fall back to CPU. There's no way to force CPU-only mode for debugging/comparison. An explicit `PipelineMode` enum (CpuOnly, Gpu, Fused) would allow:
- CPU-only benchmark to measure pure CPU speed
- Direct comparison of fused vs unfused GPU paths
- Easier debugging (CPU path is deterministic, GPU may have driver variance)

## Output Divergence

The C and Rust engines produce different token sequences from the same prompt and model. The C engine hits EOS at ~328 tokens; the Rust engine runs all 500 without EOS. This indicates a correctness issue:

* MoE routing: different gate projection results → different expert selection → different layer outputs
* Linear attention: fused GPU path vs CPU path may have numerical differences
* Full attention: GPU batched (C) vs CPU scalar (Rust) with different KV cache formats
* RMS norm: GPU kernel vs CPU computation with different float accumulation order

To debug: compare hidden state vectors after each layer between C and Rust for a single token.

## Files

| Directory | Purpose |
|-----------|---------|
| `moe_infer_c/` | Original C vendor code, patched for 35B model (hardcoded prompt IDs, no tokenizer) |
| `moe_infer_c/infer.m` | Original ~7000 line inference engine (397B, patched to 35B) |
| `moe_infer_c/bench.m` | Generated benchmark binary (29 hardcoded prompt token IDs) |
| `moe_infer_c/shaders.metal` | Metal compute shaders |
| `moe_infer_c/patch_bench.py` | Script to generate bench.m from infer.m |
| `moe_infer_rs/` | Rust port |
| `moe_infer_rs/src/gpu_forward.rs` | Fused layer forward, linear attention, MoE routing |
| `moe_infer_rs/src/bin/bench.rs` | Pure Rust benchmark (no HTTP) |
| `moe_infer_rs/src/kernels.rs` | GPU kernel dispatch wrappers |
| `moe_infer_rs/src/metal_context.rs` | Metal init, pipeline creation, GpuWeightCtx |
| `moe_infer_rs/shaders/shaders.metal` | Metal compute shaders (embedded at compile time) |

## Building and Running

### C benchmark
```bash
cd moe_infer_c
python3 patch_bench.py
clang -O2 -Wall -fobjc-arc -framework Metal -framework Foundation \
      -framework Accelerate bench.m -lpthread -lcompression -o bench
./bench --prompt "bench" --tokens 500 --k 8
```

### Rust benchmark
```bash
cd moe_infer_rs
cargo run --release --bin bench -- \
  --model /Volumes/Hippopotamus/vault/code/flash-moe/data/models--mlx-community--Qwen3.5-35B-A3B-4bit \
  --tokens 500
```

## Next Steps

1. **Add `PipelineMode` enum** — CpuOnly, GpuOnly, Fused variants for `linear_attention_forward` and `moe_layer_forward`
2. **Wire fused path into bench.rs** — replace inline CPU attention with `gpu_forward.rs` calls
3. **Investigate output divergence** — compare hidden states after each layer between C and Rust
4. **Implement CMD3 async expert dispatch** — use unsafe ObjC retain/release for true async
5. **GPU-side combine** — use `moe_combine_residual` kernel

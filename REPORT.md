# Rust Port Report: Flash-MoE → Qwen3.5-35B-A3B-4bit

## Status

The Rust port (`moe_infer_rs/`) builds, runs, and generates coherent text on Apple M4 via PyO3 Python bindings. The original vendor C code lives in `moe_infer_c/` (patched for the 35B model) and serves as the performance baseline.

**Fused3 is now complete.** All 40 layers run with 2 sync points (matching the C engine's 3-CMD architecture). `Fused3` is available as a pipeline mode.

## Architecture Comparison

| Aspect | C (`moe_infer_c/infer.m`) | Rust (`moe_infer_rs/`) |
|--------|--------------------------|------------------------|
| Model config | `#define` compile-time constants | JSON-driven at runtime |
| Weight loading | mmap + zero-copy Metal buffer | Same (`newBufferWithBytesNoCopy`) |
| Shader compilation | Runtime (`newLibraryWithSource`) | Same (embedded via `include_str!`) |
| Expert I/O | `pread` from per-layer files | Same (`libc::pread`) |
| Linear attention | Fused CMD1: qkv/z/b/a + conv1d + SSM | Fused CMD1 + out_proj + residual (same) |
| Full attention | GPU batched CMD2: attn + o_proj + residual + norm + gate | Same |
| MoE routing | CMD2: o_proj + residual + norm + gate | Same (full-attn in CMD2, linear in CMD1) |
| Expert dispatch | CMD3: async commit + GPU combine | Same |
| KV cache | GPU bf16 buffers | CPU f32 buffers |
| Python bindings | None (C-only) | PyO3 + Maturin (Context, Cache classes) |

## Stage-by-Stage Comparison: C vs Rust Fused3

Each row is one stage in the per-layer pipeline. Device shows where the work executes.

### Full-Attention Layers (indices 3, 7, 11, ...)

| # | Stage | C (bench.m) | Rust (Fused3) | Device |
|---|-------|-------------|---------------|--------|
| 0 | Deferred wait | `wait_until_completed` on prev CMD3 | `DeferredExperts::complete()` — wait + readback | GPU→CPU |
| 1 | Input norm | CPU `rms_norm` | CPU `rms_norm` | CPU |
| 2 | CMD1 encode | Q/K/V projections (3× `dequant_matvec_4bit`) | Same (3× `encode_matvec_into`) | GPU |
| 3 | CMD1 wait | `commit` + `wait_until_completed` | Same | GPU |
| 4 | Q/K norms | CPU per-head RMS norm | CPU per-head RMS norm | CPU |
| 5 | RoPE | CPU `apply_rope` on Q/K | CPU `apply_rope` on Q/K | CPU |
| 6 | KV cache | Append to GPU `gpu_k_cache`/`gpu_v_cache` | Append to CPU `FullAttnCache` (uploaded next layer) | GPU / **CPU** |
| 7 | Attn scores | GPU `attn_scores_batched` (in CMD2) | GPU `attn_scores_batched` (in CMD2) | GPU |
| 8 | Attn softmax | GPU `attn_softmax_batched` (in CMD2) | GPU `attn_softmax_batched` (in CMD2) | GPU |
| 9 | Attn values | GPU `attn_values_batched` (in CMD2) | GPU `attn_values_batched` (in CMD2) | GPU |
| 10 | Sigmoid gate | GPU `sigmoid` (in CMD2) | GPU `sigmoid_gate` (in CMD2) | GPU |
| 11 | o_proj | GPU `dequant_matvec_4bit` (in CMD2) | GPU `encode_matvec_into` (in CMD2) | GPU |
| 12 | Residual add | GPU `residual_add` (in CMD2) | GPU `residual_add` (in CMD2) | GPU |
| 13 | Post-attn norm | GPU `rms_norm_sum` + `rms_norm_apply` (in CMD2) | GPU `rms_norm_sum` + `rms_norm_apply_bf16` (in CMD2) | GPU |
| 14 | MoE gate | GPU `dequant_matvec_4bit` (in CMD2) | GPU `encode_matvec_into` (in CMD2) | GPU |
| 15 | Shared gate/up | GPU 2× `dequant_matvec_4bit` (in CMD2) | GPU 2× `encode_matvec_into` (in CMD2) | GPU |
| 16 | Shared gate score | GPU `dequant_matvec_4bit` (in CMD2) | GPU `encode_matvec_into` (in CMD2) | GPU |
| 17 | CMD2 wait | `commit` + `wait_until_completed` (single CMD for steps 7–16) | Same (single CMD) | GPU |
| 18 | Routing | CPU `softmax` + `topk` | CPU `cpu_softmax` + `cpu_topk` | CPU |
| 19 | Expert I/O | `pread` K experts from `packed_experts/layer_NN.bin` | Same (`libc::pread`) | Disk→CPU |
| 20 | CMD3: gate/up | GPU K×2 `dequant_matvec_offset` | GPU K×2 `encode_matvec_offset` | GPU |
| 21 | CMD3: SwiGLU | GPU K× `swiglu` | GPU K× `encode_swiglu` | GPU |
| 22 | CMD3: down | GPU K× `dequant_matvec_offset` | GPU K× `encode_matvec_offset` | GPU |
| 23 | CMD3: shared SwiGLU | GPU `swiglu` | GPU `encode_swiglu` | GPU |
| 24 | CMD3: shared down | GPU `dequant_matvec_4bit` | GPU `encode_matvec_into` | GPU |
| 25 | CMD3: combine | GPU `moe_combine_residual` | GPU `moe_combine_residual` | GPU |
| 26 | CMD3 commit | `commit` (async, **no wait**) | `commit` (async, **no wait**) — returns `DeferredExperts` | GPU |

**Full-attn diff**: Only stage 6 — C appends KV to persistent GPU buffers; Rust appends to CPU buffer and uploads next layer.

### Linear-Attention Layers (indices 0, 1, 2, 4, 5, 6, ...)

| # | Stage | C (bench.m) | Rust (Fused3) | Device |
|---|-------|-------------|---------------|--------|
| 0 | Deferred wait | `wait_until_completed` on prev CMD3 | `DeferredExperts::complete()` | GPU→CPU |
| 1 | Input norm | CPU `rms_norm` | CPU `rms_norm` | CPU |
| 2 | CMD1: qkvz | GPU `dequant_matvec_4bit` | GPU `encode_matvec_into` | GPU |
| 3 | CMD1: ba | GPU `dequant_matvec_4bit` | GPU `encode_matvec_into` | GPU |
| 4 | CMD1: conv1d | GPU `conv1d` (depthwise, kernel=4) | GPU `conv1d` (depthwise, kernel=4) | GPU |
| 5 | CMD1: SSM | GPU `gated_delta_net_step` | GPU `gated_delta_net_step` | GPU |
| 6 | CMD1: gated norm | CPU (readback + `_precise_swiglu`) | GPU `gated_norm` | **CPU** / GPU |
| 7 | CMD1: out_proj | — (in CMD2) | GPU `encode_matvec_into` (fused into CMD1) | GPU |
| 8 | CMD1: residual add | — (in CMD2) | GPU `residual_add` (fused into CMD1) | GPU |
| 9 | CMD1 wait | `commit` + `wait_until_completed` | `commit` + `wait_until_completed` (single CMD for steps 2–8) | GPU |
| 10 | CMD2: out_proj | GPU `dequant_matvec_4bit` | — (already done in CMD1) | GPU / — |
| 11 | CMD2: residual add | GPU `residual_add` | — (already done in CMD1) | GPU / — |
| 12 | CMD2: post-attn norm | GPU `rms_norm_sum` + `rms_norm_apply` | — (not needed, residual already in CMD1) | GPU / — |
| 13 | CMD2: MoE gate | GPU `dequant_matvec_4bit` | GPU `encode_matvec_into` (router CMD) | GPU |
| 14 | CMD2: shared gate/up | GPU 2× `dequant_matvec_4bit` | GPU 2× `encode_matvec_into` (router CMD) | GPU |
| 15 | CMD2: shared gate score | GPU `dequant_matvec_4bit` | GPU `encode_matvec_into` (router CMD) | GPU |
| 16 | CMD2 wait | `commit` + `wait_until_completed` | `commit` + `wait_until_completed` (router CMD) | GPU |
| 17 | Routing | CPU `softmax` + `topk` | CPU `cpu_softmax` + `cpu_topk` | CPU |
| 18 | Expert I/O | `pread` K experts | Same (`libc::pread`) | Disk→CPU |
| 19 | CMD3: gate/up/down/SwiGLU | GPU (same as full-attn) | GPU (same as full-attn) | GPU |
| 20 | CMD3: shared SwiGLU+down | GPU (same) | GPU (same) | GPU |
| 21 | CMD3: combine | GPU `moe_combine_residual` | GPU `moe_combine_residual` | GPU |
| 22 | CMD3 commit | `commit` (async) | `commit` (async) — returns `DeferredExperts` | GPU |

**Linear-attn diffs:**
- Stage 6: C does gated norm on CPU (readback after SSM); Rust does it on GPU in CMD1
- Stages 7–12: Rust fuses out_proj + residual into CMD1; C keeps them in CMD2
- Net CMD count: C = 3 (CMD1+CMD2+CMD3), Rust = 3 (CMD1+router+CMD3). Same count, different boundaries.

### Summary of Differences

| Diff | Impact |
|------|--------|
| KV cache: CPU (Rust) vs GPU persistent (C) | Rust uploads K/V per layer (~0.03ms/layer overhead) |
| SSM state: CPU↔GPU transfer (Rust) vs GPU persistent (C) | Rust uploads/downloads state per layer |
| Gated norm: GPU (Rust) vs CPU (C) | Rust offloads more work to GPU |
| out_proj+residual: in CMD1 (Rust) vs CMD2 (C) for linear layers | Same total work, different CMD boundaries |
| Speculative routing | C only; Rust doesn't implement expert prediction |
| Expert cache (LRU) | C only; Rust always reads from disk |

## Pipeline Mode: Fused3 — COMPLETE

Fused3 matches the C engine's 3-command-buffer architecture. All steps complete:

| Feature | Status |
|---------|--------|
| Fused CMD1 (linear attention: qkv/z/b/a + conv1d + SSM + gated_norm + out_proj + residual) | Done |
| GPU batched full attention (scores + softmax + values + sigmoid) | Done |
| GPU moe_combine_residual (expert weighted sum + shared expert + residual in one kernel) | Done |
| Async CMD3 (commit without wait, complete on next layer) | Done |
| CMD2 fusion for full-attn (batched attn + o_proj + residual + norm + gate + shared) | Done |
| CMD1 out_proj fusion for linear (out_proj + residual in CMD1, not separate CMD) | Done |
| PyO3 bindings (Context, Cache, telemetry, stream_generate) | Done |
| Ctrl-C interrupt handling (`py.check_signals()`) | Done |

## Sync points per layer

- **Linear attention** (30/40): 2 sync points — CMD1 (fused linear + out_proj + residual) + router CMD + async CMD3
- **Full attention** (10/40): 2 sync points — QKV CMD + CMD2 (batched attn + o_proj + residual + norm + gate) + async CMD3

**Matches C engine: 2 sync points for all 40 layers.**

## Remaining gaps (non-critical)

| Gap | Description |
|-----|-------------|
| GPU KV cache | KV cache stored as CPU f32 buffers, uploaded per layer. C engine stores on GPU persistently |
| GPU RoPE | Q/K norms and RoPE are CPU-side (C engine also does this on CPU) |
| Linear router CMD fusion | Gate + shared projections are a separate CMD in linear layers. Could fuse into CMD1 or CMD2 — would not reduce sync points further |

## Output Coherence

The Rust engine produces coherent, sensible output verified against the same prompt. The CMD2 combine bug (attention contribution lost in full-attn CMD2 path) has been fixed — `temp_buf` (h_mid + attn_out) is now passed through as `hmid_gpu` to the moe_combine_residual kernel.

## Recent Fix: CpuOnly full-attention layer MoE bug (2026-05-21)

**Bug**: In CpuOnly mode, `full_attention_forward` used the GPU attention path (since Metal pipelines were initialized and available) and returned `FullAttnCmd2State` without adding the attention output to `hidden`. Then `moe_layer_forward` in CpuOnly entered the non-fused path (`use_gpu = false`), which saved `h_mid = hidden` (pre-attention) and used it as the residual in CMD3's `moe_combine_residual`. The attention output was completely lost, causing CpuOnly logits to diverge from Fused3 logits by up to 4.88 (max absolute diff).

**Root cause** in `gpu_forward.rs`:
- `full_attention_forward` had no `mode` parameter, so it couldn't distinguish CpuOnly from GPU modes.
- In CpuOnly mode, `use_gpu_attn` was `true` because Metal was available, causing the GPU path (early return with buffers) to be taken instead of the CPU fallback (which adds `o_out` to `hidden` via residual at line 446).
- The non-fused MoE path used `h_mid = hidden` without attention output, causing the divergence.

**Fix** (3 changes in `gpu_forward.rs`, 1 in `python_bindings.rs`):
1. Added `mode: PipelineMode` parameter to `full_attention_forward`
2. Guard `use_gpu_attn` with `mode != PipelineMode::CpuOnly`
3. Guard GPU o_proj in CPU fallback with `mode != PipelineMode::CpuOnly`, adding CPU dequant fallback
4. Pass `mode` from the caller in `python_bindings.rs`

**Verification**: CpuOnly vs Fused3 logits now match within 1e-5 (was 4.88 max diff). 100% of 248,320 vocabulary elements match within 1e-4.

## Verification Methodology

### Quick verification: CpuOnly vs Fused3 on stripped model

```bash
# 1. Build and install the Rust engine
cd moe_infer_rs
maturin build --release --features python-bindings
python -m pip install --force-reinstall target/wheels/moe_infer-0.1.0-*.whl

# 2. Run comparison (CpuOnly vs Fused3, 1 token, stripped 4-layer model)
python -c "
import numpy as np
from moe_infer import Context, Cache

MODEL_DIR = 'hub/models--mlx-community--Qwen3.5-35B-A3B-4bit-stripped'

results = {}
for mode in ['CpuOnly', 'Fused3']:
    ctx = Context()
    ctx.load_model(MODEL_DIR, pipeline_mode=mode)
    cache = ctx.new_cache()
    logits = np.array(ctx.forward(np.array([248045], dtype=np.int64), cache)[-1], dtype=np.float32)
    results[mode] = logits
    ctx.unload_model()

diff = np.abs(results['CpuOnly'] - results['Fused3'])
print(f'max_diff={diff.max():.8f}')
print(f'matching within 1e-3: {(diff < 1e-3).sum()}/{len(diff)}')
assert diff.max() < 1e-3, 'VERIFICATION FAILED'
print('PASS')
"
```

### Full verification: C vs Rust on full model

```bash
# Requires: C bench compiled with --verify-logits patch, full 35B model
python verify.py
# Compares C GPU (Fused3) logits vs Rust GPU (Fused3) logits on 100-token sequence
# Success criterion: max_diff < 1e-3
```

### CPU path correctness verification

Layer-0 linear attention (GatedDeltaNet) CPU path was verified to match a Python CPU reference implementation within 5e-6 across all intermediate steps (norm, qkv/z/b/a projections, conv1d, q/k RMS norms, SSM state update, gated norm, out_proj, residual add). The reference script is at `helpers/compare_cpu_ref.py`.

## Python API

```python
from moe_infer import Context, Cache

ctx = Context()
ctx.load_model("/path/to/model", pipeline_mode="Fused3")
cache = ctx.new_cache()

# Forward / generate / stream
logits = ctx.forward(input_ids, cache)
new_ids = ctx.generate(input_ids, cache, max_tokens=256, temperature=0.7)
results = ctx.stream_generate(input_ids, cache, max_tokens=256)

# Telemetry
info = ctx.telemetry()
# {"ttft_ms": ..., "tokens_per_sec": ..., "tokens_generated": ...}
```

## Model Stripping (`helpers/strip_model.py`)

For fast-iteration verification, we created a stripped version of the Qwen3.5-35B-A3B-4bit model with 4 layers and 4 experts (down from 40 and 256).

### How it works

`helpers/strip_model.py` reads the full MLX safetensors model and writes a smaller one:

1. **Config**: Updates `config.json` — sets `num_hidden_layers=4`, `num_experts=4`, truncates `layer_types` to 4 entries
2. **Layers kept**: Layers 0–3 (preserves the 1-full-attn-per-4-layers pattern — layer 3 is full attention)
3. **Expert slicing**: For tensors under `mlp.gate.*`, `mlp.switch_mlp.*`, or `mlp.experts.*`, the first dimension (num_experts) is sliced from 256 → 4. All 4 expert weight sets are retained per layer
4. **Non-expert weights**: All layers-0–3 weights (attention, norms, shared experts, etc.) are kept verbatim
5. **Output**: Single `model.safetensors` file (680 MB → ~16 MB per expert, ~650 MB total)

```bash
python helpers/strip_model.py \
    --input hub/models--mlx-community--Qwen3.5-35B-A3B-4bit \
    --output hub/models--mlx-community--Qwen3.5-35B-A3B-4bit-stripped \
    --num-layers 4 --num-experts 4
```

### Stripped model layout

```
stripped/
├── config.json                    # 4 layers, 4 experts
├── model.safetensors              # MLX-LM format (single file)
├── model.safetensors.index.json
├── model_weights.bin              # C-engine format (generated by helpers/dump_mlx_intermediates.py)
├── model_weights.json             # Tensor manifest (offsets into .bin)
├── packed_experts/                # Per-layer expert files (layer_00.bin .. layer_03.bin)
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

### 3-Way Verification on Stripped Model

We compared C bench, Rust Fused3, and MLX-LM on the stripped model (BOS token, greedy):

| Pair | Max Diff | Within 1e-3 |
|------|----------|-------------|
| C vs Rust | 0.000005 | 100% |
| C vs MLX-LM | 0.109 | 3.36% |
| Rust vs MLX-LM | 0.109 | 3.36% |

**C and Rust are numerically identical.** Both diverge from MLX-LM by ~0.02 mean / ~0.11 max. This means the C→Rust port is faithful, but the C engine has a pre-existing discrepancy from the MLX-LM reference. The root cause is still under investigation — likely candidates are dequantization precision, norm epsilon handling, or SSM state initialization.

## Performance Benchmark

Ran on Apple M4 (unified memory), Qwen3.5-35B-A3B-4bit full model (40 layers, 256 experts, hidden=2048), K=8 experts, 32-token prompt, 100-token greedy generation.

| Metric | C | Rust | Ratio |
|--------|---|------|-------|
| TTFT (prefill 32 tok) | 10,722 ms | 16,357 ms | 1.53x slower |
| Generation speed | 3.36 tok/s | 2.14 tok/s | 0.64x |
| Init time | — | 33 ms | — |

### C per-layer breakdown (avg 7.25 ms)

| Phase | ms | % |
|-------|-----|---|
| expert_io (disk read) | 5.194 | 71.7% |
| cmd1_wait (GPU linear attn) | 1.403 | 19.4% |
| cmd2_wait (GPU full attn) | 0.549 | 7.6% |
| cmd3_encode | 0.047 | 0.6% |
| cmd1_submit | 0.024 | 0.3% |
| cmd2_encode | 0.020 | 0.3% |
| routing_cpu | 0.003 | 0.0% |
| deferred_cpu | 0.002 | 0.0% |

Expert I/O dominates at 72% of per-layer time — this is `pread` from disk for expert weight files.

### Rust slowdown analysis

Rust is ~1.5x slower on prefill and ~1.6x slower on generation. Likely causes (needs profiling):

1. **KV cache upload/download**: Rust stores KV caches on CPU and uploads/downloads per layer. C stores them persistently on GPU.
2. **SSM state transfer**: Linear attention SSM states are transferred CPU↔GPU per layer in Rust.
3. **Debug prints**: `[RUST-PRE]`/`[RUST-LAYER]` stderr prints on every token×layer add overhead.
4. **No Metal command buffer reuse**: Rust creates new command buffers per layer rather than reusing.

## Next Steps

1. **GPU KV cache** — store K/V caches persistently on GPU instead of uploading per layer
2. **GPU RoPE kernel** — port C `apply_rope` shader (stretch goal; C engine also does RoPE on CPU)
3. **GPU SSM state** — linear attention SSM state currently uploaded/downloaded per layer in non-fused path

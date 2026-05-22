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
| 1 | Input norm | **GPU** (CMD3 already put normed input in buf_input) | CPU `rms_norm` (then uploaded to GPU) | **GPU** / CPU |
| 2 | CMD1 L0: qkvz/ba | GPU `dequant_matvec_4bit` (from buf_input) | GPU `encode_matvec_into` (from uploaded buffer) | GPU |
| 3 | CMD1 L1: conv1d | GPU `conv1d_step` | GPU `conv1d_step` | GPU |
| 4 | CMD1 L2: rms_norm_qk | GPU `rms_norm_qk` | GPU `rms_norm_qk` | GPU |
| 5 | CMD1 L3: decay_beta | GPU `compute_decay_beta` | GPU `compute_decay_beta` | GPU |
| 6 | CMD1 L4: SSM | GPU `gated_delta_net_step` | GPU `gated_delta_net_step` | GPU |
| **7** | **CMD1 L5: gated_rms_norm** | **GPU `gated_rms_norm` → batch_out[6]** | **MISSING** — done on CPU after CMD1 | **GPU** / CPU |
| 8 | CMD1 wait | `commit` + `wait_until_completed` | `commit` + `wait_until_completed` | GPU |
| 9 | Post-CMD1 | Data stays in GPU (batch_out[6]) | **CPU readback** SSM output + z (16K floats), **CPU gated_norm**, re-upload gated_out + h_mid for CMD2 (~26K floats memcpy) | CPU+GPU |
| 10 | CMD2 Enc 1: o_proj | GPU `matvec_fast` (batch_out[6] → buf_output) | GPU `encode_matvec_into` (uploaded gated_buf → o_proj_buf) | GPU |
| 11 | CMD2 Enc 2: residual_add | GPU `residual_add` (buf_output + buf_residual → buf_h_mid) | GPU `residual_add` (o_proj_buf + hmid_buf → temp_buf) | GPU |
| 12 | CMD2 Enc 3: rms_norm_sum | GPU `rms_norm_sum` | GPU `rms_norm_sum` | GPU |
| 13 | CMD2 Enc 4: rms_norm_apply | GPU `rms_norm_apply_bf16` → buf_input | GPU `rms_norm_apply_bf16` → normed_buf | GPU |
| 14 | CMD2 Enc 5-8: routing | GPU `batch_matvec` × 4 (gate, sg, su, seg) | GPU `encode_matvec_into` × 4 (gate, sg, su, seg) | GPU |
| 15 | CMD2 wait | `commit` + `wait_until_completed` | `commit` + `wait_until_completed` | GPU |
| 16 | CMD2 readback | routing results + h_mid + h_post | routing results + normed_buf → hidden, h_post | GPU→CPU |
| 17 | Routing | CPU `softmax` + `topk` | CPU `cpu_softmax` + `cpu_topk` | CPU |
| 18 | Expert I/O | `pread` K experts with LRU cache | Same (`libc::pread` with LRU cache) | Disk→CPU |
| 19 | CMD3: gate/up/down/SwiGLU | GPU K× expert matvecs | GPU K× expert matvecs | GPU |
| 20 | CMD3: shared SwiGLU+down | GPU shared SwiGLU + down_proj | GPU shared SwiGLU + down_proj | GPU |
| 21 | CMD3: combine | GPU `moe_combine_residual` | GPU `moe_combine_residual` | GPU |
| 22 | CMD3: input_norm (next layer) | GPU — norm of combine output → buf_input | **MISSING** | GPU / — |
| 23 | CMD3 commit | `commit` (async, **no wait**) | `commit` (async) — returns `DeferredExperts` | GPU |

### Summary of Differences (as of 2026-05-22)

| # | Diff | Impact |
|---|------|--------|
| 1 | **CMD1 L5 (gated_rms_norm) on CPU vs GPU** | C does gated_rms_norm in CMD1 on GPU (encoder L5, output to batch_out[6]). Rust Fused3 omits L5 from CMD1, reads back SSM output + z to CPU (~16K floats), does CPU gated_norm, then re-uploads gated_out + h_mid for CMD2 (~26K floats). Extra GPU↔CPU round trip per linear layer. |
| 2 | **No GPU-side input_norm for next layer** | C's CMD3 computes input_norm of the combine output into buf_input, so the next layer's CMD1 reads directly from GPU. Rust always does CPU input_norm at the start of each linear_attention_forward, adding another GPU↔CPU round trip. |
| 3 | KV cache: CPU (Rust) vs GPU persistent (C) | Rust uploads K/V per layer (~0.03ms/layer overhead) |
| 4 | Speculative routing | C only; Rust doesn't implement expert prediction |
| 5 | Expert cache (LRU) | Both have LRU cache + parallel pread now (Rust added 2026-05-22) |

## Pipeline Mode: Fused3 — CURRENT STATE

Fused3 matches the C engine's 3-command-buffer architecture but with pipeline differences in CMD1/CMD2 boundaries for linear-attention layers.

| Feature | Status |
|---------|--------|
| Fused CMD1 (linear attention: qkv/z/b/a + conv1d + SSM) | Done |
| **CMD1 L5: gated_rms_norm on GPU** | **MISSING** — done on CPU after CMD1 (see Discrepancy #1) |
| GPU batched full attention (scores + softmax + values + sigmoid) | Done |
| CMD2 fusion (o_proj + residual_add + rms_norm + gate + shared) | Done |
| GPU moe_combine_residual (expert weighted sum + shared expert + residual in one kernel) | Done |
| Async CMD3 (commit without wait, complete on next layer) | Done |
| CMD3 GPU-side input_norm for next layer | **MISSING** — always done on CPU (see Discrepancy #2) |
| Expert LRU cache + parallel pread | Done |
| PyO3 bindings (Context, Cache, telemetry, stream_generate) | Done |
| Ctrl-C interrupt handling (`py.check_signals()`) | Done |

## Sync points per layer

- **Linear attention** (30/40): 3 sync points — CMD1 (linear SSM, no gated_norm) + CMD2 (o_proj + residual + norm + routing) + async CMD3
  - C has same 3 sync points but CMD1 includes gated_rms_norm, keeping data on GPU
- **Full attention** (10/40): 2 sync points — QKV CMD + CMD2 (batched attn + o_proj + residual + norm + gate) + async CMD3
  - Matches C exactly

## Key discrepancies vs C (to fix)

| # | Discrepancy | Fix |
|---|------------|-----|
| 1 | CMD1 missing L5 (gated_rms_norm on GPU) — Rust does it on CPU with GPU↔CPU round trip | Add gated_rms_norm as encoder L5 in CMD1, write to a GPU buffer (like C's batch_out[6]), remove CPU readback + CPU gated_norm |
| 2 | No GPU-side input_norm for next layer in CMD3 | Add input_norm compute to CMD3 (norm of combine output → buf_input for next layer's CMD1) |

## Remaining gaps (non-critical)

| Gap | Description |
|-----|-------------|
| GPU KV cache | KV cache stored as CPU f32 buffers, uploaded per layer. C engine stores on GPU persistently |
| GPU RoPE | Q/K norms and RoPE are CPU-side (C engine also does this on CPU) |
| Speculative routing | C predicts experts with pre-attention normed input, does async pread in parallel with attention compute |

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

**C and Rust are numerically identical.** Both diverge from MLX-LM by ~0.02 mean / ~0.11 max. This means the C→Rust port is faithful, but the C engine has a pre-existing discrepancy from the MLX-LM reference.

## Rust Cpu vs MLX-LM: Divergence Investigation

### Bug Found: RoPE Element Pairing (2026-05-22)

**Root cause**: `apply_rope()` in `moe_infer_rs/src/gpu_forward.rs` used traditional consecutive element pairing instead of NeoX-style non-consecutive pairing. MLX uses `nn.RoPE(dims, traditional=False)` which implements NeoX-style: elements at `(i, i + dims/2)` are rotated as a pair.

#### Before (broken)

```rust
// Traditional: pairs (0,1), (2,3), (4,5), ...
for d in (0..rotary_dim).step_by(2) {
    let theta = pos * rope_theta.powf(-2.0 * d / rotary_dim);
    let (q0, q1) = (qh[d], qh[d + 1]);
    qh[d] = q0 * cos - q1 * sin;
    qh[d + 1] = q0 * sin + q1 * cos;
}
```

#### After (fixed)

```rust
// NeoX-style (traditional=False): pairs (0, half), (1, half+1), ...
let half = rotary_dim / 2;
for i in 0..half {
    let theta = pos * rope_theta.powf(-2.0 * i / rotary_dim);
    let (q0, q1) = (qh[i], qh[i + half]);
    qh[i] = q0 * cos - q1 * sin;
    qh[i + half] = q0 * sin + q1 * cos;
}
```

### Impact on 29-Token Batch Verification

Verification is run with 29 tokens through the stripped 4-layer model. Per-layer hidden states are captured from both Rust Cpu and MLX (via class-level `DecoderLayer.__call__` monkey-patch) and compared.

#### Before RoPE Fix

| Metric | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Logits |
|--------|---------|---------|---------|---------|--------|
| max_diff | 0.006 | 0.12 | 0.80 | 2.03 | 0.835 |
| cos_sim | 0.99999 | 0.99997 | 0.9980 | 0.985 | 0.99823 |
| pass 1e-4? | NO | NO | NO | NO | NO |

#### After RoPE Fix

| Metric | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Logits |
|--------|---------|---------|---------|---------|--------|
| max_diff | 1.9e-03 | 1.6e-03 | 2.7e-03 | 2.2e-03 | 0.113 |
| cos_sim | 0.99999 | 0.99999 | 0.99999 | 1.00000 | 0.99996 |
| pass 1e-4? | NO | NO | NO | NO | **NO** |

**Verdict: FAIL.** Verification is binary — either it matches or it doesn't. Despite the dramatic improvement (7.4x reduction in logit max_diff, cos_sim now 0.99996), Cpu still does not match MLX-LM within the 1e-4 threshold. The divergence is reduced but not eliminated.

### Remaining Divergence After RoPE Fix

1. **Linear attention layers (0–2)**: max_diff ~1.6e-03 to ~2.7e-03. These layers use GatedDeltaNet SSM — the SSM state accumulates error over 29 tokens from 4-bit dequant precision differences, conv1d precision, and floating-point order differences in SSM recurrence.

2. **Full attention layer (3)**: max_diff ~2.2e-03. Attention softmax and value aggregation are sensitive to element ordering.

3. **Logits show 0.113 max_diff but 0.99996 cos_sim**: The large max_diff concentrated at a single token index (1269) suggests a systematic bias in one output dimension rather than random noise. This position is sensitive to accumulated per-layer differences across all 29 tokens.

4. **Candidates for residual divergence**:
   - 4-bit dequant order: float32 vs float64 accumulation order in `nibble * scale + bias`
   - RMS norm epsilon: 1e-6 in Rust, possibly different in MLX
   - SSM recurrence: `g = exp(-exp(A_log) * softplus(a + dt_bias))` — small errors compound over 29 steps
   - Attention softmax: numerical order differences in exp-sum-reciprocal chain

## Performance Benchmark (2026-05-22)

Ran on Apple M4 (unified memory), Qwen3.5-35B-A3B-4bit full model (40 layers, 256 experts, hidden=2048), K=8 experts, 32-token prompt, 100-token greedy generation.

| Metric | C | Rust (pre-opt) | Rust (post expert-IO opt) | Ratio vs C |
|--------|---|----------------|--------------------------|------------|
| Generation speed | 2.82 tok/s | 2.14 tok/s | 2.69 tok/s | 0.96x |
| Expert I/O (disk read) | ~5.2 ms/layer | — | ~5.8 ms/layer | — |

Expert I/O dominates at 72% of per-layer time in C.

### Expert I/O optimizations applied to Rust (2026-05-22)

1. **Parallel 4-thread `pread`** via `rayon::scope` — matches C's `io_pool_dispatch` (4 pthreads)
2. **LRU expert cache** (32 entries) — avoids re-reading experts that repeat across tokens
3. **Pre-allocated Metal buffers** — reused across all layers instead of per-layer `metal_buf_shared()` calls
4. **CMD3 deferred commit** — CMD3 committed without wait, completed at start of next layer for GPU/CPU overlap

Result: Rust went from 0.76x to 0.96x of C throughput after expert I/O optimization.

## Next Steps

1. **Fix remaining Cpu vs MLX-LM divergence** — RoPE is fixed but residual differences remain in 4-bit dequant precision, SSM recurrence, and attention softmax. Track down the remaining 0.113 logit max_diff.
2. **GPU KV cache** — store K/V caches persistently on GPU instead of uploading per layer
3. **GPU RoPE kernel** — port C `apply_rope` shader (stretch goal; C engine also does RoPE on CPU)
4. **GPU SSM state** — linear attention SSM state currently uploaded/downloaded per layer in non-fused path

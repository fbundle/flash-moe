# MoE-Infer: Technical Report

## Overview

MoE-Infer is a high-performance inference engine for Mixture-of-Experts models on Apple Silicon. It streams expert weights from SSD on demand (with optional LZ4 compression), uses hand-tuned Metal compute shaders for all GPU operations, and exposes Python bindings via PyO3. No Python ML frameworks at runtime — just Rust, Metal, and ~0.65 GB of mmap'd weights.

**Supported models**: `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.6-35B-A3B-4bit`

**Hardware**: Apple Silicon (M1–M4) with unified memory. Tested on M1 Pro (10-core CPU, 14-core GPU).

## Architecture

### Model Structure

Qwen3.5-35B-A3B-4bit: 40 layers, 256 experts, K=8 active experts per token.

| Parameter | Value |
|-----------|-------|
| Hidden dim | 2048 |
| Vocab size | 248,320 |
| Layers | 40 (30 linear attention + 10 full attention, every 4th layer) |
| Experts | 256 (8 active per token) |
| Expert intermediate | 512 |
| Shared expert intermediate | 512 |
| Linear attention | 16 K-heads (dim 128), 32 V-heads (dim 128), conv kernel 4 |
| Full attention | 16 Q-heads (dim 256), 2 KV-heads (dim 256), RoPE dim 64 |
| Quantization | 4-bit affine (group_size=64), nibble * scale + bias |
| Weight format | U32 packed weights + BF16 scales/biases |

### Data Flow (per token, per layer)

```
Input → RMS Norm → Attention (linear or full) → Residual Add
  → Post-Attention Norm → MoE Gate → Top-K Routing
  → Expert I/O (SSD pread) → Expert Matvecs (Gate/Up, SwiGLU, Down)
  → Shared Expert (SwiGLU + Down) → MoE Combine + Residual → Output
```

### Expert I/O

Expert weights (~19 GB 4-bit) live on SSD in per-layer files (`packed_experts/layer_NN.bin`). Only K=8 active experts are read per layer (~1.77 MB each) via parallel `pread()` across 4 threads, with an LRU cache (512 entries) to avoid re-reading repeated experts.

**LZ4 compression** (optional): `helpers/compress_experts_lz4.py` compresses the per-layer expert files with LZ4, reducing total expert size by ~40-55%. The engine auto-detects `packed_experts_lz4/` at load time and transparently decompresses on read via `lz4_flex`. This is a drop-in replacement for the raw packed files and reduces SSD bandwidth by roughly 30-50%. Both `ExpertFile::Raw` and `ExpertFile::Lz4` variants share the same `read_expert()` interface.

#### Why `pread()` and not `mmap()`

Non-expert weights (0.65 GB) use `mmap()` — they fit in memory and are accessed every layer, every token. Experts (19 GB, 30× larger) use `pread()` into pre-allocated Metal buffers. The reasoning:

**1. DMA alignment (3.6× speedup).** Expert data buffers are allocated with `posix_memalign(..., 2MB)` and wrapped via `newBufferWithBytesNoCopy`. The DMA controller that handles `pread()` from Apple's SSD achieves 3.6× higher throughput with 2 MB alignment vs the 16 KB page alignment that `mmap()` guarantees. This is the single biggest factor.

**2. One syscall, not 110 page faults.** A 1.77 MB expert spans ~110 pages (16 KB each on Apple Silicon). With `mmap()`, the first access to each page triggers a synchronous kernel trap: page fault → I/O dispatch → TLB fill. That's 110 individual round-trips through the kernel. With `pread()`, the kernel reads the entire blob in a single efficient I/O operation — one syscall, one I/O submission, one completion.

**3. Double-buffering for prediction preads.** The engine uses an A/B buffer pair. While the GPU processes expert A's results, prediction preads fill the B buffer for the next layer. `mmap()` can't provide independent buffer copies — it's a single mapping. The double-buffer scheme is essential for overlapping I/O with compute.

**4. Explicit eviction control.** The LRU cache (512 entries) decides which experts stay resident based on application-level routing patterns. With `mmap()` + memory pressure, the kernel's page reclaimer makes that decision instead — and it has no knowledge of MoE routing. Under the wrong access pattern, the kernel evicts the wrong pages and thrashing results. With `pread()`, eviction is deterministic and application-controlled.

**5. Scale mismatch.** Non-expert weights (0.65 GB) are small enough to mmap once at startup and keep resident forever — the `newBufferWithBytesNoCopy` Metal buffer wrapping the mmap'd region is valid for the lifetime of the process. Experts (19 GB) can't be kept resident alongside KV caches, activations, and scratch buffers. `pread()` is the correct primitive for "read this blob, use it on GPU, discard it."

### Metal Compute Pipeline

All matrix-vector multiplies run on GPU via Metal compute shaders:
- **4-bit dequant matvec**: `nibble * scale + bias` fused with dot product via FMA
- **Fused linear attention (CMD1)**: QKV/Z/B/A projections + Conv1d + Q/K RMS norms + SSM state update + Gated RMS norm — single command buffer
- **Fused full attention (CMD2)**: QKV projections + Q/K norms + RoPE (CPU) + Batched attention (scores, softmax, values) + Sigmoid gate + o_proj + Residual add + Post-attn norm + MoE gate + Shared expert projections — single command buffer
- **Expert dispatch (CMD3)**: K expert gate/up + SwiGLU + down + Shared expert SwiGLU + down + MoE combine + residual — async commit, completed next layer

## Weight File Format

MoE-Infer uses a custom binary weight format optimized for mmap and pread, converted from the HuggingFace/MLX safetensors format. The conversion is done by helper scripts in `helpers/`.

### HF/MLX Format (Input)

The source model is stored in the standard MLX-quantized safetensors layout:

- **Multiple `.safetensors` files** with a `model.safetensors.index.json` index mapping tensor names to shard files.
- **4-bit affine quantization**: weights stored as nibble-packed U32 arrays `[out_dim, in_dim/8]`, scales and biases as BF16 arrays `[out_dim, in_dim/64]`. Group size is 64.
- **Expert tensors** use 3D shapes: `[num_experts, out_dim, packed_in_dim]` for each of gate_proj, up_proj, down_proj (weight + scales + biases = 9 tensors per layer).
- **Tensor naming**: `language_model.model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}` for experts; `language_model.model.layers.N.self_attn.*` for full attention; `language_model.model.layers.N.linear_attn.*` for linear attention.
- **Gate tensors** (router and shared expert gate) may use 8-bit quantization (`{weight,scales,biases}` with INT8 dtype) on newer models (Qwen3.6+). These are dequantized and kept as BF16 during extraction.

### MoE-Infer Non-Expert Weights

Single mmap'd file `model_weights.bin` + JSON manifest `model_weights.json`. Produced by `helpers/extract_weights.py`.

**`model_weights.bin`**: All non-expert tensors packed contiguously with 64-byte alignment. Each tensor stored in its native format (U32 packed for 4-bit, BF16 for scales/biases, F32 for norms). The file is mmap'd at startup for zero-copy GPU access via `newBufferWithBytesNoCopy`.

**`model_weights.json`**: Manifest mapping sanitized tensor names to `{offset, size, shape, dtype}`. Also includes a `config` block with all model dimensions (hidden_size, num_layers, head counts, MoE params, etc.) and per-layer types. The Rust engine uses this to resolve tensors by name at runtime.

Key differences from HF format:
- **Single file** vs multi-shard: all non-expert tensors in one contiguous binary.
- **Name sanitization**: `language_model.model.layers.N.X` → `model.layers.N.X`; `language_model.lm_head` → `lm_head`.
- **8-bit gate tensors**: Router gate and shared expert gate should remain BF16 (not 4-bit quantized) for routing precision. These represent <0.25% of total parameters.
- **Excluded**: vision tower weights, expert tensors (stored separately), and MTP (Multi-Token Prediction) expert layers.

### Dtype mappings

Weights are stored in three dtypes, each chosen for its precision/throughput tradeoff:

| Dtype | Used for |
|-------|----------|
| **U32 (packed int4)** | All `nn.Linear` weight matrices: attention Q/K/V/O projections, expert gate/up/down, shared expert. 8 nibbles per u32, dequantized on the fly by the Metal shader: `nibble * scale + bias` |
| **BF16 (u16)** | **(a)** Scales and biases for every 4-bit weight (one pair per group of 64 weights). **(b)** RMS norm weights (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `model.norm`). Cast to f32 at runtime via `bf16_to_f32`. Scale/biases never leave the Metal buffer — the shader reads them directly. **(c)** Routing gate and shared expert gate (kept BF16 for precision) |
| **F32** | Embedding (`embed_tokens`), output head (`lm_head`), and SSM decay parameter (`A_log`). Embeddings and lm_head stay f32 to avoid accumulating precision loss at the pipeline boundaries. `A_log` is exponentiated at runtime, which amplifies BF16 round-off error |

### MoE-Infer Expert Weights

Per-layer flat binary files `packed_experts/layer_NN.bin`. Produced by `helpers/repack_experts_4bit.py`.

Each layer file is a concatenation of expert weight blobs:

```
[expert_0][expert_1]...[expert_{num_experts-1}]
```

Where each expert blob is:

```
gate_proj.weight (U32)  gate_proj.scales (BF16)  gate_proj.biases (BF16)
up_proj.weight   (U32)  up_proj.scales   (BF16)  up_proj.biases   (BF16)
down_proj.weight (U32)  down_proj.scales (BF16)  down_proj.biases (BF16)
```

Sizes for a typical expert (hidden_dim=2048, moe_intermediate=512, group_size=64):

| Component | Dims | Bytes |
|-----------|------|-------|
| gate_proj.weight | 512 × 256 U32 | 524,288 |
| gate_proj.scales | 512 × 32 BF16 | 32,768 |
| gate_proj.biases | 512 × 32 BF16 | 32,768 |
| up_proj.weight | 512 × 256 U32 | 524,288 |
| up_proj.scales | 512 × 32 BF16 | 32,768 |
| up_proj.biases | 512 × 32 BF16 | 32,768 |
| down_proj.weight | 2048 × 64 U32 | 524,288 |
| down_proj.scales | 2048 × 8 BF16 | 32,768 |
| down_proj.biases | 2048 × 8 BF16 | 32,768 |
| **Total per expert** | | **~1.77 MB** |

Key differences from HF format:
- **Per-layer files** vs 3D tensors in multi-shard safetensors: the 3D `[num_experts, out_dim, packed_in_dim]` arrays are sliced by expert and repacked into flat per-expert layouts.
- **Flat binary** vs safetensors container: no JSON header, no tensor metadata — just raw concatenated blobs. Offsets are known from compile-time constants in the `ModelConfig` trait.
- **pread-friendly**: fixed-size records at known offsets enable direct `pread(expert_id * expert_size)` from SSD without parsing.

### Model Config

The engine reads HF `config.json` directly at runtime via `model::config::load_model_config()`. All dimensions and expert layout offsets are derived from HF fields — no intermediate config format needed.

Derived fields computed at load time:
- `expert_size_4bit` / `expert_size_2bit`: total bytes per expert in the packed layer files.
- `expert_layout_4bit` / `expert_layout_2bit`: byte offsets within each expert blob for each projection's weight/scales/biases.
- `rotary_dim`, `linear_total_key`, `linear_total_value`, `linear_conv_dim`: derived from head counts and dimensions.
- `num_full_attn_layers`, `num_linear_layers`: computed from `num_layers` and `full_attention_interval`.

### Conversion Pipeline

```
HF config.json ──► copied directly ──► config.json

HF safetensors/ ──► helpers/extract_weights.py ──► model_weights.bin
                                                   model_weights.json

HF safetensors/ ──► helpers/repack_experts_4bit.py ──► packed_experts/layer_00.bin
                                                       packed_experts/layer_01.bin
                                                       ...

              ┌──► helpers/compress_experts_lz4.py ──► packed_experts_lz4/
              │                                       (optional, ~40-55% compression)
              │
              └──► helpers/repack_experts_2bit.py   ──► packed_experts_2bit/
                                                      (experimental, 2-bit quant)
```

All scripts read from the same MLX-format model directory and output to a single MoE-Infer model directory. The conversion is a one-time offline step; at inference time only the binary files are needed.

`helpers/convert.py` automates the entire pipeline with a single command. `helpers/quantize_from_hf.py` converts directly from HuggingFace unquantized models. `helpers/strip_model.py` builds a small 4-layer model for fast verification iteration.

### Cache Format

KV cache and linear attention state are stored in the same flat binary + JSON manifest format as model weights:

```
cache.bin       # Flat concatenation of all cache tensors (f32 + u32 scalars)
cache.json      # Manifest: name → {offset, size, shape, dtype}
```

Full-attention layers store `k_cache`, `v_cache`, and `len`. Linear-attention layers store `conv_state` and `ssm_state`. The sequence position `pos` is a u32 scalar.

## Pipeline Modes

| Mode | Description |
|------|-------------|
| `Fused4bit` | Full model: 40 layers, 256 experts, K=8. 3-CMD GPU pipeline with expert dispatch every layer |
| `Fused4bitStripped` | Stripped model: 4 layers, 4 experts, K=4. For verification |
| `Cpu` (Rust only) | Pure-CPU reference engine using `ndarray`. Not exposed via Python bindings |

All GPU modes use the 3-CMD pipeline. The stripped variant uses a reduced 4-layer 4-expert model suitable for fast verification iteration.

### Fused4bit Command Buffer Layout

**Linear attention layers (30/40)**:
- CMD1: QKV/Z/B/A projections → Conv1d → Q/K RMS norms → SSM → Gated RMS norm → out_proj → Residual add
- CMD2: Post-attn norm → Gate + Shared expert projections + Shared expert gate
- CMD3 (async): Expert gate/up + SwiGLU + down × K → Shared SwiGLU + down → MoE combine + residual → Input norm for next layer

**Full attention layers (10/40)**:
- CMD1: Q/K/V projections → Q/K norms → RoPE (CPU) → KV cache append
- CMD2: Batched attention (scores, softmax, values) → Sigmoid gate → o_proj → Residual add → Post-attn norm → Gate + Shared expert projections
- CMD3 (async): Same as linear, plus explicit input norm for next layer

### CPU Engine

The `CpuEngine<C: ModelConfig>` in `engine/qwen35_moe/cpu.rs` is a pure-CPU reference implementation using `ndarray::Array1<f32>`. All computation is in f32. It follows the same data flow as the GPU pipeline:

- `pre_expert_full()`: input_layernorm → QKV projections → Q/K head norm + RoPE → KV cache append → attention (scores/softmax/values) → sigmoid gate → o_proj → residual add → post_attention_layernorm → gate projections
- `pre_expert_linear()`: input_layernorm → QKV/Z/B/A projections → conv1d_step with state update → RMS norm Q/K → decay/beta → gated delta net → gated RMS norm → out_proj → residual add → post_attention_layernorm → gate projections
- `post_expert()`: dequant_matvec_4bit + swiglu per expert → shared expert swiglu + down → sigmoid-gated residual combine

The CPU engine serves as a numerical reference for verifying the GPU pipeline, and runs at ~0.15 tok/s (vs ~10 tok/s for Fused4bit on M1 Pro).

## Performance

Benchmarked on M1 Pro (10-core CPU, 14-core GPU), Qwen3.5-35B-A3B-4bit full model (40 layers, 256 experts, K=8), 32-token prompt, 100-token greedy generation:

| Mode | tok/s |
|------|-------|
| Fused4bit (Rust) | ~10 |
| Cpu (reference) | ~0.15 |

Expert I/O (SSD reads) dominates at ~70% of per-layer time.

### Fused4bit per-phase telemetry (full model, 20 tokens, prompt prefill)

| Stage | Mean (ms) | Share |
|-------|-----------|-------|
| Wall time | 1996 | — |
| engine.expert_io_ms | 671 | 33.6% |
| engine.full_attention_layer | 1.8 | <0.1% |
| engine.linear_group | 6.9 | 0.3% |
| engine.total_ms | 1996 | — |

Expert I/O (SSD pread) dominates at ~34% of wall time for short prompts and ~43% for 100-token prompts. Full-attention and linear SSM GPU compute are negligible (<10 ms total).

## Numerical Verification

### CPU vs MLX-LM (Stripped 4-Layer Model)

All verification uses the stripped model (4 layers, 4 experts) to enable fast iteration.

**Algorithmic bugs found and fixed**:
1. **RoPE element pairing** (2026-05-22): `apply_rope()` used traditional consecutive pairs (d, d+1) instead of NeoX-style pairs (i, i + dims/2) used by MLX's `nn.RoPE(traditional=False)`. Fix reduced logit max_diff from 0.835 to 0.113 (7.4× improvement).

2. **Full-attention MoE bug** (2026-05-21): In the CPU engine, the full-attention path returned early without adding attention output to hidden, causing MoE to use pre-attention hidden as residual. Attention contribution was lost (max_diff 4.88).

3. **conv_state not updated** (2026-05-24): conv1d_step was called but conv_state was never shifted/updated for the next token — would produce incorrect results for multi-token sequences. Fixed by adding shift-and-append logic after conv1d_step.

**Per-operation verification** (Layer 0, token 0): Every intermediate tensor in GatedDeltaNet compared between Rust f32 and MLX bf16. All operations match within bf16 precision limits (~0.4% relative). No remaining algorithmic bugs.

**Current state**: After all fixes, max logit diff = 0.113, cos_sim = 0.99996. The residual divergence is entirely attributable to bf16 vs f32 precision differences across ~40 operations per token. Per-layer hidden state error is bounded at ~2e-3 and does not grow across layers. The lm_head projection (2048 → 248320) amplifies this to the observed 0.113 logit max_diff.

### GPU vs CPU

CpuEngine and Fused4bit are numerically identical (max_diff < 1e-5, within ULP-level tolerance). The CPU engine uses `ndarray` and f32 throughout, providing a trustworthy reference for the Metal GPU pipeline.

#### Performance Differences (GPU vs CPU)

1. **Expert prediction**. During CMD1 wait, the engine can predict the next layer's experts (based on the previous token's routing) and start async `pread()` into the B buffer. By the time CMD3 runs, if prediction was correct (experts often repeat between consecutive tokens), the I/O is already done. Not yet implemented in Rust.

2. **LZ4-compressed experts**. The engine auto-detects `packed_experts_lz4/` at model load time and transparently decompresses via `lz4_flex`.

3. **2-bit quantization**. The `dequant_matvec_2bit` Metal shader exists and `model/config.rs` computes 2-bit layout offsets, but no engine code path dispatches it yet. `helpers/repack_experts_2bit.py` generates `packed_experts_2bit/` files. Expert size drops from ~1.77 MB to ~1.44 MB at the cost of ~4% relative logit error (pilot experiments).

4. **Persistent I/O thread pool**. Uses `rayon::scope()` for parallel expert reads. On macOS, GCD may have lower wake-up latency due to kernel integration.

5. **Full attention GPU gating**. Currently unconditionally uses GPU attention. For very short sequences (seq_len < 32), CPU attention may be faster because GPU encoder overhead dominates.

6. **Expert dispatch encoding pattern**. Currently puts all K experts in a single encoder (4 dispatches each), serializing all expert work within CMD3. Splitting across multiple encoders could allow the Metal driver to overlap gate/up of one expert with SwiGLU/down of another.

## Key Design Decisions

1. **SSD expert streaming over GPU preloading**: Expert weights are too large (~19 GB) for unified memory alongside KV caches and activations. On-demand SSD reads with LRU caching are the pragmatic choice.

2. **CPU KV cache**: KV caches stored as CPU f32 buffers rather than GPU persistent buffers. Adds ~0.03 ms/layer upload overhead but simplifies memory management.

3. **CPU RoPE**: RoPE rotations computed on CPU. The rotary dimension is only 64 elements per head — GPU kernel launch overhead exceeds CPU compute time.

4. **Single mmap for non-expert weights**: All 0.65 GB of non-expert weights (embeddings, norms, attention projections, shared experts, gates) in one mmap'd file. Zero-copy GPU access via `newBufferWithBytesNoCopy`.

5. **Per-layer expert files**: Each layer's 256 experts in a separate file (`packed_experts/layer_NN.bin`). Enables `pread()` with offset — no seeking needed.

6. **All compute in f32**: While weights are stored in 4-bit + BF16, all math on both CPU and GPU runs in f32. This avoids precision accumulation issues while keeping memory/IO footprint small.

7. **File-based module convention**: No `mod.rs` files — Rust module declarations use `#[path]` attributes. The `qwen35_moe/` directory lives alongside `qwen35_moe.rs`, which declares its submodules with explicit `#[path = "qwen35_moe/foo.rs"]` attributes.

## Known Limitations

1. **No batched inference**: Single-token-at-a-time generation. The prefill processes tokens sequentially rather than in parallel.

2. **No continuous batching**: One sequence per `Engine`. Multiple concurrent users require multiple `Engine` instances.

3. **No expert prediction**: The engine does not predict experts for the next token to overlap pread with attention compute.

4. **No 2-bit expert engine path**: The `dequant_matvec_2bit` Metal shader and `repack_experts_2bit.py` helper exist, but the engine code path does not dispatch it yet.

5. **No KV cache quantization**: KV cache stored as f32. Quantizing to bf16 or int8 would reduce memory and upload bandwidth.

6. **CPU engine not exposed via Python bindings**: The `CpuEngine` is Rust-only (`use moe_infer::engine::qwen35_moe::CpuEngine`).

## Project Structure

```
moe_infer_rs/                   Rust engine + Python bindings
  moe_infer/
    __init__.py                 Re-exports from native module
  src/
    lib.rs                      Module declarations + #[pymodule] init
    engine.rs                   Engine trait, DynEngine, EngineEnum, telemetry
    engine/
      qwen35_moe.rs             Module file (uses #[path] for submodules)
      qwen35_moe/
        constants.rs            ModelConfig trait + FullModel/StrippedModel impls
        cpu.rs                  CPU reference engine (ndarray, pure f32)
        fused_4bit.rs             Fused4bit GPU pipeline (3-CMD, Metal)
        metal_context.rs        Metal device/pipelines, ExpertCache LRU, scratch bufs
        metal_kernels.rs        Metal kernel dispatch (matvec, SwiGLU, conv1d, SSM, attention)
        shaders.metal           Metal compute shaders (embedded via include_str!)
    model.rs                    Module file (uses #[path] for submodules)
    model/
      config.rs                 ModelConfig derived from HF config.json
      expert.rs                 ExpertFile enum (Raw pread / Lz4 decompress)
      weights.rs                Mmap'd weight file + tensor lookup (model_weights.bin/.json)
    cache.rs                    KV cache + linear attention state (flat binary + JSON manifest I/O)
    math_util.rs                RMS norm, softmax, RoPE, dequant, SwiGLU, SSM, conv1d
    error.rs                    Error types
    constants.rs                Shared constants + backward-compat re-exports
    timer.rs                    Wall-clock timer
    python_bindings.rs          PyO3 bindings (Model, Cache, Engine, record_engine_telemetry)
  Cargo.toml
  pyproject.toml

helpers/                        Model conversion scripts
  convert.py                    One-step MLX → MoE-Infer conversion
  extract_weights.py            Non-expert weights → model_weights.bin + .json
  repack_experts_4bit.py        MLX 4-bit experts → packed_experts/
  compress_experts_lz4.py       packed_experts/ → packed_experts_lz4/ (~40-55% compression)
  repack_experts_2bit.py        packed_experts/ → packed_experts_2bit/ (experimental)
  strip_model.py                Build 4-layer stripped model for verification
  quantize_from_hf.py           HF unquantized → MoE-Infer 4-bit format

bench.py                        Multi-engine performance benchmark
verify_nway.py                  N-way logit comparison (Cpu, Fused4bit, C, mlx-lm)
chat.py                         Interactive chat demo
```

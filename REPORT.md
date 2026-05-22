# MoE-Infer: Technical Report

## Overview

MoE-Infer is a high-performance inference engine for Mixture-of-Experts models on Apple Silicon. It streams expert weights from SSD on demand, uses hand-tuned Metal compute shaders for all GPU operations, and exposes Python bindings via PyO3. No Python ML frameworks at runtime — just Rust, Metal, and ~0.65 GB of mmap'd weights.

Built on the 3-command-buffer GPU pipeline architecture designed by Dan Woods in the original C/Metal engine. The `FusedWoods` pipeline mode is named in his honor.

**Supported models**: `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.6-35B-A3B-4bit`

**Hardware**: Apple Silicon (M1–M4) with unified memory. Tested on M4.

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

Expert weights (~19 GB 4-bit) live on SSD in per-layer files (`packed_experts/layer_NN.bin`). Only K=8 active experts are read per layer (~1.77 MB each) via parallel `pread()` across 4 threads, with an LRU cache (32 entries) to avoid re-reading repeated experts.

### Metal Compute Pipeline

All matrix-vector multiplies run on GPU via Metal compute shaders:
- **4-bit dequant matvec**: `nibble * scale + bias` fused with dot product via FMA
- **Fused linear attention (CMD1)**: QKV/Z/B/A projections + Conv1d + Q/K RMS norms + SSM state update + Gated RMS norm — single command buffer
- **Fused full attention (CMD2)**: QKV projections + Q/K norms + RoPE (CPU) + Batched attention (scores, softmax, values) + Sigmoid gate + o_proj + Residual add + Post-attn norm + MoE gate + Shared expert projections — single command buffer
- **Expert dispatch (CMD3)**: K expert gate/up + SwiGLU + down + Shared expert SwiGLU + down + MoE combine + residual — async commit, completed next layer

## Weight File Format

MoE-Infer uses a custom binary weight format optimized for mmap and pread, converted from the HuggingFace/MLX safetensors format. The conversion is done by three helper scripts in `helpers/`.

### HF/MLX Format (Input)

The source model is stored in the standard MLX-quantized safetensors layout:

- **Multiple `.safetensors` files** with a `model.safetensors.index.json` index mapping tensor names to shard files.
- **4-bit affine quantization**: weights stored as nibble-packed U32 arrays `[out_dim, in_dim/8]`, scales and biases as BF16 arrays `[out_dim, in_dim/64]`. Group size is 64.
- **Expert tensors** use 3D shapes: `[num_experts, out_dim, packed_in_dim]` for each of gate_proj, up_proj, down_proj (weight + scales + biases = 9 tensors per layer).
- **Tensor naming**: `language_model.model.layers.N.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}` for experts; `language_model.model.layers.N.self_attn.*` for full attention; `language_model.model.layers.N.linear_attn.*` for linear attention.
- **Gate tensors** (router and shared expert gate) may use 8-bit quantization (`{weight,scales,biases}` with INT8 dtype) on newer models (Qwen3.6+). These are dequantized and re-quantized to 4-bit during extraction.

### MoE-Infer Non-Expert Weights

Single mmap'd file `model_weights.bin` + JSON manifest `model_weights.json`. Produced by `helpers/extract_weights.py`.

**`model_weights.bin`**: All non-expert tensors packed contiguously with 64-byte alignment. Each tensor stored in its native format (U32 packed for 4-bit, BF16 for scales/biases, F32 for norms). The file is mmap'd at startup for zero-copy GPU access via `newBufferWithBytesNoCopy`.

**`model_weights.json`**: Manifest mapping sanitized tensor names to `{offset, size, shape, dtype}`. Also includes a `config` block with all model dimensions (hidden_size, num_layers, head counts, MoE params, etc.) and per-layer types. The C/Rust engines use this to resolve tensors by name at runtime.

Key differences from HF format:
- **Single file** vs multi-shard: all non-expert tensors in one contiguous binary.
- **Name sanitization**: `language_model.model.layers.N.X` → `model.layers.N.X`; `language_model.lm_head` → `lm_head`. The C engine's `get_tensor_ptr()` looks up tensors by sanitized name.
- **8-bit → 4-bit conversion**: INT8 gate tensors are dequantized to f32 then re-quantized to 4-bit affine (group_size=64) to match the engine's uniform 4-bit code path.
- **Excluded**: vision tower weights, expert tensors (stored separately), and MTP (Multi-Token Prediction) expert layers.

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
- **Flat binary** vs safetensors container: no JSON header, no tensor metadata — just raw concatenated blobs. Offsets are known from `model_config.json` (see below).
- **pread-friendly**: fixed-size records at known offsets enable direct `pread(expert_id * expert_size)` from SSD without parsing.

### Model Config

`model_config.json` is generated by `helpers/gen_model_config.py` from the HF `config.json`. It contains all derived dimensions and expert layout offsets. The C engine has these as compile-time `#define`s; the Rust engine reads them from the JSON at runtime via `load_model_config()`.

Key computed fields:
- `expert_size_4bit` / `expert_size_2bit`: total bytes per expert in the packed layer files.
- `expert_layout_4bit` / `expert_layout_2bit`: byte offsets within each expert blob for each projection's weight/scales/biases.
- `rotary_dim`, `linear_total_key`, `linear_total_value`, `linear_conv_dim`: derived from head counts and dimensions.
- `num_full_attn_layers`, `num_linear_layers`: computed from `num_layers` and `full_attention_interval`.
- `layer_types`: per-layer type array (e.g., `["linear_attention", "linear_attention", "linear_attention", "full_attention"]` for stripped model).

### Conversion Pipeline

```
HF config.json ──► helpers/gen_model_config.py ──► model_config.json

HF safetensors/ ──► helpers/extract_weights.py ──► model_weights.bin
                                                   model_weights.json

HF safetensors/ ──► helpers/repack_experts_4bit.py ──► packed_experts/layer_00.bin
                                                       packed_experts/layer_01.bin
                                                       ...
```

All three scripts read from the same MLX-format model directory and output to a single MoE-Infer model directory. The conversion is a one-time offline step; at inference time only the binary files are needed.

## Pipeline Modes

| Mode | Description | Sync Points/Layer |
|------|-------------|-------------------|
| `Cpu` | Pure CPU reference. All dequant matvecs, norms, attention, SSM on CPU. | N/A (sequential) |
| `Gpu` | Individual GPU kernel dispatch per operation. No command buffer fusion. | 10+ |
| `FusedExp` | Linear attention fused into CMD1. MoE experts dispatched individually. | 4–6 |
| `FusedWoods` | Full C-engine architecture: CMD1 (linear attn) + CMD2 (full attn + o_proj + routing) + async CMD3 (experts + combine). | 2–3 |

`FusedWoods` (named after Dan Woods, author of the original C inference engine) is the recommended mode and matches the C engine's 3-command-buffer design.

### FusedWoods Command Buffer Layout

**Linear attention layers (30/40)**:
- CMD1: QKV/Z/B/A projections → Conv1d → Q/K RMS norms → SSM → Gated RMS norm → out_proj → Residual add
- CMD2: Post-attn norm → Gate + Shared expert projections + Shared expert gate
- CMD3 (async): Expert gate/up + SwiGLU + down × K → Shared SwiGLU + down → MoE combine + residual → Input norm for next layer

**Full attention layers (10/40)**:
- CMD1: Q/K/V projections → Q/K norms → RoPE (CPU) → KV cache append
- CMD2: Batched attention (scores, softmax, values) → Sigmoid gate → o_proj → Residual add → Post-attn norm → Gate + Shared expert projections
- CMD3 (async): Same as linear, plus explicit input norm for next layer

## Performance

Benchmarked on Apple M4, Qwen3.5-35B-A3B-4bit full model, K=8 experts, 32-token prompt, 100-token greedy generation.

| Metric | Value |
|--------|-------|
| Generation speed (FusedWoods) | 2.69 tok/s |
| Expert I/O (disk read) | ~5.8 ms/layer |
| Expert I/O share of per-layer time | ~72% |
| Weight file (mmap) | 0.65 GB |
| Expert files (SSD) | ~19 GB |

Expert I/O dominates latency. The LRU cache and parallel pread bring Rust within 4% of the C baseline.

## Numerical Verification

### Cpu vs MLX-LM (Stripped 4-Layer Model)

All verification uses the stripped model (4 layers, 4 experts) to enable fast iteration.

**Algorithmic bugs found and fixed**:
1. **RoPE element pairing** (2026-05-22): `apply_rope()` used traditional consecutive pairs (d, d+1) instead of NeoX-style pairs (i, i + dims/2) used by MLX's `nn.RoPE(traditional=False)`. Fix reduced logit max_diff from 0.835 to 0.113 (7.4× improvement).

2. **CpuOnly full-attention MoE bug** (2026-05-21): In CpuOnly mode, the GPU attention path returned early without adding attention output to hidden, causing MoE to use pre-attention hidden as residual. Attention contribution was lost (max_diff 4.88 vs FusedWoods).

**Per-operation verification** (Layer 0, token 0): Every intermediate tensor in GatedDeltaNet compared between Rust f32 and MLX bf16. All operations match within bf16 precision limits (~0.4% relative). No remaining algorithmic bugs.

**Current state**: After all fixes, max logit diff = 0.113, cos_sim = 0.99996. The residual divergence is entirely attributable to bf16 vs f32 precision differences across ~40 operations per token. Per-layer hidden state error is bounded at ~2e-3 and does not grow across layers. The lm_head projection (2048 → 248320) amplifies this to the observed 0.113 logit max_diff.

### C vs Rust FusedWoods (Stripped Model)

C and Rust FusedWoods are numerically identical (max_diff < 1e-5, 100% within 1e-3). The C→Rust port is faithful.

## Key Design Decisions

1. **SSD expert streaming over GPU preloading**: Expert weights are too large (~19 GB) for unified memory alongside KV caches and activations. On-demand SSD reads with LRU caching are the pragmatic choice.

2. **CPU KV cache**: KV caches stored as CPU f32 buffers rather than GPU persistent buffers. Adds ~0.03 ms/layer upload overhead but simplifies memory management. The C engine does the same for full attention layers.

3. **CPU RoPE**: RoPE rotations computed on CPU (both Rust and C). The rotary dimension is only 64 elements per head — GPU kernel launch overhead exceeds CPU compute time.

4. **Single mmap for non-expert weights**: All 0.65 GB of non-expert weights (embeddings, norms, attention projections, shared experts, gates) in one mmap'd file. Zero-copy GPU access via `newBufferWithBytesNoCopy`.

5. **Per-layer expert files**: Each layer's 256 experts in a separate file (`packed_experts/layer_NN.bin`). Enables `pread()` with offset — no seeking needed.

## Known Limitations

1. **No batched inference**: Single-token-at-a-time generation. The prefill processes tokens sequentially rather than in parallel.

2. **No continuous batching**: One sequence per `Context`. Multiple concurrent users require multiple `Context` instances.

3. **No speculative decoding**: C engine predicts experts for the next token to overlap pread with attention compute. Rust does not implement this.

4. **No KV cache quantization**: KV cache stored as f32. Quantizing to bf16 or int8 would reduce memory and upload bandwidth.

5. **No model conversion pipeline**: Converting from HuggingFace/MLX format to MoE-Infer format requires separate helper scripts (`helpers/`). Not yet unified into a single command.

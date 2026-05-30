# MoE-Infer: Technical Report

## Overview

MoE-Infer is a Rust-native inference engine for Mixture-of-Experts models on Apple Silicon. It builds on the techniques pioneered by [flash-moe](https://github.com/danielhanchen/flash-moe) — on-demand SSD expert streaming, hand-tuned Metal compute shaders, deferred GPU command dispatch — and extends them with a larger kernel surface, a novel block-aware quantization scheme (BQ4), compile-time safety guarantees, and a full numerical verification framework. The engine streams expert weights from SSD on demand (with optional LZ4 compression), runs all GPU operations through custom Metal kernels, and exposes Python bindings via PyO3. No Python ML frameworks at runtime — just Rust, Metal, and ~0.65 GB of mmap'd weights.

On an M1 Pro, MoE-Infer achieves **10 tok/s** on Qwen3.5-35B-A3B-4bit — a 25% improvement over the flash-moe C reference at 8 tok/s on the same model and hardware. The BQ4 variant trades 0.5 tok/s (9.5 tok/s) for better accuracy: it only quantizes expert weights (INT4) while keeping attention projections and router gates in BF16, and the lm_head in INT8.

**Supported models**: `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.6-35B-A3B-4bit`

**Hardware**: Apple Silicon (M1–M4) with unified memory. Tested on M1 Pro (10-core CPU, 14-core GPU).

## The Two Core Contributions

MoE-Infer builds on the insight that flash-moe proved: a 397B-parameter MoE model can run on a laptop by streaming only the K active experts from SSD per layer (4.4 tok/s on M3 Max, 48 GB). The architecture — per-layer expert files, `pread()` into Metal buffers, deferred GPU command dispatch — is sound. What MoE-Infer contributes are the two things that determine *how much* of the forward pass stays on the GPU and *how few* times the CPU must intervene. Together they deliver a **25% throughput improvement**: 10 tok/s vs flash-moe's 8 tok/s on the same M1 Pro hardware running the same Qwen3.5-35B-A3B-4bit model.

### 1. Maximal GPU Fusion: Only Wait for Routing

The forward pass for a single token through a single layer has exactly one hard synchronization point: the router must decide which experts to activate. Everything else — attention, norms, projections, activations, residual connections — is pure arithmetic on known dimensions. There is no data-dependent branching in any of it.

flash-moe's pipeline respects this partially. It uses three command buffers per layer — one for attention input projections, one for the attention output projection + residual + norm + gate routing projection, and one (deferred) for expert matvecs. But between the first two it round-trips to the CPU for Q/K head normalization, RoPE rotation, KV cache writes, and the attention softmax — operations that are pure element-wise or reduction arithmetic with no branching.

MoE-Infer's principle is: **fuse every GPU operation into the fewest possible command buffers, and only commit+wait when the CPU needs data to make a decision.** Two things happen:

First, flash-moe's two pre-expert command buffers are fused into one. Second — and more importantly — adjacent layers share a command buffer: the post-expert work of layer *N* (expert matvecs, combine, next-layer input norm) and the pre-expert work of layer *N+1* (attention, projections, gate) are encoded into the **same** Metal command buffer and committed together. This produces an interleaved N+1 buffers for N layers:

```
Layer 0:  [pre-expert L0]                          → commit + wait
Layer 1:  [post-expert L0 + pre-expert L1]         → commit + wait
Layer 2:  [post-expert L1 + pre-expert L2]         → commit + wait
  ...
Layer N-1:[post-expert L(N-2) + pre-expert L(N-1)] → commit + wait
Final:    [post-expert L(N-1)]                     → commit + wait
```

The post-expert buffer is encoded but left uncommitted at the end of each layer. The next layer's `op1_wait` takes that buffer, appends its pre-expert work, and commits both together. This means the GPU can execute expert matvecs from the previous layer and attention projections for the current layer in one uninterrupted stretch — no intervening CPU handshake.

Concretely, MoE-Infer moves these operations from CPU to GPU and fuses them into the interleaved buffer scheme:

| Operation | flash-moe | MoE-Infer |
|-----------|:---------:|:---------:|
| Q per-head RMS norm + deinterleave + RoPE | CPU (between GPU dispatches) | GPU (`q_head_norm_rope`) |
| K per-head RMS norm + RoPE | CPU (between GPU dispatches) | GPU (`k_head_norm_rope`) |
| KV cache write | CPU memcpy (between GPU dispatches) | GPU (`kv_cache_append`) |
| Attention softmax + values | CPU (between GPU dispatches) | GPU (`attn_sdpa_fused`) |
| o_proj + residual add | GPU (separate command buffer) | GPU (fused into pre-expert buffer) |
| Post-attention RMS norm | GPU two-pass (separate buffer) | GPU 1-pass (`rms_norm_fused_bf16`, fused in) |
| Router gate projection | GPU (separate command buffer) | GPU (fused into pre-expert buffer) |
| Shared expert gate/up projections | GPU (separate command buffer) | GPU (fused into pre-expert buffer) |

The CPU's only job per layer is `softmax(gate_scores)` and `top_k()`. Everything else — including the transition between layers — stays on the Metal command queue.

### 2. New Metal Kernels: 35 vs flash-moe's 26

flash-moe's `shaders.metal` defines 26 compute kernels covering the essential operations: dequantized matvec (naive, SIMD, v3/v4/v5, batched, 2-bit), SwiGLU, weighted sum, RMS norm (two-pass), residual add, batched attention (3-pass), GatedDeltaNet SSM, conv1d, Q/K RMS norm, decay/beta, gated RMS norm, MoE combine+residual, sigmoid gate, and fused gate+up+SwiGLU.

MoE-Infer retains every one of these kernels and adds **9 more**. These are not incremental — they are the kernels that make maximal GPU fusion possible:

| Kernel | What it replaces | Impact |
|--------|-----------------|--------|
| `q_head_norm_rope` | CPU deinterleave + RMS norm + RoPE (3 CPU steps per head) | Eliminates CPU read-back of Q projection |
| `k_head_norm_rope` | CPU RMS norm + RoPE (2 CPU steps per head) | Eliminates CPU read-back of K projection |
| `kv_cache_append` | CPU memcpy of K and V into cache buffers | Keeps KV cache entirely on GPU |
| `attn_sdpa_fused` | 3-pass batched GPU (scores → softmax → values) | Single-pass with online softmax; fewer dispatches |
| `attn_sdpa_block` | — (no flash-moe equivalent) | Block-sparse SDPA for long sequences |
| `attn_sdpa_reduce` | — (companion to block) | Reduce pass for 2-pass SDPA |
| `rms_norm_fused_bf16` | Two-pass RMS norm (sum-sq dispatch + apply dispatch) | Single dispatch; halves encoder overhead |
| `matvec_bf16` | — (flash-moe quantizes everything to 4-bit) | Direct BF16 matvec for BQ4 sensitive blocks |
| `matvec_int8` | — (no flash-moe equivalent) | INT8 matvec for BQ4 lm_head |

The first three (Q/K norm+RoPE, KV cache append) are the critical enablers for elimination of CPU round-trips. The next three (SDPA variants) improve attention throughput. The RMS norm fusion reduces dispatch count. The last two (BF16/INT8 matvecs) are the substrate for BQ4 quantization, which flash-moe had no equivalent for.

## Supporting Improvements

Beyond the two core contributions, MoE-Infer advances flash-moe's design on every supporting axis:

### BQ4: Block-Aware Quantization

flash-moe quantizes everything to 4-bit uniformly — all attention projections, all expert projections, router gates, and the lm_head. This works but leaves quality on the table: attention QKV projections and the router gate are disproportionately sensitive to quantization error because they determine which experts are activated.

MoE-Infer's BQ4 (`src/quantize/qwen35_moe/bq4.rs`, see also `quant/README.md`) takes the opposite approach: **only expert weights are quantized**. Every other weight matrix — attention projections, router gate, lm_head, norms, embeddings — stays in BF16 or FP32. The classification is determined by the tensor's role, not by any runtime sensitivity analysis:

| Block | Quantization | Kernel | Rationale |
|-------|-------------|--------|-----------|
| Expert gate/up/down (`mlp.switch_mlp.*`, `mlp.shared_expert.*`) | INT4 affine (group_size=64) | `dequant_matvec_4bit_v3` | 95%+ of parameters; bulk quantization is safe |
| Attention Q/K/V/O, router gate, embeddings (`self_attn.*`, `mlp.gate`, `attn.*`, `embed_tokens`) | BF16 | `matvec_bf16` | Q·Kᵀ amplifies quantization noise quadratically; router error misroutes tokens |
| lm_head (248,320 × 2048) | INT8 per-channel symmetric | `matvec_int8` | Half the memory of BF16 (485 MB vs 947 MB); applied once at the final layer so error does not compound |
| Norms, conv1d, quantization metadata | BF16 or FP32 | `matvec_bf16` | Vectors; quantization would distort normalization |

This is a design decision flash-moe never explored — their experiments focused on fixed 4-bit or fixed 2-bit quantization for all weights. BQ4 gives 4-bit-class memory efficiency for the expert bulk (the only part that needs it) while keeping the precision-critical blocks in BF16. The cost is modest: 9.5 tok/s vs 10 tok/s for the all-4-bit engine — a 5% throughput reduction because BF16 matvecs read twice the bytes of 4-bit dequant matvecs from the weight buffer.

### Type Safety and Memory Model

flash-moe is ~7,000 lines of C and Objective-C with manual memory management: `malloc`/`free`, raw pointer arithmetic, manual buffer lifecycle tracking. A buffer used after free is a segfault; a leaked buffer is a slow memory exhaustion. The code is correct but the language provides no guardrails.

MoE-Infer is Rust throughout. The borrow checker prevents use-after-free and data races at compile time. The `MetalContext` struct holds all GPU state with clear ownership; `ExpertBuffer` is a separate allocation with its own lifetime. The `Engine` trait enforces a uniform interface (`forward_token`, `reset_cache`) that every pipeline mode must satisfy. The `ModelConfig` trait uses Rust's const generics and associated constants to encode model dimensions at the type level — passing a `FullModel` vs `StrippedModel` selects a completely different code path with zero runtime overhead.

The C code handles threading with `pthread_create` and `dispatch_group`; the Rust code uses `rayon::scope()` with Rust's `Send`/`Sync` guarantees, making it impossible to accidentally share non-thread-safe state across expert I/O workers.

### Python Bindings

flash-moe embeds an HTTP server (`infer.m` lines 5635–6200) directly in the inference binary, with a separate C chat client (`chat.m`) that connects via SSE. This split requires serializing/deserializing tensors over HTTP for every token, and the server must manage multiple client connections manually.

MoE-Infer compiles to a native Python module via PyO3 (`src/python_bindings.rs`). The Python side (`moe_infer/pipeline.py`) handles tokenization, chat templates, and vision encoding using standard HuggingFace libraries, while the Rust engine handles inference through direct function calls — no serialization, no HTTP overhead, no socket management. The same binary serves interactive chat (`chat.py`), benchmarking (`bench.py`), and n-way verification (`verify_nway.py`) as library code.

### Numerical Verification Infrastructure

flash-moe has no reference implementation — the Metal shaders are the ground truth, and correctness is validated by inspecting output quality.

MoE-Infer provides three independent verification paths:

1. **CPU reference engine** (`src/engine/qwen35_moe/cpu.rs`): A pure-CPU, pure-f32 implementation of the entire forward pass using `ndarray`. Every operation — RMS norm, RoPE, attention, GatedDeltaNet, dequant matvec, SwiGLU — is implemented in scalar Rust with no GPU involvement. This serves as the numerical ground truth against which the GPU pipeline is verified.

2. **Stripped model** (`helpers/strip_model.py`): A 4-layer, 4-expert variant of the full model, suitable for fast verification iteration. Running the CPU reference on this model takes seconds, not minutes.

3. **N-way logit comparison** (`verify_nway.py`): Compares logits from CpuEngine, Fused4bit, the C reference, and MLX across multiple prompts, reporting max_diff and cosine similarity for each.

This infrastructure caught three algorithmic bugs during development that would have manifested as subtle quality degradation:

- **RoPE element pairing bug**: `apply_rope()` used traditional consecutive pairs instead of NeoX-style (i, i + dims/2). Fix reduced logit max_diff from 0.835 to 0.113 (7.4× improvement).
- **Full-attention MoE bug**: The full-attention path returned early without adding attention output to hidden, causing MoE to use pre-attention hidden as residual. Attention contribution was entirely lost (max_diff 4.88).
- **conv_state not updated**: conv1d_step was called but conv_state was never shifted for the next token — would produce incorrect results for multi-token sequences.

After all fixes, GPU vs CPU max_diff < 1e-5 (ULP-level); CPU vs MLX max logit diff = 0.113 with cos_sim = 0.99996. The residual divergence is entirely attributable to bf16 vs f32 precision differences.

### Application-Level Expert Cache

flash-moe's headline design principle is "Trust the OS" — the page cache is the expert cache, and their experiments showed that custom caching (Metal LRU, malloc cache, LZ4 compressed cache) was uniformly slower.

MoE-Infer takes the opposite approach with an application-level LRU cache (512 entries). This is feasible because the models are smaller (35B-A3B vs 397B-A17B): expert data is ~19 GB vs ~209 GB, and the working set of frequently-accessed experts fits more comfortably. The LRU cache provides:

- **Deterministic eviction**: The kernel's page reclaimer has no knowledge of MoE routing patterns. Under adversarial access, it can evict experts that will be needed next token. The application-level LRU knows the routing distribution.
- **Explicit lifecycle**: Cache entries are pread'd into pre-allocated Metal buffers with 2 MB alignment (3.6× DMA throughput vs page-aligned mmap). The kernel can't guarantee this alignment for page cache hits.
- **Predictable latency**: Cache hit → skip pread entirely. Cache miss → parallel pread across 4 threads. No variance from kernel page reclamation decisions.

### LZ4 Compression: Production Feature

flash-moe experimented with LZ4 expert compression and discarded it ("-13% — decompress overhead > warm cache savings"). Their workload (397B model, 209 GB experts) meant decompression competed with SSD bandwidth on every read.

MoE-Infer ships LZ4 compression as a production feature with transparent auto-detection. `helpers/compress_experts_lz4.py` compresses per-layer expert files (~40–55% compression ratio), and the engine auto-detects `packed_experts_lz4/` at load time, transparently decompressing via `lz4_flex`. The smaller model size (19 GB experts) and Apple Silicon's hardware LZ4 decode make the overhead negligible for most configurations. Both `ExpertFile::Raw` and `ExpertFile::Lz4` share the same `read_expert()` interface — switching is a filesystem-level change, not a code change.

### Shader Embedding

flash-moe requires `shaders.metal` on disk at runtime and compiles it with `newLibraryWithSource`. If the file is missing or moved, the engine fails at startup.

MoE-Infer embeds the Metal shader source at compile time via Rust's `include_str!()` macro. The shaders are compiled from the embedded string at runtime (same `newLibraryWithSource` path), but there is no external file dependency. The engine binary is self-contained.

### Multi-Model Architecture

flash-moe is hardcoded for one specific architecture (Qwen3.5-397B-A17B: 60 layers, 512 experts, K=10). Model dimensions are `#define` macros; supporting a different model requires recompiling with different constants.

MoE-Infer uses a `ModelConfig` trait with associated constants. The `FullModel` and `StrippedModel` implementations provide different dimension sets, and the engine is generic over `C: ModelConfig`. Adding a new model size means implementing the trait — no engine code changes needed. The same binary can load different models at runtime, selecting dimensions from `config.json`.

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
| Quantization | 4-bit affine (group_size=64), nibble * scale + bias (BQ4: selective BF16/INT8) |
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

**1. DMA alignment (3.6× speedup).** Expert data buffers are allocated with 2 MB alignment and wrapped via `newBufferWithBytesNoCopy`. The DMA controller that handles `pread()` from Apple's SSD achieves 3.6× higher throughput with 2 MB alignment vs the 16 KB page alignment that `mmap()` guarantees. This is the single biggest factor.

**2. One syscall, not 110 page faults.** A 1.77 MB expert spans ~110 pages (16 KB each on Apple Silicon). With `mmap()`, the first access to each page triggers a synchronous kernel trap: page fault → I/O dispatch → TLB fill. That's 110 individual round-trips through the kernel. With `pread()`, the kernel reads the entire blob in a single efficient I/O operation — one syscall, one I/O submission, one completion.

**3. Double-buffering for prediction preads.** The engine uses an A/B buffer pair. While the GPU processes expert A's results, prediction preads fill the B buffer for the next layer. `mmap()` can't provide independent buffer copies — it's a single mapping. The double-buffer scheme is essential for overlapping I/O with compute.

**4. Explicit eviction control.** The LRU cache (512 entries) decides which experts stay resident based on application-level routing patterns. With `mmap()` + memory pressure, the kernel's page reclaimer makes that decision instead — and it has no knowledge of MoE routing. Under the wrong access pattern, the kernel evicts the wrong pages and thrashing results. With `pread()`, eviction is deterministic and application-controlled.

**5. Scale mismatch.** Non-expert weights (0.65 GB) are small enough to mmap once at startup and keep resident forever — the `newBufferWithBytesNoCopy` Metal buffer wrapping the mmap'd region is valid for the lifetime of the process. Experts (19 GB) can't be kept resident alongside KV caches, activations, and scratch buffers. `pread()` is the correct primitive for "read this blob, use it on GPU, discard it."

### Metal Compute Pipeline

All matrix-vector multiplies run on GPU via Metal compute shaders. The kernel fleet (35 kernels, embedded via `include_str!`) covers every operation in the forward pass:

- **4-bit dequant matvec**: 6 variants (naive, SIMD, v3 optimized, v4, v5 LUT-based, batched). The v3 kernel tiles output rows across SIMD groups, caches the input vector in threadgroup shared memory, and uses `fma(nibble, scale*x, bias*x)` to fuse dequantization with the dot product in a single instruction.
- **BF16 matvec**: Direct BF16→f32 matvec with no dequantization, used by BQ4 for sensitive weight blocks (attention projections, router gate).
- **INT8 matvec**: Per-channel symmetric dequant matvec, used by BQ4 for the lm_head.
- **2-bit dequant matvec**: 2-bit variant for experimental ultra-low-bitwidth expert quantization.
- **Fused gate+up SwiGLU**: Reads the input vector once, computes both gate_proj and up_proj in the same kernel, and applies the SwiGLU nonlinearity — saves one input read and one kernel dispatch per expert.
- **RMS norm**: Two approaches — a two-pass (sum-sq reduction + apply) used standalone, and a single-pass fused variant (`rms_norm_fused_bf16`) that combines both in one dispatch.
- **SDPA attention**: Three variants — fused online-softmax (single-pass), and 2-pass block-sparse (block + reduce) for long sequences. Complemented by the original batched 3-pass (scores → softmax → values).
- **Fused Q/K head norm + RoPE**: Eliminates CPU round-trips by applying per-head RMS norm and rotary position embeddings on-GPU.
- **KV cache append**: Writes K and V directly into persistent GPU cache buffers.
- **GatedDeltaNet SSM**: GPU implementation of the SSM recurrence — one threadgroup per V-head, shared memory reduction.
- **Other fused kernels**: conv1d step, compute_decay_beta, gated RMS norm, MoE combine+residual, sigmoid gate, weighted sum.

#### Pipeline Structure (N+1 buffers for N layers)

flash-moe uses 3 command buffers per layer, committing and waiting on each before proceeding. MoE-Infer interleaves adjacent layers into shared buffers, producing N+1 command buffers for N layers:

```
Layer 0:  [pre-expert L0]                          → commit + wait → CPU routing → I/O
Layer 1:  [post-expert L0 + pre-expert L1]         → commit + wait → CPU routing → I/O
Layer 2:  [post-expert L1 + pre-expert L2]         → commit + wait → CPU routing → I/O
  ...
Layer N-1:[post-expert L(N-2) + pre-expert L(N-1)] → commit + wait → CPU routing → I/O
Final:    [post-expert L(N-1)]                     → commit + wait
```

Each pre-expert buffer contains a single Metal compute encoder with the full attention-and-gate pipeline:

**Linear attention layers (30/40)** — single encoder:
- in_proj_qkv/z/b/a → Conv1d → Q/K RMS norms → compute_decay_beta → gated_delta_net_step → gated_rms_norm → out_proj → residual_add → post_attention_norm → gate + shared_expert.gate_proj + shared_expert.up_proj + shared_expert_gate

**Full attention layers (10/40)** — single encoder:
- input_norm → Q/K/V projections → Q/K head norm + RoPE (fused) → KV cache append (GPU) → attn_scores_batched → attn_softmax_batched → attn_values_batched → sigmoid_gate → o_proj → residual_add → post_attention_norm → gate + shared_expert.gate_proj + shared_expert.up_proj + shared_expert_gate

Each post-expert buffer (appended to the next layer's pre-expert buffer):
- Expert gate/up + SwiGLU + down × K → Shared SwiGLU + down → MoE combine + residual → Input norm for next layer

The interleaving means the GPU executes expert matvecs from layer *N* and immediately continues into attention projections for layer *N+1* without a commit boundary. The only CPU synchronization point is after the combined buffer completes, when the CPU reads gate scores for routing.

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

Weights are stored in four dtypes under BQ4, each chosen for its precision/throughput tradeoff:

| Dtype | Used for |
|-------|----------|
| **U32 (packed int4)** | Bulk `nn.Linear` weight matrices: expert gate/up/down, shared expert. 8 nibbles per u32, dequantized on the fly: `nibble * scale + bias` |
| **BF16 (u16)** | **(a)** Sensitive blocks: attention Q/K/V/O projections, router gate, shared expert gate. **(b)** Scales and biases for 4-bit weights. **(c)** RMS norm weights. |
| **INT8 (i8)** | lm_head output projection (248,320 × 2048) with per-channel f32 scales. Half the memory of BF16 while preserving vocabulary precision |
| **F32** | Embedding (`embed_tokens`) and SSM decay parameter (`A_log`). Embeddings stay f32 to avoid accumulating precision loss at pipeline boundaries |

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
              ├──► helpers/repack_experts_2bit.py   ──► packed_experts_2bit/
              │                                       (experimental, 2-bit quant)
              │
              └──► helpers/quantize_from_hf.py +    ──► BQ4 model directory
                   src/quantize/qwen35_moe/bq4.rs       (block-aware quantization)
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

Cache persistence enables conversation resume across engine restarts — the Python pipeline saves cache after each user turn and restores it on the next invocation.

## Pipeline Modes

| Mode | Description |
|------|-------------|
| `Qwen35MoEBq4Exp1` | Full model: 40 layers, 256 experts, K=8. N+1 interleaved command buffers. BQ4 dispatch: expert weights in INT4, attention/gates in BF16, lm_head in INT8 |
| `Qwen35MoEBq4Exp2` | Full model, experimental variant of the BQ4 pipeline |
| `Qwen35MoEBq4Exp1` (Stripped) | Stripped model: 4 layers, 4 experts, K=4. For verification |
| `Cpu` | Pure-CPU reference engine using `ndarray`. Not exposed via Python bindings |

All GPU modes use the N+1 interleaved buffer scheme. The stripped variant uses a reduced 4-layer 4-expert model suitable for fast verification iteration. BQ4 dispatch is transparent — `WeightBuffer::encode_matvec_into` reads each tensor's dtype from the weight manifest and dispatches `matvec_bf16`, `matvec_int8`, or `dequant_matvec_4bit_v3` accordingly.

### FusedBq4 Command Buffer Layout

N layers produce N+1 Metal command buffers via interleaving (see "Maximal GPU Fusion" above). Each pre-expert buffer is a single Metal compute encoder:

**Linear attention layers (30/40)**:
- in_proj_qkv/z/b/a → Conv1d → Q/K RMS norms → compute_decay_beta → gated_delta_net_step → gated_rms_norm → out_proj → residual_add → post_attention_norm → gate + shared_expert.gate_proj + shared_expert.up_proj + shared_expert_gate

**Full attention layers (10/40)**:
- input_norm → Q/K/V projections → Q/K head norm + RoPE (fused) → KV cache append (GPU) → attn_scores_batched → attn_softmax_batched → attn_values_batched → sigmoid_gate → o_proj → residual_add → post_attention_norm → gate + shared_expert.gate_proj + shared_expert.up_proj + shared_expert_gate

Each post-expert buffer is appended to the next layer's pre-expert buffer before commit:
- Expert gate/up + SwiGLU + down × K → Shared SwiGLU + down → MoE combine + residual → Input norm for next layer

### CPU Engine

The `CpuEngine<C: ModelConfig>` in `engine/qwen35_moe/cpu.rs` is a pure-CPU reference implementation using `ndarray::Array1<f32>`. All computation is in f32. It follows the same data flow as the GPU pipeline:

- `pre_expert_full()`: input_layernorm → QKV projections → Q/K head norm + RoPE → KV cache append → attention (scores/softmax/values) → sigmoid gate → o_proj → residual add → post_attention_layernorm → gate projections
- `pre_expert_linear()`: input_layernorm → QKV/Z/B/A projections → conv1d_step with state update → RMS norm Q/K → decay/beta → gated delta net → gated RMS norm → out_proj → residual add → post_attention_layernorm → gate projections
- `post_expert()`: dequant_matvec_4bit + swiglu per expert → shared expert swiglu + down → sigmoid-gated residual combine

The CPU engine serves as a numerical reference for verifying the GPU pipeline, and runs at ~0.15 tok/s (vs ~10 tok/s for Fused4bit on M1 Pro). It is not exposed via Python bindings — it exists solely for verification.

## Performance

Benchmarked on M1 Pro (10-core CPU, 14-core GPU), Qwen3.5-35B-A3B-4bit full model (40 layers, 256 experts, K=8), 32-token prompt, 100-token greedy generation:

| Mode | tok/s | Notes |
|------|-------|-------|
| MoE-Infer (all-4bit) | **10.0** | All-4bit, N+1 interleaved command buffers |
| MoE-Infer (BQ4) | **9.5** | BQ4: only expert weights quantized (INT4); attention/gates in BF16, lm_head INT8 — costs 0.5 tok/s for better accuracy |
| flash-moe (C reference) | **8.0** | Same model, same hardware, 3-CMD pipeline |
| Cpu (reference) | ~0.15 | Pure Rust ndarray, f32 throughout |

The 25% throughput gain over flash-moe comes from the two core contributions: (1) interleaving adjacent layers into shared command buffers (N+1 buffers for N layers, vs flash-moe's 3N) and moving all CPU attention compute (Q/K norm, RoPE, KV cache writes, SDPA) onto GPU; (2) fused SDPA attention and single-pass RMS norm reduce Metal encoder dispatch overhead within each buffer. Expert I/O (SSD reads) dominates per-layer time at ~70% in all GPU modes.

### Fused4bit per-phase telemetry (full model, 20 tokens, prompt prefill)

| Stage | Mean (ms) | Share |
|-------|-----------|-------|
| Wall time | 1996 | — |
| engine.expert_io_ms | 671 | 33.6% |
| engine.full_attention_layer | 1.8 | <0.1% |
| engine.linear_group | 6.9 | 0.3% |
| engine.total_ms | 1996 | — |

Expert I/O (SSD pread) dominates at ~34% of wall time for short prompts and ~43% for 100-token prompts. Full-attention and linear SSM GPU compute are negligible (<10 ms total).

### Batched Prefill (branch: `batched-prefill`)

For prompts of more than one token, the token-serial engine above leaves the GPU underutilized. Each step issues matvecs against large weight matrices with a tiny input vector, and the GPU's wide units are barely loaded. The batched-prefill path processes N tokens together at the same point in the layer stack, sharing weight reads and collapsing per-token command-buffer commits into one per layer.

Seven landed optimizations, each preserving bit-near-identical logits (`max_diff ~ 2e-5`, top-1 match):

1. **Batched matvec kernels** — `matvec_bf16_n`, `matvec_int8_n`, `dequant_matvec_4bit_n`. Same simdgroup-based reduction as the single-input kernels, with one threadgroup per (row tile, token) (grid linearized; Metal disallows mixing scalar and vector grid attributes).
2. **Causal batched SDPA** (`attn_sdpa_causal_n`) — one TG per (new-token i, query head h); online softmax over positions [0, past_pos + i]; causal mask implicit in the loop bound. Paired with `kv_cache_append_n` for the N-position cache write.
3. **`op1_full_batched`** — full-attn pre-MoE work for N tokens in one cmd buffer. Q/K head-norm + RoPE loop within the encoder with offset addressing (per-token positions differ for RoPE).
4. **`op1_linear_batched`** — linear-attn pre-MoE work for N tokens in one cmd buffer. DeltaNet recurrence is sequential, but Metal serializes the per-token dispatches via implicit barriers on conv_state and delta_state. A tiny `buffer_copy_f32` GPU kernel saves/loads per-token ctx slices (CPU memcpy can't interleave with encoding because shared-storage writes are only visible at commit).
5. **Single-commit batched MoE op2** — all N op2 dispatches into ONE cmd buffer per layer. `encode_post_expert_at` reads per-token slices of `BatchedFullBuffers` via offset arithmetic. Per-token `combine_params` live in `bufs.combine_params_n[ti*10..]`.
6. **Unique-expert pool** — the biggest single win. Without it, N×K expert pre-reads per layer; with it, each unique expert is loaded ONCE per layer regardless of how many tokens chose it. At N=128, K=8, E=256 with random routing: ~200–256 unique vs 1024 pre-reads. ~4× pread bandwidth reduction. Pool memory: E × expert_size ≈ 450 MB on Qwen3.6.
7. **GEMM-tiled `matvec_bf16_gemm_n`** — NCOLS_PER_TG=4 tokens per threadgroup with K-axis tiling. Weight reads shared across the 4 tokens, ~4× weight bandwidth saving on the BF16 matvecs.

#### Measured speedup (M4, mins of 5 runs)

| N | seq tok/s | bat tok/s | speedup | seq ms | bat ms |
|---|---:|---:|---:|---:|---:|
| 8 | 6.51 | 11.66 | 1.79× | 1229 | 686 |
| 16 | 6.29 | 12.01 | 1.91× | 2543 | 1333 |
| 32 | 6.77 | 18.45 | **2.73×** | 4728 | 1735 |
| 64 | 6.29 | 14.92 | 2.37× | 10170 | 4289 |
| 128 | 6.89 | 17.82 | 2.59× | 18582 | 7182 |

Speedup grows with N — the unique-expert pool savings compound. Throughput climbs from ~7 tok/s to 12–18 tok/s.

#### Incremental contribution at N=128

| Cumulative optimizations | Speedup |
|---|---:|
| batched lm_head only | 1.05× |
| + `op1_full_batched` | 1.16× |
| + single-commit op2 (per-token pool) | 1.62× |
| + batched op1 for linear-attn | 1.72× |
| + GEMM-tiled `matvec_bf16_n` | 1.88× |
| + unique-expert pool | **2.59×** |

The python entry point is `Engine.forward_batched(tokens, cache)`, parallel to `Engine.forward(tokens, cache)`. N=1 falls through to the token-serial path (batched overhead exceeds the win at single tokens). Numerics validated by `verify_nway.py` and `helpers/verify_vs_original.py`; the bench is `helpers/bench_prefill.py`.

### Negative-result experiments

Two well-known weight-compression techniques that failed on Qwen3.6 (the failures themselves are informative):

- **Low-rank expert factorization** (`helpers/analyze_expert_lowrank.py`). Layer 0 alone is meaningfully low-rank (~28% storage saving at 99% variance). Layers 1–40 are rank-saturated: rank > 485 of 512 needed to keep 99% variance, and the low-rank decomposition is actually *larger* than the dense INT4 matrix (+21%). Verdict: experts have been trained to use full rank. Dead on 40 of 41 layers.
- **Rotated quantization** (QuIP#-style, `helpers/analyze_rotated_quant.py`). Random orthogonal rotation gives ≤0.2% per-matrix cosine lift, *only* at INT2 where the absolute baseline is already broken. The per-group BF16-scale INT4 scheme (group size 64) already does the local outlier adaptation that rotation would unlock against coarser schemes.

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

## Key Design Decisions

1. **SSD expert streaming over GPU preloading**: Expert weights are too large (~19 GB) for unified memory alongside KV caches and activations. On-demand SSD reads with LRU caching are the pragmatic choice.

2. **Application-level LRU cache over "trust the OS"**: While flash-moe demonstrated that the OS page cache works well for 209 GB expert sets, the smaller 19 GB working set here benefits from deterministic, application-controlled eviction with 2 MB DMA-aligned buffers. The LRU cache knows MoE routing patterns; the kernel's page reclaimer does not.

3. **CPU KV cache with GPU-side write**: KV caches stored as CPU f32 buffers, but written on-GPU via the `kv_cache_append` kernel to eliminate upload bandwidth. flash-moe does all KV cache management on CPU.

4. **GPU RoPE via fused kernels**: Unlike flash-moe which computes RoPE on CPU (rotary dim is only 64 elements), MoE-Infer uses fused `q_head_norm_rope` and `k_head_norm_rope` GPU kernels that combine deinterleaving, RMS norm, and rotation in a single dispatch. This eliminates CPU read-back for Q/K tensors.

5. **BQ4: tiered quantization**: Not all weight matrices have equal sensitivity. Keeping attention projections, routers, and lm_head at higher precision while quantizing expert bulk to 4-bit gives better quality at negligible memory cost.

6. **Single mmap for non-expert weights**: All 0.65 GB of non-expert weights (embeddings, norms, attention projections, shared experts, gates) in one mmap'd file. Zero-copy GPU access via `newBufferWithBytesNoCopy`.

7. **Per-layer expert files**: Each layer's 256 experts in a separate file (`packed_experts/layer_NN.bin`). Enables `pread()` with offset — no seeking needed.

8. **All compute in f32**: While weights are stored in 4-bit + BF16 + INT8, all math on both CPU and GPU runs in f32. This avoids precision accumulation issues while keeping memory/IO footprint small.

9. **Type-safe generic engine**: The `Engine` trait and `ModelConfig` trait use Rust's type system to enforce correctness at compile time. Wrong model dimensions or mismatched buffer sizes are caught by the compiler, not at runtime.

10. **Compile-time shader embedding**: Metal shaders embedded via `include_str!()` — no external file dependency. The engine binary is self-contained.

11. **File-based module convention**: No `mod.rs` files — Rust module declarations use `#[path]` attributes. The `qwen35_moe/` directory lives alongside `qwen35_moe.rs`, which declares its submodules with explicit `#[path = "qwen35_moe/foo.rs"]` attributes.

12. **Cache persistence**: Flat binary + JSON manifest format for saving/restoring full conversation state. Enables session resume across engine restarts without replaying history.

## Known Limitations

1. **No batched inference**: Single-token-at-a-time generation. The prefill processes tokens sequentially rather than in parallel.

2. **No continuous batching**: One sequence per `Engine`. Multiple concurrent users require multiple `Engine` instances.

3. **No expert prediction**: The engine does not predict experts for the next token to overlap pread with attention compute. flash-moe experimented with this (temporal prediction, MLP predictor) and found net-negative results due to cache pollution, but the smaller model size here may change the tradeoff.

4. **No 2-bit expert engine path**: The `dequant_matvec_2bit` Metal shader and `repack_experts_2bit.py` helper exist, but the engine code path does not dispatch it yet. flash-moe found 2-bit breaks JSON/tool calling quality.

5. **No KV cache quantization**: KV cache stored as f32. Quantizing to bf16 or int8 would reduce memory and upload bandwidth.

6. **CPU engine not exposed via Python bindings**: The `CpuEngine` is Rust-only — it exists for verification, not production use.

## Project Structure

```
moe_infer_rs/                   Rust engine + Python bindings
  moe_infer/
    __init__.py                 Re-exports from native module
  src/
    lib.rs                      Module declarations + #[pymodule] init
    engine.rs                   Engine trait, DynEngine, EngineEnum, telemetry
    engine/
      qwen35_moe/
        constants.rs            ModelConfig trait + FullModel/StrippedModel impls
        cpu.rs                  CPU reference engine (ndarray, pure f32)
        fused_bq4_exp1.rs       BQ4 GPU pipeline (N+1 interleaved buffers), primary engine
        fused_bq4_exp2.rs       BQ4 GPU pipeline, experimental variant
        metal_context.rs        Metal device/pipelines, ExpertBuffer, persistent GPU state
        metal_kernels.rs        Metal kernel dispatch (matvec, SwiGLU, conv1d, SSM, attention)
        shaders.metal           Metal compute shaders (embedded via include_str!)
    model.rs                    Module file (uses #[path] for submodules)
    model/
      config.rs                 ModelConfig derived from HF config.json
      expert.rs                 ExpertFile enum (Raw pread / Lz4 decompress)
      weights.rs                Mmap'd weight file + tensor lookup (model_weights.bin/.json)
    cache.rs                    KV cache + linear attention state (flat binary + JSON manifest I/O)
    math_util.rs                RMS norm, softmax, RoPE, dequant, SwiGLU, SSM, conv1d
    quant.rs                    Quantization dtype enum + tensor encoding
    quantize/
      qwen35_moe/
        bq4.rs                  BQ4 quantization: sensitivity analysis + tiered encoding
        name_mapping.json       Tensor name mapping for BQ4 conversion
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

//! Batched prefill path — MVP scope.
//!
//! Goal: process N tokens through one layer at a time (instead of one token
//! through all layers), so the per-token compute that's currently matvec
//! becomes batched matmul.
//!
//! MVP scope (what gets batched):
//! - All matvecs: q/k/v/o, in_proj_*, out_proj, gate, shared_expert.*,
//!   shared_expert_gate. New `dequant_matvec_4bit_n` shape supports
//!   [out_dim, N] from [in_dim, N] input.
//! - Element-wise: rms_norm, residual_add, sigmoid_gate, swiglu — same kernels,
//!   dispatched with one threadgroup per token.
//! - Q/K head norm + RoPE: per-token pos array passed into kernel.
//! - KV-cache append: writes positions [past_pos .. past_pos + N).
//! - Causal SDPA: new kernel — N queries vs (past_pos + N) K/V with causal
//!   triangular mask.
//!
//! MVP scope (what stays sequential — per-token loop within layer):
//! - DeltaNet (linear-attn) layers: conv1d_step + gated_delta_net_step are
//!   recurrent (state(t) = f(state(t-1), x(t))). Chunked-parallel form
//!   (Mamba2-style) is the future optimization; out of scope for MVP.
//! - MoE: route_experts (CPU) + op2 (8 experts × 3 matvecs each). The current
//!   per-token pread + per-expert matvec path stays unchanged. Batching MoE
//!   requires grouping tokens by chosen expert and a completely different
//!   data flow — Phase 2 work.
//!
//! Expected speedup (analysis from telemetry breakdown):
//!   125 ms/tok total = 60 ms GPU (op1 + op2) + 50 ms pread + 15 ms misc.
//!   Of the 60 ms GPU, the MoE expert matmuls dominate (~40-45 ms).
//!   The non-MoE GPU work (~15-20 ms/tok) is what batching collapses.
//!   On a 4K prefill: 4K × 125 ms = 8.3 min sequential.
//!   With batched non-MoE: 4K × (60 + 50 + saved_15) → ~7 min. ~1.4× wins.
//!
//! Buffer-layout decision:
//!   For MVP we allocate per-call N-sized buffers from `metal_buf_shared`.
//!   This costs ~milliseconds at the start of forward_batched but avoids
//!   changing the persistent MetalContext buffer sizes (which would balloon
//!   memory for users who never call the batched path).

#![allow(dead_code)]

// Implementation will land here as tasks #2-#7 complete.

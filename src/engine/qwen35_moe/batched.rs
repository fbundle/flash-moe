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

// ─── Status (batched-prefill branch) ──────────────────────────────────────
//
// DONE — Kernel infrastructure:
//   matvec_bf16_n, matvec_int8_n, dequant_matvec_4bit_n  — batched matvec
//                                                          variants for the
//                                                          [N, in_dim] → [N, out_dim] shape
//   attn_sdpa_causal_n                                   — causal batched
//                                                          attention with online softmax
//   kv_cache_append_n                                    — write K/V for N
//                                                          tokens at positions [past_pos..past_pos+N)
//
//   All 5 kernels compile cleanly and pipelines load. They are presently
//   unused by the runtime — the integration is the next phase.
//
// TODO — Integration (the bulk of the remaining work):
//
//   1. forward_hidden_batched entry point on FusedExp2 (mirrors
//      forward_hidden's signature) — initially delegates to the token-serial
//      forward_hidden so we have an A/B-testable Python entry point.
//
//   2. Replace internal loop body, layer at a time, with batched compute:
//      - Full-attn op1_full_batched: rms_norm + 3 batched matvecs (q/k/v)
//        + per-token loop for Q/K head_norm_rope (pos changes per token)
//        + kv_cache_append_n + attn_sdpa_causal_n + sigmoid_gate (loop or batched)
//        + o_proj batched matvec + residual_add + post_attn_norm (loop or batched)
//        + 4 batched matvecs for gate / shared_expert.*.
//      - Linear-attn op1_linear_batched: 4 batched matvecs (in_proj_*)
//        + per-token serial loop for conv1d_step, rms_norm_qk,
//          compute_decay_beta, gated_delta_net_step, gated_rms_norm
//          (these are recurrent / cheap enough that batching isn't worth it
//          for MVP) + out_proj batched matvec + residual + post_attn_norm
//        + 4 batched matvecs.
//      - MoE (per-token): keep existing route_experts + op2 per token. Loop
//        over N tokens within the layer. This stays serial — batching MoE
//        is Phase 2.
//      - Final norm + lm_head: batched.
//
//   3. Numerical equivalence test: forward_hidden(N) ≡ forward_hidden_batched(N)
//      for N ∈ {1, 2, 4, 32}, validated against transformers via verify_nway.
//
//   4. Bench prefill: helpers/bench_prefill.py comparing both paths at
//      prompt lengths {64, 256, 1K, 4K}.
//
// Expected: ~1.4× prefill speedup (analysis: non-MoE GPU work collapses;
// MoE stays per-token and dominates; net win bounded by non-MoE fraction).

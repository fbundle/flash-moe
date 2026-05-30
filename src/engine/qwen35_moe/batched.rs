//! Batched prefill path — implementation.
//!
//! Provides `op1_full_batched` (full-attn pre-MoE work for N tokens at once)
//! and the supporting buffer struct. The MoE part (op2) still runs per-token,
//! invoked from the integration in `fused_exp2.rs`'s `forward_hidden_batched`.
//!
//! ─── Speedup ceiling (next step) ────────────────────────────────────────
//!
//! End-to-end batched_prefill is functionally correct (verified against the
//! token-serial path: max_diff ~2e-5, top-1 match) but only ~1.0-1.17×
//! faster at N=16-64. The ceiling is the per-token op2 commits inside each
//! layer — they dominate wall time.
//!
//! Real ~1.4× requires encoding all N op2 dispatches into one command
//! buffer per layer (drop commits from N+1 to 2 per layer). Blocked by:
//!
//!   - `expert_buffer.expert_data[0..K]` is shared scratch — pread for
//!     token T+1 overwrites T's experts mid-encoding.
//!   - `ctx.buf_post_normed`, `ctx.buf_temp_residual`, `ctx.buf_moe_hidden`
//!     etc. are single-token buffers — N op2s all reading/writing them
//!     would clobber each other.
//!
//! Cleanest fix: refactor `encode_post_expert` to accept per-token
//! input/output buffer offsets so it can read/write slices of
//! `BatchedFullBuffers` directly. Then route_experts must allocate per-call
//! expert_data shadows OR require LRU cache ≥ unique experts in the batch
//! (~ N*K), so cached buffer refs stay stable across all N routes done
//! before any op2 encoding.

#![allow(dead_code)]

use std::ffi::c_void;
use metal::*;

use crate::constants::{RMS_NORM_EPS, GROUP_SIZE};
use crate::engine::metal_context::{MetalContext, WeightBuffer, ExpertBuffer, MAX_K, metal_buf_shared};
use crate::engine::qwen35_constants::ModelConfig;
use crate::model::weights::WeightFile;
use crate::engine::metal_kernels;

/// Scratch + output buffers for op1_full_batched, sized for N tokens.
/// Allocated once per forward_hidden_batched call and reused across all
/// 40 layers. On unified memory the allocation is cheap; reusing avoids
/// re-allocating the same buffers each layer.
pub struct BatchedFullBuffers {
    pub n: usize,
    // Layer-to-layer input/output (N hidden states).
    pub hidden_n: Buffer,
    // Op1 outputs needed by the per-token MoE step:
    pub post_normed_n: Buffer,       // [N, hidden]
    pub gate_scores_n: Buffer,       // [N, num_experts]
    pub shared_gate_n: Buffer,       // [N, shared_inter]
    pub shared_up_n: Buffer,         // [N, shared_inter]
    pub shared_gate_score_n: Buffer, // [N, 1]
    // Per-token combine_params for moe_combine_residual.
    // Layout: each token gets 10 f32 (expert_weights[0..8] + shared_gate_score + pad).
    pub combine_params_n: Buffer,    // [N, 10] f32
    // Per-token expert intermediate outputs for batched op2.
    // expert_out_n[ti * MAX_K + ki] holds the down-proj output of expert ki for token ti.
    pub expert_out_n: Vec<Buffer>,   // length N * MAX_K, each [hidden] f32
    pub shared_down_n: Buffer,       // [N, hidden] — shared expert down output per token
    // Scratch (lifetimes: within op1_full_batched only)
    pub qkv_x_n: Buffer,
    pub qbuf_n: Buffer,
    pub kbuf_n: Buffer,
    pub vbuf_n: Buffer,
    pub q_out_n: Buffer,
    pub q_gate_n: Buffer,
    pub attn_out_n: Buffer,
    pub o_proj_n: Buffer,
    pub temp_residual_n: Buffer,
}

impl BatchedFullBuffers {
    pub fn new<C: ModelConfig>(device: &Device, n: usize) -> Self {
        let hidden = C::HIDDEN_DIM;
        let num_q = C::NUM_ATTN_HEADS;
        let num_kv = C::NUM_KV_HEADS;
        let head_dim = C::HEAD_DIM;
        let q_dim = num_q * head_dim;
        let q_proj_dim = q_dim * 2; // Q + Q_gate concatenated
        let kv_dim = num_kv * head_dim;
        let num_experts = C::NUM_EXPERTS;
        let shared_inter = C::SHARED_INTERMEDIATE;

        let alloc = |elements: usize| metal_buf_shared(device, elements * 4);

        let mut expert_out_n = Vec::with_capacity(n * MAX_K);
        for _ in 0..(n * MAX_K) {
            expert_out_n.push(metal_buf_shared(device, hidden * 4));
        }

        Self {
            n,
            hidden_n:            alloc(n * hidden),
            post_normed_n:       alloc(n * hidden),
            gate_scores_n:       alloc(n * num_experts),
            shared_gate_n:       alloc(n * shared_inter),
            shared_up_n:         alloc(n * shared_inter),
            shared_gate_score_n: alloc(n * 1),
            combine_params_n:    alloc(n * 10),
            expert_out_n,
            shared_down_n:       alloc(n * hidden),
            qkv_x_n:    alloc(n * hidden),
            qbuf_n:     alloc(n * q_proj_dim),
            kbuf_n:     alloc(n * kv_dim),
            vbuf_n:     alloc(n * kv_dim),
            q_out_n:    alloc(n * q_dim),
            q_gate_n:   alloc(n * q_dim),
            attn_out_n: alloc(n * q_dim),
            o_proj_n:   alloc(n * hidden),
            temp_residual_n: alloc(n * hidden),
        }
    }
}

/// Batched op1 for a full-attn layer.
///
/// Reads `bufs.hidden_n` (N tokens × hidden_dim), writes:
///  - K/V appended to the layer's KV cache at positions [past_pos..past_pos+N)
///  - `bufs.temp_residual_n` (hidden_n + attn output, used as input to post_attn_norm)
///  - `bufs.post_normed_n` (input for MoE & shared expert)
///  - `bufs.gate_scores_n`, `bufs.shared_gate_n`, `bufs.shared_up_n`,
///    `bufs.shared_gate_score_n` (per-token outputs consumed by per-token MoE)
///
/// Caller must commit + wait on the returned `CommandBuffer`.
pub fn op1_full_batched<C: ModelConfig>(
    wf: &WeightFile,
    weight_buffer: &WeightBuffer,
    ctx: &MetalContext,
    layer: usize,
    fa_idx: usize,
    past_pos: usize,
    n: usize,
    bufs: &BatchedFullBuffers,
) -> CommandBuffer {
    let hidden_dim = C::HIDDEN_DIM;
    let num_q = C::NUM_ATTN_HEADS;
    let num_kv = C::NUM_KV_HEADS;
    let head_dim = C::HEAD_DIM;
    let rotary_dim = C::ROTARY_DIM;
    let rope_theta = C::ROPE_THETA as f32;
    let num_experts = C::NUM_EXPERTS;
    let shared_inter = C::SHARED_INTERMEDIATE;
    let q_dim = num_q * head_dim;
    let q_proj_dim = q_dim * 2;
    let kv_dim = num_kv * head_dim;

    let prefix = format!("language_model.model.layers.{}.self_attn", layer);
    let cm = ctx.queue.new_command_buffer().to_owned();
    let enc = cm.new_compute_command_encoder();

    let kc_buf = &ctx.buf_kv_k[fa_idx];
    let vc_buf = &ctx.buf_kv_v[fa_idx];

    // ── 1. input_norm: hidden_n → qkv_x_n (loop N dispatches of existing kernel) ──
    let in_norm_name = format!("language_model.model.layers.{}.input_layernorm.weight", layer);
    let pnw_ptr = wf.get_tensor_ptr(&in_norm_name).expect("input_layernorm.weight missing");
    let pnw_off = (pnw_ptr as usize - weight_buffer.base as usize) as u64;
    let rms_pipe = ctx.rms_norm_fused_bf16.as_ref().unwrap();
    for ti in 0..n {
        enc.set_compute_pipeline_state(rms_pipe);
        enc.set_buffer(0, Some(&bufs.hidden_n), (ti * hidden_dim * 4) as u64);
        enc.set_buffer(1, Some(&weight_buffer.buf), pnw_off);
        enc.set_buffer(2, Some(&bufs.qkv_x_n), (ti * hidden_dim * 4) as u64);
        unsafe {
            enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
        }
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    // ── 2. Q/K/V projections (batched matvec_n) ──
    weight_buffer.encode_matvec_n_into(wf, ctx, &enc, &format!("{}.q_proj", prefix),
        &bufs.qkv_x_n, 0, &bufs.qbuf_n, 0, q_proj_dim, hidden_dim, n as u32);
    weight_buffer.encode_matvec_n_into(wf, ctx, &enc, &format!("{}.k_proj", prefix),
        &bufs.qkv_x_n, 0, &bufs.kbuf_n, 0, kv_dim, hidden_dim, n as u32);
    weight_buffer.encode_matvec_n_into(wf, ctx, &enc, &format!("{}.v_proj", prefix),
        &bufs.qkv_x_n, 0, &bufs.vbuf_n, 0, kv_dim, hidden_dim, n as u32);

    // ── 3. Q head norm + RoPE per token (pos differs per token) ──
    let qn_ptr = wf.get_tensor_ptr(&format!("{}.q_norm.weight", prefix)).expect("q_norm.weight missing");
    let qn_off = (qn_ptr as usize - weight_buffer.base as usize) as u64;
    let q_pipe = ctx.q_head_norm_rope.as_ref().unwrap();
    for ti in 0..n {
        let pos = (past_pos + ti) as u32;
        enc.set_compute_pipeline_state(q_pipe);
        enc.set_buffer(0, Some(&bufs.qbuf_n), (ti * q_proj_dim * 4) as u64);
        enc.set_buffer(1, Some(&weight_buffer.buf), qn_off);
        enc.set_buffer(2, Some(&bufs.q_out_n),  (ti * q_dim * 4) as u64);
        enc.set_buffer(3, Some(&bufs.q_gate_n), (ti * q_dim * 4) as u64);
        unsafe {
            enc.set_bytes(4, 4, &(head_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &(rotary_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &rope_theta as *const f32 as *const c_void);
            enc.set_bytes(7, 4, &pos as *const u32 as *const c_void);
            enc.set_bytes(8, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
        }
        enc.dispatch_thread_groups(
            MTLSize::new(num_q as u64, 1, 1),
            MTLSize::new(head_dim as u64, 1, 1),
        );
    }

    // ── 4. K head norm + RoPE per token (in-place on kbuf_n[ti..]) ──
    let kn_ptr = wf.get_tensor_ptr(&format!("{}.k_norm.weight", prefix)).expect("k_norm.weight missing");
    let kn_off = (kn_ptr as usize - weight_buffer.base as usize) as u64;
    let k_pipe = ctx.k_head_norm_rope.as_ref().unwrap();
    for ti in 0..n {
        let pos = (past_pos + ti) as u32;
        enc.set_compute_pipeline_state(k_pipe);
        enc.set_buffer(0, Some(&bufs.kbuf_n), (ti * kv_dim * 4) as u64);
        enc.set_buffer(1, Some(&weight_buffer.buf), kn_off);
        unsafe {
            enc.set_bytes(2, 4, &(head_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(3, 4, &(rotary_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &rope_theta as *const f32 as *const c_void);
            enc.set_bytes(5, 4, &pos as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
        }
        enc.dispatch_thread_groups(
            MTLSize::new(num_kv as u64, 1, 1),
            MTLSize::new(head_dim as u64, 1, 1),
        );
    }

    // ── 5. KV-cache append for all N tokens ──
    metal_kernels::encode_kv_cache_append_n(
        ctx, &enc,
        &bufs.kbuf_n, 0, &bufs.vbuf_n, 0,
        kc_buf, vc_buf,
        past_pos as u32, kv_dim as u32, n as u32,
    );

    // ── 6. Causal batched SDPA: q_out_n vs (K_cache, V_cache) → attn_out_n ──
    metal_kernels::encode_attn_sdpa_causal_n(
        ctx, &enc,
        &bufs.q_out_n, 0,
        kc_buf, vc_buf,
        &bufs.attn_out_n, 0,
        past_pos as u32, num_q as u32, head_dim as u32, n as u32,
    );

    // ── 7. sigmoid_gate: attn_out_n *= sigmoid(q_gate_n) per token ──
    let sg_pipe = ctx.sigmoid_gate.as_ref().unwrap();
    for ti in 0..n {
        enc.set_compute_pipeline_state(sg_pipe);
        enc.set_buffer(0, Some(&bufs.attn_out_n), (ti * q_dim * 4) as u64);
        enc.set_buffer(1, Some(&bufs.q_gate_n),   (ti * q_dim * 4) as u64);
        unsafe { enc.set_bytes(2, 4, &(q_dim as u32) as *const u32 as *const c_void); }
        enc.dispatch_thread_groups(
            MTLSize::new(((q_dim as u32 + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    // ── 8. o_proj (batched) ──
    weight_buffer.encode_matvec_n_into(wf, ctx, &enc, &format!("{}.o_proj", prefix),
        &bufs.attn_out_n, 0, &bufs.o_proj_n, 0, hidden_dim, q_dim, n as u32);

    // ── 9. residual_add: o_proj_n + hidden_n → temp_residual_n (per token) ──
    let res_pipe = ctx.residual_add.as_ref().unwrap();
    for ti in 0..n {
        enc.set_compute_pipeline_state(res_pipe);
        enc.set_buffer(0, Some(&bufs.o_proj_n),        (ti * hidden_dim * 4) as u64);
        enc.set_buffer(1, Some(&bufs.hidden_n),        (ti * hidden_dim * 4) as u64);
        enc.set_buffer(2, Some(&bufs.temp_residual_n), (ti * hidden_dim * 4) as u64);
        unsafe { enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void); }
        enc.dispatch_thread_groups(
            MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    // ── 10. post_attention_layernorm per token: temp_residual_n → post_normed_n ──
    let post_norm_name = format!("language_model.model.layers.{}.post_attention_layernorm.weight", layer);
    let post_norm_ptr = wf.get_tensor_ptr(&post_norm_name).unwrap();
    let post_norm_off = (post_norm_ptr as usize - weight_buffer.base as usize) as u64;
    for ti in 0..n {
        enc.set_compute_pipeline_state(rms_pipe);
        enc.set_buffer(0, Some(&bufs.temp_residual_n), (ti * hidden_dim * 4) as u64);
        enc.set_buffer(1, Some(&weight_buffer.buf), post_norm_off);
        enc.set_buffer(2, Some(&bufs.post_normed_n), (ti * hidden_dim * 4) as u64);
        unsafe {
            enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
        }
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    // ── 11. Gate + shared expert projections (4 batched matvecs) ──
    let mlp_prefix = format!("language_model.model.layers.{}.mlp", layer);
    weight_buffer.encode_matvec_n_into(wf, ctx, &enc, &format!("{}.gate", mlp_prefix),
        &bufs.post_normed_n, 0, &bufs.gate_scores_n, 0, num_experts, hidden_dim, n as u32);
    weight_buffer.encode_matvec_n_into(wf, ctx, &enc, &format!("{}.shared_expert.gate_proj", mlp_prefix),
        &bufs.post_normed_n, 0, &bufs.shared_gate_n, 0, shared_inter, hidden_dim, n as u32);
    weight_buffer.encode_matvec_n_into(wf, ctx, &enc, &format!("{}.shared_expert.up_proj", mlp_prefix),
        &bufs.post_normed_n, 0, &bufs.shared_up_n, 0, shared_inter, hidden_dim, n as u32);
    weight_buffer.encode_matvec_n_into(wf, ctx, &enc, &format!("{}.shared_expert_gate", mlp_prefix),
        &bufs.post_normed_n, 0, &bufs.shared_gate_score_n, 0, 1, hidden_dim, n as u32);

    enc.end_encoding();
    cm
}


// ─── Per-token op2 encoded with explicit batched buffer offsets ───────────
//
// Encodes one token's op2 dispatches into `enc`. All inputs and outputs
// are slices of BatchedFullBuffers indexed by `ti`. Per-token combine_params
// are written to `bufs.combine_params_n[ti * 10..]` and read via offset.
//
// Reads:
//   - bufs.post_normed_n[ti]                   (input to expert + shared-expert matvecs)
//   - bufs.temp_residual_n[ti]                 (residual baseline for moe_combine_residual)
//   - bufs.shared_gate_n[ti], shared_up_n[ti]  (shared expert swiglu inputs)
// Writes:
//   - bufs.expert_out_n[ti*MAX_K..(ti+1)*MAX_K] (per-expert outputs, per-token)
//   - bufs.shared_down_n[ti]                    (shared expert down output)
//   - bufs.hidden_n[ti]                         (final layer output — input to next layer)
//
// expert_data[ki] points to the pread'd expert weights for this token; the
// caller is responsible for ensuring these refs remain stable through GPU
// execution (i.e., per-token expert pool, not the shared scratch).
#[allow(clippy::too_many_arguments)]
pub fn encode_post_expert_at<C: ModelConfig>(
    wf: &WeightFile,
    weight_buffer: &WeightBuffer,
    ctx: &MetalContext,
    enc: &ComputeCommandEncoderRef,
    layer_idx: usize,
    ti: usize,
    expert_weights: &[f32],
    shared_gate_score: f32,
    expert_data: &[Buffer],
    expert_buffer: &ExpertBuffer,
    num_experts_per_tok: usize,
    bufs: &BatchedFullBuffers,
) {
    let hidden_dim = C::HIDDEN_DIM;
    let moe_inter = C::MOE_INTERMEDIATE;
    let shared_inter = C::SHARED_INTERMEDIATE;
    let hidden_u32 = hidden_dim as u32;
    let inter_u32 = moe_inter as u32;
    let gs_u32 = GROUP_SIZE as u32;
    let actual_k = num_experts_per_tok.min(MAX_K);
    let prefix = format!("language_model.model.layers.{}.mlp", layer_idx);

    let post_normed_off = (ti * hidden_dim * 4) as u64;
    let temp_residual_off = (ti * hidden_dim * 4) as u64;
    let shared_gate_off = (ti * shared_inter * 4) as u64;
    let shared_up_off = (ti * shared_inter * 4) as u64;
    let hidden_out_off = (ti * hidden_dim * 4) as u64;
    let combine_off = (ti * 10 * 4) as u64;

    for ki in 0..actual_k {
        let eb = &expert_data[ki];
        if eb.length() == 0 { continue; }

        metal_kernels::encode_matvec_offset(ctx, enc,
            eb, C::GATE_W_OFF as u64,
            eb, C::GATE_S_OFF as u64,
            eb, C::GATE_B_OFF as u64,
            &bufs.post_normed_n, post_normed_off,
            &expert_buffer.scratch_gate, 0,
            inter_u32, hidden_u32, gs_u32, 3);

        metal_kernels::encode_matvec_offset(ctx, enc,
            eb, C::UP_W_OFF as u64,
            eb, C::UP_S_OFF as u64,
            eb, C::UP_B_OFF as u64,
            &bufs.post_normed_n, post_normed_off,
            &expert_buffer.scratch_up, 0,
            inter_u32, hidden_u32, gs_u32, 3);

        metal_kernels::encode_swiglu(ctx, enc,
            &expert_buffer.scratch_gate, 0,
            &expert_buffer.scratch_up, 0,
            &expert_buffer.scratch_act, 0, inter_u32);

        let eout = &bufs.expert_out_n[ti * MAX_K + ki];
        metal_kernels::encode_matvec_offset(ctx, enc,
            eb, C::DOWN_W_OFF as u64,
            eb, C::DOWN_S_OFF as u64,
            eb, C::DOWN_B_OFF as u64,
            &expert_buffer.scratch_act, 0,
            eout, 0,
            hidden_u32, inter_u32, gs_u32, 3);
    }

    // Shared expert: swiglu(shared_gate, shared_up) → shared_act, then down → shared_down_n[ti]
    metal_kernels::encode_swiglu(ctx, enc,
        &bufs.shared_gate_n, shared_gate_off,
        &bufs.shared_up_n, shared_up_off,
        &expert_buffer.shared_act, 0, shared_inter as u32);

    let sd_name = format!("{}.shared_expert.down_proj", prefix);
    let ok = weight_buffer.encode_matvec_into(wf, ctx, enc, &sd_name,
        &expert_buffer.shared_act, 0,
        &bufs.shared_down_n, hidden_out_off,
        hidden_dim, shared_inter);
    debug_assert!(ok, "shared_expert.down_proj missing");

    // Write per-token combine_params (CPU memcpy into bufs.combine_params_n[ti*10..]).
    let mut cparams = [0.0f32; 10];
    for (i, &w) in expert_weights.iter().enumerate() { cparams[i] = w; }
    cparams[8] = shared_gate_score;
    unsafe {
        let dst = (bufs.combine_params_n.contents() as *mut f32).add(ti * 10);
        std::ptr::copy_nonoverlapping(cparams.as_ptr(), dst, 10);
    }

    // moe_combine_residual: sums per-expert outputs + shared_down + temp_residual → hidden_n[ti]
    {
        let mcr_pipe = ctx.moe_combine_residual.as_ref().unwrap();
        enc.set_compute_pipeline_state(mcr_pipe);
        enc.set_buffer(0, Some(&bufs.temp_residual_n), temp_residual_off);
        enc.set_buffer(1, Some(&bufs.shared_down_n), hidden_out_off);
        enc.set_buffer(2, Some(&bufs.hidden_n), hidden_out_off);
        for ei in 0..MAX_K {
            if ei < actual_k {
                enc.set_buffer(3 + ei as u64, Some(&bufs.expert_out_n[ti * MAX_K + ei]), 0);
            } else {
                // Padding: bind any valid buffer; the kernel ignores past actual_k.
                enc.set_buffer(3 + ei as u64, Some(&bufs.hidden_n), hidden_out_off);
            }
        }
        enc.set_buffer(11, Some(&bufs.combine_params_n), combine_off);
        unsafe {
            enc.set_bytes(12, 4, &hidden_u32 as *const u32 as *const c_void);
            let ku = actual_k as u32;
            enc.set_bytes(13, 4, &ku as *const u32 as *const c_void);
        }
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }
}


// ─── Per-call expert pool ────────────────────────────────────────────────
//
// For batched op2 commits, each token's experts must live in distinct GPU
// memory through the whole layer's encoding (otherwise the next token's
// pread overwrites the previous token's data mid-encoding). This pool of
// N * MAX_K Metal buffers serves as per-token pread destinations.
//
// Memory cost: N * MAX_K * expert_size. For Qwen3.6 with N=32: ~450 MB.
pub struct ExpertPool {
    pub n: usize,
    pub buffers: Vec<Buffer>,  // length n * MAX_K
}

impl ExpertPool {
    pub fn new(device: &Device, n: usize, expert_size: usize) -> Self {
        let mut buffers = Vec::with_capacity(n * MAX_K);
        for _ in 0..(n * MAX_K) {
            buffers.push(metal_buf_shared(device, expert_size));
        }
        Self { n, buffers }
    }

    /// Get the `ki`-th expert slot for token `ti`.
    pub fn slot(&self, ti: usize, ki: usize) -> &Buffer {
        &self.buffers[ti * MAX_K + ki]
    }
}

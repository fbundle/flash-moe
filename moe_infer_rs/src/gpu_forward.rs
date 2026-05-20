// GPU layer forward pass — Metal dispatch for MoE experts, attention, delta-net.
// Port of moe_infer_mlx/core_src/layer_forward.h GPU paths (CMD1/CMD2/CMD3).
//
// The CPU-only forward pass lives in cpu_forward.rs.

use crate::types::*;
use crate::weights::OwnedTensorHashTable;
use crate::metal::MetalCtx;
use crate::gpu_ops;
use crate::expert_io::{ExpertLRUCache, IOPool, direct_expert_read};
use crate::constants::MAX_K;
use crate::kernels::cpu_vec_madd;
use crate::cpu_forward::{
    step_input_norm, step_full_attention, step_linear_attention,
    step_moe_routing, step_final_combine,
    step_cpu_expert, step_shared_expert,
    cpu_full_attn_compute, cpu_linear_attn_compute,
    CpuForwardScratch,
};
use crate::kernels::cpu_dequant_matvec;
use metal::MTLSize;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::fd::AsRawFd;

extern "C" {
    fn pread(fd: i32, buf: *mut u8, count: usize, offset: i64) -> isize;
}

#[cfg(feature = "timing")]
use std::time::Instant;

// ---- Layer weight cache ----

/// Cached weight pointers per layer — avoids repeated tensor lookups.
#[derive(Debug, Clone)]
pub struct LayerWeightCache {
    // Norms
    pub input_norm_w: Option<usize>,
    pub post_attn_norm_w: Option<usize>,

    // Full attention weights
    pub q_w: Option<usize>, pub q_s: Option<usize>, pub q_b: Option<usize>,
    pub k_w: Option<usize>, pub k_s: Option<usize>, pub k_b: Option<usize>,
    pub v_w: Option<usize>, pub v_s: Option<usize>, pub v_b: Option<usize>,
    pub o_w: Option<usize>, pub o_s: Option<usize>, pub o_b: Option<usize>,
    pub q_norm_w: Option<usize>, pub k_norm_w: Option<usize>,

    // Linear attention weights
    pub qkv_w: Option<usize>, pub qkv_s: Option<usize>, pub qkv_b: Option<usize>,
    pub z_w: Option<usize>,   pub z_s: Option<usize>,   pub z_b: Option<usize>,
    pub b_w: Option<usize>,   pub b_s: Option<usize>,   pub b_b: Option<usize>,
    pub a_w: Option<usize>,   pub a_s: Option<usize>,   pub a_b: Option<usize>,
    pub conv1d_w: Option<usize>,
    pub a_log: Option<usize>,
    pub dt_bias: Option<usize>,
    pub gated_norm_w: Option<usize>,
    pub out_proj_w: Option<usize>, pub out_proj_s: Option<usize>, pub out_proj_b: Option<usize>,

    // MoE weights
    pub gate_w: Option<usize>, pub gate_s: Option<usize>, pub gate_b: Option<usize>,
    pub sg_w: Option<usize>,  pub sg_s: Option<usize>,  pub sg_b: Option<usize>,
    pub su_w: Option<usize>,  pub su_s: Option<usize>,  pub su_b: Option<usize>,
    pub sd_w: Option<usize>,  pub sd_s: Option<usize>,  pub sd_b: Option<usize>,
    pub seg_w: Option<usize>, pub seg_s: Option<usize>, pub seg_b: Option<usize>,
}

impl LayerWeightCache {
    fn empty() -> Self {
        Self {
            input_norm_w: None, post_attn_norm_w: None,
            q_w: None, q_s: None, q_b: None,
            k_w: None, k_s: None, k_b: None,
            v_w: None, v_s: None, v_b: None,
            o_w: None, o_s: None, o_b: None,
            q_norm_w: None, k_norm_w: None,
            qkv_w: None, qkv_s: None, qkv_b: None,
            z_w: None, z_s: None, z_b: None,
            b_w: None, b_s: None, b_b: None,
            a_w: None, a_s: None, a_b: None,
            conv1d_w: None, a_log: None, dt_bias: None, gated_norm_w: None,
            out_proj_w: None, out_proj_s: None, out_proj_b: None,
            gate_w: None, gate_s: None, gate_b: None,
            sg_w: None, sg_s: None, sg_b: None,
            su_w: None, su_s: None, su_b: None,
            sd_w: None, sd_s: None, sd_b: None,
            seg_w: None, seg_s: None, seg_b: None,
        }
    }
}

/// Build per-layer weight pointer cache.
pub unsafe fn build_layer_cache(
    cfg: &ModelConfig,
    ht: &OwnedTensorHashTable,
    wf_data: *const u8,
) -> Vec<LayerWeightCache> {
    let mut cache = Vec::with_capacity(cfg.num_layers as usize);

    for i in 0..cfg.num_layers {
        let is_full = ((i + 1) % cfg.full_attn_interval) == 0;
        let mut lc = LayerWeightCache::empty();

        // Norms
        lc.input_norm_w = ht_offset(ht, wf_data, &format!("model.layers.{}.input_layernorm.weight", i));
        lc.post_attn_norm_w = ht_offset(ht, wf_data, &format!("model.layers.{}.post_attention_layernorm.weight", i));

        if is_full {
            lc.q_w = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.q_proj.weight", i));
            lc.q_s = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.q_proj.scales", i));
            lc.q_b = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.q_proj.biases", i));
            lc.k_w = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.k_proj.weight", i));
            lc.k_s = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.k_proj.scales", i));
            lc.k_b = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.k_proj.biases", i));
            lc.v_w = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.v_proj.weight", i));
            lc.v_s = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.v_proj.scales", i));
            lc.v_b = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.v_proj.biases", i));
            lc.o_w = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.o_proj.weight", i));
            lc.o_s = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.o_proj.scales", i));
            lc.o_b = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.o_proj.biases", i));
            lc.q_norm_w = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.q_norm.weight", i));
            lc.k_norm_w = ht_offset(ht, wf_data, &format!("model.layers.{}.self_attn.k_norm.weight", i));
        } else {
            lc.qkv_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_qkv.weight", i));
            lc.qkv_s = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_qkv.scales", i));
            lc.qkv_b = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_qkv.biases", i));
            lc.z_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_z.weight", i));
            lc.z_s = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_z.scales", i));
            lc.z_b = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_z.biases", i));
            lc.b_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_b.weight", i));
            lc.b_s = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_b.scales", i));
            lc.b_b = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_b.biases", i));
            lc.a_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_a.weight", i));
            lc.a_s = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_a.scales", i));
            lc.a_b = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.in_proj_a.biases", i));
            lc.conv1d_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.conv1d.weight", i));
            lc.a_log = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.A_log", i));
            lc.dt_bias = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.dt_bias", i));
            lc.gated_norm_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.norm.weight", i));
            lc.out_proj_w = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.out_proj.weight", i));
            lc.out_proj_s = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.out_proj.scales", i));
            lc.out_proj_b = ht_offset(ht, wf_data, &format!("model.layers.{}.linear_attn.out_proj.biases", i));
        }

        // MoE weights (same for all layers)
        lc.gate_w = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.gate.weight", i));
        lc.gate_s = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.gate.scales", i));
        lc.gate_b = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.gate.biases", i));
        lc.sg_w = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.gate_proj.weight", i));
        lc.sg_s = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.gate_proj.scales", i));
        lc.sg_b = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.gate_proj.biases", i));
        lc.su_w = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.up_proj.weight", i));
        lc.su_s = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.up_proj.scales", i));
        lc.su_b = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.up_proj.biases", i));
        lc.sd_w = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.down_proj.weight", i));
        lc.sd_s = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.down_proj.scales", i));
        lc.sd_b = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert.down_proj.biases", i));
        lc.seg_w = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert_gate.weight", i));
        lc.seg_s = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert_gate.scales", i));
        lc.seg_b = ht_offset(ht, wf_data, &format!("model.layers.{}.mlp.shared_expert_gate.biases", i));

        cache.push(lc);
    }

    println!("[cache] Pre-computed weight pointers for {} layers", cfg.num_layers);
    cache
}

pub unsafe fn ht_offset(ht: &OwnedTensorHashTable, _wf_data: *const u8, name: &str) -> Option<usize> {
    ht.find(name).map(|t| t.offset as usize)
}

// ---- Active expert size ----

pub fn active_expert_size(cfg: &ModelConfig, use_2bit: bool) -> usize {
    if use_2bit { cfg.expert_size_2bit as usize } else { cfg.expert_size_4bit as usize }
}

// ============================================================================
// GPU pipeline modes
// ============================================================================

/// Which GPU pipeline to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMode {
    /// Full async 3-command pipeline with GPU-side combine and deferred CMD3.
    /// CMD1: attention projections → CMD2: o_proj+norm+routing → CMD3: experts+combine (async)
    ThreeCommand,
    /// Simpler 2-command pipeline.  Attention projections on GPU; o_proj+routing
    /// on CPU; experts+combine in a single sync CMD2.  No deferred CMD3.
    TwoCommand,
}

// ============================================================================
// Deferred CMD3 state — stored between layers for async expert completion
// ============================================================================

/// Holds the asynchronously-running CMD3 so the next layer can wait on it.
pub struct DeferredCmd3 {
    pub cmd_buf: Option<metal::CommandBuffer>,  // running CMD3 (None if already completed)
    pub gpu_combined: bool,
    pub actual_k: usize,
    pub shared_gate_score: f32,
    pub expert_weights: [f32; MAX_K],
    pub valid: [bool; MAX_K],
    pub pinned_entries: Vec<usize>,  // cache entry indices pinned for this CMD3
    pub event_value: u64,  // MTLSharedEvent value to wait for (0 = no event pipeline)
}

// ============================================================================
// Deferred CMD3 helpers
// ============================================================================

/// Finalize CMD3 results into hidden state.  Waits for the command buffer
/// if CMD3 was submitted asynchronously.
unsafe fn finalize_cmd3(
    ctx: &MetalCtx,
    deferred: &mut DeferredCmd3,
    hidden: &mut [f32],
    hd: usize,
    scratch: &mut CpuForwardScratch,
) {
    if let Some(ref cmd) = deferred.cmd_buf {
        cmd.wait_until_completed();
    }
    if deferred.gpu_combined {
        let src = ctx.buf_moe_hidden.contents() as *const f32;
        std::ptr::copy_nonoverlapping(src, hidden.as_mut_ptr(), hd);
    } else {
        scratch.moe_out[..hd].fill(0.0);
        for k in 0..deferred.actual_k {
            if !deferred.valid[k] { continue; }
            let out_ptr = ctx.buf_multi_expert_out[k].contents() as *const f32;
            let expert_out = std::slice::from_raw_parts(out_ptr, hd);
            cpu_vec_madd(&mut scratch.moe_out, expert_out, deferred.expert_weights[k], hd);
        }
        scratch.shared_out[..hd].fill(0.0);
        let shared_out_ptr = ctx.buf_shared_out.contents() as *const f32;
        let sw = crate::kernels::cpu_sigmoid(deferred.shared_gate_score);
        for i in 0..hd {
            scratch.shared_out[i] = (*shared_out_ptr.add(i)) * sw;
        }
        step_final_combine(&scratch.h_mid, &scratch.moe_out, &scratch.shared_out, hidden, hd);
    }
}

/// Complete any pending deferred CMD3 — wait for GPU, finalize, unpin cache entries.
pub unsafe fn complete_deferred_experts(
    ctx: &MetalCtx,
    deferred: &mut Option<DeferredCmd3>,
    hidden: &mut [f32],
    hd: usize,
    scratch: &mut CpuForwardScratch,
    expert_cache: Option<&mut ExpertLRUCache>,
) {
    if let Some(ref mut d) = *deferred {
        finalize_cmd3(ctx, d, hidden, hd, scratch);
        // Unpin cache entries that were pinned for this CMD3
        if let Some(cache) = expert_cache {
            cache.unpin_batch(&d.pinned_entries);
        }
    }
    *deferred = None;
}

// ============================================================================
// GPU availability check
// ============================================================================

fn gpu_ready(ctx: Option<&MetalCtx>) -> bool {
    ctx.and_then(|c| c.wf_buf.as_ref()).is_some()
}

// ============================================================================
// Encode CMD2: o_proj + residual_add + rms_norm + routing + shared expert
// ============================================================================

/// Encode the fused CMD2 into a command buffer.
/// On entry: residual in buf_residual, attn_out in batch_out[6] (or buf_attn_out for GPU attn).
/// On exit:  buf_h_mid = residual + o_proj(attn_out), buf_input = norm(buf_h_mid),
///           batch_out[0..3] = routing gate + shared gate/up/seg scores.
unsafe fn encode_cmd2_fused(
    cfg: &ModelConfig,
    ctx: &MetalCtx,
    cmd: &metal::CommandBufferRef,
    lc: &LayerWeightCache,
    oproj_w: *const u32, oproj_s: *const u16, oproj_b: *const u16,
    oproj_in_dim: u32,
    attn_out_src: &metal::Buffer,  // batch_out[6] or buf_attn_out
    wf_base: *const u8,
    gpu_attn_fuse: bool,
    fa_idx: usize,
    seq_len: u32,
) {
    let wf_ptr = wf_base;

    // ---- GPU attention dispatches (prepended before o_proj) ----
    if gpu_attn_fuse {
        let kv_dim = (cfg.num_kv_heads * cfg.head_dim) as u32;
        let head_dim = cfg.head_dim as u32;
        let heads_per_kv = (cfg.num_attn_heads / cfg.num_kv_heads) as u32;
        let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let seq_stride = cfg.gpu_kv_seq as u32;

        // Enc A1: attn_scores_batched — Q @ K^T with GQA
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(ctx.attn_scores_pipe.as_ref().unwrap());
            enc.set_buffer(0, Some(&ctx.buf_attn_q), 0);
            enc.set_buffer(1, Some(&ctx.buf_kv_k[fa_idx]), 0);
            enc.set_buffer(2, Some(&ctx.buf_attn_scores), 0);
            enc.set_bytes(3, 4, &head_dim as *const u32 as *const _);
            enc.set_bytes(4, 4, &kv_dim as *const u32 as *const _);
            enc.set_bytes(5, 4, &seq_len as *const u32 as *const _);
            enc.set_bytes(6, 4, &seq_stride as *const u32 as *const _);
            enc.set_bytes(7, 4, &scale as *const f32 as *const _);
            enc.set_bytes(8, 4, &heads_per_kv as *const u32 as *const _);
            enc.set_bytes(9, 4, &seq_len as *const u32 as *const _);
            let total_tgs = seq_len as u64 * cfg.num_attn_heads as u64;
            enc.dispatch_thread_groups(MTLSize::new(total_tgs, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }
        // Enc A2: attn_softmax_batched
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(ctx.attn_softmax_pipe.as_ref().unwrap());
            enc.set_buffer(0, Some(&ctx.buf_attn_scores), 0);
            enc.set_bytes(1, 4, &seq_len as *const u32 as *const _);
            enc.set_bytes(2, 4, &seq_stride as *const u32 as *const _);
            enc.dispatch_thread_groups(
                MTLSize::new(cfg.num_attn_heads as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }
        // Enc A3: attn_values_batched — softmax scores @ V
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(ctx.attn_values_pipe.as_ref().unwrap());
            enc.set_buffer(0, Some(&ctx.buf_attn_scores), 0);
            enc.set_buffer(1, Some(&ctx.buf_kv_v[fa_idx]), 0);
            enc.set_buffer(2, Some(&ctx.buf_attn_out), 0);
            enc.set_bytes(3, 4, &head_dim as *const u32 as *const _);
            enc.set_bytes(4, 4, &kv_dim as *const u32 as *const _);
            enc.set_bytes(5, 4, &seq_len as *const u32 as *const _);
            enc.set_bytes(6, 4, &seq_stride as *const u32 as *const _);
            enc.set_bytes(7, 4, &heads_per_kv as *const u32 as *const _);
            let total_threads = head_dim as u64 * cfg.num_attn_heads as u64;
            let tgs = (total_threads + 255) / 256;
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }
        // Enc A4: sigmoid_gate — apply Q gate to attn_out
        {
            let enc = cmd.new_compute_command_encoder();
            let q_dim = (cfg.num_attn_heads * cfg.head_dim) as u32;
            enc.set_compute_pipeline_state(ctx.sigmoid_gate_pipe.as_ref().unwrap());
            enc.set_buffer(0, Some(&ctx.buf_attn_out), 0);
            enc.set_buffer(1, Some(&ctx.buf_attn_gate), 0);
            enc.set_bytes(2, 4, &q_dim as *const u32 as *const _);
            let tgs = (q_dim as u64 + 255) / 256;
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }
    }

    // Enc 1 (or 5 w/ GPU attn): o_proj matvec (attn_out → buf_output)
    {
        let w_off = (oproj_w as *const u8).offset_from(wf_ptr) as u64;
        let s_off = (oproj_s as *const u8).offset_from(wf_ptr) as u64;
        let b_off = (oproj_b as *const u8).offset_from(wf_ptr) as u64;

        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.matvec_fast);
        enc.set_buffer(0, ctx.wf_buf.as_deref(), w_off);
        enc.set_buffer(1, ctx.wf_buf.as_deref(), s_off);
        enc.set_buffer(2, ctx.wf_buf.as_deref(), b_off);
        enc.set_buffer(3, Some(attn_out_src), 0);
        enc.set_buffer(4, Some(&ctx.buf_output), 0);
        let hd = cfg.hidden_dim as u32;
        let gs = cfg.group_size as u32;
        enc.set_bytes(5, 4, &hd as *const u32 as *const _);
        enc.set_bytes(6, 4, &oproj_in_dim as *const u32 as *const _);
        enc.set_bytes(7, 4, &gs as *const u32 as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new(hd as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
    }

    // Enc 2: residual_add (buf_output + buf_residual → buf_h_mid)
    {
        let enc = cmd.new_compute_command_encoder();
        let dim = cfg.hidden_dim as u32;
        enc.set_compute_pipeline_state(&ctx.residual_add);
        enc.set_buffer(0, Some(&ctx.buf_residual), 0);
        enc.set_buffer(1, Some(&ctx.buf_output), 0);
        enc.set_buffer(2, Some(&ctx.buf_h_mid), 0);
        enc.set_bytes(3, 4, &dim as *const u32 as *const _);
        let tgs = (dim + 255) / 256;
        enc.dispatch_thread_groups(MTLSize::new(tgs as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }

    // Enc 3: rms_norm_sum_sq (buf_h_mid → buf_sum_sq)
    {
        let enc = cmd.new_compute_command_encoder();
        let dim = cfg.hidden_dim as u32;
        enc.set_compute_pipeline_state(&ctx.rms_norm_sum);
        enc.set_buffer(0, Some(&ctx.buf_h_mid), 0);
        enc.set_buffer(1, Some(&ctx.buf_sum_sq), 0);
        enc.set_bytes(2, 4, &dim as *const u32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }

    // Enc 4: rms_norm_apply_bf16 (buf_h_mid + norm_w → buf_input)
    {
        let post_norm = lc.post_attn_norm_w.expect("post_attn_norm_w required");
        let norm_off = (wf_ptr.add(post_norm) as *const u8).offset_from(wf_ptr) as u64;
        let enc = cmd.new_compute_command_encoder();
        let dim = cfg.hidden_dim as u32;
        let eps = cfg.rms_norm_eps;
        enc.set_compute_pipeline_state(&ctx.rms_norm_apply_bf16);
        enc.set_buffer(0, Some(&ctx.buf_h_mid), 0);
        enc.set_buffer(1, ctx.wf_buf.as_deref(), norm_off);
        enc.set_buffer(2, Some(&ctx.buf_sum_sq), 0);
        enc.set_buffer(3, Some(&ctx.buf_input), 0);
        enc.set_bytes(4, 4, &dim as *const u32 as *const _);
        enc.set_bytes(5, 4, &eps as *const f32 as *const _);
        let tgs = (dim + 255) / 256;
        enc.dispatch_thread_groups(MTLSize::new(tgs as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }

    // Enc 5-8: routing gate + shared expert gate/up + shared_expert_gate score
    let gate_w = lc.gate_w.map_or(std::ptr::null(), |o| wf_ptr.add(o) as *const u32);
    let gate_s = lc.gate_s.map_or(std::ptr::null(), |o| wf_ptr.add(o) as *const u16);
    let gate_b = lc.gate_b.map_or(std::ptr::null(), |o| wf_ptr.add(o) as *const u16);
    let sg_w = lc.sg_w.map_or(std::ptr::null(), |o| wf_ptr.add(o) as *const u32);
    let sg_s = lc.sg_s.map_or(std::ptr::null(), |o| wf_ptr.add(o) as *const u16);
    let sg_b = lc.sg_b.map_or(std::ptr::null(), |o| wf_ptr.add(o) as *const u16);
    let su_w = lc.su_w.map_or(std::ptr::null(), |o| wf_ptr.add(o) as *const u32);
    let su_s = lc.su_s.map_or(std::ptr::null(), |o| wf_ptr.add(o) as *const u16);
    let su_b = lc.su_b.map_or(std::ptr::null(), |o| wf_ptr.add(o) as *const u16);
    let seg_w = lc.seg_w.map_or(std::ptr::null(), |o| wf_ptr.add(o) as *const u32);
    let seg_s = lc.seg_s.map_or(std::ptr::null(), |o| wf_ptr.add(o) as *const u16);
    let seg_b = lc.seg_b.map_or(std::ptr::null(), |o| wf_ptr.add(o) as *const u16);

    let moe_specs = [
        gpu_ops::BatchMatvecSpec { w: gate_w, scales: gate_s, biases: gate_b, out_cpu: std::ptr::null_mut(), out_dim: cfg.num_experts as u32, in_dim: cfg.hidden_dim as u32, group_size: cfg.group_size as u32, batch_slot: 0 },
        gpu_ops::BatchMatvecSpec { w: sg_w, scales: sg_s, biases: sg_b, out_cpu: std::ptr::null_mut(), out_dim: cfg.shared_intermediate as u32, in_dim: cfg.hidden_dim as u32, group_size: cfg.group_size as u32, batch_slot: 1 },
        gpu_ops::BatchMatvecSpec { w: su_w, scales: su_s, biases: su_b, out_cpu: std::ptr::null_mut(), out_dim: cfg.shared_intermediate as u32, in_dim: cfg.hidden_dim as u32, group_size: cfg.group_size as u32, batch_slot: 2 },
        gpu_ops::BatchMatvecSpec { w: seg_w, scales: seg_s, biases: seg_b, out_cpu: std::ptr::null_mut(), out_dim: 1, in_dim: cfg.hidden_dim as u32, group_size: cfg.group_size as u32, batch_slot: 3 },
    ];
    gpu_ops::gpu_encode_batch_matvec(ctx, cmd, &moe_specs);
}

// ============================================================================
// Encode CMD3: K expert forwards + shared SwiGLU + shared down + GPU combine
// ============================================================================

/// Encode expert forwards into cmd3.
/// `expert_data_bufs`: if Some, use these cache buffers instead of buf_multi_expert_data[k]
/// (zero-copy — avoids memcpy from cache to GPU buffer).
/// On exit: expert outputs in buf_multi_expert_out[k], shared in buf_shared_out,
/// buf_moe_hidden = combine result, buf_input = normed for next layer.
unsafe fn encode_cmd3_experts(
    cfg: &ModelConfig,
    ctx: &MetalCtx,
    cmd: &metal::CommandBufferRef,
    lc: &LayerWeightCache,
    actual_k: usize,
    shared_gate_score: f32,
    expert_weights: &[f32; MAX_K],
    valid: &[bool; MAX_K],
    use_2bit: bool,
    wf_base: *const u8,
    _layer_idx: usize,
    expert_data_bufs: &[Option<&metal::Buffer>],
) -> bool {
    // Expert forwards (batched into one cmd)
    for k in 0..actual_k {
        if valid[k] {
            let data_buf = expert_data_bufs.get(k).and_then(|b| *b);
            gpu_ops::gpu_encode_expert_forward_slot(cfg, ctx, cmd, k, use_2bit, data_buf);
        }
    }

    // Shared expert gate+up+SwiGLU+down
    let sg_w = lc.sg_w.map_or(std::ptr::null(), |o| wf_base.add(o) as *const u32);
    let sg_s = lc.sg_s.map_or(std::ptr::null(), |o| wf_base.add(o) as *const u16);
    let sg_b = lc.sg_b.map_or(std::ptr::null(), |o| wf_base.add(o) as *const u16);
    let su_w = lc.su_w.map_or(std::ptr::null(), |o| wf_base.add(o) as *const u32);
    let su_s = lc.su_s.map_or(std::ptr::null(), |o| wf_base.add(o) as *const u16);
    let su_b = lc.su_b.map_or(std::ptr::null(), |o| wf_base.add(o) as *const u16);
    let sd_w = lc.sd_w.map_or(std::ptr::null(), |o| wf_base.add(o) as *const u32);
    let sd_s = lc.sd_s.map_or(std::ptr::null(), |o| wf_base.add(o) as *const u16);
    let sd_b = lc.sd_b.map_or(std::ptr::null(), |o| wf_base.add(o) as *const u16);
    gpu_ops::gpu_encode_shared_down_swiglu(cfg, ctx, cmd, sg_w, sg_s, sg_b, su_w, su_s, su_b, sd_w, sd_s, sd_b);

    // GPU-side combine (if not last layer and moe_combine_residual pipeline available)
    let gpu_combine = ctx.moe_combine_residual.is_some()
        && ctx.wf_buf.is_some()
        && cfg.num_layers as usize > 1  // must have at least 2 layers for next input_norm_w
        && lc.input_norm_w.is_some();

    // We need the NEXT layer's input_norm_w for GPU combine
    // This is handled by the caller passing the appropriate weight

    if gpu_combine {
        // Prepare combine params
        let params_ptr = ctx.buf_combine_params.contents() as *mut f32;
        std::ptr::write_bytes(params_ptr as *mut u8, 0, 10 * 4);
        for k in 0..actual_k {
            *params_ptr.add(k) = if valid[k] { expert_weights[k] } else { 0.0 };
        }
        *params_ptr.add(8) = shared_gate_score;

        // Enc C1: moe_combine_residual
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(ctx.moe_combine_residual.as_ref().unwrap());
            enc.set_buffer(0, Some(&ctx.buf_h_mid), 0);
            enc.set_buffer(1, Some(&ctx.buf_shared_out), 0);
            enc.set_buffer(2, Some(&ctx.buf_moe_hidden), 0);
            for k in 0..MAX_K {
                enc.set_buffer((3 + k) as u64, Some(&ctx.buf_multi_expert_out[k]), 0);
            }
            enc.set_buffer(11, Some(&ctx.buf_combine_params), 0);
            let dim = cfg.hidden_dim as u32;
            let kval = actual_k as u32;
            enc.set_bytes(12, 4, &dim as *const u32 as *const _);
            enc.set_bytes(13, 4, &kval as *const u32 as *const _);
            let tgs = (dim + 255) / 256;
            enc.dispatch_thread_groups(MTLSize::new(tgs as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }

        // Enc C2: rms_norm_sum_sq (buf_moe_hidden → buf_cmd3_sum_sq)
        {
            let enc = cmd.new_compute_command_encoder();
            let dim = cfg.hidden_dim as u32;
            enc.set_compute_pipeline_state(&ctx.rms_norm_sum);
            enc.set_buffer(0, Some(&ctx.buf_moe_hidden), 0);
            enc.set_buffer(1, Some(&ctx.buf_cmd3_sum_sq), 0);
            enc.set_bytes(2, 4, &dim as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
        }

        // Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + N+1 norm_w → buf_input)
        {
            // Use next layer's input_norm_w — passed via the layer_cache
            // lc is THIS layer's cache; we need NEXT layer's input_norm_w
            // For now, disable GPU combine on last layer (checked above)
            let next_norm_w = lc.input_norm_w; // caller will pass N+1 cache for combine
            if let Some(nw_off) = next_norm_w {
                let norm_off = (wf_base.add(nw_off) as *const u8).offset_from(wf_base) as u64;
                let enc = cmd.new_compute_command_encoder();
                let dim = cfg.hidden_dim as u32;
                let eps = cfg.rms_norm_eps;
                enc.set_compute_pipeline_state(&ctx.rms_norm_apply_bf16);
                enc.set_buffer(0, Some(&ctx.buf_moe_hidden), 0);
                enc.set_buffer(1, ctx.wf_buf.as_deref(), norm_off);
                enc.set_buffer(2, Some(&ctx.buf_cmd3_sum_sq), 0);
                enc.set_buffer(3, Some(&ctx.buf_input), 0);
                enc.set_bytes(4, 4, &dim as *const u32 as *const _);
                enc.set_bytes(5, 4, &eps as *const f32 as *const _);
                let tgs = (dim + 255) / 256;
                enc.dispatch_thread_groups(MTLSize::new(tgs as u64, 1, 1), MTLSize::new(256, 1, 1));
                enc.end_encoding();
            }
        }
    }

    gpu_combine
}

// ============================================================================
// 3-Command GPU pipeline — full async with GPU-side combine
// ============================================================================

/// Run one layer with the 3-command pipeline:
///   1. Wait for prev CMD3, input_norm
///   2. CMD1: attention projections (batch matvec)
///   3. CPU: attention compute
///   4. CMD2: o_proj + residual + norm + routing + shared (fused, 1 wait)
///   5. CPU: softmax + top-K + expert I/O
///   6. CMD3: experts + shared swiglu+down + GPU combine (async, deferred)
#[allow(clippy::too_many_arguments)]
unsafe fn gpu_layer_forward_3cmd(
    cfg: &ModelConfig,
    wf_data: *const u8,
    lc: &LayerWeightCache,
    _next_lc: Option<&LayerWeightCache>,  // N+1 layer cache (for GPU combine)
    hidden: &mut [f32],
    kv: Option<&mut KVCache>,
    la_state: Option<&mut LinearAttnState>,
    pos: i32,
    packed_fd: Option<&File>,
    use_2bit: bool,
    scratch: &mut CpuForwardScratch,
    ctx: &MetalCtx,
    mut expert_cache: Option<&mut ExpertLRUCache>,
    io_pool: Option<&IOPool>,
    layer_idx: usize,
    deferred: &mut Option<DeferredCmd3>,
    pred_experts: &mut [i32],
    pred_count: &mut [i32],
    pred_valid: &mut bool,
) {
    let hd = cfg.hidden_dim as usize;
    let is_full = kv.is_some();

    let u16o = |o: Option<usize>| o.map(|x| wf_data.add(x) as *const u16);
    let f32o = |o: Option<usize>| o.map(|x| wf_data.add(x) as *const f32);
    let u32o = |o: Option<usize>| o.map(|x| wf_data.add(x) as *const u32);

    // ---- Phase 0: Wait for deferred CMD3 from previous layer ----
    // The previous layer submitted CMD3 asynchronously.  Wait for it now.
    if let Some(ref mut d) = *deferred {
        finalize_cmd3(ctx, d, hidden, hd, scratch);
        // Unpin cache entries from previous CMD3
        if let Some(ref mut cache) = expert_cache {
            cache.unpin_batch(&d.pinned_entries);
        }
    }
    *deferred = None;

    // Save residual
    scratch.residual[..hd].copy_from_slice(hidden);

    // Input norm
    step_input_norm(hidden, u16o(lc.input_norm_w), &mut scratch.normed, hd, cfg.rms_norm_eps);

    // Copy normed to GPU buf_input for CMD1
    {
        let dst = ctx.buf_input.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(scratch.normed.as_ptr(), dst, hd);
    }

    // ---- Phase 1: CMD1 — attention projections (batch matvec) ----
    #[cfg(feature = "timing")]
    let t_p0 = Instant::now();
    let mut cmd1_specs: [gpu_ops::BatchMatvecSpec; 4] = [
        gpu_ops::BatchMatvecSpec { w: std::ptr::null(), scales: std::ptr::null(), biases: std::ptr::null(), out_cpu: std::ptr::null_mut(), out_dim: 0, in_dim: 0, group_size: 0, batch_slot: 0 },
        gpu_ops::BatchMatvecSpec { w: std::ptr::null(), scales: std::ptr::null(), biases: std::ptr::null(), out_cpu: std::ptr::null_mut(), out_dim: 0, in_dim: 0, group_size: 0, batch_slot: 1 },
        gpu_ops::BatchMatvecSpec { w: std::ptr::null(), scales: std::ptr::null(), biases: std::ptr::null(), out_cpu: std::ptr::null_mut(), out_dim: 0, in_dim: 0, group_size: 0, batch_slot: 2 },
        gpu_ops::BatchMatvecSpec { w: std::ptr::null(), scales: std::ptr::null(), biases: std::ptr::null(), out_cpu: std::ptr::null_mut(), out_dim: 0, in_dim: 0, group_size: 0, batch_slot: 3 },
    ];
    let num_attn_specs: usize;

    if is_full {
        let q_proj_dim = cfg.num_attn_heads as usize * cfg.head_dim as usize * 2;
        let kv_dim = cfg.num_kv_heads as usize * cfg.head_dim as usize;
        cmd1_specs[0] = gpu_ops::BatchMatvecSpec {
            w: lc.q_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.q_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.q_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.q_proj_out.as_mut_ptr(),
            out_dim: q_proj_dim as u32, in_dim: cfg.hidden_dim as u32, group_size: cfg.group_size as u32, batch_slot: 0,
        };
        cmd1_specs[1] = gpu_ops::BatchMatvecSpec {
            w: lc.k_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.k_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.k_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.k_buf.as_mut_ptr(),
            out_dim: kv_dim as u32, in_dim: cfg.hidden_dim as u32, group_size: cfg.group_size as u32, batch_slot: 1,
        };
        cmd1_specs[2] = gpu_ops::BatchMatvecSpec {
            w: lc.v_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.v_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.v_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.v_buf.as_mut_ptr(),
            out_dim: kv_dim as u32, in_dim: cfg.hidden_dim as u32, group_size: cfg.group_size as u32, batch_slot: 2,
        };
        num_attn_specs = 3;
    } else {
        let qkv_dim = cfg.linear_conv_dim as usize;
        let z_dim = cfg.linear_total_value as usize;
        let vh = cfg.linear_num_v_heads as usize;
        cmd1_specs[0] = gpu_ops::BatchMatvecSpec {
            w: lc.qkv_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.qkv_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.qkv_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.qkv_out.as_mut_ptr(),
            out_dim: qkv_dim as u32, in_dim: cfg.hidden_dim as u32, group_size: cfg.group_size as u32, batch_slot: 0,
        };
        cmd1_specs[1] = gpu_ops::BatchMatvecSpec {
            w: lc.z_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.z_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.z_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.z_out.as_mut_ptr(),
            out_dim: z_dim as u32, in_dim: cfg.hidden_dim as u32, group_size: cfg.group_size as u32, batch_slot: 1,
        };
        cmd1_specs[2] = gpu_ops::BatchMatvecSpec {
            w: lc.b_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.b_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.b_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.beta_out.as_mut_ptr(),
            out_dim: vh as u32, in_dim: cfg.hidden_dim as u32, group_size: cfg.group_size as u32, batch_slot: 2,
        };
        cmd1_specs[3] = gpu_ops::BatchMatvecSpec {
            w: lc.a_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.a_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.a_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.alpha_out.as_mut_ptr(),
            out_dim: vh as u32, in_dim: cfg.hidden_dim as u32, group_size: cfg.group_size as u32, batch_slot: 3,
        };
        num_attn_specs = 4;
    }

    // Prediction state (scoped outside CMD1 block so Phase 5 can access)
    let mut pred_handles: Vec<std::thread::JoinHandle<(usize, isize)>> = Vec::new();
    let mut pred_started = false;

    // Check GPU linear attention availability (before CMD1 encoding)
    let linear_layer_idx = if !is_full {
        layer_idx.wrapping_sub((layer_idx + 1) / cfg.full_attn_interval as usize)
    } else {
        usize::MAX
    };
    // NOTE: C code guards this with gpu_linear_attn_enabled (always 0 in C).
    // GPU linear attention in CMD1 is experimental — disabled for correctness.
    let _gpu_linear_attn_enabled = false;
    let can_gpu_linear = _gpu_linear_attn_enabled
        && !is_full
        && linear_layer_idx < ctx.buf_conv_state.len()
        && ctx.delta_net_step.is_some()
        && ctx.conv1d_step.is_some()
        && ctx.rms_norm_qk.is_some()
        && ctx.compute_decay_beta.is_some()
        && ctx.gated_rms_norm.is_some()
        && lc.conv1d_w.is_some()
        && lc.a_log.is_some()
        && lc.dt_bias.is_some()
        && lc.gated_norm_w.is_some();

    let mut gpu_attn_fuse = false;
    let mut gpu_linear_attn = false;

    if num_attn_specs > 0 {
        let cmd1 = ctx.queue.new_command_buffer();
        gpu_ops::gpu_encode_batch_matvec(ctx, &cmd1, &cmd1_specs[..num_attn_specs]);

        // GPU linear attention: encode conv1d + norm + decay/beta + delta-net + gated_norm into CMD1
        if can_gpu_linear {
            let conv_dim = cfg.linear_conv_dim as u32;
            let conv_w_off = (lc.conv1d_w.unwrap() as *const u8).offset_from(wf_data as *const u8) as u64;

            // Enc L1: conv1d_step
            {
                let enc = cmd1.new_compute_command_encoder();
                enc.set_compute_pipeline_state(ctx.conv1d_step.as_ref().unwrap());
                enc.set_buffer(0, Some(&ctx.buf_conv_state[linear_layer_idx]), 0);
                enc.set_buffer(1, Some(&ctx.batch_out[0]), 0); // qkv projection output
                enc.set_buffer(2, ctx.wf_buf.as_deref(), conv_w_off);
                enc.set_buffer(3, Some(&ctx.buf_conv_output), 0);
                enc.set_bytes(4, 4, &conv_dim as *const u32 as *const _);
                let tgs = (conv_dim + 255) / 256;
                enc.dispatch_thread_groups(MTLSize::new(tgs as u64, 1, 1), MTLSize::new(256, 1, 1));
                enc.end_encoding();
            }

            // Enc L2: rms_norm_qk
            {
                let key_dim = cfg.linear_key_dim as u32;
                let inv_scale = 1.0f32 / (cfg.linear_key_dim as f32).sqrt();
                let enc = cmd1.new_compute_command_encoder();
                enc.set_compute_pipeline_state(ctx.rms_norm_qk.as_ref().unwrap());
                enc.set_buffer(0, Some(&ctx.buf_conv_output), 0); // q at offset 0
                enc.set_buffer(1, Some(&ctx.buf_conv_output), cfg.linear_total_key as u64 * 4); // k
                enc.set_bytes(2, 4, &key_dim as *const u32 as *const _);
                enc.set_bytes(3, 4, &inv_scale as *const f32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(cfg.linear_num_k_heads as u64, 1, 1),
                    MTLSize::new(cfg.linear_key_dim as u64, 1, 1),
                );
                enc.end_encoding();
            }

            // Enc L3: compute_decay_beta
            {
                let a_log_off = (wf_data.add(lc.a_log.unwrap()) as *const u8).offset_from(wf_data as *const u8) as u64;
                let dt_bias_off = (wf_data.add(lc.dt_bias.unwrap()) as *const u8).offset_from(wf_data as *const u8) as u64;
                let enc = cmd1.new_compute_command_encoder();
                enc.set_compute_pipeline_state(ctx.compute_decay_beta.as_ref().unwrap());
                enc.set_buffer(0, Some(&ctx.batch_out[3]), 0); // alpha
                enc.set_buffer(1, Some(&ctx.batch_out[2]), 0); // beta
                enc.set_buffer(2, ctx.wf_buf.as_deref(), a_log_off);
                enc.set_buffer(3, ctx.wf_buf.as_deref(), dt_bias_off);
                enc.set_buffer(4, Some(&ctx.buf_delta_g_decay), 0);
                enc.set_buffer(5, Some(&ctx.buf_delta_beta), 0);
                enc.dispatch_thread_groups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(cfg.linear_num_v_heads as u64, 1, 1),
                );
                enc.end_encoding();
            }

            // Enc L4: gated_delta_net_step
            {
                let khpv = (cfg.linear_num_v_heads / cfg.linear_num_k_heads) as u32;
                let enc = cmd1.new_compute_command_encoder();
                enc.set_compute_pipeline_state(ctx.delta_net_step.as_ref().unwrap());
                enc.set_buffer(0, Some(&ctx.buf_delta_state[linear_layer_idx]), 0); // state
                enc.set_buffer(1, Some(&ctx.buf_conv_output), 0); // q
                enc.set_buffer(2, Some(&ctx.buf_conv_output), cfg.linear_total_key as u64 * 4); // k
                enc.set_buffer(3, Some(&ctx.buf_conv_output), 2 * cfg.linear_total_key as u64 * 4); // v
                enc.set_buffer(4, Some(&ctx.buf_delta_g_decay), 0);
                enc.set_buffer(5, Some(&ctx.buf_delta_beta), 0);
                enc.set_buffer(6, Some(&ctx.buf_delta_output), 0); // output
                enc.set_bytes(7, 4, &khpv as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(cfg.linear_num_v_heads as u64, 1, 1),
                    MTLSize::new(128, 1, 1),
                );
                enc.end_encoding();
            }

            // Enc L5: gated_rms_norm → writes to batch_out[6] for CMD2 o_proj
            {
                let gnorm_w_off = (wf_data.add(lc.gated_norm_w.unwrap()) as *const u8).offset_from(wf_data as *const u8) as u64;
                let value_dim = cfg.linear_value_dim as u32;
                let eps = cfg.rms_norm_eps;
                let enc = cmd1.new_compute_command_encoder();
                enc.set_compute_pipeline_state(ctx.gated_rms_norm.as_ref().unwrap());
                enc.set_buffer(0, Some(&ctx.buf_delta_output), 0); // values
                enc.set_buffer(1, Some(&ctx.batch_out[1]), 0); // z projection (slot 1)
                enc.set_buffer(2, ctx.wf_buf.as_deref(), gnorm_w_off);
                enc.set_buffer(3, Some(&ctx.batch_out[6]), 0); // output → batch_out[6] for CMD2
                enc.set_bytes(4, 4, &value_dim as *const u32 as *const _);
                enc.set_bytes(5, 4, &eps as *const f32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(cfg.linear_num_v_heads as u64, 1, 1),
                    MTLSize::new(cfg.linear_value_dim as u64, 1, 1),
                );
                enc.end_encoding();
            }

            gpu_linear_attn = true;
        }

        cmd1.commit();

        // ---- Phase 1.5: Launch prediction preads during CMD1 GPU execution ----
        // Spawn threads to pread predicted experts (from prev token) into buf_B.
        // These run concurrently with CMD1 on GPU and subsequent CPU work.
        pred_started = if *pred_valid && pred_count[layer_idx] > 0 {
            if let Some(fd) = packed_fd {
                let raw_fd = fd.as_raw_fd();
                let esz = active_expert_size(cfg, use_2bit);
                let base = layer_idx * MAX_K;
                for p in 0..pred_count[layer_idx] as usize {
                    let expert_idx = pred_experts[base + p];
                    let dst_addr = ctx.buf_multi_expert_data_b[p].contents() as usize;
                    let fd2 = raw_fd;
                    pred_handles.push(std::thread::spawn(move || {
                        let dst = dst_addr as *mut u8;
                        let result = unsafe { pread(fd2, dst, esz, (expert_idx as u64 * esz as u64) as i64) };
                        (p, result)
                    }));
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        cmd1.wait_until_completed();

        // Flush non-GPU results (specs 0-3 for CPU attention, specs 0-3 for GPU linear too)
        // For GPU linear: batch_out[6] already has gated_rms_norm output, no CPU flush needed
        // For CPU: flush all specs
        if !gpu_linear_attn {
            gpu_ops::gpu_flush_batch_results(ctx, &cmd1_specs[..num_attn_specs]);
        }
    }

    // ---- Phase 2: Attention compute (GPU if possible, CPU fallback) ----
    #[cfg(feature = "timing")]
    let t_p1 = Instant::now();

    let mut attn_kv_len: u32 = 0;

    if is_full {
        let kv_cache = kv.unwrap();
        let kv_dim = cfg.num_kv_heads as usize * cfg.head_dim as usize;
        let q_dim = cfg.num_attn_heads as usize * cfg.head_dim as usize;
        let fa_idx = ((layer_idx + 1) / cfg.full_attn_interval as usize).wrapping_sub(1);

        // Always run CPU prep (Q/K norm, RoPE, KV cache update) + attention compute
        cpu_full_attn_compute(
            cfg,
            &scratch.q_proj_out,
            &mut scratch.k_buf,
            &scratch.v_buf,
            u16o(lc.q_norm_w), u16o(lc.k_norm_w),
            kv_cache, pos,
            &mut scratch.attn_out,
            &mut scratch.q_buf, &mut scratch.q_gate,
        );

        // GPU attention ready when KV cache is long enough (checked after increment)
        let gpu_attn_ready = fa_idx < ctx.buf_kv_k.len()
            && ctx.attn_scores_pipe.is_some()
            && kv_cache.len >= 32
            && (kv_cache.len as i32) < cfg.gpu_kv_seq;

        if gpu_attn_ready {
            // Copy K/V cache for current position to GPU KV buffers
            let cache_pos = kv_cache.len as usize - 1;
            let elem_size = 2usize; // bf16
            let k_gpu_dst = ctx.buf_kv_k[fa_idx].contents() as *mut u8;
            let k_gpu_offset = cache_pos * kv_dim * elem_size;
            let k_cpu_src = &kv_cache.k_cache[cache_pos * kv_dim..(cache_pos + 1) * kv_dim];
            std::ptr::copy_nonoverlapping(
                k_cpu_src.as_ptr() as *const u8,
                k_gpu_dst.add(k_gpu_offset),
                kv_dim * elem_size,
            );
            let v_gpu_dst = ctx.buf_kv_v[fa_idx].contents() as *mut u8;
            let v_gpu_offset = cache_pos * kv_dim * elem_size;
            let v_cpu_src = &kv_cache.v_cache[cache_pos * kv_dim..(cache_pos + 1) * kv_dim];
            std::ptr::copy_nonoverlapping(
                v_cpu_src.as_ptr() as *const u8,
                v_gpu_dst.add(v_gpu_offset),
                kv_dim * elem_size,
            );

            // Copy Q and gate to GPU (q_buf and q_gate populated by cpu_full_attn_compute)
            let q_dst = ctx.buf_attn_q.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(scratch.q_buf.as_ptr(), q_dst, q_dim);
            let gate_dst = ctx.buf_attn_gate.contents() as *mut f32;
            let num_q_heads = cfg.num_attn_heads as usize;
            let head_dim = cfg.head_dim as usize;
            std::ptr::copy_nonoverlapping(scratch.q_gate.as_ptr(), gate_dst, num_q_heads * head_dim);

            gpu_attn_fuse = true;
        } else {
            // Copy CPU attn_out to GPU for CMD2
            let dst = ctx.batch_out[6].contents() as *mut f32;
            std::ptr::copy_nonoverlapping(scratch.attn_out.as_ptr(), dst, q_dim);
        }
        attn_kv_len = kv_cache.len as u32;
    } else {
        if !gpu_linear_attn {
            let la = la_state.unwrap();
            cpu_linear_attn_compute(
                cfg,
                &scratch.qkv_out, &scratch.z_out,
                &scratch.beta_out, &scratch.alpha_out,
                u16o(lc.conv1d_w), f32o(lc.a_log), u16o(lc.dt_bias),
                u16o(lc.gated_norm_w),
                la,
                &mut scratch.gated_out,
                &mut scratch.conv_out, &mut scratch.out_values,
            );

            // Copy gated_out to GPU for CMD2 out_proj
            let dst = ctx.batch_out[6].contents() as *mut f32;
            std::ptr::copy_nonoverlapping(
                scratch.gated_out.as_ptr(), dst,
                cfg.linear_total_value as usize,
            );
        }
        // When gpu_linear_attn: GPU already wrote to batch_out[6] via gated_rms_norm in CMD1
    }

    // Copy residual to GPU for CMD2
    {
        let dst = ctx.buf_residual.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(scratch.residual.as_ptr(), dst, hd);
    }

    // ---- Phase 3: CMD2 — o_proj + residual + norm + routing + shared ----
    #[cfg(feature = "timing")]
    let t_p2 = Instant::now();
    let (oproj_w, oproj_s, oproj_b, oproj_in_dim): (*const u32, *const u16, *const u16, u32) = if is_full {
        (
            lc.o_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            lc.o_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            lc.o_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            (cfg.num_attn_heads * cfg.head_dim) as u32,
        )
    } else {
        (
            lc.out_proj_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            lc.out_proj_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            lc.out_proj_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            cfg.linear_total_value as u32,
        )
    };

    // moe_specs for reading back routing results after CMD2
    let mut moe_specs_readback: [gpu_ops::BatchMatvecSpec; 4] = [
        gpu_ops::BatchMatvecSpec { w: std::ptr::null(), scales: std::ptr::null(), biases: std::ptr::null(), out_cpu: scratch.gate_scores.as_mut_ptr(), out_dim: cfg.num_experts as u32, in_dim: 0, group_size: 0, batch_slot: 0 },
        gpu_ops::BatchMatvecSpec { w: std::ptr::null(), scales: std::ptr::null(), biases: std::ptr::null(), out_cpu: scratch.shared_out.as_mut_ptr(), out_dim: cfg.shared_intermediate as u32, in_dim: 0, group_size: 0, batch_slot: 1 },
        gpu_ops::BatchMatvecSpec { w: std::ptr::null(), scales: std::ptr::null(), biases: std::ptr::null(), out_cpu: scratch.moe_out.as_mut_ptr(), out_dim: cfg.shared_intermediate as u32, in_dim: 0, group_size: 0, batch_slot: 2 },
        gpu_ops::BatchMatvecSpec { w: std::ptr::null(), scales: std::ptr::null(), biases: std::ptr::null(), out_cpu: std::ptr::null_mut(), out_dim: 1, in_dim: 0, group_size: 0, batch_slot: 3 },
    ];

    let cmd2 = ctx.queue.new_command_buffer();
    let attn_src: &metal::Buffer = if gpu_attn_fuse { &ctx.buf_attn_out } else { &ctx.batch_out[6] };
    encode_cmd2_fused(
        cfg, ctx, &cmd2, lc,
        oproj_w, oproj_s, oproj_b, oproj_in_dim,
        attn_src,
        wf_data,
        gpu_attn_fuse,
        ((layer_idx + 1) / cfg.full_attn_interval as usize).wrapping_sub(1),
        attn_kv_len,
    );
    cmd2.commit();
    cmd2.wait_until_completed();

    // Read back CMD2 results
    // h_mid (hidden = residual + o_proj)
    {
        let src = ctx.buf_h_mid.contents() as *const f32;
        std::ptr::copy_nonoverlapping(src, hidden.as_mut_ptr(), hd);
        scratch.h_mid[..hd].copy_from_slice(&hidden[..hd]);
    }
    // h_post (normed hidden for MoE)
    {
        let src = ctx.buf_input.contents() as *const f32;
        std::ptr::copy_nonoverlapping(src, scratch.h_post.as_mut_ptr(), hd);
    }
    // Routing results (batch slots 0-3)
    moe_specs_readback[0].out_cpu = scratch.gate_scores.as_mut_ptr();
    moe_specs_readback[1].out_cpu = scratch.shared_out.as_mut_ptr(); // shared_gate
    moe_specs_readback[2].out_cpu = scratch.moe_out.as_mut_ptr();    // shared_up
    moe_specs_readback[3].out_cpu = scratch.expert_tmp.as_mut_ptr(); // shared_gate_score (scalar)
    gpu_ops::gpu_flush_batch_results(ctx, &moe_specs_readback[..4]);

    // ---- Phase 4: CPU softmax + top-K ----
    #[cfg(feature = "timing")]
    let t_p3 = Instant::now();
    scratch.gate_scores[..cfg.num_experts as usize].fill(0.0);
    let routing = step_moe_routing(
        cfg,
        lc.gate_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
        lc.gate_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
        lc.gate_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
        u32o(lc.seg_w), u16o(lc.seg_s), u16o(lc.seg_b),
        &scratch.h_post,
        &mut scratch.gate_scores,
    );
    let actual_k = routing.expert_indices.len().min(MAX_K);

    // ---- Phase 5: Expert I/O (prediction-aware) ----
    #[cfg(feature = "timing")]
    let t_p4 = Instant::now();
    let esz = active_expert_size(cfg, use_2bit);

    let mut expert_data_bufs: Vec<Option<metal::Buffer>> = Vec::with_capacity(actual_k);
    let mut pinned_entries: Vec<usize> = Vec::new();

    // Initialize all slots to None
    for _ in 0..actual_k { expert_data_bufs.push(None); }

    if let Some(fd) = packed_fd {
        let base = layer_idx * MAX_K;

        if pred_started {
            // ---- Prediction path: wait for preads, match against actual routing ----
            let mut pred_results: Vec<(usize, isize)> = Vec::new();
            for h in pred_handles.drain(..) {
                if let Ok(r) = h.join() { pred_results.push(r); }
            }

            let mut miss_ei: Vec<i32> = Vec::with_capacity(actual_k);
            let mut miss_slots: Vec<usize> = Vec::with_capacity(actual_k);
            let mut hit_count: usize = 0;

            for k in 0..actual_k {
                let mut found = false;
                for &(p_slot, result) in &pred_results {
                    if result == esz as isize
                        && routing.expert_indices[k] == pred_experts[base + p_slot]
                    {
                        // Hit! Expert was pre-loaded into buf_multi_expert_data_b[p_slot]
                        expert_data_bufs[k] = Some(ctx.buf_multi_expert_data_b[p_slot].clone());
                        found = true;
                        hit_count += 1;
                        break;
                    }
                }
                if !found {
                    miss_ei.push(routing.expert_indices[k]);
                    miss_slots.push(k);
                }
            }
            #[cfg(feature = "timing")]
            if hit_count > 0 || miss_ei.len() > 0 {
                if pos == 0 { eprintln!("[L{:02}-pred] hits={} misses={}", layer_idx, hit_count, miss_ei.len()); }
            }

            // ---- Read misses via direct I/O (OS page cache, no LRU) ----
            if !miss_ei.is_empty() {
                let num_misses = miss_ei.len();
                let raw_fd = fd.as_raw_fd();
                if let Some(pool) = io_pool {
                    let mut tasks: Vec<crate::expert_io::IOPreadTask> = Vec::with_capacity(num_misses);
                    for i in 0..num_misses {
                        let k = miss_slots[i];
                        tasks.push(crate::expert_io::IOPreadTask {
                            fd: raw_fd,
                            dst: ctx.buf_multi_expert_data[k].contents() as *mut u8,
                            offset: miss_ei[i] as u64 * esz as u64,
                            size: esz,
                            result: 0,
                            lz4_comp_buf: std::ptr::null_mut(),
                            lz4_comp_size: 0,
                        });
                    }
                    pool.dispatch(&mut tasks);
                    for i in 0..num_misses {
                        let k = miss_slots[i];
                        if tasks[i].result == esz as isize {
                            expert_data_bufs[k] = Some(ctx.buf_multi_expert_data[k].clone());
                        }
                    }
                }
            }
        } else {
            // ---- No prediction: direct I/O path (OS page cache only, no LRU) ----
            let valid = direct_expert_read(
                fd, &routing.expert_indices[..actual_k], actual_k, esz,
                &ctx.buf_multi_expert_data, io_pool,
            );
            for k in 0..actual_k {
                if valid[k] {
                    expert_data_bufs[k] = Some(ctx.buf_multi_expert_data[k].clone());
                }
            }
        }
    }

    // Copy h_post to GPU multi-expert input
    {
        let dst = ctx.buf_multi_expert_input.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(scratch.h_post.as_ptr(), dst, hd);
    }

    // Copy shared gate/up to GPU buffers
    {
        let src_gate = ctx.batch_out[1].contents() as *const f32;
        let dst_gate = ctx.buf_shared_gate.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(src_gate, dst_gate, cfg.shared_intermediate as usize);
        let src_up = ctx.batch_out[2].contents() as *const f32;
        let dst_up = ctx.buf_shared_up.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(src_up, dst_up, cfg.shared_intermediate as usize);
    }

    // ---- Phase 6: CMD3 — experts + shared + GPU combine (ASYNC) ----
    #[cfg(feature = "timing")]
    let t_p5 = Instant::now();
    let cmd3_ref = ctx.queue.new_command_buffer();

    let mut valid_arr = [false; MAX_K];
    let mut expert_weights_arr = [0.0f32; MAX_K];
    for k in 0..actual_k {
        valid_arr[k] = expert_data_bufs[k].is_some();
        expert_weights_arr[k] = routing.expert_weights[k];
    }

    // Convert owned buffers to refs for encoding
    let expert_buf_refs: Vec<Option<&metal::Buffer>> = expert_data_bufs.iter().map(|b| b.as_ref()).collect();
    let gpu_combine = encode_cmd3_experts(
        cfg, ctx, cmd3_ref, lc,
        actual_k, routing.shared_gate_score,
        &expert_weights_arr, &valid_arr,
        use_2bit, wf_data, layer_idx,
        &expert_buf_refs,
    );

    // Commit CMD3 asynchronously — don't wait.
    // The next layer will wait for it in Phase 0 (or complete_deferred_experts
    // for the last layer).
    cmd3_ref.commit();

    // Store deferred CMD3 for the next layer to consume.
    // Retain the CommandBuffer so it survives beyond this function.
    let owned_cmd: metal::CommandBuffer = cmd3_ref.to_owned();
    *deferred = Some(DeferredCmd3 {
        cmd_buf: Some(owned_cmd),
        gpu_combined: gpu_combine,
        actual_k,
        shared_gate_score: routing.shared_gate_score,
        expert_weights: expert_weights_arr,
        valid: valid_arr,
        pinned_entries,
        event_value: 0,
    });

    // ---- Save routing for next token's temporal prediction ----
    // MUST happen AFTER prediction hit check (which reads pred_experts).
    let base = layer_idx * MAX_K;
    for k in 0..actual_k {
        pred_experts[base + k] = routing.expert_indices[k];
    }
    pred_count[layer_idx] = actual_k as i32;
    if layer_idx == cfg.num_layers as usize - 1 {
        *pred_valid = true;
    }

    #[cfg(feature = "timing")]
    {
        let t_p6 = Instant::now();
        eprintln!(
            "[L{:02}-3cmd] cmd1={}µs cpu_attn={}µs cmd2={}µs routing={}µs expert_io={}µs cmd3={}µs total={}µs",
            layer_idx,
            t_p1.duration_since(t_p0).as_micros(),
            t_p2.duration_since(t_p1).as_micros(),
            t_p3.duration_since(t_p2).as_micros(),
            t_p4.duration_since(t_p3).as_micros(),
            t_p5.duration_since(t_p4).as_micros(),
            t_p6.duration_since(t_p5).as_micros(),
            t_p6.duration_since(t_p0).as_micros(),
        );
    }
}

// ============================================================================
// 2-Command GPU pipeline — simpler, sync, no deferred CMD3
// ============================================================================

/// Run one layer with the 2-command pipeline:
///   1. CMD1: attention projections (batch matvec) → wait
///   2. CPU: attention compute + o_proj(CPU) + residual + norm + routing + I/O
///   3. CMD2: experts + shared + combine → sync wait → readback → final combine
#[allow(clippy::too_many_arguments)]
unsafe fn gpu_layer_forward_2cmd(
    cfg: &ModelConfig,
    wf_data: *const u8,
    lc: &LayerWeightCache,
    hidden: &mut [f32],
    kv: Option<&mut KVCache>,
    la_state: Option<&mut LinearAttnState>,
    pos: i32,
    packed_fd: Option<&File>,
    use_2bit: bool,
    scratch: &mut CpuForwardScratch,
    ctx: &MetalCtx,
    mut expert_cache: Option<&mut ExpertLRUCache>,
    io_pool: Option<&IOPool>,
    layer_idx: usize,
    _pred_experts: &mut [i32],
    _pred_count: &mut [i32],
    _pred_valid: &mut bool,
) {
    let hd = cfg.hidden_dim as usize;
    let is_full = kv.is_some();
    let gs = cfg.group_size as usize;

    let u16o = |o: Option<usize>| o.map(|x| wf_data.add(x) as *const u16);
    let f32o = |o: Option<usize>| o.map(|x| wf_data.add(x) as *const f32);

    // Save residual
    scratch.residual[..hd].copy_from_slice(hidden);

    // Input norm
    step_input_norm(hidden, u16o(lc.input_norm_w), &mut scratch.normed, hd, cfg.rms_norm_eps);

    // Copy normed to GPU buf_input for CMD1
    {
        let dst = ctx.buf_input.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(scratch.normed.as_ptr(), dst, hd);
    }

    // ---- CMD1: Attention projections (batch matvec) ----
    let mut cmd1_specs: [gpu_ops::BatchMatvecSpec; 4] = [
        gpu_ops::BatchMatvecSpec { w: std::ptr::null(), scales: std::ptr::null(), biases: std::ptr::null(), out_cpu: std::ptr::null_mut(), out_dim: 0, in_dim: 0, group_size: 0, batch_slot: 0 },
        gpu_ops::BatchMatvecSpec { w: std::ptr::null(), scales: std::ptr::null(), biases: std::ptr::null(), out_cpu: std::ptr::null_mut(), out_dim: 0, in_dim: 0, group_size: 0, batch_slot: 1 },
        gpu_ops::BatchMatvecSpec { w: std::ptr::null(), scales: std::ptr::null(), biases: std::ptr::null(), out_cpu: std::ptr::null_mut(), out_dim: 0, in_dim: 0, group_size: 0, batch_slot: 2 },
        gpu_ops::BatchMatvecSpec { w: std::ptr::null(), scales: std::ptr::null(), biases: std::ptr::null(), out_cpu: std::ptr::null_mut(), out_dim: 0, in_dim: 0, group_size: 0, batch_slot: 3 },
    ];
    let num_attn_specs: usize;

    if is_full {
        let q_proj_dim = cfg.num_attn_heads as usize * cfg.head_dim as usize * 2;
        let kv_dim = cfg.num_kv_heads as usize * cfg.head_dim as usize;
        cmd1_specs[0] = gpu_ops::BatchMatvecSpec {
            w: lc.q_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.q_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.q_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.q_proj_out.as_mut_ptr(),
            out_dim: q_proj_dim as u32, in_dim: cfg.hidden_dim as u32, group_size: gs as u32, batch_slot: 0,
        };
        cmd1_specs[1] = gpu_ops::BatchMatvecSpec {
            w: lc.k_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.k_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.k_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.k_buf.as_mut_ptr(),
            out_dim: kv_dim as u32, in_dim: cfg.hidden_dim as u32, group_size: gs as u32, batch_slot: 1,
        };
        cmd1_specs[2] = gpu_ops::BatchMatvecSpec {
            w: lc.v_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.v_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.v_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.v_buf.as_mut_ptr(),
            out_dim: kv_dim as u32, in_dim: cfg.hidden_dim as u32, group_size: gs as u32, batch_slot: 2,
        };
        num_attn_specs = 3;
    } else {
        let qkv_dim = cfg.linear_conv_dim as usize;
        let z_dim = cfg.linear_total_value as usize;
        let vh = cfg.linear_num_v_heads as usize;
        cmd1_specs[0] = gpu_ops::BatchMatvecSpec {
            w: lc.qkv_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.qkv_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.qkv_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.qkv_out.as_mut_ptr(),
            out_dim: qkv_dim as u32, in_dim: cfg.hidden_dim as u32, group_size: gs as u32, batch_slot: 0,
        };
        cmd1_specs[1] = gpu_ops::BatchMatvecSpec {
            w: lc.z_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.z_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.z_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.z_out.as_mut_ptr(),
            out_dim: z_dim as u32, in_dim: cfg.hidden_dim as u32, group_size: gs as u32, batch_slot: 1,
        };
        cmd1_specs[2] = gpu_ops::BatchMatvecSpec {
            w: lc.b_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.b_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.b_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.beta_out.as_mut_ptr(),
            out_dim: vh as u32, in_dim: cfg.hidden_dim as u32, group_size: gs as u32, batch_slot: 2,
        };
        cmd1_specs[3] = gpu_ops::BatchMatvecSpec {
            w: lc.a_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
            scales: lc.a_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            biases: lc.a_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
            out_cpu: scratch.alpha_out.as_mut_ptr(),
            out_dim: vh as u32, in_dim: cfg.hidden_dim as u32, group_size: gs as u32, batch_slot: 3,
        };
        num_attn_specs = 4;
    }

    if num_attn_specs > 0 {
        let cmd1 = ctx.queue.new_command_buffer();
        gpu_ops::gpu_encode_batch_matvec(ctx, &cmd1, &cmd1_specs[..num_attn_specs]);
        cmd1.commit();
        cmd1.wait_until_completed();
        gpu_ops::gpu_flush_batch_results(ctx, &cmd1_specs[..num_attn_specs]);
    }

    // DEBUG: Compare GPU path vs full CPU path for first linear layer
    // Save GPU-path hidden for comparison
    let mut gpu_hidden_before_layer = if pos == 0 && layer_idx == 0 && !is_full {
        hidden.to_vec()
    } else {
        Vec::new()
    };

    // ---- CPU: Attention compute ----
    if is_full {
        let kv_cache = kv.unwrap();
        cpu_full_attn_compute(
            cfg,
            &scratch.q_proj_out,
            &mut scratch.k_buf,
            &scratch.v_buf,
            u16o(lc.q_norm_w), u16o(lc.k_norm_w),
            kv_cache, pos,
            &mut scratch.attn_out,
            &mut scratch.q_buf, &mut scratch.q_gate,
        );

        // o_proj on CPU (2cmd mode)
        scratch.attn_proj[..hd].fill(0.0);
        let oproj_in_dim = cfg.num_attn_heads as usize * cfg.head_dim as usize;
        let o_w = lc.o_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32);
        let o_s = lc.o_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16);
        let o_b = lc.o_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16);
        if !o_w.is_null() {
            let num_groups = oproj_in_dim / gs;
            let packed_cols = oproj_in_dim / 8;
            cpu_dequant_matvec(
                std::slice::from_raw_parts(o_w, hd * packed_cols),
                std::slice::from_raw_parts(o_s, hd * num_groups),
                std::slice::from_raw_parts(o_b, hd * num_groups),
                &scratch.attn_out, &mut scratch.attn_proj, hd, oproj_in_dim, gs,
            );
        }
    } else {
        let la = la_state.unwrap();
        let linear_layer_idx = layer_idx.wrapping_sub((layer_idx + 1) / cfg.full_attn_interval as usize);

        // Determine if GPU delta-net is available (matches C version's logic)
        // TODO: GPU delta-net shader needs debugging — produces garbled output when enabled
        let use_gpu_delta = false;

        if use_gpu_delta {
            // ---- CPU: conv1d + q/k norm + decay/beta (same as C) ----
            let qkv_dim = cfg.linear_conv_dim as usize;
            let num_kh = cfg.linear_num_k_heads as usize;
            let key_dim = cfg.linear_key_dim as usize;
            let val_dim = cfg.linear_value_dim as usize;
            let num_vh = cfg.linear_num_v_heads as usize;
            let total_key = cfg.linear_total_key as usize;

            // Conv1d step
            scratch.conv_out[..qkv_dim].fill(0.0);
            if let Some(cw) = u16o(lc.conv1d_w) {
                let cw_slice = unsafe {
                    std::slice::from_raw_parts(cw, qkv_dim * cfg.conv_kernel_size as usize)
                };
                crate::kernels::cpu_conv1d_step(
                    &la.conv_state, &scratch.qkv_out, cw_slice,
                    &mut scratch.conv_out[..qkv_dim],
                    qkv_dim, cfg.conv_kernel_size as usize,
                );
            }
            // Update conv state
            let state_skip = qkv_dim;
            let state_len = la.conv_state.len();
            if state_len > state_skip {
                la.conv_state.copy_within(state_skip.., 0);
                let dst = state_len - state_skip;
                la.conv_state[dst..].copy_from_slice(&scratch.qkv_out);
            }

            // Split into q, k, v — use split_at_mut to avoid borrow conflicts
            let (conv_prefix, lin_v) = scratch.conv_out.split_at_mut(2 * total_key);
            let (lin_q_conv, lin_k_conv) = conv_prefix.split_at_mut(total_key);

            // RMS norm q and k (in-place on conv_out splits)
            let inv_scale = 1.0 / (key_dim as f32).sqrt();
            for h in 0..num_kh {
                let off = h * key_dim;
                let sq: f32 = lin_q_conv[off..off+key_dim].iter().map(|x| x*x).sum();
                let inv_rms = 1.0 / (sq / key_dim as f32 + cfg.rms_norm_eps).sqrt();
                let q_scale = inv_scale * inv_scale;
                for d in 0..key_dim { lin_q_conv[off+d] *= inv_rms * q_scale; }
            }
            for h in 0..num_kh {
                let off = h * key_dim;
                let sq: f32 = lin_k_conv[off..off+key_dim].iter().map(|x| x*x).sum();
                let inv_rms = 1.0 / (sq / key_dim as f32 + cfg.rms_norm_eps).sqrt();
                for d in 0..key_dim { lin_k_conv[off+d] *= inv_rms * inv_scale; }
            }

            // Compute g_decay, beta_gate
            let k_heads_per_v = num_vh / num_kh;
            for vh in 0..num_vh {
                let a_val = scratch.alpha_out[vh];
                let dt_b = u16o(lc.dt_bias).map_or(0.0, |o| {
                    crate::kernels::bf16_to_f32(unsafe { *o.add(vh) })
                });
                let a_log_val = f32o(lc.a_log).map_or(1.0f32, |o| unsafe { *o.add(vh) });
                let softplus = (1.0 + (a_val + dt_b).exp()).ln();
                scratch.g_decay[vh] = (-a_log_val.exp() * softplus).exp();
                scratch.beta_gate[vh] = crate::kernels::cpu_sigmoid(scratch.beta_out[vh]);
            }

            // Upload to GPU delta buffers
            let dq_ptr = ctx.buf_delta_q.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(lin_q_conv.as_ptr(), dq_ptr, total_key);
            let dk_ptr = ctx.buf_delta_k.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(lin_k_conv.as_ptr(), dk_ptr, total_key);
            let dv_ptr = ctx.buf_delta_v.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(lin_v.as_ptr(), dv_ptr, cfg.linear_total_value as usize);
            let dg_ptr = ctx.buf_delta_g_decay.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(scratch.g_decay.as_ptr(), dg_ptr, num_vh);
            let db_ptr = ctx.buf_delta_beta.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(scratch.beta_gate.as_ptr(), db_ptr, num_vh);

            // Dispatch GPU delta_net_step
            {
                let cmd_dn = ctx.queue.new_command_buffer();
                let enc = cmd_dn.new_compute_command_encoder();
                enc.set_compute_pipeline_state(ctx.delta_net_step.as_ref().unwrap());
                enc.set_buffer(0, Some(&ctx.buf_delta_state[linear_layer_idx]), 0);
                enc.set_buffer(1, Some(&ctx.buf_delta_q), 0);
                enc.set_buffer(2, Some(&ctx.buf_delta_k), 0);
                enc.set_buffer(3, Some(&ctx.buf_delta_v), 0);
                enc.set_buffer(4, Some(&ctx.buf_delta_g_decay), 0);
                enc.set_buffer(5, Some(&ctx.buf_delta_beta), 0);
                enc.set_buffer(6, Some(&ctx.buf_delta_output), 0);
                let khpv = k_heads_per_v as u32;
                enc.set_bytes(7, 4, &khpv as *const u32 as *const _);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_vh as u64, 1, 1),
                    MTLSize::new(128, 1, 1),
                );
                enc.end_encoding();
                cmd_dn.commit();
                cmd_dn.wait_until_completed();
            }

            // Read back GPU output
            let do_ptr = ctx.buf_delta_output.contents() as *const f32;
            scratch.out_values[..cfg.linear_total_value as usize].fill(0.0);
            std::ptr::copy_nonoverlapping(
                do_ptr,
                scratch.out_values.as_mut_ptr(),
                cfg.linear_total_value as usize,
            );

            // RMSNormGated on CPU
            let z_dim = cfg.linear_total_value as usize;
            scratch.gated_out[..z_dim].fill(0.0);
            for vh in 0..num_vh {
                let oh = &scratch.out_values[vh * val_dim..(vh + 1) * val_dim];
                let zh = &scratch.z_out[vh * val_dim..(vh + 1) * val_dim];
                let gh = &mut scratch.gated_out[vh * val_dim..(vh + 1) * val_dim];
                if let Some(gnw) = u16o(lc.gated_norm_w) {
                    let gnw_s = unsafe { std::slice::from_raw_parts(gnw, val_dim) };
                    let sum_sq: f32 = oh.iter().map(|v| v * v).sum();
                    let inv_rms = 1.0 / (sum_sq / val_dim as f32 + cfg.rms_norm_eps).sqrt();
                    for i in 0..val_dim {
                        let w = crate::kernels::bf16_to_f32(gnw_s[i]);
                        let silu_z = zh[i] / (1.0 + (-zh[i]).exp());
                        gh[i] = oh[i] * inv_rms * w * silu_z;
                    }
                } else {
                    gh.copy_from_slice(oh);
                }
            }
        } else {
            // CPU fallback: full linear attention on CPU
            cpu_linear_attn_compute(
                cfg,
                &scratch.qkv_out, &scratch.z_out,
                &scratch.beta_out, &scratch.alpha_out,
                u16o(lc.conv1d_w), f32o(lc.a_log), u16o(lc.dt_bias),
                u16o(lc.gated_norm_w),
                la,
                &mut scratch.gated_out,
                &mut scratch.conv_out, &mut scratch.out_values,
            );
        }

        // DEBUG: print intermediate RMS for linear attn layers
        if pos == 0 && layer_idx <= 7 {
            let sq_qkv: f32 = scratch.qkv_out.iter().map(|x| x*x).sum();
            let sq_z: f32 = scratch.z_out.iter().map(|x| x*x).sum();
            let sq_gated: f32 = scratch.gated_out.iter().map(|x| x*x).sum();
            let rms_qkv = (sq_qkv / scratch.qkv_out.len() as f32).sqrt();
            let rms_z = (sq_z / scratch.z_out.len() as f32).sqrt();
            let rms_gated = (sq_gated / scratch.gated_out.len() as f32).sqrt();
            let rms_ssm = if use_gpu_delta {
                // GPU state, can't easily read
                0.0
            } else {
                let sq_ssm: f32 = la.ssm_state.iter().map(|x| x*x).sum();
                (sq_ssm / la.ssm_state.len() as f32).sqrt()
            };
            eprintln!("[DBG-L{:02}] qkv_rms={:.6} z_rms={:.6} gated_rms={:.6} ssm_rms={:.6} gpu_delta={}",
                layer_idx, rms_qkv, rms_z, rms_gated, rms_ssm, use_gpu_delta);
        }

        // out_proj on CPU (2cmd mode)
        scratch.attn_proj[..hd].fill(0.0);
        let oproj_in_dim = cfg.linear_total_value as usize;
        let o_w = lc.out_proj_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32);
        let o_s = lc.out_proj_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16);
        let o_b = lc.out_proj_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16);
        if !o_w.is_null() {
            let num_groups = oproj_in_dim / gs;
            let packed_cols = oproj_in_dim / 8;
            cpu_dequant_matvec(
                std::slice::from_raw_parts(o_w, hd * packed_cols),
                std::slice::from_raw_parts(o_s, hd * num_groups),
                std::slice::from_raw_parts(o_b, hd * num_groups),
                &scratch.gated_out, &mut scratch.attn_proj, hd, oproj_in_dim, gs,
            );
        }
    }

    // Apply residual
    for i in 0..hd {
        hidden[i] = scratch.residual[i] + scratch.attn_proj[i];
    }
    scratch.h_mid[..hd].copy_from_slice(hidden);

    // DEBUG: print after residual
    if pos == 0 && layer_idx <= 7 {
        let sq_proj: f32 = scratch.attn_proj.iter().map(|x| x*x).sum();
        let sq_hid: f32 = hidden.iter().map(|x| x*x).sum();
        let rms_proj = (sq_proj / hd as f32).sqrt();
        let rms_hid = (sq_hid / hd as f32).sqrt();
        eprintln!("[DBG-L{:02}] attn_proj_rms={:.6} hidden_after_residual_rms={:.6}",
            layer_idx, rms_proj, rms_hid);
    }

    // Post-attention norm
    step_input_norm(hidden, u16o(lc.post_attn_norm_w), &mut scratch.h_post, hd, cfg.rms_norm_eps);

    // DEBUG: print after post-attn norm
    if pos == 0 && layer_idx <= 7 {
        let sq_post: f32 = scratch.h_post.iter().map(|x| x*x).sum();
        let rms_post = (sq_post / hd as f32).sqrt();
        eprintln!("[DBG-L{:02}] h_post_rms={:.6}", layer_idx, rms_post);
    }

    // ---- CPU: MoE routing ----
    let routing = step_moe_routing(
        cfg,
        lc.gate_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32),
        lc.gate_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
        lc.gate_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16),
        lc.seg_w.map(|o| wf_data.add(o) as *const u32),
        lc.seg_s.map(|o| wf_data.add(o) as *const u16),
        lc.seg_b.map(|o| wf_data.add(o) as *const u16),
        &scratch.h_post,
        &mut scratch.gate_scores,
    );
    let actual_k = routing.expert_indices.len().min(MAX_K);

    // ---- GPU: Expert forward + Shared expert (CMD2) ----
    // Upload h_post to GPU
    {
        let dst = ctx.buf_multi_expert_input.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(scratch.h_post.as_ptr(), dst, hd);
    }

    // Expert I/O (direct read into GPU buffers)
    let esz = active_expert_size(cfg, use_2bit);
    let mut expert_data_bufs: Vec<Option<metal::Buffer>> = Vec::new();
    if let Some(fd) = packed_fd {
        let valid = direct_expert_read(
            fd, &routing.expert_indices[..actual_k], actual_k, esz,
            &ctx.buf_multi_expert_data, io_pool,
        );
        for k in 0..actual_k {
            if valid[k] {
                expert_data_bufs.push(Some(ctx.buf_multi_expert_data[k].clone()));
            } else {
                expert_data_bufs.push(None);
            }
        }
    }

    let cmd2 = ctx.queue.new_command_buffer();

    // Expert forwards with zero-copy cache buffers
    for k in 0..actual_k {
        let data_buf = expert_data_bufs.get(k).and_then(|b| b.as_ref());
        gpu_ops::gpu_encode_expert_forward_slot(cfg, ctx, &cmd2, k, use_2bit, data_buf);
    }

    // Shared expert gate+up+swiglu+down
    let sg_w = lc.sg_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32);
    let sg_s = lc.sg_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16);
    let sg_b = lc.sg_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16);
    let su_w = lc.su_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32);
    let su_s = lc.su_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16);
    let su_b = lc.su_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16);
    let sd_w = lc.sd_w.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u32);
    let sd_s = lc.sd_s.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16);
    let sd_b = lc.sd_b.map_or(std::ptr::null(), |o| wf_data.add(o) as *const u16);
    gpu_ops::gpu_encode_shared_down_swiglu(cfg, ctx, &cmd2, sg_w, sg_s, sg_b, su_w, su_s, su_b, sd_w, sd_s, sd_b);

    cmd2.commit();
    cmd2.wait_until_completed();

    // Manual combine on CPU
    scratch.moe_out[..hd].fill(0.0);
    for k in 0..actual_k {
        let out_ptr = ctx.buf_multi_expert_out[k].contents() as *const f32;
        cpu_vec_madd(&mut scratch.moe_out,
            std::slice::from_raw_parts(out_ptr, hd),
            routing.expert_weights[k], hd);
    }
    scratch.shared_out[..hd].fill(0.0);
    let shared_ptr = ctx.buf_shared_out.contents() as *const f32;
    let sw = crate::kernels::cpu_sigmoid(routing.shared_gate_score);
    for i in 0..hd {
        scratch.shared_out[i] = (*shared_ptr.add(i)) * sw;
    }

    // ---- Final combine ----
    step_final_combine(&scratch.h_mid, &scratch.moe_out, &scratch.shared_out, hidden, hd);

    // DEBUG: print after final combine
    if pos == 0 && layer_idx <= 7 {
        let sq_moe: f32 = scratch.moe_out.iter().map(|x| x*x).sum();
        let sq_sh: f32 = scratch.shared_out.iter().map(|x| x*x).sum();
        let sq_hid: f32 = hidden.iter().map(|x| x*x).sum();
        let rms_moe = (sq_moe / hd as f32).sqrt();
        let rms_sh = (sq_sh / hd as f32).sqrt();
        let rms_hid = (sq_hid / hd as f32).sqrt();
        eprintln!("[DBG-L{:02}] after_combine: moe_rms={:.6} shared_rms={:.6} final_hidden_rms={:.6}",
            layer_idx, rms_moe, rms_sh, rms_hid);
    }
}

// ============================================================================
// Main dispatch — select between 3-command / 2-command / CPU fallback
// ============================================================================

/// Run one GPU-accelerated layer.  Chooses between 3cmd and 2cmd modes.
/// Falls back to CPU path when Metal is unavailable.
///
/// # Safety
///
/// `wf_data` must point to a valid mmap'd weight file. `lc` offsets must
/// be valid within that mapping. Expert file descriptors must be open.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gpu_layer_forward(
    cfg: &ModelConfig,
    wf_data: *const u8,
    layer_caches: &[LayerWeightCache],
    hidden: &mut [f32],
    kv: Option<&mut KVCache>,
    la_state: Option<&mut LinearAttnState>,
    pos: i32,
    packed_fd: Option<&File>,
    use_2bit: bool,
    scratch: &mut CpuForwardScratch,
    metal_ctx: Option<&MetalCtx>,
    mut expert_cache: Option<&mut ExpertLRUCache>,
    io_pool: Option<&IOPool>,
    layer_idx: usize,
    deferred: &mut Option<DeferredCmd3>,
    mode: GpuMode,
    pred_experts: &mut [i32],
    pred_count: &mut [i32],
    pred_valid: &mut bool,
) {
    let lc = &layer_caches[layer_idx];
    let hd = cfg.hidden_dim as usize;
    let is_full = kv.is_some();
    let gpu_available = metal_ctx.is_some() && metal_ctx.as_ref().unwrap().wf_buf.is_some();

    // Enable GPU dequant for all attention/routing projections via thread-local
    crate::cpu_forward::set_tl_gpu_ctx(metal_ctx);

    // Helper: offset → pointer (null fallback for required weights)
    let u32p = |o: Option<usize>| o.map_or(std::ptr::null(), |x| wf_data.add(x) as *const u32);
    let u16p = |o: Option<usize>| o.map_or(std::ptr::null(), |x| wf_data.add(x) as *const u16);
    let u16o = |o: Option<usize>| o.map(|x| wf_data.add(x) as *const u16);
    let f32o = |o: Option<usize>| o.map(|x| wf_data.add(x) as *const f32);
    let u32o = |o: Option<usize>| o.map(|x| wf_data.add(x) as *const u32);

    // -- save residual --
    scratch.residual[..hd].copy_from_slice(hidden);

    // -- step 1: input norm --
    step_input_norm(hidden, u16o(lc.input_norm_w), &mut scratch.normed, hd, cfg.rms_norm_eps);

    // -- step 2: attention (CPU) --
    if is_full {
        step_full_attention(
            cfg, wf_data,
            u32p(lc.q_w), u16p(lc.q_s), u16p(lc.q_b),
            u32p(lc.k_w), u16p(lc.k_s), u16p(lc.k_b),
            u32p(lc.v_w), u16p(lc.v_s), u16p(lc.v_b),
            u32p(lc.o_w), u16p(lc.o_s), u16p(lc.o_b),
            u16o(lc.q_norm_w), u16o(lc.k_norm_w),
            &scratch.normed, kv.unwrap(), pos,
            &mut scratch.attn_proj,
            &mut scratch.q_proj_out, &mut scratch.q_buf, &mut scratch.q_gate,
            &mut scratch.k_buf, &mut scratch.v_buf, &mut scratch.attn_out,
        );
    } else {
        step_linear_attention(
            cfg, wf_data,
            u32p(lc.qkv_w), u16p(lc.qkv_s), u16p(lc.qkv_b),
            u32p(lc.z_w), u16p(lc.z_s), u16p(lc.z_b),
            u32p(lc.b_w), u16p(lc.b_s), u16p(lc.b_b),
            u32p(lc.a_w), u16p(lc.a_s), u16p(lc.a_b),
            u16o(lc.conv1d_w), f32o(lc.a_log), u16o(lc.dt_bias),
            u16o(lc.gated_norm_w),
            u32p(lc.out_proj_w), u16p(lc.out_proj_s), u16p(lc.out_proj_b),
            &scratch.normed, la_state.unwrap(),
            &mut scratch.attn_proj,
            &mut scratch.qkv_out, &mut scratch.z_out,
            &mut scratch.beta_out, &mut scratch.alpha_out,
            &mut scratch.conv_out, &mut scratch.out_values, &mut scratch.gated_out,
        );
    }

    // -- apply residual: hidden = residual + attn_proj --
    for i in 0..hd {
        hidden[i] = scratch.residual[i] + scratch.attn_proj[i];
    }

    // h_mid snapshot for final combine
    scratch.h_mid[..hd].copy_from_slice(hidden);

    // -- step 1b: post-attention norm --
    step_input_norm(hidden, u16o(lc.post_attn_norm_w), &mut scratch.h_post, hd, cfg.rms_norm_eps);

    // -- step 4: MoE routing (CPU) --
    scratch.gate_scores[..cfg.num_experts as usize].fill(0.0);
    let routing = step_moe_routing(
        cfg,
        u32p(lc.gate_w), u16p(lc.gate_s), u16p(lc.gate_b),
        u32o(lc.seg_w), u16o(lc.seg_s), u16o(lc.seg_b),
        &scratch.h_post,
        &mut scratch.gate_scores,
    );
    let actual_k = routing.expert_indices.len().min(MAX_K);

    // -- step 5-6: Expert compute --
    scratch.moe_out[..hd].fill(0.0);
    scratch.shared_out[..hd].fill(0.0);

    if gpu_available && actual_k > 0 {
        // ---- GPU EXPERT PATH ----
        let ctx = metal_ctx.unwrap();
        let esz = active_expert_size(cfg, use_2bit);

        // Load expert data + GPU compute + readback
        if let Some(fd) = packed_fd {
            let mut cpu_buf = vec![0u8; esz];
            for k in 0..actual_k {
                let expert_idx = routing.expert_indices[k];
                let cached = expert_cache.as_mut()
                    .and_then(|c| c.lookup(layer_idx as i32, expert_idx, cfg.num_experts));
                if let Some(cached_buf) = cached {
                    let src = cached_buf.contents() as *const u8;
                    let dst = ctx.buf_multi_expert_data[k].contents() as *mut u8;
                    std::ptr::copy_nonoverlapping(src, dst, esz);
                } else {
                    let offset = (expert_idx as usize * esz) as u64;
                    if fd.read_exact_at(&mut cpu_buf, offset).is_ok() {
                        let dst = ctx.buf_multi_expert_data[k].contents() as *mut u8;
                        std::ptr::copy_nonoverlapping(cpu_buf.as_ptr(), dst, esz);
                        if let Some(c) = expert_cache.as_mut() {
                            let cache_buf = c.insert(layer_idx as i32, expert_idx, cfg.num_experts);
                            let cache_dst = cache_buf.contents() as *mut u8;
                            std::ptr::copy_nonoverlapping(cpu_buf.as_ptr(), cache_dst, esz);
                        }
                    }
                }
            }
        }

        // Copy input hidden state to GPU multi-expert input buffer
        let input_ptr = ctx.buf_multi_expert_input.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(scratch.h_post.as_ptr(), input_ptr, hd);

        // Encode GPU expert forwards
        let cmd = ctx.queue.new_command_buffer();
        for k in 0..actual_k {
            gpu_ops::gpu_encode_expert_forward_slot(cfg, ctx, &cmd, k, use_2bit, None);
        }
        cmd.commit();
        cmd.wait_until_completed();

        // Read back expert outputs and accumulate
        for k in 0..actual_k {
            let out_ptr = ctx.buf_multi_expert_out[k].contents() as *const f32;
            let expert_out = std::slice::from_raw_parts(out_ptr, hd);
            cpu_vec_madd(&mut scratch.moe_out, expert_out, routing.expert_weights[k], hd);
        }

        // Shared expert on CPU (lightweight)
        scratch.shared_out[..hd].fill(0.0);
        step_shared_expert(
            cfg,
            u32p(lc.sg_w), u16p(lc.sg_s), u16p(lc.sg_b),
            u32p(lc.su_w), u16p(lc.su_s), u16p(lc.su_b),
            u32p(lc.sd_w), u16p(lc.sd_s), u16p(lc.sd_b),
            &scratch.h_post,
            routing.shared_gate_score,
            &mut scratch.shared_out,
        );
    } else {
        // ---- CPU EXPERT PATH ----
        let esz = active_expert_size(cfg, use_2bit);
        if let Some(fd) = packed_fd {
            let mut buf = vec![0u8; esz];
            for k in 0..actual_k {
                let offset = (routing.expert_indices[k] as usize * esz) as u64;
                if fd.read_exact_at(&mut buf, offset).is_ok() {
                    step_cpu_expert(cfg, &buf, &scratch.h_post, &mut scratch.expert_tmp, use_2bit);
                    cpu_vec_madd(&mut scratch.moe_out, &scratch.expert_tmp, routing.expert_weights[k], hd);
                }
            }
        }

        // Shared expert (CPU)
        step_shared_expert(
            cfg,
            u32p(lc.sg_w), u16p(lc.sg_s), u16p(lc.sg_b),
            u32p(lc.su_w), u16p(lc.su_s), u16p(lc.su_b),
            u32p(lc.sd_w), u16p(lc.sd_s), u16p(lc.sd_b),
            &scratch.h_post,
            routing.shared_gate_score,
            &mut scratch.shared_out,
        );
    }

    // -- step 7: final combine --
    step_final_combine(&scratch.h_mid, &scratch.moe_out, &scratch.shared_out, hidden, hd);
}

use std::os::fd::RawFd;
use std::ffi::c_void;

use metal::{Buffer, CommandBuffer, MTLSize};

use crate::constants::{GROUP_SIZE, RMS_NORM_EPS};
use crate::error::MoEError;
use crate::metal_kernels;
use crate::metal_context::{metal_buf_shared, ExpertBuffer, WeightBuffer, MetalContext, MAX_K};
use crate::model::config::ModelConfig;
use crate::model::weights::WeightFile;

use crate::math_full_attention::FullAttnCmd2State;
use crate::math::{bf16_to_f32, dequant_matvec_4bit, softmax, topk, normalize_weights, rms_norm, sigmoid};
use crate::math_linear_attention::LinearAttnFusedWoodsState;

// ─── Deferred expert results (CMD3 async dispatch) ───────────────────────

pub struct DeferredExperts {
    pub(crate) cmd_buf: Option<CommandBuffer>,
    pub(crate) out_buf: Option<Buffer>,
    pub(crate) _keep_alive: Vec<Buffer>,
    pub gpu_combined: bool,
}

impl DeferredExperts {
    pub fn complete(&mut self, hidden: &mut [f32], hidden_dim: usize) {
        if let Some(ref cmd_buf) = self.cmd_buf {
            cmd_buf.wait_until_completed();
        }
        if let Some(ref out_buf) = self.out_buf {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_buf.contents() as *const f32,
                    hidden.as_mut_ptr(),
                    hidden_dim,
                );
            }
        }
        self.cmd_buf = None;
        self.out_buf = None;
        self._keep_alive.clear();
    }

    pub fn complete_fast(&mut self, hidden: &mut [f32], hidden_dim: usize) {
        // Wait on CMD3's own command buffer for CPU cache coherence.
        // Even though a later CMD1 on the same serial queue has completed
        // (guaranteeing CMD3 finished first), Metal requires waiting on the
        // specific command buffer that wrote the data for CPU visibility.
        if let Some(ref cmd_buf) = self.cmd_buf {
            cmd_buf.wait_until_completed();
        }
        if let Some(ref out_buf) = self.out_buf {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_buf.contents() as *const f32,
                    hidden.as_mut_ptr(),
                    hidden_dim,
                );
            }
        }
        self.cmd_buf = None;
        self.out_buf = None;
        self._keep_alive.clear();
    }
}

// ─── MoE layer forward ─────────────────────────────────────────────────────

/// Run the full MoE block for a single layer: routing, shared expert, K routed experts, combine.
///
/// When `attn_state` is `Some`, fuses batched attention + o_proj + residual + norm + gate
/// into a single CMD2 (matching the C engine's 3-CMD architecture).
/// When `attn_state` is `None` and `lin_attn` is `Some`, fuses out_proj + residual + norm + gate
/// into a single CMD2 for linear attention layers (FusedWoods mode, matching C exactly).
///
/// Returns `Some(DeferredExperts)` when GPU expert dispatch is used (async CMD3).
pub fn moe_layer_forward(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    packed_fd: RawFd,
    ctx: Option<&MetalContext>,
    gpu_wf: Option<&WeightBuffer>,
    config: &ModelConfig,
    attn_state: Option<FullAttnCmd2State>,
    lin_attn: Option<LinearAttnFusedWoodsState>,
    mut expert_gpu_buffer: Option<&mut ExpertBuffer>,
    gpu_combined: bool,
) -> Result<Option<DeferredExperts>, MoEError> {
    let hidden_dim = config.hidden_dim;
    let num_experts = config.num_experts;
    let moe_inter = config.moe_intermediate;
    let shared_inter = config.shared_intermediate;
    let expert_size = config.expert_size_4bit;
    let layout = &config.expert_layout_4bit;
    let k = config.num_experts_per_tok;

    let use_gpu = ctx.is_some() && gpu_wf.is_some();

    // Save h_mid (residual) — already completed by caller before this layer
    let h_mid = hidden.to_vec();

    let post_norm_name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
    let mut h_post = vec![0.0f32; hidden_dim];

    // ── Router gate + shared expert projections ──
    let mut gate_scores = vec![0.0f32; num_experts];
    let mut shared_gate = vec![0.0f32; shared_inter];
    let mut shared_up = vec![0.0f32; shared_inter];
    let mut shared_gate_score = 0.0f32;

    let prefix = format!("model.layers.{}.mlp", layer_idx);

    // GPU buffers (preserved for expert dispatch combine)
    let mut sg_buf_gpu: Option<Buffer> = None;
    let mut su_buf_gpu: Option<Buffer> = None;
    // When set, CMD3 uses this instead of uploading h_mid from CPU
    let mut hmid_gpu_override: Option<Buffer> = None;

    // ── CMD2 fusion path: batched attn + o_proj + residual + norm + gate ──
    let use_cmd2_fusion = attn_state.is_some()
        && use_gpu
        && ctx.is_some()
        && ctx.unwrap().attn_scores_batched.is_some()
        && ctx.unwrap().attn_softmax_batched.is_some()
        && ctx.unwrap().attn_values_batched.is_some()
        && ctx.unwrap().sigmoid_gate.is_some()
        && ctx.unwrap().residual_add.is_some()
        && ctx.unwrap().rms_norm_apply_bf16.is_some();

    if use_cmd2_fusion {
        let c = ctx.unwrap();
        let gw = gpu_wf.unwrap();
        let attn = attn_state.unwrap();

        // Allocate intermediate buffers
        let o_proj_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let temp_buf = metal_buf_shared(&c.device, hidden_dim * 4);  // residual_add output
        let sum_sq_buf = metal_buf_shared(&c.device, 4);
        let normed_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let gate_buf = metal_buf_shared(&c.device, num_experts * 4);
        let sg_buf = metal_buf_shared(&c.device, shared_inter * 4);
        let su_buf = metal_buf_shared(&c.device, shared_inter * 4);
        let sge_buf = metal_buf_shared(&c.device, 4);

        let cmd_buf = c.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        // Step 1: attn_scores_batched
        {
            let pipe = c.attn_scores_batched.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&attn.q_buf), 0);
            enc.set_buffer(1, Some(&attn.kc_buf), 0);
            enc.set_buffer(2, Some(&attn.scores_buf), 0);
                enc.set_bytes(3, 4, &attn.head_dim as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &attn.kv_dim as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &attn.seq_len as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &attn.seq_stride as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &attn.scale as *const f32 as *const c_void);
                enc.set_bytes(8, 4, &attn.heads_per_kv as *const u32 as *const c_void);
                enc.set_bytes(9, 4, &attn.seq_len as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new((attn.num_attn_heads * attn.seq_len) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 2: attn_softmax_batched
        {
            let pipe = c.attn_softmax_batched.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&attn.scores_buf), 0);
                enc.set_bytes(1, 4, &attn.seq_len as *const u32 as *const c_void);
                enc.set_bytes(2, 4, &attn.seq_stride as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(attn.num_attn_heads as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 3: attn_values_batched
        {
            let pipe = c.attn_values_batched.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&attn.scores_buf), 0);
            enc.set_buffer(1, Some(&attn.vc_buf), 0);
            enc.set_buffer(2, Some(&attn.out_buf), 0);
                enc.set_bytes(3, 4, &attn.head_dim as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &attn.kv_dim as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &attn.seq_len as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &attn.seq_stride as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &attn.heads_per_kv as *const u32 as *const c_void);
            let total_threads = attn.num_attn_heads * attn.head_dim;
            enc.dispatch_thread_groups(
                MTLSize::new(((total_threads + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 4: sigmoid_gate (attn_out *= sigmoid(q_gate))
        {
            let pipe = c.sigmoid_gate.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&attn.out_buf), 0);
            enc.set_buffer(1, Some(&attn.q_gate_buf), 0);
            enc.set_bytes(2, 4, &attn.q_dim as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((attn.q_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 5: o_proj matvec (gated attention → hidden_dim)
        gw.encode_matvec_into(wf, c, &enc, &attn.o_prefix, &attn.out_buf, 0, &o_proj_buf, 0, hidden_dim, attn.q_dim as usize);

        // Step 6: residual_add (o_proj_out + h_mid → temp_buf)
        {
            let pipe = c.residual_add.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&o_proj_buf), 0);
            enc.set_buffer(1, Some(&attn.hidden_buf), 0);
            enc.set_buffer(2, Some(&temp_buf), 0);
            enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 7: post_attention_layernorm (rms_norm_sum + rms_norm_apply_bf16)
        {
            enc.set_compute_pipeline_state(&c.rms_norm_sum);
            enc.set_buffer(0, Some(&temp_buf), 0);
            enc.set_buffer(1, Some(&sum_sq_buf), 0);
            enc.set_bytes(2, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        }
        {
            let pipe = c.rms_norm_apply_bf16.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&temp_buf), 0);
            let pnw_ptr = wf.get_tensor_ptr(&post_norm_name).unwrap();
            let pnw_off = (pnw_ptr as usize - gw.base as usize) as u64;
            enc.set_buffer(1, Some(&gw.buf), pnw_off);
            enc.set_buffer(2, Some(&sum_sq_buf), 0);
            enc.set_buffer(3, Some(&normed_buf), 0);
                enc.set_bytes(4, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 8-11: gate, shared_gate, shared_up, shared_expert_gate matvecs (on normed hidden)
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.gate", prefix), &normed_buf, 0, &gate_buf, 0, num_experts, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.gate_proj", prefix), &normed_buf, 0, &sg_buf, 0, shared_inter, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.up_proj", prefix), &normed_buf, 0, &su_buf, 0, shared_inter, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert_gate", prefix), &normed_buf, 0, &sge_buf, 0, 1, hidden_dim);

        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read back results
        unsafe {
            std::ptr::copy_nonoverlapping(gate_buf.contents() as *const f32, gate_scores.as_mut_ptr(), num_experts);
            std::ptr::copy_nonoverlapping(sg_buf.contents() as *const f32, shared_gate.as_mut_ptr(), shared_inter);
            std::ptr::copy_nonoverlapping(su_buf.contents() as *const f32, shared_up.as_mut_ptr(), shared_inter);
            shared_gate_score = *(sge_buf.contents() as *const f32);
            // Update hidden to post-normed value (h_post for expert input)
            std::ptr::copy_nonoverlapping(normed_buf.contents() as *const f32, hidden.as_mut_ptr(), hidden_dim);
        }

        // Keep shared gate/up on GPU for CMD3 SwiGLU
        sg_buf_gpu = Some(sg_buf);
        su_buf_gpu = Some(su_buf);

        // h_post already set from normed_buf readback above (stored in hidden[])
        h_post.copy_from_slice(hidden);
        // temp_buf = h_mid + attn_out — use as hmid_gpu in CMD3 combine
        hmid_gpu_override = Some(temp_buf);
    } else if lin_attn.is_some() && use_gpu
        && ctx.is_some()
        && ctx.unwrap().residual_add.is_some()
        && ctx.unwrap().rms_norm_apply_bf16.is_some()
    {
        // ── FusedWoods linear CMD2: out_proj + residual_add + rms_norm + gate + shared ──
        // Matches C engine exactly: CMD1(SSM) → CPU(gated_norm) → CMD2(this block) → CMD3
        let c = ctx.unwrap();
        let gw = gpu_wf.unwrap();
        let la = lin_attn.unwrap();

        // Gated buffer already on GPU from CMD1 L5 — no upload needed (matches C: batch_out[6])
        let gated_buf = &la.gated_buf;
        let hmid_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(la.h_mid.as_ptr(), hmid_buf.contents() as *mut f32, hidden_dim);
        }

        let o_proj_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let temp_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let sum_sq_buf = metal_buf_shared(&c.device, 4);
        let normed_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let gate_buf = metal_buf_shared(&c.device, num_experts * 4);
        let sg_buf = metal_buf_shared(&c.device, shared_inter * 4);
        let su_buf = metal_buf_shared(&c.device, shared_inter * 4);
        let sge_buf = metal_buf_shared(&c.device, 4);

        let cmd_buf = c.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        // Step 1: out_proj (gated_out → hidden_dim) — reads directly from GPU buffer
        gw.encode_matvec_into(wf, c, &enc, &la.o_prefix, gated_buf, 0, &o_proj_buf, 0, hidden_dim, la.total_value);

        // Step 2: residual_add (o_proj_out + h_mid → temp_buf)
        {
            let pipe = c.residual_add.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&o_proj_buf), 0);
            enc.set_buffer(1, Some(&hmid_buf), 0);
            enc.set_buffer(2, Some(&temp_buf), 0);
            enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 3: post_attention_layernorm (rms_norm_sum + rms_norm_apply_bf16)
        {
            enc.set_compute_pipeline_state(&c.rms_norm_sum);
            enc.set_buffer(0, Some(&temp_buf), 0);
            enc.set_buffer(1, Some(&sum_sq_buf), 0);
            enc.set_bytes(2, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        }
        {
            let pipe = c.rms_norm_apply_bf16.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&temp_buf), 0);
            let pnw_ptr = wf.get_tensor_ptr(&la.post_norm_name).unwrap();
            let pnw_off = (pnw_ptr as usize - gw.base as usize) as u64;
            enc.set_buffer(1, Some(&gw.buf), pnw_off);
            enc.set_buffer(2, Some(&sum_sq_buf), 0);
            enc.set_buffer(3, Some(&normed_buf), 0);
                enc.set_bytes(4, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 4: gate, shared_gate, shared_up, shared_expert_gate matvecs (on normed hidden)
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.gate", prefix), &normed_buf, 0, &gate_buf, 0, num_experts, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.gate_proj", prefix), &normed_buf, 0, &sg_buf, 0, shared_inter, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.up_proj", prefix), &normed_buf, 0, &su_buf, 0, shared_inter, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert_gate", prefix), &normed_buf, 0, &sge_buf, 0, 1, hidden_dim);

        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read back results
        unsafe {
            std::ptr::copy_nonoverlapping(gate_buf.contents() as *const f32, gate_scores.as_mut_ptr(), num_experts);
            std::ptr::copy_nonoverlapping(sg_buf.contents() as *const f32, shared_gate.as_mut_ptr(), shared_inter);
            std::ptr::copy_nonoverlapping(su_buf.contents() as *const f32, shared_up.as_mut_ptr(), shared_inter);
            shared_gate_score = *(sge_buf.contents() as *const f32);
            // Update hidden to post-normed value
            std::ptr::copy_nonoverlapping(normed_buf.contents() as *const f32, hidden.as_mut_ptr(), hidden_dim);
        }

        sg_buf_gpu = Some(sg_buf);
        su_buf_gpu = Some(su_buf);
        h_post.copy_from_slice(hidden);
        // temp_buf = h_mid + out_proj(attn_out) — use as hmid_gpu in CMD3 combine
        hmid_gpu_override = Some(temp_buf);
    } else {
        // ── Non-fused path: post-norm on CPU, router CMD separately ──
        let pnw = wf.get_tensor_u16(&post_norm_name);
        if let Some(pnw) = pnw {
            let pnw_f32: Vec<f32> = pnw.iter().map(|&v| bf16_to_f32(v)).collect();
            rms_norm(hidden, &pnw_f32, &mut h_post, hidden_dim, RMS_NORM_EPS);
        } else {
            h_post.copy_from_slice(hidden);
        }

        // Router gate + shared expert projections: all independent (same input) → batch
        if use_gpu {
            let gw = gpu_wf.unwrap();
            let c = ctx.unwrap();
            let x_buf = metal_buf_shared(&c.device, hidden_dim * 4);
            unsafe { let dst = x_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(h_post.as_ptr(), dst, hidden_dim); }
            let gate_buf = metal_buf_shared(&c.device, num_experts * 4);
            let sg_buf = metal_buf_shared(&c.device, shared_inter * 4);
            let su_buf = metal_buf_shared(&c.device, shared_inter * 4);
            let sge_buf = metal_buf_shared(&c.device, 4);

            let cmd_buf = c.queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.gate", prefix), &x_buf, 0, &gate_buf, 0, num_experts, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.gate_proj", prefix), &x_buf, 0, &sg_buf, 0, shared_inter, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.up_proj", prefix), &x_buf, 0, &su_buf, 0, shared_inter, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert_gate", prefix), &x_buf, 0, &sge_buf, 0, 1, hidden_dim);
            enc.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            unsafe {
                std::ptr::copy_nonoverlapping(gate_buf.contents() as *const f32, gate_scores.as_mut_ptr(), num_experts);
                std::ptr::copy_nonoverlapping(sg_buf.contents() as *const f32, shared_gate.as_mut_ptr(), shared_inter);
                std::ptr::copy_nonoverlapping(su_buf.contents() as *const f32, shared_up.as_mut_ptr(), shared_inter);
                let tmp = sge_buf.contents() as *const f32;
                shared_gate_score = *tmp;
            }
            sg_buf_gpu = Some(sg_buf);
            su_buf_gpu = Some(su_buf);
        } else {
            // CPU fallback
            if let (Some(gw_p), Some(gs), Some(gb)) = (
                wf.get_tensor_u32(&format!("{}.gate.weight", prefix)),
                wf.get_tensor_u16(&format!("{}.gate.scales", prefix)),
                wf.get_tensor_u16(&format!("{}.gate.biases", prefix)),
            ) { dequant_matvec_4bit(gw_p, gs, gb, &h_post, &mut gate_scores, num_experts, hidden_dim, GROUP_SIZE); }
            if let (Some(sgw), Some(sgs), Some(sgb)) = (
                wf.get_tensor_u32(&format!("{}.shared_expert.gate_proj.weight", prefix)),
                wf.get_tensor_u16(&format!("{}.shared_expert.gate_proj.scales", prefix)),
                wf.get_tensor_u16(&format!("{}.shared_expert.gate_proj.biases", prefix)),
            ) { dequant_matvec_4bit(sgw, sgs, sgb, &h_post, &mut shared_gate, shared_inter, hidden_dim, GROUP_SIZE); }
            if let (Some(suw), Some(sus), Some(sub)) = (
                wf.get_tensor_u32(&format!("{}.shared_expert.up_proj.weight", prefix)),
                wf.get_tensor_u16(&format!("{}.shared_expert.up_proj.scales", prefix)),
                wf.get_tensor_u16(&format!("{}.shared_expert.up_proj.biases", prefix)),
            ) { dequant_matvec_4bit(suw, sus, sub, &h_post, &mut shared_up, shared_inter, hidden_dim, GROUP_SIZE); }
            if let (Some(segw), Some(segs), Some(segb)) = (
                wf.get_tensor_u32(&format!("{}.shared_expert_gate.weight", prefix)),
                wf.get_tensor_u16(&format!("{}.shared_expert_gate.scales", prefix)),
                wf.get_tensor_u16(&format!("{}.shared_expert_gate.biases", prefix)),
            ) {
                let mut tmp = [0.0f32];
                dequant_matvec_4bit(segw, segs, segb, &h_post, &mut tmp, 1, hidden_dim, GROUP_SIZE);
                shared_gate_score = tmp[0];
            }
        }
    }

    // ── Routing: softmax + topk ──
    softmax(&mut gate_scores);

    let mut expert_indices = vec![0usize; k];
    let mut expert_weights = vec![0.0f32; k];
    topk(&gate_scores, k, &mut expert_indices, &mut expert_weights);
    normalize_weights(&mut expert_weights);

    // ── Routed expert computation ──
    let mut moe_out = vec![0.0f32; hidden_dim];

    if use_gpu {
        let ctx = ctx.unwrap();
        let gw = gpu_wf.unwrap();
        let k = expert_indices.len();
        let actual_k = k.min(MAX_K);

        let hidden_u32 = hidden_dim as u32;
        let inter_u32 = moe_inter as u32;
        let gs_u32 = GROUP_SIZE as u32;

        // ── Phase 1: Parallel pread (cache hits skip I/O) ──
        let mut valid = [false; MAX_K];
        let mut fallback_expert_bufs: Vec<Buffer> = Vec::new();

        // Mutable phase: check cache, insert misses, parallel pread
        if let Some(ref mut io) = expert_gpu_buffer {
            let mut miss_ei = [0usize; MAX_K];
            let mut miss_k_slot = [0usize; MAX_K];
            let mut miss_count = 0;

            for ki in 0..actual_k {
                let eidx = expert_indices[ki];
                if let Some(buf) = io.cache.lookup(layer_idx, eidx) {
                    io.expert_data[ki] = buf;
                    valid[ki] = true;
                } else {
                    miss_ei[miss_count] = eidx;
                    miss_k_slot[miss_count] = ki;
                    miss_count += 1;
                }
            }

            // Insert cache entries for misses BEFORE rayon::scope
            for m in 0..miss_count {
                let ki = miss_k_slot[m];
                let eidx = miss_ei[m];
                let buf = io.cache.insert_get_buf(layer_idx, eidx);
                io.expert_data[ki] = buf;
            }

            // Parallel pread cache misses — raw pointers only, no &io in scope
            // Use usize to transmit pointers across threads (raw pointers aren't Send)
            if miss_count > 0 {
                let mut pread_tasks: Vec<(RawFd, usize, usize, i64)> = Vec::with_capacity(miss_count);
                for m in 0..miss_count {
                    let ki = miss_k_slot[m];
                    let eidx = miss_ei[m];
                    let ptr = io.expert_data[ki].contents() as usize;
                    pread_tasks.push((packed_fd, ptr, expert_size, (eidx as i64) * (expert_size as i64)));
                }
                rayon::scope(|s| {
                    for (fd, dst, sz, off) in pread_tasks {
                        s.spawn(move |_| {
                            unsafe { libc::pread(fd, dst as *mut std::ffi::c_void, sz, off); }
                        });
                    }
                });
            }
            for m in 0..miss_count {
                valid[miss_k_slot[m]] = true;
            }
        } else {
            // Fallback: sequential pread into ad-hoc buffers
            for ki in 0..actual_k {
                let eidx = expert_indices[ki];
                let buf = metal_buf_shared(&ctx.device, expert_size);
                let nread = unsafe {
                    let ptr = buf.contents() as *mut u8;
                    let slice = std::slice::from_raw_parts_mut(ptr, expert_size);
                    libc::pread(packed_fd, slice.as_mut_ptr() as *mut std::ffi::c_void, expert_size, (eidx as i64) * (expert_size as i64))
                };
                if nread == expert_size as isize {
                    valid[ki] = true;
                }
                fallback_expert_bufs.push(buf);
            }
        }
        // Mutable borrow of expert_gpu_buffer ends here

        let any_valid = valid.iter().take(actual_k).any(|&v| v);

        if any_valid {
            // ── Phase 2: GPU dispatch using pre-allocated or ad-hoc buffers ──
            let io_ref = expert_gpu_buffer.as_deref();  // Option<&ExpertBuffer>

            let (x_buf, gate_out, up_out, act_out, out_bufs,
                 shared_act_gpu, shared_down_gpu, _hidden_out, params_buf)
                = if let Some(io) = io_ref {
                // Pre-allocated path: reuse persistent Metal buffers
                unsafe { let dst = io.input_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(h_post.as_ptr(), dst, hidden_dim); }
                let ob: Vec<Buffer> = io.expert_out.iter().take(actual_k).cloned().collect();
                (io.input_buf.clone(), io.scratch_gate.clone(), io.scratch_up.clone(),
                 io.scratch_act.clone(), ob,
                 io.shared_act.clone(), io.shared_down.clone(),
                 io.combine_out.clone(), io.combine_params.clone())
            } else {
                // Legacy path: allocate per-layer
                let x = metal_buf_shared(&ctx.device, hidden_dim * 4);
                unsafe { let dst = x.contents() as *mut f32; std::ptr::copy_nonoverlapping(h_post.as_ptr(), dst, hidden_dim); }
                (x,
                 metal_buf_shared(&ctx.device, moe_inter * 4),
                 metal_buf_shared(&ctx.device, moe_inter * 4),
                 metal_buf_shared(&ctx.device, moe_inter * 4),
                 (0..actual_k).map(|_| metal_buf_shared(&ctx.device, hidden_dim * 4)).collect(),
                 metal_buf_shared(&ctx.device, shared_inter * 4),
                 metal_buf_shared(&ctx.device, hidden_dim * 4),
                 metal_buf_shared(&ctx.device, hidden_dim * 4),
                 metal_buf_shared(&ctx.device, 40))
            };

            // h_mid on GPU for moe_combine_residual
            let hmid_gpu = if let Some(buf) = hmid_gpu_override.take() {
                buf
            } else {
                let buf = metal_buf_shared(&ctx.device, hidden_dim * 4);
                unsafe { let dst = buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(h_mid.as_ptr(), dst, hidden_dim); }
                buf
            };

            // ── FUSED CMD: K experts + shared SwiGLU + shared down_proj + moe_combine_residual ──
            let cmd_buf = ctx.queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();

            for ki in 0..actual_k {
                if !valid[ki] { continue; }
                // Expert data: from pre-allocated io.expert_data[ki] or fallback
                let expert_buf: &Buffer = if let Some(io) = io_ref {
                    &io.expert_data[ki]
                } else if ki < fallback_expert_bufs.len() {
                    &fallback_expert_bufs[ki]
                } else {
                    continue;
                };
                metal_kernels::encode_matvec_offset(ctx, &enc,
                    expert_buf, layout.gate_w_off as u64,
                    expert_buf, layout.gate_s_off as u64,
                    expert_buf, layout.gate_b_off as u64,
                    &x_buf, 0, &gate_out, 0,
                    inter_u32, hidden_u32, gs_u32, 3);

                metal_kernels::encode_matvec_offset(ctx, &enc,
                    expert_buf, layout.up_w_off as u64,
                    expert_buf, layout.up_s_off as u64,
                    expert_buf, layout.up_b_off as u64,
                    &x_buf, 0, &up_out, 0,
                    inter_u32, hidden_u32, gs_u32, 3);

                metal_kernels::encode_swiglu(ctx, &enc, &gate_out, 0, &up_out, 0, &act_out, 0, inter_u32);

                metal_kernels::encode_matvec_offset(ctx, &enc,
                    expert_buf, layout.down_w_off as u64,
                    expert_buf, layout.down_s_off as u64,
                    expert_buf, layout.down_b_off as u64,
                    &act_out, 0, &out_bufs[ki], 0,
                    hidden_u32, inter_u32, gs_u32, 3);
            }

            // Shared expert SwiGLU on GPU
            if let (Some(ref sg), Some(ref su)) = (sg_buf_gpu.as_ref(), su_buf_gpu.as_ref()) {
                metal_kernels::encode_swiglu(ctx, &enc, sg, 0, su, 0, &shared_act_gpu, 0, shared_inter as u32);
            }

            gw.encode_matvec_into(wf, ctx, &enc,
                &format!("{}.shared_expert.down_proj", prefix),
                &shared_act_gpu, 0, &shared_down_gpu, 0, hidden_dim, shared_inter);

            // ── moe_combine_residual (writes to persistent buf_moe_hidden for GPU input_norm) ──
            {
                let mcr_pipe = ctx.moe_combine_residual.as_ref().unwrap();
                enc.set_compute_pipeline_state(mcr_pipe);
                enc.set_buffer(0, Some(&hmid_gpu), 0);
                enc.set_buffer(1, Some(&shared_down_gpu), 0);
                enc.set_buffer(2, Some(ctx.buf_moe_hidden.as_ref().unwrap()), 0);
                for ei in 0..MAX_K {
                    if ei < actual_k && valid[ei] {
                        enc.set_buffer(3 + ei as u64, Some(&out_bufs[ei]), 0);
                    } else {
                        enc.set_buffer(3 + ei as u64, Some(ctx.buf_moe_hidden.as_ref().unwrap()), 0);
                    }
                }
                let mut params = [0.0f32; 10];
                for (i, &w) in expert_weights.iter().enumerate() { params[i] = w; }
                params[8] = shared_gate_score;
                unsafe { std::ptr::copy_nonoverlapping(params.as_ptr(), params_buf.contents() as *mut f32, 10); }
                enc.set_buffer(11, Some(&params_buf), 0);
                    enc.set_bytes(12, 4, &hidden_u32 as *const u32 as *const c_void);
                    let ku = actual_k as u32;
                    enc.set_bytes(13, 4, &ku as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }

            // GPU-side input_norm for next layer (matches C Enc C2 + Enc C3)
            // Only safe in fused modes where CMD1 reads directly from buf_input,
            // guaranteeing GPU queue serialization of CMD3-then-CMD1.
            let do_gpu_norm = gpu_combined
                && layer_idx + 1 < config.num_layers
                && ctx.rms_norm_apply_bf16.is_some()
                && wf.get_tensor_ptr(&format!("model.layers.{}.input_layernorm.weight", layer_idx + 1)).is_some();
            if do_gpu_norm {
                let next_norm_ptr = wf.get_tensor_ptr(
                    &format!("model.layers.{}.input_layernorm.weight", layer_idx + 1)).unwrap();
                let next_norm_off = (next_norm_ptr as usize - gw.base as usize) as u64;
                let buf_moe = ctx.buf_moe_hidden.as_ref().unwrap();

                // Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
                {
                    enc.set_compute_pipeline_state(&ctx.rms_norm_sum);
                    enc.set_buffer(0, Some(buf_moe), 0);
                    enc.set_buffer(1, Some(ctx.buf_cmd3_sum_sq.as_ref().unwrap()), 0);
                    enc.set_bytes(2, 4, &hidden_u32 as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
                }

                // Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_norm_w -> buf_input)
                {
                    let pipe = ctx.rms_norm_apply_bf16.as_ref().unwrap();
                    enc.set_compute_pipeline_state(pipe);
                    enc.set_buffer(0, Some(buf_moe), 0);
                    enc.set_buffer(1, Some(&gw.buf), next_norm_off);
                    enc.set_buffer(2, Some(ctx.buf_cmd3_sum_sq.as_ref().unwrap()), 0);
                    enc.set_buffer(3, Some(ctx.buf_input.as_ref().unwrap()), 0);
                        enc.set_bytes(4, 4, &hidden_u32 as *const u32 as *const c_void);
                        let eps = RMS_NORM_EPS;
                        enc.set_bytes(5, 4, &eps as *const f32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                }
            }

            enc.end_encoding();
            cmd_buf.commit();

            let mut keep_alive = Vec::with_capacity(4);
            keep_alive.push(hmid_gpu);
            if io_ref.is_none() {
                keep_alive.push(shared_act_gpu);
                keep_alive.push(shared_down_gpu);
                keep_alive.push(params_buf);
                keep_alive.push(x_buf);
                keep_alive.push(gate_out);
                keep_alive.push(up_out);
                keep_alive.push(act_out);
                keep_alive.extend(out_bufs);
                keep_alive.extend(fallback_expert_bufs);
            }
            // hidden_out no longer in keep_alive — buf_moe_hidden is persistent (MetalContext)
            if let Some(b) = sg_buf_gpu.take() { keep_alive.push(b); }
            if let Some(b) = su_buf_gpu.take() { keep_alive.push(b); }

            return Ok(Some(DeferredExperts {
                cmd_buf: Some(cmd_buf.to_owned()),
                out_buf: ctx.buf_moe_hidden.clone(),
                _keep_alive: keep_alive,
                gpu_combined,
            }));
        }
        // No experts loaded — fall through to CPU below
    }

    let gpu_done = !moe_out.iter().all(|&v| v == 0.0);
    if !gpu_done {
        // ── CPU fallback: compute everything synchronously ──
        let mut expert_data = vec![0u8; expert_size];
        let mut gate_tmp = vec![0.0f32; moe_inter];
        let mut up_tmp = vec![0.0f32; moe_inter];
        let mut act_tmp = vec![0.0f32; moe_inter];
        let mut eout = vec![0.0f32; hidden_dim];

        for (&eidx, &ew) in expert_indices.iter().zip(expert_weights.iter()) {
            let expert_offset = (eidx as i64) * (expert_size as i64);
            let nread = unsafe {
                libc::pread(
                    packed_fd,
                    expert_data.as_mut_ptr() as *mut std::ffi::c_void,
                    expert_size,
                    expert_offset,
                )
            };
            if nread != expert_size as isize {
                continue;
            }

            // gate_proj
            let gw = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.gate_w_off) as *const u32, layout.gate_w_size / 4) };
            let gs = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.gate_s_off) as *const u16, layout.gate_s_size / 2) };
            let gb = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.gate_b_off) as *const u16, layout.gate_b_size / 2) };
            dequant_matvec_4bit(gw, gs, gb, &h_post, &mut gate_tmp, moe_inter, hidden_dim, GROUP_SIZE);

            // up_proj
            let uw = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.up_w_off) as *const u32, layout.up_w_size / 4) };
            let us = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.up_s_off) as *const u16, layout.up_s_size / 2) };
            let ub = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.up_b_off) as *const u16, layout.up_b_size / 2) };
            dequant_matvec_4bit(uw, us, ub, &h_post, &mut up_tmp, moe_inter, hidden_dim, GROUP_SIZE);

            // SwiGLU
            for i in 0..moe_inter {
                let g = gate_tmp[i];
                let silu_g = g / (1.0 + (-g).exp());
                act_tmp[i] = silu_g * up_tmp[i];
            }

            // down_proj
            let dw = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.down_w_off) as *const u32, layout.down_w_size / 4) };
            let ds = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.down_s_off) as *const u16, layout.down_s_size / 2) };
            let db = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.down_b_off) as *const u16, layout.down_b_size / 2) };
            dequant_matvec_4bit(dw, ds, db, &act_tmp, &mut eout, hidden_dim, moe_inter, GROUP_SIZE);

            for d in 0..hidden_dim {
                moe_out[d] += eout[d] * ew;
            }
        }
    }

    // ── Shared expert SwiGLU + down_proj ──
    let mut shared_out = vec![0.0f32; hidden_dim];
    let mut shared_act = vec![0.0f32; shared_inter];

    // SwiGLU on shared gate/up
    for i in 0..shared_inter {
        let g = shared_gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        shared_act[i] = silu_g * shared_up[i];
    }

    // Shared expert down_proj
    if use_gpu {
        let gw = gpu_wf.unwrap();
        let c = ctx.unwrap();
        let sa_buf = metal_buf_shared(&c.device, shared_inter * 4);
        unsafe { let dst = sa_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(shared_act.as_ptr(), dst, shared_inter); }
        let so_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let cmd_buf = c.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.down_proj", prefix), &sa_buf, 0, &so_buf, 0, hidden_dim, shared_inter);
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        unsafe { std::ptr::copy_nonoverlapping(so_buf.contents() as *const f32, shared_out.as_mut_ptr(), hidden_dim); }
    } else if let (Some(sdw), Some(sds), Some(sdb)) = (
        wf.get_tensor_u32(&format!("{}.shared_expert.down_proj.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.shared_expert.down_proj.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.shared_expert.down_proj.biases", prefix)),
    ) {
        dequant_matvec_4bit(sdw, sds, sdb, &shared_act, &mut shared_out, hidden_dim, shared_inter, GROUP_SIZE);
    }

    let shared_weight = sigmoid(shared_gate_score);

    // ── Final combine: hidden = h_mid + moe_out + shared_weight * shared_out ──
    for i in 0..hidden_dim {
        hidden[i] = h_mid[i] + moe_out[i] + shared_weight * shared_out[i];
    }

    Ok(None)
}

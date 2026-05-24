/// Qwen3.6-35B-A3B-4bit FusedExp engine — all model dimensions are compile-time constants.
use crate::constants::qwen35_35b_stripped::*;
use crate::constants::{MAX_SEQ, RMS_NORM_EPS, FULL_ATTN_INTERVAL, GROUP_SIZE, CONV_KERNEL_SIZE};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::time::Instant;

use metal::{Buffer, CommandBuffer, ComputeCommandEncoderRef, MTLSize};

use crate::metal_kernels;
use crate::metal_context::{metal_buf_shared, WeightBuffer, MetalContext, ExpertBuffer, MAX_K};
use crate::cache::{Cache, LinearAttnState};
use crate::engine::Engine;
use crate::model::Model;
use crate::model::expert::ExpertFile;
use crate::error::MoEError;
use crate::engine::{SignalCheckFn, TelemetryValue};
use crate::model::weights::WeightFile;
use crate::math::{
    apply_rope, bf16_to_f32, dequant_matvec_4bit,
    embed_lookup, final_norm, normalize_weights, rms_norm, sigmoid,
    softmax, topk,
};

// ─── Full attention GPU output (local copy) ───────────────────────────────

struct FullAttnGpuOut {
    q_buf: Buffer,
    q_gate_buf: Buffer,
    kc_buf: Buffer,
    vc_buf: Buffer,
    scores_buf: Buffer,
    out_buf: Buffer,
    hidden_buf: Buffer,
    seq_len: u32,
    seq_stride: u32,
    num_attn_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    heads_per_kv: u32,
    scale: f32,
    q_dim: u32,
    o_prefix: String,
}

// ─── Deferred expert results (local copy) ─────────────────────────────────

struct DeferredExperts {
    cmd_buf: Option<CommandBuffer>,
    out_buf: Option<Buffer>,
    _keep_alive: Vec<Buffer>,
}

impl DeferredExperts {
    fn complete(&mut self, hidden: &mut [f32], hidden_dim: usize) {
        if let Some(ref cmd_buf) = self.cmd_buf {
            cmd_buf.wait_until_completed();
        }
        if let Some(ref out_buf) = self.out_buf {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_buf.contents() as *const f32, hidden.as_mut_ptr(), hidden_dim);
            }
        }
        self.cmd_buf = None;
        self.out_buf = None;
        self._keep_alive.clear();
    }
}

// ─── Timing helpers ──────────────────────────────────────────────────────

fn timing_add(tm: &mut BTreeMap<String, TelemetryValue>, key: &str, dt: f64) {
    if !crate::engine::record_telemetry() { return; }
    match tm.entry(key.into()) {
        std::collections::btree_map::Entry::Occupied(mut e) => {
            if let TelemetryValue::Scalar(ref mut v) = e.get_mut() { *v += dt; }
        }
        std::collections::btree_map::Entry::Vacant(e) => { e.insert(TelemetryValue::Scalar(dt)); }
    }
}

fn timing_push(tm: &mut BTreeMap<String, TelemetryValue>, key: &str, dt: f64) {
    if !crate::engine::record_telemetry() { return; }
    match tm.entry(key.into()) {
        std::collections::btree_map::Entry::Occupied(mut e) => {
            if let TelemetryValue::List(ref mut v) = e.get_mut() { v.push(dt); }
        }
        std::collections::btree_map::Entry::Vacant(e) => { e.insert(TelemetryValue::List(vec![dt])); }
    }
}

// ─── Execution context ─────────────────────────────────────────────────────

struct ExecCtx<'a> {
    model: &'a Model,
    cache: &'a mut Cache,
    ctx: &'a MetalContext,
    gpu_wf: &'a WeightBuffer,
    expert_gpu_buffer: Option<&'a mut ExpertBuffer>,
    k: usize,
    timing: BTreeMap<String, TelemetryValue>,
}

impl<'a> ExecCtx<'a> {
    // ── Embedding ──────────────────────────────────────────────────────────

    fn embed(&self, token_id: usize, out: &mut [f32]) {
        embed_lookup(&self.model.wf, token_id, out, HIDDEN_DIM);
    }

    // ── Full attention layer (breaks pipeline) ─────────────────────────────

    fn full_attention_layer(
        &mut self, layer: usize, hidden: &mut [f32], pos: usize,
    ) -> Option<DeferredExperts> {
        let wf = &self.model.wf;
        let hd = HIDDEN_DIM;
        let num_q = NUM_ATTN_HEADS;
        let num_kv = NUM_KV_HEADS;
        let head_dim = HEAD_DIM;
        let q_dim = num_q * head_dim;
        let q_proj_dim = q_dim * 2;
        let kv_dim = num_kv * head_dim;

        // Input RMS norm
        let norm_name = format!("model.layers.{}.input_layernorm.weight", layer);
        let nw = wf.get_tensor_u16(&norm_name);
        let mut normed = vec![0.0f32; hd];
        if let Some(nw) = nw {
            let nw_f32: Vec<f32> = nw.iter().map(|&v| bf16_to_f32(v)).collect();
            rms_norm(hidden, &nw_f32, &mut normed, hd, RMS_NORM_EPS);
        } else {
            normed.copy_from_slice(hidden);
        }

        // QKV projections (GPU)
        let mut q_proj_out = vec![0.0f32; q_proj_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        {
            let gw = self.gpu_wf;
            let c = self.ctx;
            let x_buf = metal_buf_shared(&c.device, hd * 4);
            unsafe { let dst = x_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(normed.as_ptr(), dst, hd); }
            let qbuf = metal_buf_shared(&c.device, q_proj_dim * 4);
            let kbuf = metal_buf_shared(&c.device, kv_dim * 4);
            let vbuf = metal_buf_shared(&c.device, kv_dim * 4);
            let cm = c.queue.new_command_buffer();
            let enc = cm.new_compute_command_encoder();
            let q_name = format!("model.layers.{}.self_attn.q_proj", layer);
            let k_name = format!("model.layers.{}.self_attn.k_proj", layer);
            let v_name = format!("model.layers.{}.self_attn.v_proj", layer);
            gw.encode_matvec_into(wf, c, &enc, &q_name, &x_buf, 0, &qbuf, 0, q_proj_dim, hd);
            gw.encode_matvec_into(wf, c, &enc, &k_name, &x_buf, 0, &kbuf, 0, kv_dim, hd);
            gw.encode_matvec_into(wf, c, &enc, &v_name, &x_buf, 0, &vbuf, 0, kv_dim, hd);
            enc.end_encoding(); cm.commit(); cm.wait_until_completed();
            unsafe {
                std::ptr::copy_nonoverlapping(qbuf.contents() as *const f32, q_proj_out.as_mut_ptr(), q_proj_dim);
                std::ptr::copy_nonoverlapping(kbuf.contents() as *const f32, k.as_mut_ptr(), kv_dim);
                std::ptr::copy_nonoverlapping(vbuf.contents() as *const f32, v.as_mut_ptr(), kv_dim);
            }
        }

        // Split Q / Q-gate
        let mut q = vec![0.0f32; q_dim];
        let mut q_gate = vec![0.0f32; q_dim];
        for h in 0..num_q {
            let src = &q_proj_out[h * 2 * head_dim..];
            q[h * head_dim..h * head_dim + head_dim].copy_from_slice(&src[..head_dim]);
            q_gate[h * head_dim..h * head_dim + head_dim].copy_from_slice(&src[head_dim..2 * head_dim]);
        }

        // Q/K per-head norms
        let prefix = format!("model.layers.{}.self_attn", layer);
        let qn_name = format!("{}.q_norm.weight", prefix);
        let kn_name = format!("{}.k_norm.weight", prefix);
        if let Some(qnw) = wf.get_tensor_u16(&qn_name) {
            for h in 0..num_q {
                let qh = &mut q[h * head_dim..(h + 1) * head_dim];
                let ssq: f32 = qh.iter().map(|&x| x * x).sum();
                let inv = 1.0 / (ssq / head_dim as f32 + RMS_NORM_EPS).sqrt();
                for i in 0..head_dim.min(qnw.len()) { qh[i] *= inv * bf16_to_f32(qnw[i]); }
            }
        }
        if let Some(knw) = wf.get_tensor_u16(&kn_name) {
            for h in 0..num_kv {
                let kh = &mut k[h * head_dim..(h + 1) * head_dim];
                let ssq: f32 = kh.iter().map(|&x| x * x).sum();
                let inv = 1.0 / (ssq / head_dim as f32 + RMS_NORM_EPS).sqrt();
                for i in 0..head_dim.min(knw.len()) { kh[i] *= inv * bf16_to_f32(knw[i]); }
            }
        }

        // RoPE
        apply_rope(&mut q, &mut k, pos, num_q, num_kv, head_dim,
            ROTARY_DIM, ROPE_THETA);

        // Append K, V to cache
        let kv_cache = self.cache.kv[layer].as_mut().unwrap();
        let cache_pos = kv_cache.len;
        assert!(cache_pos < MAX_SEQ, "sequence length {} exceeds MAX_SEQ ({})", cache_pos, MAX_SEQ);
        kv_cache.k_cache[cache_pos * kv_dim..(cache_pos + 1) * kv_dim].copy_from_slice(&k);
        kv_cache.v_cache[cache_pos * kv_dim..(cache_pos + 1) * kv_dim].copy_from_slice(&v);
        kv_cache.len += 1;

        // Build FullAttnGpuOut for CMD2
        let seq_len = kv_cache.len;
        let c = self.ctx;
        let q_buf = metal_buf_shared(&c.device, q_dim * 4);
        let kc_buf = metal_buf_shared(&c.device, MAX_SEQ * kv_dim * 4);
        let vc_buf = metal_buf_shared(&c.device, MAX_SEQ * kv_dim * 4);
        let scores_buf = metal_buf_shared(&c.device, num_q * MAX_SEQ * 4);
        let out_buf = metal_buf_shared(&c.device, q_dim * 4);
        let q_gate_buf = metal_buf_shared(&c.device, q_dim * 4);
        let hidden_buf = metal_buf_shared(&c.device, hd * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(q.as_ptr(), q_buf.contents() as *mut f32, q_dim);
            std::ptr::copy_nonoverlapping(kv_cache.k_cache.as_ptr(), kc_buf.contents() as *mut f32, seq_len * kv_dim);
            std::ptr::copy_nonoverlapping(kv_cache.v_cache.as_ptr(), vc_buf.contents() as *mut f32, seq_len * kv_dim);
            std::ptr::copy_nonoverlapping(q_gate.as_ptr(), q_gate_buf.contents() as *mut f32, q_dim);
            std::ptr::copy_nonoverlapping(hidden.as_ptr(), hidden_buf.contents() as *mut f32, hd);
        }

        let attn = FullAttnGpuOut {
            q_buf, q_gate_buf, kc_buf, vc_buf, scores_buf, out_buf, hidden_buf,
            seq_len: seq_len as u32,
            seq_stride: MAX_SEQ as u32,
            num_attn_heads: num_q as u32,
            head_dim: head_dim as u32,
            kv_dim: kv_dim as u32,
            heads_per_kv: (num_q / num_kv) as u32,
            scale: 1.0f32 / (head_dim as f32).sqrt(),
            q_dim: q_dim as u32,
            o_prefix: format!("{}.o_proj", prefix),
        };

        moe_layer_forward(
            wf, layer, hidden, &self.model.expert_files[layer],
            self.ctx, self.gpu_wf, self.k,
            Some(attn), self.expert_gpu_buffer.as_deref_mut(),
            &mut self.timing,
        )
    }

    // ── Linear layer group (N+1 pipelined) ─────────────────────────────────

    fn linear_group(&mut self, first_layer: usize, count: usize, hidden: &mut [f32]) {
        let layers: Vec<usize> = (first_layer..first_layer + count).collect();
        let m = layers.len();
        if m == 0 { return; }
        let last_layer = layers[m - 1];

        let hd = HIDDEN_DIM;
        let num_experts = NUM_EXPERTS;
        let moe_inter = MOE_INTERMEDIATE;
        let shared_inter = SHARED_INTERMEDIATE;
        let k = self.k;
        let qkv_dim = LINEAR_CONV_DIM;
        let total_key = LINEAR_TOTAL_KEY;
        let total_val = LINEAR_TOTAL_VALUE;
        let num_k_heads = LINEAR_NUM_K_HEADS;
        let num_v_heads = LINEAR_NUM_V_HEADS;
        let key_dim = total_key / num_k_heads;
        let val_dim = total_val / num_v_heads;
        let inv_scale = 1.0 / (key_dim as f32).sqrt();
        let k_heads_per_v = num_v_heads / num_k_heads;

        // Upload first layer: buf_moe_hidden = hidden (h_mid), buf_input = input_norm(hidden)
        {
            let buf_moe = self.ctx.buf_moe_hidden.as_ref().unwrap();
            unsafe { std::ptr::copy_nonoverlapping(hidden.as_ptr(), buf_moe.contents() as *mut f32, hd); }
            let buf_in = self.ctx.buf_input.as_ref().unwrap();
            let norm_name = format!("model.layers.{}.input_layernorm.weight", first_layer);
            if let Some(nw_u16) = self.model.wf.get_tensor_u16(&norm_name) {
                let nw: Vec<f32> = nw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
                let ssq: f32 = hidden[..hd].iter().map(|v| v * v).sum();
                let inv_rms = 1.0 / (ssq / hd as f32 + RMS_NORM_EPS).sqrt();
                unsafe {
                    let dst = buf_in.contents() as *mut f32;
                    for i in 0..hd { *dst.add(i) = hidden[i] * inv_rms * nw[i]; }
                }
            } else {
                unsafe { std::ptr::copy_nonoverlapping(hidden.as_ptr(), buf_in.contents() as *mut f32, hd); }
            }
        }

        // CMD 0: pre_expert(first_layer)
        {
            let li = first_layer - (first_layer + 1) / FULL_ATTN_INTERVAL;
            let cmd = self.ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            encode_pre_expert(
                &self.model.wf, self.gpu_wf, self.ctx, &enc, first_layer, li,
                hd, num_k_heads, num_v_heads, total_key, total_val, qkv_dim,
                key_dim, val_dim, k_heads_per_v, inv_scale, num_experts, shared_inter,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            if let Some(ref mut s) = self.cache.lin[first_layer] {
                update_conv_state(s, qkv_dim);
            }
        }

        let (_, mut prev_weights, mut prev_gate_score) = route_and_pread(
            self.ctx, self.expert_gpu_buffer.as_mut().unwrap(),
            first_layer, &self.model.expert_files[first_layer],
            num_experts, k,
            &mut self.timing,
        );

        // CMD 1..N-1: fused post_expert(prev) + pre_expert(curr)
        for gi in 1..m {
            let prev_layer = layers[gi - 1];
            let curr_layer = layers[gi];
            let curr_li = curr_layer - (curr_layer + 1) / FULL_ATTN_INTERVAL;

            let next_norm_info = self.model.wf.get_tensor_ptr(
                &format!("model.layers.{}.input_layernorm.weight", curr_layer))
                .map(|p| (p as *const c_void, self.gpu_wf.base as usize));

            let cmd = self.ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            {
                let io = self.expert_gpu_buffer.as_ref().unwrap();
                encode_post_expert(
                    &self.model.wf, self.gpu_wf, self.ctx, &enc, prev_layer,
                    &prev_weights, prev_gate_score,
                    &io.expert_data, &io.scratch_gate, &io.scratch_up, &io.scratch_act,
                    &io.expert_out, &io.shared_act, &io.shared_down, &io.combine_params,
                    next_norm_info,
                    hd, moe_inter, shared_inter, k,
                );
            }
            encode_pre_expert(
                &self.model.wf, self.gpu_wf, self.ctx, &enc, curr_layer, curr_li,
                hd, num_k_heads, num_v_heads, total_key, total_val, qkv_dim,
                key_dim, val_dim, k_heads_per_v, inv_scale, num_experts, shared_inter,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            if let Some(ref mut s) = self.cache.lin[curr_layer] {
                update_conv_state(s, qkv_dim);
            }

            let (_, weights, gate_score) = route_and_pread(
                self.ctx, self.expert_gpu_buffer.as_mut().unwrap(),
                curr_layer, &self.model.expert_files[curr_layer],
                num_experts, k,
                &mut self.timing,
            );
            prev_weights = weights;
            prev_gate_score = gate_score;
        }

        // Last CMD: post_expert(last_layer)
        {
            let next_norm_info = if last_layer + 1 < NUM_LAYERS {
                self.model.wf.get_tensor_ptr(
                    &format!("model.layers.{}.input_layernorm.weight", last_layer + 1))
                    .map(|p| (p as *const c_void, self.gpu_wf.base as usize))
            } else {
                None
            };

            let cmd = self.ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            {
                let io = self.expert_gpu_buffer.as_ref().unwrap();
                encode_post_expert(
                    &self.model.wf, self.gpu_wf, self.ctx, &enc, last_layer,
                    &prev_weights, prev_gate_score,
                    &io.expert_data, &io.scratch_gate, &io.scratch_up, &io.scratch_act,
                    &io.expert_out, &io.shared_act, &io.shared_down, &io.combine_params,
                    next_norm_info,
                    hd, moe_inter, shared_inter, k,
                );
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.ctx.buf_moe_hidden.as_ref().unwrap().contents() as *const f32,
                    hidden.as_mut_ptr(), hd);
            }
        }
    }

    // ── Final norm + LM head ───────────────────────────────────────────────

    fn final_norm_and_lm_head(&self, hidden: &mut [f32], logits: &mut [f32]) {
        final_norm(&self.model.wf, hidden, HIDDEN_DIM);
        gpu_lm_head(&self.model.wf, hidden, logits, self.gpu_wf, self.ctx);
    }
}

// ─── MoE layer forward (local copy, fusedexp-specific) ────────────────────

fn moe_layer_forward(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    expert_file: &ExpertFile,
    ctx: &MetalContext,
    gpu_wf: &WeightBuffer,
    
    k: usize,
    attn_state: Option<FullAttnGpuOut>,
    mut expert_gpu_buffer: Option<&mut ExpertBuffer>,
    timing: &mut BTreeMap<String, TelemetryValue>,
) -> Option<DeferredExperts> {
    let hidden_dim = HIDDEN_DIM;
    let num_experts = NUM_EXPERTS;
    let moe_inter = MOE_INTERMEDIATE;
    let shared_inter = SHARED_INTERMEDIATE;
    let expert_size = EXPERT_SIZE_4BIT;

    // Save h_mid (residual)
    let h_mid = hidden.to_vec();

    let post_norm_name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
    let mut h_post = vec![0.0f32; hidden_dim];

    // ── Router gate + shared expert projections ──
    let mut gate_scores = vec![0.0f32; num_experts];
    let mut shared_gate = vec![0.0f32; shared_inter];
    let mut shared_up = vec![0.0f32; shared_inter];
    let shared_gate_score;

    let prefix = format!("model.layers.{}.mlp", layer_idx);

    // GPU buffers preserved for expert dispatch combine
    let mut sg_buf_gpu: Option<Buffer>;
    let mut su_buf_gpu: Option<Buffer>;
    let mut hmid_gpu_override: Option<Buffer> = None;

    // ── CMD2 fusion path: batched attn + o_proj + residual + norm + gate ──
    let use_cmd2_fusion = attn_state.is_some()
        && ctx.attn_scores_batched.is_some()
        && ctx.attn_softmax_batched.is_some()
        && ctx.attn_values_batched.is_some()
        && ctx.sigmoid_gate.is_some()
        && ctx.residual_add.is_some()
        && ctx.rms_norm_apply_bf16.is_some();

    if use_cmd2_fusion {
        let attn = attn_state.unwrap();

        // Use pre-allocated buffers from MetalContext (matching C)
        let o_proj_buf = ctx.buf_out_proj.as_ref().unwrap().clone();
        let temp_buf = ctx.buf_temp_residual.as_ref().unwrap().clone();
        let sum_sq_buf = ctx.buf_post_sum_sq.as_ref().unwrap().clone();
        let normed_buf = ctx.buf_post_normed.as_ref().unwrap().clone();
        let gate_buf = ctx.buf_gate_scores.as_ref().unwrap().clone();
        let sg_buf = ctx.buf_shared_gate.as_ref().unwrap().clone();
        let su_buf = ctx.buf_shared_up.as_ref().unwrap().clone();
        let sge_buf = ctx.buf_shared_gate_score.as_ref().unwrap().clone();

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        // Step 1: attn_scores_batched
        {
            let pipe = ctx.attn_scores_batched.as_ref().unwrap();
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
            let pipe = ctx.attn_softmax_batched.as_ref().unwrap();
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
            let pipe = ctx.attn_values_batched.as_ref().unwrap();
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

        // Step 4: sigmoid_gate
        {
            let pipe = ctx.sigmoid_gate.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&attn.out_buf), 0);
            enc.set_buffer(1, Some(&attn.q_gate_buf), 0);
            enc.set_bytes(2, 4, &attn.q_dim as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((attn.q_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 5: o_proj matvec
        gpu_wf.encode_matvec_into(wf, ctx, &enc, &attn.o_prefix, &attn.out_buf, 0, &o_proj_buf, 0, hidden_dim, attn.q_dim as usize);

        // Step 6: residual_add
        {
            let pipe = ctx.residual_add.as_ref().unwrap();
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

        // Step 7: post_attention_layernorm
        {
            enc.set_compute_pipeline_state(&ctx.rms_norm_sum);
            enc.set_buffer(0, Some(&temp_buf), 0);
            enc.set_buffer(1, Some(&sum_sq_buf), 0);
            enc.set_bytes(2, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        }
        {
            let pipe = ctx.rms_norm_apply_bf16.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&temp_buf), 0);
            let pnw_ptr = wf.get_tensor_ptr(&post_norm_name).unwrap();
            let pnw_off = (pnw_ptr as usize - gpu_wf.base as usize) as u64;
            enc.set_buffer(1, Some(&gpu_wf.buf), pnw_off);
            enc.set_buffer(2, Some(&sum_sq_buf), 0);
            enc.set_buffer(3, Some(&normed_buf), 0);
                enc.set_bytes(4, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Steps 8-11: gate, shared_gate, shared_up, shared_expert_gate matvecs
        gpu_wf.encode_matvec_into(wf, ctx, &enc, &format!("{}.gate", prefix), &normed_buf, 0, &gate_buf, 0, num_experts, hidden_dim);
        gpu_wf.encode_matvec_into(wf, ctx, &enc, &format!("{}.shared_expert.gate_proj", prefix), &normed_buf, 0, &sg_buf, 0, shared_inter, hidden_dim);
        gpu_wf.encode_matvec_into(wf, ctx, &enc, &format!("{}.shared_expert.up_proj", prefix), &normed_buf, 0, &su_buf, 0, shared_inter, hidden_dim);
        gpu_wf.encode_matvec_into(wf, ctx, &enc, &format!("{}.shared_expert_gate", prefix), &normed_buf, 0, &sge_buf, 0, 1, hidden_dim);

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

        sg_buf_gpu = Some(sg_buf);
        su_buf_gpu = Some(su_buf);
        h_post.copy_from_slice(hidden);
        hmid_gpu_override = Some(temp_buf);
    } else {
        // Non-fused path: post-norm on CPU, router CMD separately
        // (fallback, shouldn't normally be reached in FusedExp)
        let pnw = wf.get_tensor_u16(&post_norm_name);
        if let Some(pnw) = pnw {
            let pnw_f32: Vec<f32> = pnw.iter().map(|&v| bf16_to_f32(v)).collect();
            rms_norm(hidden, &pnw_f32, &mut h_post, hidden_dim, RMS_NORM_EPS);
        } else {
            h_post.copy_from_slice(hidden);
        }

        // Router gate + shared expert projections on GPU (reuse pre-allocated buffers)
        let x_buf = ctx.buf_post_normed.as_ref().unwrap().clone();
        unsafe { let dst = x_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(h_post.as_ptr(), dst, hidden_dim); }
        let gate_buf = ctx.buf_gate_scores.as_ref().unwrap().clone();
        let sg_buf = ctx.buf_shared_gate.as_ref().unwrap().clone();
        let su_buf = ctx.buf_shared_up.as_ref().unwrap().clone();
        let sge_buf = ctx.buf_shared_gate_score.as_ref().unwrap().clone();

        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        gpu_wf.encode_matvec_into(wf, ctx, &enc, &format!("{}.gate", prefix), &x_buf, 0, &gate_buf, 0, num_experts, hidden_dim);
        gpu_wf.encode_matvec_into(wf, ctx, &enc, &format!("{}.shared_expert.gate_proj", prefix), &x_buf, 0, &sg_buf, 0, shared_inter, hidden_dim);
        gpu_wf.encode_matvec_into(wf, ctx, &enc, &format!("{}.shared_expert.up_proj", prefix), &x_buf, 0, &su_buf, 0, shared_inter, hidden_dim);
        gpu_wf.encode_matvec_into(wf, ctx, &enc, &format!("{}.shared_expert_gate", prefix), &x_buf, 0, &sge_buf, 0, 1, hidden_dim);
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe {
            std::ptr::copy_nonoverlapping(gate_buf.contents() as *const f32, gate_scores.as_mut_ptr(), num_experts);
            std::ptr::copy_nonoverlapping(sg_buf.contents() as *const f32, shared_gate.as_mut_ptr(), shared_inter);
            std::ptr::copy_nonoverlapping(su_buf.contents() as *const f32, shared_up.as_mut_ptr(), shared_inter);
            shared_gate_score = *(sge_buf.contents() as *const f32);
        }
        sg_buf_gpu = Some(sg_buf);
        su_buf_gpu = Some(su_buf);
    }

    // ── Routing: softmax + topk ──
    softmax(&mut gate_scores);
    let mut expert_indices = vec![0usize; k];
    let mut expert_weights = vec![0.0f32; k];
    topk(&gate_scores, k, &mut expert_indices, &mut expert_weights);
    normalize_weights(&mut expert_weights);

    // ── Routed expert computation ──
    let mut moe_out = vec![0.0f32; hidden_dim];
    let actual_k = k.min(MAX_K);
    let hidden_u32 = hidden_dim as u32;
    let inter_u32 = moe_inter as u32;
    let gs_u32 = GROUP_SIZE as u32;

    // Phase 1: Parallel pread (cache hits skip I/O)
    let mut valid = [false; MAX_K];
    let mut fallback_expert_bufs: Vec<Buffer> = Vec::new();

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

        for m in 0..miss_count {
            let ki = miss_k_slot[m];
            let eidx = miss_ei[m];
            let buf = io.cache.insert_get_buf(layer_idx, eidx);
            io.expert_data[ki] = buf;
        }

        if miss_count > 0 {
            let t_io = Instant::now();
            for m in 0..miss_count {
                let ki = miss_k_slot[m];
                let eidx = miss_ei[m];
                let ptr = io.expert_data[ki].contents() as *mut u8;
                let dst = unsafe { std::slice::from_raw_parts_mut(ptr, expert_size) };
                expert_file.read_expert(eidx, dst).unwrap();
            }
            let dt = t_io.elapsed().as_secs_f64() * 1000.0;
            timing_add(timing, "engine.expert_io_ms", dt);
        }
        for m in 0..miss_count {
            valid[miss_k_slot[m]] = true;
        }
    } else {
        let t_io = Instant::now();
        for ki in 0..actual_k {
            let eidx = expert_indices[ki];
            let buf = metal_buf_shared(&ctx.device, expert_size);
            let mut dst = vec![0u8; expert_size];
            if expert_file.read_expert(eidx, &mut dst).is_ok() {
                unsafe {
                    let ptr = buf.contents() as *mut u8;
                    std::ptr::copy_nonoverlapping(dst.as_ptr(), ptr, expert_size);
                }
                valid[ki] = true;
            }
            fallback_expert_bufs.push(buf);
        }
        let dt = t_io.elapsed().as_secs_f64() * 1000.0;
        timing_add(timing, "engine.expert_io_ms", dt);
    }

    let any_valid = valid.iter().take(actual_k).any(|&v| v);

    if any_valid {
        let io_ref = expert_gpu_buffer.as_deref();

        let (x_buf, gate_out, up_out, act_out, out_bufs,
             shared_act_gpu, shared_down_gpu, _hidden_out, params_buf)
            = if let Some(io) = io_ref {
            unsafe { let dst = io.input_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(h_post.as_ptr(), dst, hidden_dim); }
            let ob: Vec<Buffer> = io.expert_out.iter().take(actual_k).cloned().collect();
            (io.input_buf.clone(), io.scratch_gate.clone(), io.scratch_up.clone(),
             io.scratch_act.clone(), ob,
             io.shared_act.clone(), io.shared_down.clone(),
             io.combine_out.clone(), io.combine_params.clone())
        } else {
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

        let hmid_gpu = if let Some(buf) = hmid_gpu_override.take() {
            buf
        } else {
            let buf = metal_buf_shared(&ctx.device, hidden_dim * 4);
            unsafe { let dst = buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(h_mid.as_ptr(), dst, hidden_dim); }
            buf
        };

        // Fused CMD: K experts + shared SwiGLU + shared down_proj + moe_combine_residual
        let cmd_buf = ctx.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        for ki in 0..actual_k {
            if !valid[ki] { continue; }
            let expert_buf: &Buffer = if let Some(io) = io_ref {
                &io.expert_data[ki]
            } else if ki < fallback_expert_bufs.len() {
                &fallback_expert_bufs[ki]
            } else {
                continue;
            };
            metal_kernels::encode_matvec_offset(ctx, &enc,
                expert_buf, GATE_W_OFF as u64,
                expert_buf, GATE_S_OFF as u64,
                expert_buf, GATE_B_OFF as u64,
                &x_buf, 0, &gate_out, 0,
                inter_u32, hidden_u32, gs_u32, 3);

            metal_kernels::encode_matvec_offset(ctx, &enc,
                expert_buf, UP_W_OFF as u64,
                expert_buf, UP_S_OFF as u64,
                expert_buf, UP_B_OFF as u64,
                &x_buf, 0, &up_out, 0,
                inter_u32, hidden_u32, gs_u32, 3);

            metal_kernels::encode_swiglu(ctx, &enc, &gate_out, 0, &up_out, 0, &act_out, 0, inter_u32);

            metal_kernels::encode_matvec_offset(ctx, &enc,
                expert_buf, DOWN_W_OFF as u64,
                expert_buf, DOWN_S_OFF as u64,
                expert_buf, DOWN_B_OFF as u64,
                &act_out, 0, &out_bufs[ki], 0,
                hidden_u32, inter_u32, gs_u32, 3);
        }

        // Shared expert SwiGLU on GPU
        if let (Some(ref sg), Some(ref su)) = (sg_buf_gpu.as_ref(), su_buf_gpu.as_ref()) {
            metal_kernels::encode_swiglu(ctx, &enc, sg, 0, su, 0, &shared_act_gpu, 0, shared_inter as u32);
        }

        gpu_wf.encode_matvec_into(wf, ctx, &enc,
            &format!("{}.shared_expert.down_proj", prefix),
            &shared_act_gpu, 0, &shared_down_gpu, 0, hidden_dim, shared_inter);

        // moe_combine_residual
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

        // GPU-side input_norm for next layer
        let do_gpu_norm = layer_idx + 1 < NUM_LAYERS
            && ctx.rms_norm_apply_bf16.is_some()
            && wf.get_tensor_ptr(&format!("model.layers.{}.input_layernorm.weight", layer_idx + 1)).is_some();
        if do_gpu_norm {
            let next_norm_ptr = wf.get_tensor_ptr(
                &format!("model.layers.{}.input_layernorm.weight", layer_idx + 1)).unwrap();
            let next_norm_off = (next_norm_ptr as usize - gpu_wf.base as usize) as u64;
            let buf_moe = ctx.buf_moe_hidden.as_ref().unwrap();

            {
                enc.set_compute_pipeline_state(&ctx.rms_norm_sum);
                enc.set_buffer(0, Some(buf_moe), 0);
                enc.set_buffer(1, Some(ctx.buf_cmd3_sum_sq.as_ref().unwrap()), 0);
                enc.set_bytes(2, 4, &hidden_u32 as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            }
            {
                let pipe = ctx.rms_norm_apply_bf16.as_ref().unwrap();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(buf_moe), 0);
                enc.set_buffer(1, Some(&gpu_wf.buf), next_norm_off);
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
        if let Some(b) = sg_buf_gpu.take() { keep_alive.push(b); }
        if let Some(b) = su_buf_gpu.take() { keep_alive.push(b); }

        return Some(DeferredExperts {
            cmd_buf: Some(cmd_buf.to_owned()),
            out_buf: ctx.buf_moe_hidden.clone(),
            _keep_alive: keep_alive,
        });
    }

    // ── CPU fallback: compute everything synchronously ──
    let mut expert_data = vec![0u8; expert_size];
    let mut gate_tmp = vec![0.0f32; moe_inter];
    let mut up_tmp = vec![0.0f32; moe_inter];
    let mut act_tmp = vec![0.0f32; moe_inter];
    let mut eout = vec![0.0f32; hidden_dim];

    for (&eidx, &ew) in expert_indices.iter().zip(expert_weights.iter()) {
        if expert_file.read_expert(eidx, &mut expert_data).is_err() { continue; }

        let gw = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(GATE_W_OFF) as *const u32, GATE_W_SIZE / 4) };
        let gs = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(GATE_S_OFF) as *const u16, GATE_S_SIZE / 2) };
        let gb = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(GATE_B_OFF) as *const u16, GATE_B_SIZE / 2) };
        dequant_matvec_4bit(gw, gs, gb, &h_post, &mut gate_tmp, moe_inter, hidden_dim, GROUP_SIZE);

        let uw = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(UP_W_OFF) as *const u32, UP_W_SIZE / 4) };
        let us = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(UP_S_OFF) as *const u16, UP_S_SIZE / 2) };
        let ub = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(UP_B_OFF) as *const u16, UP_B_SIZE / 2) };
        dequant_matvec_4bit(uw, us, ub, &h_post, &mut up_tmp, moe_inter, hidden_dim, GROUP_SIZE);

        for i in 0..moe_inter {
            let g = gate_tmp[i];
            let silu_g = g / (1.0 + (-g).exp());
            act_tmp[i] = silu_g * up_tmp[i];
        }

        let dw = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(DOWN_W_OFF) as *const u32, DOWN_W_SIZE / 4) };
        let ds = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(DOWN_S_OFF) as *const u16, DOWN_S_SIZE / 2) };
        let db = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(DOWN_B_OFF) as *const u16, DOWN_B_SIZE / 2) };
        dequant_matvec_4bit(dw, ds, db, &act_tmp, &mut eout, hidden_dim, moe_inter, GROUP_SIZE);

        for d in 0..hidden_dim { moe_out[d] += eout[d] * ew; }
    }

    // ── Shared expert SwiGLU + down_proj ──
    let mut shared_out = vec![0.0f32; hidden_dim];
    let mut shared_act = vec![0.0f32; shared_inter];
    for i in 0..shared_inter {
        let g = shared_gate[i];
        shared_act[i] = (g / (1.0 + (-g).exp())) * shared_up[i];
    }

    if let (Some(sdw), Some(sds), Some(sdb)) = (
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

    None
}

// ─── Private helpers ──────────────────────────────────────────────────────

/// Shift the CPU-side conv1d ring buffer one slot left, zero the new slot.
///
/// GPU paths process conv1d entirely on GPU (`buf_conv_state`). The CPU-side
/// `conv_state` is shadow state that's never read — shifting + zeroing
/// preserves the ring-buffer structure without a costly GPU→CPU readback.
fn update_conv_state(state: &mut LinearAttnState, qkv_dim: usize) {
    let state_off = (CONV_KERNEL_SIZE - 2) * qkv_dim;
    state.conv_state.copy_within(qkv_dim.., 0);
    state.conv_state[state_off..state_off + qkv_dim].fill(0.0);
}

fn route_and_pread(
    ctx: &MetalContext,
    expert_gpu_buffer: &mut ExpertBuffer,
    layer_idx: usize,
    expert_file: &ExpertFile,
    num_experts: usize,
    k: usize,
    timing: &mut BTreeMap<String, TelemetryValue>,
) -> (Vec<usize>, Vec<f32>, f32) {
    let gate_buf = ctx.buf_gate_scores.as_ref().unwrap();
    let mut gate_scores = vec![0.0f32; num_experts];
    unsafe {
        std::ptr::copy_nonoverlapping(
            gate_buf.contents() as *const f32, gate_scores.as_mut_ptr(), num_experts);
    }
    let shared_gate_score =
        unsafe { *(ctx.buf_shared_gate_score.as_ref().unwrap().contents() as *const f32) };

    softmax(&mut gate_scores);
    let mut expert_indices = vec![0usize; k];
    let mut expert_weights = vec![0.0f32; k];
    topk(&gate_scores, k, &mut expert_indices, &mut expert_weights);
    normalize_weights(&mut expert_weights);

    let actual_k = k.min(MAX_K);
    let mut miss_ei = [0usize; MAX_K];
    let mut miss_k_slot = [0usize; MAX_K];
    let mut miss_count = 0;
    for ki in 0..actual_k {
        let eidx = expert_indices[ki];
        if let Some(buf) = expert_gpu_buffer.cache.lookup(layer_idx, eidx) {
            expert_gpu_buffer.expert_data[ki] = buf;
        } else {
            miss_ei[miss_count] = eidx;
            miss_k_slot[miss_count] = ki;
            miss_count += 1;
        }
    }
    for m in 0..miss_count {
        let ki = miss_k_slot[m];
        let eidx = miss_ei[m];
        let buf = expert_gpu_buffer.cache.insert_get_buf(layer_idx, eidx);
        expert_gpu_buffer.expert_data[ki] = buf;
    }
    if miss_count > 0 {
        let t_io = Instant::now();
        let expert_size = expert_file.expert_size();
        for m in 0..miss_count {
            let ki = miss_k_slot[m];
            let eidx = miss_ei[m];
            let ptr = expert_gpu_buffer.expert_data[ki].contents() as *mut u8;
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr, expert_size) };
            expert_file.read_expert(eidx, dst).unwrap();
        }
        let dt = t_io.elapsed().as_secs_f64() * 1000.0;
        timing_add(timing, "engine.expert_io_ms", dt);
    }
    (expert_indices, expert_weights, shared_gate_score)
}

fn encode_pre_expert(
    wf: &WeightFile,
    gpu_wf: &WeightBuffer,
    ctx: &MetalContext,
    enc: &ComputeCommandEncoderRef,
    layer_idx: usize,
    linear_idx: usize,
    hidden_dim: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    total_key: usize,
    total_value: usize,
    qkv_dim: usize,
    key_dim: usize,
    value_dim: usize,
    k_heads_per_v: usize,
    inv_scale: f32,
    num_experts: usize,
    shared_inter: usize,
) {
    let c = ctx;
    let gw = gpu_wf;
    debug_assert!(linear_idx < c.buf_conv_state.len(),
        "linear_idx {} out of bounds for buf_conv_state (len {})", linear_idx, c.buf_conv_state.len());
    debug_assert!(linear_idx < c.buf_delta_state.len(),
        "linear_idx {} out of bounds for buf_delta_state (len {})", linear_idx, c.buf_delta_state.len());
    debug_assert!(c.batch_out.len() >= 7,
        "batch_out too short (len {}), need >= 7", c.batch_out.len());
    let prefix = format!("model.layers.{}.linear_attn", layer_idx);

    let input_buf = c.buf_input.as_ref().unwrap();
    {
        gw.encode_matvec_into(wf, c, enc, &format!("{}.in_proj_qkv", prefix), input_buf, 0, &c.batch_out[0], 0, qkv_dim, hidden_dim);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.in_proj_z", prefix), input_buf, 0, &c.batch_out[1], 0, total_value, hidden_dim);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.in_proj_b", prefix), input_buf, 0, &c.batch_out[2], 0, num_v_heads, hidden_dim);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.in_proj_a", prefix), input_buf, 0, &c.batch_out[3], 0, num_v_heads, hidden_dim);
    }

    if let Some(conv_w_ptr) = wf.get_tensor_ptr(&format!("{}.conv1d.weight", prefix)) {
        let conv_w_off = (conv_w_ptr as usize - gw.base as usize) as u64;
        metal_kernels::encode_conv1d_step(c, enc,
            &c.buf_conv_state[linear_idx],
            &c.batch_out[0],
            &gw.buf, conv_w_off,
            c.buf_conv_output.as_ref().unwrap(),
            qkv_dim as u32);
    }

    metal_kernels::encode_rms_norm_qk(c, enc,
        c.buf_conv_output.as_ref().unwrap(), 0,
        c.buf_conv_output.as_ref().unwrap(), (total_key * 4) as u64,
        num_k_heads as u32, key_dim as u32, inv_scale);

    {
        let a_log_ptr = wf.get_tensor_ptr(&format!("{}.A_log", prefix));
        let dt_bias_ptr = wf.get_tensor_ptr(&format!("{}.dt_bias", prefix));
        let a_log_off = a_log_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
        let dt_bias_off = dt_bias_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
        metal_kernels::encode_compute_decay_beta(c, enc,
            &c.batch_out[3],
            &c.batch_out[2],
            if a_log_ptr.is_some() { &gw.buf } else { &c.batch_out[3] }, a_log_off,
            if dt_bias_ptr.is_some() { &gw.buf } else { &c.batch_out[2] }, dt_bias_off,
            c.buf_delta_g_decay.as_ref().unwrap(),
            c.buf_delta_beta.as_ref().unwrap(),
            num_v_heads as u32);
    }

    {
        let q_off = 0u64;
        let k_off = (total_key * 4) as u64;
        let v_off = (2 * total_key * 4) as u64;
        let conv_out = c.buf_conv_output.as_ref().unwrap();
        metal_kernels::encode_gated_delta_net_step(c, enc,
            &c.buf_delta_state[linear_idx],
            conv_out, q_off,
            conv_out, k_off,
            conv_out, v_off,
            c.buf_delta_g_decay.as_ref().unwrap(),
            c.buf_delta_beta.as_ref().unwrap(),
            c.buf_delta_output.as_ref().unwrap(),
            num_v_heads as u32, k_heads_per_v as u32, key_dim as u32, value_dim as u32);
    }

    {
        let gnw_ptr = wf.get_tensor_ptr(&format!("{}.norm.weight", prefix));
        if let Some(gnw_p) = gnw_ptr {
            let gnw_off = (gnw_p as usize - gw.base as usize) as u64;
            metal_kernels::encode_gated_rms_norm(c, enc,
                c.buf_delta_output.as_ref().unwrap(),
                &c.batch_out[1],
                &gw.buf, gnw_off,
                &c.batch_out[6],
                num_v_heads as u32, value_dim as u32);
        }
    }

    let out_proj_buf = c.buf_out_proj.as_ref().unwrap();
    gw.encode_matvec_into(wf, c, enc, &format!("{}.out_proj", prefix),
        &c.batch_out[6], 0, out_proj_buf, 0, hidden_dim, total_value);

    {
        let pipe = c.residual_add.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(out_proj_buf), 0);
        enc.set_buffer(1, Some(c.buf_moe_hidden.as_ref().unwrap()), 0);
        enc.set_buffer(2, Some(c.buf_temp_residual.as_ref().unwrap()), 0);
        enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }

    let post_norm_name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
    let pnw_ptr = wf.get_tensor_ptr(&post_norm_name).unwrap();
    let pnw_off = (pnw_ptr as usize - gw.base as usize) as u64;
    let temp_res = c.buf_temp_residual.as_ref().unwrap();
    let post_sum = c.buf_post_sum_sq.as_ref().unwrap();
    {
        enc.set_compute_pipeline_state(&c.rms_norm_sum);
        enc.set_buffer(0, Some(temp_res), 0);
        enc.set_buffer(1, Some(post_sum), 0);
        enc.set_bytes(2, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
        enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));
    }
    {
        let pipe = c.rms_norm_apply_bf16.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(temp_res), 0);
        enc.set_buffer(1, Some(&gw.buf), pnw_off);
        enc.set_buffer(2, Some(post_sum), 0);
        enc.set_buffer(3, Some(c.buf_post_normed.as_ref().unwrap()), 0);
        {
            enc.set_bytes(4, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
        }
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }

    let mlp_prefix = format!("model.layers.{}.mlp", layer_idx);
    let post_normed = c.buf_post_normed.as_ref().unwrap();
    gw.encode_matvec_into(wf, c, enc, &format!("{}.gate", mlp_prefix), post_normed, 0, c.buf_gate_scores.as_ref().unwrap(), 0, num_experts, hidden_dim);
    gw.encode_matvec_into(wf, c, enc, &format!("{}.shared_expert.gate_proj", mlp_prefix), post_normed, 0, c.buf_shared_gate.as_ref().unwrap(), 0, shared_inter, hidden_dim);
    gw.encode_matvec_into(wf, c, enc, &format!("{}.shared_expert.up_proj", mlp_prefix), post_normed, 0, c.buf_shared_up.as_ref().unwrap(), 0, shared_inter, hidden_dim);
    gw.encode_matvec_into(wf, c, enc, &format!("{}.shared_expert_gate", mlp_prefix), post_normed, 0, c.buf_shared_gate_score.as_ref().unwrap(), 0, 1, hidden_dim);
}

fn encode_post_expert(
    wf: &WeightFile,
    gpu_wf: &WeightBuffer,
    ctx: &MetalContext,
    enc: &ComputeCommandEncoderRef,
    layer_idx: usize,
    expert_weights: &[f32],
    shared_gate_score: f32,
    expert_data: &[Buffer],
    expert_scratch_gate: &Buffer,
    expert_scratch_up: &Buffer,
    expert_scratch_act: &Buffer,
    expert_out_bufs: &[Buffer],
    shared_scratch: &Buffer,
    shared_down_buf: &Buffer,
    combine_params_buf: &Buffer,
    next_norm_weight: Option<(*const c_void, usize)>,
    hidden_dim: usize,
    moe_inter: usize,
    shared_inter: usize,
    num_experts_per_tok: usize,
    
) {
    let hidden_u32 = hidden_dim as u32;
    let inter_u32 = moe_inter as u32;
    let gs_u32 = GROUP_SIZE as u32;
    let actual_k = num_experts_per_tok.min(MAX_K);
    let prefix = format!("model.layers.{}.mlp", layer_idx);

    let post_normed = ctx.buf_post_normed.as_ref().unwrap();

    for ki in 0..actual_k {
        let expert_buf = &expert_data[ki];
        if expert_buf.length() == 0 { continue; }

        metal_kernels::encode_matvec_offset(ctx, enc,
            expert_buf, GATE_W_OFF as u64,
            expert_buf, GATE_S_OFF as u64,
            expert_buf, GATE_B_OFF as u64,
            post_normed, 0, expert_scratch_gate, 0,
            inter_u32, hidden_u32, gs_u32, 3);

        metal_kernels::encode_matvec_offset(ctx, enc,
            expert_buf, UP_W_OFF as u64,
            expert_buf, UP_S_OFF as u64,
            expert_buf, UP_B_OFF as u64,
            post_normed, 0, expert_scratch_up, 0,
            inter_u32, hidden_u32, gs_u32, 3);

        metal_kernels::encode_swiglu(ctx, enc, expert_scratch_gate, 0, expert_scratch_up, 0,
            expert_scratch_act, 0, inter_u32);

        metal_kernels::encode_matvec_offset(ctx, enc,
            expert_buf, DOWN_W_OFF as u64,
            expert_buf, DOWN_S_OFF as u64,
            expert_buf, DOWN_B_OFF as u64,
            expert_scratch_act, 0, &expert_out_bufs[ki], 0,
            hidden_u32, inter_u32, gs_u32, 3);
    }

    {
        let sg = ctx.buf_shared_gate.as_ref().unwrap();
        let su = ctx.buf_shared_up.as_ref().unwrap();
        metal_kernels::encode_swiglu(ctx, enc, sg, 0, su, 0, shared_scratch, 0, shared_inter as u32);
    }

    let sd_name = format!("{}.shared_expert.down_proj", prefix);
    if !gpu_wf.encode_matvec_into(wf, ctx, enc, &sd_name, shared_scratch, 0,
        shared_down_buf, 0, hidden_dim, shared_inter)
    {
        eprintln!("[fusedexp] WARNING: shared expert down_proj tensor not found: {}", sd_name);
    }

    {
        let mcr_pipe = ctx.moe_combine_residual.as_ref().unwrap();
        enc.set_compute_pipeline_state(mcr_pipe);
        let hmid_src = ctx.buf_temp_residual.as_ref().unwrap();
        enc.set_buffer(0, Some(hmid_src), 0);
        enc.set_buffer(1, Some(shared_down_buf), 0);
        enc.set_buffer(2, Some(ctx.buf_moe_hidden.as_ref().unwrap()), 0);
        for ei in 0..MAX_K {
            if ei < actual_k {
                enc.set_buffer(3 + ei as u64, Some(&expert_out_bufs[ei]), 0);
            } else {
                enc.set_buffer(3 + ei as u64, Some(ctx.buf_moe_hidden.as_ref().unwrap()), 0);
            }
        }
        let mut cparams = [0.0f32; 10];
        for (i, &w) in expert_weights.iter().enumerate() { cparams[i] = w; }
        cparams[8] = shared_gate_score;
        unsafe { std::ptr::copy_nonoverlapping(cparams.as_ptr(), combine_params_buf.contents() as *mut f32, 10); }
        enc.set_buffer(11, Some(combine_params_buf), 0);
        {
            enc.set_bytes(12, 4, &hidden_u32 as *const u32 as *const c_void);
            let ku = actual_k as u32;
            enc.set_bytes(13, 4, &ku as *const u32 as *const c_void);
        }
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }

    if let Some((norm_ptr, base)) = next_norm_weight {
        let norm_off = (norm_ptr as usize - base) as u64;
        let buf_moe = ctx.buf_moe_hidden.as_ref().unwrap();
        let sum_sq = ctx.buf_cmd3_sum_sq.as_ref().unwrap();

        enc.set_compute_pipeline_state(&ctx.rms_norm_sum);
        enc.set_buffer(0, Some(buf_moe), 0);
        enc.set_buffer(1, Some(sum_sq), 0);
        enc.set_bytes(2, 4, &hidden_u32 as *const u32 as *const c_void);
        enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));

        let pipe = ctx.rms_norm_apply_bf16.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(buf_moe), 0);
        enc.set_buffer(1, Some(&gpu_wf.buf), norm_off);
        enc.set_buffer(2, Some(sum_sq), 0);
        enc.set_buffer(3, Some(ctx.buf_input.as_ref().unwrap()), 0);
        {
            enc.set_bytes(4, 4, &hidden_u32 as *const u32 as *const c_void);
            let eps = RMS_NORM_EPS;
            enc.set_bytes(5, 4, &eps as *const f32 as *const c_void);
        }
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }
}

// ─── gpu_lm_head (local copy) ────────────────────────────────────────────

fn gpu_lm_head(
    wf: &WeightFile, hidden: &[f32], logits: &mut [f32],
    gpu_wf: &WeightBuffer, ctx: &MetalContext,
) {
    let x_buf = metal_buf_shared(&ctx.device, hidden.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(hidden.as_ptr(), x_buf.contents() as *mut f32, hidden.len());
    }
    let out_buf = metal_buf_shared(&ctx.device, logits.len() * 4);
    let cm = ctx.queue.new_command_buffer();
    let enc = cm.new_compute_command_encoder();
    gpu_wf.encode_matvec_into(wf, ctx, &enc, "lm_head", &x_buf, 0, &out_buf, 0, logits.len(), hidden.len());
    enc.end_encoding();
    cm.commit();
    cm.wait_until_completed();
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents() as *const f32, logits.as_mut_ptr(), logits.len());
    }
}

// ─── FusedExp ───────────────────────────────────────────────────────

pub struct FusedExp<'a> {
    pub model: &'a Model,
    pub ctx: &'a MetalContext,
    pub gpu_wf: &'a WeightBuffer,
    pub expert_gpu_buffer: Option<&'a mut ExpertBuffer>,
    pub k: usize,
    pub timing: BTreeMap<String, TelemetryValue>,
}

impl<'a> FusedExp<'a> {
    pub fn new(
        model: &'a Model,
        ctx: &'a MetalContext,
        gpu_wf: &'a WeightBuffer,
        expert_gpu_buffer: Option<&'a mut ExpertBuffer>,
        k: usize,
    ) -> Result<Self, MoEError> {
        let c = &model.config;
        validate_config(
            c.hidden_dim, c.num_layers, c.num_experts,
            c.num_experts_per_tok, c.moe_intermediate,
            c.shared_intermediate, c.num_attn_heads,
            c.num_kv_heads, c.head_dim, c.vocab_size,
            c.linear_num_v_heads, c.linear_num_k_heads,
            c.linear_total_key, c.linear_total_value,
        ).map_err(MoEError::Config)?;
        Ok(FusedExp {
            model, ctx, gpu_wf, expert_gpu_buffer, k,
            timing: BTreeMap::new(),
        })
    }
}

impl<'a> Engine for FusedExp<'a> {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError> {
        assert!(self.k <= NUM_EXPERTS_PER_TOK,
            "k ({}) must not exceed model's num_experts_per_tok ({})",
            self.k, NUM_EXPERTS_PER_TOK);

        let t0 = Instant::now();
        let n = input_ids.len();
        let hd = HIDDEN_DIM;
        let vs = VOCAB_SIZE;
        let num_layers = NUM_LAYERS;

        let mut logits = vec![0.0f32; n * vs];
        if n == 0 {
            return Ok(logits);
        }

        let mut exec = ExecCtx {
            model: self.model,
            cache,
            ctx: self.ctx,
            gpu_wf: self.gpu_wf,
            expert_gpu_buffer: self.expert_gpu_buffer.as_deref_mut(),
            k: self.k,
            timing: BTreeMap::new(),
        };

        let mut embed = vec![0.0f32; n * hd];
        for (i, &id) in input_ids.iter().enumerate() {
            exec.embed(id as usize, &mut embed[i * hd..(i + 1) * hd]);
        }

        let mut hidden = vec![0.0f32; hd];
        for (ti, _) in input_ids.iter().enumerate() {
            hidden.copy_from_slice(&embed[ti * hd..(ti + 1) * hd]);

            let mut pending: Option<DeferredExperts> = None;
            let mut layer = 0;
            let pos = exec.cache.pos;
            while layer < num_layers {
                if check_signal() {
                    return Err(MoEError::Metal("interrupted".into()));
                }

                let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;
                if is_full {
                    if let Some(ref mut def) = pending.take() {
                        def.complete(&mut hidden, hd);
                    }
                    let t0 = Instant::now();
                    pending = exec.full_attention_layer(layer, &mut hidden, pos);
                    timing_push(&mut exec.timing, "engine.full_attention_layer", t0.elapsed().as_secs_f64() * 1000.0);
                    layer += 1;
                } else {
                    if let Some(ref mut def) = pending.take() {
                        def.complete(&mut hidden, hd);
                    }
                    let group_start = layer;
                    while layer < num_layers && (layer + 1) % FULL_ATTN_INTERVAL != 0 {
                        layer += 1;
                    }
                    let t0 = Instant::now();
                    exec.linear_group(group_start, layer - group_start, &mut hidden);
                    timing_push(&mut exec.timing, "engine.linear_group", t0.elapsed().as_secs_f64() * 1000.0);
                }
            }

            if let Some(ref mut def) = pending.take() {
                def.complete(&mut hidden, hd);
            }

            exec.cache.pos = pos + 1;
            exec.final_norm_and_lm_head(&mut hidden, &mut logits[ti * vs..(ti + 1) * vs]);
        }

        timing_add(&mut exec.timing, "engine.total_ms", t0.elapsed().as_secs_f64() * 1000.0);
        self.timing = exec.timing.clone();
        Ok(logits)
    }

    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        self.timing.clone()
    }
}

/// Qwen3.6-35B-A3B-4bit FusedExp engine — all model dimensions are compile-time constants.
use super::constants::ModelConfig;
use crate::constants::{MAX_SEQ, RMS_NORM_EPS, FULL_ATTN_INTERVAL, GROUP_SIZE, CONV_KERNEL_SIZE};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::ffi::c_void;
use std::time::Instant;

use metal::{Buffer, CommandBuffer, ComputeCommandEncoderRef, MTLSize};

use crate::metal_kernels;
use crate::metal_context::{metal_buf_shared, WeightBuffer, MetalContext, ExpertBuffer, MAX_K};
use crate::cache::{Cache, LinearAttnState};
use crate::engine::Engine;
use crate::model::Model;

use crate::error::MoEError;
use crate::engine::{SignalCheckFn, TelemetryValue};
use crate::model::weights::WeightFile;
use crate::math::{
    bf16_to_f32,
    embed_lookup, final_norm, normalize_weights,
    softmax, topk,
};

// ─── Deferred expert results (local copy) ─────────────────────────────────

struct DeferredExperts {
    cmd_buf: Option<CommandBuffer>,
    _keep_alive: Vec<Buffer>,
}

// ─── Per-layer gate scores and routing ────────────────────────────────────

struct GateScores {
    scores: Vec<f32>,
    shared_gate_score: f32,
}

struct Routing {
    expert_weights: Vec<f32>,
    shared_gate_score: f32,
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


// ─── Execution context ─────────────────────────────────────────────────────

struct ExecCtx<'a, C: ModelConfig> {
    engine: FusedExp<'a, C>,
    cache: &'a mut Cache,
    pending: Option<DeferredExperts>,
}

impl<'a, C: ModelConfig> ExecCtx<'a, C> {
    // ── Initialise GPU hidden buffers from embedding ────────────────────────

    fn init_hidden(self, hidden: &[f32]) -> Self {
        let hd = C::HIDDEN_DIM;
        {
            let buf_moe = self.engine.ctx.buf_moe_hidden.as_ref().unwrap();
            unsafe { std::ptr::copy_nonoverlapping(hidden.as_ptr(), buf_moe.contents() as *mut f32, hd); }
        }
        // Pre-compute input_norm for the first layer into buf_input
        {
            let buf_in = self.engine.ctx.buf_input.as_ref().unwrap();
            let norm_name = format!("model.layers.{}.input_layernorm.weight", 0);
            if let Some(nw_u16) = self.engine.model.wf.get_tensor_u16(&norm_name) {
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
        self
    }

    // ── op1: pre-expert (attention + gate projections) ─────────────────────

    fn op1_wait(self, layer: usize) -> (Self, GateScores) {
        let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;
        if is_full {
            self.op1_wait_full(layer)
        } else {
            self.op1_wait_linear(layer)
        }
    }

    /// Full-attn op1: entirely on GPU — input_norm → QKV → Q/K norms → RoPE → KV-cache → attention → gate.
    fn op1_wait_full(mut self, layer: usize) -> (Self, GateScores) {
        let wf = &self.engine.model.wf;
        let gw = self.engine.gpu_wf;
        let c = self.engine.ctx;

        let hd = C::HIDDEN_DIM;
        let num_q = C::NUM_ATTN_HEADS;
        let num_kv = C::NUM_KV_HEADS;
        let head_dim = C::HEAD_DIM;
        let q_dim = num_q * head_dim;
        let q_proj_dim = q_dim * 2;
        let kv_dim = num_kv * head_dim;
        let pos = self.cache.pos;
        let fa_idx = layer / FULL_ATTN_INTERVAL;
        let cache_pos = self.cache.kv[layer].as_ref().unwrap().len;
        let seq_len = cache_pos + 1;

        // Reuse pending op2 buffer from previous layer, or create new
        let mut pending = self.pending.take();
        let keep_alive = pending.as_mut()
            .map(|d| std::mem::take(&mut d._keep_alive))
            .unwrap_or_default();
        let cmd = pending
            .and_then(|d| d.cmd_buf)
            .unwrap_or_else(|| c.queue.new_command_buffer().to_owned());

        let prefix = format!("model.layers.{}.self_attn", layer);

        // Buffer bindings
        let buf_moe = c.buf_moe_hidden.as_ref().unwrap();
        let qkv_x = c.buf_qkv_x.as_ref().unwrap();
        let sum_sq = c.buf_post_sum_sq.as_ref().unwrap();
        let qbuf = c.buf_qkv_q.as_ref().unwrap();
        let kbuf = c.buf_qkv_k.as_ref().unwrap();
        let vbuf = c.buf_qkv_v.as_ref().unwrap();
        let q_out_buf = c.buf_attn_q.as_ref().unwrap();
        let q_gate_buf = c.buf_attn_q_gate.as_ref().unwrap();
        let scores_buf = c.buf_attn_scores.as_ref().unwrap();
        let out_buf = c.buf_attn_out.as_ref().unwrap();
        let kc_buf = &c.buf_kv_k[fa_idx];
        let vc_buf = &c.buf_kv_v[fa_idx];
        let o_proj_buf = c.buf_out_proj.as_ref().unwrap();
        let temp_buf = c.buf_temp_residual.as_ref().unwrap();
        let normed_buf = c.buf_post_normed.as_ref().unwrap();
        let gate_buf = c.buf_gate_scores.as_ref().unwrap();
        let sg_buf = c.buf_shared_gate.as_ref().unwrap();
        let su_buf = c.buf_shared_up.as_ref().unwrap();
        let sge_buf = c.buf_shared_gate_score.as_ref().unwrap();

        {
            let enc = cmd.new_compute_command_encoder();

            // ── input_norm(buf_moe) → qkv_x ──
            let pnw_ptr = wf.get_tensor_ptr(
                &format!("model.layers.{}.input_layernorm.weight", layer)).unwrap();
            let pnw_off = (pnw_ptr as usize - gw.base as usize) as u64;
            enc.set_compute_pipeline_state(&c.rms_norm_sum);
            enc.set_buffer(0, Some(buf_moe), 0);
            enc.set_buffer(1, Some(sum_sq), 0);
            enc.set_bytes(2, 4, &(hd as u32) as *const u32 as *const c_void);
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            {
                let pipe = c.rms_norm_apply_bf16.as_ref().unwrap();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(buf_moe), 0);
                enc.set_buffer(1, Some(&gw.buf), pnw_off);
                enc.set_buffer(2, Some(sum_sq), 0);
                enc.set_buffer(3, Some(qkv_x), 0);
                enc.set_bytes(4, 4, &(hd as u32) as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(((hd + 255) / 256) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }

            // ── QKV projections ──
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.q_proj", prefix), qkv_x, 0, qbuf, 0, q_proj_dim, hd);
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.k_proj", prefix), qkv_x, 0, kbuf, 0, kv_dim, hd);
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.v_proj", prefix), qkv_x, 0, vbuf, 0, kv_dim, hd);

            // ── Q head norm + RoPE (split Q/Q-gate, norm, rotate) ──
            {
                let qn_ptr = wf.get_tensor_ptr(
                    &format!("{}.q_norm.weight", prefix)).expect("q_norm.weight missing");
                let qn_off = (qn_ptr as usize - gw.base as usize) as u64;
                let pipe = c.q_head_norm_rope.as_ref().unwrap();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(qbuf), 0);
                enc.set_buffer(1, Some(&gw.buf), qn_off);
                enc.set_buffer(2, Some(q_out_buf), 0);
                enc.set_buffer(3, Some(q_gate_buf), 0);
                enc.set_bytes(4, 4, &(head_dim as u32) as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &(C::ROTARY_DIM as u32) as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &(C::ROPE_THETA as f32) as *const f32 as *const c_void);
                enc.set_bytes(7, 4, &(pos as u32) as *const u32 as *const c_void);
                enc.set_bytes(8, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_q as u64, 1, 1),
                    MTLSize::new(head_dim as u64, 1, 1),
                );
            }

            // ── K head norm + RoPE (in-place on kbuf) ──
            {
                let kn_ptr = wf.get_tensor_ptr(
                    &format!("{}.k_norm.weight", prefix)).expect("k_norm.weight missing");
                let kn_off = (kn_ptr as usize - gw.base as usize) as u64;
                let pipe = c.k_head_norm_rope.as_ref().unwrap();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(kbuf), 0);
                enc.set_buffer(1, Some(&gw.buf), kn_off);
                enc.set_bytes(2, 4, &(head_dim as u32) as *const u32 as *const c_void);
                enc.set_bytes(3, 4, &(C::ROTARY_DIM as u32) as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &(C::ROPE_THETA as f32) as *const f32 as *const c_void);
                enc.set_bytes(5, 4, &(pos as u32) as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_kv as u64, 1, 1),
                    MTLSize::new(head_dim as u64, 1, 1),
                );
            }

            // ── KV-cache append ──
            {
                let pipe = c.kv_cache_append.as_ref().unwrap();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(kbuf), 0);
                enc.set_buffer(1, Some(vbuf), 0);
                enc.set_buffer(2, Some(kc_buf), 0);
                enc.set_buffer(3, Some(vc_buf), 0);
                enc.set_bytes(4, 4, &(cache_pos as u32) as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &(kv_dim as u32) as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(((kv_dim + 255) / 256) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }

            // ── attn_scores_batched ──
            {
                let pipe = c.attn_scores_batched.as_ref().unwrap();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(q_out_buf), 0);
                enc.set_buffer(1, Some(kc_buf), 0);
                enc.set_buffer(2, Some(scores_buf), 0);
                enc.set_bytes(3, 4, &(head_dim as u32) as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &(kv_dim as u32) as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &(seq_len as u32) as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &(MAX_SEQ as u32) as *const u32 as *const c_void);
                let scale: f32 = 1.0 / (head_dim as f32).sqrt();
                enc.set_bytes(7, 4, &scale as *const f32 as *const c_void);
                enc.set_bytes(8, 4, &((num_q / num_kv) as u32) as *const u32 as *const c_void);
                enc.set_bytes(9, 4, &(seq_len as u32) as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new((num_q as u32 * seq_len as u32) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            // ── attn_softmax_batched ──
            {
                let pipe = c.attn_softmax_batched.as_ref().unwrap();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(scores_buf), 0);
                enc.set_bytes(1, 4, &(seq_len as u32) as *const u32 as *const c_void);
                enc.set_bytes(2, 4, &(MAX_SEQ as u32) as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_q as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            // ── attn_values_batched ──
            {
                let pipe = c.attn_values_batched.as_ref().unwrap();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(scores_buf), 0);
                enc.set_buffer(1, Some(vc_buf), 0);
                enc.set_buffer(2, Some(out_buf), 0);
                enc.set_bytes(3, 4, &(head_dim as u32) as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &(kv_dim as u32) as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &(seq_len as u32) as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &(MAX_SEQ as u32) as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &((num_q / num_kv) as u32) as *const u32 as *const c_void);
                let total_threads = num_q as u32 * head_dim as u32;
                enc.dispatch_thread_groups(
                    MTLSize::new(((total_threads + 255) / 256) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            // ── sigmoid_gate ──
            {
                let pipe = c.sigmoid_gate.as_ref().unwrap();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(out_buf), 0);
                enc.set_buffer(1, Some(q_gate_buf), 0);
                enc.set_bytes(2, 4, &(q_dim as u32) as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(((q_dim as u32 + 255) / 256) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            // ── o_proj matvec ──
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.o_proj", prefix), out_buf, 0, o_proj_buf, 0, hd, q_dim);
            // ── residual_add: o_proj + buf_moe → temp_buf ──
            {
                let pipe = c.residual_add.as_ref().unwrap();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(o_proj_buf), 0);
                enc.set_buffer(1, Some(buf_moe), 0);
                enc.set_buffer(2, Some(temp_buf), 0);
                enc.set_bytes(3, 4, &(hd as u32) as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(((hd + 255) / 256) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            // ── post_attention_layernorm ──
            let post_norm_name = format!("model.layers.{}.post_attention_layernorm.weight", layer);
            {
                enc.set_compute_pipeline_state(&c.rms_norm_sum);
                enc.set_buffer(0, Some(temp_buf), 0);
                enc.set_buffer(1, Some(sum_sq), 0);
                enc.set_bytes(2, 4, &(hd as u32) as *const u32 as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
            }
            {
                let pnw_ptr = wf.get_tensor_ptr(&post_norm_name).unwrap();
                let pnw_off = (pnw_ptr as usize - gw.base as usize) as u64;
                let pipe = c.rms_norm_apply_bf16.as_ref().unwrap();
                enc.set_compute_pipeline_state(pipe);
                enc.set_buffer(0, Some(temp_buf), 0);
                enc.set_buffer(1, Some(&gw.buf), pnw_off);
                enc.set_buffer(2, Some(sum_sq), 0);
                enc.set_buffer(3, Some(normed_buf), 0);
                enc.set_bytes(4, 4, &(hd as u32) as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(((hd + 255) / 256) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            // ── Gate + shared expert projections ──
            let mlp_prefix = format!("model.layers.{}.mlp", layer);
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.gate", mlp_prefix), normed_buf, 0, gate_buf, 0, C::NUM_EXPERTS, hd);
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.gate_proj", mlp_prefix), normed_buf, 0, sg_buf, 0, C::SHARED_INTERMEDIATE, hd);
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.up_proj", mlp_prefix), normed_buf, 0, su_buf, 0, C::SHARED_INTERMEDIATE, hd);
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert_gate", mlp_prefix), normed_buf, 0, sge_buf, 0, 1, hd);

            enc.end_encoding();
        }

        cmd.commit();
        cmd.wait_until_completed();
        drop(keep_alive);

        // Update CPU-side KV-cache position tracking
        {
            let kv_cache = self.cache.kv[layer].as_mut().unwrap();
            assert!(cache_pos < MAX_SEQ, "sequence length {} exceeds MAX_SEQ ({})", cache_pos, MAX_SEQ);
            kv_cache.len += 1;
        }

        // Read back gate scores
        let mut gate_scores = vec![0.0f32; C::NUM_EXPERTS];
        let shared_gate_score: f32;
        unsafe {
            std::ptr::copy_nonoverlapping(gate_buf.contents() as *const f32, gate_scores.as_mut_ptr(), C::NUM_EXPERTS);
            shared_gate_score = *(sge_buf.contents() as *const f32);
        }

        self.pending = None;
        (self, GateScores { scores: gate_scores, shared_gate_score })
    }

    /// Linear-attn op1: uses the pending op2 buffer (or creates a new one)
    /// to form a fused post_expert(prev) + pre_expert(curr) command when pipelining.
    fn op1_wait_linear(mut self, layer: usize) -> (Self, GateScores) {
        let hd = C::HIDDEN_DIM;
        let num_experts = C::NUM_EXPERTS;
        let shared_inter = C::SHARED_INTERMEDIATE;
        let qkv_dim = C::LINEAR_CONV_DIM;
        let total_key = C::LINEAR_TOTAL_KEY;
        let total_val = C::LINEAR_TOTAL_VALUE;
        let num_k_heads = C::LINEAR_NUM_K_HEADS;
        let num_v_heads = C::LINEAR_NUM_V_HEADS;
        let key_dim = total_key / num_k_heads;
        let val_dim = total_val / num_v_heads;
        let inv_scale = 1.0 / (key_dim as f32).sqrt();
        let k_heads_per_v = num_v_heads / num_k_heads;
        let li = layer - (layer + 1) / FULL_ATTN_INTERVAL;

        // Take the pending op2 buffer from the previous layer, or create fresh
        let mut pending_def = self.pending.take();
        let cmd = if let Some(def) = pending_def.take() {
            def.cmd_buf.unwrap_or_else(|| self.engine.ctx.queue.new_command_buffer().to_owned())
        } else {
            self.engine.ctx.queue.new_command_buffer().to_owned()
        };
        let keep_alive = pending_def.map(|d| d._keep_alive).unwrap_or_default();

        {
            let enc = cmd.new_compute_command_encoder();
            encode_pre_expert(
                &self.engine.model.wf, self.engine.gpu_wf, self.engine.ctx, &enc, layer, li,
                hd, num_k_heads, num_v_heads, total_key, total_val, qkv_dim,
                key_dim, val_dim, k_heads_per_v, inv_scale, num_experts, shared_inter,
            );
            enc.end_encoding();
        }
        cmd.commit();
        cmd.wait_until_completed();
        drop(keep_alive);

        // Update conv state shadow
        if let Some(ref mut s) = self.cache.lin[layer] {
            update_conv_state(s, qkv_dim);
        }

        // Read back gate scores
        let mut gate_scores = vec![0.0f32; num_experts];
        let shared_gate_score: f32;
        {
            let c = self.engine.ctx;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    c.buf_gate_scores.as_ref().unwrap().contents() as *const f32,
                    gate_scores.as_mut_ptr(), num_experts);
                shared_gate_score = *(c.buf_shared_gate_score.as_ref().unwrap().contents() as *const f32);
            }
        }

        self.pending = None;
        (self, GateScores { scores: gate_scores, shared_gate_score })
    }

    // ── Routing ───────────────────────────────────────────────────────────

    fn route_experts(&mut self, layer: usize, mut gate_scores: GateScores) -> Routing {
        let k = self.engine.k;
        softmax(&mut gate_scores.scores);
        let mut expert_indices = vec![0usize; k];
        let mut expert_weights = vec![0.0f32; k];
        topk(&gate_scores.scores, k, &mut expert_indices, &mut expert_weights);
        normalize_weights(&mut expert_weights);

        let actual_k = k.min(MAX_K);
        let io = self.engine.expert_gpu_buffer.as_mut().unwrap();
        let mut miss_ei = [0usize; MAX_K];
        let mut miss_k_slot = [0usize; MAX_K];
        let mut miss_count = 0;
        for ki in 0..actual_k {
            let eidx = expert_indices[ki];
            if let Some(buf) = io.cache.lookup(layer, eidx) {
                io.expert_data[ki] = buf;
            } else {
                miss_ei[miss_count] = eidx;
                miss_k_slot[miss_count] = ki;
                miss_count += 1;
            }
        }
        for m in 0..miss_count {
            let ki = miss_k_slot[m];
            let eidx = miss_ei[m];
            let buf = io.cache.insert_get_buf(layer, eidx);
            io.expert_data[ki] = buf;
        }
        if miss_count > 0 {
            let t_io = Instant::now();
            let expert_file = &self.engine.model.expert_files[layer];
            let expert_size = expert_file.expert_size();
            let mut reads: [(usize, usize); MAX_K] = [(0, 0); MAX_K];
            for m in 0..miss_count {
                let ki = miss_k_slot[m];
                reads[m] = (miss_ei[m], io.expert_data[ki].contents() as usize);
            }
            rayon::scope(|s| {
                for m in 0..miss_count {
                    let (eidx, ptr_u) = reads[m];
                    let dst = unsafe { std::slice::from_raw_parts_mut(ptr_u as *mut u8, expert_size) };
                    s.spawn(move |_| {
                        expert_file.read_expert(eidx, dst).unwrap();
                    });
                }
            });
            let dt = t_io.elapsed().as_secs_f64() * 1000.0;
            timing_add(&mut self.engine.timing, "engine.expert_io_ms", dt);
        }

        Routing {
            expert_weights,
            shared_gate_score: gate_scores.shared_gate_score,
        }
    }

    // ── op2: post-expert (encodes into pending buffer, does NOT commit) ────

    fn op2(mut self, layer: usize, routing: &Routing) -> Self {
        let hd = C::HIDDEN_DIM;
        let moe_inter = C::MOE_INTERMEDIATE;
        let shared_inter = C::SHARED_INTERMEDIATE;
        let k = self.engine.k;

        let next_norm_info = if layer + 1 < C::NUM_LAYERS {
            self.engine.model.wf.get_tensor_ptr(
                &format!("model.layers.{}.input_layernorm.weight", layer + 1))
                .map(|p| (p as *const c_void, self.engine.gpu_wf.base as usize))
        } else {
            None
        };

        let cmd = self.engine.ctx.queue.new_command_buffer().to_owned();
        let mut keep_alive = Vec::with_capacity(4);

        {
            let enc = cmd.new_compute_command_encoder();
            let io = self.engine.expert_gpu_buffer.as_ref().unwrap();

            // Keep expert out buffers alive
            let actual_k = k.min(MAX_K);
            for ki in 0..actual_k {
                keep_alive.push(io.expert_out[ki].clone());
            }
            keep_alive.push(io.shared_act.clone());
            keep_alive.push(io.shared_down.clone());

            encode_post_expert::<C>(
                &self.engine.model.wf, self.engine.gpu_wf, self.engine.ctx, &enc, layer,
                &routing.expert_weights, routing.shared_gate_score,
                &io.expert_data, &io.scratch_gate, &io.scratch_up, &io.scratch_act,
                &io.expert_out, &io.shared_act, &io.shared_down, &io.combine_params,
                next_norm_info,
                hd, moe_inter, shared_inter, k,
            );
            enc.end_encoding();
        }

        self.pending = Some(DeferredExperts {
            cmd_buf: Some(cmd),
            _keep_alive: keep_alive,
        });
        self
    }

    // ── Commit final pending work & read hidden from GPU ──────────────────

    fn hidden_wait(&mut self) -> Vec<f32> {
        let hd = C::HIDDEN_DIM;
        let pending = self.pending.take();
        if let Some(ref def) = pending {
            if let Some(ref cmd) = def.cmd_buf {
                cmd.commit();
                cmd.wait_until_completed();
            }
        }
        drop(pending);
        let mut hidden = vec![0.0f32; hd];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.engine.ctx.buf_moe_hidden.as_ref().unwrap().contents() as *const f32,
                hidden.as_mut_ptr(), hd);
        }
        hidden
    }

    // ── Final norm + LM head ───────────────────────────────────────────────

    fn final_norm_and_lm_head(&self, hidden: &mut [f32], logits: &mut [f32]) {
        final_norm(&self.engine.model.wf, hidden, C::HIDDEN_DIM);
        gpu_lm_head(&self.engine.model.wf, hidden, logits, self.engine.gpu_wf, self.engine.ctx);
    }
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

fn encode_post_expert<C: ModelConfig>(
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
            expert_buf, C::GATE_W_OFF as u64,
            expert_buf, C::GATE_S_OFF as u64,
            expert_buf, C::GATE_B_OFF as u64,
            post_normed, 0, expert_scratch_gate, 0,
            inter_u32, hidden_u32, gs_u32, 3);

        metal_kernels::encode_matvec_offset(ctx, enc,
            expert_buf, C::UP_W_OFF as u64,
            expert_buf, C::UP_S_OFF as u64,
            expert_buf, C::UP_B_OFF as u64,
            post_normed, 0, expert_scratch_up, 0,
            inter_u32, hidden_u32, gs_u32, 3);

        metal_kernels::encode_swiglu(ctx, enc, expert_scratch_gate, 0, expert_scratch_up, 0,
            expert_scratch_act, 0, inter_u32);

        metal_kernels::encode_matvec_offset(ctx, enc,
            expert_buf, C::DOWN_W_OFF as u64,
            expert_buf, C::DOWN_S_OFF as u64,
            expert_buf, C::DOWN_B_OFF as u64,
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

pub struct FusedExp<'a, C: ModelConfig> {
    pub model: &'a Model,
    pub ctx: &'a MetalContext,
    pub gpu_wf: &'a WeightBuffer,
    pub expert_gpu_buffer: Option<&'a mut ExpertBuffer>,
    pub k: usize,
    pub timing: BTreeMap<String, TelemetryValue>,
    _phantom: PhantomData<C>,
}

impl<'a, C: ModelConfig> FusedExp<'a, C> {
    pub fn new(
        model: &'a Model,
        ctx: &'a MetalContext,
        gpu_wf: &'a WeightBuffer,
        expert_gpu_buffer: Option<&'a mut ExpertBuffer>,
        k: usize,
    ) -> Result<Self, MoEError> {
        let c = &model.config;
        let get = |k| c.get_usize(k).unwrap_or(0);
        C::validate_config(
            get("hidden_dim"), get("num_layers"), get("num_experts"),
            get("num_experts_per_tok"), get("moe_intermediate"),
            get("shared_intermediate"), get("num_attn_heads"),
            get("num_kv_heads"), get("head_dim"), get("vocab_size"),
            get("linear_num_v_heads"), get("linear_num_k_heads"),
            get("linear_total_key"), get("linear_total_value"),
        ).map_err(MoEError::Config)?;
        Ok(FusedExp {
            model, ctx, gpu_wf, expert_gpu_buffer, k,
            timing: BTreeMap::new(),
            _phantom: PhantomData,
        })
    }
}

impl<'a, C: ModelConfig> Engine for FusedExp<'a, C> {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError> {
        assert!(self.k <= C::NUM_EXPERTS_PER_TOK,
            "k ({}) must not exceed model's num_experts_per_tok ({})",
            self.k, C::NUM_EXPERTS_PER_TOK);

        let t0 = Instant::now();
        let n = input_ids.len();
        let hd = C::HIDDEN_DIM;
        let vs = C::VOCAB_SIZE;
        let num_layers = C::NUM_LAYERS;

        let mut logits = vec![0.0f32; n * vs];
        if n == 0 {
            return Ok(logits);
        }

        let engine: FusedExp<'_, C> = FusedExp {
            model: self.model,
            ctx: self.ctx,
            gpu_wf: self.gpu_wf,
            expert_gpu_buffer: self.expert_gpu_buffer.take(),
            k: self.k,
            timing: std::mem::take(&mut self.timing),
            _phantom: PhantomData,
        };
        let mut exec = ExecCtx { engine, cache, pending: None };

        let mut embed = vec![0.0f32; n * hd];
        for (i, &id) in input_ids.iter().enumerate() {
            embed_lookup(&exec.engine.model.wf, id as usize, &mut embed[i * hd..(i + 1) * hd], C::HIDDEN_DIM);
        }

        for (ti, _) in input_ids.iter().enumerate() {
            let embed_hidden = &embed[ti * hd..(ti + 1) * hd];
            exec = exec.init_hidden(embed_hidden);

            for layer in 0..num_layers {
                if check_signal() {
                    return Err(MoEError::Metal("interrupted".into()));
                }
                let (new_ctx, gate_scores) = exec.op1_wait(layer);
                exec = new_ctx;
                let routing = exec.route_experts(layer, gate_scores);
                exec = exec.op2(layer, &routing);
            }

            let mut hidden = exec.hidden_wait();
            exec.cache.pos += 1;
            exec.final_norm_and_lm_head(&mut hidden, &mut logits[ti * vs..(ti + 1) * vs]);
        }

        timing_add(&mut exec.engine.timing, "engine.total_ms", t0.elapsed().as_secs_f64() * 1000.0);
        self.timing = exec.engine.timing.clone();
        Ok(logits)
    }

    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        self.timing.clone()
    }
}

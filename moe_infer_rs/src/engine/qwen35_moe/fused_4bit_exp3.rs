/// Qwen3.6-35B-A3B-4bit Fused4bitExp3 engine — speculative expert prefetch.
///
/// exp3 extends exp2's 2-pass SDPA with speculative expert prefetch:
///   route_experts computes top-2K instead of top-K, sync-reads the first K,
///   and stores spare K..2K cache-miss info.  On the next layer's op1 commit,
///   the spare experts are read asynchronously while the GPU runs op1,
///   overlapping SSD I/O with GPU compute.
///
/// Only one commit per layer (op1 bundles the previous layer's deferred op2).
use super::constants::ModelConfig;
use crate::constants::{MAX_SEQ, RMS_NORM_EPS, FULL_ATTN_INTERVAL, GROUP_SIZE};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::marker::PhantomData;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use metal::{Buffer, CommandBuffer, ComputeCommandEncoderRef, MTLSize};

use crate::engine::qwen35_moe::metal_kernels;
use crate::engine::qwen35_moe::metal_context::{metal_buf_shared, WeightBuffer, MetalContext, ExpertBuffer, MAX_K};
use crate::cache::Cache;
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

// ─── Per-layer gate scores and routing ────────────────────────────────────

struct GateScores {
    scores: Vec<f32>,
    shared_gate_score: f32,
}

struct Routing {
    expert_weights: Vec<f32>,
    shared_gate_score: f32,
}

/// Spare expert indices and cache buffers that need SSD reads.
/// These are fire-and-forget: a background thread reads them while the GPU runs.
struct SpareInfo {
    miss_indices: Vec<usize>,
    miss_buffers: Vec<Buffer>,
}

// ─── Deferred expert results ─────────────────────────────────────────────

struct DeferredExperts {
    cmd_buf: Option<CommandBuffer>,
    _keep_alive: Vec<Buffer>,
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

struct ExecCtx<'b, C: ModelConfig> {
    engine: &'b mut Fused4bitExp3<C>,
    /// Deferred op2 from the previous layer — committed with this layer's op1.
    pending: Option<DeferredExperts>,
    /// Spare expert info from the previous layer — loaded while GPU runs op1.
    pending_spare: Option<SpareInfo>,
    /// Active spare-read thread with cancel flag (at most one at a time).
    spare_thread: Option<(Arc<AtomicBool>, std::thread::JoinHandle<()>)>,
}

impl<'b, C: ModelConfig> ExecCtx<'b, C> {
    // ── Initialise GPU hidden buffers from embedding ────────────────────────

    fn init_hidden(&mut self, hidden: &[f32]) {
        let hidden_dim = C::HIDDEN_DIM;
        {
            let buf_moe = self.engine.ctx.buf_moe_hidden.as_ref().unwrap();
            unsafe { std::ptr::copy_nonoverlapping(hidden.as_ptr(), buf_moe.contents() as *mut f32, hidden_dim); }
        }
        // Pre-compute input_norm for the first layer into buf_input
        {
            let buf_in = self.engine.ctx.buf_input.as_ref().unwrap();
            let norm_name = format!("model.layers.{}.input_layernorm.weight", 0);
            if let Some(nw_u16) = self.engine.model.weight_file.get_tensor_u16(&norm_name) {
                let nw: Vec<f32> = nw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
                let ssq: f32 = hidden[..hidden_dim].iter().map(|v| v * v).sum();
                let inv_rms = 1.0 / (ssq / hidden_dim as f32 + RMS_NORM_EPS).sqrt();
                unsafe {
                    let dst = buf_in.contents() as *mut f32;
                    for i in 0..hidden_dim { *dst.add(i) = hidden[i] * inv_rms * nw[i]; }
                }
            } else {
                unsafe { std::ptr::copy_nonoverlapping(hidden.as_ptr(), buf_in.contents() as *mut f32, hidden_dim); }
            }
        }
    }

    // ── op1: pre-expert (attention + gate projections) ─────────────────────
    //
    /// Takes the previous layer's deferred op2 and this layer's op1, bundles them
    /// into one command buffer, commits (single commit per layer), then spawns
    /// background spare-expert reads while the GPU runs.  After the GPU completes,
    /// returns the gate scores for routing.

    fn op1_wait(&mut self, layer: usize, pos: usize) -> GateScores {
        // Take previous layer's deferred op2 (if any) and spare info
        let (cmd, keep_alive) = match self.pending.take() {
            Some(def) => {
                let cmd = def.cmd_buf.unwrap_or_else(|| self.engine.ctx.queue.new_command_buffer().to_owned());
                (cmd, def._keep_alive)
            }
            None => {
                (self.engine.ctx.queue.new_command_buffer().to_owned(), Vec::new())
            }
        };
        let prev_spare = self.pending_spare.take();

        let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;
        if is_full {
            self.op1_full(layer, pos, &cmd);
        } else {
            self.op1_linear(layer, &cmd);
        }

        // Single commit: GPU starts running (op2_{L-1} + op1_L)
        cmd.commit();

        // While GPU runs: load spare experts from the previous layer
        if let Some(spare_info) = prev_spare {
            if layer > 0 {
                self.spawn_spare_reads(layer - 1, &spare_info);
            }
        }

        // Wait for GPU to finish, then cancel any still-running spare reads
        cmd.wait_until_completed();
        self.stop_spare_reads();
        drop(keep_alive);
        self.pending = None;

        let c = &self.engine.ctx;
        let mut gate_scores = vec![0.0f32; C::NUM_EXPERTS];
        let shared_gate_score: f32;
        unsafe {
            std::ptr::copy_nonoverlapping(
                c.buf_gate_scores.as_ref().unwrap().contents() as *const f32,
                gate_scores.as_mut_ptr(), C::NUM_EXPERTS);
            shared_gate_score = *(c.buf_shared_gate_score.as_ref().unwrap().contents() as *const f32);
        }
        GateScores { scores: gate_scores, shared_gate_score }
    }

    /// Full-attn op1: encode pre-expert GPU work into the supplied command buffer.
    fn op1_full(&mut self, layer: usize, pos: usize, cmd: &CommandBuffer) {
        let fa_idx = layer / FULL_ATTN_INTERVAL;
        {
            let enc = cmd.new_compute_command_encoder();
            encode_pre_expert_full(
                &self.engine.model.weight_file, &self.engine.weight_buffer, &self.engine.ctx, &enc,
                layer, fa_idx, pos,
                C::HIDDEN_DIM, C::NUM_ATTN_HEADS, C::NUM_KV_HEADS, C::HEAD_DIM,
                C::ROTARY_DIM, C::ROPE_THETA as f32, C::NUM_EXPERTS, C::SHARED_INTERMEDIATE,
            );
            enc.end_encoding();
        }
        assert!(pos < MAX_SEQ, "sequence position {} exceeds MAX_SEQ ({})", pos, MAX_SEQ);
    }

    /// Linear-attn op1: encode pre-expert GPU work into the supplied command buffer.
    fn op1_linear(&mut self, layer: usize, cmd: &CommandBuffer) {
        let hidden_dim = C::HIDDEN_DIM;
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
        let linear_idx = layer - (layer + 1) / FULL_ATTN_INTERVAL;

        {
            let enc = cmd.new_compute_command_encoder();
            encode_pre_expert_linear(
                &self.engine.model.weight_file, &self.engine.weight_buffer, &self.engine.ctx, &enc, layer, linear_idx,
                hidden_dim, num_k_heads, num_v_heads, total_key, total_val, qkv_dim,
                key_dim, val_dim, k_heads_per_v, inv_scale, num_experts, shared_inter,
            );
            enc.end_encoding();
        }
    }

    // ── Routing (top-2K with speculative prefetch) ─────────────────────────

    fn route_experts(&mut self, layer: usize, mut gate_scores: GateScores) -> (Routing, SpareInfo) {
        let k = self.engine.k;
        let k2 = (k * 2).min(C::NUM_EXPERTS);

        softmax(&mut gate_scores.scores);

        // Compute top-2K expert indices and weights
        let mut expert_indices_2k = vec![0usize; k2];
        let mut expert_weights_2k = vec![0.0f32; k2];
        topk(&gate_scores.scores, k2, &mut expert_indices_2k, &mut expert_weights_2k);

        // Primary weights: [0..k]
        let mut expert_weights = vec![0.0f32; k];
        expert_weights.copy_from_slice(&expert_weights_2k[..k]);
        normalize_weights(&mut expert_weights);

        let actual_k = k.min(MAX_K);
        let expert_buf = &mut self.engine.expert_buffer;

        // ── Primary K: cache lookup + sync SSD read ──
        let mut miss_ei = [0usize; MAX_K];
        let mut miss_k_slot = [0usize; MAX_K];
        let mut miss_count = 0;
        for ki in 0..actual_k {
            let eidx = expert_indices_2k[ki];
            if let Some(buf) = expert_buf.cache.lookup(layer, eidx) {
                expert_buf.expert_data[ki] = buf;
            } else {
                miss_ei[miss_count] = eidx;
                miss_k_slot[miss_count] = ki;
                miss_count += 1;
            }
        }
        for m in 0..miss_count {
            let ki = miss_k_slot[m];
            let eidx = miss_ei[m];
            let buf = expert_buf.cache.insert_get_buf(layer, eidx);
            expert_buf.expert_data[ki] = buf;
        }
        if miss_count > 0 {
            let t_io = Instant::now();
            let expert_file = &self.engine.model.expert_files[layer];
            let expert_size = expert_file.expert_size();
            let mut reads: [(usize, usize); MAX_K] = [(0, 0); MAX_K];
            for m in 0..miss_count {
                let ki = miss_k_slot[m];
                reads[m] = (miss_ei[m], expert_buf.expert_data[ki].contents() as usize);
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

        // ── Spare K..2K: cache lookup, collect misses for async prefetch ──
        let spare_start = k;
        let spare_end = k2;
        let mut spare_miss_indices = Vec::new();
        let mut spare_miss_buffers = Vec::new();

        for idx in spare_start..spare_end {
            let eidx = expert_indices_2k[idx];
            if expert_buf.cache.lookup(layer, eidx).is_none() {
                // Cache miss — get a buffer and queue for background read
                let buf = expert_buf.cache.insert_get_buf(layer, eidx);
                spare_miss_indices.push(eidx);
                spare_miss_buffers.push(buf);
            }
            // Cache hit — expert already resident, nothing to do
        }

        let routing = Routing {
            expert_weights,
            shared_gate_score: gate_scores.shared_gate_score,
        };
        let spare_info = SpareInfo {
            miss_indices: spare_miss_indices,
            miss_buffers: spare_miss_buffers,
        };

        (routing, spare_info)
    }

    // ── op2: post-expert (encodes into pending, does NOT commit) ───────────
    //
    /// Encodes post-expert GPU work but does NOT commit — deferred to the
    /// next layer's op1_wait which bundles it with that layer's op1.

    fn op2(&mut self, layer: usize, routing: &Routing) {

        let hidden_dim = C::HIDDEN_DIM;
        let moe_inter = C::MOE_INTERMEDIATE;
        let shared_inter = C::SHARED_INTERMEDIATE;
        let k = self.engine.k;

        let next_norm_info = if layer + 1 < C::NUM_LAYERS {
            self.engine.model.weight_file.get_tensor_ptr(
                &format!("model.layers.{}.input_layernorm.weight", layer + 1))
                .map(|p| (p as *const c_void, self.engine.weight_buffer.base as usize))
        } else {
            None
        };

        let cmd = self.engine.ctx.queue.new_command_buffer().to_owned();
        let mut keep_alive = Vec::with_capacity(4);

        let expert_buf = &self.engine.expert_buffer;

        let actual_k = k.min(MAX_K);
        for ki in 0..actual_k {
            keep_alive.push(expert_buf.expert_out[ki].clone());
        }
        keep_alive.push(expert_buf.shared_act.clone());
        keep_alive.push(expert_buf.shared_down.clone());
        {
            let enc = cmd.new_compute_command_encoder();
            encode_post_expert::<C>(
                &self.engine.model.weight_file, &self.engine.weight_buffer, &self.engine.ctx, &enc, layer,
                &routing.expert_weights, routing.shared_gate_score,
                &expert_buf.expert_data, &expert_buf.scratch_gate, &expert_buf.scratch_up, &expert_buf.scratch_act,
                &expert_buf.expert_out, &expert_buf.shared_act, &expert_buf.shared_down, &expert_buf.combine_params,
                next_norm_info,
                hidden_dim, moe_inter, shared_inter, k,
            );
            enc.end_encoding();
        }

        self.pending = Some(DeferredExperts {
            cmd_buf: Some(cmd),
            _keep_alive: keep_alive,
        });
    }

    // ── Spawn asynchronous reads for spare experts ─────────────────────────
    //
    /// Fire-and-forget background thread that reads spare expert weights into
    /// cache buffers.  These reads overlap with the GPU running op2 for this
    /// layer.  Errors are silently ignored (best-effort prefetch).

    fn spawn_spare_reads(&mut self, layer: usize, spare_info: &SpareInfo) {
        if spare_info.miss_indices.is_empty() {
            return;
        }

        let ef_addr = &self.engine.model.expert_files[layer] as *const _ as usize;
        let expert_size = self.engine.model.expert_files[layer].expert_size();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_flag = cancel.clone();

        let mut reads: Vec<(usize, usize)> = Vec::with_capacity(spare_info.miss_indices.len());
        for i in 0..spare_info.miss_indices.len() {
            reads.push((
                spare_info.miss_indices[i],
                spare_info.miss_buffers[i].contents() as usize,
            ));
        }

        let handle = std::thread::spawn(move || {
            let ef: &crate::model::ExpertFileType = unsafe { &*(ef_addr as *const _) };
            for (eidx, ptr_u) in &reads {
                if cancel_flag.load(Ordering::Relaxed) {
                    break;
                }
                let dst = unsafe { std::slice::from_raw_parts_mut(*ptr_u as *mut u8, expert_size) };
                ef.read_expert(*eidx, dst).ok();
            }
        });
        // Replace any previous spare thread (should already be stopped)
        self.stop_spare_reads();
        self.spare_thread = Some((cancel, handle));
    }

    /// Signal the outstanding spare-read thread to stop, then detach it.
    /// Does NOT block — the thread exits on its next loop iteration.
    fn stop_spare_reads(&mut self) {
        if let Some((cancel, _handle)) = self.spare_thread.take() {
            cancel.store(true, Ordering::Relaxed);
            // handle dropped → thread detached, finishes on its own
        }
    }

    // ── Commit final pending work & read hidden from GPU ──────────────────

    fn hidden_wait(&mut self) -> Vec<f32> {
        let hidden_dim = C::HIDDEN_DIM;
        let pending = self.pending.take();
        if let Some(ref def) = pending {
            if let Some(ref cmd) = def.cmd_buf {
                cmd.commit();
                cmd.wait_until_completed();
            }
        }
        drop(pending);
        let mut hidden = vec![0.0f32; hidden_dim];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.engine.ctx.buf_moe_hidden.as_ref().unwrap().contents() as *const f32,
                hidden.as_mut_ptr(), hidden_dim);
        }
        hidden
    }

    // ── Final norm + LM head ───────────────────────────────────────────────

    fn final_norm_and_lm_head(&self, hidden: &mut [f32], logits: &mut [f32]) {
        final_norm(&self.engine.model.weight_file, hidden, C::HIDDEN_DIM);
        gpu_lm_head(&self.engine.model.weight_file, hidden, logits, &self.engine.weight_buffer, &self.engine.ctx);
    }
}


// ─── Private helpers ──────────────────────────────────────────────────────

fn encode_pre_expert_linear(
    wf: &WeightFile,
    weight_buffer: &WeightBuffer,
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
    let gw = weight_buffer;
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
            let gate_norm_weight_off = (gnw_p as usize - gw.base as usize) as u64;
            metal_kernels::encode_gated_rms_norm(c, enc,
                c.buf_delta_output.as_ref().unwrap(),
                &c.batch_out[1],
                &gw.buf, gate_norm_weight_off,
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
    let post_norm_weight_off = (pnw_ptr as usize - gw.base as usize) as u64;
    let temp_res = c.buf_temp_residual.as_ref().unwrap();
    {
        let pipe = c.rms_norm_fused_bf16.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(temp_res), 0);
        enc.set_buffer(1, Some(&gw.buf), post_norm_weight_off);
        enc.set_buffer(2, Some(c.buf_post_normed.as_ref().unwrap()), 0);
        enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
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

fn encode_pre_expert_full(
    wf: &WeightFile,
    weight_buffer: &WeightBuffer,
    ctx: &MetalContext,
    enc: &ComputeCommandEncoderRef,
    layer: usize,
    fa_idx: usize,
    pos: usize,
    hidden_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    num_experts: usize,
    shared_intermediate: usize,
) {
    let c = ctx;
    let gw = weight_buffer;
    let num_q = num_q_heads;
    let num_kv = num_kv_heads;
    let q_dim = num_q * head_dim;
    let q_proj_dim = q_dim * 2;
    let kv_dim = num_kv * head_dim;
    let seq_len = pos + 1;

    let prefix = format!("model.layers.{}.self_attn", layer);

    let buf_moe = c.buf_moe_hidden.as_ref().unwrap();
    let qkv_x = c.buf_qkv_x.as_ref().unwrap();
    let qbuf = c.buf_qkv_q.as_ref().unwrap();
    let kbuf = c.buf_qkv_k.as_ref().unwrap();
    let vbuf = c.buf_qkv_v.as_ref().unwrap();
    let q_out_buf = c.buf_attn_q.as_ref().unwrap();
    let q_gate_buf = c.buf_attn_q_gate.as_ref().unwrap();
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

    // ── input_norm(buf_moe) → qkv_x ──
    let pnw_ptr = wf.get_tensor_ptr(
        &format!("model.layers.{}.input_layernorm.weight", layer)).unwrap();
    let post_norm_weight_off = (pnw_ptr as usize - gw.base as usize) as u64;
    {
        let pipe = c.rms_norm_fused_bf16.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(buf_moe), 0);
        enc.set_buffer(1, Some(&gw.buf), post_norm_weight_off);
        enc.set_buffer(2, Some(qkv_x), 0);
        enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    // ── QKV projections ──
    gw.encode_matvec_into(wf, c, enc, &format!("{}.q_proj", prefix), qkv_x, 0, qbuf, 0, q_proj_dim, hidden_dim);
    gw.encode_matvec_into(wf, c, enc, &format!("{}.k_proj", prefix), qkv_x, 0, kbuf, 0, kv_dim, hidden_dim);
    gw.encode_matvec_into(wf, c, enc, &format!("{}.v_proj", prefix), qkv_x, 0, vbuf, 0, kv_dim, hidden_dim);

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
        enc.set_bytes(5, 4, &(rotary_dim as u32) as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &rope_theta as *const f32 as *const c_void);
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
        enc.set_bytes(3, 4, &(rotary_dim as u32) as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &rope_theta as *const f32 as *const c_void);
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
        enc.set_bytes(4, 4, &(pos as u32) as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &(kv_dim as u32) as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((kv_dim + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    // ── 2-pass Fused SDPA (block + reduce) ──
    {
        // Block size = 32 (matches SIMD width BD).  Smaller blocks → more TGs,
        // better occupancy on GPUs with many cores.  Larger blocks (64, 128) → fewer
        // TGs, less reduce overhead, better for weaker GPUs or short sequences.
        // Must be a multiple of 32 (SIMD width) for lane-parallel score reduction.
        let num_blocks: u32 = (seq_len as u32 + 31) / 32;  // ceil(seq_len / 32)
        let stride: usize = 2 + head_dim;  // max, sum + output

        // Allocate partials buffer: [num_q * num_blocks * stride] f32
        let partials_len = num_q * num_blocks as usize * stride;
        let partials = c.device.new_buffer(
            (partials_len * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // ── Pass 1: block processing ──
        {
            let pipe = c.attn_sdpa_block.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(q_out_buf), 0);
            enc.set_buffer(1, Some(kc_buf), 0);
            enc.set_buffer(2, Some(vc_buf), 0);
            enc.set_buffer(3, Some(&partials), 0);
            enc.set_bytes(4, 4, &(seq_len as u32) as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &num_blocks as *const u32 as *const c_void);
            let scale: f32 = 1.0 / (head_dim as f32).sqrt();
            enc.set_bytes(6, 4, &scale as *const f32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(num_q as u64, num_blocks as u64, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // ── Pass 2: reduce ──
        {
            let pipe = c.attn_sdpa_reduce.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&partials), 0);
            enc.set_buffer(1, Some(out_buf), 0);
            enc.set_bytes(2, 4, &num_blocks as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(num_q as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }
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
    gw.encode_matvec_into(wf, c, enc, &format!("{}.o_proj", prefix), out_buf, 0, o_proj_buf, 0, hidden_dim, q_dim);
    // ── residual_add: o_proj + buf_moe → temp_buf ──
    {
        let pipe = c.residual_add.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(o_proj_buf), 0);
        enc.set_buffer(1, Some(buf_moe), 0);
        enc.set_buffer(2, Some(temp_buf), 0);
        enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
    // ── post_attention_layernorm ──
    let post_norm_name = format!("model.layers.{}.post_attention_layernorm.weight", layer);
    {
        let pnw_ptr = wf.get_tensor_ptr(&post_norm_name).unwrap();
        let post_norm_weight_off = (pnw_ptr as usize - gw.base as usize) as u64;
        let pipe = c.rms_norm_fused_bf16.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(temp_buf), 0);
        enc.set_buffer(1, Some(&gw.buf), post_norm_weight_off);
        enc.set_buffer(2, Some(normed_buf), 0);
        enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }
    // ── Gate + shared expert projections ──
    let mlp_prefix = format!("model.layers.{}.mlp", layer);
    gw.encode_matvec_into(wf, c, enc, &format!("{}.gate", mlp_prefix), normed_buf, 0, gate_buf, 0, num_experts, hidden_dim);
    gw.encode_matvec_into(wf, c, enc, &format!("{}.shared_expert.gate_proj", mlp_prefix), normed_buf, 0, sg_buf, 0, shared_intermediate, hidden_dim);
    gw.encode_matvec_into(wf, c, enc, &format!("{}.shared_expert.up_proj", mlp_prefix), normed_buf, 0, su_buf, 0, shared_intermediate, hidden_dim);
    gw.encode_matvec_into(wf, c, enc, &format!("{}.shared_expert_gate", mlp_prefix), normed_buf, 0, sge_buf, 0, 1, hidden_dim);
}

fn encode_post_expert<C: ModelConfig>(
    wf: &WeightFile,
    weight_buffer: &WeightBuffer,
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
    if !weight_buffer.encode_matvec_into(wf, ctx, enc, &sd_name, shared_scratch, 0,
        shared_down_buf, 0, hidden_dim, shared_inter)
    {
        eprintln!("[fused_4bit_exp3] WARNING: shared expert down_proj tensor not found: {}", sd_name);
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

        let pipe = ctx.rms_norm_fused_bf16.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(buf_moe), 0);
        enc.set_buffer(1, Some(&weight_buffer.buf), norm_off);
        enc.set_buffer(2, Some(ctx.buf_input.as_ref().unwrap()), 0);
        enc.set_bytes(3, 4, &hidden_u32 as *const u32 as *const c_void);
        {
            let eps = RMS_NORM_EPS;
            enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
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
    weight_buffer: &WeightBuffer, ctx: &MetalContext,
) {
    let x_buf = metal_buf_shared(&ctx.device, hidden.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(hidden.as_ptr(), x_buf.contents() as *mut f32, hidden.len());
    }
    let out_buf = metal_buf_shared(&ctx.device, logits.len() * 4);
    let cm = ctx.queue.new_command_buffer();
    let enc = cm.new_compute_command_encoder();
    weight_buffer.encode_matvec_into(wf, ctx, &enc, "lm_head", &x_buf, 0, &out_buf, 0, logits.len(), hidden.len());
    enc.end_encoding();
    cm.commit();
    cm.wait_until_completed();
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents() as *const f32, logits.as_mut_ptr(), logits.len());
    }
}

// ─── Fused4bitExp3 ──────────────────────────────────────────────────────

pub struct Fused4bitExp3<C: ModelConfig> {
    pub model: Arc<Model>,
    pub ctx: MetalContext,
    pub weight_buffer: WeightBuffer,
    pub expert_buffer: ExpertBuffer,
    pub k: usize,
    pub timing: BTreeMap<String, TelemetryValue>,
    _phantom: PhantomData<C>,
}

impl<C: ModelConfig> Fused4bitExp3<C> {
    pub fn new(model: Arc<Model>, k: usize) -> Result<Self, MoEError> {
        C::validate_config(&model.config).map_err(MoEError::Config)?;
        let (ctx, weight_buffer, expert_buffer) = MetalContext::new::<C>(&model.weight_file, k, "Fused4bitExp3")?;
        Ok(Fused4bitExp3 {
            model, ctx, weight_buffer, expert_buffer,
            k: if k == 0 { C::NUM_EXPERTS_PER_TOK } else { k },
            timing: BTreeMap::new(),
            _phantom: PhantomData,
        })
    }
}

impl<C: ModelConfig> Engine for Fused4bitExp3<C> {
    fn upload_cache(&self, cache: &Cache) {
        self.ctx.upload_cache(cache);
    }

    fn download_cache(&self, cache: &mut Cache) {
        self.ctx.download_cache(cache);
    }

    fn forward(
        &mut self,
        input_ids: &[i64],
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError> {
        assert!(self.k <= C::NUM_EXPERTS_PER_TOK,
            "k ({}) must not exceed model's num_experts_per_tok ({})",
            self.k, C::NUM_EXPERTS_PER_TOK);

        let t0 = Instant::now();
        let n_tokens = input_ids.len();
        let hidden_dim = C::HIDDEN_DIM;
        let vocab_size = C::VOCAB_SIZE;
        let num_layers = C::NUM_LAYERS;

        let mut logits = vec![0.0f32; n_tokens * vocab_size];
        if n_tokens == 0 {
            return Ok(logits);
        }

        let mut pos = self.ctx.pos.get();
        {
            let mut exec = ExecCtx {
                engine: self,
                pending: None,
                pending_spare: None,
                spare_thread: None,
            };

            let mut embed = vec![0.0f32; n_tokens * hidden_dim];
            for (i, &id) in input_ids.iter().enumerate() {
                embed_lookup(&exec.engine.model.weight_file, id as usize, &mut embed[i * hidden_dim..(i + 1) * hidden_dim], C::HIDDEN_DIM);
            }

            for (ti, _) in input_ids.iter().enumerate() {
                let embed_hidden = &embed[ti * hidden_dim..(ti + 1) * hidden_dim];
                exec.init_hidden(embed_hidden);

                for layer in 0..num_layers {
                    if check_signal() {
                        return Err(MoEError::Metal("interrupted".into()));
                    }
                    // op1 bundles prev layer's deferred op2 + this layer's op1 into
                    // one commit, then loads prev layer's spare experts while GPU runs
                    let gate_scores = exec.op1_wait(layer, pos);
                    let (routing, spare_info) = exec.route_experts(layer, gate_scores);
                    exec.pending_spare = Some(spare_info);
                    exec.op2(layer, &routing);
                }

                let mut hidden = exec.hidden_wait();
                pos += 1;
                exec.engine.ctx.pos.set(pos);
                exec.final_norm_and_lm_head(&mut hidden, &mut logits[ti * vocab_size..(ti + 1) * vocab_size]);
            }
        } // exec dropped — ends borrow of self

        timing_add(&mut self.timing, "engine.total_ms", t0.elapsed().as_secs_f64() * 1000.0);
        Ok(logits)
    }

    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        self.timing.clone()
    }
}

/// Pipelined FusedExp: N+1 command buffers for N layers.
///
/// Layer boundary: [post_expert(L-1) + pre_expert(L)] fused into one command buffer.
/// GPU data flows between layers through persistent buffers — no CPU round-trip.
///
/// ```
/// CMD 0: pre_expert(0)
/// CMD 1: post_expert(0) + pre_expert(1)
/// ...
/// CMD_{N-1}: post_expert(N-2) + pre_expert(N-1)
/// CMD_N: post_expert(N-1)  [+ GPU-side input_norm for next token]
/// ```
use std::os::fd::RawFd;

use metal::{Buffer, ComputeCommandEncoderRef};

use crate::metal_kernels;
use crate::metal_context::{GpuWeightCtx, MetalContext, MAX_K};
use crate::constants::{CONV_KERNEL_SIZE, FULL_ATTN_INTERVAL, GROUP_SIZE, RMS_NORM_EPS};
use crate::engine::{Cache, Engine, FullAttnCache, LinearAttnState, Model};
use crate::metal_context::ExpertBuffer;
use crate::math::{
    bf16_to_f32, normalize_weights, softmax, topk,
    embed_lookup, final_norm, gpu_lm_head,
    DeferredExperts, ExecCtxGpu, FullAttnCmd2State,
    SignalCheckFn,
};
use crate::math::{gpu_full_attention_forward, moe_layer_forward};
use crate::model_weights::WeightFile;

/// Encode post_expert for a previously-routed layer into a command encoder.
///
/// Reads from persistent GPU buffers written by pre_expert(L-1):
///   - buf_post_normed → expert input (h_post)
///   - buf_shared_gate, buf_shared_up → shared SwiGLU inputs
/// Uses caller-provided scratch buffers for expert intermediates (no conflict
/// with pre_expert which also writes to buf_shared_gate/up for the NEXT layer).
///
/// Writes:
///   - buf_moe_hidden = moe_combine output (hidden state after this layer)
///   - buf_input = input_layernorm(buf_moe_hidden) for next layer's pre_expert
///     (skipped when next_norm_weight is None, e.g. last layer)
pub fn encode_post_expert(
    wf: &WeightFile,
    gpu_wf: &GpuWeightCtx,
    ctx: &MetalContext,
    enc: &ComputeCommandEncoderRef,
    layer_idx: usize,
    expert_weights: &[f32],
    shared_gate_score: f32,
    expert_data: &[Buffer],         // [K] pread expert weight buffers
    expert_scratch_gate: &Buffer,   // moe_inter * 4
    expert_scratch_up: &Buffer,     // moe_inter * 4
    expert_scratch_act: &Buffer,    // moe_inter * 4
    expert_out_bufs: &[Buffer],     // [K] per-expert down_proj outputs
    shared_scratch: &Buffer,        // shared_inter * 4 (shared SwiGLU activation)
    shared_down_buf: &Buffer,       // hidden_dim * 4 (shared down_proj output)
    combine_params_buf: &Buffer,    // 40 bytes (10 f32)
    next_norm_weight: Option<(*const std::ffi::c_void, usize)>, // (ptr, base) for next layer's input_layernorm
    hidden_dim: usize,
    moe_inter: usize,
    shared_inter: usize,
    num_experts_per_tok: usize,
    layout: &crate::model_config::ExpertLayout,
) {
    let hidden_u32 = hidden_dim as u32;
    let inter_u32 = moe_inter as u32;
    let gs_u32 = GROUP_SIZE as u32;
    let actual_k = num_experts_per_tok.min(MAX_K);
    let prefix = format!("model.layers.{}.mlp", layer_idx);

    let post_normed = ctx.buf_post_normed.as_ref().unwrap();

    // ── Expert dispatch: gate/up → SwiGLU → down for each expert ──
    // Each expert writes down_proj to its own expert_out_bufs[ki] for combine.
    for ki in 0..actual_k {
        let expert_buf = &expert_data[ki];
        if expert_buf.length() == 0 { continue; }

        metal_kernels::encode_matvec_offset(ctx, enc,
            expert_buf, layout.gate_w_off as u64,
            expert_buf, layout.gate_s_off as u64,
            expert_buf, layout.gate_b_off as u64,
            post_normed, 0, expert_scratch_gate, 0,
            inter_u32, hidden_u32, gs_u32, 3);

        metal_kernels::encode_matvec_offset(ctx, enc,
            expert_buf, layout.up_w_off as u64,
            expert_buf, layout.up_s_off as u64,
            expert_buf, layout.up_b_off as u64,
            post_normed, 0, expert_scratch_up, 0,
            inter_u32, hidden_u32, gs_u32, 3);

        metal_kernels::encode_swiglu(ctx, enc, expert_scratch_gate, 0, expert_scratch_up, 0,
            expert_scratch_act, 0, inter_u32);

        metal_kernels::encode_matvec_offset(ctx, enc,
            expert_buf, layout.down_w_off as u64,
            expert_buf, layout.down_s_off as u64,
            expert_buf, layout.down_b_off as u64,
            expert_scratch_act, 0, &expert_out_bufs[ki], 0,
            hidden_u32, inter_u32, gs_u32, 3);
    }

    // ── Shared expert SwiGLU (reads buf_shared_gate, buf_shared_up from pre_expert) ──
    {
        let sg = ctx.buf_shared_gate.as_ref().unwrap();
        let su = ctx.buf_shared_up.as_ref().unwrap();
        metal_kernels::encode_swiglu(ctx, enc, sg, 0, su, 0, shared_scratch, 0, shared_inter as u32);
    }

    // ── Shared down_proj ──
    let sd_name = format!("{}.shared_expert.down_proj", prefix);
    let _ = gpu_wf.encode_matvec_into(wf, ctx, enc, &sd_name, shared_scratch, 0,
        shared_down_buf, 0, hidden_dim, shared_inter);

    // ── moe_combine_residual ──
    // h_mid = buf_temp_residual = previous_layer_output + current_attention_output.
    // This is set by pre_expert(L)'s residual_add in the previous CMD (or by CPU
    // upload for the very first layer). Using buf_moe_hidden directly would miss
    // the attention contribution.
    // Writes: buf_moe_hidden = h_mid + moe_out + shared_out
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
        // Upload combine params (reused each CMD — caller fills before encoding)
        let mut cparams = [0.0f32; 10];
        for (i, &w) in expert_weights.iter().enumerate() { cparams[i] = w; }
        cparams[8] = shared_gate_score;
        unsafe { std::ptr::copy_nonoverlapping(cparams.as_ptr(), combine_params_buf.contents() as *mut f32, 10); }
        enc.set_buffer(11, Some(combine_params_buf), 0);
        unsafe {
            enc.set_bytes(12, 4, &hidden_u32 as *const u32 as *const std::ffi::c_void);
            let ku = actual_k as u32;
            enc.set_bytes(13, 4, &ku as *const u32 as *const std::ffi::c_void);
        }
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }

    // ── GPU-side input_norm for next layer: buf_moe_hidden → buf_input ──
    if let Some((norm_ptr, base)) = next_norm_weight {
        let norm_off = (norm_ptr as usize - base) as u64;
        let buf_moe = ctx.buf_moe_hidden.as_ref().unwrap();
        let sum_sq = ctx.buf_cmd3_sum_sq.as_ref().unwrap();

        enc.set_compute_pipeline_state(&ctx.rms_norm_sum);
        enc.set_buffer(0, Some(buf_moe), 0);
        enc.set_buffer(1, Some(sum_sq), 0);
        unsafe { enc.set_bytes(2, 4, &hidden_u32 as *const u32 as *const std::ffi::c_void); }
        enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));

        let pipe = ctx.rms_norm_apply_bf16.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(buf_moe), 0);
        enc.set_buffer(1, Some(&gpu_wf.buf), norm_off);
        enc.set_buffer(2, Some(sum_sq), 0);
        enc.set_buffer(3, Some(ctx.buf_input.as_ref().unwrap()), 0);
        unsafe {
            enc.set_bytes(4, 4, &hidden_u32 as *const u32 as *const std::ffi::c_void);
            let eps = RMS_NORM_EPS;
            enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
        }
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }
}

/// Encode pre_expert for a layer into a command encoder.
///
/// Reads from persistent GPU buffers:
///   - buf_input → normed hidden for attention (from previous post_expert or CPU upload)
///   - buf_moe_hidden → h_mid for residual_add (from previous post_expert or CPU upload)
/// Writes:
///   - buf_gate_scores → gate projection output (CPU reads for routing)
///   - buf_post_normed → post-attn normed hidden (next post_expert's input)
///   - buf_shared_gate, buf_shared_up → shared projections (next post_expert's SwiGLU input)
///   - buf_shared_gate_score → shared gate scalar
pub fn encode_pre_expert(
    wf: &WeightFile,
    gpu_wf: &GpuWeightCtx,
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
    let prefix = format!("model.layers.{}.linear_attn", layer_idx);

    // ── Attention projections (4 matvecs, reads buf_input) ──
    let input_buf = c.buf_input.as_ref().unwrap();
    {
        gw.encode_matvec_into(wf, c, enc, &format!("{}.in_proj_qkv", prefix), input_buf, 0, &c.batch_out[0], 0, qkv_dim, hidden_dim);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.in_proj_z", prefix), input_buf, 0, &c.batch_out[1], 0, total_value, hidden_dim);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.in_proj_b", prefix), input_buf, 0, &c.batch_out[2], 0, num_v_heads, hidden_dim);
        gw.encode_matvec_into(wf, c, enc, &format!("{}.in_proj_a", prefix), input_buf, 0, &c.batch_out[3], 0, num_v_heads, hidden_dim);
    }

    // ── conv1d_step ──
    if let Some(conv_w_ptr) = wf.get_tensor_ptr(&format!("{}.conv1d.weight", prefix)) {
        let conv_w_off = (conv_w_ptr as usize - gw.base as usize) as u64;
        metal_kernels::encode_conv1d_step(c, enc,
            &c.buf_conv_state[linear_idx],
            &c.batch_out[0],
            &gw.buf, conv_w_off,
            c.buf_conv_output.as_ref().unwrap(),
            qkv_dim as u32);
    }

    // ── rms_norm_qk ──
    metal_kernels::encode_rms_norm_qk(c, enc,
        c.buf_conv_output.as_ref().unwrap(), 0,
        c.buf_conv_output.as_ref().unwrap(), (total_key * 4) as u64,
        num_k_heads as u32, key_dim as u32, inv_scale);

    // ── compute_decay_beta ──
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

    // ── gated_delta_net_step ──
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

    // ── gated_rms_norm → batch_out[6] (gated on GPU for out_proj) ──
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

    // ── out_proj (batch_out[6] → buf_out_proj) ──
    let out_proj_buf = c.buf_out_proj.as_ref().unwrap();
    gw.encode_matvec_into(wf, c, enc, &format!("{}.out_proj", prefix),
        &c.batch_out[6], 0, out_proj_buf, 0, hidden_dim, total_value);

    // ── residual_add (out_proj + h_mid → buf_temp_residual) ──
    // h_mid = buf_moe_hidden (from previous post_expert)
    {
        let pipe = c.residual_add.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(out_proj_buf), 0);
        enc.set_buffer(1, Some(c.buf_moe_hidden.as_ref().unwrap()), 0);
        enc.set_buffer(2, Some(c.buf_temp_residual.as_ref().unwrap()), 0);
        unsafe { enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const std::ffi::c_void); }
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }

    // ── Post-attn layernorm (buf_temp_residual → buf_post_normed) ──
    let post_norm_name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
    let pnw_ptr = wf.get_tensor_ptr(&post_norm_name).unwrap();
    let pnw_off = (pnw_ptr as usize - gw.base as usize) as u64;
    let temp_res = c.buf_temp_residual.as_ref().unwrap();
    let post_sum = c.buf_post_sum_sq.as_ref().unwrap();
    {
        enc.set_compute_pipeline_state(&c.rms_norm_sum);
        enc.set_buffer(0, Some(temp_res), 0);
        enc.set_buffer(1, Some(post_sum), 0);
        unsafe { enc.set_bytes(2, 4, &(hidden_dim as u32) as *const u32 as *const std::ffi::c_void); }
        enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));
    }
    {
        let pipe = c.rms_norm_apply_bf16.as_ref().unwrap();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(temp_res), 0);
        enc.set_buffer(1, Some(&gw.buf), pnw_off);
        enc.set_buffer(2, Some(post_sum), 0);
        enc.set_buffer(3, Some(c.buf_post_normed.as_ref().unwrap()), 0);
        unsafe {
            enc.set_bytes(4, 4, &(hidden_dim as u32) as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &RMS_NORM_EPS as *const f32 as *const std::ffi::c_void);
        }
        enc.dispatch_thread_groups(
            metal::MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
    }

    // ── Gate + shared projections (read buf_post_normed) ──
    let mlp_prefix = format!("model.layers.{}.mlp", layer_idx);
    let post_normed = c.buf_post_normed.as_ref().unwrap();
    let gate_buf = c.buf_gate_scores.as_ref().unwrap();
    let sg_buf = c.buf_shared_gate.as_ref().unwrap();
    let su_buf = c.buf_shared_up.as_ref().unwrap();
    let sge_buf = c.buf_shared_gate_score.as_ref().unwrap();

    gw.encode_matvec_into(wf, c, enc, &format!("{}.gate", mlp_prefix), post_normed, 0, gate_buf, 0, num_experts, hidden_dim);
    gw.encode_matvec_into(wf, c, enc, &format!("{}.shared_expert.gate_proj", mlp_prefix), post_normed, 0, sg_buf, 0, shared_inter, hidden_dim);
    gw.encode_matvec_into(wf, c, enc, &format!("{}.shared_expert.up_proj", mlp_prefix), post_normed, 0, su_buf, 0, shared_inter, hidden_dim);
    gw.encode_matvec_into(wf, c, enc, &format!("{}.shared_expert_gate", mlp_prefix), post_normed, 0, sge_buf, 0, 1, hidden_dim);
}

/// Update CPU conv_state after CMD1 completes (must be called after wait_until_completed).
pub fn update_conv_state(state: &mut LinearAttnState, qkv_dim: usize) {
    let state_off = (CONV_KERNEL_SIZE - 2) * qkv_dim;
    state.conv_state.copy_within(qkv_dim.., 0);
    state.conv_state[state_off..state_off + qkv_dim].fill(0.0);
}

// ─── FusedExp pipeline orchestrator ──────────────────────────────────────────

/// Run a group of consecutive linear layers using the FusedExp pipelined
/// N+1 command buffer approach.
///
/// Full attention layers break the pipeline and are handled with the existing
/// CMD2 fusion approach. Consecutive linear layers are pipelined.
pub fn process_token_fusedexp_pipelined(
    exec: &mut ExecCtxGpu<'_>,
    hidden: &mut [f32],
    pos: usize,
    kv: &mut [Option<FullAttnCache>],
    lin: &mut [Option<LinearAttnState>],
    check_signal: SignalCheckFn<'_>,
    capture_per_layer: bool,
    layer_outputs: &mut Vec<Vec<f32>>,
) -> Result<(), String> {
    let hd = exec.config.hidden_dim;
    let num_layers = exec.config.num_layers;
    let num_experts = exec.config.num_experts;
    let moe_inter = exec.config.moe_intermediate;
    let shared_inter = exec.config.shared_intermediate;
    let k = exec.config.num_experts_per_tok;
    let expert_size = exec.config.expert_size_4bit;
    let layout = &exec.config.expert_layout_4bit;
    let qkv_dim = exec.config.linear_conv_dim;
    let total_key = exec.config.linear_total_key;
    let total_val = exec.config.linear_total_value;
    let num_k_heads = exec.config.linear_num_k_heads;
    let num_v_heads = exec.config.linear_num_v_heads;
    let key_dim = total_key / num_k_heads;
    let val_dim = total_val / num_v_heads;
    let inv_scale = 1.0 / (key_dim as f32).sqrt();
    let k_heads_per_v = num_v_heads / num_k_heads;

    let upload_first_layer =
        |ctx: &MetalContext, wf: &WeightFile, hidden: &[f32], layer_idx: usize| {
            let buf_moe = ctx.buf_moe_hidden.as_ref().unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(hidden.as_ptr(), buf_moe.contents() as *mut f32, hd);
            }
            let norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
            let buf_in = ctx.buf_input.as_ref().unwrap();
            if let Some(nw_u16) = wf.get_tensor_u16(&norm_name) {
                let nw: Vec<f32> = nw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
                let sum_sq: f32 = hidden[..hd].iter().map(|v| v * v).sum();
                let inv_rms = 1.0 / (sum_sq / hd as f32 + RMS_NORM_EPS).sqrt();
                unsafe {
                    let dst = buf_in.contents() as *mut f32;
                    for i in 0..hd {
                        *dst.add(i) = hidden[i] * inv_rms * nw[i];
                    }
                }
            } else {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        hidden.as_ptr(), buf_in.contents() as *mut f32, hd);
                }
            }
        };

    let route_and_pread = |ctx: &MetalContext,
                           expert_gpu_buffer: &mut crate::metal_context::ExpertBuffer,
                           layer_idx: usize,
                           layer_fd: RawFd|
        -> (Vec<usize>, Vec<f32>, f32)
    {
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
            let mut pread_tasks: Vec<(RawFd, usize, usize, i64)> =
                Vec::with_capacity(miss_count);
            for m in 0..miss_count {
                let ki = miss_k_slot[m];
                let eidx = miss_ei[m];
                let ptr = expert_gpu_buffer.expert_data[ki].contents() as usize;
                pread_tasks.push((
                    layer_fd, ptr, expert_size, (eidx as i64) * (expert_size as i64)));
            }
            rayon::scope(|s| {
                for (fd, dst, sz, off) in pread_tasks {
                    s.spawn(move |_| {
                        unsafe { libc::pread(fd, dst as *mut std::ffi::c_void, sz, off); }
                    });
                }
            });
        }
        (expert_indices, expert_weights, shared_gate_score)
    };

    let mut layer = 0;
    let mut pending_deferred: Option<DeferredExperts> = None;
    while layer < num_layers {
        if check_signal() {
            return Err("interrupted".into());
        }
        let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;

        if is_full {
            if let Some(ref mut def) = pending_deferred.take() {
                def.complete(hidden, hd);
                if capture_per_layer {
                    layer_outputs.push(hidden.to_vec());
                }
            }
            let mut attn_state: Option<FullAttnCmd2State> = None;
            if let Some(ref mut kv_entry) = kv[layer] {
                attn_state = gpu_full_attention_forward(
                    exec.wf, layer, hidden, kv_entry, pos, exec.config,
                    exec.gpu_wf, exec.ctx,
                );
            }
            let r = moe_layer_forward(
                exec.wf, layer, hidden, exec.expert_fds[layer],
                Some(exec.ctx), Some(exec.gpu_wf), exec.config,
                attn_state, None, exec.expert_gpu_buffer.as_mut().map(|x| &mut **x),
                true,
            );
            pending_deferred = r.unwrap_or(None);
            layer += 1;
            continue;
        }

        let group_start = layer;
        while layer < num_layers && (layer + 1) % FULL_ATTN_INTERVAL != 0 {
            layer += 1;
        }
        let group_layers: Vec<usize> = (group_start..layer).collect();
        let m_layers = group_layers.len();
        if m_layers == 0 { continue; }

        let first_layer = group_layers[0];
        let last_layer = group_layers[m_layers - 1];

        if let Some(ref mut def) = pending_deferred.take() {
            def.complete(hidden, hd);
            if capture_per_layer {
                layer_outputs.push(hidden.to_vec());
            }
        }

        upload_first_layer(exec.ctx, exec.wf, hidden, first_layer);

        // CMD 0: pre_expert(first_layer)
        {
            let li = first_layer - (first_layer + 1) / FULL_ATTN_INTERVAL;
            let cmd = exec.ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            encode_pre_expert(
                exec.wf, exec.gpu_wf, exec.ctx, &enc, first_layer, li,
                hd, num_k_heads, num_v_heads, total_key, total_val, qkv_dim,
                key_dim, val_dim, k_heads_per_v, inv_scale, num_experts, shared_inter,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            if let Some(ref mut s) = lin[first_layer] {
                update_conv_state(s, qkv_dim);
            }
        }

        let (expert_indices_0, expert_weights_0, shared_gate_score_0) =
            route_and_pread(
                exec.ctx, exec.expert_gpu_buffer.as_mut().unwrap(),
                first_layer, exec.expert_fds[first_layer]);

        let mut _prev_expert_indices = expert_indices_0;
        let mut prev_expert_weights = expert_weights_0;
        let mut prev_shared_gate_score = shared_gate_score_0;

        for gi in 1..m_layers {
            let prev_layer = group_layers[gi - 1];
            let curr_layer = group_layers[gi];
            let curr_li = curr_layer - (curr_layer + 1) / FULL_ATTN_INTERVAL;

            let next_norm = exec.wf.get_tensor_ptr(
                &format!("model.layers.{}.input_layernorm.weight", curr_layer));
            let next_norm_info = next_norm
                .map(|p| (p as *const std::ffi::c_void, exec.gpu_wf.base as usize));

            let cmd = exec.ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            {
                let io = exec.expert_gpu_buffer.as_ref().unwrap();
                encode_post_expert(
                    exec.wf, exec.gpu_wf, exec.ctx, &enc, prev_layer,
                    &prev_expert_weights, prev_shared_gate_score,
                    &io.expert_data, &io.scratch_gate, &io.scratch_up, &io.scratch_act,
                    &io.expert_out, &io.shared_act, &io.shared_down, &io.combine_params,
                    next_norm_info,
                    hd, moe_inter, shared_inter, k, layout,
                );
            }

            encode_pre_expert(
                exec.wf, exec.gpu_wf, exec.ctx, &enc, curr_layer, curr_li,
                hd, num_k_heads, num_v_heads, total_key, total_val, qkv_dim,
                key_dim, val_dim, k_heads_per_v, inv_scale, num_experts, shared_inter,
            );

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            if let Some(ref mut s) = lin[curr_layer] {
                update_conv_state(s, qkv_dim);
            }

            if capture_per_layer {
                let mut h = vec![0.0f32; hd];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        exec.ctx.buf_moe_hidden.as_ref().unwrap().contents() as *const f32,
                        h.as_mut_ptr(), hd);
                }
                layer_outputs.push(h);
            }

            {
                let (indices, weights, gate_score) = route_and_pread(
                    exec.ctx, exec.expert_gpu_buffer.as_mut().unwrap(),
                    curr_layer, exec.expert_fds[curr_layer]);
                _prev_expert_indices = indices;
                prev_expert_weights = weights;
                prev_shared_gate_score = gate_score;
            }
        }

        // Last CMD: post_expert(last_layer)
        {
            let next_norm_info = if last_layer + 1 < num_layers {
                exec.wf.get_tensor_ptr(
                    &format!("model.layers.{}.input_layernorm.weight", last_layer + 1))
                    .map(|p| (p as *const std::ffi::c_void, exec.gpu_wf.base as usize))
            } else {
                None
            };

            let cmd = exec.ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            {
                let io = exec.expert_gpu_buffer.as_ref().unwrap();
                encode_post_expert(
                    exec.wf, exec.gpu_wf, exec.ctx, &enc, last_layer,
                    &prev_expert_weights, prev_shared_gate_score,
                    &io.expert_data, &io.scratch_gate, &io.scratch_up, &io.scratch_act,
                    &io.expert_out, &io.shared_act, &io.shared_down, &io.combine_params,
                    next_norm_info,
                    hd, moe_inter, shared_inter, k, layout,
                );
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            unsafe {
                std::ptr::copy_nonoverlapping(
                    exec.ctx.buf_moe_hidden.as_ref().unwrap().contents() as *const f32,
                    hidden.as_mut_ptr(), hd);
            }

            if capture_per_layer {
                layer_outputs.push(hidden.to_vec());
            }
        }
    }

    if let Some(ref mut def) = pending_deferred.take() {
        def.complete(hidden, hd);
        if capture_per_layer {
            layer_outputs.push(hidden.to_vec());
        }
    }

    Ok(())
}

// ─── EngineFusedExp ───────────────────────────────────────────────────────

pub struct EngineFusedExp<'a> {
    pub model: &'a Model,
    pub ctx: &'a MetalContext,
    pub gpu_wf: &'a GpuWeightCtx,
    pub expert_gpu_buffer: Option<&'a mut ExpertBuffer>,
}

impl<'a> EngineFusedExp<'a> {
    fn make_exec_ctx(&mut self) -> ExecCtxGpu<'_> {
        ExecCtxGpu {
            wf: &self.model.wf,
            ctx: self.ctx,
            gpu_wf: self.gpu_wf,
            config: &self.model.config,
            expert_fds: &self.model.expert_fds,
            expert_gpu_buffer: self.expert_gpu_buffer.as_deref_mut(),
        }
    }
}

impl<'a> Engine for EngineFusedExp<'a> {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        mut check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, String> {
        let n = input_ids.len();
        let hd = self.model.config.hidden_dim;
        let vs = self.model.config.vocab_size;

        let mut logits = vec![0.0f32; n * vs];
        if n == 0 {
            return Ok(logits);
        }

        let mut embed = vec![0.0f32; n * hd];
        for (i, &id) in input_ids.iter().enumerate() {
            embed_lookup(&self.model.wf, id as usize, &mut embed[i * hd..(i + 1) * hd], hd);
        }

        let mut hidden = vec![0.0f32; hd];
        for (ti, _) in input_ids.iter().enumerate() {
            hidden.copy_from_slice(&embed[ti * hd..(ti + 1) * hd]);
            let mut exec = self.make_exec_ctx();
            process_token_fusedexp_pipelined(
                &mut exec, &mut hidden,
                cache.pos, &mut cache.kv, &mut cache.lin,
                &mut || check_signal(), false, &mut Vec::new(),
            )?;
            cache.pos += 1;
            final_norm(exec.wf, &mut hidden, hd);
            gpu_lm_head(exec.wf, &hidden,
                &mut logits[ti * vs..(ti + 1) * vs],
                exec.gpu_wf, exec.ctx);
        }

        Ok(logits)
    }
}


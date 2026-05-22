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
use metal::{Buffer, ComputeCommandEncoderRef};
use crate::kernels;
use crate::metal_context::{GpuWeightCtx, MetalContext, MAX_K};
use crate::weights::WeightFile;
use crate::pipeline_common::{LinearAttnState, CONV_KERNEL_SIZE};

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
    layout: &crate::config::ExpertLayout,
) {
    let hidden_u32 = hidden_dim as u32;
    let inter_u32 = moe_inter as u32;
    let gs_u32 = crate::pipeline_common::GROUP_SIZE as u32;
    let actual_k = num_experts_per_tok.min(MAX_K);
    let prefix = format!("model.layers.{}.mlp", layer_idx);

    let post_normed = ctx.buf_post_normed.as_ref().unwrap();

    // ── Expert dispatch: gate/up → SwiGLU → down for each expert ──
    // Each expert writes down_proj to its own expert_out_bufs[ki] for combine.
    for ki in 0..actual_k {
        let expert_buf = &expert_data[ki];
        if expert_buf.length() == 0 { continue; }

        kernels::encode_matvec_offset(ctx, enc,
            expert_buf, layout.gate_w_off as u64,
            expert_buf, layout.gate_s_off as u64,
            expert_buf, layout.gate_b_off as u64,
            post_normed, 0, expert_scratch_gate, 0,
            inter_u32, hidden_u32, gs_u32, 3);

        kernels::encode_matvec_offset(ctx, enc,
            expert_buf, layout.up_w_off as u64,
            expert_buf, layout.up_s_off as u64,
            expert_buf, layout.up_b_off as u64,
            post_normed, 0, expert_scratch_up, 0,
            inter_u32, hidden_u32, gs_u32, 3);

        kernels::encode_swiglu(ctx, enc, expert_scratch_gate, 0, expert_scratch_up, 0,
            expert_scratch_act, 0, inter_u32);

        kernels::encode_matvec_offset(ctx, enc,
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
        kernels::encode_swiglu(ctx, enc, sg, 0, su, 0, shared_scratch, 0, shared_inter as u32);
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
            let eps = crate::pipeline_common::RMS_NORM_EPS;
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
        kernels::encode_conv1d_step(c, enc,
            &c.buf_conv_state[linear_idx],
            &c.batch_out[0],
            &gw.buf, conv_w_off,
            c.buf_conv_output.as_ref().unwrap(),
            qkv_dim as u32);
    }

    // ── rms_norm_qk ──
    kernels::encode_rms_norm_qk(c, enc,
        c.buf_conv_output.as_ref().unwrap(), 0,
        c.buf_conv_output.as_ref().unwrap(), (total_key * 4) as u64,
        num_k_heads as u32, key_dim as u32, inv_scale);

    // ── compute_decay_beta ──
    {
        let a_log_ptr = wf.get_tensor_ptr(&format!("{}.A_log", prefix));
        let dt_bias_ptr = wf.get_tensor_ptr(&format!("{}.dt_bias", prefix));
        let a_log_off = a_log_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
        let dt_bias_off = dt_bias_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
        kernels::encode_compute_decay_beta(c, enc,
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
        kernels::encode_gated_delta_net_step(c, enc,
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
            kernels::encode_gated_rms_norm(c, enc,
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
            enc.set_bytes(5, 4, &crate::pipeline_common::RMS_NORM_EPS as *const f32 as *const std::ffi::c_void);
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

/// Fused3 pipeline mode: 3-CMD architecture matching the C engine.
///
/// CMD1: attention projections + conv1d + SSM + gated_rms_norm (no out_proj/residual)
/// CMD2: out_proj + residual_add + rms_norm + gate + shared (1 fused encoder)
/// CMD3: experts + combine + GPU-side input_norm (async, deferred commit)
use metal::Buffer;
use crate::kernels;
use crate::metal_context::{metal_buf_shared, GpuWeightCtx, MetalContext};
use crate::weights::WeightFile;
use crate::pipeline_common::{LinearAttnFused3State, LinearAttnState, CONV_KERNEL_SIZE};

/// Run Fused3 CMD1: attention projections → conv1d → SSM → gated_rms_norm.
///
/// Returns `LinearAttnFused3State` with gated_buf (= batch_out[6]) and h_mid (residual).
/// CMD2 will read from batch_out[6] for out_proj and use h_mid for residual_add.
pub fn fused3_cmd1(
    wf: &WeightFile,
    gpu_wf: &GpuWeightCtx,
    ctx: &MetalContext,
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
    normed: &[f32],
    residual: Vec<f32>,
    state: &mut LinearAttnState,
    prev_gpu_combined: bool,
) -> LinearAttnFused3State {
    let c = ctx;
    let gw = gpu_wf;
    let prefix = format!("model.layers.{}.linear_attn", layer_idx);
    let prefix_std = format!("{}.in_proj_qkv", prefix);
    let prefix_z = format!("{}.in_proj_z", prefix);
    let prefix_b = format!("{}.in_proj_b", prefix);
    let prefix_a = format!("{}.in_proj_a", prefix);

    // When prev CMD3 did GPU-side input_norm, buf_input has normed hidden on GPU.
    // Skip CPU upload and use buf_input directly (matches C FAST PATH).
    let input_buf: Buffer;
    if prev_gpu_combined && c.buf_input.is_some() {
        input_buf = c.buf_input.as_ref().unwrap().clone();
    } else {
        let x = metal_buf_shared(&c.device, hidden_dim * 4);
        unsafe { std::ptr::copy_nonoverlapping(normed.as_ptr(), x.contents() as *mut f32, hidden_dim); }
        input_buf = x;
    }

    // CMD1: encoders 1-6 (projections + conv1d + rms_norm_qk + decay_beta + SSM + gated_norm)
    let cmd_buf = c.queue.new_command_buffer();

    // Encoder 1: 4 attention projections
    {
        let enc = cmd_buf.new_compute_command_encoder();
        gw.encode_matvec_into(wf, c, &enc, &prefix_std, &input_buf, 0, &c.batch_out[0], 0, qkv_dim, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &prefix_z, &input_buf, 0, &c.batch_out[1], 0, total_value, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &prefix_b, &input_buf, 0, &c.batch_out[2], 0, num_v_heads, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &prefix_a, &input_buf, 0, &c.batch_out[3], 0, num_v_heads, hidden_dim);
        enc.end_encoding();
    }

    // Encoder 2: conv1d_step
    if let Some(conv_w_ptr) = wf.get_tensor_ptr(&format!("{}.conv1d.weight", prefix)) {
        let conv_w_off = (conv_w_ptr as usize - gw.base as usize) as u64;
        let enc = cmd_buf.new_compute_command_encoder();
        kernels::encode_conv1d_step(c, &enc,
            &c.buf_conv_state[linear_idx],
            &c.batch_out[0],
            &gw.buf, conv_w_off,
            c.buf_conv_output.as_ref().unwrap(),
            qkv_dim as u32);
        enc.end_encoding();
    }

    // Encoder 3: rms_norm_qk
    {
        let enc = cmd_buf.new_compute_command_encoder();
        kernels::encode_rms_norm_qk(c, &enc,
            c.buf_conv_output.as_ref().unwrap(), 0,
            c.buf_conv_output.as_ref().unwrap(), (total_key * 4) as u64,
            num_k_heads as u32, key_dim as u32, inv_scale);
        enc.end_encoding();
    }

    // Encoder 4: compute_decay_beta
    {
        let a_log_ptr = wf.get_tensor_ptr(&format!("{}.A_log", prefix));
        let dt_bias_ptr = wf.get_tensor_ptr(&format!("{}.dt_bias", prefix));
        let a_log_off = a_log_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
        let dt_bias_off = dt_bias_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
        let enc = cmd_buf.new_compute_command_encoder();
        kernels::encode_compute_decay_beta(c, &enc,
            &c.batch_out[3],
            &c.batch_out[2],
            if a_log_ptr.is_some() { &gw.buf } else { &c.batch_out[3] }, a_log_off,
            if dt_bias_ptr.is_some() { &gw.buf } else { &c.batch_out[2] }, dt_bias_off,
            c.buf_delta_g_decay.as_ref().unwrap(),
            c.buf_delta_beta.as_ref().unwrap(),
            num_v_heads as u32);
        enc.end_encoding();
    }

    // Encoder 5: gated_delta_net_step
    {
        let q_off = 0u64;
        let k_off = (total_key * 4) as u64;
        let v_off = (2 * total_key * 4) as u64;
        let conv_out = c.buf_conv_output.as_ref().unwrap();
        let enc = cmd_buf.new_compute_command_encoder();
        kernels::encode_gated_delta_net_step(c, &enc,
            &c.buf_delta_state[linear_idx],
            conv_out, q_off,
            conv_out, k_off,
            conv_out, v_off,
            c.buf_delta_g_decay.as_ref().unwrap(),
            c.buf_delta_beta.as_ref().unwrap(),
            c.buf_delta_output.as_ref().unwrap(),
            num_v_heads as u32, k_heads_per_v as u32, key_dim as u32, value_dim as u32);
        enc.end_encoding();
    }

    // Encoder 6: gated_rms_norm → batch_out[6] (matches C's CMD1 encoder L5)
    {
        let gnw_ptr = wf.get_tensor_ptr(&format!("{}.norm.weight", prefix));
        let enc = cmd_buf.new_compute_command_encoder();
        if let Some(gnw_p) = gnw_ptr {
            let gnw_off = (gnw_p as usize - gw.base as usize) as u64;
            kernels::encode_gated_rms_norm(c, &enc,
                c.buf_delta_output.as_ref().unwrap(),
                &c.batch_out[1],
                &gw.buf, gnw_off,
                &c.batch_out[6],
                num_v_heads as u32, value_dim as u32);
        }
        enc.end_encoding();
    }

    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Update CPU conv_state for consistency
    let state_off = (CONV_KERNEL_SIZE - 2) * qkv_dim;
    state.conv_state.copy_within(qkv_dim.., 0);
    state.conv_state[state_off..state_off + qkv_dim].fill(0.0);

    LinearAttnFused3State {
        gated_buf: c.batch_out[6].clone(),
        h_mid: residual,
        total_value,
        o_prefix: format!("{}.out_proj", prefix),
        post_norm_name: format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
    }
}

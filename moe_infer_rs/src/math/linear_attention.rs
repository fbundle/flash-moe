use std::ffi::c_void;

use metal::{Buffer, MTLSize};

use crate::cache::LinearAttnState;
use crate::constants::{CONV_KERNEL_SIZE, GROUP_SIZE, RMS_NORM_EPS};
use crate::metal_kernels;
use crate::metal_context::{metal_buf_shared, WeightBuffer, MetalContext};
use crate::model::weights::WeightFile;

use crate::math::{bf16_to_f32, conv1d_step, dequant_matvec_4bit, rms_norm, rms_norm_bare, rms_norm_gated, sigmoid};

// ─── FusedWoods shared state ─────────────────────────────────────────────
/// GPU/CPU state from linear attention CMD1 for FusedWoods.
pub struct LinearAttnGpuOut {
    pub gated_buf: Buffer,
    pub h_mid: Vec<f32>,
    pub total_value: usize,
    pub o_prefix: String,
    pub post_norm_name: String,
}

// ─── Linear attention forward (GatedDeltaNet) ─────────────────────────────

/// Full linear attention forward (GatedDeltaNet) for single-token incremental inference.
/// Port of fused_layer_forward from layer_forward.h (CMD1 linear attention pipeline).
///
/// When `prev_gpu_combined` is true, buf_input already holds normed hidden from the
/// previous layer's CMD3, so CPU rms_norm is skipped and CMD1 reads directly from
/// buf_input (FAST PATH matching C's prev_gpu_combined optimization).
pub fn gpu_linear_attention(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    state: &mut LinearAttnState,
    hidden_dim: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    total_key: usize,
    total_value: usize,
    qkv_dim: usize,
    gpu_wf: Option<&WeightBuffer>,
    ctx: Option<&MetalContext>,
    linear_idx: usize,  // index into persistent GPU state buffers
    use_fused_cmd1: bool,
    use_fusedwoods_cmd1: bool,
    prev_gpu_combined: bool,
) -> Option<LinearAttnGpuOut> {
    let use_gpu = gpu_wf.is_some() && ctx.is_some();

    // Input RMS norm
    let norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
    let nw = wf.get_tensor_u16(&norm_name);
    let mut normed = vec![0.0f32; hidden_dim];
    let mut residual = vec![0.0f32; hidden_dim];
    residual.copy_from_slice(hidden);

    if let Some(nw) = nw {
        let nw_f32: Vec<f32> = nw.iter().map(|&v| bf16_to_f32(v)).collect();
        rms_norm(hidden, &nw_f32, &mut normed, hidden_dim, RMS_NORM_EPS);
    } else {
        normed.copy_from_slice(hidden);
    }

    let prefix = format!("model.layers.{}.linear_attn", layer_idx);

    let key_dim = total_key / num_k_heads;
    let value_dim = total_value / num_v_heads;
    let inv_scale = 1.0 / (key_dim as f32).sqrt();
    let k_heads_per_v = num_v_heads / num_k_heads;

    // ── Fused GPU path (CMD1): attention projections + conv1d + SSM in ONE command buffer ──
    let gpu_compatible = key_dim == 128 && value_dim == 128 && use_gpu;
    let use_fused_gpu = use_fused_cmd1
        && gpu_compatible
        && ctx.is_some()
        && ctx.unwrap().buf_conv_output.is_some()
        && linear_idx < ctx.unwrap().buf_conv_state.len()
        && linear_idx < ctx.unwrap().buf_delta_state.len()
        && ctx.unwrap().batch_out.len() >= 4
        && ctx.unwrap().residual_add.is_some();

    if use_fused_gpu {
        let c = ctx.unwrap();
        let gw = gpu_wf.unwrap();
        let prefix_std = format!("{}.in_proj_qkv", prefix);
        let prefix_z = format!("{}.in_proj_z", prefix);
        let prefix_b = format!("{}.in_proj_b", prefix);
        let prefix_a = format!("{}.in_proj_a", prefix);

        // Upload normed input + residual once
        let x_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let residual_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        unsafe {
            let dst = x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(normed.as_ptr(), dst, hidden_dim);
            let dst_r = residual_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(residual.as_ptr(), dst_r, hidden_dim);
        }

        // CMD1: Single command buffer — attention projs + full linear attn pipeline
        let cmd_buf = c.queue.new_command_buffer();

        // ── Encoder 1: 4 attention projections → batch_out[0..3] ──
        {
            let enc = cmd_buf.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, &prefix_std, &x_buf, 0, &c.batch_out[0], 0, qkv_dim, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_z, &x_buf, 0, &c.batch_out[1], 0, total_value, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_b, &x_buf, 0, &c.batch_out[2], 0, num_v_heads, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_a, &x_buf, 0, &c.batch_out[3], 0, num_v_heads, hidden_dim);
            enc.end_encoding();
        }

        // ── Encoder 2: conv1d_step (reads qkv from batch_out[0], writes buf_conv_output, updates buf_conv_state) ──
        if let Some(conv_w_ptr) = wf.get_tensor_ptr(&format!("{}.conv1d.weight", prefix)) {
            let conv_w_off = (conv_w_ptr as usize - gw.base as usize) as u64;
            let enc = cmd_buf.new_compute_command_encoder();
            metal_kernels::encode_conv1d_step(c, &enc,
                &c.buf_conv_state[linear_idx],      // persistent conv state
                &c.batch_out[0],                     // input = QKV projection
                &gw.buf, conv_w_off,                 // weights from wf_buf with offset
                c.buf_conv_output.as_ref().unwrap(),  // output
                qkv_dim as u32);
            enc.end_encoding();
        }

        // ── Encoder 3: rms_norm_qk (reads q/k from buf_conv_output at offsets) ──
        {
            let enc = cmd_buf.new_compute_command_encoder();
            metal_kernels::encode_rms_norm_qk(c, &enc,
                c.buf_conv_output.as_ref().unwrap(), 0,                             // q at offset 0
                c.buf_conv_output.as_ref().unwrap(), (total_key * 4) as u64,         // k at offset total_key*f32
                num_k_heads as u32, key_dim as u32, inv_scale);
            enc.end_encoding();
        }

        // ── Encoder 4: compute_decay_beta (reads alpha/beta from batch_out[3]/[2], A_log/dt_bias from wf_buf) ──
        {
            let a_log_ptr = wf.get_tensor_ptr(&format!("{}.A_log", prefix));
            let dt_bias_ptr = wf.get_tensor_ptr(&format!("{}.dt_bias", prefix));
            let a_log_off = a_log_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
            let dt_bias_off = dt_bias_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
            let enc = cmd_buf.new_compute_command_encoder();
            metal_kernels::encode_compute_decay_beta(c, &enc,
                &c.batch_out[3],                             // alpha
                &c.batch_out[2],                             // beta
                if a_log_ptr.is_some() { &gw.buf } else { &c.batch_out[3] }, a_log_off,   // A_log (or dummy)
                if dt_bias_ptr.is_some() { &gw.buf } else { &c.batch_out[2] }, dt_bias_off, // dt_bias (or dummy)
                c.buf_delta_g_decay.as_ref().unwrap(),
                c.buf_delta_beta.as_ref().unwrap(),
                num_v_heads as u32);
            enc.end_encoding();
        }

        // ── Encoder 5: gated_delta_net_step (reads q/k/v from buf_conv_output at offsets, updates buf_delta_state) ──
        {
            let q_off = 0u64;
            let k_off = (total_key * 4) as u64;
            let v_off = (2 * total_key * 4) as u64;
            let conv_out = c.buf_conv_output.as_ref().unwrap();
            let enc = cmd_buf.new_compute_command_encoder();
            metal_kernels::encode_gated_delta_net_step(c, &enc,
                &c.buf_delta_state[linear_idx],   // persistent SSM state
                conv_out, q_off,                   // q at offset 0
                conv_out, k_off,                   // k at offset total_key*4
                conv_out, v_off,                   // v at offset 2*total_key*4
                c.buf_delta_g_decay.as_ref().unwrap(),
                c.buf_delta_beta.as_ref().unwrap(),
                c.buf_delta_output.as_ref().unwrap(),
                num_v_heads as u32, k_heads_per_v as u32, key_dim as u32, value_dim as u32);
            enc.end_encoding();
        }

        // ── Encoder 6: gated_rms_norm (reads buf_delta_output, z from batch_out[1], weight from wf_buf) ──
        let gated_gpu = metal_buf_shared(&c.device, total_value * 4);
        {
            let gnw_ptr = wf.get_tensor_ptr(&format!("{}.norm.weight", prefix));
            let enc = cmd_buf.new_compute_command_encoder();
            if let Some(gnw_p) = gnw_ptr {
                let gnw_off = (gnw_p as usize - gw.base as usize) as u64;
                metal_kernels::encode_gated_rms_norm(c, &enc,
                    c.buf_delta_output.as_ref().unwrap(),
                    &c.batch_out[1],                // z
                    &gw.buf, gnw_off,               // norm weight from wf_buf
                    &gated_gpu,
                    num_v_heads as u32, value_dim as u32);
            }
            enc.end_encoding();
        }

        // ── Encoder 7: out_proj matvec (gated_out → hidden_dim) ──
        let o_proj_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        {
            let enc = cmd_buf.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.out_proj", prefix),
                &gated_gpu, 0, &o_proj_buf, 0, hidden_dim, total_value);
            enc.end_encoding();
        }

        // ── Encoder 8: residual_add (o_proj_out + residual → hidden_out) ──
        let hidden_out = metal_buf_shared(&c.device, hidden_dim * 4);
        {
            let enc = cmd_buf.new_compute_command_encoder();
            let pipe = c.residual_add.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&o_proj_buf), 0);
            enc.set_buffer(1, Some(&residual_buf), 0);
            enc.set_buffer(2, Some(&hidden_out), 0);
            enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read final hidden (already has residual + attn_out)
        unsafe {
            std::ptr::copy_nonoverlapping(hidden_out.contents() as *const f32,
                hidden.as_mut_ptr(), hidden_dim);
        }

        // Update CPU conv_state for non-fused fallback / debugging
        let state_off = (CONV_KERNEL_SIZE - 2) * qkv_dim;
        state.conv_state.copy_within(qkv_dim.., 0);
        state.conv_state[state_off..state_off + qkv_dim].fill(0.0);

        // Skip separate out_proj + residual (already done in CMD1)
        return None;
    }

    // ── FusedWoods path (matching C exactly): CMD1 without gated_norm/out_proj/residual ──
    // C does: CMD1(qkvz+ba+conv1d+SSM) → CPU(gated_norm) → CMD2(out_proj+residual+norm+gate+shared)
    let use_fusedwoods_gpu = use_fusedwoods_cmd1
        && gpu_compatible
        && ctx.is_some()
        && ctx.unwrap().buf_conv_output.is_some()
        && linear_idx < ctx.unwrap().buf_conv_state.len()
        && linear_idx < ctx.unwrap().buf_delta_state.len()
        && ctx.unwrap().batch_out.len() >= 8;

    if use_fusedwoods_gpu {
        let c = ctx.unwrap();
        let gw = gpu_wf.unwrap();
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

        // CMD1: encoders 1-5 only (projections + conv1d + rms_norm_qk + decay_beta + SSM)
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
            metal_kernels::encode_conv1d_step(c, &enc,
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
            metal_kernels::encode_rms_norm_qk(c, &enc,
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
            metal_kernels::encode_compute_decay_beta(c, &enc,
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
            metal_kernels::encode_gated_delta_net_step(c, &enc,
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
                metal_kernels::encode_gated_rms_norm(c, &enc,
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

        return Some(LinearAttnGpuOut {
            gated_buf: c.batch_out[6].clone(),
            h_mid: residual,  // pre-attention hidden (saved before norm)
            total_value,
            o_prefix: format!("{}.out_proj", prefix),
            post_norm_name: format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
        });
    }

    // ── Non-fused or CPU path ──
    // CPU: attention projections
    let mut qkv = vec![0.0f32; qkv_dim];
    let mut z = vec![0.0f32; total_value];
    let mut beta = vec![0.0f32; num_v_heads];
    let mut alpha = vec![0.0f32; num_v_heads];

    if let (Some(qw), Some(qs), Some(qb)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_qkv.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_qkv.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_qkv.biases", prefix)),
    ) { dequant_matvec_4bit(qw, qs, qb, &normed, &mut qkv, qkv_dim, hidden_dim, GROUP_SIZE); }
    if let (Some(zw), Some(zs), Some(zb)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_z.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_z.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_z.biases", prefix)),
    ) { dequant_matvec_4bit(zw, zs, zb, &normed, &mut z, total_value, hidden_dim, GROUP_SIZE); }
    if let (Some(bw), Some(bs), Some(bb)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_b.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_b.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_b.biases", prefix)),
    ) { dequant_matvec_4bit(bw, bs, bb, &normed, &mut beta, num_v_heads, hidden_dim, GROUP_SIZE); }
    if let (Some(aw), Some(ass), Some(ab)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_a.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_a.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_a.biases", prefix)),
    ) { dequant_matvec_4bit(aw, ass, ab, &normed, &mut alpha, num_v_heads, hidden_dim, GROUP_SIZE); }

    // Conv1d step (CPU)
    let mut conv_out = vec![0.0f32; qkv_dim];
    if let Some(conv_w) = wf.get_tensor_u16(&format!("{}.conv1d.weight", prefix)) {
        conv1d_step(&state.conv_state, &qkv, conv_w, &mut conv_out, qkv_dim, CONV_KERNEL_SIZE);
    } else {
        conv_out.copy_from_slice(&qkv);
    }
    let state_off = (CONV_KERNEL_SIZE - 2) * qkv_dim;
    state.conv_state.copy_within(qkv_dim.., 0);
    state.conv_state[state_off..state_off + qkv_dim].copy_from_slice(&qkv);
    let lin_q = conv_out[..total_key].to_vec();
    let lin_k = conv_out[total_key..2 * total_key].to_vec();
    let lin_v = conv_out[2 * total_key..].to_vec();

    // Try non-fused GPU SSM (or CPU fallback)
    let mut gated_out = vec![0.0f32; total_value];
    let gpu_ssm_ok = gpu_compatible && ctx.is_some();
    if gpu_ssm_ok {
        let c = ctx.unwrap();
        let ssm_size = num_v_heads * value_dim * key_dim;
        let ssm_gpu = state.ssm_state_gpu.get_or_insert_with(|| {
            metal_buf_shared(&c.device, ssm_size * 4)
        });
        unsafe { let dst = ssm_gpu.contents() as *mut f32; std::ptr::copy_nonoverlapping(state.ssm_state.as_ptr(), dst, ssm_size); }

        let q_gpu = metal_buf_shared(&c.device, total_key * 4);
        let k_gpu = metal_buf_shared(&c.device, total_key * 4);
        let v_gpu = metal_buf_shared(&c.device, total_value * 4);
        let z_gpu = metal_buf_shared(&c.device, total_value * 4);
        let alpha_gpu = metal_buf_shared(&c.device, num_v_heads * 4);
        let beta_gpu = metal_buf_shared(&c.device, num_v_heads * 4);
        let out_gpu = metal_buf_shared(&c.device, total_value * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(lin_q.as_ptr(), q_gpu.contents() as *mut f32, total_key);
            std::ptr::copy_nonoverlapping(lin_k.as_ptr(), k_gpu.contents() as *mut f32, total_key);
            std::ptr::copy_nonoverlapping(lin_v.as_ptr(), v_gpu.contents() as *mut f32, total_value);
            std::ptr::copy_nonoverlapping(z.as_ptr(), z_gpu.contents() as *mut f32, total_value);
            std::ptr::copy_nonoverlapping(alpha.as_ptr(), alpha_gpu.contents() as *mut f32, num_v_heads);
            std::ptr::copy_nonoverlapping(beta.as_ptr(), beta_gpu.contents() as *mut f32, num_v_heads);
        }
        let a_log_ptr = wf.get_tensor_ptr(&format!("{}.A_log", prefix));
        let dt_bias_ptr = wf.get_tensor_ptr(&format!("{}.dt_bias", prefix));
        let a_log_gpu = metal_buf_shared(&c.device, num_v_heads * 4);
        let dt_bias_gpu = metal_buf_shared(&c.device, num_v_heads * 2);
        if let Some(p) = a_log_ptr {
            unsafe { std::ptr::copy_nonoverlapping(p as *const f32, a_log_gpu.contents() as *mut f32, num_v_heads); }
        }
        if let Some(p) = dt_bias_ptr {
            unsafe { std::ptr::copy_nonoverlapping(p as *const u16, dt_bias_gpu.contents() as *mut u16, num_v_heads); }
        }
        let g_decay_gpu = metal_buf_shared(&c.device, num_v_heads * 4);
        let beta_gate_gpu = metal_buf_shared(&c.device, num_v_heads * 4);
        let gated_gpu2 = metal_buf_shared(&c.device, total_value * 4);
        let gnw_ptr = wf.get_tensor_u16(&format!("{}.norm.weight", prefix));
        let gnw_gpu = gnw_ptr.map(|gnw| {
            let buf = metal_buf_shared(&c.device, gnw.len() * 2);
            unsafe { std::ptr::copy_nonoverlapping(gnw.as_ptr(), buf.contents() as *mut u16, gnw.len()); }
            buf
        });

        let cmd_buf = c.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        metal_kernels::encode_rms_norm_qk(c, &enc, &q_gpu, 0, &k_gpu, 0, num_k_heads as u32, key_dim as u32, inv_scale);
        metal_kernels::encode_compute_decay_beta(c, &enc, &alpha_gpu, &beta_gpu, &a_log_gpu, 0, &dt_bias_gpu, 0, &g_decay_gpu, &beta_gate_gpu, num_v_heads as u32);
        metal_kernels::encode_gated_delta_net_step(c, &enc, ssm_gpu, &q_gpu, 0, &k_gpu, 0, &v_gpu, 0, &g_decay_gpu, &beta_gate_gpu, &out_gpu, num_v_heads as u32, k_heads_per_v as u32, key_dim as u32, value_dim as u32);
        if let Some(ref gnw_buf) = gnw_gpu {
            metal_kernels::encode_gated_rms_norm(c, &enc, &out_gpu, &z_gpu, gnw_buf, 0, &gated_gpu2, num_v_heads as u32, value_dim as u32);
        }
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        if gnw_gpu.is_some() {
            unsafe { std::ptr::copy_nonoverlapping(gated_gpu2.contents() as *const f32, gated_out.as_mut_ptr(), total_value); }
        } else {
            unsafe { std::ptr::copy_nonoverlapping(out_gpu.contents() as *const f32, gated_out.as_mut_ptr(), total_value); }
        }
        unsafe { std::ptr::copy_nonoverlapping(ssm_gpu.contents() as *const f32, state.ssm_state.as_mut_ptr(), ssm_size); }
    } else {
        let mut q_normed = vec![0.0f32; total_key];
        let mut k_normed = vec![0.0f32; total_key];
        for h in 0..num_k_heads {
            let qh = &lin_q[h * key_dim..(h + 1) * key_dim];
            let qh_out = &mut q_normed[h * key_dim..(h + 1) * key_dim];
            rms_norm_bare(qh, qh_out, key_dim, 1e-6);
            let q_scale = inv_scale * inv_scale;
            for d in qh_out.iter_mut() { *d *= q_scale; }
        }
        for h in 0..num_k_heads {
            let kh = &lin_k[h * key_dim..(h + 1) * key_dim];
            let kh_out = &mut k_normed[h * key_dim..(h + 1) * key_dim];
            rms_norm_bare(kh, kh_out, key_dim, 1e-6);
            for d in kh_out.iter_mut() { *d *= inv_scale; }
        }

        let a_log = wf.get_tensor_f32(&format!("{}.A_log", prefix));
        let dt_bias = wf.get_tensor_u16(&format!("{}.dt_bias", prefix));

        let mut out_values = vec![0.0f32; total_value];

        for vh in 0..num_v_heads {
            let kh = vh / k_heads_per_v;
            let a_val = a_log.map_or(1.0, |al| al[vh]);
            let dt_b = dt_bias.map_or(0.0, |db| bf16_to_f32(db[vh]));
            let softplus_val = (1.0 + (alpha[vh] + dt_b).exp()).ln();
            let g_decay = (-a_val.exp() * softplus_val).exp();
            let beta_gate = sigmoid(beta[vh]);
            let s_off = vh * value_dim * key_dim;
            let ssm = &mut state.ssm_state[s_off..s_off + value_dim * key_dim];
            let v_h = &lin_v[vh * value_dim..(vh + 1) * value_dim];
            let k_h = &k_normed[kh * key_dim..(kh + 1) * key_dim];
            for vi in 0..value_dim {
                for ki in 0..key_dim { ssm[vi * key_dim + ki] *= g_decay; }
            }
            for vi in 0..value_dim {
                let mut kv_mem = 0.0f32;
                for ki in 0..key_dim { kv_mem += ssm[vi * key_dim + ki] * k_h[ki]; }
                let delta = (v_h[vi] - kv_mem) * beta_gate;
                for ki in 0..key_dim { ssm[vi * key_dim + ki] += k_h[ki] * delta; }
            }
            let q_h = &q_normed[kh * key_dim..(kh + 1) * key_dim];
            let o_h = &mut out_values[vh * value_dim..(vh + 1) * value_dim];
            for vi in 0..value_dim {
                let mut sum = 0.0f32;
                for ki in 0..key_dim { sum += ssm[vi * key_dim + ki] * q_h[ki]; }
                o_h[vi] = sum;
            }
        }

        // RMSNormGated
        if let Some(gnw) = wf.get_tensor_u16(&format!("{}.norm.weight", prefix)) {
            for vh in 0..num_v_heads {
                let oh = &out_values[vh * value_dim..(vh + 1) * value_dim];
                let zh = &z[vh * value_dim..(vh + 1) * value_dim];
                let gh = &mut gated_out[vh * value_dim..(vh + 1) * value_dim];
                rms_norm_gated(oh, zh, gnw, gh, value_dim, RMS_NORM_EPS);
            }
        } else {
            gated_out.copy_from_slice(&out_values);
        }
    }

    // Output projection
    let mut attn_out = vec![0.0f32; hidden_dim];
    if use_gpu {
        let gw = gpu_wf.unwrap();
        let c = ctx.unwrap();
        gw.matvec(wf, c, &format!("{}.out_proj", prefix), &gated_out, &mut attn_out, hidden_dim, total_value);
    } else if let (Some(ow), Some(os), Some(ob)) = (
        wf.get_tensor_u32(&format!("{}.out_proj.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.out_proj.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.out_proj.biases", prefix)),
    ) {
        dequant_matvec_4bit(ow, os, ob, &gated_out, &mut attn_out, hidden_dim, total_value, GROUP_SIZE);
    }
    for i in 0..hidden_dim {
        hidden[i] = residual[i] + attn_out[i];
    }
    None
}

/// CPU linear attention: dequant projections → conv1d → SSM → gated_norm → out_proj → residual.
///
/// Writes final hidden (with residual) into `hidden`.
pub fn linear_attention(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    normed: &[f32],
    residual: &[f32],
    state: &mut LinearAttnState,
    num_k_heads: usize,
    num_v_heads: usize,
    total_key: usize,
    total_value: usize,
    qkv_dim: usize,
    hidden_dim: usize,
    key_dim: usize,
    value_dim: usize,
    inv_scale: f32,
    k_heads_per_v: usize,
) {
    let prefix = format!("model.layers.{}.linear_attn", layer_idx);

    // CPU: attention projections
    let mut qkv = vec![0.0f32; qkv_dim];
    let mut z = vec![0.0f32; total_value];
    let mut beta = vec![0.0f32; num_v_heads];
    let mut alpha = vec![0.0f32; num_v_heads];

    if let (Some(qw), Some(qs), Some(qb)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_qkv.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_qkv.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_qkv.biases", prefix)),
    ) { dequant_matvec_4bit(qw, qs, qb, normed, &mut qkv, qkv_dim, hidden_dim, GROUP_SIZE); }
    if let (Some(zw), Some(zs), Some(zb)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_z.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_z.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_z.biases", prefix)),
    ) { dequant_matvec_4bit(zw, zs, zb, normed, &mut z, total_value, hidden_dim, GROUP_SIZE); }
    if let (Some(bw), Some(bs), Some(bb)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_b.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_b.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_b.biases", prefix)),
    ) { dequant_matvec_4bit(bw, bs, bb, normed, &mut beta, num_v_heads, hidden_dim, GROUP_SIZE); }
    if let (Some(aw), Some(ass), Some(ab)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_a.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_a.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_a.biases", prefix)),
    ) { dequant_matvec_4bit(aw, ass, ab, normed, &mut alpha, num_v_heads, hidden_dim, GROUP_SIZE); }

    // Conv1d step (CPU)
    let mut conv_out = vec![0.0f32; qkv_dim];
    if let Some(conv_w) = wf.get_tensor_u16(&format!("{}.conv1d.weight", prefix)) {
        conv1d_step(&state.conv_state, &qkv, conv_w, &mut conv_out, qkv_dim, CONV_KERNEL_SIZE);
    } else {
        conv_out.copy_from_slice(&qkv);
    }
    let state_off = (CONV_KERNEL_SIZE - 2) * qkv_dim;
    state.conv_state.copy_within(qkv_dim.., 0);
    state.conv_state[state_off..state_off + qkv_dim].copy_from_slice(&qkv);
    let lin_q = conv_out[..total_key].to_vec();
    let lin_k = conv_out[total_key..2 * total_key].to_vec();
    let lin_v = conv_out[2 * total_key..].to_vec();

    // CPU SSM
    let mut q_normed = vec![0.0f32; total_key];
    let mut k_normed = vec![0.0f32; total_key];
    for h in 0..num_k_heads {
        let qh = &lin_q[h * key_dim..(h + 1) * key_dim];
        let qh_out = &mut q_normed[h * key_dim..(h + 1) * key_dim];
        rms_norm_bare(qh, qh_out, key_dim, 1e-6);
        let q_scale = inv_scale * inv_scale;
        for d in qh_out.iter_mut() { *d *= q_scale; }
    }
    for h in 0..num_k_heads {
        let kh = &lin_k[h * key_dim..(h + 1) * key_dim];
        let kh_out = &mut k_normed[h * key_dim..(h + 1) * key_dim];
        rms_norm_bare(kh, kh_out, key_dim, 1e-6);
        for d in kh_out.iter_mut() { *d *= inv_scale; }
    }

    let a_log = wf.get_tensor_f32(&format!("{}.A_log", prefix));
    let dt_bias = wf.get_tensor_u16(&format!("{}.dt_bias", prefix));

    let mut out_values = vec![0.0f32; total_value];

    for vh in 0..num_v_heads {
        let kh = vh / k_heads_per_v;
        let a_val = a_log.map_or(1.0, |al| al[vh]);
        let dt_b = dt_bias.map_or(0.0, |db| bf16_to_f32(db[vh]));
        let softplus_val = (1.0 + (alpha[vh] + dt_b).exp()).ln();
        let g_decay = (-a_val.exp() * softplus_val).exp();
        let beta_gate = sigmoid(beta[vh]);
        let s_off = vh * value_dim * key_dim;
        let ssm = &mut state.ssm_state[s_off..s_off + value_dim * key_dim];
        let v_h = &lin_v[vh * value_dim..(vh + 1) * value_dim];
        let k_h = &k_normed[kh * key_dim..(kh + 1) * key_dim];
        for vi in 0..value_dim {
            for ki in 0..key_dim { ssm[vi * key_dim + ki] *= g_decay; }
        }
        for vi in 0..value_dim {
            let mut kv_mem = 0.0f32;
            for ki in 0..key_dim { kv_mem += ssm[vi * key_dim + ki] * k_h[ki]; }
            let delta = (v_h[vi] - kv_mem) * beta_gate;
            for ki in 0..key_dim { ssm[vi * key_dim + ki] += k_h[ki] * delta; }
        }
        let q_h = &q_normed[kh * key_dim..(kh + 1) * key_dim];
        let o_h = &mut out_values[vh * value_dim..(vh + 1) * value_dim];
        for vi in 0..value_dim {
            let mut sum = 0.0f32;
            for ki in 0..key_dim { sum += ssm[vi * key_dim + ki] * q_h[ki]; }
            o_h[vi] = sum;
        }
    }

    // RMSNormGated
    let mut gated_out = vec![0.0f32; total_value];
    if let Some(gnw) = wf.get_tensor_u16(&format!("{}.norm.weight", prefix)) {
        for vh in 0..num_v_heads {
            let oh = &out_values[vh * value_dim..(vh + 1) * value_dim];
            let zh = &z[vh * value_dim..(vh + 1) * value_dim];
            let gh = &mut gated_out[vh * value_dim..(vh + 1) * value_dim];
            rms_norm_gated(oh, zh, gnw, gh, value_dim, RMS_NORM_EPS);
        }
    } else {
        gated_out.copy_from_slice(&out_values);
    }

    // Output projection (CPU)
    let mut attn_out = vec![0.0f32; hidden_dim];
    if let (Some(ow), Some(os), Some(ob)) = (
        wf.get_tensor_u32(&format!("{}.out_proj.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.out_proj.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.out_proj.biases", prefix)),
    ) {
        dequant_matvec_4bit(ow, os, ob, &gated_out, &mut attn_out, hidden_dim, total_value, GROUP_SIZE);
    }

    // Residual add
    for i in 0..hidden_dim {
        hidden[i] = residual[i] + attn_out[i];
    }
}

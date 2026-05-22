/// FusedWoods pipeline mode: 3-CMD architecture matching the C engine.
///
/// CMD1: attention projections + conv1d + SSM + gated_rms_norm (no out_proj/residual)
/// CMD2: out_proj + residual_add + rms_norm + gate + shared (1 fused encoder)
/// CMD3: experts + combine + GPU-side input_norm (async, deferred commit)
use std::ffi::c_void;
use std::os::fd::RawFd;

use metal::{Buffer, CommandBuffer, MTLSize};

use crate::cache::{Cache, FullAttnCache, LinearAttnState};
use crate::constants::{CONV_KERNEL_SIZE, FULL_ATTN_INTERVAL, GROUP_SIZE, MAX_SEQ, RMS_NORM_EPS};
use crate::engine::Engine;
use crate::metal_kernels;
use crate::metal_context::{metal_buf_shared, ExpertBuffer, WeightBuffer, MetalContext, MAX_K};
use crate::model::Model;
use crate::model::config::ModelConfig;
use crate::model::weights::WeightFile;
use crate::engine::SignalCheckFn;
use crate::math::{
    apply_rope, bf16_to_f32, conv1d_step, dequant_matvec_4bit,
    embed_lookup, final_norm, normalize_weights, rms_norm, rms_norm_bare,
    rms_norm_gated, sigmoid, softmax, topk,
};

// ─── FullAttnGpuOut (local copy) ──────────────────────────────────────────

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

// ─── mixed_full_attention_forward (local copy) ────────────────────────────

fn mixed_full_attention_forward(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    kv: &mut FullAttnCache,
    pos: usize,
    config: &ModelConfig,
    gpu_wf: Option<&WeightBuffer>,
    ctx: Option<&MetalContext>,
) -> Option<FullAttnGpuOut> {
    let hidden_dim = config.hidden_dim;
    let num_attn_heads = config.num_attn_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let rotary_dim = config.rotary_dim;
    let rope_theta = config.rope_theta;

    let q_proj_dim = num_attn_heads * head_dim * 2;
    let q_dim = num_attn_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Input RMS norm
    let norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
    let nw = wf.get_tensor_u16(&norm_name);
    let mut normed = vec![0.0f32; hidden_dim];
    if let Some(nw) = nw {
        let nw_f32: Vec<f32> = nw.iter().map(|&v| bf16_to_f32(v)).collect();
        rms_norm(hidden, &nw_f32, &mut normed, hidden_dim, RMS_NORM_EPS);
    } else {
        normed.copy_from_slice(hidden);
    }

    // QKV projections (GPU)
    let mut q_proj_out = vec![0.0f32; q_proj_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];
    if let (Some(gw), Some(c)) = (gpu_wf, ctx) {
        let x_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        unsafe { let dst = x_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(normed.as_ptr(), dst, hidden_dim); }
        let qbuf = metal_buf_shared(&c.device, q_proj_dim * 4);
        let kbuf = metal_buf_shared(&c.device, kv_dim * 4);
        let vbuf = metal_buf_shared(&c.device, kv_dim * 4);
        let cm = c.queue.new_command_buffer();
        let enc = cm.new_compute_command_encoder();
        let q_name = format!("model.layers.{}.self_attn.q_proj", layer_idx);
        let k_name = format!("model.layers.{}.self_attn.k_proj", layer_idx);
        let v_name = format!("model.layers.{}.self_attn.v_proj", layer_idx);
        gw.encode_matvec_into(wf, c, &enc, &q_name, &x_buf, 0, &qbuf, 0, q_proj_dim, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &k_name, &x_buf, 0, &kbuf, 0, kv_dim, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &v_name, &x_buf, 0, &vbuf, 0, kv_dim, hidden_dim);
        enc.end_encoding(); cm.commit(); cm.wait_until_completed();
        unsafe {
            std::ptr::copy_nonoverlapping(qbuf.contents() as *const f32, q_proj_out.as_mut_ptr(), q_proj_dim);
            std::ptr::copy_nonoverlapping(kbuf.contents() as *const f32, k.as_mut_ptr(), kv_dim);
            std::ptr::copy_nonoverlapping(vbuf.contents() as *const f32, v.as_mut_ptr(), kv_dim);
        }
    } else {
        // CPU fallback: dequant QKV projections
        let q_name = format!("model.layers.{}.self_attn.q_proj", layer_idx);
        let k_name = format!("model.layers.{}.self_attn.k_proj", layer_idx);
        let v_name = format!("model.layers.{}.self_attn.v_proj", layer_idx);
        if let (Some(qw), Some(qs), Some(qb)) = (
            wf.get_tensor_u32(&format!("{}.weight", q_name)),
            wf.get_tensor_u16(&format!("{}.scales", q_name)),
            wf.get_tensor_u16(&format!("{}.biases", q_name)),
        ) { dequant_matvec_4bit(qw, qs, qb, &normed, &mut q_proj_out, q_proj_dim, hidden_dim, GROUP_SIZE); }
        if let (Some(kw), Some(ks), Some(kb)) = (
            wf.get_tensor_u32(&format!("{}.weight", k_name)),
            wf.get_tensor_u16(&format!("{}.scales", k_name)),
            wf.get_tensor_u16(&format!("{}.biases", k_name)),
        ) { dequant_matvec_4bit(kw, ks, kb, &normed, &mut k, kv_dim, hidden_dim, GROUP_SIZE); }
        if let (Some(vw), Some(vs), Some(vb)) = (
            wf.get_tensor_u32(&format!("{}.weight", v_name)),
            wf.get_tensor_u16(&format!("{}.scales", v_name)),
            wf.get_tensor_u16(&format!("{}.biases", v_name)),
        ) { dequant_matvec_4bit(vw, vs, vb, &normed, &mut v, kv_dim, hidden_dim, GROUP_SIZE); }
    }

    // Split Q and Q-gate from concatenated output
    let mut q = vec![0.0f32; q_dim];
    let mut q_gate = vec![0.0f32; q_dim];
    for h in 0..num_attn_heads {
        let src = &q_proj_out[h * 2 * head_dim..];
        q[h * head_dim..h * head_dim + head_dim].copy_from_slice(&src[..head_dim]);
        q_gate[h * head_dim..h * head_dim + head_dim].copy_from_slice(&src[head_dim..2 * head_dim]);
    }

    // Q/K norms
    let qn_name = format!("model.layers.{}.self_attn.q_norm.weight", layer_idx);
    let kn_name = format!("model.layers.{}.self_attn.k_norm.weight", layer_idx);
    if let Some(qnw) = wf.get_tensor_u16(&qn_name) {
        for h in 0..num_attn_heads {
            let qh = &mut q[h * head_dim..(h + 1) * head_dim];
            let sum_sq: f32 = qh.iter().map(|&x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / head_dim as f32 + RMS_NORM_EPS).sqrt();
            for i in 0..head_dim.min(qnw.len()) { qh[i] = qh[i] * inv_rms * bf16_to_f32(qnw[i]); }
        }
    }
    if let Some(knw) = wf.get_tensor_u16(&kn_name) {
        for h in 0..num_kv_heads {
            let kh = &mut k[h * head_dim..(h + 1) * head_dim];
            let sum_sq: f32 = kh.iter().map(|&x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / head_dim as f32 + RMS_NORM_EPS).sqrt();
            for i in 0..head_dim.min(knw.len()) { kh[i] = kh[i] * inv_rms * bf16_to_f32(knw[i]); }
        }
    }

    // RoPE
    apply_rope(&mut q, &mut k, pos, num_attn_heads, num_kv_heads, head_dim, rotary_dim, rope_theta);

    // Append K, V to cache
    let cache_pos = kv.len;
    kv.k_cache[cache_pos * kv_dim..(cache_pos + 1) * kv_dim].copy_from_slice(&k);
    kv.v_cache[cache_pos * kv_dim..(cache_pos + 1) * kv_dim].copy_from_slice(&v);
    kv.len += 1;

    let heads_per_kv = num_attn_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let seq_len = kv.len;
    let seq_stride = MAX_SEQ;
    let mut o_out = vec![0.0f32; hidden_dim];

    let use_gpu_attn = ctx.is_some()
        && gpu_wf.is_some()
        && ctx.unwrap().attn_scores_batched.is_some()
        && ctx.unwrap().attn_softmax_batched.is_some()
        && ctx.unwrap().attn_values_batched.is_some();

    if use_gpu_attn {
        let c = ctx.unwrap();
        let q_buf = metal_buf_shared(&c.device, q_dim * 4);
        let kc_buf = metal_buf_shared(&c.device, seq_stride * kv_dim * 4);
        let vc_buf = metal_buf_shared(&c.device, seq_stride * kv_dim * 4);
        let scores_buf = metal_buf_shared(&c.device, num_attn_heads * seq_stride * 4);
        let out_buf = metal_buf_shared(&c.device, q_dim * 4);
        let q_gate_buf = metal_buf_shared(&c.device, q_dim * 4);
        let hidden_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(q.as_ptr(), q_buf.contents() as *mut f32, q_dim);
            std::ptr::copy_nonoverlapping(kv.k_cache.as_ptr(), kc_buf.contents() as *mut f32, seq_len * kv_dim);
            std::ptr::copy_nonoverlapping(kv.v_cache.as_ptr(), vc_buf.contents() as *mut f32, seq_len * kv_dim);
            std::ptr::copy_nonoverlapping(q_gate.as_ptr(), q_gate_buf.contents() as *mut f32, q_dim);
            std::ptr::copy_nonoverlapping(hidden.as_ptr(), hidden_buf.contents() as *mut f32, hidden_dim);
        }

        let o_prefix = format!("model.layers.{}.self_attn.o_proj", layer_idx);
        return Some(FullAttnGpuOut {
            q_buf, q_gate_buf, kc_buf, vc_buf, scores_buf, out_buf, hidden_buf,
            seq_len: seq_len as u32,
            seq_stride: seq_stride as u32,
            num_attn_heads: num_attn_heads as u32,
            head_dim: head_dim as u32,
            kv_dim: kv_dim as u32,
            heads_per_kv: heads_per_kv as u32,
            scale,
            q_dim: q_dim as u32,
            o_prefix,
        });
    }
    // CPU fallback
    {
        let mut attn_out = vec![0.0f32; q_dim];
        for h in 0..num_attn_heads {
            let kv_h = h / heads_per_kv;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = vec![0.0f32; seq_len];
            for p in 0..seq_len {
                let kp = &kv.k_cache[p * kv_dim + kv_h * head_dim..p * kv_dim + (kv_h + 1) * head_dim];
                scores[p] = qh.iter().zip(kp.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale;
            }
            let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = scores.iter().map(|&s| (s - max_val).exp()).sum();
            let inv_sum = 1.0 / sum;
            let oh = &mut attn_out[h * head_dim..(h + 1) * head_dim];
            for p in 0..seq_len {
                let weight = (scores[p] - max_val).exp() * inv_sum;
                let vp = &kv.v_cache[p * kv_dim + kv_h * head_dim..p * kv_dim + (kv_h + 1) * head_dim];
                for d in 0..head_dim { oh[d] += weight * vp[d]; }
            }
        }
        for i in 0..q_dim { attn_out[i] *= 1.0f32 / (1.0f32 + (-q_gate[i]).exp()); }

        let o_prefix = format!("model.layers.{}.self_attn.o_proj", layer_idx);
        if let (Some(gw), Some(c)) = (gpu_wf, ctx) {
            let attn_buf = metal_buf_shared(&c.device, q_dim * 4);
            unsafe { let dst = attn_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(attn_out.as_ptr(), dst, q_dim); }
            let buf = metal_buf_shared(&c.device, hidden_dim * 4);
            let cm = c.queue.new_command_buffer();
            let enc = cm.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, &o_prefix, &attn_buf, 0, &buf, 0, hidden_dim, q_dim);
            enc.end_encoding(); cm.commit(); cm.wait_until_completed();
            unsafe { std::ptr::copy_nonoverlapping(buf.contents() as *const f32, o_out.as_mut_ptr(), hidden_dim); }
        } else {
            if let (Some(ow), Some(os), Some(ob)) = (
                wf.get_tensor_u32(&format!("{}.weight", o_prefix)),
                wf.get_tensor_u16(&format!("{}.scales", o_prefix)),
                wf.get_tensor_u16(&format!("{}.biases", o_prefix)),
            ) { dequant_matvec_4bit(ow, os, ob, &attn_out, &mut o_out, hidden_dim, q_dim, GROUP_SIZE); }
        }
    }

    // Residual add
    for i in 0..hidden_dim { hidden[i] += o_out[i]; }
    None
}

// ─── LinearAttnGpuOut (local copy) ───────────────────────────────────────

struct LinearAttnGpuOut {
    gated_buf: Buffer,
    h_mid: Vec<f32>,
    total_value: usize,
    o_prefix: String,
    post_norm_name: String,
}

// ─── gpu_linear_attention (local copy) ───────────────────────────────────

fn gpu_linear_attention(
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
    linear_idx: usize,
    use_fused_cmd1: bool,
    use_fusedwoods_cmd1: bool,
    prev_gpu_combined: bool,
) -> Option<LinearAttnGpuOut> {
    let use_gpu = gpu_wf.is_some() && ctx.is_some();

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

        let x_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let residual_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        unsafe {
            let dst = x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(normed.as_ptr(), dst, hidden_dim);
            let dst_r = residual_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(residual.as_ptr(), dst_r, hidden_dim);
        }

        let cmd_buf = c.queue.new_command_buffer();

        {
            let enc = cmd_buf.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, &prefix_std, &x_buf, 0, &c.batch_out[0], 0, qkv_dim, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_z, &x_buf, 0, &c.batch_out[1], 0, total_value, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_b, &x_buf, 0, &c.batch_out[2], 0, num_v_heads, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_a, &x_buf, 0, &c.batch_out[3], 0, num_v_heads, hidden_dim);
            enc.end_encoding();
        }

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

        {
            let enc = cmd_buf.new_compute_command_encoder();
            metal_kernels::encode_rms_norm_qk(c, &enc,
                c.buf_conv_output.as_ref().unwrap(), 0,
                c.buf_conv_output.as_ref().unwrap(), (total_key * 4) as u64,
                num_k_heads as u32, key_dim as u32, inv_scale);
            enc.end_encoding();
        }

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

        let gated_gpu = metal_buf_shared(&c.device, total_value * 4);
        {
            let gnw_ptr = wf.get_tensor_ptr(&format!("{}.norm.weight", prefix));
            let enc = cmd_buf.new_compute_command_encoder();
            if let Some(gnw_p) = gnw_ptr {
                let gnw_off = (gnw_p as usize - gw.base as usize) as u64;
                metal_kernels::encode_gated_rms_norm(c, &enc,
                    c.buf_delta_output.as_ref().unwrap(),
                    &c.batch_out[1],
                    &gw.buf, gnw_off,
                    &gated_gpu,
                    num_v_heads as u32, value_dim as u32);
            }
            enc.end_encoding();
        }

        let o_proj_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        {
            let enc = cmd_buf.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.out_proj", prefix),
                &gated_gpu, 0, &o_proj_buf, 0, hidden_dim, total_value);
            enc.end_encoding();
        }

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

        unsafe {
            std::ptr::copy_nonoverlapping(hidden_out.contents() as *const f32,
                hidden.as_mut_ptr(), hidden_dim);
        }

        let state_off = (CONV_KERNEL_SIZE - 2) * qkv_dim;
        state.conv_state.copy_within(qkv_dim.., 0);
        state.conv_state[state_off..state_off + qkv_dim].fill(0.0);

        return None;
    }

    // ── FusedWoods path: CMD1 without gated_norm/out_proj/residual ──
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

        let input_buf: Buffer;
        if prev_gpu_combined && c.buf_input.is_some() {
            input_buf = c.buf_input.as_ref().unwrap().clone();
        } else {
            let x = metal_buf_shared(&c.device, hidden_dim * 4);
            unsafe { std::ptr::copy_nonoverlapping(normed.as_ptr(), x.contents() as *mut f32, hidden_dim); }
            input_buf = x;
        }

        let cmd_buf = c.queue.new_command_buffer();

        {
            let enc = cmd_buf.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, &prefix_std, &input_buf, 0, &c.batch_out[0], 0, qkv_dim, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_z, &input_buf, 0, &c.batch_out[1], 0, total_value, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_b, &input_buf, 0, &c.batch_out[2], 0, num_v_heads, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_a, &input_buf, 0, &c.batch_out[3], 0, num_v_heads, hidden_dim);
            enc.end_encoding();
        }

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

        {
            let enc = cmd_buf.new_compute_command_encoder();
            metal_kernels::encode_rms_norm_qk(c, &enc,
                c.buf_conv_output.as_ref().unwrap(), 0,
                c.buf_conv_output.as_ref().unwrap(), (total_key * 4) as u64,
                num_k_heads as u32, key_dim as u32, inv_scale);
            enc.end_encoding();
        }

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

        let state_off = (CONV_KERNEL_SIZE - 2) * qkv_dim;
        state.conv_state.copy_within(qkv_dim.., 0);
        state.conv_state[state_off..state_off + qkv_dim].fill(0.0);

        return Some(LinearAttnGpuOut {
            gated_buf: c.batch_out[6].clone(),
            h_mid: residual,
            total_value,
            o_prefix: format!("{}.out_proj", prefix),
            post_norm_name: format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
        });
    }

    // ── Non-fused or CPU path ──
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

// ─── DeferredExperts (local copy, with FusedWoods extensions) ────────────

struct DeferredExperts {
    cmd_buf: Option<CommandBuffer>,
    out_buf: Option<Buffer>,
    _keep_alive: Vec<Buffer>,
    gpu_combined: bool,
}

impl DeferredExperts {
    fn complete(&mut self, hidden: &mut [f32], hidden_dim: usize) {
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

    fn complete_fast(&mut self, hidden: &mut [f32], hidden_dim: usize) {
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

// ─── moe_layer_forward (local copy, with FusedWoods lin_attn CMD2) ───────

fn moe_layer_forward(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    packed_fd: RawFd,
    ctx: Option<&MetalContext>,
    gpu_wf: Option<&WeightBuffer>,
    config: &ModelConfig,
    attn_state: Option<FullAttnGpuOut>,
    lin_attn: Option<LinearAttnGpuOut>,
    mut expert_gpu_buffer: Option<&mut ExpertBuffer>,
    gpu_combined: bool,
) -> Option<DeferredExperts> {
    let hidden_dim = config.hidden_dim;
    let num_experts = config.num_experts;
    let moe_inter = config.moe_intermediate;
    let shared_inter = config.shared_intermediate;
    let expert_size = config.expert_size_4bit;
    let layout = &config.expert_layout_4bit;
    let k = config.num_experts_per_tok;

    let use_gpu = ctx.is_some() && gpu_wf.is_some();

    let h_mid = hidden.to_vec();

    let post_norm_name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
    let mut h_post = vec![0.0f32; hidden_dim];

    let mut gate_scores = vec![0.0f32; num_experts];
    let mut shared_gate = vec![0.0f32; shared_inter];
    let mut shared_up = vec![0.0f32; shared_inter];
    let shared_gate_score;

    let prefix = format!("model.layers.{}.mlp", layer_idx);

    let mut sg_buf_gpu: Option<Buffer>;
    let mut su_buf_gpu: Option<Buffer>;
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

        gw.encode_matvec_into(wf, c, &enc, &attn.o_prefix, &attn.out_buf, 0, &o_proj_buf, 0, hidden_dim, attn.q_dim as usize);

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

        gw.encode_matvec_into(wf, c, &enc, &format!("{}.gate", prefix), &normed_buf, 0, &gate_buf, 0, num_experts, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.gate_proj", prefix), &normed_buf, 0, &sg_buf, 0, shared_inter, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.up_proj", prefix), &normed_buf, 0, &su_buf, 0, shared_inter, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert_gate", prefix), &normed_buf, 0, &sge_buf, 0, 1, hidden_dim);

        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe {
            std::ptr::copy_nonoverlapping(gate_buf.contents() as *const f32, gate_scores.as_mut_ptr(), num_experts);
            std::ptr::copy_nonoverlapping(sg_buf.contents() as *const f32, shared_gate.as_mut_ptr(), shared_inter);
            std::ptr::copy_nonoverlapping(su_buf.contents() as *const f32, shared_up.as_mut_ptr(), shared_inter);
            shared_gate_score = *(sge_buf.contents() as *const f32);
            std::ptr::copy_nonoverlapping(normed_buf.contents() as *const f32, hidden.as_mut_ptr(), hidden_dim);
        }

        sg_buf_gpu = Some(sg_buf);
        su_buf_gpu = Some(su_buf);

        h_post.copy_from_slice(hidden);
        hmid_gpu_override = Some(temp_buf);
    } else if lin_attn.is_some() && use_gpu
        && ctx.is_some()
        && ctx.unwrap().residual_add.is_some()
        && ctx.unwrap().rms_norm_apply_bf16.is_some()
    {
        // ── FusedWoods linear CMD2: out_proj + residual_add + rms_norm + gate + shared ──
        let c = ctx.unwrap();
        let gw = gpu_wf.unwrap();
        let la = lin_attn.unwrap();

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

        gw.encode_matvec_into(wf, c, &enc, &la.o_prefix, gated_buf, 0, &o_proj_buf, 0, hidden_dim, la.total_value);

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

        gw.encode_matvec_into(wf, c, &enc, &format!("{}.gate", prefix), &normed_buf, 0, &gate_buf, 0, num_experts, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.gate_proj", prefix), &normed_buf, 0, &sg_buf, 0, shared_inter, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.up_proj", prefix), &normed_buf, 0, &su_buf, 0, shared_inter, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert_gate", prefix), &normed_buf, 0, &sge_buf, 0, 1, hidden_dim);

        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe {
            std::ptr::copy_nonoverlapping(gate_buf.contents() as *const f32, gate_scores.as_mut_ptr(), num_experts);
            std::ptr::copy_nonoverlapping(sg_buf.contents() as *const f32, shared_gate.as_mut_ptr(), shared_inter);
            std::ptr::copy_nonoverlapping(su_buf.contents() as *const f32, shared_up.as_mut_ptr(), shared_inter);
            shared_gate_score = *(sge_buf.contents() as *const f32);
            std::ptr::copy_nonoverlapping(normed_buf.contents() as *const f32, hidden.as_mut_ptr(), hidden_dim);
        }

        sg_buf_gpu = Some(sg_buf);
        su_buf_gpu = Some(su_buf);
        h_post.copy_from_slice(hidden);
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
                shared_gate_score = *(sge_buf.contents() as *const f32);
            }
            sg_buf_gpu = Some(sg_buf);
            su_buf_gpu = Some(su_buf);
        } else {
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
            } else {
                shared_gate_score = 0.0;
            }
            sg_buf_gpu = None;
            su_buf_gpu = None;
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

            if let (Some(ref sg), Some(ref su)) = (sg_buf_gpu.as_ref(), su_buf_gpu.as_ref()) {
                metal_kernels::encode_swiglu(ctx, &enc, sg, 0, su, 0, &shared_act_gpu, 0, shared_inter as u32);
            }

            gw.encode_matvec_into(wf, ctx, &enc,
                &format!("{}.shared_expert.down_proj", prefix),
                &shared_act_gpu, 0, &shared_down_gpu, 0, hidden_dim, shared_inter);

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

            let do_gpu_norm = gpu_combined
                && layer_idx + 1 < config.num_layers
                && ctx.rms_norm_apply_bf16.is_some()
                && wf.get_tensor_ptr(&format!("model.layers.{}.input_layernorm.weight", layer_idx + 1)).is_some();
            if do_gpu_norm {
                let next_norm_ptr = wf.get_tensor_ptr(
                    &format!("model.layers.{}.input_layernorm.weight", layer_idx + 1)).unwrap();
                let next_norm_off = (next_norm_ptr as usize - gw.base as usize) as u64;
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
            if let Some(b) = sg_buf_gpu.take() { keep_alive.push(b); }
            if let Some(b) = su_buf_gpu.take() { keep_alive.push(b); }

            return Some(DeferredExperts {
                cmd_buf: Some(cmd_buf.to_owned()),
                out_buf: ctx.buf_moe_hidden.clone(),
                _keep_alive: keep_alive,
                gpu_combined,
            });
        }
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

            let gw = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.gate_w_off) as *const u32, layout.gate_w_size / 4) };
            let gs = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.gate_s_off) as *const u16, layout.gate_s_size / 2) };
            let gb = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.gate_b_off) as *const u16, layout.gate_b_size / 2) };
            dequant_matvec_4bit(gw, gs, gb, &h_post, &mut gate_tmp, moe_inter, hidden_dim, GROUP_SIZE);

            let uw = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.up_w_off) as *const u32, layout.up_w_size / 4) };
            let us = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.up_s_off) as *const u16, layout.up_s_size / 2) };
            let ub = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.up_b_off) as *const u16, layout.up_b_size / 2) };
            dequant_matvec_4bit(uw, us, ub, &h_post, &mut up_tmp, moe_inter, hidden_dim, GROUP_SIZE);

            for i in 0..moe_inter {
                let g = gate_tmp[i];
                let silu_g = g / (1.0 + (-g).exp());
                act_tmp[i] = silu_g * up_tmp[i];
            }

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

    for i in 0..shared_inter {
        let g = shared_gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        shared_act[i] = silu_g * shared_up[i];
    }

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

    for i in 0..hidden_dim {
        hidden[i] = h_mid[i] + moe_out[i] + shared_weight * shared_out[i];
    }

    None
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

// ─── GPU execution context (local) ───────────────────────────────────────

struct ExecCtxGpu<'a> {
    wf: &'a WeightFile,
    ctx: &'a MetalContext,
    gpu_wf: &'a WeightBuffer,
    config: &'a ModelConfig,
    expert_fds: &'a [RawFd],
    expert_gpu_buffer: Option<&'a mut ExpertBuffer>,
}

// ─── General-purpose token processing ────────────────────────────────────

fn process_token_inner(
    exec: &mut ExecCtxGpu<'_>,
    hidden: &mut [f32],
    pos: usize,
    kv: &mut [Option<FullAttnCache>],
    lin: &mut [Option<LinearAttnState>],
    check_signal: SignalCheckFn<'_>,
    capture_per_layer: bool,
    layer_outputs: &mut Vec<Vec<f32>>,
    use_fusedwoods: bool,
) -> Result<(), String> {
    let mut deferred: Option<DeferredExperts> = None;
    let hd = exec.config.hidden_dim;
    for layer in 0..exec.config.num_layers {
        if layer % 4 == 0 && check_signal() {
            return Err("interrupted".into());
        }
        let prev_gpu_combined = deferred.as_ref().map_or(false, |d| d.gpu_combined);
        if !prev_gpu_combined {
            if let Some(ref mut def) = deferred.take() {
                def.complete(hidden, hd);
            }
        }
        let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;
        let mut attn_state: Option<FullAttnGpuOut> = None;
        let mut lin_state: Option<LinearAttnGpuOut> = None;
        let mut h_mid_saved: Option<Vec<f32>> = None;
        if is_full {
            if prev_gpu_combined {
                if let Some(ref mut def) = deferred.take() {
                    def.complete_fast(hidden, hd);
                }
            }
            if let Some(ref mut kv) = kv[layer] {
                attn_state = mixed_full_attention_forward(
                    exec.wf, layer, hidden, kv, pos, exec.config,
                    Some(exec.gpu_wf), Some(exec.ctx));
            }
        } else if let Some(ref mut s) = lin[layer] {
            let li = layer - (layer + 1) / FULL_ATTN_INTERVAL;
            if use_fusedwoods && !prev_gpu_combined {
                h_mid_saved = Some(hidden.to_vec());
            }
            lin_state = gpu_linear_attention(
                exec.wf, layer, hidden, s,
                hd,
                exec.config.linear_num_k_heads, exec.config.linear_num_v_heads,
                exec.config.linear_total_key, exec.config.linear_total_value,
                exec.config.linear_conv_dim,
                Some(exec.gpu_wf), Some(exec.ctx), li,
                false, use_fusedwoods, prev_gpu_combined,
            );
            if prev_gpu_combined {
                if let Some(ref mut def) = deferred.take() {
                    def.complete_fast(hidden, hd);
                }
                if let Some(ref mut ls) = lin_state {
                    ls.h_mid.copy_from_slice(hidden);
                }
                h_mid_saved = Some(hidden.to_vec());
            }
            if let Some(ref hmid) = h_mid_saved {
                hidden.copy_from_slice(hmid);
            }
        }
        let r = moe_layer_forward(
            exec.wf, layer, hidden, exec.expert_fds[layer],
            Some(exec.ctx), Some(exec.gpu_wf), exec.config,
            attn_state, lin_state,
            exec.expert_gpu_buffer.as_mut().map(|x| &mut **x),
            use_fusedwoods,
        );
        deferred = r;
        if capture_per_layer {
            layer_outputs.push(hidden.to_vec());
        }
    }
    if let Some(ref mut def) = deferred {
        def.complete(hidden, hd);
    }
    Ok(())
}

// ─── EngineFusedWoods ─────────────────────────────────────────────────────

pub struct EngineFusedWoods<'a> {
    pub model: &'a Model,
    pub ctx: &'a MetalContext,
    pub gpu_wf: &'a WeightBuffer,
    pub expert_gpu_buffer: Option<&'a mut ExpertBuffer>,
}

impl<'a> EngineFusedWoods<'a> {
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

impl<'a> Engine for EngineFusedWoods<'a> {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        check_signal: SignalCheckFn<'_>,
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
            process_token_inner(
                &mut exec, &mut hidden,
                cache.pos, &mut cache.kv, &mut cache.lin,
                &mut || check_signal(), false, &mut Vec::new(),
                true,
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

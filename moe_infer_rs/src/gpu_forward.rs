/// GPU-accelerated MoE forward and linear attention (GatedDeltaNet).
///
/// Port of moe_forward, linear_attention_forward, and fused_layer_forward_debug
/// from moe_infer/core_src/layer_forward.h and attention.h.
use std::os::fd::RawFd;

use crate::config::ModelConfig;
use crate::error::MoEError;
use crate::kernels;
use crate::metal_context::{metal_buf_shared, MetalContext};
use crate::quant::{bf16_to_f32, cpu_dequant_matvec_4bit, cpu_rms_norm};
use crate::weights::WeightFile;

const RMS_NORM_EPS: f32 = 1e-6;
const GROUP_SIZE: usize = 64;
pub const LINEAR_KEY_DIM: usize = 128;
pub const LINEAR_VALUE_DIM: usize = 128;
const CONV_KERNEL_SIZE: usize = 4;

// ─── CPU helper functions ──────────────────────────────────────────────────

fn cpu_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn cpu_silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

fn cpu_softmax(x: &mut [f32]) {
    let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

fn cpu_topk(scores: &[f32], k: usize, indices: &mut [usize], values: &mut [f32]) {
    // Min-heap of K smallest
    for (i, &score) in scores.iter().enumerate() {
        if i < k {
            // Insert into heap
            let mut pos = i;
            while pos > 0 && values[(pos - 1) / 2] > score {
                values[pos] = values[(pos - 1) / 2];
                indices[pos] = indices[(pos - 1) / 2];
                pos = (pos - 1) / 2;
            }
            values[pos] = score;
            indices[pos] = i;
        } else if score > values[0] {
            values[0] = score;
            indices[0] = i;
            let mut pos = 0;
            loop {
                let left = 2 * pos + 1;
                let right = 2 * pos + 2;
                let mut smallest = pos;
                if left < k && values[left] < values[smallest] { smallest = left; }
                if right < k && values[right] < values[smallest] { smallest = right; }
                if smallest == pos { break; }
                values.swap(pos, smallest);
                indices.swap(pos, smallest);
                pos = smallest;
            }
        }
    }
}

fn cpu_normalize_weights(weights: &mut [f32]) {
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for w in weights.iter_mut() { *w *= inv; }
    }
}

fn cpu_rms_norm_bare(x: &[f32], out: &mut [f32], dim: usize, eps: f32) {
    let sum_sq: f32 = x[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms;
    }
}

fn cpu_rms_norm_gated(
    x: &[f32], z: &[f32], w_bf16: &[u16],
    out: &mut [f32], dim: usize, eps: f32,
) {
    let sum_sq: f32 = x[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        let w = bf16_to_f32(w_bf16[i]);
        let silu_z = z[i] / (1.0 + (-z[i]).exp());
        out[i] = x[i] * inv_rms * w * silu_z;
    }
}

fn cpu_conv1d_step(
    conv_state: &[f32],   // [(kernel_size-1) * channels]
    new_input: &[f32],    // [channels]
    weight_bf16: &[u16],  // [channels * kernel_size]
    out: &mut [f32],      // [channels]
    channels: usize,
    kernel_size: usize,
) {
    for c in 0..channels {
        let mut acc = 0.0f32;
        for k in 0..kernel_size - 1 {
            let w = bf16_to_f32(weight_bf16[c * kernel_size + k]);
            acc += conv_state[k * channels + c] * w;
        }
        let w = bf16_to_f32(weight_bf16[c * kernel_size + (kernel_size - 1)]);
        acc += new_input[c] * w;
        out[c] = acc;
    }
    cpu_silu(&mut out[..channels]);
}

// ─── Linear attention state ────────────────────────────────────────────────

pub struct LinearAttnState {
    /// Conv1d state: [(kernel_size-1) * qkv_dim] ring buffer
    pub conv_state: Vec<f32>,
    /// SSM state: [num_v_heads * value_dim * key_dim] — the S matrix per v-head
    pub ssm_state: Vec<f32>,
}

impl LinearAttnState {
    pub fn new(num_v_heads: usize, key_dim: usize, value_dim: usize, qkv_dim: usize) -> Self {
        LinearAttnState {
            conv_state: vec![0.0f32; (CONV_KERNEL_SIZE - 1) * qkv_dim],
            ssm_state: vec![0.0f32; num_v_heads * value_dim * key_dim],
        }
    }
}

// ─── Linear attention forward (GatedDeltaNet) ─────────────────────────────

/// Full linear attention forward (GatedDeltaNet) for single-token incremental inference.
/// Port of linear_attention_forward from attention.h:330-560.
pub fn linear_attention_forward(
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
) {
    // Input RMS norm
    let norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
    let nw = wf.get_tensor_u16(&norm_name);
    let mut normed = vec![0.0f32; hidden_dim];
    let mut residual = vec![0.0f32; hidden_dim];
    residual.copy_from_slice(hidden);

    if let Some(nw) = nw {
        let nw_f32: Vec<f32> = nw.iter().map(|&v| bf16_to_f32(v)).collect();
        cpu_rms_norm(hidden, &nw_f32, &mut normed, hidden_dim, RMS_NORM_EPS);
    } else {
        normed.copy_from_slice(hidden);
    }

    // Batch projections: QKV, Z, B, A
    let mut qkv = vec![0.0f32; qkv_dim];
    let mut z = vec![0.0f32; total_value];
    let mut beta = vec![0.0f32; num_v_heads];
    let mut alpha = vec![0.0f32; num_v_heads];

    let prefix = format!("model.layers.{}.linear_attn", layer_idx);

    if let (Some(qw), Some(qs), Some(qb)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_qkv.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_qkv.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_qkv.biases", prefix)),
    ) {
        cpu_dequant_matvec_4bit(qw, qs, qb, &normed, &mut qkv, qkv_dim, hidden_dim, GROUP_SIZE);
    }
    if let (Some(zw), Some(zs), Some(zb)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_z.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_z.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_z.biases", prefix)),
    ) {
        cpu_dequant_matvec_4bit(zw, zs, zb, &normed, &mut z, total_value, hidden_dim, GROUP_SIZE);
    }
    if let (Some(bw), Some(bs), Some(bb)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_b.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_b.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_b.biases", prefix)),
    ) {
        cpu_dequant_matvec_4bit(bw, bs, bb, &normed, &mut beta, num_v_heads, hidden_dim, GROUP_SIZE);
    }
    if let (Some(aw), Some(ass), Some(ab)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_a.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_a.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_a.biases", prefix)),
    ) {
        cpu_dequant_matvec_4bit(aw, ass, ab, &normed, &mut alpha, num_v_heads, hidden_dim, GROUP_SIZE);
    }

    // Conv1d step
    let mut conv_out = vec![0.0f32; qkv_dim];
    if let Some(conv_w) = wf.get_tensor_u16(&format!("{}.conv1d.weight", prefix)) {
        cpu_conv1d_step(
            &state.conv_state, &qkv, conv_w, &mut conv_out,
            qkv_dim, CONV_KERNEL_SIZE,
        );
    }

    // Update conv state: shift left, append new input
    let state_offset = (CONV_KERNEL_SIZE - 2) * qkv_dim;
    state.conv_state.copy_within(qkv_dim.., 0);
    state.conv_state[state_offset..state_offset + qkv_dim].copy_from_slice(&qkv);

    // Split conv_out into q, k, v
    let lin_q = &conv_out[..total_key];
    let lin_k = &conv_out[total_key..2 * total_key];
    let lin_v = &conv_out[2 * total_key..];

    // RMS norm q and k (bare, no weights) then scale
    let inv_scale = 1.0 / (LINEAR_KEY_DIM as f32).sqrt();
    let key_dim = LINEAR_KEY_DIM;

    let mut q_normed = vec![0.0f32; total_key];
    let mut k_normed = vec![0.0f32; total_key];
    for h in 0..num_k_heads {
        let qh = &lin_q[h * key_dim..(h + 1) * key_dim];
        let qh_out = &mut q_normed[h * key_dim..(h + 1) * key_dim];
        cpu_rms_norm_bare(qh, qh_out, key_dim, 1e-6);
        let q_scale = inv_scale * inv_scale;
        for d in qh_out.iter_mut() { *d *= q_scale; }
    }
    for h in 0..num_k_heads {
        let kh = &lin_k[h * key_dim..(h + 1) * key_dim];
        let kh_out = &mut k_normed[h * key_dim..(h + 1) * key_dim];
        cpu_rms_norm_bare(kh, kh_out, key_dim, 1e-6);
        for d in kh_out.iter_mut() { *d *= inv_scale; }
    }

    // Gated delta net recurrence
    let a_log = wf.get_tensor_f32(&format!("{}.A_log", prefix));
    let dt_bias = wf.get_tensor_u16(&format!("{}.dt_bias", prefix));

    let k_heads_per_v = num_v_heads / num_k_heads;
    let value_dim = LINEAR_VALUE_DIM;
    let mut out_values = vec![0.0f32; total_value];

    for vh in 0..num_v_heads {
        let kh = vh / k_heads_per_v;

        let a_val = a_log.map_or(1.0, |al| al[vh]);
        let dt_b = dt_bias.map_or(0.0, |db| bf16_to_f32(db[vh]));
        let softplus_val = (1.0 + (alpha[vh] + dt_b).exp()).ln();
        let g_decay = (-a_val.exp() * softplus_val).exp();

        let beta_gate = cpu_sigmoid(beta[vh]);

        let s_off = vh * value_dim * key_dim;
        let ssm = &mut state.ssm_state[s_off..s_off + value_dim * key_dim];
        let v_h = &lin_v[vh * value_dim..(vh + 1) * value_dim];
        let k_h = &k_normed[kh * key_dim..(kh + 1) * key_dim];

        // Step 1: Decay state
        for vi in 0..value_dim {
            for ki in 0..key_dim {
                ssm[vi * key_dim + ki] *= g_decay;
            }
        }

        // Step 2: Compute kv_mem, delta, update state
        for vi in 0..value_dim {
            let mut kv_mem = 0.0f32;
            for ki in 0..key_dim {
                kv_mem += ssm[vi * key_dim + ki] * k_h[ki];
            }
            let delta = (v_h[vi] - kv_mem) * beta_gate;
            for ki in 0..key_dim {
                ssm[vi * key_dim + ki] += k_h[ki] * delta;
            }
        }

        // Step 3: Output
        let q_h = &q_normed[kh * key_dim..(kh + 1) * key_dim];
        let o_h = &mut out_values[vh * value_dim..(vh + 1) * value_dim];
        for vi in 0..value_dim {
            let mut sum = 0.0f32;
            for ki in 0..key_dim {
                sum += ssm[vi * key_dim + ki] * q_h[ki];
            }
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
            cpu_rms_norm_gated(oh, zh, gnw, gh, value_dim, RMS_NORM_EPS);
        }
    } else {
        gated_out.copy_from_slice(&out_values);
    }

    // Output projection
    if let (Some(ow), Some(os), Some(ob)) = (
        wf.get_tensor_u32(&format!("{}.out_proj.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.out_proj.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.out_proj.biases", prefix)),
    ) {
        let mut attn_out = vec![0.0f32; hidden_dim];
        cpu_dequant_matvec_4bit(ow, os, ob, &gated_out, &mut attn_out, hidden_dim, total_value, GROUP_SIZE);
        // Residual add
        for i in 0..hidden_dim {
            hidden[i] = residual[i] + attn_out[i];
        }
    }
}

// ─── MoE layer forward ─────────────────────────────────────────────────────

/// Run the full MoE block for a single layer: routing, shared expert, K routed experts, combine.
///
/// Port of moe_forward from layer_forward.h:298-503.
///
/// When `ctx` and `packed_fd` are provided, runs expert matvecs on GPU;
/// otherwise falls back to CPU.
pub fn moe_layer_forward(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    packed_fd: RawFd,
    ctx: Option<&MetalContext>,
    config: &ModelConfig,
) -> Result<(), MoEError> {
    let hidden_dim = config.hidden_dim;
    let num_experts = config.num_experts;
    let moe_inter = config.moe_intermediate;
    let shared_inter = config.shared_intermediate;
    let expert_size = config.expert_size_4bit;
    let layout = &config.expert_layout_4bit;
    let k = config.num_experts_per_tok;

    // Save h_mid (residual) and apply post-attention RMS norm → h_post
    let h_mid = hidden.to_vec();

    let post_norm_name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
    let pnw = wf.get_tensor_u16(&post_norm_name);
    let mut h_post = vec![0.0f32; hidden_dim];
    if let Some(pnw) = pnw {
        let pnw_f32: Vec<f32> = pnw.iter().map(|&v| bf16_to_f32(v)).collect();
        cpu_rms_norm(hidden, &pnw_f32, &mut h_post, hidden_dim, RMS_NORM_EPS);
    } else {
        h_post.copy_from_slice(hidden);
    }

    // ── Router gate + shared expert projections ──
    let mut gate_scores = vec![0.0f32; num_experts];
    let mut shared_gate = vec![0.0f32; shared_inter];
    let mut shared_up = vec![0.0f32; shared_inter];
    let mut shared_gate_score = 0.0f32;

    let prefix = format!("model.layers.{}.mlp", layer_idx);

    // Router gate: hidden_dim → num_experts
    if let (Some(gw), Some(gs), Some(gb)) = (
        wf.get_tensor_u32(&format!("{}.gate.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.gate.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.gate.biases", prefix)),
    ) {
        cpu_dequant_matvec_4bit(gw, gs, gb, &h_post, &mut gate_scores, num_experts, hidden_dim, GROUP_SIZE);
    }

    // Shared expert gate_proj: hidden_dim → shared_intermediate
    if let (Some(sgw), Some(sgs), Some(sgb)) = (
        wf.get_tensor_u32(&format!("{}.shared_expert.gate_proj.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.shared_expert.gate_proj.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.shared_expert.gate_proj.biases", prefix)),
    ) {
        cpu_dequant_matvec_4bit(sgw, sgs, sgb, &h_post, &mut shared_gate, shared_inter, hidden_dim, GROUP_SIZE);
    }

    // Shared expert up_proj: hidden_dim → shared_intermediate
    if let (Some(suw), Some(sus), Some(sub)) = (
        wf.get_tensor_u32(&format!("{}.shared_expert.up_proj.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.shared_expert.up_proj.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.shared_expert.up_proj.biases", prefix)),
    ) {
        cpu_dequant_matvec_4bit(suw, sus, sub, &h_post, &mut shared_up, shared_inter, hidden_dim, GROUP_SIZE);
    }

    // Shared expert gate: hidden_dim → 1 (scalar sigmoid)
    if let (Some(segw), Some(segs), Some(segb)) = (
        wf.get_tensor_u32(&format!("{}.shared_expert_gate.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.shared_expert_gate.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.shared_expert_gate.biases", prefix)),
    ) {
        let mut tmp = [0.0f32];
        cpu_dequant_matvec_4bit(segw, segs, segb, &h_post, &mut tmp, 1, hidden_dim, GROUP_SIZE);
        shared_gate_score = tmp[0];
    }

    // ── Routing: softmax + topk ──
    cpu_softmax(&mut gate_scores);

    let mut expert_indices = vec![0usize; k];
    let mut expert_weights = vec![0.0f32; k];
    cpu_topk(&gate_scores, k, &mut expert_indices, &mut expert_weights);
    cpu_normalize_weights(&mut expert_weights);

    // ── Routed expert computation ──
    let mut moe_out = vec![0.0f32; hidden_dim];

    if let Some(ctx) = ctx {
        // GPU path
        let expert_buf = metal_buf_shared(&ctx.device, expert_size);
        let gate_out = metal_buf_shared(&ctx.device, moe_inter * 4);
        let up_out = metal_buf_shared(&ctx.device, moe_inter * 4);
        let act_out = metal_buf_shared(&ctx.device, moe_inter * 4);
        let expert_out = metal_buf_shared(&ctx.device, hidden_dim * 4);

        // Upload h_post to GPU
        let x_buf = metal_buf_shared(&ctx.device, hidden_dim * 4);
        unsafe {
            let dst = x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(h_post.as_ptr(), dst, hidden_dim);
        }

        let hidden_u32 = hidden_dim as u32;
        let inter_u32 = moe_inter as u32;
        let gs_u32 = GROUP_SIZE as u32;

        for (&eidx, &ew) in expert_indices.iter().zip(expert_weights.iter()) {
            let expert_offset = (eidx as i64) * (expert_size as i64);

            // Single pread for the entire expert
            let nread = unsafe {
                let ptr = expert_buf.contents() as *mut u8;
                let slice = std::slice::from_raw_parts_mut(ptr, expert_size);
                libc::pread(packed_fd, slice.as_mut_ptr() as *mut std::ffi::c_void, expert_size, expert_offset)
            };
            if nread != expert_size as isize {
                continue;
            }

            let cmd_buf = ctx.queue.new_command_buffer();

            // gate_proj
            {
                let encoder = cmd_buf.new_compute_command_encoder();
                kernels::encode_matvec_offset(
                    ctx, encoder,
                    &expert_buf, layout.gate_w_off as u64,
                    &expert_buf, layout.gate_s_off as u64,
                    &expert_buf, layout.gate_b_off as u64,
                    &x_buf, 0, &gate_out, 0,
                    inter_u32, hidden_u32, gs_u32, 3,
                );
                encoder.end_encoding();
            }

            // up_proj
            {
                let encoder = cmd_buf.new_compute_command_encoder();
                kernels::encode_matvec_offset(
                    ctx, encoder,
                    &expert_buf, layout.up_w_off as u64,
                    &expert_buf, layout.up_s_off as u64,
                    &expert_buf, layout.up_b_off as u64,
                    &x_buf, 0, &up_out, 0,
                    inter_u32, hidden_u32, gs_u32, 3,
                );
                encoder.end_encoding();
            }

            // SwiGLU
            {
                let encoder = cmd_buf.new_compute_command_encoder();
                kernels::encode_swiglu(ctx, encoder, &gate_out, 0, &up_out, 0, &act_out, 0, inter_u32);
                encoder.end_encoding();
            }

            // down_proj
            {
                let encoder = cmd_buf.new_compute_command_encoder();
                kernels::encode_matvec_offset(
                    ctx, encoder,
                    &expert_buf, layout.down_w_off as u64,
                    &expert_buf, layout.down_s_off as u64,
                    &expert_buf, layout.down_b_off as u64,
                    &act_out, 0, &expert_out, 0,
                    hidden_u32, inter_u32, gs_u32, 3,
                );
                encoder.end_encoding();
            }

            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            // Accumulate into moe_out
            unsafe {
                let eout = expert_out.contents() as *const f32;
                for d in 0..hidden_dim {
                    moe_out[d] += (*eout.add(d)) * ew;
                }
            }
        }
    } else {
        // CPU fallback path
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
            cpu_dequant_matvec_4bit(gw, gs, gb, &h_post, &mut gate_tmp, moe_inter, hidden_dim, GROUP_SIZE);

            // up_proj
            let uw = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.up_w_off) as *const u32, layout.up_w_size / 4) };
            let us = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.up_s_off) as *const u16, layout.up_s_size / 2) };
            let ub = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.up_b_off) as *const u16, layout.up_b_size / 2) };
            cpu_dequant_matvec_4bit(uw, us, ub, &h_post, &mut up_tmp, moe_inter, hidden_dim, GROUP_SIZE);

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
            cpu_dequant_matvec_4bit(dw, ds, db, &act_tmp, &mut eout, hidden_dim, moe_inter, GROUP_SIZE);

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
    if let (Some(sdw), Some(sds), Some(sdb)) = (
        wf.get_tensor_u32(&format!("{}.shared_expert.down_proj.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.shared_expert.down_proj.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.shared_expert.down_proj.biases", prefix)),
    ) {
        cpu_dequant_matvec_4bit(sdw, sds, sdb, &shared_act, &mut shared_out, hidden_dim, shared_inter, GROUP_SIZE);
    }

    let shared_weight = cpu_sigmoid(shared_gate_score);

    // ── Final combine: hidden = h_mid + moe_out + shared_weight * shared_out ──
    for i in 0..hidden_dim {
        hidden[i] = h_mid[i] + moe_out[i] + shared_weight * shared_out[i];
    }

    Ok(())
}

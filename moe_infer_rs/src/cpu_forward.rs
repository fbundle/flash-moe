// CPU-only decomposed forward pass — platform-agnostic, no GPU dependency.
//
// Each step is a pure function with explicit inputs and outputs, designed
// for inspection, testing, and deployment on any platform.  Compose them
// sequentially to run a full layer, or swap individual steps for custom
// behaviour.

use crate::kernels::*;
use crate::types::*;

// Thread-local GPU context for attention projections.
// Set to Some(&MetalCtx) by the GPU forward path before calling step functions,
// cleared to None by the CPU forward path.  dequant_proj checks this to
// optionally dispatch to GPU for bit-identical results with C's pipeline.
use std::cell::Cell;
thread_local! {
    static TL_GPU_CTX: Cell<usize> = const { Cell::new(0) };
}

/// Set the thread-local GPU context pointer. Call before running GPU forward.
pub(crate) fn set_tl_gpu_ctx(ctx: Option<&crate::metal::MetalCtx>) {
    TL_GPU_CTX.with(|c| c.set(match ctx {
        Some(ctx) => ctx as *const crate::metal::MetalCtx as usize,
        None => 0,
    }));
}

fn tl_gpu_ctx() -> Option<*const crate::metal::MetalCtx> {
    TL_GPU_CTX.with(|c| {
        let p = c.get();
        if p == 0 { None } else { Some(p as *const crate::metal::MetalCtx) }
    })
}

// ============================================================================
// Step 1 — Input RMS normalisation
// ============================================================================

/// RMS-normalise `hidden` into `out`, using `norm_w` (bf16 weights).
/// If `norm_w` is `None`, copies `hidden` to `out` unchanged.
pub fn step_input_norm(
    hidden: &[f32],
    norm_w: Option<*const u16>,
    out: &mut [f32],
    dim: usize,
    eps: f32,
) {
    if let Some(nw) = norm_w {
        let w = unsafe { std::slice::from_raw_parts(nw, dim) };
        cpu_rms_norm(hidden, w, out, dim, eps);
    } else {
        out[..dim].copy_from_slice(&hidden[..dim]);
    }
}

// ============================================================================
// Step 2 — Full scaled-dot-product attention
// ============================================================================

/// Run full attention for one layer on a CPU.
///
/// Weights are passed as raw pointers from the mapped weight file; see
/// `LayerWeightCache` for how these are resolved.  `wf_data` is the base
/// address of the flat weight buffer.
///
/// Returns via `attn_proj` (hidden_dim floats) — the caller should later
/// apply `residual + attn_proj` to `hidden`.
#[allow(clippy::too_many_arguments)]
pub fn step_full_attention(
    cfg: &ModelConfig,
    _wf_data: *const u8,
    q_w: *const u32, q_s: *const u16, q_b: *const u16,
    k_w: *const u32, k_s: *const u16, k_b: *const u16,
    v_w: *const u32, v_s: *const u16, v_b: *const u16,
    o_w: *const u32, o_s: *const u16, o_b: *const u16,
    q_norm_w: Option<*const u16>,
    k_norm_w: Option<*const u16>,
    normed: &[f32],
    kv: &mut KVCache,
    pos: i32,
    attn_proj: &mut [f32],
    // Scratch (caller allocates once, reuses across calls):
    q_proj_out: &mut [f32],
    q_buf: &mut [f32],
    q_gate: &mut [f32],
    k_buf: &mut [f32],
    v_buf: &mut [f32],
    attn_out: &mut [f32],
) {
    let hd = cfg.hidden_dim as usize;
    let heads = cfg.num_attn_heads as usize;
    let kv_heads = cfg.num_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_proj_dim = heads * head_dim * 2;
    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    let gs = cfg.group_size as usize;

    // ---- Q / K / V projections ----
    dequant_proj(q_w, q_s, q_b, normed, q_proj_out, q_proj_dim, hd, gs);
    dequant_proj(k_w, k_s, k_b, normed, k_buf, kv_dim, hd, gs);
    dequant_proj(v_w, v_s, v_b, normed, v_buf, kv_dim, hd, gs);

    // Split q_proj_out into q and q_gate
    for h in 0..heads {
        let src = h * 2 * head_dim;
        q_buf[h * head_dim..(h + 1) * head_dim]
            .copy_from_slice(&q_proj_out[src..src + head_dim]);
        q_gate[h * head_dim..(h + 1) * head_dim]
            .copy_from_slice(&q_proj_out[src + head_dim..src + 2 * head_dim]);
    }

    // ---- Q/K RMS norm ----
    rms_norm_heads(q_buf, heads, head_dim, q_norm_w, cfg.rms_norm_eps);
    rms_norm_heads(k_buf, kv_heads, head_dim, k_norm_w, cfg.rms_norm_eps);

    // ---- Rotary embeddings ----
    crate::gpu_ops::apply_rotary_emb(
        q_buf, k_buf, pos, heads, kv_heads, head_dim,
        cfg.rotary_dim as usize, cfg.rope_theta,
    );

    // ---- KV cache update ----
    let cache_pos = kv.len as usize;
    for i in 0..kv_dim {
        let bits = k_buf[i].to_bits();
        kv.k_cache[cache_pos * kv_dim + i] = (bits >> 16) as u16;
        let vbits = v_buf[i].to_bits();
        kv.v_cache[cache_pos * kv_dim + i] = (vbits >> 16) as u16;
    }
    kv.len += 1;

    // ---- Scaled dot-product attention ----
    attn_out[..q_dim].fill(0.0);
    let heads_per_kv = heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; kv.len as usize];

    for h in 0..heads {
        let kv_h = h / heads_per_kv;
        let qh = &q_buf[h * head_dim..(h + 1) * head_dim];

        for p in 0..kv.len as usize {
            let kp = &kv.k_cache[p * kv_dim + kv_h * head_dim..][..head_dim];
            let dot: f32 = qh.iter().enumerate()
                .map(|(d, &qd)| qd * kv_elem_to_f32(kp[d]))
                .sum();
            scores[p] = dot * scale;
        }
        cpu_softmax(&mut scores, kv.len as usize);

        let oh = &mut attn_out[h * head_dim..(h + 1) * head_dim];
        for p in 0..kv.len as usize {
            let vp = &kv.v_cache[p * kv_dim + kv_h * head_dim..][..head_dim];
            let sp = scores[p];
            for d in 0..head_dim {
                oh[d] += sp * kv_elem_to_f32(vp[d]);
            }
        }
    }

    // ---- Sigmoid gate ----
    for i in 0..q_dim {
        attn_out[i] *= cpu_sigmoid(q_gate[i]);
    }

    // ---- Output projection ----
    attn_proj[..hd].fill(0.0);
    dequant_proj(o_w, o_s, o_b, attn_out, attn_proj, hd, q_dim, gs);
}

// ============================================================================
// Step 3 — Linear (Gated Delta Net) attention
// ============================================================================

#[allow(clippy::too_many_arguments)]
pub fn step_linear_attention(
    cfg: &ModelConfig,
    _wf_data: *const u8,
    qkv_w: *const u32, qkv_s: *const u16, qkv_b: *const u16,
    z_w:   *const u32, z_s:   *const u16, z_b:   *const u16,
    b_w:   *const u32, b_s:   *const u16, b_b:   *const u16,
    a_w:   *const u32, a_s:   *const u16, a_b:   *const u16,
    conv1d_w: Option<*const u16>,
    a_log: Option<*const f32>,
    dt_bias: Option<*const u16>,
    gated_norm_w: Option<*const u16>,
    out_proj_w: *const u32, out_proj_s: *const u16, out_proj_b: *const u16,
    normed: &[f32],
    state: &mut LinearAttnState,
    attn_proj: &mut [f32],
    // Scratch:
    qkv_out: &mut [f32],
    z_out: &mut [f32],
    beta_out: &mut [f32],
    alpha_out: &mut [f32],
    conv_out: &mut [f32],
    out_values: &mut [f32],
    gated_out: &mut [f32],
) {
    let hd = cfg.hidden_dim as usize;
    let qkv_dim = cfg.linear_conv_dim as usize;
    let z_dim = cfg.linear_total_value as usize;
    let num_vh = cfg.linear_num_v_heads as usize;
    let num_kh = cfg.linear_num_k_heads as usize;
    let key_dim = cfg.linear_key_dim as usize;
    let val_dim = cfg.linear_value_dim as usize;
    let gs = cfg.group_size as usize;

    // ---- QKV / Z / B / A projections ----
    dequant_proj(qkv_w, qkv_s, qkv_b, normed, qkv_out, qkv_dim, hd, gs);
    dequant_proj(z_w, z_s, z_b, normed, z_out, z_dim, hd, gs);
    dequant_proj(b_w, b_s, b_b, normed, beta_out, num_vh, hd, gs);
    dequant_proj(a_w, a_s, a_b, normed, alpha_out, num_vh, hd, gs);

    // ---- Conv1d step ----
    conv_out[..qkv_dim].fill(0.0);
    if let Some(cw) = conv1d_w {
        let cw_slice = unsafe {
            std::slice::from_raw_parts(cw, qkv_dim * cfg.conv_kernel_size as usize)
        };
        cpu_conv1d_step(&state.conv_state, qkv_out, cw_slice, conv_out,
            qkv_dim, cfg.conv_kernel_size as usize);
    }
    // Update conv state: shift left, append new input
    let state_skip = qkv_dim;
    let state_len = state.conv_state.len();
    if state_len > state_skip {
        state.conv_state.copy_within(state_skip.., 0);
        let dst = state_len - state_skip;
        state.conv_state[dst..].copy_from_slice(qkv_out);
    }

    // ---- Split conv_out into q, k, v — operate in-place ----
    let total_key = cfg.linear_total_key as usize;
    let (q_part, rest) = conv_out.split_at_mut(total_key);
    let (k_part, v_part) = rest.split_at_mut(total_key);

    // ---- RMS norm q and k (in-place) ----
    let inv_scale = 1.0 / (key_dim as f32).sqrt();
    for h in 0..num_kh {
        let qh = &mut q_part[h * key_dim..(h + 1) * key_dim];
        let sq: f32 = qh.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (sq / key_dim as f32 + 1e-6).sqrt();
        let q_scale = inv_scale * inv_scale;
        for d in 0..key_dim { qh[d] = qh[d] * inv_rms * q_scale; }
    }
    for h in 0..num_kh {
        let kh = &mut k_part[h * key_dim..(h + 1) * key_dim];
        let sq: f32 = kh.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (sq / key_dim as f32 + 1e-6).sqrt();
        for d in 0..key_dim { kh[d] *= inv_rms * inv_scale; }
    }
    // Reborrow immutably for recurrence
    let lin_q: &[f32] = q_part;
    let lin_k: &[f32] = k_part;
    let lin_v: &[f32] = v_part;

    // ---- Precompute decay and beta gate ----
    let k_heads_per_v = num_vh / num_kh;
    let mut g_decay = vec![0.0f32; num_vh];
    let mut beta_gate = vec![0.0f32; num_vh];
    for vh in 0..num_vh {
        let a_val = alpha_out[vh];
        let dt_b = dt_bias.map_or(0.0, |o| {
            bf16_to_f32(unsafe { *o.add(vh) })
        });
        let a_log_val = a_log.map_or(0.0f32, |o| unsafe { *o.add(vh) }); // default: exp(0)=1.0 matches C's default A_val=1.0
        let softplus = (1.0 + (a_val + dt_b).exp()).ln();
        g_decay[vh] = (-a_log_val.exp() * softplus).exp();
        beta_gate[vh] = cpu_sigmoid(beta_out[vh]);
    }

    // ---- Gated delta net recurrence ----
    out_values[..z_dim].fill(0.0);
    for vh in 0..num_vh {
        let kh = vh / k_heads_per_v;
        let g = g_decay[vh];
        let b_gate = beta_gate[vh];
        let s_off = vh * val_dim * key_dim;
        let v_off = vh * val_dim;
        let k_off = kh * key_dim;

        // Decay state
        for vi in 0..val_dim {
            for ki in 0..key_dim {
                state.ssm_state[s_off + vi * key_dim + ki] *= g;
            }
        }
        // Update
        for vi in 0..val_dim {
            let kv_mem: f32 = (0..key_dim)
                .map(|ki| state.ssm_state[s_off + vi * key_dim + ki] * lin_k[k_off + ki])
                .sum();
            let delta = (lin_v[v_off + vi] - kv_mem) * b_gate;
            for ki in 0..key_dim {
                state.ssm_state[s_off + vi * key_dim + ki] += lin_k[k_off + ki] * delta;
            }
        }
        // Output
        for vi in 0..val_dim {
            let sum: f32 = (0..key_dim)
                .map(|ki| state.ssm_state[s_off + vi * key_dim + ki] * lin_q[k_off + ki])
                .sum();
            out_values[v_off + vi] = sum;
        }
    }
    // ---- RMSNormGated ----
    gated_out[..z_dim].fill(0.0);
    for vh in 0..num_vh {
        let oh = &out_values[vh * val_dim..(vh + 1) * val_dim];
        let zh = &z_out[vh * val_dim..(vh + 1) * val_dim];
        let gh = &mut gated_out[vh * val_dim..(vh + 1) * val_dim];
        if let Some(gnw) = gated_norm_w {
            let gnw_s = unsafe { std::slice::from_raw_parts(gnw, val_dim) };
            let sum_sq: f32 = oh.iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / val_dim as f32 + cfg.rms_norm_eps).sqrt();
            for i in 0..val_dim {
                let w = bf16_to_f32(gnw_s[i]);
                let silu_z = zh[i] / (1.0 + (-zh[i]).exp());
                gh[i] = oh[i] * inv_rms * w * silu_z;
            }
        } else {
            gh.copy_from_slice(oh);
        }
    }
    // ---- Output projection ----
    attn_proj[..hd].fill(0.0);
    dequant_proj(out_proj_w, out_proj_s, out_proj_b, gated_out, attn_proj, hd, z_dim, gs);
}

// ============================================================================
// CPU attention compute (between projections and o_proj) — for GPU pipeline
// ============================================================================

/// CPU full attention compute after GPU projections.  Takes Q/K/V projection
/// outputs (already computed by GPU CMD1), runs RMS-norm / RoPE / KV-cache /
/// dot-product attention / sigmoid gate on CPU.  Writes attn_out ready for o_proj.
#[allow(clippy::too_many_arguments)]
pub fn cpu_full_attn_compute(
    cfg: &ModelConfig,
    q_proj_out: &[f32],   // [heads * head_dim * 2]
    k_out: &mut [f32],     // [kv_heads * head_dim] — mutated in-place by RoPE
    v_out: &[f32],         // [kv_heads * head_dim]
    q_norm_w: Option<*const u16>,
    k_norm_w: Option<*const u16>,
    kv: &mut KVCache,
    pos: i32,
    attn_out: &mut [f32],  // [heads * head_dim]
    // Scratch:
    q_buf: &mut [f32],
    q_gate: &mut [f32],
) {
    let heads = cfg.num_attn_heads as usize;
    let kv_heads = cfg.num_kv_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;

    // Split q_proj_out into q and q_gate
    for h in 0..heads {
        let src = h * 2 * head_dim;
        q_buf[h * head_dim..(h + 1) * head_dim]
            .copy_from_slice(&q_proj_out[src..src + head_dim]);
        q_gate[h * head_dim..(h + 1) * head_dim]
            .copy_from_slice(&q_proj_out[src + head_dim..src + 2 * head_dim]);
    }

    // Q/K RMS norm (k_out mutated in-place)
    rms_norm_heads(q_buf, heads, head_dim, q_norm_w, cfg.rms_norm_eps);
    rms_norm_heads(k_out, kv_heads, head_dim, k_norm_w, cfg.rms_norm_eps);

    // RoPE (modifies q_buf and k_out in-place)
    crate::gpu_ops::apply_rotary_emb(
        q_buf, k_out, pos, heads, kv_heads, head_dim,
        cfg.rotary_dim as usize, cfg.rope_theta,
    );

    // KV cache update
    let cache_pos = kv.len as usize;
    for i in 0..kv_dim {
        let bits = k_out[i].to_bits();
        kv.k_cache[cache_pos * kv_dim + i] = (bits >> 16) as u16;
        let vbits = v_out[i].to_bits();
        kv.v_cache[cache_pos * kv_dim + i] = (vbits >> 16) as u16;
    }
    kv.len += 1;

    // Scaled dot-product attention (GQA)
    attn_out[..q_dim].fill(0.0);
    let heads_per_kv = heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; kv.len as usize];

    for h in 0..heads {
        let kv_h = h / heads_per_kv;
        let qh = &q_buf[h * head_dim..(h + 1) * head_dim];

        for p in 0..kv.len as usize {
            let kp = &kv.k_cache[p * kv_dim + kv_h * head_dim..][..head_dim];
            let dot: f32 = qh.iter().enumerate()
                .map(|(d, &qd)| qd * kv_elem_to_f32(kp[d]))
                .sum();
            scores[p] = dot * scale;
        }
        cpu_softmax(&mut scores, kv.len as usize);

        let oh = &mut attn_out[h * head_dim..(h + 1) * head_dim];
        for p in 0..kv.len as usize {
            let vp = &kv.v_cache[p * kv_dim + kv_h * head_dim..][..head_dim];
            let sp = scores[p];
            for d in 0..head_dim {
                oh[d] += sp * kv_elem_to_f32(vp[d]);
            }
        }
    }

    // Sigmoid gate
    for i in 0..q_dim {
        attn_out[i] *= cpu_sigmoid(q_gate[i]);
    }
}

/// CPU linear attention compute after GPU projections.  Takes QKV/Z/B/A/Alpha
/// projection outputs (already computed by GPU CMD1), runs conv1d + delta-net +
/// gated RMS-norm on CPU.  Writes gated_out ready for out_proj.
#[allow(clippy::too_many_arguments)]
pub fn cpu_linear_attn_compute(
    cfg: &ModelConfig,
    qkv_out: &[f32],       // [linear_conv_dim]
    z_out: &[f32],          // [linear_total_value]
    beta_out: &[f32],       // [linear_num_v_heads]
    alpha_out: &[f32],      // [linear_num_v_heads]
    conv1d_w: Option<*const u16>,
    a_log: Option<*const f32>,
    dt_bias: Option<*const u16>,
    gated_norm_w: Option<*const u16>,
    state: &mut LinearAttnState,
    gated_out: &mut [f32],  // [linear_total_value]
    // Scratch:
    conv_out_buf: &mut [f32],
    out_values: &mut [f32],
) {
    let qkv_dim = cfg.linear_conv_dim as usize;
    let z_dim = cfg.linear_total_value as usize;
    let num_vh = cfg.linear_num_v_heads as usize;
    let num_kh = cfg.linear_num_k_heads as usize;
    let key_dim = cfg.linear_key_dim as usize;
    let val_dim = cfg.linear_value_dim as usize;

    // Conv1d step
    conv_out_buf[..qkv_dim].fill(0.0);
    if let Some(cw) = conv1d_w {
        let cw_slice = unsafe {
            std::slice::from_raw_parts(cw, qkv_dim * cfg.conv_kernel_size as usize)
        };
        cpu_conv1d_step(&state.conv_state, qkv_out, cw_slice, conv_out_buf,
            qkv_dim, cfg.conv_kernel_size as usize);
    }
    // Update conv state
    let state_skip = qkv_dim;
    let state_len = state.conv_state.len();
    if state_len > state_skip {
        state.conv_state.copy_within(state_skip.., 0);
        let dst = state_len - state_skip;
        state.conv_state[dst..].copy_from_slice(qkv_out);
    }

    // Split conv_out into q, k, v — operate in-place on conv_out_buf
    let total_key = cfg.linear_total_key as usize;
    let (q_part, rest) = conv_out_buf.split_at_mut(total_key);
    let (k_part, v_part) = rest.split_at_mut(total_key);

    // RMS norm q and k (in-place)
    let inv_scale = 1.0 / (key_dim as f32).sqrt();
    for h in 0..num_kh {
        let qh = &mut q_part[h * key_dim..(h + 1) * key_dim];
        let sq: f32 = qh.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (sq / key_dim as f32 + 1e-6).sqrt();
        let q_scale = inv_scale * inv_scale;
        for d in 0..key_dim { qh[d] = qh[d] * inv_rms * q_scale; }
    }
    for h in 0..num_kh {
        let kh = &mut k_part[h * key_dim..(h + 1) * key_dim];
        let sq: f32 = kh.iter().map(|x| x * x).sum();
        let inv_rms = 1.0 / (sq / key_dim as f32 + 1e-6).sqrt();
        for d in 0..key_dim { kh[d] *= inv_rms * inv_scale; }
    }
    // Reborrow immutably for recurrence
    let lin_q: &[f32] = q_part;
    let lin_k: &[f32] = k_part;
    let lin_v: &[f32] = v_part;

    // Precompute decay and beta gate
    let k_heads_per_v = num_vh / num_kh;
    let mut g_decay = vec![0.0f32; num_vh];
    let mut beta_gate = vec![0.0f32; num_vh];
    for vh in 0..num_vh {
        let a_val = alpha_out[vh];
        let dt_b = dt_bias.map_or(0.0, |o| {
            bf16_to_f32(unsafe { *o.add(vh) })
        });
        let a_log_val = a_log.map_or(0.0f32, |o| unsafe { *o.add(vh) }); // default: exp(0)=1.0 matches C's default A_val=1.0
        let softplus = (1.0 + (a_val + dt_b).exp()).ln();
        g_decay[vh] = (-a_log_val.exp() * softplus).exp();
        beta_gate[vh] = cpu_sigmoid(beta_out[vh]);
    }

    // Gated delta net recurrence
    out_values[..z_dim].fill(0.0);
    for vh in 0..num_vh {
        let kh = vh / k_heads_per_v;
        let g = g_decay[vh];
        let b_gate = beta_gate[vh];
        let s_off = vh * val_dim * key_dim;
        let v_off = vh * val_dim;
        let k_off = kh * key_dim;

        for vi in 0..val_dim {
            for ki in 0..key_dim {
                state.ssm_state[s_off + vi * key_dim + ki] *= g;
            }
        }
        for vi in 0..val_dim {
            let kv_mem: f32 = (0..key_dim)
                .map(|ki| state.ssm_state[s_off + vi * key_dim + ki] * lin_k[k_off + ki])
                .sum();
            let delta = (lin_v[v_off + vi] - kv_mem) * b_gate;
            for ki in 0..key_dim {
                state.ssm_state[s_off + vi * key_dim + ki] += lin_k[k_off + ki] * delta;
            }
        }
        for vi in 0..val_dim {
            let sum: f32 = (0..key_dim)
                .map(|ki| state.ssm_state[s_off + vi * key_dim + ki] * lin_q[k_off + ki])
                .sum();
            out_values[v_off + vi] = sum;
        }
    }

    // RMSNormGated
    gated_out[..z_dim].fill(0.0);
    for vh in 0..num_vh {
        let oh = &out_values[vh * val_dim..(vh + 1) * val_dim];
        let zh = &z_out[vh * val_dim..(vh + 1) * val_dim];
        let gh = &mut gated_out[vh * val_dim..(vh + 1) * val_dim];
        if let Some(gnw) = gated_norm_w {
            let gnw_s = unsafe { std::slice::from_raw_parts(gnw, val_dim) };
            let sum_sq: f32 = oh.iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / val_dim as f32 + cfg.rms_norm_eps).sqrt();
            for i in 0..val_dim {
                let w = bf16_to_f32(gnw_s[i]);
                let silu_z = zh[i] / (1.0 + (-zh[i]).exp());
                gh[i] = oh[i] * inv_rms * w * silu_z;
            }
        } else {
            gh.copy_from_slice(oh);
        }
    }

}

// ============================================================================
// Step 4 — MoE routing: gate projection + softmax + top-K
// ============================================================================

/// Result of MoE routing: indices of selected experts, their normalised
/// weights, and the shared expert gate score.
pub struct RoutingResult {
    pub expert_indices: Vec<i32>,
    pub expert_weights: Vec<f32>,   // normalised so they sum to 1
    pub shared_gate_score: f32,
}

/// Project the hidden state through the routing gate, softmax, then select
/// the top-K experts.  Also computes the shared-expert gate score.
pub fn step_moe_routing(
    cfg: &ModelConfig,
    gate_w: *const u32, gate_s: *const u16, gate_b: *const u16,
    seg_w: Option<*const u32>, seg_s: Option<*const u16>, seg_b: Option<*const u16>,
    h_post: &[f32],
    gate_scores: &mut [f32], // [num_experts]
) -> RoutingResult {
    let hd = cfg.hidden_dim as usize;
    let gs = cfg.group_size as usize;
    let ne = cfg.num_experts as usize;

    // Gate projection -> expert logits
    gate_scores[..ne].fill(0.0);
    dequant_proj(gate_w, gate_s, gate_b, h_post, gate_scores, ne, hd, gs);

    // Shared expert gate score (scalar)
    let mut sg_score = 0.0f32;
    if let (Some(segw), Some(segs), Some(segb)) = (seg_w, seg_s, seg_b) {
        let mut tmp = [0.0f32; 1];
        dequant_proj(segw, segs, segb, h_post, &mut tmp, 1, hd, gs);
        sg_score = tmp[0];
    }

    // Softmax + top-K
    cpu_softmax(gate_scores, ne);
    let k_count = cfg.num_experts_per_tok as usize;
    let (expert_indices, expert_weights) = cpu_topk(gate_scores, ne, k_count);
    let mut normed = expert_weights.clone();
    cpu_normalize_weights(&mut normed, k_count);

    RoutingResult {
        expert_indices,
        expert_weights: normed,
        shared_gate_score: sg_score,
    }
}

// ============================================================================
// Step 5 — Single CPU expert forward (gate + up + SwiGLU + down)
// ============================================================================

/// Run one 4-bit MoE expert on CPU.
///
/// `expert_data` is a byte slice containing the expert's packed weights
/// (sized per `active_expert_size`).  `h_post` is the post-attention-norm
/// hidden state.  Result is written into `expert_out` (hidden_dim floats).
pub fn step_cpu_expert(
    cfg: &ModelConfig,
    expert_data: &[u8],
    h_post: &[f32],
    expert_out: &mut [f32],
    use_2bit: bool,
) {
    let layout = if use_2bit { &cfg.layout_2bit } else { &cfg.layout_4bit };
    let mi = cfg.moe_intermediate as usize;
    let hd = cfg.hidden_dim as usize;
    let gs = cfg.group_size as usize;

    let gw  = unsafe { expert_data.as_ptr().add(layout.gate_w_off as usize) as *const u32 };
    let gs_p = unsafe { expert_data.as_ptr().add(layout.gate_s_off as usize) as *const u16 };
    let gb_p = unsafe { expert_data.as_ptr().add(layout.gate_b_off as usize) as *const u16 };
    let uw  = unsafe { expert_data.as_ptr().add(layout.up_w_off as usize) as *const u32 };
    let us_p = unsafe { expert_data.as_ptr().add(layout.up_s_off as usize) as *const u16 };
    let ub_p = unsafe { expert_data.as_ptr().add(layout.up_b_off as usize) as *const u16 };
    let dw  = unsafe { expert_data.as_ptr().add(layout.down_w_off as usize) as *const u32 };
    let ds_p = unsafe { expert_data.as_ptr().add(layout.down_s_off as usize) as *const u16 };
    let db_p = unsafe { expert_data.as_ptr().add(layout.down_b_off as usize) as *const u16 };

    let packed_cols = hd / 8;
    let num_groups = hd / gs;

    let mut gate_proj = vec![0.0f32; mi];
    let mut up_proj   = vec![0.0f32; mi];
    let mut act       = vec![0.0f32; mi];

    let gw_slice = unsafe { std::slice::from_raw_parts(gw, mi * packed_cols) };
    let gs_slice = unsafe { std::slice::from_raw_parts(gs_p, mi * num_groups) };
    let gb_slice = unsafe { std::slice::from_raw_parts(gb_p, mi * num_groups) };
    let uw_slice = unsafe { std::slice::from_raw_parts(uw, mi * packed_cols) };
    let us_slice = unsafe { std::slice::from_raw_parts(us_p, mi * num_groups) };
    let ub_slice = unsafe { std::slice::from_raw_parts(ub_p, mi * num_groups) };
    let dw_slice = unsafe { std::slice::from_raw_parts(dw, hd * (mi / 8)) };
    let ds_slice = unsafe { std::slice::from_raw_parts(ds_p, hd * (mi / gs)) };
    let db_slice = unsafe { std::slice::from_raw_parts(db_p, hd * (mi / gs)) };

    cpu_dequant_matvec(gw_slice, gs_slice, gb_slice, h_post, &mut gate_proj, mi, hd, gs);
    cpu_dequant_matvec(uw_slice, us_slice, ub_slice, h_post, &mut up_proj, mi, hd, gs);
    cpu_swiglu(&gate_proj, &up_proj, &mut act, mi);
    cpu_dequant_matvec(dw_slice, ds_slice, db_slice, &act, expert_out, hd, mi, gs);
}

// ============================================================================
// Step 6 — Shared expert forward
// ============================================================================

/// Run the shared expert (gate + up + SwiGLU + down) for one layer.
/// Writes `shared_gate_score * down_proj(swiglu(gate, up))` into `shared_out`.
pub fn step_shared_expert(
    cfg: &ModelConfig,
    sg_w: *const u32, sg_s: *const u16, sg_b: *const u16,
    su_w: *const u32, su_s: *const u16, su_b: *const u16,
    sd_w: *const u32, sd_s: *const u16, sd_b: *const u16,
    h_post: &[f32],
    shared_gate_score: f32,
    shared_out: &mut [f32],
) {
    let hd = cfg.hidden_dim as usize;
    let gs = cfg.group_size as usize;
    let si = cfg.shared_intermediate as usize;

    let mut gate = vec![0.0f32; si];
    let mut up   = vec![0.0f32; si];
    let mut act  = vec![0.0f32; si];

    dequant_proj(sg_w, sg_s, sg_b, h_post, &mut gate, si, hd, gs);
    dequant_proj(su_w, su_s, su_b, h_post, &mut up, si, hd, gs);
    cpu_swiglu(&gate, &up, &mut act, si);

    shared_out[..hd].fill(0.0);
    dequant_proj(sd_w, sd_s, sd_b, &act, shared_out, hd, si, gs);

    let sw = cpu_sigmoid(shared_gate_score);
    for i in 0..hd {
        shared_out[i] *= sw;
    }
}

// ============================================================================
// Step 7 — Final combine
// ============================================================================

/// Combine residual, MoE output, and shared expert output into `hidden`.
/// `hidden[i] = h_mid[i] + moe_out[i] + shared_out[i]`
pub fn step_final_combine(
    h_mid: &[f32],
    moe_out: &[f32],
    shared_out: &[f32],
    hidden: &mut [f32],
    dim: usize,
) {
    for i in 0..dim {
        hidden[i] = h_mid[i] + moe_out[i] + shared_out[i];
    }
}

// ============================================================================
// Convenience: run all steps for one layer (CPU-only, decomposed)
// ============================================================================

/// Run the full decomposed CPU forward pass for one layer on one token.
///
/// This calls each of the step_* functions in order, so you can inspect
/// intermediate results or swap individual steps.  All scratch buffers are
/// provided by the caller for zero-allocation reuse.
#[allow(clippy::too_many_arguments)]
pub fn cpu_layer_forward_decomposed(
    cfg: &ModelConfig,
    wf_data: *const u8,
    // Norms
    input_norm_w: Option<*const u16>,
    post_attn_norm_w: Option<*const u16>,
    // Full attention weights
    q_w: *const u32, q_s: *const u16, q_b: *const u16,
    k_w: *const u32, k_s: *const u16, k_b: *const u16,
    v_w: *const u32, v_s: *const u16, v_b: *const u16,
    o_w: *const u32, o_s: *const u16, o_b: *const u16,
    q_norm_w: Option<*const u16>, k_norm_w: Option<*const u16>,
    // Linear attention weights
    qkv_w: *const u32, qkv_s: *const u16, qkv_b: *const u16,
    z_w:   *const u32, z_s:   *const u16, z_b:   *const u16,
    b_w:   *const u32, b_s:   *const u16, b_b:   *const u16,
    a_w:   *const u32, a_s:   *const u16, a_b:   *const u16,
    conv1d_w: Option<*const u16>,
    a_log: Option<*const f32>, dt_bias: Option<*const u16>,
    gated_norm_w: Option<*const u16>,
    out_proj_w: *const u32, out_proj_s: *const u16, out_proj_b: *const u16,
    // MoE weights
    gate_w: *const u32, gate_s: *const u16, gate_b: *const u16,
    sg_w: *const u32, sg_s: *const u16, sg_b: *const u16,
    su_w: *const u32, su_s: *const u16, su_b: *const u16,
    sd_w: *const u32, sd_s: *const u16, sd_b: *const u16,
    seg_w: Option<*const u32>, seg_s: Option<*const u16>, seg_b: Option<*const u16>,
    // State
    is_full: bool,
    mut kv: Option<&mut KVCache>,
    mut la_state: Option<&mut LinearAttnState>,
    pos: i32,
    // Expert data
    expert_data_for_slot: &[Option<&[u8]>], // pre-read expert blobs [0..K]
    expert_weights: &[f32],                 // normalised weights [0..K]
    actual_k: usize,
    use_2bit: bool,
    // Hidden state (mutated in place)
    hidden: &mut [f32],
    // Scratch
    scratch: &mut CpuForwardScratch,
) {
    let hd = cfg.hidden_dim as usize;

    // -- save residual --
    scratch.residual[..hd].copy_from_slice(hidden);

    // -- step 1: input norm --
    step_input_norm(hidden, input_norm_w, &mut scratch.normed, hd, cfg.rms_norm_eps);

    // -- step 2 or 3: attention --
    if is_full {
        step_full_attention(
            cfg, wf_data,
            q_w, q_s, q_b, k_w, k_s, k_b, v_w, v_s, v_b,
            o_w, o_s, o_b, q_norm_w, k_norm_w,
            &scratch.normed,
            kv.as_deref_mut().unwrap(),
            pos,
            &mut scratch.attn_proj,
            &mut scratch.q_proj_out, &mut scratch.q_buf, &mut scratch.q_gate,
            &mut scratch.k_buf, &mut scratch.v_buf, &mut scratch.attn_out,
        );
    } else {
        step_linear_attention(
            cfg, wf_data,
            qkv_w, qkv_s, qkv_b,
            z_w, z_s, z_b,
            b_w, b_s, b_b,
            a_w, a_s, a_b,
            conv1d_w, a_log, dt_bias, gated_norm_w,
            out_proj_w, out_proj_s, out_proj_b,
            &scratch.normed,
            la_state.as_deref_mut().unwrap(),
            &mut scratch.attn_proj,
            &mut scratch.qkv_out, &mut scratch.z_out,
            &mut scratch.beta_out, &mut scratch.alpha_out,
            &mut scratch.conv_out, &mut scratch.out_values, &mut scratch.gated_out,
        );
    }

    // -- apply residual: hidden = residual + attn_proj --
    for i in 0..hd {
        hidden[i] = scratch.residual[i] + scratch.attn_proj[i];
    }

    // h_mid snapshot (for final combine)
    scratch.h_mid[..hd].copy_from_slice(hidden);

    // -- step 1b: post-attention norm --
    step_input_norm(hidden, post_attn_norm_w, &mut scratch.h_post, hd, cfg.rms_norm_eps);

    // -- step 4: MoE routing --
    let routing = step_moe_routing(
        cfg,
        gate_w, gate_s, gate_b,
        seg_w, seg_s, seg_b,
        &scratch.h_post,
        &mut scratch.gate_scores,
    );

    // -- step 5: routed experts --
    scratch.moe_out[..hd].fill(0.0);
    for k in 0..actual_k {
        if let Some(edata) = expert_data_for_slot[k] {
            step_cpu_expert(cfg, edata, &scratch.h_post, &mut scratch.expert_tmp, use_2bit);
            cpu_vec_madd(&mut scratch.moe_out, &scratch.expert_tmp, expert_weights[k], hd);
        }
    }

    // -- step 6: shared expert --
    scratch.shared_out[..hd].fill(0.0);
    step_shared_expert(
        cfg,
        sg_w, sg_s, sg_b,
        su_w, su_s, su_b,
        sd_w, sd_s, sd_b,
        &scratch.h_post,
        routing.shared_gate_score,
        &mut scratch.shared_out,
    );

    // -- step 7: final combine --
    step_final_combine(&scratch.h_mid, &scratch.moe_out, &scratch.shared_out, hidden, hd);
}

// ============================================================================
// Scratch buffer — reusable across layers and tokens
// ============================================================================

/// All scratch buffers needed by the decomposed CPU forward pass.
/// Allocate once, reuse across all layers and tokens.
pub struct CpuForwardScratch {
    pub normed: Vec<f32>,
    pub residual: Vec<f32>,
    pub attn_proj: Vec<f32>,
    pub h_post: Vec<f32>,
    pub h_mid: Vec<f32>,
    pub gate_scores: Vec<f32>,
    pub moe_out: Vec<f32>,
    pub shared_out: Vec<f32>,
    pub expert_tmp: Vec<f32>,
    // Full attention
    pub q_proj_out: Vec<f32>,
    pub k_buf: Vec<f32>,
    pub v_buf: Vec<f32>,
    pub q_buf: Vec<f32>,
    pub q_gate: Vec<f32>,
    pub attn_out: Vec<f32>,
    // Linear attention
    pub qkv_out: Vec<f32>,
    pub z_out: Vec<f32>,
    pub beta_out: Vec<f32>,
    pub alpha_out: Vec<f32>,
    pub conv_out: Vec<f32>,
    pub out_values: Vec<f32>,
    pub gated_out: Vec<f32>,
    // GPU delta-net scratch
    pub g_decay: Vec<f32>,
    pub beta_gate: Vec<f32>,
}

impl CpuForwardScratch {
    pub fn new(cfg: &ModelConfig) -> Self {
        let hd = cfg.hidden_dim as usize;
        let q_proj_dim = cfg.num_attn_heads as usize * cfg.head_dim as usize * 2;
        let q_dim = cfg.num_attn_heads as usize * cfg.head_dim as usize;
        let kv_dim = cfg.num_kv_heads as usize * cfg.head_dim as usize;
        let qkv_dim = cfg.linear_conv_dim as usize;
        let z_dim = cfg.linear_total_value as usize;
        let vh = cfg.linear_num_v_heads as usize;

        Self {
            normed: vec![0.0; hd],
            residual: vec![0.0; hd],
            attn_proj: vec![0.0; hd],
            h_post: vec![0.0; hd],
            h_mid: vec![0.0; hd],
            gate_scores: vec![0.0; cfg.num_experts as usize],
            moe_out: vec![0.0; hd],
            shared_out: vec![0.0; hd],
            expert_tmp: vec![0.0; hd],
            q_proj_out: vec![0.0; q_proj_dim],
            k_buf: vec![0.0; kv_dim],
            v_buf: vec![0.0; kv_dim],
            q_buf: vec![0.0; q_dim],
            q_gate: vec![0.0; q_dim],
            attn_out: vec![0.0; q_dim],
            qkv_out: vec![0.0; qkv_dim],
            z_out: vec![0.0; z_dim],
            beta_out: vec![0.0; vh],
            alpha_out: vec![0.0; vh],
            conv_out: vec![0.0; qkv_dim],
            out_values: vec![0.0; z_dim],
            gated_out: vec![0.0; z_dim],
            g_decay: vec![0.0; vh],
            beta_gate: vec![0.0; vh],
        }
    }
}

// ============================================================================
// Convenience: full layer forward calling the decomposed steps
// ============================================================================

use crate::gpu_forward::{LayerWeightCache, active_expert_size};
use std::fs::File;
use std::os::unix::fs::FileExt;

/// Full CPU layer forward.
///
/// Calls the `step_*` functions in order.  Uses `LayerWeightCache` for
/// pre-computed weight offsets and `packed_fd` for expert I/O.
/// This is the main entry point for the CPU-only path.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cpu_layer_forward(
    cfg: &ModelConfig,
    wf_data: *const u8,
    lc: &LayerWeightCache,
    hidden: &mut [f32],
    mut kv: Option<&mut KVCache>,
    mut la_state: Option<&mut LinearAttnState>,
    pos: i32,
    packed_fd: Option<&File>,
    use_2bit: bool,
    scratch: &mut CpuForwardScratch,
) {
    // Ensure GPU dequant is disabled (CPU-only path)
    set_tl_gpu_ctx(None);

    let hd = cfg.hidden_dim as usize;
    let is_full = kv.is_some();

    // Helper: offset → pointer (null fallback for required weights)
    let u32p = |o: Option<usize>| o.map_or(std::ptr::null(), |x| unsafe { wf_data.add(x) } as *const u32);
    let u16p = |o: Option<usize>| o.map_or(std::ptr::null(), |x| unsafe { wf_data.add(x) } as *const u16);
    let _f32p = |o: Option<usize>| o.map_or(std::ptr::null(), |x| unsafe { wf_data.add(x) } as *const f32);
    // Helper: offset → Option<pointer> (for optional weights)
    let u16o = |o: Option<usize>| o.map(|x| unsafe { wf_data.add(x) } as *const u16);
    let f32o = |o: Option<usize>| o.map(|x| unsafe { wf_data.add(x) } as *const f32);
    let u32o = |o: Option<usize>| o.map(|x| unsafe { wf_data.add(x) } as *const u32);

    // -- save residual --
    scratch.residual[..hd].copy_from_slice(hidden);

    // -- step 1: input norm --
    step_input_norm(hidden, u16o(lc.input_norm_w), &mut scratch.normed, hd, cfg.rms_norm_eps);

    // -- step 2 or 3: attention --
    if is_full {
        step_full_attention(
            cfg, wf_data,
            u32p(lc.q_w), u16p(lc.q_s), u16p(lc.q_b),
            u32p(lc.k_w), u16p(lc.k_s), u16p(lc.k_b),
            u32p(lc.v_w), u16p(lc.v_s), u16p(lc.v_b),
            u32p(lc.o_w), u16p(lc.o_s), u16p(lc.o_b),
            u16o(lc.q_norm_w), u16o(lc.k_norm_w),
            &scratch.normed, kv.as_deref_mut().unwrap(), pos,
            &mut scratch.attn_proj,
            &mut scratch.q_proj_out, &mut scratch.q_buf, &mut scratch.q_gate,
            &mut scratch.k_buf, &mut scratch.v_buf, &mut scratch.attn_out,
        );
    } else {
        step_linear_attention(
            cfg, wf_data,
            u32p(lc.qkv_w), u16p(lc.qkv_s), u16p(lc.qkv_b),
            u32p(lc.z_w), u16p(lc.z_s), u16p(lc.z_b),
            u32p(lc.b_w), u16p(lc.b_s), u16p(lc.b_b),
            u32p(lc.a_w), u16p(lc.a_s), u16p(lc.a_b),
            u16o(lc.conv1d_w), f32o(lc.a_log), u16o(lc.dt_bias),
            u16o(lc.gated_norm_w),
            u32p(lc.out_proj_w), u16p(lc.out_proj_s), u16p(lc.out_proj_b),
            &scratch.normed, la_state.as_deref_mut().unwrap(),
            &mut scratch.attn_proj,
            &mut scratch.qkv_out, &mut scratch.z_out,
            &mut scratch.beta_out, &mut scratch.alpha_out,
            &mut scratch.conv_out, &mut scratch.out_values, &mut scratch.gated_out,
        );
    }

    // -- apply residual: hidden = residual + attn_proj --
    for i in 0..hd {
        hidden[i] = scratch.residual[i] + scratch.attn_proj[i];
    }

    // h_mid snapshot for final combine
    scratch.h_mid[..hd].copy_from_slice(hidden);

    // -- step 1b: post-attention norm --
    step_input_norm(hidden, u16o(lc.post_attn_norm_w), &mut scratch.h_post, hd, cfg.rms_norm_eps);

    // -- step 4: MoE routing --
    scratch.gate_scores[..cfg.num_experts as usize].fill(0.0);
    let routing = step_moe_routing(
        cfg,
        u32p(lc.gate_w), u16p(lc.gate_s), u16p(lc.gate_b),
        u32o(lc.seg_w), u16o(lc.seg_s), u16o(lc.seg_b),
        &scratch.h_post,
        &mut scratch.gate_scores,
    );

    let actual_k = routing.expert_indices.len().min(crate::constants::MAX_K);

    // -- step 5: read expert data + expert forward --
    scratch.moe_out[..hd].fill(0.0);
    let esz = active_expert_size(cfg, use_2bit);
    if let Some(fd) = packed_fd {
        let mut buf = vec![0u8; esz];
        for k in 0..actual_k {
            let offset = (routing.expert_indices[k] as usize * esz) as u64;
            if fd.read_exact_at(&mut buf, offset).is_ok() {
                step_cpu_expert(cfg, &buf, &scratch.h_post, &mut scratch.expert_tmp, use_2bit);
                cpu_vec_madd(&mut scratch.moe_out, &scratch.expert_tmp, routing.expert_weights[k], hd);
            }
        }
    }

    // -- step 6: shared expert --
    scratch.shared_out[..hd].fill(0.0);
    step_shared_expert(
        cfg,
        u32p(lc.sg_w), u16p(lc.sg_s), u16p(lc.sg_b),
        u32p(lc.su_w), u16p(lc.su_s), u16p(lc.su_b),
        u32p(lc.sd_w), u16p(lc.sd_s), u16p(lc.sd_b),
        &scratch.h_post,
        routing.shared_gate_score,
        &mut scratch.shared_out,
    );

    // -- step 7: final combine --
    step_final_combine(&scratch.h_mid, &scratch.moe_out, &scratch.shared_out, hidden, hd);
}

// ============================================================================
// Internal helpers
// ============================================================================

/// bf16 cache element → f32
fn kv_elem_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// Dequant projection: out = W @ x  (4-bit packed, bf16 scales/biases)
/// Uses GPU dequant (same Metal shader as C) when the GPU path is active,
/// otherwise falls back to CPU dequant.
pub(crate) fn dequant_proj(
    w: *const u32, s: *const u16, b: *const u16,
    x: &[f32], out: &mut [f32],
    out_dim: usize, in_dim: usize, group_size: usize,
) {
    if w.is_null() || s.is_null() || b.is_null() {
        return;
    }
    if let Some(gpu_ptr) = tl_gpu_ctx() {
        let ctx = unsafe { &*gpu_ptr };
        if ctx.wf_buf.is_some() {
            unsafe {
                crate::gpu_ops::fast_dequant_matvec(ctx, w, s, b, x, out, out_dim, in_dim, group_size);
            }
            return;
        }
    }
    let num_groups = in_dim / group_size;
    let packed_cols = in_dim / 8;
    let w_slice = unsafe { std::slice::from_raw_parts(w, out_dim * packed_cols) };
    let s_slice = unsafe { std::slice::from_raw_parts(s, out_dim * num_groups) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, out_dim * num_groups) };
    cpu_dequant_matvec(w_slice, s_slice, b_slice, x, out, out_dim, in_dim, group_size);
}

/// Apply per-head RMS norm with optional weights.
fn rms_norm_heads(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    weight: Option<*const u16>,
    eps: f32,
) {
    if let Some(w) = weight {
        let w_slice = unsafe { std::slice::from_raw_parts(w, head_dim) };
        for h in 0..num_heads {
            let head = &mut data[h * head_dim..(h + 1) * head_dim];
            let sum_sq: f32 = head.iter().map(|x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / head_dim as f32 + eps).sqrt();
            for i in 0..head_dim {
                head[i] = head[i] * inv_rms * bf16_to_f32(w_slice[i]);
            }
        }
    }
}

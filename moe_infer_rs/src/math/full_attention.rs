use metal::Buffer;

use crate::constants::{GROUP_SIZE, MAX_SEQ, RMS_NORM_EPS};
use crate::cache::FullAttnCache;
use crate::metal_context::{metal_buf_shared, WeightBuffer, MetalContext};
use crate::model::config::ModelConfig;
use crate::model::weights::WeightFile;

use crate::math::{apply_rope, bf16_to_f32, dequant_matvec_4bit, rms_norm};

// ─── GPU state passed from full-attention forward to MoE for CMD2 fusion ──

pub struct FullAttnGpuOut {
    pub q_buf: Buffer,
    pub q_gate_buf: Buffer,
    pub kc_buf: Buffer,
    pub vc_buf: Buffer,
    pub scores_buf: Buffer,
    pub out_buf: Buffer,
    pub hidden_buf: Buffer,
    pub seq_len: u32,
    pub seq_stride: u32,
    pub num_attn_heads: u32,
    pub head_dim: u32,
    pub kv_dim: u32,
    pub heads_per_kv: u32,
    pub scale: f32,
    pub q_dim: u32,
    pub o_prefix: String,
}

// ─── Full attention forward ───────────────────────────────────────────────

/// Single-token full (self) attention forward: QKV proj, Q/K norms, RoPE,
/// KV cache append.
///
/// Returns `FullAttnGpuOut` for GPU-fused CMD2 (batched attn + o_proj + gate).
/// When GPU is available, skips batched attention/o_proj/residual — those are
/// deferred to `moe_layer_forward`'s CMD2. When GPU unavailable, computes
/// everything on CPU/separate CMDs and returns None.
pub fn mixed_full_attention_forward(
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

    // GPU batched attention (scores + softmax + values + sigmoid gate)
    let heads_per_kv = num_attn_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let seq_len = kv.len;
    let seq_stride = MAX_SEQ;
    // o_proj output (filled by GPU fused path or CPU fallback below)
    let mut o_out = vec![0.0f32; hidden_dim];

    let use_gpu_attn = ctx.is_some()
        && gpu_wf.is_some()
        && ctx.unwrap().attn_scores_batched.is_some()
        && ctx.unwrap().attn_softmax_batched.is_some()
        && ctx.unwrap().attn_values_batched.is_some();

    if use_gpu_attn {
        let c = ctx.unwrap();
        // Upload Q, K_cache, V_cache, Q_gate, hidden → returned for CMD2 fusion
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
    // CPU fallback — no GPU attention available, do everything on CPU
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

        // o_proj
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

/// CPU wrapper — calls mixed_full_attention_forward without GPU resources.
pub fn full_attention_forward(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    kv: &mut FullAttnCache,
    pos: usize,
    config: &ModelConfig,
) -> Option<FullAttnGpuOut> {
    mixed_full_attention_forward(wf, layer_idx, hidden, kv, pos, config, None, None)
}

/// GPU wrapper — calls mixed_full_attention_forward with GPU resources.
pub fn gpu_full_attention_forward(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    kv: &mut FullAttnCache,
    pos: usize,
    config: &ModelConfig,
    gpu_wf: &WeightBuffer,
    ctx: &MetalContext,
) -> Option<FullAttnGpuOut> {
    mixed_full_attention_forward(wf, layer_idx, hidden, kv, pos, config, Some(gpu_wf), Some(ctx))
}

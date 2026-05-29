/// MTP (Multi-Token Prediction) forward pass for Qwen3.6.
///
/// Reference: vendor/llama.cpp/src/models/qwen35moe.cpp graph_mtp
///
/// The MTP block receives the main model's pre-norm hidden state `h` and a
/// token embedding, concatenates them after separate RMS norms, projects
/// through `eh_proj`, then runs a full-attention decoder block with its own
/// KV cache and MoE FFN, followed by a (possibly shared) output norm + lm_head.

use metal::Buffer;

use crate::constants::{MAX_SEQ, RMS_NORM_EPS};
use crate::engine::metal_context::{metal_buf_shared, MetalContext, WeightBuffer, ExpertBuffer, MAX_K};
use crate::engine::metal_kernels;
use crate::math::{bf16_to_f32, normalize_weights, softmax, topk};
use crate::model::weights::WeightFile;
use crate::model::expert::ExpertFile;

/// Holds persistent GPU state for the MTP block's KV cache.
pub struct MtpState {
    /// K cache [MAX_SEQ * kv_dim] f32
    pub k_cache: Buffer,
    /// V cache [MAX_SEQ * kv_dim] f32
    pub v_cache: Buffer,
    /// Current position in KV cache
    pub pos: usize,
}

impl MtpState {
    pub fn new(device: &metal::Device, kv_dim: usize) -> Self {
        let cache_size = MAX_SEQ * kv_dim * 4; // f32 = 4 bytes
        MtpState {
            k_cache: metal_buf_shared(device, cache_size),
            v_cache: metal_buf_shared(device, cache_size),
            pos: 0,
        }
    }

    /// Reset KV cache position (e.g., after verification accept/reject).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Roll back KV cache position (after partial accept).
    pub fn rollback(&mut self, new_pos: usize) {
        self.pos = new_pos;
    }
}

// ─── MTP forward ───────────────────────────────────────────────────────────────

/// Run one MTP forward step: (h, token_id) → logits.
///
/// `h` is the pre-norm hidden state from the main model's last transformer
/// layer (before final norm).  `token_id` is the previously predicted token.
///
/// Returns logits [vocab_size] and updates the MTP KV cache position.
pub fn mtp_step(
    wf: &WeightFile,
    weight_buffer: &WeightBuffer,
    expert_buffer: &ExpertBuffer,
    mtp_state: &mut MtpState,
    ctx: &MetalContext,
    h: &[f32],
    token_id: usize,
    hidden_dim: usize,
    num_attn_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_inter: usize,
    shared_inter: usize,
    vocab_size: usize,
    expert_file: &ExpertFile,
) -> Vec<f32> {
    let pos = mtp_state.pos;
    let kv_dim = num_kv_heads * head_dim;
    let q_dim = num_attn_heads * head_dim;
    let device = &ctx.device;
    let queue = &ctx.queue;

    // ── 1. Embed token (CPU) ──────────────────────────────────────────────
    let tok_embd = embed_token_cpu(wf, token_id, hidden_dim);

    // ── 2. RMS norm: hnorm (h) and enorm (tok_embd) ───────────────────────
    let hnorm_w = wf.get_tensor_u16("mtp.pre_fc_norm_hidden.weight")
        .map(|w| w.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<f32>>())
        .unwrap_or_else(|| vec![1.0f32; hidden_dim]);
    let enorm_w = wf.get_tensor_u16("mtp.pre_fc_norm_embedding.weight")
        .map(|w| w.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<f32>>())
        .unwrap_or_else(|| vec![1.0f32; hidden_dim]);

    let h_normed = rms_norm(h, &hnorm_w, RMS_NORM_EPS);
    let e_normed = rms_norm(&tok_embd, &enorm_w, RMS_NORM_EPS);

    // ── 3. Concat + fc (eh_proj) on GPU ───────────────────────────────────
    let mut concat = vec![0.0f32; hidden_dim * 2];
    concat[..hidden_dim].copy_from_slice(&e_normed);
    concat[hidden_dim..].copy_from_slice(&h_normed);

    let fc_out = gpu_matvec(device, queue, weight_buffer, wf, ctx,
        "mtp.fc", &concat, hidden_dim, hidden_dim * 2);
    let mut cur = fc_out;

    // ── 4. Standard decoder block ─────────────────────────────────────────

    // 4a. attn_norm
    let attn_norm_w = wf.get_tensor_u16("mtp.layers.0.input_layernorm.weight")
        .map(|w| w.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<f32>>())
        .unwrap_or_else(|| vec![1.0f32; hidden_dim]);
    let attn_normed = rms_norm(&cur, &attn_norm_w, RMS_NORM_EPS);

    // 4b. Q, K, V projections
    let q = gpu_matvec(device, queue, weight_buffer, wf, ctx,
        "mtp.layers.0.self_attn.q_proj", &attn_normed, q_dim * 2, hidden_dim);
    let k = gpu_matvec(device, queue, weight_buffer, wf, ctx,
        "mtp.layers.0.self_attn.k_proj", &attn_normed, kv_dim, hidden_dim);
    let v = gpu_matvec(device, queue, weight_buffer, wf, ctx,
        "mtp.layers.0.self_attn.v_proj", &attn_normed, kv_dim, hidden_dim);

    // Split Q into Q_head and gate
    let q_head: Vec<f32> = q[..q_dim].to_vec();
    let gate: Vec<f32> = q[q_dim..].to_vec();

    // 4c. Q, K norm
    let q_norm_w = wf.get_tensor_u16("mtp.layers.0.self_attn.q_norm.weight")
        .map(|w| w.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<f32>>())
        .unwrap_or_else(|| vec![1.0f32; head_dim]);
    let k_norm_w = wf.get_tensor_u16("mtp.layers.0.self_attn.k_norm.weight")
        .map(|w| w.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<f32>>())
        .unwrap_or_else(|| vec![1.0f32; head_dim]);

    let q_normed = per_head_norm(&q_head, num_attn_heads, head_dim, &q_norm_w);
    let k_normed = per_head_norm(&k, num_kv_heads, head_dim, &k_norm_w);

    // 4d. RoPE
    let q_roped = apply_rope(&q_normed, num_attn_heads, head_dim, rotary_dim, pos, rope_theta);
    let k_roped = apply_rope(&k_normed, num_kv_heads, head_dim, rotary_dim, pos, rope_theta);

    // 4e. KV cache append
    let kv_offset = pos * kv_dim;
    unsafe {
        let k_dst = (mtp_state.k_cache.contents() as *mut u8).add(kv_offset * 4) as *mut f32;
        std::ptr::copy_nonoverlapping(k_roped.as_ptr(), k_dst, kv_dim);
        let v_dst = (mtp_state.v_cache.contents() as *mut u8).add(kv_offset * 4) as *mut f32;
        std::ptr::copy_nonoverlapping(v.as_ptr(), v_dst, kv_dim);
    }

    // 4f. Attention (CPU)
    let seq_len = pos + 1;
    let kq_scale = 1.0f32 / (head_dim as f32).sqrt();
    let attn_out = cpu_attention(&q_roped, &mtp_state.k_cache, &mtp_state.v_cache,
        num_attn_heads, num_kv_heads, head_dim, seq_len, kq_scale);

    // 4g. Gate + output projection
    let gated: Vec<f32> = attn_out.iter().zip(gate.iter())
        .map(|(&a, &g)| a * sigmoid(g))
        .collect();

    let o_out = gpu_matvec(device, queue, weight_buffer, wf, ctx,
        "mtp.layers.0.self_attn.o_proj", &gated, hidden_dim, q_dim);

    let residual = cur.clone();
    cur = o_out;
    for i in 0..hidden_dim { cur[i] += residual[i]; }

    // 4h. Post-attn norm
    let post_norm_w = wf.get_tensor_u16("mtp.layers.0.post_attention_layernorm.weight")
        .map(|w| w.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<f32>>())
        .unwrap_or_else(|| vec![1.0f32; hidden_dim]);
    let post_normed = rms_norm(&cur, &post_norm_w, RMS_NORM_EPS);

    // 4i. Router gate → expert selection
    let gate_scores = gpu_matvec(device, queue, weight_buffer, wf, ctx,
        "mtp.layers.0.mlp.gate", &post_normed, num_experts, hidden_dim);
    let mut gate_s = gate_scores.clone();
    softmax(&mut gate_s);
    let k = num_experts_per_tok.min(MAX_K);
    let mut expert_indices = vec![0usize; k];
    let mut expert_weights = vec![0.0f32; k];
    topk(&gate_s, k, &mut expert_indices, &mut expert_weights);
    normalize_weights(&mut expert_weights);

    // 4j. Load expert data from disk
    let expert_size = expert_file.expert_size();
    let mut expert_data_bufs: Vec<Vec<u8>> = (0..k)
        .map(|ki| {
            let mut data = vec![0u8; expert_size];
            expert_file.read_expert(expert_indices[ki], &mut data).unwrap();
            data
        })
        .collect();

    // 4k. MoE compute on GPU
    let moe_out = gpu_moe_compute(
        device, queue, weight_buffer, wf, ctx, expert_buffer,
        &post_normed, &mut expert_data_bufs, &expert_weights,
        hidden_dim, moe_inter, shared_inter,
    );

    // 4l. Residual
    let ffn_residual = cur.clone();
    cur = moe_out;
    for i in 0..hidden_dim { cur[i] += ffn_residual[i]; }

    // ── 5. Output norm ────────────────────────────────────────────────────
    let out_norm_w = wf.get_tensor_u16("mtp.norm.weight")
        .map(|w| w.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<f32>>())
        .unwrap_or_else(|| {
            wf.get_tensor_u16("language_model.model.norm.weight")
                .map(|w| w.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<f32>>())
                .unwrap_or_else(|| vec![1.0f32; hidden_dim])
        });
    let out_normed = rms_norm(&cur, &out_norm_w, RMS_NORM_EPS);

    // ── 6. LM head on GPU ─────────────────────────────────────────────────
    let logits = gpu_matvec(device, queue, weight_buffer, wf, ctx,
        "language_model.lm_head", &out_normed, vocab_size, hidden_dim);

    mtp_state.pos += 1;
    logits
}

// ─── CPU helpers ──────────────────────────────────────────────────────────────

fn embed_token_cpu(wf: &WeightFile, token_id: usize, hidden_dim: usize) -> Vec<f32> {
    let (Some(w), Some(s), Some(b)) = (
        wf.get_tensor_u32("language_model.model.embed_tokens.weight"),
        wf.get_tensor_u16("language_model.model.embed_tokens.scales"),
        wf.get_tensor_u16("language_model.model.embed_tokens.biases"),
    ) else {
        return vec![0.0f32; hidden_dim];
    };
    let w_info = wf.get_tensor_info("language_model.model.embed_tokens.weight").unwrap();
    let packed_cols = w_info.shape[1];
    let s_info = wf.get_tensor_info("language_model.model.embed_tokens.scales").unwrap();
    let num_groups = s_info.shape[1];
    let group_size = hidden_dim / num_groups;
    let packed_per_group = group_size / 8;
    let w_row = &w[token_id * packed_cols..];
    let s_row = &s[token_id * num_groups..];
    let b_row = &b[token_id * num_groups..];
    let mut out = vec![0.0f32; hidden_dim];
    for g in 0..num_groups {
        let scale = bf16_to_f32(s_row[g]);
        let bias = bf16_to_f32(b_row[g]);
        for p in 0..packed_per_group {
            let packed = w_row[g * packed_per_group + p];
            for n in 0..8 {
                let nibble = (packed >> (n * 4)) & 0xF;
                let idx = g * group_size + p * 8 + n;
                if idx < hidden_dim {
                    out[idx] = nibble as f32 * scale + bias;
                }
            }
        }
    }
    out
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ssq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (ssq / n as f32 + eps).sqrt();
    x.iter().zip(weight.iter()).map(|(&v, &w)| v * inv_rms * w).collect()
}

fn per_head_norm(x: &[f32], num_heads: usize, head_dim: usize, weight: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for h in 0..num_heads {
        let base = h * head_dim;
        let normed = rms_norm(&x[base..base + head_dim], weight, RMS_NORM_EPS);
        out[base..base + head_dim].copy_from_slice(&normed);
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn apply_rope(x: &[f32], num_heads: usize, head_dim: usize, rope_dim: usize, pos: usize, theta: f32) -> Vec<f32> {
    let mut out = x.to_vec();
    let pos_f = pos as f32;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..rope_dim / 2 {
            let idx = base + i * 2;
            let freq = theta.powf(-2.0 * i as f32 / rope_dim as f32);
            let angle = pos_f * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = x[idx];
            let x1 = x[idx + 1];
            out[idx] = x0 * cos_a - x1 * sin_a;
            out[idx + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
    out
}

fn cpu_attention(
    q: &[f32], k_cache: &Buffer, v_cache: &Buffer,
    num_heads: usize, num_kv_heads: usize, head_dim: usize,
    seq_len: usize, kq_scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_heads * head_dim];
    let kv_dim = num_kv_heads * head_dim;
    let group_size = num_heads / num_kv_heads;

    unsafe {
        let k_ptr = k_cache.contents() as *const f32;
        let v_ptr = v_cache.contents() as *const f32;

        for h in 0..num_heads {
            let kv_h = h / group_size;
            let q_off = h * head_dim;

            let mut scores = vec![0.0f32; seq_len];
            let mut max_score = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let k_off = t * kv_dim + kv_h * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_off + d] * *k_ptr.add(k_off + d);
                }
                scores[t] = dot * kq_scale;
                max_score = max_score.max(scores[t]);
            }

            let mut sum_exp = 0.0f32;
            for t in 0..seq_len {
                scores[t] = (scores[t] - max_score).exp();
                sum_exp += scores[t];
            }

            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t in 0..seq_len {
                    let v_off = t * kv_dim + kv_h * head_dim;
                    acc += scores[t] / sum_exp * *v_ptr.add(v_off + d);
                }
                out[q_off + d] = acc;
            }
        }
    }
    out
}

// ─── GPU helpers ──────────────────────────────────────────────────────────────

fn gpu_matvec(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    weight_buffer: &WeightBuffer,
    wf: &WeightFile,
    ctx: &MetalContext,
    prefix: &str,
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let x_buf = metal_buf_shared(device, in_dim * 4);
    unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), x_buf.contents() as *mut f32, in_dim); }
    let out_buf = metal_buf_shared(device, out_dim * 4);
    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        weight_buffer.encode_matvec_into(wf, ctx, &enc, prefix, &x_buf, 0, &out_buf, 0, out_dim, in_dim);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
    let mut out = vec![0.0f32; out_dim];
    unsafe { std::ptr::copy_nonoverlapping(out_buf.contents() as *const f32, out.as_mut_ptr(), out_dim); }
    out
}

fn gpu_moe_compute(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    weight_buffer: &WeightBuffer,
    wf: &WeightFile,
    ctx: &MetalContext,
    expert_buffer: &ExpertBuffer,
    input: &[f32],
    expert_data: &mut [Vec<u8>],
    expert_weights: &[f32],
    hidden_dim: usize,
    moe_inter: usize,
    shared_inter: usize,
) -> Vec<f32> {
    let k = expert_data.len().min(MAX_K);
    let gs = 64usize;

    // Copy expert data to GPU buffers
    for ki in 0..k {
        unsafe {
            let dst = expert_buffer.expert_data[ki].contents() as *mut u8;
            std::ptr::copy_nonoverlapping(expert_data[ki].as_ptr(), dst, expert_data[ki].len());
        }
    }

    // Copy input to GPU
    let in_buf = metal_buf_shared(device, hidden_dim * 4);
    unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), in_buf.contents() as *mut f32, hidden_dim); }

    // Expert layout offsets (same as ExpertLayout in constants.rs)
    let gate_sb_size = moe_inter * (hidden_dim / gs) * 2;
    let gate_w_size = moe_inter * (hidden_dim / 8) * 4;
    let up_w_size = gate_w_size;
    let up_sb_size = gate_sb_size;
    let down_w_size = hidden_dim * (moe_inter / 8) * 4;
    let down_sb_size = hidden_dim * (moe_inter / gs) * 2;

    let gate_w_off: u64 = 0;
    let gate_s_off: u64 = gate_w_size as u64;
    let gate_b_off: u64 = gate_s_off + gate_sb_size as u64;
    let up_w_off: u64 = gate_b_off + gate_sb_size as u64;
    let up_s_off: u64 = up_w_off + up_w_size as u64;
    let up_b_off: u64 = up_s_off + up_sb_size as u64;
    let down_w_off: u64 = up_b_off + up_sb_size as u64;
    let down_s_off: u64 = down_w_off + down_w_size as u64;
    let down_b_off: u64 = down_s_off + down_sb_size as u64;

    let gs_u32 = gs as u32;
    let moe_inter_u32 = moe_inter as u32;
    let hidden_u32 = hidden_dim as u32;

    {
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        for ki in 0..k {
            let edata = &expert_buffer.expert_data[ki];
            let eout = &expert_buffer.expert_out[ki];

            // Gate matvec
            metal_kernels::encode_matvec_offset(
                ctx, &enc,
                edata, gate_w_off, edata, gate_s_off, edata, gate_b_off,
                &in_buf, 0, &expert_buffer.scratch_gate, 0,
                moe_inter_u32, hidden_u32, gs_u32, 3,
            );

            // Up matvec
            metal_kernels::encode_matvec_offset(
                ctx, &enc,
                edata, up_w_off, edata, up_s_off, edata, up_b_off,
                &in_buf, 0, &expert_buffer.scratch_up, 0,
                moe_inter_u32, hidden_u32, gs_u32, 3,
            );

            // SiLU gate * up
            metal_kernels::encode_swiglu(
                ctx, &enc,
                &expert_buffer.scratch_gate, 0,
                &expert_buffer.scratch_up, 0,
                &expert_buffer.scratch_act, 0,
                moe_inter_u32,
            );

            // Down matvec
            metal_kernels::encode_matvec_offset(
                ctx, &enc,
                edata, down_w_off, edata, down_s_off, edata, down_b_off,
                &expert_buffer.scratch_act, 0, eout, 0,
                hidden_u32, moe_inter_u32, gs_u32, 3,
            );
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Combine weighted expert outputs on CPU
    let mut combined = vec![0.0f32; hidden_dim];
    for ki in 0..k {
        let w = expert_weights[ki];
        unsafe {
            let src = expert_buffer.expert_out[ki].contents() as *const f32;
            for i in 0..hidden_dim {
                combined[i] += *src.add(i) * w;
            }
        }
    }

    // Shared expert
    let post_normed_buf = metal_buf_shared(device, hidden_dim * 4);
    unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), post_normed_buf.contents() as *mut f32, hidden_dim); }

    let shared_gate = gpu_matvec(device, queue, weight_buffer, wf, ctx,
        "mtp.layers.0.mlp.shared_expert.gate_proj", input, shared_inter, hidden_dim);
    let shared_up = gpu_matvec(device, queue, weight_buffer, wf, ctx,
        "mtp.layers.0.mlp.shared_expert.up_proj", input, shared_inter, hidden_dim);

    // SiLU gate * up on CPU
    let shared_act: Vec<f32> = shared_gate.iter().zip(shared_up.iter())
        .map(|(&g, &u)| u * g / (1.0 + (-g).exp()))
        .collect();

    let shared_down = gpu_matvec(device, queue, weight_buffer, wf, ctx,
        "mtp.layers.0.mlp.shared_expert.down_proj", &shared_act, hidden_dim, shared_inter);

    // Shared expert gate
    let sge_out = gpu_matvec(device, queue, weight_buffer, wf, ctx,
        "mtp.layers.0.mlp.shared_expert_gate", input, 1, hidden_dim);
    let shared_gate_score = sigmoid(sge_out[0]);

    for i in 0..hidden_dim {
        combined[i] += shared_down[i] * shared_gate_score;
    }

    combined
}

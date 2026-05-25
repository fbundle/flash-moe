/// CPU reference engine — all computation on CPU using ndarray + dequant matvec.
/// Serves as a verification baseline for the GPU engines.
use ndarray::Array1;
use std::cell::RefCell;

use super::constants::ModelConfig;
use crate::constants::{FULL_ATTN_INTERVAL, GROUP_SIZE, MAX_SEQ, RMS_NORM_EPS, CONV_KERNEL_SIZE};
use crate::cache::Cache;
use crate::engine::{Engine, SignalCheckFn};
use crate::error::MoEError;
use crate::math::{
    embed_lookup, final_norm, normalize_weights, softmax, topk,
    swiglu_fused, sigmoid_gate_inplace, residual_add,
    rms_norm_bf16, matvec_lookup,
    attention_scores_batched, attention_softmax_batched, attention_values_batched,
    kv_cache_append, compute_decay_beta, gated_delta_net_step, gated_rms_norm,
    q_head_norm_rope, k_head_norm_rope, conv1d_step,
    dequant_matvec_4bit, sigmoid,
};

/// Convert a byte slice at the given offset into a `&[u32]` view (packed 4-bit weights).
unsafe fn as_u32_slice(data: &[u8], offset: usize, count: usize) -> &[u32] {
    let ptr = data.as_ptr().add(offset) as *const u32;
    std::slice::from_raw_parts(ptr, count)
}

/// Convert a byte slice at the given offset into a `&[u16]` view (bf16).
unsafe fn as_u16_slice(data: &[u8], offset: usize, count: usize) -> &[u16] {
    let ptr = data.as_ptr().add(offset) as *const u16;
    std::slice::from_raw_parts(ptr, count)
}

// ─── Routing result ────────────────────────────────────────────────────────

struct Routing {
    indices: Vec<usize>,
    weights: Vec<f32>,
}

// ─── Pre-expert result ─────────────────────────────────────────────────────

struct PreExpertResult {
    /// Gate scores for expert routing [num_experts].
    gate_scores: Array1<f32>,
    /// Scalar gate for shared expert combine weight.
    shared_gate_score: f32,
    /// Residual after attention (before post_attention_layernorm).
    h_mid: Array1<f32>,
    /// post_attention_layernorm(h_mid), used for expert matvec inputs.
    post_normed: Array1<f32>,
    /// Shared expert gate projection [shared_intermediate].
    shared_gate: Array1<f32>,
    /// Shared expert up projection [shared_intermediate].
    shared_up: Array1<f32>,
}

// ─── CpuEngine ─────────────────────────────────────────────────────────────

pub struct CpuEngine<'a, C: ModelConfig> {
    pub model: &'a crate::model::Model,
    pub k: usize,
    cache: RefCell<Cache>,
    _phantom: std::marker::PhantomData<C>,
}

impl<'a, C: ModelConfig> CpuEngine<'a, C> {
    pub fn new(model: &'a crate::model::Model, k: usize) -> Result<Self, MoEError> {
        let c = &model.config;
        C::validate_config(c).map_err(MoEError::Config)?;
        let k = if k == 0 { C::NUM_EXPERTS_PER_TOK } else { k };
        let cache = RefCell::new(Cache::new(c));
        Ok(CpuEngine { model, k, cache, _phantom: std::marker::PhantomData })
    }

    // ── Routing ─────────────────────────────────────────────────────────

    fn route_experts(gate_scores: &mut [f32], k: usize) -> Routing {
        softmax(gate_scores);
        let mut indices = vec![0usize; k];
        let mut weights = vec![0.0f32; k];
        topk(gate_scores, k, &mut indices, &mut weights);
        normalize_weights(&mut weights);
        Routing { indices, weights }
    }

    // ── pre_expert: full attention ──────────────────────────────────────

    fn pre_expert_full(&self, layer: usize, pos: usize, hidden: &[f32]) -> PreExpertResult {
        let hd = C::HIDDEN_DIM;
        let num_q_heads = C::NUM_ATTN_HEADS;
        let num_kv_heads = C::NUM_KV_HEADS;
        let head_dim = C::HEAD_DIM;
        let rotary_dim = C::ROTARY_DIM;
        let rope_theta = C::ROPE_THETA as f32;
        let num_experts = C::NUM_EXPERTS;
        let shared_inter = C::SHARED_INTERMEDIATE;
        let kv_dim = C::KV_DIM;
        let q_dim = num_q_heads * head_dim;
        let q_proj_dim = q_dim * 2;

        let wf = &self.model.weight_file;
        let prefix = format!("model.layers.{}.self_attn", layer);

        // 1. input_layernorm
        let norm_w = wf.get_tensor_u16(
            &format!("model.layers.{}.input_layernorm.weight", layer)).unwrap();
        let mut normed = Array1::zeros(hd);
        rms_norm_bf16(hidden, norm_w, normed.as_slice_mut().unwrap(), hd, RMS_NORM_EPS);

        // 2. Q, K, V projections
        let mut qbuf = Array1::zeros(q_proj_dim);
        let mut kbuf = Array1::zeros(kv_dim);
        let mut vbuf = Array1::zeros(kv_dim);
        matvec_lookup(wf, &format!("{}.q_proj", prefix), normed.as_slice().unwrap(),
            qbuf.as_slice_mut().unwrap(), q_proj_dim, hd, GROUP_SIZE);
        matvec_lookup(wf, &format!("{}.k_proj", prefix), normed.as_slice().unwrap(),
            kbuf.as_slice_mut().unwrap(), kv_dim, hd, GROUP_SIZE);
        matvec_lookup(wf, &format!("{}.v_proj", prefix), normed.as_slice().unwrap(),
            vbuf.as_slice_mut().unwrap(), kv_dim, hd, GROUP_SIZE);

        // 3. Q head norm + RoPE
        let q_norm_w = wf.get_tensor_u16(&format!("{}.q_norm.weight", prefix)).unwrap();
        let mut q_out = Array1::zeros(q_dim);
        let mut q_gate = Array1::zeros(q_dim);
        q_head_norm_rope(
            qbuf.as_slice().unwrap(), q_norm_w,
            q_out.as_slice_mut().unwrap(), q_gate.as_slice_mut().unwrap(),
            num_q_heads, head_dim, rotary_dim, rope_theta, pos, RMS_NORM_EPS);

        // 4. K head norm + RoPE (in-place)
        let k_norm_w = wf.get_tensor_u16(&format!("{}.k_norm.weight", prefix)).unwrap();
        k_head_norm_rope(kbuf.as_slice_mut().unwrap(), k_norm_w,
            num_kv_heads, head_dim, rotary_dim, rope_theta, pos, RMS_NORM_EPS);

        // 5. KV cache append
        let _fa_idx = layer / FULL_ATTN_INTERVAL;
        {
            let mut cache = self.cache.borrow_mut();
            let kv = cache.full_mut(layer);
            kv_cache_append(
                kbuf.as_slice().unwrap(), vbuf.as_slice().unwrap(),
                &mut kv.k_cache, &mut kv.v_cache, pos, kv_dim);
            kv.len = pos + 1;
        }
        let seq_len = self.cache.borrow().full(layer).len;

        // 6-8. Attention
        let mut scores = Array1::zeros(num_q_heads * MAX_SEQ);
        {
            let cache = self.cache.borrow();
            let kv = cache.full(layer);
            attention_scores_batched(
                q_out.as_slice().unwrap(), &kv.k_cache,
                scores.as_slice_mut().unwrap(),
                num_q_heads, num_kv_heads, head_dim, kv_dim, seq_len, MAX_SEQ);
            attention_softmax_batched(
                scores.as_slice_mut().unwrap(), num_q_heads, seq_len, MAX_SEQ);
            let mut attn_out = Array1::zeros(q_dim);
            attention_values_batched(
                scores.as_slice().unwrap(), &kv.v_cache,
                attn_out.as_slice_mut().unwrap(),
                num_q_heads, num_kv_heads, head_dim, kv_dim, seq_len, MAX_SEQ);

            // 9. Sigmoid gate
            sigmoid_gate_inplace(attn_out.as_slice_mut().unwrap(),
                q_gate.as_slice().unwrap(), q_dim);

            // 10. o_proj
            let mut o_proj_out = Array1::zeros(hd);
            matvec_lookup(wf, &format!("{}.o_proj", prefix),
                attn_out.as_slice().unwrap(), o_proj_out.as_slice_mut().unwrap(),
                hd, q_dim, GROUP_SIZE);

            // 11. Residual add: h_mid = o_proj_out + hidden
            let mut h_mid = Array1::zeros(hd);
            residual_add(o_proj_out.as_slice().unwrap(), hidden,
                h_mid.as_slice_mut().unwrap(), hd);

            // 12. post_attention_layernorm
            let post_norm_w = wf.get_tensor_u16(
                &format!("model.layers.{}.post_attention_layernorm.weight", layer)).unwrap();
            let mut post_normed = Array1::zeros(hd);
            rms_norm_bf16(h_mid.as_slice().unwrap(), post_norm_w,
                post_normed.as_slice_mut().unwrap(), hd, RMS_NORM_EPS);

            // 13. Gate projections (on post_normed)
            let mlp_prefix = format!("model.layers.{}.mlp", layer);
            let mut gate_scores = Array1::zeros(num_experts);
            let mut shared_gate = Array1::zeros(shared_inter);
            let mut shared_up = Array1::zeros(shared_inter);
            let mut shared_gate_score_buf = [0.0f32];

            matvec_lookup(wf, &format!("{}.gate", mlp_prefix),
                post_normed.as_slice().unwrap(), gate_scores.as_slice_mut().unwrap(),
                num_experts, hd, GROUP_SIZE);
            matvec_lookup(wf, &format!("{}.shared_expert.gate_proj", mlp_prefix),
                post_normed.as_slice().unwrap(), shared_gate.as_slice_mut().unwrap(),
                shared_inter, hd, GROUP_SIZE);
            matvec_lookup(wf, &format!("{}.shared_expert.up_proj", mlp_prefix),
                post_normed.as_slice().unwrap(), shared_up.as_slice_mut().unwrap(),
                shared_inter, hd, GROUP_SIZE);
            matvec_lookup(wf, &format!("{}.shared_expert_gate", mlp_prefix),
                post_normed.as_slice().unwrap(), &mut shared_gate_score_buf,
                1, hd, GROUP_SIZE);

            PreExpertResult {
                gate_scores,
                shared_gate_score: shared_gate_score_buf[0],
                h_mid,
                post_normed,
                shared_gate,
                shared_up,
            }
        }
    }

    // ── pre_expert: linear attention ─────────────────────────────────────

    fn pre_expert_linear(&self, layer: usize, hidden: &[f32]) -> PreExpertResult {
        let hd = C::HIDDEN_DIM;
        let num_experts = C::NUM_EXPERTS;
        let shared_inter = C::SHARED_INTERMEDIATE;
        let qkv_dim = C::LINEAR_CONV_DIM;
        let total_key = C::LINEAR_TOTAL_KEY;
        let total_value = C::LINEAR_TOTAL_VALUE;
        let num_k_heads = C::LINEAR_NUM_K_HEADS;
        let num_v_heads = C::LINEAR_NUM_V_HEADS;
        let key_dim = total_key / num_k_heads;
        let value_dim = total_value / num_v_heads;
        let k_heads_per_v = num_v_heads / num_k_heads;
        let _linear_idx = layer - (layer + 1) / FULL_ATTN_INTERVAL;

        let wf = &self.model.weight_file;
        let prefix = format!("model.layers.{}.linear_attn", layer);

        // 1. input_layernorm
        let norm_w = wf.get_tensor_u16(
            &format!("model.layers.{}.input_layernorm.weight", layer)).unwrap();
        let mut normed = Array1::zeros(hd);
        rms_norm_bf16(hidden, norm_w, normed.as_slice_mut().unwrap(), hd, RMS_NORM_EPS);

        // 2. QKV, Z, Beta, Alpha projections
        let mut qkv = Array1::zeros(qkv_dim);
        let mut z = Array1::zeros(total_value);
        let mut beta_out = Array1::zeros(num_v_heads);
        let mut alpha_out = Array1::zeros(num_v_heads);
        matvec_lookup(wf, &format!("{}.in_proj_qkv", prefix),
            normed.as_slice().unwrap(), qkv.as_slice_mut().unwrap(),
            qkv_dim, hd, GROUP_SIZE);
        matvec_lookup(wf, &format!("{}.in_proj_z", prefix),
            normed.as_slice().unwrap(), z.as_slice_mut().unwrap(),
            total_value, hd, GROUP_SIZE);
        matvec_lookup(wf, &format!("{}.in_proj_b", prefix),
            normed.as_slice().unwrap(), beta_out.as_slice_mut().unwrap(),
            num_v_heads, hd, GROUP_SIZE);
        matvec_lookup(wf, &format!("{}.in_proj_a", prefix),
            normed.as_slice().unwrap(), alpha_out.as_slice_mut().unwrap(),
            num_v_heads, hd, GROUP_SIZE);

        // 3-4. Conv1d step (if conv weight exists)
        let mut conv_out = qkv.clone();
        if wf.get_tensor_ptr(&format!("{}.conv1d.weight", prefix)).is_some() {
            let conv_w = wf.get_tensor_u16(&format!("{}.conv1d.weight", prefix)).unwrap();
            let mut cache = self.cache.borrow_mut();
            let lin = cache.lin_mut(layer);
            conv1d_step(&lin.conv_state, qkv.as_slice().unwrap(), conv_w,
                conv_out.as_slice_mut().unwrap(), qkv_dim, CONV_KERNEL_SIZE);
            // Update conv_state: shift old → older, append new input
            for k in 0..CONV_KERNEL_SIZE - 2 {
                let src = (k + 1) * qkv_dim;
                let dst = k * qkv_dim;
                lin.conv_state.copy_within(src..src + qkv_dim, dst);
            }
            let tail = (CONV_KERNEL_SIZE - 2) * qkv_dim;
            lin.conv_state[tail..tail + qkv_dim]
                .copy_from_slice(qkv.as_slice().unwrap());
        }

        // 5. RMS norm Q/K (per-head, in-place on slices of conv_out)
        let inv_scale = 1.0 / (key_dim as f32).sqrt();
        {
            let q_slice = &conv_out.as_slice().unwrap()[..total_key];
            let k_slice = &conv_out.as_slice().unwrap()[total_key..2 * total_key];
            let mut q_normed: Vec<f32> = q_slice.to_vec();
            let mut k_normed: Vec<f32> = k_slice.to_vec();
            for h in 0..num_k_heads {
                let base = h * key_dim;
                let q_head = &q_slice[base..base + key_dim];
                let k_head = &k_slice[base..base + key_dim];
                let q_sq: f32 = q_head.iter().map(|v| v * v).sum();
                let k_sq: f32 = k_head.iter().map(|v| v * v).sum();
                let q_inv = 1.0 / (q_sq / key_dim as f32 + 1e-6).sqrt();
                let k_inv = 1.0 / (k_sq / key_dim as f32 + 1e-6).sqrt();
                for d in 0..key_dim {
                    q_normed[base + d] = q_head[d] * q_inv * inv_scale * inv_scale;
                    k_normed[base + d] = k_head[d] * k_inv * inv_scale;
                }
            }
            // Write back into conv_out
            conv_out.as_slice_mut().unwrap()[..total_key].copy_from_slice(&q_normed);
            conv_out.as_slice_mut().unwrap()[total_key..2 * total_key].copy_from_slice(&k_normed);
        }

        // 6. Compute decay & beta
        let a_log_ptr = wf.get_tensor_f32(&format!("{}.A_log", prefix));
        let dt_bias_ptr = wf.get_tensor_u16(&format!("{}.dt_bias", prefix));
        let a_log_default = vec![0.0f32; num_v_heads];
        let dt_bias_default = vec![0u16; num_v_heads];
        let a_log = a_log_ptr.unwrap_or(&a_log_default);
        let dt_bias = dt_bias_ptr.unwrap_or(&dt_bias_default);
        let mut g_decay = Array1::zeros(num_v_heads);
        let mut beta_gate = Array1::zeros(num_v_heads);
        compute_decay_beta(
            alpha_out.as_slice().unwrap(), beta_out.as_slice().unwrap(),
            a_log, dt_bias,
            g_decay.as_slice_mut().unwrap(), beta_gate.as_slice_mut().unwrap(),
            num_v_heads);

        // 7. Gated delta net step
        let conv_data = conv_out.as_slice().unwrap();
        let q_normed_slice = &conv_data[..total_key];
        let k_normed_slice = &conv_data[total_key..2 * total_key];
        let v_slice = &conv_data[2 * total_key..2 * total_key + total_value];
        let mut delta_out = Array1::zeros(total_value);
        {
            let mut cache = self.cache.borrow_mut();
            let lin = cache.lin_mut(layer);
            gated_delta_net_step(
                &mut lin.ssm_state,
                q_normed_slice, k_normed_slice, v_slice,
                g_decay.as_slice().unwrap(), beta_gate.as_slice().unwrap(),
                delta_out.as_slice_mut().unwrap(),
                num_v_heads, k_heads_per_v, key_dim, value_dim);
        }

        // 8. Gated RMS norm
        let mut gated_normed = Array1::zeros(total_value);
        if let Some(gnw) = wf.get_tensor_u16(&format!("{}.norm.weight", prefix)) {
            gated_rms_norm(
                delta_out.as_slice().unwrap(), z.as_slice().unwrap(), gnw,
                gated_normed.as_slice_mut().unwrap(),
                num_v_heads, value_dim, 1e-6);
        } else {
            gated_normed.assign(&delta_out);
        }

        // 9. out_proj
        let mut o_proj_out = Array1::zeros(hd);
        matvec_lookup(wf, &format!("{}.out_proj", prefix),
            gated_normed.as_slice().unwrap(), o_proj_out.as_slice_mut().unwrap(),
            hd, total_value, GROUP_SIZE);

        // 10. Residual add: h_mid = o_proj_out + hidden
        let mut h_mid = Array1::zeros(hd);
        residual_add(o_proj_out.as_slice().unwrap(), hidden,
            h_mid.as_slice_mut().unwrap(), hd);

        // 11. post_attention_layernorm
        let post_norm_w = wf.get_tensor_u16(
            &format!("model.layers.{}.post_attention_layernorm.weight", layer)).unwrap();
        let mut post_normed = Array1::zeros(hd);
        rms_norm_bf16(h_mid.as_slice().unwrap(), post_norm_w,
            post_normed.as_slice_mut().unwrap(), hd, RMS_NORM_EPS);

        // 12. Gate projections
        let mlp_prefix = format!("model.layers.{}.mlp", layer);
        let mut gate_scores = Array1::zeros(num_experts);
        let mut shared_gate = Array1::zeros(shared_inter);
        let mut shared_up = Array1::zeros(shared_inter);
        let mut shared_gate_score_buf = [0.0f32];

        matvec_lookup(wf, &format!("{}.gate", mlp_prefix),
            post_normed.as_slice().unwrap(), gate_scores.as_slice_mut().unwrap(),
            num_experts, hd, GROUP_SIZE);
        matvec_lookup(wf, &format!("{}.shared_expert.gate_proj", mlp_prefix),
            post_normed.as_slice().unwrap(), shared_gate.as_slice_mut().unwrap(),
            shared_inter, hd, GROUP_SIZE);
        matvec_lookup(wf, &format!("{}.shared_expert.up_proj", mlp_prefix),
            post_normed.as_slice().unwrap(), shared_up.as_slice_mut().unwrap(),
            shared_inter, hd, GROUP_SIZE);
        matvec_lookup(wf, &format!("{}.shared_expert_gate", mlp_prefix),
            post_normed.as_slice().unwrap(), &mut shared_gate_score_buf,
            1, hd, GROUP_SIZE);

        PreExpertResult {
            gate_scores,
            shared_gate_score: shared_gate_score_buf[0],
            h_mid,
            post_normed,
            shared_gate,
            shared_up,
        }
    }

    // ── post_expert ──────────────────────────────────────────────────────

    fn post_expert(
        &self, layer: usize, pre: &PreExpertResult, routing: &Routing,
    ) -> Result<Array1<f32>, MoEError> {
        let hd = C::HIDDEN_DIM;
        let moe_inter = C::MOE_INTERMEDIATE;
        let shared_inter = C::SHARED_INTERMEDIATE;
        let expert_size_bytes = C::EXPERT_SIZE_4BIT;
        let k = self.k;

        let wf = &self.model.weight_file;
        let mlp_prefix = format!("model.layers.{}.mlp", layer);

        // Compute expert outputs
        let mut expert_outs: Vec<Array1<f32>> = (0..k).map(|_| Array1::zeros(hd)).collect();
        let mut expert_data = vec![0u8; expert_size_bytes];

        for ki in 0..k {
            let eidx = routing.indices[ki];
            self.model.expert_files[layer].read_expert(eidx, &mut expert_data)?;

            // Gate projection: [moe_inter] = dequant_matvec(gate_W, post_normed)
            let gate_w = unsafe { as_u32_slice(&expert_data, C::GATE_W_OFF, C::GATE_W_SIZE / 4) };
            let gate_s = unsafe { as_u16_slice(&expert_data, C::GATE_S_OFF, C::GATE_S_SIZE / 2) };
            let gate_b = unsafe { as_u16_slice(&expert_data, C::GATE_B_OFF, C::GATE_B_SIZE / 2) };
            let mut gate_out = Array1::zeros(moe_inter);
            dequant_matvec_4bit(gate_w, gate_s, gate_b,
                pre.post_normed.as_slice().unwrap(),
                gate_out.as_slice_mut().unwrap(),
                moe_inter, hd, GROUP_SIZE);

            // Up projection: [moe_inter] = dequant_matvec(up_W, post_normed)
            let up_w = unsafe { as_u32_slice(&expert_data, C::UP_W_OFF, C::UP_W_SIZE / 4) };
            let up_s = unsafe { as_u16_slice(&expert_data, C::UP_S_OFF, C::UP_S_SIZE / 2) };
            let up_b = unsafe { as_u16_slice(&expert_data, C::UP_B_OFF, C::UP_B_SIZE / 2) };
            let mut up_out = Array1::zeros(moe_inter);
            dequant_matvec_4bit(up_w, up_s, up_b,
                pre.post_normed.as_slice().unwrap(),
                up_out.as_slice_mut().unwrap(),
                moe_inter, hd, GROUP_SIZE);

            // SwiGLU: act = silu(gate) * up
            let mut act = Array1::zeros(moe_inter);
            swiglu_fused(gate_out.as_slice().unwrap(), up_out.as_slice().unwrap(),
                act.as_slice_mut().unwrap(), moe_inter);

            // Down projection: [hd] = dequant_matvec(down_W, act)
            let down_w = unsafe { as_u32_slice(&expert_data, C::DOWN_W_OFF, C::DOWN_W_SIZE / 4) };
            let down_s = unsafe { as_u16_slice(&expert_data, C::DOWN_S_OFF, C::DOWN_S_SIZE / 2) };
            let down_b = unsafe { as_u16_slice(&expert_data, C::DOWN_B_OFF, C::DOWN_B_SIZE / 2) };
            dequant_matvec_4bit(down_w, down_s, down_b,
                act.as_slice().unwrap(),
                expert_outs[ki].as_slice_mut().unwrap(),
                hd, moe_inter, GROUP_SIZE);
        }

        // Shared expert: swiglu(gate, up) → down
        let mut shared_act = Array1::zeros(shared_inter);
        swiglu_fused(
            pre.shared_gate.as_slice().unwrap(),
            pre.shared_up.as_slice().unwrap(),
            shared_act.as_slice_mut().unwrap(), shared_inter);

        let mut shared_down = Array1::zeros(hd);
        matvec_lookup(wf, &format!("{}.shared_expert.down_proj", mlp_prefix),
            shared_act.as_slice().unwrap(), shared_down.as_slice_mut().unwrap(),
            hd, shared_inter, GROUP_SIZE);

        // Combine: hidden = h_mid + sum(w_i * expert_out_i) + sigmoid(gate_score) * shared_down
        let shared_gate = sigmoid(pre.shared_gate_score);
        let mut hidden = Array1::zeros(hd);
        let expert_refs: Vec<&[f32]> = expert_outs.iter().map(|a| a.as_slice().unwrap()).collect();
        // moe_combine_residual uses &[&[f32]] for expert_outs
        let h_mid_slice = pre.h_mid.as_slice().unwrap();
        let shared_down_slice = shared_down.as_slice().unwrap();
        let expert_weights_slice = routing.weights.as_slice();
        let hidden_slice = hidden.as_slice_mut().unwrap();
        for i in 0..hd {
            let mut moe_sum = 0.0f32;
            for ki in 0..k {
                moe_sum += expert_weights_slice[ki] * expert_refs[ki][i];
            }
            hidden_slice[i] = h_mid_slice[i] + moe_sum + shared_gate * shared_down_slice[i];
        }

        Ok(hidden)
    }

    // ── lm_head ─────────────────────────────────────────────────────────

    fn lm_head(&self, hidden: &[f32], logits: &mut [f32]) {
        let vocab_size = C::VOCAB_SIZE;
        let hd = C::HIDDEN_DIM;
        matvec_lookup(&self.model.weight_file, "lm_head", hidden, logits,
            vocab_size, hd, GROUP_SIZE);
    }
}

// ─── Engine trait ──────────────────────────────────────────────────────────

impl<'a, C: ModelConfig> Engine for CpuEngine<'a, C> {
    fn upload_cache(&self, cache: &Cache) {
        self.cache.borrow_mut().copy_from(cache);
    }

    fn download_cache(&self, cache: &mut Cache) {
        cache.copy_from(&self.cache.borrow());
    }

    fn forward(
        &mut self,
        input_ids: &[i64],
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError> {
        let n_tokens = input_ids.len();
        let hd = C::HIDDEN_DIM;
        let vocab_size = C::VOCAB_SIZE;
        let num_layers = C::NUM_LAYERS;

        let mut logits = vec![0.0f32; n_tokens * vocab_size];
        if n_tokens == 0 {
            return Ok(logits);
        }

        // Embed all tokens
        let mut embed = vec![0.0f32; n_tokens * hd];
        for (i, &id) in input_ids.iter().enumerate() {
            embed_lookup(&self.model.weight_file, id as usize,
                &mut embed[i * hd..(i + 1) * hd], hd);
        }

        let pos = self.cache.borrow().pos;

        for ti in 0..n_tokens {
            let cur_pos = pos + ti;
            let embed_hidden = &embed[ti * hd..(ti + 1) * hd];

            let mut hidden = Array1::from_vec(embed_hidden.to_vec());

            for layer in 0..num_layers {
                if check_signal() {
                    return Err(MoEError::Metal("interrupted".into()));
                }

                let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;
                let pre = if is_full {
                    self.pre_expert_full(layer, cur_pos, hidden.as_slice().unwrap())
                } else {
                    self.pre_expert_linear(layer, hidden.as_slice().unwrap())
                };

                let mut gate_scores_vec = pre.gate_scores.to_vec();
                let routing = Self::route_experts(&mut gate_scores_vec, self.k);

                hidden = self.post_expert(layer, &pre, &routing)?;
            }

            // Final norm + lm_head
            let mut hidden_vec = hidden.to_vec();
            final_norm(&self.model.weight_file, &mut hidden_vec, hd);
            let logit_slice = &mut logits[ti * vocab_size..(ti + 1) * vocab_size];
            self.lm_head(&hidden_vec, logit_slice);
        }

        self.cache.borrow_mut().set_pos(pos + n_tokens);
        Ok(logits)
    }
}

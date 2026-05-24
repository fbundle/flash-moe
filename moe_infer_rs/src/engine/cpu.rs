/// CPU-only engine using ndarray: reference implementation for correctness.
/// No GPU resources, no performance micro-optimizations — readable ndarray ops.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayViewMut1};

use crate::cache::Cache;
use crate::constants::{CONV_KERNEL_SIZE, FULL_ATTN_INTERVAL, GROUP_SIZE, MAX_SEQ, RMS_NORM_EPS};
use crate::engine::Engine;
use crate::error::MoEError;
use crate::model::Model;
use crate::engine::SignalCheckFn;
use crate::math::{
    apply_rope, bf16_to_f32, conv1d_step,
    normalize_weights, rms_norm_gated, sigmoid,
    topk,
};

// ─── ndarray helpers ─────────────────────────────────────────────────────

/// rms_norm using ndarray: out = (x / rms(x)) * weight
fn rms_norm_nd(x: &ArrayView1<f32>, weight: &ArrayView1<f32>, eps: f32) -> Array1<f32> {
    let n = x.len() as f32;
    let ssq: f32 = x.mapv(|v| v * v).sum();
    let inv_rms = 1.0 / (ssq / n + eps).sqrt();
    x * inv_rms * weight
}

/// rms_norm without weight (bare norm): out = x / rms(x)
fn rms_norm_bare_nd(x: &ArrayView1<f32>, eps: f32) -> Array1<f32> {
    let n = x.len() as f32;
    let ssq: f32 = x.mapv(|v| v * v).sum();
    let inv_rms = 1.0 / (ssq / n + eps).sqrt();
    x * inv_rms
}

/// Dequantize 4-bit packed weights to a dense Array1<f32> row.
fn dequant_row(
    w_packed: &[u32], scales: &[u16], biases: &[u16],
    _out_dim: usize, in_dim: usize, group_size: usize, row: usize,
) -> Array1<f32> {
    let num_groups = in_dim / group_size;
    let packed_cols = in_dim / 8;
    let s_row = &scales[row * num_groups..];
    let b_row = &biases[row * num_groups..];
    let mut out = Array1::zeros(in_dim);
    for g in 0..num_groups {
        let scale = bf16_to_f32(s_row[g]);
        let bias = bf16_to_f32(b_row[g]);
        let base = g * group_size;
        for p in 0..(group_size / 8) {
            let packed = w_packed[row * packed_cols + g * (group_size / 8) + p];
            for n in 0..8 {
                let nibble = (packed >> (n * 4)) & 0xF;
                out[base + p * 8 + n] = (nibble as f32) * scale + bias;
            }
        }
    }
    out
}

/// Fully dequantize a weight matrix from 4-bit to dense Array2<f32>.
fn dequant_matrix(
    w_packed: &[u32], scales: &[u16], biases: &[u16],
    out_dim: usize, in_dim: usize, group_size: usize,
) -> Array2<f32> {
    let rows: Vec<Array1<f32>> = (0..out_dim)
        .map(|r| dequant_row(w_packed, scales, biases, out_dim, in_dim, group_size, r))
        .collect();
    // Stack rows into an Array2: shape (out_dim, in_dim)
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().cloned()).collect();
    Array2::from_shape_vec((out_dim, in_dim), flat).unwrap()
}

/// Softmax in-place on ArrayViewMut1.
fn softmax_nd(x: &mut ArrayViewMut1<f32>) {
    let max_val = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    x.mapv_inplace(|v| (v - max_val).exp());
    let sum = x.sum();
    let inv = 1.0 / sum;
    x.mapv_inplace(|v| v * inv);
}

/// SiLU activation on ArrayViewMut1.
fn silu_nd(x: &mut ArrayViewMut1<f32>) {
    x.mapv_inplace(|v| v / (1.0 + (-v).exp()));
}

// ─── Execution context ─────────────────────────────────────────────────────

struct ExecCtx<'a> {
    model: &'a Model,
    cache: &'a mut Cache,
}

impl<'a> ExecCtx<'a> {
    // ── Embedding ──────────────────────────────────────────────────────────

    fn embed(&self, token_id: usize) -> Array1<f32> {
        let hd = self.model.config.hidden_dim;
        let (Some(w), Some(s), Some(b)) = (
            self.model.wf.get_tensor_u32("model.embed_tokens.weight"),
            self.model.wf.get_tensor_u16("model.embed_tokens.scales"),
            self.model.wf.get_tensor_u16("model.embed_tokens.biases"),
        ) else { return Array1::zeros(hd) };
        let w_info = self.model.wf.get_tensor_info("model.embed_tokens.weight").unwrap();
        let packed_cols = w_info.shape[1];
        let s_info = self.model.wf.get_tensor_info("model.embed_tokens.scales").unwrap();
        let num_groups = s_info.shape[1];
        let group_size = hd / num_groups;
        let w_row = &w[token_id * packed_cols..];
        let s_row = &s[token_id * num_groups..];
        let b_row = &b[token_id * num_groups..];
        let mut out = Array1::zeros(hd);
        for g in 0..num_groups {
            let scale = bf16_to_f32(s_row[g]);
            let bias = bf16_to_f32(b_row[g]);
            let base = g * group_size;
            for p in 0..(group_size / 8) {
                let packed = w_row[g * (group_size / 8) + p];
                for n in 0..8 {
                    let nibble = (packed >> (n * 4)) & 0xF;
                    out[base + p * 8 + n] = (nibble as f32) * scale + bias;
                }
            }
        }
        out
    }

    // ── Input RMS norm ─────────────────────────────────────────────────────

    fn input_norm(&self, layer_idx: usize, hidden: &ArrayView1<f32>) -> Array1<f32> {
        let name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
        if let Some(nw_u16) = self.model.wf.get_tensor_u16(&name) {
            let nw: Vec<f32> = nw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
            rms_norm_nd(hidden, &ArrayView1::from(&nw), RMS_NORM_EPS)
        } else {
            hidden.to_owned()
        }
    }

    // ── Post-attention RMS norm ────────────────────────────────────────────

    fn post_norm(&self, layer_idx: usize, hidden: &ArrayView1<f32>) -> Array1<f32> {
        let name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
        if let Some(pnw_u16) = self.model.wf.get_tensor_u16(&name) {
            let pnw: Vec<f32> = pnw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
            rms_norm_nd(hidden, &ArrayView1::from(&pnw), RMS_NORM_EPS)
        } else {
            hidden.to_owned()
        }
    }

    // ── Full (self) attention ──────────────────────────────────────────────

    fn full_attention(
        &mut self, layer_idx: usize,
        hidden: &mut Array1<f32>, residual: &ArrayView1<f32>, pos: usize,
        normed: &ArrayView1<f32>,
    ) {
        let hd = self.model.config.hidden_dim;
        let num_q = self.model.config.num_attn_heads;
        let num_kv = self.model.config.num_kv_heads;
        let head_dim = self.model.config.head_dim;
        let q_dim = num_q * head_dim;
        let q_proj_dim = q_dim * 2;
        let kv_dim = num_kv * head_dim;
        let prefix = format!("model.layers.{}.self_attn", layer_idx);

        // QKV projections: dequant weight row-by-row and compute dot products
        let q_weight = self.model.wf.get_tensor_u32(&format!("{}.q_proj.weight", prefix));
        let q_scale = self.model.wf.get_tensor_u16(&format!("{}.q_proj.scales", prefix));
        let q_bias = self.model.wf.get_tensor_u16(&format!("{}.q_proj.biases", prefix));
        let k_weight = self.model.wf.get_tensor_u32(&format!("{}.k_proj.weight", prefix));
        let k_scale = self.model.wf.get_tensor_u16(&format!("{}.k_proj.scales", prefix));
        let k_bias = self.model.wf.get_tensor_u16(&format!("{}.k_proj.biases", prefix));
        let v_weight = self.model.wf.get_tensor_u32(&format!("{}.v_proj.weight", prefix));
        let v_scale = self.model.wf.get_tensor_u16(&format!("{}.v_proj.scales", prefix));
        let v_bias = self.model.wf.get_tensor_u16(&format!("{}.v_proj.biases", prefix));

        let mut q_proj = Array1::zeros(q_proj_dim);
        let mut k = Array1::zeros(kv_dim);
        let mut v = Array1::zeros(kv_dim);

        if let (Some(qw), Some(qs), Some(qb)) = (q_weight, q_scale, q_bias) {
            let q_mat = dequant_matrix(qw, qs, qb, q_proj_dim, hd, GROUP_SIZE);
            q_proj = q_mat.dot(normed);
        }
        if let (Some(kw), Some(ks), Some(kb)) = (k_weight, k_scale, k_bias) {
            let k_mat = dequant_matrix(kw, ks, kb, kv_dim, hd, GROUP_SIZE);
            k = k_mat.dot(normed);
        }
        if let (Some(vw), Some(vs), Some(vb)) = (v_weight, v_scale, v_bias) {
            let v_mat = dequant_matrix(vw, vs, vb, kv_dim, hd, GROUP_SIZE);
            v = v_mat.dot(normed);
        }

        // Split Q / Q-gate from concatenated output
        let mut q = Array1::zeros(q_dim);
        let mut q_gate = Array1::zeros(q_dim);
        for h in 0..num_q {
            let src = &q_proj.slice(s![h * 2 * head_dim..(h + 1) * 2 * head_dim]);
            q.slice_mut(s![h * head_dim..(h + 1) * head_dim])
                .assign(&src.slice(s![..head_dim]));
            q_gate.slice_mut(s![h * head_dim..(h + 1) * head_dim])
                .assign(&src.slice(s![head_dim..2 * head_dim]));
        }

        // Q/K per-head norms
        let qn_name = format!("{}.q_norm.weight", prefix);
        let kn_name = format!("{}.k_norm.weight", prefix);
        if let Some(qnw) = self.model.wf.get_tensor_u16(&qn_name) {
            for h in 0..num_q {
                let mut qh = q.slice_mut(s![h * head_dim..(h + 1) * head_dim]);
                let ssq: f32 = qh.mapv(|x| x * x).sum();
                let inv = 1.0 / (ssq / head_dim as f32 + RMS_NORM_EPS).sqrt();
                for i in 0..head_dim.min(qnw.len()) {
                    qh[i] *= inv * bf16_to_f32(qnw[i]);
                }
            }
        }
        if let Some(knw) = self.model.wf.get_tensor_u16(&kn_name) {
            for h in 0..num_kv {
                let mut kh = k.slice_mut(s![h * head_dim..(h + 1) * head_dim]);
                let ssq: f32 = kh.mapv(|x| x * x).sum();
                let inv = 1.0 / (ssq / head_dim as f32 + RMS_NORM_EPS).sqrt();
                for i in 0..head_dim.min(knw.len()) {
                    kh[i] *= inv * bf16_to_f32(knw[i]);
                }
            }
        }

        // RoPE — uses existing helper (operates on slices, works with ndarray-owned data)
        apply_rope(
            q.as_slice_mut().unwrap(), k.as_slice_mut().unwrap(),
            pos, num_q, num_kv, head_dim,
            self.model.config.rotary_dim, self.model.config.rope_theta,
        );

        // Append K, V to cache
        let kv_cache = self.cache.kv[layer_idx].as_mut().unwrap();
        let cache_pos = kv_cache.len;
        assert!(cache_pos < MAX_SEQ);
        kv_cache.k_cache[cache_pos * kv_dim..(cache_pos + 1) * kv_dim]
            .copy_from_slice(k.as_slice().unwrap());
        kv_cache.v_cache[cache_pos * kv_dim..(cache_pos + 1) * kv_dim]
            .copy_from_slice(v.as_slice().unwrap());
        kv_cache.len += 1;
        let seq_len = kv_cache.len;

        let heads_per_kv = num_q / num_kv;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut attn_out = Array1::<f32>::zeros(q_dim);
        for h in 0..num_q {
            let kv_h = h / heads_per_kv;
            let qh = q.slice(s![h * head_dim..(h + 1) * head_dim]);

            // Compute attention scores: q @ K^T for this head
            let mut scores = Array1::zeros(seq_len);
            for p in 0..seq_len {
                let kp_start = p * kv_dim + kv_h * head_dim;
                let kp = &kv_cache.k_cache[kp_start..kp_start + head_dim];
                scores[p] = qh.iter().zip(kp.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale;
            }

            // Softmax
            let max_val = scores.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_scores: Array1<f32> = scores.mapv(|s| (s - max_val).exp());
            let sum = exp_scores.sum();
            let inv_sum = 1.0 / sum;

            // Weighted sum of values
            let mut oh = attn_out.slice_mut(s![h * head_dim..(h + 1) * head_dim]);
            for p in 0..seq_len {
                let weight = exp_scores[p] * inv_sum;
                let vp_start = p * kv_dim + kv_h * head_dim;
                let vp = &kv_cache.v_cache[vp_start..vp_start + head_dim];
                for d in 0..head_dim {
                    oh[d] += weight * vp[d];
                }
            }
        }

        // Sigmoid gate on attention output
        attn_out.mapv_inplace(|x| x / (1.0 + (-x).exp()));

        // o_proj: dequant and matvec
        let o_name = format!("{}.o_proj", prefix);
        let mut o_out = Array1::zeros(hd);
        if let (Some(ow), Some(os), Some(ob)) = (
            self.model.wf.get_tensor_u32(&format!("{}.weight", o_name)),
            self.model.wf.get_tensor_u16(&format!("{}.scales", o_name)),
            self.model.wf.get_tensor_u16(&format!("{}.biases", o_name)),
        ) {
            let o_mat = dequant_matrix(ow, os, ob, hd, q_dim, GROUP_SIZE);
            o_out = o_mat.dot(&attn_out);
        }

        // Residual add
        *hidden = residual + &o_out;
    }

    // ── Linear attention (GatedDeltaNet) ────────────────────────────────────

    fn linear_attention(
        &mut self, layer_idx: usize,
        hidden: &mut Array1<f32>, normed: &ArrayView1<f32>,
        residual: &ArrayView1<f32>,
    ) {
        let hd = self.model.config.hidden_dim;
        let n_k = self.model.config.linear_num_k_heads;
        let n_v = self.model.config.linear_num_v_heads;
        let total_key = self.model.config.linear_total_key;
        let total_value = self.model.config.linear_total_value;
        let qkv_dim = self.model.config.linear_conv_dim;
        let key_dim = total_key / n_k;
        let value_dim = total_value / n_v;
        let inv_scale = 1.0 / (key_dim as f32).sqrt();
        let k_heads_per_v = n_v / n_k;
        let prefix = format!("model.layers.{}.linear_attn", layer_idx);

        // QKV + Z + B + A projections
        let mut qkv = Array1::zeros(qkv_dim);
        let mut z = Array1::zeros(total_value);
        let mut beta = Array1::zeros(n_v);
        let mut alpha = Array1::zeros(n_v);

        if let (Some(qw), Some(qs), Some(qb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.in_proj_qkv.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_qkv.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_qkv.biases", prefix)),
        ) {
            let q_mat = dequant_matrix(qw, qs, qb, qkv_dim, hd, GROUP_SIZE);
            qkv = q_mat.dot(normed);
        }
        if let (Some(zw), Some(zs), Some(zb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.in_proj_z.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_z.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_z.biases", prefix)),
        ) {
            let z_mat = dequant_matrix(zw, zs, zb, total_value, hd, GROUP_SIZE);
            z = z_mat.dot(normed);
        }
        if let (Some(bw), Some(bs), Some(bb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.in_proj_b.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_b.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_b.biases", prefix)),
        ) {
            let b_mat = dequant_matrix(bw, bs, bb, n_v, hd, GROUP_SIZE);
            beta = b_mat.dot(normed);
        }
        if let (Some(aw), Some(ass), Some(ab)) = (
            self.model.wf.get_tensor_u32(&format!("{}.in_proj_a.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_a.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.in_proj_a.biases", prefix)),
        ) {
            let a_mat = dequant_matrix(aw, ass, ab, n_v, hd, GROUP_SIZE);
            alpha = a_mat.dot(normed);
        }

        // Conv1d step
        let state = self.cache.lin[layer_idx].as_mut().unwrap();
        let conv_out = if let Some(conv_w) = self.model.wf.get_tensor_u16(&format!("{}.conv1d.weight", prefix)) {
            let mut conv_out_slice = vec![0.0f32; qkv_dim];
            conv1d_step(&state.conv_state, qkv.as_slice().unwrap(), conv_w,
                        &mut conv_out_slice, qkv_dim, CONV_KERNEL_SIZE);
            Array1::from_vec(conv_out_slice)
        } else {
            qkv.clone()
        };
        // Shift conv ring buffer
        let shift = (CONV_KERNEL_SIZE - 2) * qkv_dim;
        state.conv_state.copy_within(qkv_dim.., 0);
        state.conv_state[shift..shift + qkv_dim].copy_from_slice(qkv.as_slice().unwrap());

        let lq = conv_out.slice(s![..total_key]);
        let lk = conv_out.slice(s![total_key..2 * total_key]);
        let lv = conv_out.slice(s![2 * total_key..]);

        // Q/K norms per head
        let mut q_normed = Array1::zeros(total_key);
        for h in 0..n_k {
            let qh = lq.slice(s![h * key_dim..(h + 1) * key_dim]);
            let qn = rms_norm_bare_nd(&qh, 1e-6);
            q_normed.slice_mut(s![h * key_dim..(h + 1) * key_dim])
                .assign(&(&qn * (inv_scale * inv_scale)));
        }
        let mut k_normed = Array1::zeros(total_key);
        for h in 0..n_k {
            let kh = lk.slice(s![h * key_dim..(h + 1) * key_dim]);
            let kn = rms_norm_bare_nd(&kh, 1e-6);
            k_normed.slice_mut(s![h * key_dim..(h + 1) * key_dim])
                .assign(&(&kn * inv_scale));
        }

        // GatedDeltaNet SSM per value head
        let a_log = self.model.wf.get_tensor_f32(&format!("{}.A_log", prefix));
        let dt_bias = self.model.wf.get_tensor_u16(&format!("{}.dt_bias", prefix));
        let mut out_vals = Array1::zeros(total_value);

        for vh in 0..n_v {
            let kh = vh / k_heads_per_v;
            let a_val = a_log.map_or(1.0, |al| al[vh]);
            let dt_b = dt_bias.map_or(0.0, |db| bf16_to_f32(db[vh]));
            let sp = (1.0 + (alpha[vh] + dt_b).exp()).ln();
            let g_decay = (-a_val.exp() * sp).exp();
            let beta_gate = sigmoid(beta[vh]);

            let so = vh * value_dim * key_dim;
            let ssm = &mut state.ssm_state[so..so + value_dim * key_dim];
            let v_h = lv.slice(s![vh * value_dim..(vh + 1) * value_dim]);
            let k_h = k_normed.slice(s![kh * key_dim..(kh + 1) * key_dim]);

            // Decay
            for vi in 0..value_dim {
                for ki in 0..key_dim {
                    ssm[vi * key_dim + ki] *= g_decay;
                }
            }
            // Update
            for vi in 0..value_dim {
                let kv_mem: f32 = (0..key_dim)
                    .map(|ki| ssm[vi * key_dim + ki] * k_h[ki]).sum();
                let delta = (v_h[vi] - kv_mem) * beta_gate;
                for ki in 0..key_dim {
                    ssm[vi * key_dim + ki] += k_h[ki] * delta;
                }
            }
            // Read
            let q_h = q_normed.slice(s![kh * key_dim..(kh + 1) * key_dim]);
            let mut oh = out_vals.slice_mut(s![vh * value_dim..(vh + 1) * value_dim]);
            for vi in 0..value_dim {
                oh[vi] = (0..key_dim).map(|ki| ssm[vi * key_dim + ki] * q_h[ki]).sum();
            }
        }

        // RMSNormGated
        let mut gated_out = Array1::zeros(total_value);
        if let Some(gnw) = self.model.wf.get_tensor_u16(&format!("{}.norm.weight", prefix)) {
            for vh in 0..n_v {
                let oh = out_vals.slice(s![vh * value_dim..(vh + 1) * value_dim]);
                let zh = z.slice(s![vh * value_dim..(vh + 1) * value_dim]);
                let mut gh = gated_out.slice_mut(s![vh * value_dim..(vh + 1) * value_dim]);
                let mut gh_slice = vec![0.0f32; value_dim];
                rms_norm_gated(
                    oh.as_slice().unwrap(), zh.as_slice().unwrap(), gnw,
                    &mut gh_slice, value_dim, RMS_NORM_EPS,
                );
                for i in 0..value_dim { gh[i] = gh_slice[i]; }
            }
        } else {
            gated_out = out_vals.clone();
        }

        // Output projection
        let mut attn_out = Array1::zeros(hd);
        if let (Some(ow), Some(os), Some(ob)) = (
            self.model.wf.get_tensor_u32(&format!("{}.out_proj.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.out_proj.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.out_proj.biases", prefix)),
        ) {
            let o_mat = dequant_matrix(ow, os, ob, hd, total_value, GROUP_SIZE);
            attn_out = o_mat.dot(&gated_out);
        }

        // Residual add
        *hidden = residual + &attn_out;
    }

    // ── MoE layer ──────────────────────────────────────────────────────────

    fn moe_layer(&mut self, layer_idx: usize, hidden: &mut Array1<f32>,
                 h_post: &ArrayView1<f32>)
    {
        let hd = self.model.config.hidden_dim;
        let n_experts = self.model.config.num_experts;
        let moe_inter = self.model.config.moe_intermediate;
        let shared_inter = self.model.config.shared_intermediate;
        let k = self.model.config.num_experts_per_tok;
        let prefix = format!("model.layers.{}.mlp", layer_idx);

        // h_mid = input hidden (residual for combine)
        let h_mid = hidden.clone();

        // Router gate + shared expert projections
        let mut gate_scores = Array1::zeros(n_experts);
        let mut shared_gate = Array1::zeros(shared_inter);
        let mut shared_up = Array1::zeros(shared_inter);
        let mut shared_gate_score = 0.0f32;

        if let (Some(gw), Some(gs), Some(gb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.gate.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.gate.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.gate.biases", prefix)),
        ) {
            let gate_mat = dequant_matrix(gw, gs, gb, n_experts, hd, GROUP_SIZE);
            gate_scores = gate_mat.dot(h_post);
        }
        if let (Some(sgw), Some(sgs), Some(sgb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.shared_expert.gate_proj.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert.gate_proj.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert.gate_proj.biases", prefix)),
        ) {
            let sg_mat = dequant_matrix(sgw, sgs, sgb, shared_inter, hd, GROUP_SIZE);
            shared_gate = sg_mat.dot(h_post);
        }
        if let (Some(suw), Some(sus), Some(sub)) = (
            self.model.wf.get_tensor_u32(&format!("{}.shared_expert.up_proj.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert.up_proj.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert.up_proj.biases", prefix)),
        ) {
            let su_mat = dequant_matrix(suw, sus, sub, shared_inter, hd, GROUP_SIZE);
            shared_up = su_mat.dot(h_post);
        }
        if let (Some(segw), Some(segs), Some(segb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.shared_expert_gate.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert_gate.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert_gate.biases", prefix)),
        ) {
            let seg_vec = dequant_row(segw, segs, segb, 1, hd, GROUP_SIZE, 0);
            shared_gate_score = seg_vec.dot(h_post);
        }

        // Routing: softmax + topk
        softmax_nd(&mut gate_scores.view_mut());
        let mut expert_indices = vec![0usize; k];
        let mut expert_weights = Array1::zeros(k);
        topk(gate_scores.as_slice().unwrap(), k,
             &mut expert_indices,
             expert_weights.as_slice_mut().unwrap());
        normalize_weights(expert_weights.as_slice_mut().unwrap());

        // Expert computation (CPU)
        let mut moe_out = Array1::<f32>::zeros(hd);
        let expert_size = self.model.config.expert_size_4bit;
        let mut expert_data = vec![0u8; expert_size];
        let layout = &self.model.config.expert_layout_4bit;
        let expert_file = &self.model.expert_files[layer_idx];

        for (&eidx, &ew) in expert_indices.iter().zip(expert_weights.iter()) {
            if expert_file.read_expert(eidx, &mut expert_data).is_err() { continue; }

            let gw = unsafe {
                std::slice::from_raw_parts(
                    expert_data.as_ptr().add(layout.gate_w_off) as *const u32,
                    layout.gate_w_size / 4)
            };
            let gs = unsafe {
                std::slice::from_raw_parts(
                    expert_data.as_ptr().add(layout.gate_s_off) as *const u16,
                    layout.gate_s_size / 2)
            };
            let gb = unsafe {
                std::slice::from_raw_parts(
                    expert_data.as_ptr().add(layout.gate_b_off) as *const u16,
                    layout.gate_b_size / 2)
            };
            let gate_mat = dequant_matrix(gw, gs, gb, moe_inter, hd, GROUP_SIZE);
            let gate_out = gate_mat.dot(h_post);

            let uw = unsafe {
                std::slice::from_raw_parts(
                    expert_data.as_ptr().add(layout.up_w_off) as *const u32,
                    layout.up_w_size / 4)
            };
            let us = unsafe {
                std::slice::from_raw_parts(
                    expert_data.as_ptr().add(layout.up_s_off) as *const u16,
                    layout.up_s_size / 2)
            };
            let ub = unsafe {
                std::slice::from_raw_parts(
                    expert_data.as_ptr().add(layout.up_b_off) as *const u16,
                    layout.up_b_size / 2)
            };
            let up_mat = dequant_matrix(uw, us, ub, moe_inter, hd, GROUP_SIZE);
            let up_out = up_mat.dot(h_post);

            // SwiGLU: silu(gate) * up
            let mut act = gate_out.clone();
            silu_nd(&mut act.view_mut());
            act = &act * &up_out;

            let dw = unsafe {
                std::slice::from_raw_parts(
                    expert_data.as_ptr().add(layout.down_w_off) as *const u32,
                    layout.down_w_size / 4)
            };
            let ds = unsafe {
                std::slice::from_raw_parts(
                    expert_data.as_ptr().add(layout.down_s_off) as *const u16,
                    layout.down_s_size / 2)
            };
            let db = unsafe {
                std::slice::from_raw_parts(
                    expert_data.as_ptr().add(layout.down_b_off) as *const u16,
                    layout.down_b_size / 2)
            };
            let down_mat = dequant_matrix(dw, ds, db, hd, moe_inter, GROUP_SIZE);
            let eout = down_mat.dot(&act);

            moe_out = &moe_out + &(eout * ew);
        }

        // Shared expert SwiGLU + down_proj
        let mut shared_act = shared_gate.clone();
        silu_nd(&mut shared_act.view_mut());
        shared_act = &shared_act * &shared_up;

        let mut shared_out = Array1::zeros(hd);
        if let (Some(sdw), Some(sds), Some(sdb)) = (
            self.model.wf.get_tensor_u32(&format!("{}.shared_expert.down_proj.weight", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert.down_proj.scales", prefix)),
            self.model.wf.get_tensor_u16(&format!("{}.shared_expert.down_proj.biases", prefix)),
        ) {
            let sd_mat = dequant_matrix(sdw, sds, sdb, hd, shared_inter, GROUP_SIZE);
            shared_out = sd_mat.dot(&shared_act);
        }

        let shared_weight = sigmoid(shared_gate_score);

        // Final combine: hidden = h_mid + moe_out + shared_weight * shared_out
        *hidden = &h_mid + &moe_out + &(&shared_out * shared_weight);
    }

    // ── Final norm + LM head ───────────────────────────────────────────────

    fn final_norm_and_lm_head(&self, hidden: &mut Array1<f32>) -> Array1<f32> {
        let hd = self.model.config.hidden_dim;
        let vs = self.model.config.vocab_size;

        // Final RMS norm (in-place on hidden)
        if let Some(fnw_u16) = self.model.wf.get_tensor_u16("model.norm.weight") {
            let fnw: Vec<f32> = fnw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
            let ssq: f32 = hidden.mapv(|v| v * v).sum();
            let inv_rms = 1.0 / (ssq / hd as f32 + RMS_NORM_EPS).sqrt();
            for i in 0..hd {
                hidden[i] *= inv_rms * fnw[i];
            }
        }

        // LM head (vocab projection)
        if let (Some(w), Some(s), Some(b)) = (
            self.model.wf.get_tensor_u32("lm_head.weight"),
            self.model.wf.get_tensor_u16("lm_head.scales"),
            self.model.wf.get_tensor_u16("lm_head.biases"),
        ) {
            let lm_mat = dequant_matrix(w, s, b, vs, hd, GROUP_SIZE);
            lm_mat.dot(&hidden.view())
        } else {
            Array1::zeros(vs)
        }
    }
}

// ─── EngineCPU ─────────────────────────────────────────────────────────────

pub struct EngineCPU<'a> {
    pub model: &'a Model,
}

impl<'a> Engine for EngineCPU<'a> {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError> {
        let vs = self.model.config.vocab_size;
        let num_layers = self.model.config.num_layers;
        let n = input_ids.len();
        if n == 0 { return Ok(vec![]); }

        let mut logits = vec![0.0f32; n * vs];
        let mut exec = ExecCtx { model: self.model, cache };

        for (ti, &id) in input_ids.iter().enumerate() {
            let pos = exec.cache.pos;
            let mut hidden = exec.embed(id as usize);

            for layer in 0..num_layers {
                if layer % 4 == 0 && check_signal() {
                    return Err(MoEError::Metal("interrupted".into()));
                }

                let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;
                let residual = hidden.clone();
                let normed = exec.input_norm(layer, &hidden.view());

                if is_full {
                    exec.full_attention(layer, &mut hidden,
                                        &residual.view(), pos,
                                        &normed.view());
                } else {
                    exec.linear_attention(layer, &mut hidden,
                                          &normed.view(),
                                          &residual.view());
                }

                let h_post = exec.post_norm(layer, &hidden.view());
                exec.moe_layer(layer, &mut hidden, &h_post.view());
            }

            exec.cache.pos += 1;
            let logit_arr = exec.final_norm_and_lm_head(&mut hidden);
            logits[ti * vs..(ti + 1) * vs]
                .copy_from_slice(logit_arr.as_slice().unwrap());
        }

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Model;
    use crate::math::dequant_matvec_4bit;

    #[test]
    fn test_embed_consistency() {
        let model = Model::load("../data/models--mlx-community--Qwen3.6-35B-A3B-4bit-stripped").unwrap();
        let token_id = 248045;
        let hd = model.config.hidden_dim;

        // Original logic (from old cpu.rs)
        let mut old_out = vec![0.0f32; hd];
        {
            let (Some(w), Some(s), Some(b)) = (
                model.wf.get_tensor_u32("model.embed_tokens.weight"),
                model.wf.get_tensor_u16("model.embed_tokens.scales"),
                model.wf.get_tensor_u16("model.embed_tokens.biases"),
            ) else { return };
            let w_info = model.wf.get_tensor_info("model.embed_tokens.weight").unwrap();
            let packed_cols = w_info.shape[1];
            let s_info = model.wf.get_tensor_info("model.embed_tokens.scales").unwrap();
            let num_groups = s_info.shape[1];
            let group_size = hd / num_groups;
            let w_row = &w[token_id * packed_cols..];
            let s_row = &s[token_id * num_groups..];
            let b_row = &b[token_id * num_groups..];
            for g in 0..num_groups {
                let scale = bf16_to_f32(s_row[g]);
                let bias = bf16_to_f32(b_row[g]);
                let base = g * group_size;
                for p in 0..(group_size / 8) {
                    let packed = w_row[g * (group_size / 8) + p];
                    for n in 0..8 {
                        let nibble = (packed >> (n * 4)) & 0xF;
                        old_out[base + p * 8 + n] = (nibble as f32) * scale + bias;
                    }
                }
            }
        }

        // New ndarray logic
        let new_out = {
            let (Some(w), Some(s), Some(b)) = (
                model.wf.get_tensor_u32("model.embed_tokens.weight"),
                model.wf.get_tensor_u16("model.embed_tokens.scales"),
                model.wf.get_tensor_u16("model.embed_tokens.biases"),
            ) else { return };
            let w_info = model.wf.get_tensor_info("model.embed_tokens.weight").unwrap();
            let packed_cols = w_info.shape[1];
            let s_info = model.wf.get_tensor_info("model.embed_tokens.scales").unwrap();
            let num_groups = s_info.shape[1];
            let group_size = hd / num_groups;
            let w_row = &w[token_id * packed_cols..];
            let s_row = &s[token_id * num_groups..];
            let b_row = &b[token_id * num_groups..];
            let mut out = Array1::zeros(hd);
            for g in 0..num_groups {
                let scale = bf16_to_f32(s_row[g]);
                let bias = bf16_to_f32(b_row[g]);
                let base = g * group_size;
                for p in 0..(group_size / 8) {
                    let packed = w_row[g * (group_size / 8) + p];
                    for n in 0..8 {
                        let nibble = (packed >> (n * 4)) & 0xF;
                        out[base + p * 8 + n] = (nibble as f32) * scale + bias;
                    }
                }
            }
            out
        };

        let max_diff = old_out.iter().zip(new_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("embed max_diff: {:e}", max_diff);
        assert!(max_diff < 1e-6, "embed mismatch: max_diff={}", max_diff);
    }

    #[test]
    fn test_dequant_matvec_consistency() {
        let model = Model::load("../data/models--mlx-community--Qwen3.6-35B-A3B-4bit-stripped").unwrap();
        let hd = model.config.hidden_dim;
        let token_id = 248045;

        // Get normed input (just use embedding as test vector)
        let mut test_vec = vec![0.0f32; hd];
        // Original embed lookup
        {
            let (Some(w), Some(s), Some(b)) = (
                model.wf.get_tensor_u32("model.embed_tokens.weight"),
                model.wf.get_tensor_u16("model.embed_tokens.scales"),
                model.wf.get_tensor_u16("model.embed_tokens.biases"),
            ) else { return };
            let w_info = model.wf.get_tensor_info("model.embed_tokens.weight").unwrap();
            let packed_cols = w_info.shape[1];
            let s_info = model.wf.get_tensor_info("model.embed_tokens.scales").unwrap();
            let num_groups = s_info.shape[1];
            let group_size = hd / num_groups;
            let w_row = &w[token_id * packed_cols..];
            let s_row = &s[token_id * num_groups..];
            let b_row = &b[token_id * num_groups..];
            for g in 0..num_groups {
                let scale = bf16_to_f32(s_row[g]);
                let bias = bf16_to_f32(b_row[g]);
                let base = g * group_size;
                for p in 0..(group_size / 8) {
                    let packed = w_row[g * (group_size / 8) + p];
                    for n in 0..8 {
                        let nibble = (packed >> (n * 4)) & 0xF;
                        test_vec[base + p * 8 + n] = (nibble as f32) * scale + bias;
                    }
                }
            }
        }

        // Test dequant_matvec for gate projection
        let n_experts = model.config.num_experts;
        let prefix = "model.layers.0.mlp";
        let (Some(gw), Some(gs), Some(gb)) = (
            model.wf.get_tensor_u32(&format!("{}.gate.weight", prefix)),
            model.wf.get_tensor_u16(&format!("{}.gate.scales", prefix)),
            model.wf.get_tensor_u16(&format!("{}.gate.biases", prefix)),
        ) else { return };

        // Old way: direct dequant_matvec
        let mut old_out = vec![0.0f32; n_experts];
        dequant_matvec_4bit(gw, gs, gb, &test_vec, &mut old_out, n_experts, hd, GROUP_SIZE);

        // New way: dequant_matrix then dot
        let new_mat = dequant_matrix(gw, gs, gb, n_experts, hd, GROUP_SIZE);
        let new_out = new_mat.dot(&ArrayView1::from(&test_vec));

        let max_diff = old_out.iter().zip(new_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("dequant_matvec max_diff: {:e}", max_diff);
        assert!(max_diff < 1e-5, "dequant_matvec mismatch: max_diff={}", max_diff);
    }

    #[test]
    fn test_full_attention_qkv() {
        let model = Model::load("../data/models--mlx-community--Qwen3.6-35B-A3B-4bit-stripped").unwrap();
        let hd = model.config.hidden_dim;
        let num_q = model.config.num_attn_heads;
        let num_kv = model.config.num_kv_heads;
        let head_dim = model.config.head_dim;
        let q_dim = num_q * head_dim;
        let q_proj_dim = q_dim * 2;
        let kv_dim = num_kv * head_dim;
        let layer_idx = 3; // full attention layer
        let prefix = format!("model.layers.{}.self_attn", layer_idx);

        // Create test normed vector (simple ones)
        let normed = Array1::ones(hd);

        // New ndarray way
        let (q_weight, q_scale, q_bias) = (
            model.wf.get_tensor_u32(&format!("{}.q_proj.weight", prefix)),
            model.wf.get_tensor_u16(&format!("{}.q_proj.scales", prefix)),
            model.wf.get_tensor_u16(&format!("{}.q_proj.biases", prefix)),
        );
        let (k_weight, k_scale, k_bias) = (
            model.wf.get_tensor_u32(&format!("{}.k_proj.weight", prefix)),
            model.wf.get_tensor_u16(&format!("{}.k_proj.scales", prefix)),
            model.wf.get_tensor_u16(&format!("{}.k_proj.biases", prefix)),
        );
        let new_q_proj = if let (Some(qw), Some(qs), Some(qb)) = (q_weight, q_scale, q_bias) {
            let q_mat = dequant_matrix(qw, qs, qb, q_proj_dim, hd, GROUP_SIZE);
            q_mat.dot(&normed.view())
        } else { Array1::zeros(q_proj_dim) };

        // Old way
        let mut old_q_proj = vec![0.0f32; q_proj_dim];
        if let (Some(qw), Some(qs), Some(qb)) = (q_weight, q_scale, q_bias) {
            dequant_matvec_4bit(qw, qs, qb, normed.as_slice().unwrap(),
                &mut old_q_proj, q_proj_dim, hd, GROUP_SIZE);
        }

        let max_diff = old_q_proj.iter().zip(new_q_proj.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("full_attn Q proj max_diff: {:e}", max_diff);
        assert!(max_diff < 1e-5, "Q proj mismatch: max_diff={}", max_diff);
    }

    #[test]
    fn test_rms_norm_consistency() {
        use crate::math::rms_norm;
        let hd = 2048;
        let x: Vec<f32> = (0..hd).map(|i| (i as f32).sin()).collect();
        let w: Vec<f32> = (0..hd).map(|i| ((i as f32) * 0.5).cos() + 1.0).collect();

        let mut old_out = vec![0.0f32; hd];
        rms_norm(&x, &w, &mut old_out, hd, RMS_NORM_EPS);

        let new_out = rms_norm_nd(
            &ArrayView1::from(&x),
            &ArrayView1::from(&w),
            RMS_NORM_EPS,
        );

        let max_diff = old_out.iter().zip(new_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("rms_norm max_diff: {:e}", max_diff);
        assert!(max_diff < 1e-5, "rms_norm mismatch: max_diff={}", max_diff);
    }
}

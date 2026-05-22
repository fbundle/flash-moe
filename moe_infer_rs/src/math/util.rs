/// Shared types, constants, and CPU helpers used across engine modules.
use crate::constants::RMS_NORM_EPS;
use crate::model::weights::WeightFile;

// ─── bf16 / f32 conversion ───────────────────────────────────────────────

/// Convert bf16 (uint16) to f32.
pub fn bf16_to_f32(bf16: u16) -> f32 {
    f32::from_bits((bf16 as u32) << 16)
}

/// CPU reference: 4-bit dequantized matrix-vector multiply.
pub fn dequant_matvec_4bit(
    w_packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    x: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) {
    let num_groups = in_dim / group_size;
    let packed_per_group = group_size / 8;
    let packed_cols = in_dim / 8;

    for row in 0..out_dim {
        let mut acc = 0.0f32;
        let w_row = &w_packed[row * packed_cols..];
        let s_row = &scales[row * num_groups..];
        let b_row = &biases[row * num_groups..];

        for g in 0..num_groups {
            let scale = bf16_to_f32(s_row[g]);
            let bias = bf16_to_f32(b_row[g]);

            let base_packed = g * packed_per_group;
            let base_x = g * group_size;

            for p in 0..packed_per_group {
                let packed = w_row[base_packed + p];
                let x_base = base_x + p * 8;

                for n in 0..8 {
                    let nibble = (packed >> (n * 4)) & 0xF;
                    let w_val = (nibble as f32) * scale + bias;
                    acc += w_val * x[x_base + n];
                }
            }
        }
        out[row] = acc;
    }
}

/// CPU reference: RMS normalization.
pub fn rms_norm(x: &[f32], weight: &[f32], out: &mut [f32], dim: usize, eps: f32) {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / dim as f32 + eps).sqrt().recip();
    for i in 0..dim {
        out[i] = x[i] * rms * weight[i];
    }
}

// ─── CPU helper functions ────────────────────────────────────────────────

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

pub fn softmax(x: &mut [f32]) {
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

pub fn topk(scores: &[f32], k: usize, indices: &mut [usize], values: &mut [f32]) {
    for (i, &score) in scores.iter().enumerate() {
        if i < k {
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

pub fn normalize_weights(weights: &mut [f32]) {
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for w in weights.iter_mut() { *w *= inv; }
    }
}

pub fn rms_norm_bare(x: &[f32], out: &mut [f32], dim: usize, eps: f32) {
    let sum_sq: f32 = x[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms;
    }
}

pub fn rms_norm_gated(
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

pub fn conv1d_step(
    conv_state: &[f32],
    new_input: &[f32],
    weight_bf16: &[u16],
    out: &mut [f32],
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
    silu(&mut out[..channels]);
}

// ─── Token embedding lookup ────────────────────────────────────────────────

pub fn embed_lookup(wf: &WeightFile, token_id: usize, out: &mut [f32], hidden_dim: usize) {
    let (Some(w), Some(s), Some(b)) = (
        wf.get_tensor_u32("model.embed_tokens.weight"),
        wf.get_tensor_u16("model.embed_tokens.scales"),
        wf.get_tensor_u16("model.embed_tokens.biases"),
    ) else {
        out.fill(0.0);
        return;
    };
    let w_info = wf.get_tensor_info("model.embed_tokens.weight").unwrap();
    let packed_cols = w_info.shape[1];
    let s_info = wf.get_tensor_info("model.embed_tokens.scales").unwrap();
    let num_groups = s_info.shape[1];
    let group_size = hidden_dim / num_groups;
    let packed_per_group = group_size / 8;
    let w_row = &w[token_id * packed_cols..];
    let s_row = &s[token_id * num_groups..];
    let b_row = &b[token_id * num_groups..];
    for g in 0..num_groups {
        let scale = bf16_to_f32(s_row[g]);
        let bias = bf16_to_f32(b_row[g]);
        let base = g * group_size;
        for p in 0..packed_per_group {
            let packed = w_row[g * packed_per_group + p];
            for n in 0..8 {
                let nibble = (packed >> (n * 4)) & 0xF;
                out[base + p * 8 + n] = (nibble as f32) * scale + bias;
            }
        }
    }
}

// ─── Final RMS norm ────────────────────────────────────────────────────────

pub fn final_norm(wf: &WeightFile, hidden: &mut [f32], hidden_dim: usize) {
    let Some(fnw_u16) = wf.get_tensor_u16("model.norm.weight") else { return };
    let fnw_f32: Vec<f32> = fnw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
    let sum_sq: f32 = hidden[..hidden_dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / hidden_dim as f32 + RMS_NORM_EPS).sqrt();
    for i in 0..hidden_dim {
        hidden[i] *= inv_rms * fnw_f32[i];
    }
}

// ─── LM head (moved to lm_head.rs) ──────────────────────────────────────────

//! CPU computation kernels — ported from `moe_infer_mlx/core_src/cpu_kernels.h`.
//!
//! All kernels operate on flat slices. The caller is responsible for ensuring
//! that input slices have sufficient length for the declared dimensions.

use half::bf16;

// ---------------------------------------------------------------------------
// bf16 <-> f32 conversion
// ---------------------------------------------------------------------------

/// Reinterpret a `u16` as the lower 16 bits of a bf16 value and widen to `f32`.
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    bf16::from_bits(b).to_f32()
}

/// Convert an `f32` to bf16 and return the raw upper-16-bit representation.
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    bf16::from_f32(val).to_bits()
}

// ---------------------------------------------------------------------------
// 4-bit dequantized matrix-vector multiply
// ---------------------------------------------------------------------------

/// 4-bit dequant matvec: `out[out_dim] = W * x[in_dim]`.
///
/// `W` is stored as packed `u32` (eight 4-bit values per `u32`).
/// `scales` and `biases` are bf16 per group.
pub fn cpu_dequant_matvec(
    w: &[u32],
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
        let w_row = &w[row * packed_cols..];
        let s_row = &scales[row * num_groups..];
        let b_row = &biases[row * num_groups..];

        for g in 0..num_groups {
            let scale = bf16_to_f32(s_row[g]);
            let bias = bf16_to_f32(b_row[g]);
            let base_packed = g * packed_per_group;
            let base_x = g * group_size;

            for p in 0..packed_per_group {
                let packed = w_row[base_packed + p];
                let xb = base_x + p * 8;

                let nib0 = ((packed >> 0) & 0xF) as f32;
                let nib1 = ((packed >> 4) & 0xF) as f32;
                let nib2 = ((packed >> 8) & 0xF) as f32;
                let nib3 = ((packed >> 12) & 0xF) as f32;
                let nib4 = ((packed >> 16) & 0xF) as f32;
                let nib5 = ((packed >> 20) & 0xF) as f32;
                let nib6 = ((packed >> 24) & 0xF) as f32;
                let nib7 = ((packed >> 28) & 0xF) as f32;

                // Match C's accumulation order exactly:
                //   acc += (nibble * scale + bias) * x[n]
                acc += (nib0 * scale + bias) * x[xb];
                acc += (nib1 * scale + bias) * x[xb + 1];
                acc += (nib2 * scale + bias) * x[xb + 2];
                acc += (nib3 * scale + bias) * x[xb + 3];
                acc += (nib4 * scale + bias) * x[xb + 4];
                acc += (nib5 * scale + bias) * x[xb + 5];
                acc += (nib6 * scale + bias) * x[xb + 6];
                acc += (nib7 * scale + bias) * x[xb + 7];
            }
        }
        out[row] = acc;
    }
}

// ---------------------------------------------------------------------------
// RMS Layer Normalisation
// ---------------------------------------------------------------------------

/// RMS normalisation: `out[i] = x[i] * w[i] / rms(x)`.
pub fn cpu_rms_norm(x: &[f32], w_bf16: &[u16], out: &mut [f32], dim: usize, eps: f32) {
    let sum_sq: f32 = x[..dim].iter().map(|&v| v * v).sum();
    let rms = (sum_sq / dim as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..dim {
        out[i] = x[i] * inv_rms * bf16_to_f32(w_bf16[i]);
    }
}

/// RMS norm without weights: `out[i] = x[i] / rms(x)`.
pub fn cpu_rms_norm_bare(x: &[f32], out: &mut [f32], dim: usize, eps: f32) {
    let sum_sq: f32 = x[..dim].iter().map(|&v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms;
    }
}

// ---------------------------------------------------------------------------
// SwiGLU activation
// ---------------------------------------------------------------------------

/// SwiGLU: `out[i] = silu(gate[i]) * up[i]`.
pub fn cpu_swiglu(gate: &[f32], up: &[f32], out: &mut [f32], dim: usize) {
    for i in 0..dim {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        out[i] = silu_g * up[i];
    }
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------

/// Logistic sigmoid: `1 / (1 + exp(-x))`.
#[inline]
pub fn cpu_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Softmax (in-place)
// ---------------------------------------------------------------------------

/// In-place softmax over `dim` elements: subtract max, exponentiate, normalise.
pub fn cpu_softmax(x: &mut [f32], dim: usize) {
    let max_val = x[..dim].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x[..dim]
        .iter_mut()
        .map(|v| {
            let e = (*v - max_val).exp();
            *v = e;
            e
        })
        .sum();
    let inv_sum = 1.0 / sum;
    for v in x[..dim].iter_mut() {
        *v *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Top-K
// ---------------------------------------------------------------------------

/// Find the `k` largest values in `scores[..dim]`, returning `(indices, values)`.
///
/// Uses a min-heap (`O(dim * log k)`), equivalent to the `USE_HEAP_TOPK` path
/// in the C++ reference, which is unconditionally enabled.
pub fn cpu_topk(scores: &[f32], dim: usize, k: usize) -> (Vec<i32>, Vec<f32>) {
    let mut values: Vec<f32> = Vec::with_capacity(k);
    let mut indices: Vec<i32> = Vec::with_capacity(k);

    for i in 0..dim {
        if values.len() < k {
            // --- insert into heap, bubble up ---
            values.push(scores[i]);
            indices.push(i as i32);
            let mut pos = values.len() - 1;
            while pos > 0 {
                let parent = (pos - 1) / 2;
                if values[parent] <= values[pos] {
                    break;
                }
                values.swap(pos, parent);
                indices.swap(pos, parent);
                pos = parent;
            }
        } else if scores[i] > values[0] {
            // --- replace heap root, bubble down ---
            values[0] = scores[i];
            indices[0] = i as i32;
            let mut pos = 0;
            loop {
                let left = 2 * pos + 1;
                let right = 2 * pos + 2;
                let mut smallest = pos;
                if left < k && values[left] < values[smallest] {
                    smallest = left;
                }
                if right < k && values[right] < values[smallest] {
                    smallest = right;
                }
                if smallest == pos {
                    break;
                }
                values.swap(pos, smallest);
                indices.swap(pos, smallest);
                pos = smallest;
            }
        }
    }

    (indices, values)
}

// ---------------------------------------------------------------------------
// Normalise top-K weights
// ---------------------------------------------------------------------------

/// Normalise top-K weights so that they sum to 1 (in-place, first `k` elements).
pub fn cpu_normalize_weights(weights: &mut [f32], k: usize) {
    let sum: f32 = weights[..k].iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for w in weights[..k].iter_mut() {
            *w *= inv;
        }
    }
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

/// Element-wise add: `dst[i] += src[i]`.
pub fn cpu_vec_add(dst: &mut [f32], src: &[f32], dim: usize) {
    for i in 0..dim {
        dst[i] += src[i];
    }
}

/// Element-wise multiply-add: `dst[i] += scale * src[i]`.
pub fn cpu_vec_madd(dst: &mut [f32], src: &[f32], scale: f32, dim: usize) {
    for i in 0..dim {
        dst[i] += scale * src[i];
    }
}

/// Element-wise multiply: `dst[i] = a[i] * b[i]`.
pub fn cpu_vec_mul(dst: &mut [f32], a: &[f32], b: &[f32], dim: usize) {
    for i in 0..dim {
        dst[i] = a[i] * b[i];
    }
}

/// Copy: `dst[..dim] = src[..dim]`.
pub fn cpu_vec_copy(dst: &mut [f32], src: &[f32], dim: usize) {
    dst[..dim].copy_from_slice(&src[..dim]);
}

/// Zero: set the first `dim` elements of `dst` to `0.0`.
pub fn cpu_vec_zero(dst: &mut [f32], dim: usize) {
    dst[..dim].fill(0.0);
}

// ---------------------------------------------------------------------------
// Argmax
// ---------------------------------------------------------------------------

/// Return the index of the maximum element in `x[..dim]`.
/// Returns `0` if `dim == 0`.
pub fn cpu_argmax(x: &[f32], dim: usize) -> i32 {
    x[..dim]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// SiLU activation (in-place)
// ---------------------------------------------------------------------------

/// In-place SiLU (Sigmoid Linear Unit): `x[i] = x[i] / (1 + exp(-x[i]))`.
pub fn cpu_silu(x: &mut [f32], dim: usize) {
    for v in x[..dim].iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

// ---------------------------------------------------------------------------
// Depthwise Conv1d (single step)
// ---------------------------------------------------------------------------

/// One step of depthwise Conv1d for incremental inference.
///
/// - `conv_state` has `(kernel_size - 1) * channels` elements (row-major,
///   `state[k][c] = conv_state[k * channels + c]`).
/// - `new_input` has `channels` elements (the current step's input).
/// - `weight_bf16` has `channels * kernel_size` elements (bf16, flattened:
///   `weight[c][k]`).
/// - `out` receives `channels` outputs, then SiLU is applied in-place.
pub fn cpu_conv1d_step(
    conv_state: &[f32],
    new_input: &[f32],
    weight_bf16: &[u16],
    out: &mut [f32],
    channels: usize,
    kernel_size: usize,
) {
    for c in 0..channels {
        let mut acc = 0.0f32;

        // Dot product with previous states (first kernel_size - 1 taps)
        for k in 0..(kernel_size - 1) {
            let w = bf16_to_f32(weight_bf16[c * kernel_size + k]);
            acc += conv_state[k * channels + c] * w;
        }

        // Dot product with the new input (last kernel tap)
        let w_last = bf16_to_f32(weight_bf16[c * kernel_size + (kernel_size - 1)]);
        acc += new_input[c] * w_last;

        out[c] = acc;
    }

    // Apply SiLU activation on the output
    cpu_silu(out, channels);
}

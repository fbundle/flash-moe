#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

// ============================================================================
// CPU computation kernels
// ============================================================================

// 4-bit dequant matvec: out[out_dim] = W * x[in_dim]
// W is stored as packed uint32 (8 x 4-bit values per uint32)
// scales/biases are bfloat16 per group
static void cpu_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size
) {
    int num_groups = in_dim / group_size;
    int packed_per_group = group_size / 8;
    int packed_cols = in_dim / 8;

    for (int row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t *w_row = W + row * packed_cols;
        const uint16_t *s_row = scales + row * num_groups;
        const uint16_t *b_row = biases + row * num_groups;

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(s_row[g]);
            float bias = bf16_to_f32(b_row[g]);
            int base_packed = g * packed_per_group;
            int base_x = g * group_size;

            for (int p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[base_packed + p];
                int x_base = base_x + p * 8;

                for (int n = 0; n < 8; n++) {
                    uint32_t nibble = (packed >> (n * 4)) & 0xF;
                    acc += ((float)nibble * scale + bias) * x[x_base + n];
                }
            }
        }
        out[row] = acc;
    }
}

#if USE_CPU_DEQUANT_FMA
// Optimized variant: precompute scale*x and bias*x once per group
// for the packed add path. Reduces inner-loop multiplies.
static void cpu_dequant_matvec_fma(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size
) {
    int num_groups = in_dim / group_size;
    int packed_per_group = group_size / 8;
    int packed_cols = in_dim / 8;

    for (int row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t *w_row = W + row * packed_cols;
        const uint16_t *s_row = scales + row * num_groups;
        const uint16_t *b_row = biases + row * num_groups;

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(s_row[g]);
            float bias = bf16_to_f32(b_row[g]);
            int base_packed = g * packed_per_group;
            int base_x = g * group_size;

            for (int p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[base_packed + p];
                int xb = base_x + p * 8;

                // Precompute scale*x and bias*x, use FMA: nibble*(scale*x) + (bias*x)
                float sx0 = scale * x[xb + 0], bx0 = bias * x[xb + 0];
                float sx1 = scale * x[xb + 1], bx1 = bias * x[xb + 1];
                float sx2 = scale * x[xb + 2], bx2 = bias * x[xb + 2];
                float sx3 = scale * x[xb + 3], bx3 = bias * x[xb + 3];
                float sx4 = scale * x[xb + 4], bx4 = bias * x[xb + 4];
                float sx5 = scale * x[xb + 5], bx5 = bias * x[xb + 5];
                float sx6 = scale * x[xb + 6], bx6 = bias * x[xb + 6];
                float sx7 = scale * x[xb + 7], bx7 = bias * x[xb + 7];

                acc += fmaf((float)((packed >>  0) & 0xF), sx0, bx0);
                acc += fmaf((float)((packed >>  4) & 0xF), sx1, bx1);
                acc += fmaf((float)((packed >>  8) & 0xF), sx2, bx2);
                acc += fmaf((float)((packed >> 12) & 0xF), sx3, bx3);
                acc += fmaf((float)((packed >> 16) & 0xF), sx4, bx4);
                acc += fmaf((float)((packed >> 20) & 0xF), sx5, bx5);
                acc += fmaf((float)((packed >> 24) & 0xF), sx6, bx6);
                acc += fmaf((float)((packed >> 28) & 0xF), sx7, bx7);
            }
        }
        out[row] = acc;
    }
}
#endif // USE_CPU_DEQUANT_FMA

// RMS normalization: out = x * w / rms(x)
static void cpu_rms_norm(const float *x, const uint16_t *w_bf16, float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / dim + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < dim; i++) {
        float weight = bf16_to_f32(w_bf16[i]);
        out[i] = x[i] * inv_rms * weight;
    }
}

// SwiGLU: out = silu(gate) * up
static void cpu_swiglu(const float *gate, const float *up, float *out, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

// Sigmoid
static float cpu_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Softmax over a vector
static void cpu_softmax(float *x, int dim) {
    float max_val = x[0];
    for (int i = 1; i < dim; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < dim; i++) {
        x[i] *= inv_sum;
    }
}

// Top-K: find K largest indices from scores[dim].
#if USE_HEAP_TOPK
// Min-heap implementation: O(dim * log(K)) vs O(dim * K) for selection sort.
// With K=8, ~2.5x fewer comparisons.
// Pattern from llama.cpp: maintain a min-heap of the K smallest values found so far.
static void cpu_topk(const float *scores, int dim, int K, int *indices, float *values) {
    // Build initial heap with first K elements
    int heap_size = 0;
    for (int i = 0; i < dim; i++) {
        if (heap_size < K) {
            // Insert into heap (bubble up)
            int pos = heap_size;
            while (pos > 0 && values[(pos - 1) / 2] > scores[i]) {
                values[pos] = values[(pos - 1) / 2];
                indices[pos] = indices[(pos - 1) / 2];
                pos = (pos - 1) / 2;
            }
            values[pos] = scores[i];
            indices[pos] = i;
            heap_size++;
        } else if (scores[i] > values[0]) {
            // Replace heap root with new value, then bubble down
            values[0] = scores[i];
            indices[0] = i;
            int pos = 0;
            while (1) {
                int left = 2 * pos + 1, right = 2 * pos + 2, smallest = pos;
                if (left < K && values[left] < values[smallest]) smallest = left;
                if (right < K && values[right] < values[smallest]) smallest = right;
                if (smallest == pos) break;
                float tmp_v = values[pos];
                int tmp_i = indices[pos];
                values[pos] = values[smallest];
                indices[pos] = indices[smallest];
                values[smallest] = tmp_v;
                indices[smallest] = tmp_i;
                pos = smallest;
            }
        }
    }
}
#else
// Simple selection sort: O(dim * K). Used when heap implementation is disabled.
static void cpu_topk(const float *scores, int dim, int K, int *indices, float *values) {
    for (int k = 0; k < K; k++) {
        values[k] = -1e30f;
        indices[k] = 0;
    }
    for (int i = 0; i < dim; i++) {
        int min_k = 0;
        for (int k = 1; k < K; k++) {
            if (values[k] < values[min_k]) min_k = k;
        }
        if (scores[i] > values[min_k]) {
            values[min_k] = scores[i];
            indices[min_k] = i;
        }
    }
}
#endif // USE_HEAP_TOPK

// Normalize top-K weights to sum to 1
static void cpu_normalize_weights(float *weights, int K) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += weights[k];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int k = 0; k < K; k++) weights[k] *= inv;
    }
}

// Element-wise add: dst += src
__attribute__((unused))
static void cpu_vec_add(float *dst, const float *src, int dim) {
    for (int i = 0; i < dim; i++) dst[i] += src[i];
}

// Element-wise multiply-add: dst += scale * src
static void cpu_vec_madd(float *dst, const float *src, float scale, int dim) {
    for (int i = 0; i < dim; i++) dst[i] += scale * src[i];
}

// Element-wise multiply: dst = a * b
__attribute__((unused))
static void cpu_vec_mul(float *dst, const float *a, const float *b, int dim) {
    for (int i = 0; i < dim; i++) dst[i] = a[i] * b[i];
}

// Copy
static void cpu_vec_copy(float *dst, const float *src, int dim) {
    memcpy(dst, src, dim * sizeof(float));
}

// Zero
__attribute__((unused))
static void cpu_vec_zero(float *dst, int dim) {
    memset(dst, 0, dim * sizeof(float));
}

// Argmax
int cpu_argmax(const float *x, int dim) {
    int best = 0;
    float best_val = x[0];
    for (int i = 1; i < dim; i++) {
        if (x[i] > best_val) {
            best_val = x[i];
            best = i;
        }
    }
    return best;
}

// SiLU activation
static void cpu_silu(float *x, int dim) {
    for (int i = 0; i < dim; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

// Conv1d depthwise: one step (for incremental inference)
// Input: conv_state[kernel_size-1][channels] + new_input[channels]
// Output: result[channels]
// Weight: [channels, kernel_size, 1] stored as bf16
// This is a depthwise conv1d: each channel is independent
static void cpu_conv1d_step(
    const float *conv_state,    // [(kernel_size-1) * channels] row-major
    const float *new_input,     // [channels]
    const uint16_t *weight_bf16, // [channels * kernel_size] flattened
    float *out,                 // [channels]
    int channels,
    int kernel_size
) {
    // For each channel, compute dot product of [conv_state..., new_input] with weight
    for (int c = 0; c < channels; c++) {
        float acc = 0.0f;
        // Process previous states from conv_state
        for (int k = 0; k < kernel_size - 1; k++) {
            float w = bf16_to_f32(weight_bf16[c * kernel_size + k]);
            acc += conv_state[k * channels + c] * w;
        }
        // Process new input (last position in kernel)
        float w = bf16_to_f32(weight_bf16[c * kernel_size + (kernel_size - 1)]);
        acc += new_input[c] * w;
        out[c] = acc;
    }
    // Apply SiLU
    cpu_silu(out, channels);
}


#endif // CPU_KERNELS_H

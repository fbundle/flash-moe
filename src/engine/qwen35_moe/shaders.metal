/*
 * shaders.metal — Optimized Metal compute shaders for 4-bit quantized MoE inference
 *
 * Sources:
 *   - MLX SDPA / quantized kernels: https://github.com/ml-explore/mlx (Apple)
 *     vendored at vendor/mlx/mlx/backend/metal/kernels/
 *   - Metal Flash Attention: https://github.com/philipturner/metal-flash-attention
 *     vendored at vendor/metal-flash-attention/
 *
 * Core operations:
 *   1. dequant_matvec_4bit: Naive 4-bit affine dequant matvec (reference)
 *   2. dequant_matvec_4bit_fast: SIMD-optimized with simd_sum reduction
 *   3. dequant_matvec_4bit_v3: Fully optimized — tiled threadgroup, vector loads,
 *      coalesced access, shared input cache. Target: <0.1ms per matmul.
 *   4. swiglu_fused / swiglu_fused_vec4: SwiGLU activation
 *   5. weighted_sum: combine expert outputs with routing weights
 *   6. rms_norm: RMS normalization
 *
 * Quantization format (MLX affine 4-bit, group_size=64):
 *   - Weights stored as uint32, each holding 8 x 4-bit values
 *   - Per-group scale and bias in bfloat16
 *   - Dequantized value = uint4_val * scale + bias
 *   - Groups of 64 elements share one (scale, bias) pair
 *
 * Matrix layout for expert projections:
 *   gate_proj/up_proj: [1024, 512] uint32 = [1024, 4096] logical (out=1024, in=4096)
 *   down_proj: [4096, 128] uint32 = [4096, 1024] logical (out=4096, in=1024)
 *
 *   Scales/biases: [out_dim, in_dim/group_size]
 *   gate/up scales: [1024, 64]   (4096/64 = 64 groups)
 *   down scales:    [4096, 16]   (1024/64 = 16 groups)
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Model dimension constants (Qwen3.5/3.6-35B-A3B)
// ============================================================================
#define HEAD_DIM        256
#define NUM_KV_HEADS    2
#define KV_DIM          (NUM_KV_HEADS * HEAD_DIM)  // 512
#define HEADS_PER_KV    8

// ============================================================================
// BFloat16 helpers
// ============================================================================

inline float bf16_to_f32(uint16_t bf16) {
    return as_type<float>(uint(bf16) << 16);
}

inline uint16_t f32_to_bf16(float f) {
    return uint16_t(as_type<uint>(f) >> 16);
}


// ============================================================================
// Kernel 1: 4-bit dequantized matrix-vector multiply (NAIVE — reference)
// ============================================================================

kernel void dequant_matvec_4bit(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    float acc = 0.0f;

    device const uint32_t* w_row = W_packed + tid * packed_cols;
    device const uint16_t* s_row = scales + tid * num_groups;
    device const uint16_t* b_row = biases + tid * num_groups;

    for (uint g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            for (uint n = 0; n < 8; n++) {
                uint nibble = (packed >> (n * 4)) & 0xF;
                float w_val = float(nibble) * scale + bias;
                acc += w_val * x[x_base + n];
            }
        }
    }

    out[tid] = acc;
}


// ============================================================================
// Kernel 1b: 4-bit dequant matvec — SIMD-optimized (legacy, kept for compat)
// ============================================================================

kernel void dequant_matvec_4bit_fast(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;

    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;

    device const uint32_t* w_row = W_packed + tgid * packed_cols;
    device const uint16_t* s_row = scales + tgid * num_groups;
    device const uint16_t* b_row = biases + tgid * num_groups;

    float acc = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint base_packed = g * packed_per_group;
        uint base_x = g * group_size;

        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[base_packed + p];
            uint x_base = base_x + p * 8;

            acc += (float((packed >>  0) & 0xF) * scale + bias) * x[x_base + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x[x_base + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x[x_base + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x[x_base + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x[x_base + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x[x_base + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x[x_base + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x[x_base + 7];
        }
    }

    threadgroup float shared[32];
    float simd_val = simd_sum(acc);

    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = shared[simd_lane];
        val = simd_sum(val);
        if (simd_lane == 0) {
            out[tgid] = val;
        }
    }
}

// ============================================================================
// Fused gate+up+SwiGLU: reads x ONCE, computes silu(gate(x)) * up(x)
// Saves one input read + one kernel dispatch per expert
// ============================================================================
kernel void fused_gate_up_swiglu(
    device const uint32_t* gate_W    [[buffer(0)]],
    device const uint16_t* gate_s    [[buffer(1)]],
    device const uint16_t* gate_b    [[buffer(2)]],
    device const uint32_t* up_W      [[buffer(3)]],
    device const uint16_t* up_s      [[buffer(4)]],
    device const uint16_t* up_b      [[buffer(5)]],
    device const float*    x         [[buffer(6)]],
    device float*          out       [[buffer(7)]],
    constant uint&         out_dim   [[buffer(8)]],
    constant uint&         in_dim    [[buffer(9)]],
    constant uint&         group_size [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= out_dim) return;
    uint num_groups = in_dim / group_size;
    uint packed_per_group = group_size / 8;
    uint packed_cols = in_dim / 8;
    device const uint32_t* gr = gate_W + tgid * packed_cols;
    device const uint16_t* gs = gate_s + tgid * num_groups;
    device const uint16_t* gb = gate_b + tgid * num_groups;
    device const uint32_t* ur = up_W   + tgid * packed_cols;
    device const uint16_t* us = up_s   + tgid * num_groups;
    device const uint16_t* ub = up_b   + tgid * num_groups;
    float ga = 0.0f, ua = 0.0f;
    for (uint g = lid; g < num_groups; g += tg_size) {
        float gsc = bf16_to_f32(gs[g]), gbi = bf16_to_f32(gb[g]);
        float usc = bf16_to_f32(us[g]), ubi = bf16_to_f32(ub[g]);
        uint bp = g * packed_per_group, bx = g * group_size;
        for (uint p = 0; p < packed_per_group; p++) {
            uint32_t gp = gr[bp+p], up = ur[bp+p];
            for (uint i = 0; i < 8; i++) {
                float xv = x[bx + p*8 + i];
                ga += (float((gp>>(i*4))&0xF)*gsc+gbi)*xv;
                ua += (float((up>>(i*4))&0xF)*usc+ubi)*xv;
            }
        }
    }
    threadgroup float sg[32], su[32];
    float rg = simd_sum(ga), ru = simd_sum(ua);
    uint sl = lid%32, si = lid/32, ns = (tg_size+31)/32;
    if (sl==0) { sg[si]=rg; su[si]=ru; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (si==0 && sl<ns) {
        float vg=simd_sum(sg[sl]), vu=simd_sum(su[sl]);
        if (sl==0) out[tgid] = (vg/(1.0f+exp(-vg))) * vu;
    }
}

// ============================================================================
// Kernel 1c: FULLY OPTIMIZED 4-bit dequant matvec
// ============================================================================
//
// Design for M3 Max (40-core GPU, SIMD width 32):
//
// Strategy: Each threadgroup handles ROWS_PER_TG output rows.
//   - Threadgroup size = 256 (8 SIMD groups of 32)
//   - Each SIMD group handles one output row
//   - Within a SIMD group, 32 threads split the input dimension
//   - Each thread processes in_dim/32 input elements using vector loads
//   - Reduction via simd_sum (single instruction)
//
// Memory optimizations:
//   - Input vector x cached in threadgroup shared memory (loaded once)
//   - uint4 vector loads for weights (128 bits = 32 nibbles per load)
//   - float4 vector loads for x (128 bits = 4 floats per load)
//   - Coalesced weight reads: adjacent threads read adjacent uint4 vectors
//
// For gate/up_proj [1024, 4096]: 1024/8 = 128 threadgroups, 256 threads each
//   - 128 * 256 = 32768 threads across 40 cores = good occupancy
//   - Each thread processes 4096/32 = 128 input elements = 16 uint32 packed words
//     = 4 uint4 loads per thread per row
//
// For down_proj [4096, 1024]: 4096/8 = 512 threadgroups
//   - Each thread processes 1024/32 = 32 input elements = 4 uint32 packed words
//     = 1 uint4 load per thread per row

// Number of output rows per threadgroup = number of SIMD groups (256/32 = 8)
#define ROWS_PER_TG 8

kernel void dequant_matvec_4bit_v3(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],     // which tile of rows
    uint lid    [[thread_position_in_threadgroup]],    // 0..255
    uint simd_lane  [[thread_index_in_simdgroup]],    // 0..31
    uint simd_group [[simdgroup_index_in_threadgroup]] // 0..7
) {
    // Which output row this SIMD group handles
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;      // uint32 columns per row
    uint num_groups  = in_dim / group_size;

    // ---- Cache input vector in threadgroup shared memory ----
    // Max in_dim = 4096, so we need 4096 floats = 16KB shared memory
    // This is well within the 32KB threadgroup memory limit on M3
    threadgroup float x_shared[4096];

    // Cooperative load: 256 threads load 4096 floats (16 per thread)
    // ALL threads must participate in this load + barrier, even if their
    // row is out of bounds. Early return before the barrier causes only
    // partial loading of x_shared, corrupting results for valid rows.
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Now safe to bail out for out-of-bounds rows
    if (row >= out_dim) return;

    // ---- Pointer setup for this row ----
    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    // ---- Each lane processes a strided slice of the packed columns ----
    // Lane k processes columns: k, k+32, k+64, ...
    // This gives coalesced reads: adjacent lanes read adjacent uint32 words.

    float acc = 0.0f;

    // Process packed columns in strides of 32 (one per SIMD lane)
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // Determine which group this column belongs to
        // packed_per_group = group_size / 8 = 64 / 8 = 8
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        // Dequantize 8 nibbles and multiply with cached x
        // Rearranged: (nibble * scale + bias) * x = nibble * (scale*x) + bias*x
        // Pre-compute scale*x and bias*x, then use FMA for dequant+multiply in one op.
        // This reduces per-nibble from (convert + mul + add + mul + add) to (convert + FMA + add).
        float sx0 = scale * x_shared[x_base + 0];  float bx0 = bias * x_shared[x_base + 0];
        float sx1 = scale * x_shared[x_base + 1];  float bx1 = bias * x_shared[x_base + 1];
        float sx2 = scale * x_shared[x_base + 2];  float bx2 = bias * x_shared[x_base + 2];
        float sx3 = scale * x_shared[x_base + 3];  float bx3 = bias * x_shared[x_base + 3];
        float sx4 = scale * x_shared[x_base + 4];  float bx4 = bias * x_shared[x_base + 4];
        float sx5 = scale * x_shared[x_base + 5];  float bx5 = bias * x_shared[x_base + 5];
        float sx6 = scale * x_shared[x_base + 6];  float bx6 = bias * x_shared[x_base + 6];
        float sx7 = scale * x_shared[x_base + 7];  float bx7 = bias * x_shared[x_base + 7];

        acc += fma(float((packed >>  0) & 0xF), sx0, bx0);
        acc += fma(float((packed >>  4) & 0xF), sx1, bx1);
        acc += fma(float((packed >>  8) & 0xF), sx2, bx2);
        acc += fma(float((packed >> 12) & 0xF), sx3, bx3);
        acc += fma(float((packed >> 16) & 0xF), sx4, bx4);
        acc += fma(float((packed >> 20) & 0xF), sx5, bx5);
        acc += fma(float((packed >> 24) & 0xF), sx6, bx6);
        acc += fma(float((packed >> 28) & 0xF), sx7, bx7);
    }

    // ---- SIMD reduction: sum across 32 lanes ----
    float sum = simd_sum(acc);

    // Lane 0 writes the result
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1f: 4-bit dequant matvec with LUT (eliminates uint→float conversions)
// ============================================================================
// Instead of converting each nibble to float (expensive conversion instruction),
// pre-compute a 16-entry LUT per group: lut[v] = float(v) * scale + bias.
// Then inner loop is just: acc += lut[nibble] * x_shared[i] — pure math, no conversions.
// The LUT is recomputed every group_size/8 iterations (amortized).

#define ROWS_PER_TG_V5 8

kernel void dequant_matvec_4bit_v5(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG_V5 + simd_group;
    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;
    uint packed_per_group = group_size / 8;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;
    uint prev_g = 0xFFFFFFFF;
    float lut[16];

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / packed_per_group;

        // Rebuild LUT when group changes
        if (g != prev_g) {
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);
            for (uint v = 0; v < 16; v++) {
                lut[v] = float(v) * scale + bias;
            }
            prev_g = g;
        }

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        acc += lut[(packed >>  0) & 0xF] * x_shared[x_base + 0];
        acc += lut[(packed >>  4) & 0xF] * x_shared[x_base + 1];
        acc += lut[(packed >>  8) & 0xF] * x_shared[x_base + 2];
        acc += lut[(packed >> 12) & 0xF] * x_shared[x_base + 3];
        acc += lut[(packed >> 16) & 0xF] * x_shared[x_base + 4];
        acc += lut[(packed >> 20) & 0xF] * x_shared[x_base + 5];
        acc += lut[(packed >> 24) & 0xF] * x_shared[x_base + 6];
        acc += lut[(packed >> 28) & 0xF] * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Kernel 1g: FP4_E2M1 dequant matvec
// ============================================================================
// Same structure as v3/v5 but decodes FP4 E2M1 nibbles via a static lookup
// table.  No bias — FP4's symmetric encoding handles the zero point natively.
//
//   dequant_val = fp4_lut[nibble] * scale
//
// The LUT is hard-coded to match the Rust-side FP4_E2M1_LUT.

constant float fp4_e2m1_lut[16] = {
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

kernel void dequant_matvec_fp4_e2m1(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/8]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(2)]],  // [in_dim]
    device float*          out        [[buffer(3)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(4)]],
    constant uint&         in_dim     [[buffer(5)]],
    constant uint&         group_size [[buffer(6)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        // Dequant with FP4 LUT: val = lut[nibble] * scale, then multiply with x
        acc += fp4_e2m1_lut[(packed >>  0) & 0xF] * scale * x_shared[x_base + 0];
        acc += fp4_e2m1_lut[(packed >>  4) & 0xF] * scale * x_shared[x_base + 1];
        acc += fp4_e2m1_lut[(packed >>  8) & 0xF] * scale * x_shared[x_base + 2];
        acc += fp4_e2m1_lut[(packed >> 12) & 0xF] * scale * x_shared[x_base + 3];
        acc += fp4_e2m1_lut[(packed >> 16) & 0xF] * scale * x_shared[x_base + 4];
        acc += fp4_e2m1_lut[(packed >> 20) & 0xF] * scale * x_shared[x_base + 5];
        acc += fp4_e2m1_lut[(packed >> 24) & 0xF] * scale * x_shared[x_base + 6];
        acc += fp4_e2m1_lut[(packed >> 28) & 0xF] * scale * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Kernel 1e: 2-bit affine dequant matvec (same structure as v3)
// ============================================================================
// Packs 16 x 2-bit values per uint32. Each value is 0-3, dequantized as:
//   val = uint2 * scale + bias (same affine quantization, just 2-bit range)
// Same group structure: group_size elements share one (scale, bias) pair.
// packed_cols = in_dim / 16 (16 values per uint32, vs 8 for 4-bit)

kernel void dequant_matvec_2bit(
    device const uint32_t* W_packed   [[buffer(0)]],  // [out_dim, in_dim/16]
    device const uint16_t* scales     [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const uint16_t* biases     [[buffer(2)]],  // [out_dim, num_groups] bf16
    device const float*    x          [[buffer(3)]],  // [in_dim]
    device float*          out        [[buffer(4)]],  // [out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint lid        [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;
    uint packed_cols = in_dim / 16;  // 16 values per uint32 for 2-bit
    uint num_groups  = in_dim / group_size;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row >= out_dim) return;

    device const uint32_t* w_row = W_packed + row * packed_cols;
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    float acc = 0.0f;

    // Each lane processes strided columns (16 values per uint32)
    for (uint col = simd_lane; col < packed_cols; col += 32) {
        // group_size/16 packed words per group
        uint g = col / (group_size / 16);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 16;

        // Unroll 16 x 2-bit extractions
        acc += (float((packed >>  0) & 0x3) * scale + bias) * x_shared[x_base +  0];
        acc += (float((packed >>  2) & 0x3) * scale + bias) * x_shared[x_base +  1];
        acc += (float((packed >>  4) & 0x3) * scale + bias) * x_shared[x_base +  2];
        acc += (float((packed >>  6) & 0x3) * scale + bias) * x_shared[x_base +  3];
        acc += (float((packed >>  8) & 0x3) * scale + bias) * x_shared[x_base +  4];
        acc += (float((packed >> 10) & 0x3) * scale + bias) * x_shared[x_base +  5];
        acc += (float((packed >> 12) & 0x3) * scale + bias) * x_shared[x_base +  6];
        acc += (float((packed >> 14) & 0x3) * scale + bias) * x_shared[x_base +  7];
        acc += (float((packed >> 16) & 0x3) * scale + bias) * x_shared[x_base +  8];
        acc += (float((packed >> 18) & 0x3) * scale + bias) * x_shared[x_base +  9];
        acc += (float((packed >> 20) & 0x3) * scale + bias) * x_shared[x_base + 10];
        acc += (float((packed >> 22) & 0x3) * scale + bias) * x_shared[x_base + 11];
        acc += (float((packed >> 24) & 0x3) * scale + bias) * x_shared[x_base + 12];
        acc += (float((packed >> 26) & 0x3) * scale + bias) * x_shared[x_base + 13];
        acc += (float((packed >> 28) & 0x3) * scale + bias) * x_shared[x_base + 14];
        acc += (float((packed >> 30) & 0x3) * scale + bias) * x_shared[x_base + 15];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1d: FULLY OPTIMIZED with uint4 vector loads
// ============================================================================
//
// Same structure as v3 but uses uint4 loads (128-bit / 16 bytes) to maximize
// memory bandwidth per thread. Each uint4 = 4 uint32 = 32 nibbles.
//
// For gate/up (packed_cols=512): each thread processes 512/32 = 16 uint32
//   = 4 uint4 loads per thread
// For down (packed_cols=128): each thread processes 128/32 = 4 uint32
//   = 1 uint4 load per thread

kernel void dequant_matvec_4bit_v4(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x          [[buffer(3)]],
    device float*          out        [[buffer(4)]],
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache input vector — ALL threads must participate before the barrier
    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    // Pointers — cast to uint4 for vector loads
    device const uint4* w_row_v = (device const uint4*)(W_packed + row * packed_cols);
    device const uint16_t* s_row = scales + row * num_groups;
    device const uint16_t* b_row = biases + row * num_groups;

    uint vec4_cols = packed_cols / 4;  // number of uint4 vectors per row

    float acc = 0.0f;

    // Each lane processes vec4_cols / 32 vectors (coalesced: adjacent lanes read adjacent uint4)
    for (uint vi = simd_lane; vi < vec4_cols; vi += 32) {
        uint4 packed4 = w_row_v[vi];

        // Each uint4 covers 4 * 8 = 32 input elements
        // Starting packed column index = vi * 4
        uint base_col = vi * 4;
        uint x_base = base_col * 8;  // starting input element

        // Process each of the 4 uint32 words in the uint4
        // Unroll all 4 words x 8 nibbles = 32 multiply-adds
        #pragma unroll
        for (uint w = 0; w < 4; w++) {
            uint32_t packed = packed4[w];
            uint col = base_col + w;
            uint g = col / (group_size / 8);
            float scale = bf16_to_f32(s_row[g]);
            float bias  = bf16_to_f32(b_row[g]);

            uint xb = x_base + w * 8;
            acc += (float((packed >>  0) & 0xF) * scale + bias) * x_shared[xb + 0];
            acc += (float((packed >>  4) & 0xF) * scale + bias) * x_shared[xb + 1];
            acc += (float((packed >>  8) & 0xF) * scale + bias) * x_shared[xb + 2];
            acc += (float((packed >> 12) & 0xF) * scale + bias) * x_shared[xb + 3];
            acc += (float((packed >> 16) & 0xF) * scale + bias) * x_shared[xb + 4];
            acc += (float((packed >> 20) & 0xF) * scale + bias) * x_shared[xb + 5];
            acc += (float((packed >> 24) & 0xF) * scale + bias) * x_shared[xb + 6];
            acc += (float((packed >> 28) & 0xF) * scale + bias) * x_shared[xb + 7];
        }
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}


// ============================================================================
// Kernel 1e: Multi-expert batched matvec
// ============================================================================
//
// Dispatch multiple experts simultaneously. The grid's Y dimension indexes
// the expert, so K experts' matmuls run as parallel threadgroups.
//
// Buffer layout: W_packed, scales, biases are arrays of K experts concatenated.
// x_inputs:  K input vectors concatenated [K * in_dim]
// out:       K output vectors concatenated [K * out_dim]
// expert_offsets: byte offset into W_packed buffer for each expert's weights
//                 (allows non-contiguous expert data in a shared buffer)

kernel void dequant_matvec_4bit_batched(
    device const uint32_t* W_packed   [[buffer(0)]],
    device const uint16_t* scales     [[buffer(1)]],
    device const uint16_t* biases     [[buffer(2)]],
    device const float*    x_inputs   [[buffer(3)]],  // [K, in_dim]
    device float*          out        [[buffer(4)]],  // [K, out_dim]
    constant uint&         out_dim    [[buffer(5)]],
    constant uint&         in_dim     [[buffer(6)]],
    constant uint&         group_size [[buffer(7)]],
    // Per-expert offsets into the weight/scale/bias buffers (in elements)
    device const uint*     w_offsets  [[buffer(8)]],  // [K] offset in uint32 elements
    device const uint*     s_offsets  [[buffer(9)]],  // [K] offset in uint16 elements
    device const uint*     b_offsets  [[buffer(10)]], // [K] offset in uint16 elements
    constant uint&         num_row_tiles [[buffer(11)]], // ceil(out_dim / ROWS_PER_TG)
    uint tgid_flat [[threadgroup_position_in_grid]],  // linearized (row_tile + expert * num_row_tiles)
    uint lid       [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // De-linearize: tgid_flat = row_tile + expert_k * num_row_tiles
    uint expert_k = tgid_flat / num_row_tiles;
    uint row_tile = tgid_flat % num_row_tiles;
    uint row = row_tile * ROWS_PER_TG + simd_group;
    if (row >= out_dim) return;

    uint packed_cols = in_dim / 8;
    uint num_groups  = in_dim / group_size;

    // Cache this expert's input vector
    threadgroup float x_shared[4096];
    device const float* x_k = x_inputs + expert_k * in_dim;
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x_k[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Point to this expert's weights
    device const uint32_t* w_row = W_packed + w_offsets[expert_k] + row * packed_cols;
    device const uint16_t* s_row = scales   + s_offsets[expert_k] + row * num_groups;
    device const uint16_t* b_row = biases   + b_offsets[expert_k] + row * num_groups;

    float acc = 0.0f;

    for (uint col = simd_lane; col < packed_cols; col += 32) {
        uint g = col / (group_size / 8);
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        uint32_t packed = w_row[col];
        uint x_base = col * 8;

        acc += (float((packed >>  0) & 0xF) * scale + bias) * x_shared[x_base + 0];
        acc += (float((packed >>  4) & 0xF) * scale + bias) * x_shared[x_base + 1];
        acc += (float((packed >>  8) & 0xF) * scale + bias) * x_shared[x_base + 2];
        acc += (float((packed >> 12) & 0xF) * scale + bias) * x_shared[x_base + 3];
        acc += (float((packed >> 16) & 0xF) * scale + bias) * x_shared[x_base + 4];
        acc += (float((packed >> 20) & 0xF) * scale + bias) * x_shared[x_base + 5];
        acc += (float((packed >> 24) & 0xF) * scale + bias) * x_shared[x_base + 6];
        acc += (float((packed >> 28) & 0xF) * scale + bias) * x_shared[x_base + 7];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[expert_k * out_dim + row] = sum;
    }
}


// ============================================================================
// Kernel 2: SwiGLU activation
// ============================================================================

kernel void swiglu_fused(
    device const float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant uint&      dim  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}

// Vectorized SwiGLU: process 4 elements per thread
kernel void swiglu_fused_vec4(
    device const float4* gate [[buffer(0)]],
    device const float4* up   [[buffer(1)]],
    device float4*       out  [[buffer(2)]],
    constant uint&       dim  [[buffer(3)]],  // original dim (must be multiple of 4)
    uint tid [[thread_position_in_grid]]
) {
    uint vec_dim = dim / 4;
    if (tid >= vec_dim) return;

    float4 g = gate[tid];
    float4 silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 2b: Batched SwiGLU for K experts
// ============================================================================

kernel void swiglu_fused_batched(
    device const float* gate [[buffer(0)]],  // [K * dim]
    device const float* up   [[buffer(1)]],  // [K * dim]
    device float*       out  [[buffer(2)]],  // [K * dim]
    constant uint&      dim  [[buffer(3)]],
    constant uint&      K    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = K * dim;
    if (tid >= total) return;

    float g = gate[tid];
    float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}


// ============================================================================
// Kernel 3: Weighted sum of expert outputs
// ============================================================================

kernel void weighted_sum(
    device const float* expert_outs [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device float*       out         [[buffer(2)]],
    constant uint&      K           [[buffer(3)]],
    constant uint&      dim         [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += weights[k] * expert_outs[k * dim + tid];
    }
    out[tid] = acc;
}


// ============================================================================
// Kernel 4: RMS Normalization — single-pass fused (MLX rms_single_row pattern)
// ============================================================================
//
// Computes RMS norm in one pass: sum of squares → rsqrt → normalize with bf16 weight.
// Eliminates the intermediate sum_sq buffer and second kernel dispatch.
//
// Dispatch: one threadgroup, 256 threads.  All threads cooperate on reduction.

kernel void rms_norm_fused_bf16(
    device const float*    x       [[buffer(0)]],
    device const uint16_t* weight  [[buffer(1)]],  // bf16 weights
    device float*          out     [[buffer(2)]],
    constant uint&         dim     [[buffer(3)]],
    constant float&        eps     [[buffer(4)]],
    uint lid       [[thread_position_in_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sums[32];

    // Pass 1: compute sum of squares
    float acc = 0.0f;
    for (uint i = lid; i < dim; i += tg_size) {
        float val = x[i];
        acc += val * val;
    }

    // SIMD reduction
    float simd_val = simd_sum(acc);
    if (simd_lane == 0) {
        shared_sums[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_sq = 0.0f;
    uint num_simd = (tg_size + 31) / 32;
    if (simd_group == 0 && simd_lane < num_simd) {
        sum_sq = simd_sum(shared_sums[simd_lane]);
    }

    // Broadcast sum_sq to all threads
    threadgroup float broadcast_sum = 0.0f;
    if (simd_group == 0 && simd_lane == 0) {
        broadcast_sum = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_sq = broadcast_sum;

    // Compute rms
    float inv_rms = metal::precise::rsqrt(sum_sq / float(dim) + eps);

    // Pass 2: normalize and write output
    for (uint i = lid; i < dim; i += tg_size) {
        float w = bf16_to_f32(weight[i]);
        out[i] = x[i] * inv_rms * w;
    }
}


// ============================================================================
// Kernel 4b: RMS Normalization — two-pass (legacy, kept for compat)
// ============================================================================

kernel void rms_norm_sum_sq(
    device const float* x       [[buffer(0)]],
    device float*       sum_sq  [[buffer(1)]],
    constant uint&      dim     [[buffer(2)]],
    uint tid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared[32];

    float acc = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        acc += val * val;
    }

    float simd_val = simd_sum(acc);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;

    if (simd_lane == 0) {
        shared[simd_group] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float val = (simd_lane < (tg_size + 31) / 32) ? shared[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            sum_sq[0] = val;
        }
    }
}

kernel void rms_norm_apply(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device const float* sum_sq  [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      dim     [[buffer(4)]],
    constant float&     eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    out[tid] = x[tid] * rms * weight[tid];
}


// ============================================================================
// Kernel 4b: RMS Normalization with bf16 weights
// ============================================================================
// Same as rms_norm_apply but reads weights as bfloat16 (uint16_t) and
// converts to float32 inline. Used in the fused o_proj+norm+routing path
// where norm weights come directly from the mmap'd weight file (bf16).

kernel void rms_norm_apply_bf16(
    device const float*    x       [[buffer(0)]],
    device const uint16_t* weight  [[buffer(1)]],  // bf16 weights
    device const float*    sum_sq  [[buffer(2)]],
    device float*          out     [[buffer(3)]],
    constant uint&         dim     [[buffer(4)]],
    constant float&        eps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float rms = rsqrt(sum_sq[0] / float(dim) + eps);
    float w = bf16_to_f32(weight[tid]);
    out[tid] = x[tid] * rms * w;
}


// ============================================================================
// Kernel 5: Residual add
// ============================================================================
// out[i] = a[i] + b[i]
// Used to fuse the residual connection into a GPU command buffer,
// eliminating a CPU round-trip between o_proj and routing.

kernel void residual_add(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device float*       out [[buffer(2)]],
    constant uint&      dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    out[tid] = a[tid] + b[tid];
}


// ============================================================================
// Kernel 6: Fused SDPA — scores + online softmax + value aggregation
// ============================================================================
//
// Replaces the three-kernel attn_scores/softmax/values pipeline with a single
// fused kernel.  Adapted from MLX's sdpa_vector pattern.
//
// One threadgroup per query head (grid = [num_q, 1]).  Within the TG, 8 SIMD
// groups split the KV sequence — each handles every 8th position.  Online
// softmax (running max / exp-sum, base-2 exponents) avoids materializing the
// full [num_heads, seq_len] score buffer.  After the KV loop, each SIMD group
// publishes its rescaled partial outputs to shared memory, then every lane
// sums across groups for its v_per_thread output elements.
//
// TG: 256 threads (8 SIMD groups x 32 lanes), head_dim=256 → v_per_thread=8.
// GQA mapping: kv_head = head / heads_per_kv.

kernel void attn_sdpa_fused(
    device const float* Q          [[buffer(0)]],   // [num_heads, HEAD_DIM]
    device const float* K_cache    [[buffer(1)]],   // [max_seq, KV_DIM]
    device const float* V_cache    [[buffer(2)]],   // [max_seq, KV_DIM]
    device float*       output     [[buffer(3)]],   // [num_heads, HEAD_DIM]
    constant uint&      seq_len    [[buffer(4)]],   // current sequence length
    constant float&     scale      [[buffer(5)]],   // 1/sqrt(HEAD_DIM)
    uint tgid   [[threadgroup_position_in_grid]],   // = query head index
    uint simd_lane  [[thread_index_in_simdgroup]],  // 0..31
    uint simd_group [[simdgroup_index_in_threadgroup]] // 0..7
) {
    constexpr uint BD = 32;                  // SIMD width
    constexpr uint BN = 8;                   // SIMD groups per TG
    constexpr uint V = HEAD_DIM / BD;        // 256/32 = 8

    uint h    = tgid;  // one TG per query head (no tiling)
    uint kv_h = h / HEADS_PER_KV;

    device const float* qh = Q + h * HEAD_DIM;
    device const float* k_base = K_cache + kv_h * HEAD_DIM;
    device const float* v_base = V_cache + kv_h * HEAD_DIM;

    float q_vals[V];
    float o_vals[V] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Pre-scale Q by log2(e) so inner loop can use fast::exp2 instead of exp.
    constexpr float log2_e = 1.442695041f;
    float q_scale = scale * log2_e;
    uint elem_base = simd_lane * V;  // 0, 8, 16, …, 248
    for (uint j = 0; j < V; j++) {
        q_vals[j] = q_scale * qh[elem_base + j];
    }

    // Online softmax state (identical across all lanes in a SIMD group
    // because simd_sum broadcasts the full dot-product to every lane).
    float max_score = -1e30f;
    float sum_exp   = 0.0f;

    // KV loop — each SIMD group handles a strided subset of positions
    for (uint pos = simd_group; pos < seq_len; pos += BN) {
        device const float* kp = k_base + pos * KV_DIM + elem_base;
        device const float* vp = v_base + pos * KV_DIM + elem_base;

        float score = 0.0f;
        for (uint j = 0; j < V; j++) {
            score += q_vals[j] * kp[j];
        }
        score = simd_sum(score);  // full dot product across all 256 elements

        float new_max   = max(max_score, score);
        float factor    = fast::exp2(max_score - new_max);
        float exp_score = fast::exp2(score - new_max);

        max_score = new_max;
        sum_exp   = sum_exp * factor + exp_score;

        for (uint j = 0; j < V; j++) {
            o_vals[j] = o_vals[j] * factor + exp_score * vp[j];
        }
    }

    // ── Merge partial results across SIMD groups ──

    // sg_max: per-group maxima (size BD so simd_lane indexing is safe)
    // sg_sum: per-group rescaled exp-sums
    // sg_partial: per-group partial outputs, indexed [group * HEAD_DIM + elem]
    threadgroup float sg_max[BD];
    threadgroup float sg_sum[BN];
    threadgroup float sg_partial[BN * BD * V];  // 8 * 256 = 2048

    // Initialize sg_max so lanes 8..31 don't inject garbage into simd_max.
    sg_max[simd_lane] = -1e30f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Lane 0 of each SIMD group publishes its group's max_score.
    if (simd_lane == 0) {
        sg_max[simd_group] = max_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global max across all 8 SIMD groups.
    float local_max  = sg_max[simd_lane];
    float global_max = simd_max(local_max);

    // Each group rescales its partials using its own max (broadcast from lane 0).
    float group_max     = simd_broadcast_first(max_score);
    float group_sum     = simd_broadcast_first(sum_exp);
    float rescale       = fast::exp2(group_max - global_max);
    float rescaled_sum  = group_sum * rescale;

    for (uint j = 0; j < V; j++) {
        o_vals[j] *= rescale;
    }

    // Publish rescaled partial outputs and per-group sum.
    for (uint j = 0; j < V; j++) {
        sg_partial[simd_group * HEAD_DIM + elem_base + j] = o_vals[j];
    }
    if (simd_lane == 0) {
        sg_sum[simd_group] = rescaled_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each lane sums its V output elements across all 8 SIMD groups.
    for (uint j = 0; j < V; j++) {
        float sum = 0.0f;
        for (uint g = 0; g < BN; g++) {
            sum += sg_partial[g * HEAD_DIM + elem_base + j];
        }
        o_vals[j] = sum;
    }

    // Global sum of rescaled per-group exp-sums.
    float local_sum  = (simd_lane < BN) ? sg_sum[simd_lane] : 0.0f;
    float global_sum = simd_sum(local_sum);

    // Normalize and write output.
    for (uint j = 0; j < V; j++) {
        o_vals[j] = (global_sum == 0.0f) ? 0.0f : (o_vals[j] / global_sum);
    }

    device float* out_ptr = output + h * HEAD_DIM + elem_base;
    for (uint j = 0; j < V; j++) {
        out_ptr[j] = o_vals[j];
    }
}


// ============================================================================
// Kernel 6a: Fused SDPA — block pass (KV-sequence tiled into 32-pos blocks)
// ============================================================================
//
// One threadgroup per (query head, KV block).  The KV sequence is split into
// blocks of 32 positions; each TG processes one block with online softmax,
// then writes partial results (max, sum, output[HEAD_DIM]) to an intermediate
// buffer.  A second reduce kernel merges across blocks per head.
//
// Grid: 2D [num_q, num_blocks] where num_blocks = (seq_len + 31) / 32.
// TG:   256 threads (8 SIMD groups x 32 lanes).

kernel void attn_sdpa_block(
    device const float* Q          [[buffer(0)]],   // [num_heads, HEAD_DIM]
    device const float* K_cache    [[buffer(1)]],   // [max_seq, KV_DIM]
    device const float* V_cache    [[buffer(2)]],   // [max_seq, KV_DIM]
    device float*       partials   [[buffer(3)]],   // [num_q * num_blocks * stride]
    constant uint&      seq_len    [[buffer(4)]],
    constant uint&      num_blocks [[buffer(5)]],
    constant float&     scale      [[buffer(6)]],
    uint2 tid   [[threadgroup_position_in_grid]],    // (head, block)
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint BD = 32;
    constexpr uint BN = 8;
    constexpr uint V  = HEAD_DIM / BD;       // 8
    constexpr uint STRIDE = 2 + HEAD_DIM;     // max, sum + 256 output = 258

    uint h     = tid.x;
    uint block = tid.y;
    uint kv_h  = h / HEADS_PER_KV;

    // Range of KV positions for this block
    uint block_start = block * 32;
    uint block_end   = min(block_start + 32, seq_len);
    if (block_start >= seq_len) return;

    device const float* qh = Q + h * HEAD_DIM;
    device const float* k_base = K_cache + kv_h * HEAD_DIM;
    device const float* v_base = V_cache + kv_h * HEAD_DIM;

    float q_vals[V];
    float o_vals[V] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    constexpr float log2_e = 1.442695041f;
    float q_scale = scale * log2_e;
    uint elem_base = simd_lane * V;
    for (uint j = 0; j < V; j++) {
        q_vals[j] = q_scale * qh[elem_base + j];
    }

    float max_score = -1e30f;
    float sum_exp   = 0.0f;

    // KV loop over this block — each SIMD group handles strided positions
    for (uint pos = block_start + simd_group; pos < block_end; pos += BN) {
        device const float* kp = k_base + pos * KV_DIM + elem_base;
        device const float* vp = v_base + pos * KV_DIM + elem_base;

        float score = 0.0f;
        for (uint j = 0; j < V; j++) {
            score += q_vals[j] * kp[j];
        }
        score = simd_sum(score);

        float new_max   = max(max_score, score);
        float factor    = fast::exp2(max_score - new_max);
        float exp_score = fast::exp2(score - new_max);

        max_score = new_max;
        sum_exp   = sum_exp * factor + exp_score;

        for (uint j = 0; j < V; j++) {
            o_vals[j] = o_vals[j] * factor + exp_score * vp[j];
        }
    }

    // ── Merge across SIMD groups within the TG ──
    threadgroup float sg_max[BD];
    threadgroup float sg_sum[BN];
    threadgroup float sg_partial[BN * BD * V];  // 8 * 256 = 2048

    sg_max[simd_lane] = -1e30f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lane == 0) {
        sg_max[simd_group] = max_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max  = sg_max[simd_lane];
    float global_max = simd_max(local_max);

    float group_max    = simd_broadcast_first(max_score);
    float group_sum    = simd_broadcast_first(sum_exp);
    float rescale      = fast::exp2(group_max - global_max);
    float rescaled_sum = group_sum * rescale;

    for (uint j = 0; j < V; j++) {
        o_vals[j] *= rescale;
    }

    for (uint j = 0; j < V; j++) {
        sg_partial[simd_group * HEAD_DIM + elem_base + j] = o_vals[j];
    }
    if (simd_lane == 0) {
        sg_sum[simd_group] = rescaled_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < V; j++) {
        float sum = 0.0f;
        for (uint g = 0; g < BN; g++) {
            sum += sg_partial[g * HEAD_DIM + elem_base + j];
        }
        o_vals[j] = sum;
    }

    float local_sum  = (simd_lane < BN) ? sg_sum[simd_lane] : 0.0f;
    float tg_sum     = simd_sum(local_sum);

    // Write partial results: {max, sum, output[HEAD_DIM]}
    uint p_base = (h * num_blocks + block) * STRIDE;
    if (simd_lane == 0 && simd_group == 0) {
        partials[p_base]     = global_max;
        partials[p_base + 1] = tg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < V; j++) {
        partials[p_base + 2 + elem_base + j] = o_vals[j];
    }
}


// ============================================================================
// Kernel 6b: Fused SDPA — reduce pass (merge block partials per head)
// ============================================================================
//
// One threadgroup per query head.  Reads block-pass partials, finds the global
// max across all blocks, rescales each block's output, sums them, normalizes,
// and writes the final attention output.
//
// Grid: 1D [num_q, 1].
// TG:   256 threads (8 SIMD groups x 32 lanes).

kernel void attn_sdpa_reduce(
    device const float* partials   [[buffer(0)]],   // [num_q * num_blocks * stride]
    device float*       output     [[buffer(1)]],   // [num_q, HEAD_DIM]
    constant uint&      num_blocks [[buffer(2)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint BD = 32;
    constexpr uint BN = 8;
    constexpr uint V  = HEAD_DIM / BD;       // 8
    constexpr uint STRIDE = 2 + HEAD_DIM;     // 258

    uint h = tgid;
    uint elem_base = simd_lane * V;

    // Pass 1: find global max across all blocks
    float global_max = -1e30f;
    for (uint b = simd_group * BD + simd_lane; b < num_blocks; b += BN * BD) {
        float b_max = partials[(h * num_blocks + b) * STRIDE];
        global_max = max(global_max, b_max);
    }
    float simd_global = simd_max(global_max);
    // Reduce across SIMD groups
    threadgroup float tg_max[BD];
    tg_max[simd_lane] = -1e30f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tg_max[simd_group] = simd_global;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = simd_max(tg_max[simd_lane]);

    // Pass 2: each SIMD group sums its strided subset of blocks
    float o_vals[V] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float global_sum = 0.0f;

    for (uint b = simd_group; b < num_blocks; b += BN) {
        uint p_base = (h * num_blocks + b) * STRIDE;
        float b_max = partials[p_base];
        float b_sum = partials[p_base + 1];

        float rescale = fast::exp2(b_max - global_max);
        global_sum += b_sum * rescale;

        for (uint j = 0; j < V; j++) {
            o_vals[j] += partials[p_base + 2 + elem_base + j] * rescale;
        }
    }

    // ── Merge o_vals and global_sum across SIMD groups ──
    threadgroup float tg_sum[BN];
    threadgroup float tg_partial[BN * BD * V];  // 8 * 256 = 2048

    // Publish per-group sums and partial outputs
    if (simd_lane == 0) {
        tg_sum[simd_group] = global_sum;
    }
    for (uint j = 0; j < V; j++) {
        tg_partial[simd_group * HEAD_DIM + elem_base + j] = o_vals[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-group sum of global_sum
    float lane_sum  = (simd_lane < BN) ? tg_sum[simd_lane] : 0.0f;
    global_sum = simd_sum(lane_sum);

    // Cross-group sum of o_vals
    for (uint j = 0; j < V; j++) {
        float sum = 0.0f;
        for (uint g = 0; g < BN; g++) {
            sum += tg_partial[g * HEAD_DIM + elem_base + j];
        }
        o_vals[j] = sum;
    }

    // Normalize and write
    for (uint j = 0; j < V; j++) {
        o_vals[j] = (global_sum == 0.0f) ? 0.0f : (o_vals[j] / global_sum);
    }

    device float* out_ptr = output + h * HEAD_DIM + elem_base;
    for (uint j = 0; j < V; j++) {
        out_ptr[j] = o_vals[j];
    }
}


// ============================================================================
// Kernel 6c: Batched GPU attention scores (Q @ K^T, scaled) — all heads at once
// ============================================================================
//
// Computes scores[h, p] = sum_d(Q[h, d] * K[p, kv_h*head_dim + d]) * scale
// for all heads h in [0, num_heads) and positions p in [0, seq_len).
//
// Grid: linearized (pos + h * num_seq_tgs) — one threadgroup per (position, head).
// Each threadgroup of 256 threads reduces over head_dim=256.
//
// GQA mapping: kv_head = h / heads_per_kv (e.g. 16 query heads share 1 KV head)
//
// Output layout: scores[h * seq_stride + p] where seq_stride = MAX_SEQ_LEN

kernel void attn_scores_batched(
    device const float* Q          [[buffer(0)]],  // [num_heads, head_dim]
    device const float* K_cache    [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       scores     [[buffer(2)]],  // [num_heads, seq_stride]
    constant uint&      head_dim   [[buffer(3)]],  // 256
    constant uint&      kv_dim     [[buffer(4)]],  // 512
    constant uint&      seq_len    [[buffer(5)]],  // current seq length
    constant uint&      seq_stride [[buffer(6)]],  // MAX_SEQ_LEN
    constant float&     scale      [[buffer(7)]],  // 1/sqrt(head_dim)
    constant uint&      heads_per_kv [[buffer(8)]], // 16 (GQA ratio)
    constant uint&      num_seq_tgs  [[buffer(9)]],  // = seq_len
    uint tgid  [[threadgroup_position_in_grid]],    // linearized: pos + h * num_seq_tgs
    uint lid   [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint pos = tgid % num_seq_tgs;
    uint h = tgid / num_seq_tgs;
    if (pos >= seq_len) return;

    uint kv_h = h / heads_per_kv;
    device const float* qh = Q + h * head_dim;
    device const float* kp = K_cache + pos * kv_dim + kv_h * head_dim;

    float acc = 0.0f;
    for (uint d = lid; d < head_dim; d += tg_size) {
        acc += qh[d] * kp[d];
    }

    // SIMD reduction
    float simd_val = simd_sum(acc);
    threadgroup float shared[32];
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared[simd_group] = simd_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane < num_simd_groups) {
        float val = simd_sum(shared[simd_lane]);
        if (simd_lane == 0) {
            scores[h * seq_stride + pos] = val * scale;
        }
    }
}


// ============================================================================
// Kernel 7: Batched softmax — one threadgroup per head
// ============================================================================

kernel void attn_softmax_batched(
    device float*    scores     [[buffer(0)]],  // [num_heads, seq_stride]
    constant uint&   seq_len    [[buffer(1)]],
    constant uint&   seq_stride [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],     // head index
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    device float* s = scores + tgid * seq_stride;

    // Pass 1: find max
    threadgroup float shared_max[32];
    float local_max = -1e30f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        local_max = max(local_max, s[i]);
    }
    float sm = simd_max(local_max);
    uint simd_lane = lid % 32;
    uint simd_group = lid / 32;
    uint num_simd_groups = (tg_size + 31) / 32;
    if (simd_lane == 0) shared_max[simd_group] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e30f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_max = simd_max(shared_max[simd_lane]);
    }
    threadgroup float broadcast_max;
    if (lid == 0) broadcast_max = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = broadcast_max;

    // Pass 2: exp and sum
    threadgroup float shared_sum[32];
    float local_sum = 0.0f;
    for (uint i = lid; i < seq_len; i += tg_size) {
        float val = exp(s[i] - global_max);
        s[i] = val;
        local_sum += val;
    }
    float simd_s = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_group] = simd_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_sum = 0.0f;
    if (simd_group == 0 && simd_lane < num_simd_groups) {
        global_sum = simd_sum(shared_sum[simd_lane]);
    }
    threadgroup float broadcast_sum;
    if (lid == 0) broadcast_sum = global_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_sum = broadcast_sum;

    // Pass 3: normalize
    float inv_sum = 1.0f / global_sum;
    for (uint i = lid; i < seq_len; i += tg_size) {
        s[i] *= inv_sum;
    }
}


// ============================================================================
// Kernel 8: Batched attention value aggregation (scores @ V) — all heads
// ============================================================================
//
// For each head h: output[h*head_dim + d] = sum_p(scores[h*seq_stride+p] * V[p*kv_dim + kv_h*head_dim + d])
//
// Grid: linearized over (head_dim * num_heads) — one thread per (dimension, head).

kernel void attn_values_batched(
    device const float* scores   [[buffer(0)]],  // [num_heads, seq_stride]
    device const float* V_cache  [[buffer(1)]],  // [max_seq, kv_dim]
    device float*       out      [[buffer(2)]],  // [num_heads, head_dim]
    constant uint&      head_dim  [[buffer(3)]],  // 256
    constant uint&      kv_dim    [[buffer(4)]],  // 512
    constant uint&      seq_len   [[buffer(5)]],
    constant uint&      seq_stride [[buffer(6)]],
    constant uint&      heads_per_kv [[buffer(7)]],
    uint tid [[thread_position_in_grid]]          // linearized: d + h * head_dim
) {
    uint d = tid % head_dim;
    uint h = tid / head_dim;

    uint kv_h = h / heads_per_kv;
    device const float* s = scores + h * seq_stride;

    float acc = 0.0f;
    for (uint p = 0; p < seq_len; p++) {
        acc += s[p] * V_cache[p * kv_dim + kv_h * head_dim + d];
    }
    out[h * head_dim + d] = acc;
}


// ============================================================================
// Kernel 9: Sigmoid element-wise gate
// ============================================================================
// out[i] = x[i] * sigmoid(gate[i])

kernel void sigmoid_gate(
    device float*       x_out  [[buffer(0)]],  // [dim] in/out
    device const float* gate   [[buffer(1)]],  // [dim] gate values
    constant uint&      dim    [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float g = 1.0f / (1.0f + exp(-gate[tid]));
    x_out[tid] = x_out[tid] * g;
}


// ============================================================================
// Kernel 10: GatedDeltaNet linear attention step (single token, all heads)
// ============================================================================
//
// Implements the GatedDeltaNet recurrence for autoregressive generation:
//   1. State decay:  S[vi][ki] *= g_decay
//   2. Memory read:  kv_mem[vi] = sum_ki(S[vi][ki] * k[ki])
//   3. Delta:        delta[vi] = (v[vi] - kv_mem[vi]) * beta_gate
//   4. State update: S[vi][ki] += k[ki] * delta[vi]
//   5. Output:       out[vi] = sum_ki(S[vi][ki] * q[ki])
//
// Dispatch: 64 threadgroups (one per v-head), 128 threads each (one per vi).
// Each thread owns one row S[head_id][vi][:] of the 128x128 state matrix.
//
// State layout: [64 * 128 * 128] float = 4MB total, persisted across tokens.
// k-head sharing: 4 v-heads share 1 k-head (64 v-heads / 16 k-heads).

kernel void gated_delta_net_step(
    device float *state,             // [64 * 128 * 128] persistent state
    device const float *q,           // [2048] (16 k-heads * 128)
    device const float *k,           // [2048] (16 k-heads * 128)
    device const float *v,           // [8192] (64 v-heads * 128)
    device const float *g_decay,     // [64] per v-head
    device const float *beta_gate,   // [64] per v-head
    device float *output,            // [8192] (64 v-heads * 128)
    constant uint &k_heads_per_v,    // = 4
    uint head_id [[threadgroup_position_in_grid]],
    uint vi [[thread_position_in_threadgroup]]
) {
    uint kh = head_id / k_heads_per_v;
    float g = g_decay[head_id];
    float beta = beta_gate[head_id];

    uint state_base = head_id * 128 * 128 + vi * 128;
    uint k_base = kh * 128;
    uint v_base = head_id * 128;

    // Step 1+2: Decay state row and compute kv_mem = dot(S[vi][:], k[:])
    float kv_mem = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        float s = state[state_base + ki] * g;
        state[state_base + ki] = s;
        kv_mem += s * k[k_base + ki];
    }

    // Step 3+4: Delta update — S[vi][ki] += k[ki] * delta
    float delta = (v[v_base + vi] - kv_mem) * beta;
    for (uint ki = 0; ki < 128; ki++) {
        state[state_base + ki] += k[k_base + ki] * delta;
    }

    // Step 5: Output — out[vi] = dot(S[vi][:], q[:])
    float out_val = 0.0f;
    for (uint ki = 0; ki < 128; ki++) {
        out_val += state[state_base + ki] * q[k_base + ki];
    }
    output[v_base + vi] = out_val;
}


// ============================================================================
// Kernel 11: Conv1d depthwise step (single token, incremental inference)
// ============================================================================
//
// Depthwise 1D convolution for one new input token:
//   output[c] = sum_k(history[k][c] * weight[c][k]) + input[c] * weight[c][3]
//   then SiLU activation: output[c] = output[c] / (1 + exp(-output[c]))
//
// After computing, shifts the history buffer left and appends the new input.
//
// Weight layout: [channels * kernel_size] bf16, weight[c * kernel_size + k]
// Conv state layout: [(kernel_size-1) * channels] row-major, state[k * channels + c]
// kernel_size = 4 (hardcoded), so 3 history slots + 1 new input.
//
// Dispatch: conv_dim threads (12288), one per channel.

kernel void conv1d_step(
    device float *conv_state,         // [(kernel_size-1) * conv_dim] = [3 * conv_dim]
    device const float *input,        // [conv_dim] current input
    device const uint16_t *weights,   // [conv_dim * 4] bf16 as uint16
    device float *output,             // [conv_dim] convolution output
    constant uint &conv_dim,          // = 12288
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= conv_dim) return;

    // Convolution: dot product of history + new input with weights
    // weight layout: weight[c * 4 + k] for channel c, position k
    uint w_base = idx * 4;
    float acc = 0.0f;

    // 3 history slots (k=0,1,2)
    acc += conv_state[0 * conv_dim + idx] * bf16_to_f32(weights[w_base + 0]);
    acc += conv_state[1 * conv_dim + idx] * bf16_to_f32(weights[w_base + 1]);
    acc += conv_state[2 * conv_dim + idx] * bf16_to_f32(weights[w_base + 2]);

    // New input (k=3)
    float inp = input[idx];
    acc += inp * bf16_to_f32(weights[w_base + 3]);

    // SiLU activation
    output[idx] = acc / (1.0f + exp(-acc));

    // Shift history: move slots 1,2 -> 0,1, append input at slot 2
    conv_state[0 * conv_dim + idx] = conv_state[1 * conv_dim + idx];
    conv_state[1 * conv_dim + idx] = conv_state[2 * conv_dim + idx];
    conv_state[2 * conv_dim + idx] = inp;
}


// ============================================================================
// Kernel 12: Per-head RMS normalize for q and k vectors
// ============================================================================
// q: [num_k_heads * key_dim], k: [num_k_heads * key_dim]
// Normalize each head independently, then scale by 1/sqrt(key_dim)^2 for q, 1/sqrt(key_dim) for k
// Dispatch: num_k_heads threadgroups, key_dim threads each

kernel void rms_norm_qk(
    device float *q,              // [num_k_heads * key_dim] in/out
    device float *k,              // [num_k_heads * key_dim] in/out
    constant uint &key_dim,       // = 128
    constant float &inv_scale,    // = 1/sqrt(key_dim)
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint base = head * key_dim;

    // RMS norm for q
    threadgroup float q_sum_sq;
    if (tid == 0) q_sum_sq = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qval = (tid < key_dim) ? q[base + tid] : 0;
    // Use threadgroup atomic add for sum of squares
    float q_sq_local = qval * qval;
    // Simple reduction: thread 0 accumulates (key_dim=128, fits in one pass)
    threadgroup float q_partial[128];
    q_partial[tid] = q_sq_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < key_dim; i++) s += q_partial[i];
        q_sum_sq = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_inv_rms = rsqrt(q_sum_sq / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        q[base + tid] = qval * q_inv_rms * inv_scale * inv_scale;  // q gets extra scale
    }

    // RMS norm for k
    threadgroup float k_sum_sq;
    float kval = (tid < key_dim) ? k[base + tid] : 0;
    threadgroup float k_partial[128];
    k_partial[tid] = kval * kval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < key_dim; i++) s += k_partial[i];
        k_sum_sq = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float k_inv_rms = rsqrt(k_sum_sq / float(key_dim) + 1e-6f);
    if (tid < key_dim) {
        k[base + tid] = kval * k_inv_rms * inv_scale;
    }
}


// ============================================================================
// Kernel 13: Compute g_decay and beta_gate for GatedDeltaNet
// ============================================================================
// Per v-head: g_decay = exp(-A * softplus(alpha + dt_bias)), beta_gate = sigmoid(beta)
// Dispatch: num_v_heads threads (64)

kernel void compute_decay_beta(
    device const float *alpha_out,   // [num_v_heads] from projection
    device const float *beta_out,    // [num_v_heads] from projection
    device const float *A_log,       // [num_v_heads] log of decay base (persistent)
    device const uint16_t *dt_bias,  // [num_v_heads] bf16
    device float *g_decay,           // [num_v_heads] output
    device float *beta_gate,         // [num_v_heads] output
    uint idx [[thread_position_in_grid]]
) {
    float a_val = alpha_out[idx];
    float dt_b = bf16_to_f32(dt_bias[idx]);
    float A_val = exp(A_log[idx]);
    float softplus_val = log(1.0f + exp(a_val + dt_b));
    g_decay[idx] = exp(-A_val * softplus_val);
    beta_gate[idx] = 1.0f / (1.0f + exp(-beta_out[idx]));
}


// ============================================================================
// Kernel 14: Gated RMS norm (z-gated output normalization)
// ============================================================================
// output[i] = rms_norm(values[i]) * SiLU(z[i]) * weight[i]
// Per v-head: normalize values, gate with z, scale with weight
// Dispatch: num_v_heads threadgroups, value_dim threads each

kernel void gated_rms_norm(
    device const float *values,       // [num_v_heads * value_dim] delta-net output
    device const float *z,            // [num_v_heads * value_dim] gate values
    device const uint16_t *weight,    // [value_dim] bf16 norm weights (shared across heads)
    device float *output,             // [num_v_heads * value_dim]
    constant uint &value_dim,         // = 128
    constant float &eps,              // = 1e-6
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    uint base = head * value_dim;

    float val = (tid < value_dim) ? values[base + tid] : 0;

    // RMS norm reduction
    threadgroup float partial[128];
    partial[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < value_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(partial[0] / float(value_dim) + eps);

    if (tid < value_dim) {
        float normed = val * inv_rms;
        float zval = z[base + tid];
        float gate = zval / (1.0f + exp(-zval));  // SiLU
        float w = bf16_to_f32(weight[tid]);
        output[base + tid] = normed * gate * w;
    }
}


// ============================================================================
// Kernel 12: MoE combine + residual + shared expert gate (fused)
// ============================================================================
// Fused operation for CMD3 GPU-side combine:
//   hidden[i] = h_mid[i] + sum_k(expert_weight[k] * expert_out[k][i])
//               + sigmoid(shared_gate_score) * shared_out[i]
//
// All 8 expert output buffers are always bound (unused ones have weight=0).
// This avoids variable buffer bindings and keeps the dispatch simple.
//
// Dispatch: (dim + 255) / 256 threadgroups, 256 threads each.

kernel void moe_combine_residual(
    device const float* h_mid       [[buffer(0)]],   // [dim]
    device const float* shared_out  [[buffer(1)]],   // [dim]
    device float*       hidden_out  [[buffer(2)]],   // [dim] output
    device const float* expert_out0 [[buffer(3)]],   // [dim] expert 0
    device const float* expert_out1 [[buffer(4)]],   // [dim] expert 1
    device const float* expert_out2 [[buffer(5)]],   // [dim] expert 2
    device const float* expert_out3 [[buffer(6)]],   // [dim] expert 3
    device const float* expert_out4 [[buffer(7)]],   // [dim] expert 4
    device const float* expert_out5 [[buffer(8)]],   // [dim] expert 5
    device const float* expert_out6 [[buffer(9)]],   // [dim] expert 6
    device const float* expert_out7 [[buffer(10)]],  // [dim] expert 7
    device const float* params      [[buffer(11)]],  // [10]: weights[0..7], shared_gate_score, (unused)
    constant uint&      dim         [[buffer(12)]],
    constant uint&      K           [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    // Read expert weights and shared gate from params buffer
    float shared_gate = 1.0f / (1.0f + exp(-params[8]));  // sigmoid(shared_gate_score)

    // Weighted sum of expert outputs
    float moe = 0.0f;
    // Unrolled for MAX_K=8 with branch on K to avoid reading invalid buffers
    if (K > 0) moe += params[0] * expert_out0[tid];
    if (K > 1) moe += params[1] * expert_out1[tid];
    if (K > 2) moe += params[2] * expert_out2[tid];
    if (K > 3) moe += params[3] * expert_out3[tid];
    if (K > 4) moe += params[4] * expert_out4[tid];
    if (K > 5) moe += params[5] * expert_out5[tid];
    if (K > 6) moe += params[6] * expert_out6[tid];
    if (K > 7) moe += params[7] * expert_out7[tid];

    hidden_out[tid] = h_mid[tid] + moe + shared_gate * shared_out[tid];
}

// ============================================================================
// Kernel 17: Q head norm + RoPE — split Q/Q-gate, apply per-head RMS norm, rotate
// ============================================================================
// Dispatch: num_q_heads threadgroups, head_dim threads each.

kernel void q_head_norm_rope(
    device const float*    q_proj       [[buffer(0)]],  // [num_q * 2 * head_dim]
    device const uint16_t* q_norm_w     [[buffer(1)]],  // [head_dim] bf16, shared across heads
    device float*          q_out        [[buffer(2)]],  // [num_q * head_dim]
    device float*          q_gate_out   [[buffer(3)]],  // [num_q * head_dim]
    constant uint&         head_dim     [[buffer(4)]],
    constant uint&         rotary_dim   [[buffer(5)]],
    constant float&        rope_theta   [[buffer(6)]],
    constant uint&         pos          [[buffer(7)]],
    constant float&        eps          [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    uint src_base = head * 2 * head_dim;
    uint out_base = head * head_dim;

    float q_val = q_proj[src_base + tid];
    q_gate_out[out_base + tid] = q_proj[src_base + head_dim + tid];

    // RMS norm reduction
    threadgroup float partial[256];
    partial[tid] = q_val * q_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < head_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(partial[0] / float(head_dim) + eps);

    q_val *= inv_rms * bf16_to_f32(q_norm_w[tid]);

    // RoPE
    if (tid < rotary_dim) {
        uint rot_half = rotary_dim / 2;
        float theta;
        float pair_val;
        if (tid < rot_half) {
            theta = float(pos) * pow(rope_theta, -2.0f * float(tid) / float(rotary_dim));
            pair_val = q_proj[src_base + tid + rot_half]
                       * inv_rms * bf16_to_f32(q_norm_w[tid + rot_half]);
        } else {
            uint pair = tid - rot_half;
            theta = float(pos) * pow(rope_theta, -2.0f * float(pair) / float(rotary_dim));
            pair_val = q_proj[src_base + pair]
                       * inv_rms * bf16_to_f32(q_norm_w[pair]);
        }
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        float my_normed = q_val;
        if (tid < rot_half) {
            q_out[out_base + tid] = my_normed * cos_t - pair_val * sin_t;
        } else {
            q_out[out_base + tid] = pair_val * sin_t + my_normed * cos_t;
        }
    } else {
        q_out[out_base + tid] = q_val;
    }
}

// ============================================================================
// Kernel 18: K head norm + RoPE — apply per-head RMS norm and rotate K in-place
// ============================================================================
// Dispatch: num_kv_heads threadgroups, head_dim threads each.

kernel void k_head_norm_rope(
    device float*          k_buf        [[buffer(0)]],  // [num_kv * head_dim] in/out
    device const uint16_t* k_norm_w     [[buffer(1)]],  // [head_dim] bf16, shared across heads
    constant uint&         head_dim     [[buffer(2)]],
    constant uint&         rotary_dim   [[buffer(3)]],
    constant float&        rope_theta   [[buffer(4)]],
    constant uint&         pos          [[buffer(5)]],
    constant float&        eps          [[buffer(6)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    uint base = head * head_dim;
    float k_val = k_buf[base + tid];

    // RMS norm reduction
    threadgroup float partial[256];
    partial[tid] = k_val * k_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float s = 0;
        for (uint i = 0; i < head_dim; i++) s += partial[i];
        partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(partial[0] / float(head_dim) + eps);

    k_val *= inv_rms * bf16_to_f32(k_norm_w[tid]);

    // RoPE
    if (tid < rotary_dim) {
        uint rot_half = rotary_dim / 2;
        float theta;
        float pair_val;
        if (tid < rot_half) {
            theta = float(pos) * pow(rope_theta, -2.0f * float(tid) / float(rotary_dim));
            pair_val = k_buf[base + tid + rot_half]
                       * inv_rms * bf16_to_f32(k_norm_w[tid + rot_half]);
        } else {
            uint pair = tid - rot_half;
            theta = float(pos) * pow(rope_theta, -2.0f * float(pair) / float(rotary_dim));
            pair_val = k_buf[base + pair]
                       * inv_rms * bf16_to_f32(k_norm_w[pair]);
        }
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        if (tid < rot_half) {
            k_buf[base + tid] = k_val * cos_t - pair_val * sin_t;
        } else {
            k_buf[base + tid] = pair_val * sin_t + k_val * cos_t;
        }
    } else {
        k_buf[base + tid] = k_val;
    }
}

// ============================================================================
// Kernel 19: KV-cache append — copy K and V into persistent cache at position pos
// ============================================================================
// Dispatch: (kv_dim + 255) / 256 threadgroups, 256 threads each.

kernel void kv_cache_append(
    device const float*  k       [[buffer(0)]],  // [kv_dim]
    device const float*  v       [[buffer(1)]],  // [kv_dim]
    device float*        k_cache [[buffer(2)]],  // [MAX_SEQ * kv_dim]
    device float*        v_cache [[buffer(3)]],  // [MAX_SEQ * kv_dim]
    constant uint&       pos     [[buffer(4)]],
    constant uint&       kv_dim  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= kv_dim) return;
    uint dst = pos * kv_dim + tid;
    k_cache[dst] = k[tid];
    v_cache[dst] = v[tid];
}

// ============================================================================
// Kernel 20: BF16 matrix-vector multiply (direct, no dequant)
// ============================================================================
// For BQ4: sensitive blocks (attention, routers, lm_head) stay in BF16 and
// use a direct bf16→f32 matvec instead of 4-bit dequant.
// Dispatch: ceil(out_dim / ROWS_PER_TG) threadgroups, 256 threads each.

kernel void matvec_bf16(
    device const uint16_t* W_bf16 [[buffer(0)]],  // [out_dim, in_dim]
    device const float*    x      [[buffer(1)]],  // [in_dim]
    device float*          out    [[buffer(2)]],  // [out_dim]
    constant uint&         out_dim [[buffer(3)]],
    constant uint&         in_dim  [[buffer(4)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    threadgroup float x_shared[4096];

    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uint16_t* w_row = W_bf16 + row * in_dim;

    float acc = 0.0f;
    for (uint col = simd_lane; col < in_dim; col += 32) {
        acc += bf16_to_f32(w_row[col]) * x_shared[col];
    }

    float sum = simd_sum(acc);

    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Kernel 21: INT8 per-channel symmetric matrix-vector multiply
// ============================================================================
// For BQ4: lm_head stored as int8 weights + f32 per-channel scales.
// Dequant: w_f32 = int8(w_q) * scale[row], then dot product.
// Dispatch: ceil(out_dim / ROWS_PER_TG) threadgroups, 256 threads each.

kernel void matvec_int8(
    device const char*      W_i8    [[buffer(0)]],  // [out_dim, in_dim]
    device const float*     scales  [[buffer(1)]],  // [out_dim] per-channel
    device const float*     x       [[buffer(2)]],  // [in_dim]
    device float*           out     [[buffer(3)]],  // [out_dim]
    constant uint&          out_dim [[buffer(4)]],
    constant uint&          in_dim  [[buffer(5)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    threadgroup float x_shared[4096];

    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    float scale = scales[row];
    device const char* w_row = W_i8 + row * in_dim;

    float acc = 0.0f;
    for (uint col = simd_lane; col < in_dim; col += 32) {
        acc += float(w_row[col]) * scale * x_shared[col];
    }

    float sum = simd_sum(acc);

    if (simd_lane == 0) {
        out[row] = sum;
    }
}

// ============================================================================
// Kernel 22: FP8_E4M3 dequant matvec
// ============================================================================
// Per-group scaled FP8 E4M3 weights.  No bias — FP8's symmetric encoding
// handles the zero point natively.  Uses a 256-entry LUT for byte→float decode.
//
//   dequant_val = lut[byte] * scale

constant float fp8_e4m3_lut[256] = {
     0.0000000f,  0.0019531f,  0.0039062f,  0.0058594f,  0.0078125f,  0.0097656f,  0.0117188f,  0.0136719f,
     0.0156250f,  0.0175781f,  0.0195312f,  0.0214844f,  0.0234375f,  0.0253906f,  0.0273438f,  0.0292969f,
     0.0312500f,  0.0351562f,  0.0390625f,  0.0429688f,  0.0468750f,  0.0507812f,  0.0546875f,  0.0585938f,
     0.0625000f,  0.0703125f,  0.0781250f,  0.0859375f,  0.0937500f,  0.1015625f,  0.1093750f,  0.1171875f,
     0.1250000f,  0.1406250f,  0.1562500f,  0.1718750f,  0.1875000f,  0.2031250f,  0.2187500f,  0.2343750f,
     0.2500000f,  0.2812500f,  0.3125000f,  0.3437500f,  0.3750000f,  0.4062500f,  0.4375000f,  0.4687500f,
     0.5000000f,  0.5625000f,  0.6250000f,  0.6875000f,  0.7500000f,  0.8125000f,  0.8750000f,  0.9375000f,
     1.0000000f,  1.1250000f,  1.2500000f,  1.3750000f,  1.5000000f,  1.6250000f,  1.7500000f,  1.8750000f,
     2.0000000f,  2.2500000f,  2.5000000f,  2.7500000f,  3.0000000f,  3.2500000f,  3.5000000f,  3.7500000f,
     4.0000000f,  4.5000000f,  5.0000000f,  5.5000000f,  6.0000000f,  6.5000000f,  7.0000000f,  7.5000000f,
     8.0000000f,  9.0000000f, 10.0000000f, 11.0000000f, 12.0000000f, 13.0000000f, 14.0000000f, 15.0000000f,
    16.0000000f, 18.0000000f, 20.0000000f, 22.0000000f, 24.0000000f, 26.0000000f, 28.0000000f, 30.0000000f,
    32.0000000f, 36.0000000f, 40.0000000f, 44.0000000f, 48.0000000f, 52.0000000f, 56.0000000f, 60.0000000f,
    64.0000000f, 72.0000000f, 80.0000000f, 88.0000000f, 96.0000000f,104.0000000f,112.0000000f,120.0000000f,
   128.0000000f,144.0000000f,160.0000000f,176.0000000f,192.0000000f,208.0000000f,224.0000000f,240.0000000f,
   240.0000000f,240.0000000f,240.0000000f,240.0000000f,240.0000000f,240.0000000f,240.0000000f,240.0000000f,
    -0.0000000f, -0.0019531f, -0.0039062f, -0.0058594f, -0.0078125f, -0.0097656f, -0.0117188f, -0.0136719f,
    -0.0156250f, -0.0175781f, -0.0195312f, -0.0214844f, -0.0234375f, -0.0253906f, -0.0273438f, -0.0292969f,
    -0.0312500f, -0.0351562f, -0.0390625f, -0.0429688f, -0.0468750f, -0.0507812f, -0.0546875f, -0.0585938f,
    -0.0625000f, -0.0703125f, -0.0781250f, -0.0859375f, -0.0937500f, -0.1015625f, -0.1093750f, -0.1171875f,
    -0.1250000f, -0.1406250f, -0.1562500f, -0.1718750f, -0.1875000f, -0.2031250f, -0.2187500f, -0.2343750f,
    -0.2500000f, -0.2812500f, -0.3125000f, -0.3437500f, -0.3750000f, -0.4062500f, -0.4375000f, -0.4687500f,
    -0.5000000f, -0.5625000f, -0.6250000f, -0.6875000f, -0.7500000f, -0.8125000f, -0.8750000f, -0.9375000f,
    -1.0000000f, -1.1250000f, -1.2500000f, -1.3750000f, -1.5000000f, -1.6250000f, -1.7500000f, -1.8750000f,
    -2.0000000f, -2.2500000f, -2.5000000f, -2.7500000f, -3.0000000f, -3.2500000f, -3.5000000f, -3.7500000f,
    -4.0000000f, -4.5000000f, -5.0000000f, -5.5000000f, -6.0000000f, -6.5000000f, -7.0000000f, -7.5000000f,
    -8.0000000f, -9.0000000f,-10.0000000f,-11.0000000f,-12.0000000f,-13.0000000f,-14.0000000f,-15.0000000f,
   -16.0000000f,-18.0000000f,-20.0000000f,-22.0000000f,-24.0000000f,-26.0000000f,-28.0000000f,-30.0000000f,
   -32.0000000f,-36.0000000f,-40.0000000f,-44.0000000f,-48.0000000f,-52.0000000f,-56.0000000f,-60.0000000f,
   -64.0000000f,-72.0000000f,-80.0000000f,-88.0000000f,-96.0000000f,-104.000000f,-112.000000f,-120.000000f,
  -128.000000f,-144.000000f,-160.000000f,-176.000000f,-192.000000f,-208.000000f,-224.000000f,-240.000000f,
  -240.000000f,-240.000000f,-240.000000f,-240.000000f,-240.000000f,-240.000000f,-240.000000f,-240.000000f,
};

kernel void matvec_fp8_e4m3(
    device const uchar*     W_u8     [[buffer(0)]],  // [out_dim, in_dim]
    device const uint16_t*  scales   [[buffer(1)]],  // [out_dim, num_groups] bf16
    device const float*     x        [[buffer(2)]],  // [in_dim]
    device float*           out      [[buffer(3)]],  // [out_dim]
    constant uint&          out_dim   [[buffer(4)]],
    constant uint&          in_dim    [[buffer(5)]],
    constant uint&          group_size [[buffer(6)]],
    uint tgid   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid * ROWS_PER_TG + simd_group;

    threadgroup float x_shared[4096];
    for (uint i = lid; i < in_dim; i += 256) {
        x_shared[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= out_dim) return;

    device const uchar* w_row = W_u8 + row * in_dim;
    device const uint16_t* s_row = scales + row * (in_dim / group_size);

    float acc = 0.0f;
    for (uint col = simd_lane; col < in_dim; col += 32) {
        uint g = col / group_size;
        float scale = bf16_to_f32(s_row[g]);
        float w = fp8_e4m3_lut[w_row[col]] * scale;
        acc += w * x_shared[col];
    }

    float sum = simd_sum(acc);
    if (simd_lane == 0) {
        out[row] = sum;
    }
}

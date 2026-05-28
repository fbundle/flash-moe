// Shared quantization types and format-specific encode/decode.
//
// Imported by both:
//   - quantize_pipeline.rs  (writes dtype strings into manifest JSON)
//   - metal_context.rs      (reads dtype strings to dispatch GPU kernels)
//
// Only encodes binary format info.  Sanitization (norm shift, conv1d moveaxis)
// is a Qwen3.6 pipeline concern handled separately in quantize_pipeline.rs.

use std::vec::Vec;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Group size for per-group INT4 quantization (64 elements per scale+bias pair).
pub const GROUP_SIZE: usize = 64;

/// Group size for FP8 per-group quantization (128 elements per scale).
pub const FP8_GROUP_SIZE: usize = 128;

// ─── DType enum ──────────────────────────────────────────────────────────────

/// DTypeization format — the binary representation of a tensor's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Fp32,
    Bf16,
    Int4,
    Int8,
    Fp4E2m1,
    Fp8E4m3,
}

impl DType {
    pub const fn as_str(self) -> &'static str {
        match self {
            DType::Fp32 => "f32",
            DType::Bf16 => "bf16",
            DType::Int4 => "u32",
            DType::Int8 => "u8",
            DType::Fp4E2m1 => "fp4_e2m1",
            DType::Fp8E4m3 => "fp8_e4m3",
        }
    }
}

/// Parse a manifest dtype string back to a DType variant.
pub fn string_to_dtype(dtype: &str) -> Option<DType> {
    Some(match dtype {
        "f32"  => DType::Fp32,
        "bf16" => DType::Bf16,
        "u32"  => DType::Int4,
        "u8"   => DType::Int8,
        "fp4_e2m1" => DType::Fp4E2m1,
        "fp8_e4m3" => DType::Fp8E4m3,
        _ => return None,
    })
}

// ─── Encoded tensor ──────────────────────────────────────────────────────────

/// One output tensor produced by `DType::encode()`.
pub struct EncodedTensor {
    pub data: Vec<u8>,
    pub suffix: &'static str,
    pub shape: Vec<usize>,
    pub dtype: &'static str,
}

impl DType {
    pub fn encode(self, f32_vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<EncodedTensor> {
        match self {
            DType::Int4 => encode_int4(f32_vals, out_dim, in_dim),
            DType::Int8 => encode_int8(f32_vals, out_dim, in_dim),
            DType::Bf16 => encode_bf16(f32_vals),
            DType::Fp32 => encode_fp32(f32_vals),
            DType::Fp4E2m1 => encode_fp4_e2m1(f32_vals, out_dim, in_dim),
            DType::Fp8E4m3 => encode_fp8_e4m3(f32_vals, out_dim, in_dim),
        }
    }
}

// ─── BF16 conversion ─────────────────────────────────────────────────────────

#[inline]
pub fn f32_to_bf16_u16_single(v: f32) -> u16 {
    let bits = v.to_bits();
    let round_up = ((bits >> 15) & 1) & ((bits & 0x7FFF) | ((bits >> 16) & 1));
    ((bits.wrapping_add(round_up << 16)) >> 16) as u16
}

pub fn f32_to_bf16_u16(arr: &[f32]) -> Vec<u16> {
    arr.iter().map(|&v| f32_to_bf16_u16_single(v)).collect()
}

#[inline]
pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

// ─── INT4 quantization ───────────────────────────────────────────────────────

pub fn quant_f32_to_int4(f32_vals: &[f32], out_dim: usize, in_dim: usize) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    let num_groups = in_dim / GROUP_SIZE;
    let words_per_row = in_dim / 8;
    let mut packed = vec![0u32; out_dim * words_per_row];
    let mut scales = vec![0u16; out_dim * num_groups];
    let mut biases = vec![0u16; out_dim * num_groups];
    for row in 0..out_dim {
        let row_base = row * in_dim;
        for g in 0..num_groups {
            let g_base = row_base + g * GROUP_SIZE;
            let group = &f32_vals[g_base..g_base + GROUP_SIZE];
            let mut vmin = group[0];
            let mut vmax = group[0];
            for &v in &group[1..] { vmin = vmin.min(v); vmax = vmax.max(v); }
            if vmax == vmin { vmax = vmin + 1.0; }
            let fscale = (vmax - vmin) / 15.0;
            let fbias = vmin;
            let s_idx = row * num_groups + g;
            scales[s_idx] = f32_to_bf16_u16_single(fscale);
            biases[s_idx] = f32_to_bf16_u16_single(fbias);
            let inv_scale = 1.0 / fscale;
            for p in 0..8 {
                let mut word: u32 = 0;
                for n in 0..8 {
                    let v = group[p * 8 + n];
                    let q = ((v - fbias) * inv_scale + 0.5) as i32;
                    word |= ((q.clamp(0, 15) as u32) & 0xF) << (n * 4);
                }
                packed[row * words_per_row + g * 8 + p] = word;
            }
        }
    }
    (packed, scales, biases)
}

#[allow(dead_code)]
pub fn int4_to_f32(packed: &[u32], scales: &[u16], biases: &[u16], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let num_groups = in_dim / GROUP_SIZE;
    let words_per_row = in_dim / 8;
    let mut result = vec![0.0f32; out_dim * in_dim];
    for row in 0..out_dim {
        let w_row = &packed[row * words_per_row..(row + 1) * words_per_row];
        let s_row = &scales[row * num_groups..(row + 1) * num_groups];
        let b_row = &biases[row * num_groups..(row + 1) * num_groups];
        for g in 0..num_groups {
            let scale = bf16_to_f32(s_row[g]);
            let bias = bf16_to_f32(b_row[g]);
            let out_base = row * in_dim + g * GROUP_SIZE;
            for p in 0..8 {
                let word = w_row[g * 8 + p];
                for n in 0..8 {
                    let nibble = (word >> (n * 4)) & 0xF;
                    result[out_base + p * 8 + n] = (nibble as f32) * scale + bias;
                }
            }
        }
    }
    result
}

// ─── INT8 per-channel symmetric ──────────────────────────────────────────────

pub fn quant_f32_to_int8(f32_vals: &[f32], out_dim: usize, in_dim: usize) -> (Vec<i8>, Vec<f32>) {
    let mut packed = vec![0i8; out_dim * in_dim];
    let mut scales = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let row_slice = &f32_vals[row * in_dim..(row + 1) * in_dim];
        let max_abs = row_slice.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 / 127.0 };
        scales[row] = scale;
        let inv_scale = 1.0 / scale;
        let dst = &mut packed[row * in_dim..(row + 1) * in_dim];
        for (j, &v) in row_slice.iter().enumerate() {
            dst[j] = ((v * inv_scale).round() as i32).clamp(-127, 127) as i8;
        }
    }
    (packed, scales)
}

#[allow(dead_code)]
pub fn int8_to_f32(packed: &[i8], scales: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; out_dim * in_dim];
    for row in 0..out_dim {
        let scale = scales[row];
        let src = &packed[row * in_dim..(row + 1) * in_dim];
        let dst = &mut result[row * in_dim..(row + 1) * in_dim];
        for (j, &q) in src.iter().enumerate() { dst[j] = (q as f32) * scale; }
    }
    result
}

// ─── FP4_E2M1 ───────────────────────────────────────────────────────────────

/// FP4 E2M1 lookup table: nibble → f32.
///
/// Format: 1 sign | 2 exponent | 1 mantissa.
///   normal (e > 0):  (-1)^s × 2^(e-1) × (1 + m/2)
///   subnormal (e=0): (-1)^s × 2^(-1) × (m/2)      [m=0 → 0, m=1 → 0.5]
pub const FP4_E2M1_LUT: [f32; 16] = [
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode a packed FP4_E2M1 nibble to f32.
#[inline]
pub fn fp4_e2m1_to_f32(nibble: u32) -> f32 {
    FP4_E2M1_LUT[(nibble & 0xF) as usize]
}

/// Quantize f32 values to FP4_E2M1 with per-group BF16 scale.
///
/// Returns (packed_u32, scales_bf16).  Unlike INT4 there is no bias —
/// FP4's symmetric representation handles the zero point natively.
pub fn quant_f32_to_fp4_e2m1(f32_vals: &[f32], out_dim: usize, in_dim: usize)
    -> (Vec<u32>, Vec<u16>)
{
    let num_groups = in_dim / GROUP_SIZE;
    let words_per_row = in_dim / 8;
    let mut packed = vec![0u32; out_dim * words_per_row];
    let mut scales = vec![0u16; out_dim * num_groups];

    // Build reverse LUT for encoding: use only positive half (indices 0..8).
    // Sign is applied separately via the MSB — avoids ambiguity from
    // duplicated absolute values in the negative half.
    let mut thresholds: [(f32, u8); 8] = [(0.0, 0); 8];
    for i in 0..8u8 {
        thresholds[i as usize] = (FP4_E2M1_LUT[i as usize], i);
    }
    thresholds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for row in 0..out_dim {
        let row_base = row * in_dim;
        for g in 0..num_groups {
            let g_base = row_base + g * GROUP_SIZE;
            let group = &f32_vals[g_base..g_base + GROUP_SIZE];
            let max_abs = group.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let max_val = if max_abs < 1e-8 { 1e-8 } else { max_abs };

            // Scale so that max_abs maps to 6.0 (largest representable magnitude)
            let scale = max_val / 6.0f32;
            scales[row * num_groups + g] = f32_to_bf16_u16_single(scale);
            let inv_scale = 1.0f32 / scale;

            for p in 0..8 {
                let mut word: u32 = 0;
                for n in 0..8 {
                    let v = group[p * 8 + n];
                    let norm = (v * inv_scale).abs();
                    // Binary search in thresholds to find nearest match
                    let idx = match thresholds.binary_search_by(|t| t.0.partial_cmp(&norm).unwrap()) {
                        Ok(i) => i,
                        Err(i) => {
                            if i == 0 { 0 }
                            else if i >= 8 { 7 }
                            else {
                                let lo = thresholds[i - 1].0;
                                let hi = thresholds[i].0;
                                if norm - lo < hi - norm { i - 1 } else { i }
                            }
                        }
                    };
                    let mut nibble = thresholds[idx].1;
                    if v < 0.0 { nibble |= 0x8; } // set sign bit
                    word |= ((nibble as u32) & 0xF) << (n * 4);
                }
                packed[row * words_per_row + g * 8 + p] = word;
            }
        }
    }
    (packed, scales)
}

/// Dequantize FP4_E2M1 packed weights to f32.
#[allow(dead_code)]
pub fn fp4_e2m1_to_f32_full(
    packed: &[u32], scales: &[u16], out_dim: usize, in_dim: usize,
) -> Vec<f32> {
    let num_groups = in_dim / GROUP_SIZE;
    let words_per_row = in_dim / 8;
    let mut result = vec![0.0f32; out_dim * in_dim];
    for row in 0..out_dim {
        let w_row = &packed[row * words_per_row..(row + 1) * words_per_row];
        let s_row = &scales[row * num_groups..(row + 1) * num_groups];
        for g in 0..num_groups {
            let scale = bf16_to_f32(s_row[g]);
            let out_base = row * in_dim + g * GROUP_SIZE;
            for p in 0..8 {
                let word = w_row[g * 8 + p];
                for n in 0..8 {
                    let nibble = (word >> (n * 4)) & 0xF;
                    result[out_base + p * 8 + n] = fp4_e2m1_to_f32(nibble) * scale;
                }
            }
        }
    }
    result
}

// ─── FP8_E4M3 ───────────────────────────────────────────────────────────────

/// FP8 E4M3 lookup table: byte → f32 (256 entries).
///
/// Format: 1 sign | 4 exponent | 3 mantissa.  Exponent bias = 7.
///   subnormal (e=0, m≠0): (-1)^s × 2^(-6) × m/8
///   normal   (e=1..14):   (-1)^s × 2^(e-7) × (1 + m/8)
///   NaN/Inf  (e=15):      mapped to ±MAX (used as sentinel, not for weights)
pub fn fp8_e4m3_lut() -> &'static [f32; 256] {
    use std::sync::LazyLock;
    static LUT: LazyLock<[f32; 256]> = LazyLock::new(|| {
        let mut lut = [0.0f32; 256];
        let pow2: [f32; 15] = [
            1.0/64.0, 1.0/32.0, 1.0/16.0, 1.0/8.0, 1.0/4.0, 1.0/2.0, 1.0,
            2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
        ];
        for (i, v) in lut.iter_mut().enumerate() {
            let byte = i as u8;
            let sign = if (byte >> 7) != 0 { -1.0f32 } else { 1.0f32 };
            let exp = (byte >> 3) & 0xF;
            let mant = byte & 0x7;
            let mag = if exp == 0 {
                if mant == 0 { 0.0 }
                else { mant as f32 * (1.0 / 512.0) } // 2^(-6) * m/8 = m / 512
            } else if exp == 15 {
                240.0
            } else {
                pow2[(exp - 1) as usize] * (1.0 + mant as f32 / 8.0)
            };
            *v = sign * mag;
        }
        lut
    });
    &LUT
}

/// Decode an FP8 E4M3 byte to f32 via LUT.
#[inline]
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    fp8_e4m3_lut()[byte as usize]
}

/// Encode a single f32 value to FP8 E4M3 via nearest-neighbor LUT search.
#[inline]
fn f32_to_fp8_e4m3(val: f32) -> u8 {
    let abs_val = val.abs();
    if abs_val < fp8_e4m3_lut()[1] * 0.5 { return ((val < 0.0) as u8) << 7; } // zero
    let max_norm = 240.0f32;
    let clamped = abs_val.min(max_norm);

    let sign_bit = ((val < 0.0) as u8) << 7;
    let mut best: u8 = 0;
    let mut best_dist = f32::MAX;

    let lut = fp8_e4m3_lut();
    for i in 0u8..128 {
        let dist = (clamped - lut[i as usize]).abs();
        if dist < best_dist {
            best_dist = dist;
            best = i;
        }
    }
    best | sign_bit
}

/// Quantize f32 values to FP8 E4M3 with per-group BF16 scale.
///
/// Returns (packed_u8, scales_bf16) using group_size = FP8_GROUP_SIZE (128).
pub fn quant_f32_to_fp8_e4m3(f32_vals: &[f32], out_dim: usize, in_dim: usize)
    -> (Vec<u8>, Vec<u16>)
{
    let gs = FP8_GROUP_SIZE;
    let num_groups = in_dim / gs;
    let mut packed = vec![0u8; out_dim * in_dim];
    let mut scales = vec![0u16; out_dim * num_groups];

    for row in 0..out_dim {
        let row_base = row * in_dim;
        for g in 0..num_groups {
            let g_base = row_base + g * gs;
            let group = &f32_vals[g_base..g_base + gs];
            let max_abs = group.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let max_val = if max_abs < 1e-8 { 1e-8 } else { max_abs };

            // Scale so max_abs maps to 240 (FP8 E4M3 max normal)
            let scale = max_val / 240.0f32;
            scales[row * num_groups + g] = f32_to_bf16_u16_single(scale);
            let inv_scale = 1.0f32 / scale;

            for j in 0..gs {
                let v = group[j] * inv_scale;
                packed[g_base + j] = f32_to_fp8_e4m3(v);
            }
        }
    }
    (packed, scales)
}

/// Dequantize FP8 E4M3 packed weights to f32.
#[allow(dead_code)]
pub fn fp8_e4m3_to_f32_full(
    packed: &[u8], scales: &[u16], out_dim: usize, in_dim: usize,
) -> Vec<f32> {
    let gs = FP8_GROUP_SIZE;
    let num_groups = in_dim / gs;
    let mut result = vec![0.0f32; out_dim * in_dim];
    for row in 0..out_dim {
        let row_base = row * in_dim;
        let s_row = &scales[row * num_groups..(row + 1) * num_groups];
        for g in 0..num_groups {
            let scale = bf16_to_f32(s_row[g]);
            let g_base = row_base + g * gs;
            for j in 0..gs {
                result[g_base + j] = fp8_e4m3_to_f32(packed[g_base + j]) * scale;
            }
        }
    }
    result
}

// ─── Encode helpers ──────────────────────────────────────────────────────────

fn encode_fp8_e4m3(f32_vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<EncodedTensor> {
    let gs = FP8_GROUP_SIZE;
    let num_groups = in_dim / gs;
    let (packed, scales) = quant_f32_to_fp8_e4m3(f32_vals, out_dim, in_dim);
    vec![
        EncodedTensor { data: packed, suffix: ".weight", shape: vec![out_dim, in_dim], dtype: DType::Fp8E4m3.as_str() },
        EncodedTensor { data: unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 2).to_vec() }, suffix: ".scales", shape: vec![out_dim, num_groups], dtype: DType::Bf16.as_str() },
    ]
}

fn encode_fp4_e2m1(f32_vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<EncodedTensor> {
    let num_groups = in_dim / GROUP_SIZE;
    let (packed, scales) = quant_f32_to_fp4_e2m1(f32_vals, out_dim, in_dim);
    let packed_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4).to_vec() };
    let scales_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 2).to_vec() };
    vec![
        EncodedTensor { data: packed_bytes, suffix: ".weight", shape: vec![out_dim, in_dim / 8], dtype: DType::Fp4E2m1.as_str() },
        EncodedTensor { data: scales_bytes, suffix: ".scales", shape: vec![out_dim, num_groups], dtype: DType::Bf16.as_str() },
    ]
}

fn encode_int4(f32_vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<EncodedTensor> {
    let num_groups = in_dim / GROUP_SIZE;
    let (packed, scales, biases) = quant_f32_to_int4(f32_vals, out_dim, in_dim);
    let packed_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4).to_vec() };
    let scales_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 2).to_vec() };
    let biases_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(biases.as_ptr() as *const u8, biases.len() * 2).to_vec() };
    vec![
        EncodedTensor { data: packed_bytes, suffix: ".weight", shape: vec![out_dim, in_dim / 8], dtype: DType::Int4.as_str() },
        EncodedTensor { data: scales_bytes, suffix: ".scales", shape: vec![out_dim, num_groups], dtype: DType::Bf16.as_str() },
        EncodedTensor { data: biases_bytes, suffix: ".biases", shape: vec![out_dim, num_groups], dtype: DType::Bf16.as_str() },
    ]
}

fn encode_bf16(f32_vals: &[f32]) -> Vec<EncodedTensor> {
    let v = f32_to_bf16_u16(f32_vals);
    let bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2).to_vec() };
    vec![EncodedTensor { data: bytes, suffix: "", shape: vec![f32_vals.len()], dtype: DType::Bf16.as_str() }]
}

fn encode_fp32(f32_vals: &[f32]) -> Vec<EncodedTensor> {
    let bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(f32_vals.as_ptr() as *const u8, f32_vals.len() * 4).to_vec() };
    vec![EncodedTensor { data: bytes, suffix: "", shape: vec![f32_vals.len()], dtype: DType::Fp32.as_str() }]
}

fn encode_int8(f32_vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<EncodedTensor> {
    let (packed, scales) = quant_f32_to_int8(f32_vals, out_dim, in_dim);
    let packed_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len()).to_vec() };
    let scales_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4).to_vec() };
    vec![
        EncodedTensor { data: packed_bytes, suffix: ".weight", shape: vec![out_dim, in_dim], dtype: DType::Int8.as_str() },
        EncodedTensor { data: scales_bytes, suffix: ".scales", shape: vec![out_dim], dtype: DType::Fp32.as_str() },
    ]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int4_roundtrip() {
        let out_dim = 2;
        let in_dim = 128;
        let vals: Vec<f32> = (0..(out_dim * in_dim)).map(|i| (i as f32).sin()).collect();
        let (p, s, b) = quant_f32_to_int4(&vals, out_dim, in_dim);
        let r = int4_to_f32(&p, &s, &b, out_dim, in_dim);
        for g in 0..(in_dim / GROUP_SIZE) {
            let base = g * GROUP_SIZE;
            let max_err = vals[base..base + GROUP_SIZE].iter()
                .zip(r[base..base + GROUP_SIZE].iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let range = vals[base..base + GROUP_SIZE].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
                - vals[base..base + GROUP_SIZE].iter().fold(f32::INFINITY, |a, &b| a.min(b));
            if range > 0.001 { assert!(max_err / range < 0.1); }
        }
    }

    #[test]
    fn test_int8_roundtrip() {
        let vals: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.1).sin() * 3.0).collect();
        let (p, s) = quant_f32_to_int8(&vals, 4, 128);
        let r = int8_to_f32(&p, &s, 4, 128);
        let range = vals.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v))
            - vals.iter().fold(f32::INFINITY, |a, &v| a.min(v));
        for (a, b) in vals.iter().zip(r.iter()) {
            assert!((a - b).abs() / range.max(0.01) < 0.02, "err={}", (a-b).abs());
        }
    }

    #[test]
    fn test_fp4_e2m1_roundtrip() {
        let out_dim = 2;
        let in_dim = 128;
        let vals: Vec<f32> = (0..(out_dim * in_dim))
            .map(|i| ((i as f32) * 0.13).sin() * 2.5)
            .collect();
        let (p, s) = quant_f32_to_fp4_e2m1(&vals, out_dim, in_dim);
        let r = fp4_e2m1_to_f32_full(&p, &s, out_dim, in_dim);
        let max_abs = vals.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let mut max_err = 0.0f32;
        for (a, b) in vals.iter().zip(r.iter()) {
            let err = (a - b).abs();
            if err > max_err { max_err = err; }
        }
        // FP4 has ~12.5% relative error ceiling (6 bits of range / 2^4 = 6/16)
        let rel = max_err / max_abs.max(0.001);
        assert!(rel < 0.5, "max relative error {} exceeds 0.5", rel);
    }

    #[test]
    fn test_fp8_e4m3_roundtrip() {
        let out_dim = 2;
        let in_dim = 256; // must be multiple of FP8_GROUP_SIZE (128)
        let vals: Vec<f32> = (0..(out_dim * in_dim))
            .map(|i| ((i as f32) * 0.07).sin() * 5.0)
            .collect();
        let (p, s) = quant_f32_to_fp8_e4m3(&vals, out_dim, in_dim);
        let r = fp8_e4m3_to_f32_full(&p, &s, out_dim, in_dim);
        let max_abs = vals.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let mut max_err = 0.0f32;
        for (a, b) in vals.iter().zip(r.iter()) {
            let err = (a - b).abs();
            if err > max_err { max_err = err; }
        }
        // FP8 E4M3 has ~3-6% worst-case relative error (step=16 near 240)
        let rel = max_err / max_abs.max(0.001);
        assert!(rel < 0.10, "max relative error {} exceeds 10%", rel);
    }

    #[test]
    fn test_bf16_roundtrip() {
        let vals: Vec<f32> = vec![0.0, 1.0, -1.0, 3.14159, 0.001, 1000.0];
        let bf = f32_to_bf16_u16(&vals);
        let back: Vec<f32> = bf.iter().map(|&v| bf16_to_f32(v)).collect();
        for (orig, &recon) in vals.iter().zip(back.iter()) {
            let rel = if *orig == 0.0 { (orig - recon).abs() } else { (orig - recon).abs() / orig.abs() };
            assert!(rel < 0.01);
        }
    }
}

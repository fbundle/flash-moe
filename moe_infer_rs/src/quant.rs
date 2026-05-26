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

// ─── Quant enum ──────────────────────────────────────────────────────────────

/// Quantization format — the binary representation of a tensor's data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    Fp32,
    Bf16,
    Int4,
    Int8,
}

impl Quant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Quant::Fp32 => "f32",
            Quant::Bf16 => "bf16",
            Quant::Int4 => "u32",
            Quant::Int8 => "u8",
        }
    }
}

/// Parse a manifest dtype string back to a Quant variant.
pub fn string_to_quant(dtype: &str) -> Option<Quant> {
    Some(match dtype {
        "f32"  => Quant::Fp32,
        "bf16" => Quant::Bf16,
        "u32"  => Quant::Int4,
        "u8"   => Quant::Int8,
        _ => return None,
    })
}

// ─── Encoded tensor ──────────────────────────────────────────────────────────

/// One output tensor produced by `Quant::encode()`.
pub struct EncodedTensor {
    pub data: Vec<u8>,
    pub suffix: &'static str,
    pub shape: Vec<usize>,
    pub dtype: &'static str,
}

impl Quant {
    pub fn encode(self, f32_vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<EncodedTensor> {
        match self {
            Quant::Int4 => encode_int4(f32_vals, out_dim, in_dim),
            Quant::Int8 => encode_int8(f32_vals, out_dim, in_dim),
            Quant::Bf16 => encode_bf16(f32_vals),
            Quant::Fp32 => encode_fp32(f32_vals),
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

// ─── Encode helpers ──────────────────────────────────────────────────────────

fn encode_int4(f32_vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<EncodedTensor> {
    let num_groups = in_dim / GROUP_SIZE;
    let (packed, scales, biases) = quant_f32_to_int4(f32_vals, out_dim, in_dim);
    let packed_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4).to_vec() };
    let scales_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 2).to_vec() };
    let biases_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(biases.as_ptr() as *const u8, biases.len() * 2).to_vec() };
    vec![
        EncodedTensor { data: packed_bytes, suffix: ".weight", shape: vec![out_dim, in_dim / 8], dtype: Quant::Int4.as_str() },
        EncodedTensor { data: scales_bytes, suffix: ".scales", shape: vec![out_dim, num_groups], dtype: Quant::Bf16.as_str() },
        EncodedTensor { data: biases_bytes, suffix: ".biases", shape: vec![out_dim, num_groups], dtype: Quant::Bf16.as_str() },
    ]
}

fn encode_bf16(f32_vals: &[f32]) -> Vec<EncodedTensor> {
    let v = f32_to_bf16_u16(f32_vals);
    let bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2).to_vec() };
    vec![EncodedTensor { data: bytes, suffix: "", shape: vec![f32_vals.len()], dtype: Quant::Bf16.as_str() }]
}

fn encode_fp32(f32_vals: &[f32]) -> Vec<EncodedTensor> {
    let bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(f32_vals.as_ptr() as *const u8, f32_vals.len() * 4).to_vec() };
    vec![EncodedTensor { data: bytes, suffix: "", shape: vec![f32_vals.len()], dtype: Quant::Fp32.as_str() }]
}

fn encode_int8(f32_vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<EncodedTensor> {
    let (packed, scales) = quant_f32_to_int8(f32_vals, out_dim, in_dim);
    let packed_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len()).to_vec() };
    let scales_bytes: Vec<u8> = unsafe { std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4).to_vec() };
    vec![
        EncodedTensor { data: packed_bytes, suffix: ".weight", shape: vec![out_dim, in_dim], dtype: Quant::Int8.as_str() },
        EncodedTensor { data: scales_bytes, suffix: ".scales", shape: vec![out_dim], dtype: Quant::Fp32.as_str() },
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

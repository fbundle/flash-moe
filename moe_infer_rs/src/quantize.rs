// BQ4 quantization: classification, encoding, conversion.
//
// Pure Rust — no Python dependencies.  The PyO3 wrappers live in
// python_bindings.rs behind the "python-bindings" feature flag.
//
// quant/quant.py is the canonical reference for all logic below.

#![allow(dead_code)]

// ─── Constants ───────────────────────────────────────────────────────────────

pub const GROUP_SIZE: usize = 64;

// ─── Quant enum ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    /// Full 32-bit float — used only for A_log scalars.
    Fp32,
    /// BF16 passthrough WITH sanitization (norm shift, conv1d moveaxis).
    Bf16,
    /// BF16 passthrough WITHOUT sanitization — sensitive attention/routing matrices.
    Bf16Pass,
    /// 4-bit affine quantization: weight (packed u32) + scales (bf16) + biases (bf16).
    Int4,
}

impl Quant {
    /// Manifest dtype string ("f32", "bf16", "u32").
    pub fn as_str(self) -> &'static str {
        match self {
            Quant::Fp32 => "f32",
            Quant::Bf16 | Quant::Bf16Pass => "bf16",
            Quant::Int4 => "u32",
        }
    }

    /// Whether sanitization applies (norm shift, conv1d moveaxis).
    pub fn needs_sanitization(self) -> bool {
        matches!(self, Quant::Bf16)
    }
}

// ─── Encoded tensor ──────────────────────────────────────────────────────────

/// One output tensor produced by `Quant::encode()`.
pub struct EncodedTensor {
    pub data: Vec<u8>,
    /// Suffix appended to the base name, e.g. ".weight", ".scales", ".biases", or "".
    pub suffix: &'static str,
    pub shape: Vec<usize>,
    /// Manifest dtype string.
    pub dtype: &'static str,
}

impl Quant {
    /// Encode f32 values into one or more output tensors.
    ///
    /// `f32_vals` is a flat row-major slice of shape `[out_dim, in_dim]` for
    /// INT4, or any shape for BF16/FP32 passthrough.
    pub fn encode(self, f32_vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<EncodedTensor> {
        match self {
            Quant::Int4 => encode_int4(f32_vals, out_dim, in_dim),
            Quant::Bf16 | Quant::Bf16Pass => encode_bf16(f32_vals),
            Quant::Fp32 => encode_fp32(f32_vals),
        }
    }
}

// ─── Name parsing ────────────────────────────────────────────────────────────

/// Split on last dot → (prefix, kind).  If no dot, kind is empty.
pub fn split_on_last_dot(name: &str) -> (&str, &str) {
    match name.rfind('.') {
        Some(idx) => (&name[..idx], &name[idx + 1..]),
        None => (name, ""),
    }
}

/// Strip the layer/model prefix to get the relative block name used for
/// classification.  Matches `_PREFIX_RE` from quant/quant.py.
///
/// Prefixes stripped (tried in order):
///   language_model.model.layers.{N}.
///   language_model.
///   vision_tower.blocks.{N}.
///   vision_tower.
///   mtp.layers.{N}.
///   mtp.
pub fn strip_layer_prefix(name: &str) -> &str {
    // language_model.model.layers.{N}.
    if let Some(after) = name.strip_prefix("language_model.model.layers.") {
        return after.find('.').map_or(after, |d| &after[d + 1..]);
    }
    // language_model.
    if let Some(after) = name.strip_prefix("language_model.") {
        return after;
    }
    // vision_tower.blocks.{N}.
    if let Some(after) = name.strip_prefix("vision_tower.blocks.") {
        return after.find('.').map_or(after, |d| &after[d + 1..]);
    }
    // vision_tower.
    if let Some(after) = name.strip_prefix("vision_tower.") {
        return after;
    }
    // mtp.layers.{N}.
    if let Some(after) = name.strip_prefix("mtp.layers.") {
        return after.find('.').map_or(after, |d| &after[d + 1..]);
    }
    // mtp.
    if let Some(after) = name.strip_prefix("mtp.") {
        return after;
    }
    name
}

// ─── matrixTable — relative block → Quant ────────────────────────────────────

/// Map a relative block name (after stripping layer prefix) to its Quant variant.
/// Anything not listed falls through to INT4.
///
/// Edit this table to change which matrices stay full-precision.
pub fn matrix_table(block: &str) -> Quant {
    match block {
        "self_attn.q_proj"
        | "self_attn.k_proj"
        | "self_attn.v_proj"
        | "self_attn.o_proj"
        | "mlp.gate"
        | "lm_head"
        | "attn.qkv"
        | "attn.proj"
        | "patch_embed.proj"
        | "pos_embed" => Quant::Bf16Pass,
        _ => Quant::Int4,
    }
}

// ─── bq4 — main classification ───────────────────────────────────────────────

/// Classify a tensor by its MLX name and shape → Quant variant.
///
/// Rules:
///   1. kind == "A_log"            → FP32   (scalar, exact)
///   2. kind ∈ {scales,biases,bias,dt_bias} → BF16
///   3. kind == "weight", ndim ≠ 2 → BF16   (1D vector passthrough)
///   4. kind == "weight", ndim = 2 → matrix_table(block)
pub fn bq4(mlx_name: &str, shape: &[usize]) -> Quant {
    let (prefix, kind) = split_on_last_dot(mlx_name);
    let ndim = shape.len();

    if kind == "A_log" {
        debug_assert!(ndim <= 1, "A_log must be scalar/vector, got ndim={}: {}", ndim, mlx_name);
        return Quant::Fp32;
    }

    if matches!(kind, "scales" | "biases" | "bias" | "dt_bias") {
        debug_assert!(ndim <= 2, "{} must be vector, got ndim={}: {}", kind, ndim, mlx_name);
        return Quant::Bf16;
    }

    if kind == "weight" {
        if ndim != 2 {
            return Quant::Bf16;
        }
        let block = strip_layer_prefix(prefix);
        return matrix_table(block);
    }

    panic!("unknown kind: {:?} in {}", kind, mlx_name);
}

/// Convenience: return the manifest dtype string for a weight tensor.
pub fn classify_weight(mlx_name: &str, shape: &[usize]) -> String {
    bq4(mlx_name, shape).as_str().to_string()
}

// ─── BF16 conversion ─────────────────────────────────────────────────────────

/// Convert a single f32 to bf16 (uint16), round-to-nearest-even.
#[inline]
pub fn f32_to_bf16_u16_single(v: f32) -> u16 {
    let bits = v.to_bits();
    let round_bit = (bits >> 15) & 1;
    let sticky = bits & 0x7FFF;
    let lsb = (bits >> 16) & 1;
    let round_up = round_bit & (sticky | lsb);
    // Adding (round_up << 16) may carry into exponent — that's correct.
    ((bits.wrapping_add(round_up << 16)) >> 16) as u16
}

/// Convert a slice of f32 to bf16 (uint16).
pub fn f32_to_bf16_u16(arr: &[f32]) -> Vec<u16> {
    arr.iter().map(|&v| f32_to_bf16_u16_single(v)).collect()
}

/// Convert bf16 (uint16) to f32.
#[inline]
pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

// ─── INT4 quantization ───────────────────────────────────────────────────────

/// Quantize f32 [out_dim, in_dim] row-major → (packed_u32, scales_bf16, biases_bf16).
///
/// Each group of GROUP_SIZE (64) contiguous elements is quantized independently:
///   scale  = (max - min) / 15
///   bias   = min
///   nibble = round((v - bias) / scale), clamped to [0, 15]
///
/// 8 nibbles are packed LSB-first into one u32.  The degenerate case
/// (max == min) is handled by setting max = min + 1.0.
pub fn quant_f32_to_int4(
    f32_vals: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
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

            // Find min/max
            let mut vmin = group[0];
            let mut vmax = group[0];
            for &v in &group[1..] {
                vmin = vmin.min(v);
                vmax = vmax.max(v);
            }
            if vmax == vmin {
                vmax = vmin + 1.0;
            }

            let fscale = (vmax - vmin) / 15.0;
            let fbias = vmin;

            let s_idx = row * num_groups + g;
            scales[s_idx] = f32_to_bf16_u16_single(fscale);
            biases[s_idx] = f32_to_bf16_u16_single(fbias);

            let inv_scale = 1.0 / fscale;
            // Pack 8 groups of 8 nibbles → 8 u32 words
            for p in 0..8 {
                let mut word: u32 = 0;
                for n in 0..8 {
                    let v = group[p * 8 + n];
                    let q = ((v - fbias) * inv_scale + 0.5) as i32;
                    let nibble = (q.clamp(0, 15) as u32) & 0xF;
                    word |= nibble << (n * 4);
                }
                packed[row * words_per_row + g * 8 + p] = word;
            }
        }
    }

    (packed, scales, biases)
}

// ─── INT4 dequantization (for verification) ─────────────────────────────────

/// Dequantize affine INT4 packed weights back to f32 [out_dim, in_dim].
pub fn int4_to_f32(
    packed: &[u32],
    scales: &[u16],
    biases: &[u16],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
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

// ─── Encode helpers (called by Quant::encode) ────────────────────────────────

fn encode_int4(f32_vals: &[f32], out_dim: usize, in_dim: usize) -> Vec<EncodedTensor> {
    let num_groups = in_dim / GROUP_SIZE;
    let (packed, scales, biases) = quant_f32_to_int4(f32_vals, out_dim, in_dim);

    // Convert Vec<u32> → bytes, Vec<u16> → bytes
    let packed_bytes: Vec<u8> = unsafe {
        let ptr = packed.as_ptr() as *const u8;
        let len = packed.len() * 4;
        std::slice::from_raw_parts(ptr, len).to_vec()
    };
    let scales_bytes: Vec<u8> = unsafe {
        let ptr = scales.as_ptr() as *const u8;
        let len = scales.len() * 2;
        std::slice::from_raw_parts(ptr, len).to_vec()
    };
    let biases_bytes: Vec<u8> = unsafe {
        let ptr = biases.as_ptr() as *const u8;
        let len = biases.len() * 2;
        std::slice::from_raw_parts(ptr, len).to_vec()
    };

    vec![
        EncodedTensor {
            data: packed_bytes,
            suffix: ".weight",
            shape: vec![out_dim, in_dim / 8],
            dtype: "u32",
        },
        EncodedTensor {
            data: scales_bytes,
            suffix: ".scales",
            shape: vec![out_dim, num_groups],
            dtype: "bf16",
        },
        EncodedTensor {
            data: biases_bytes,
            suffix: ".biases",
            shape: vec![out_dim, num_groups],
            dtype: "bf16",
        },
    ]
}

fn encode_bf16(f32_vals: &[f32]) -> Vec<EncodedTensor> {
    let u16 = f32_to_bf16_u16(f32_vals);
    let bytes: Vec<u8> = unsafe {
        let ptr = u16.as_ptr() as *const u8;
        let len = u16.len() * 2;
        std::slice::from_raw_parts(ptr, len).to_vec()
    };
    vec![EncodedTensor {
        data: bytes,
        suffix: "",
        shape: vec![f32_vals.len()],
        dtype: "bf16",
    }]
}

fn encode_fp32(f32_vals: &[f32]) -> Vec<EncodedTensor> {
    let bytes: Vec<u8> = unsafe {
        let ptr = f32_vals.as_ptr() as *const u8;
        let len = f32_vals.len() * 4;
        std::slice::from_raw_parts(ptr, len).to_vec()
    };
    vec![EncodedTensor {
        data: bytes,
        suffix: "",
        shape: vec![f32_vals.len()],
        dtype: "f32",
    }]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_on_last_dot() {
        assert_eq!(
            split_on_last_dot("language_model.model.layers.0.self_attn.q_proj.weight"),
            ("language_model.model.layers.0.self_attn.q_proj", "weight")
        );
        assert_eq!(split_on_last_dot("model.norm.weight"), ("model.norm", "weight"));
        assert_eq!(split_on_last_dot("nodot"), ("nodot", ""));
    }

    #[test]
    fn test_strip_layer_prefix() {
        assert_eq!(
            strip_layer_prefix("language_model.model.layers.3.self_attn.q_proj"),
            "self_attn.q_proj"
        );
        assert_eq!(
            strip_layer_prefix("language_model.lm_head"),
            "lm_head"
        );
        assert_eq!(
            strip_layer_prefix("vision_tower.blocks.0.self_attn.q_proj"),
            "self_attn.q_proj"
        );
        assert_eq!(
            strip_layer_prefix("vision_tower.patch_embed.proj"),
            "patch_embed.proj"
        );
        assert_eq!(
            strip_layer_prefix("mtp.layers.0.fc"),
            "fc"
        );
        assert_eq!(
            strip_layer_prefix("mtp.shared_head.norm"),
            "shared_head.norm"
        );
        // No prefix → unchanged
        assert_eq!(
            strip_layer_prefix("self_attn.q_proj"),
            "self_attn.q_proj"
        );
    }

    #[test]
    fn test_matrix_table_known_blocks() {
        assert_eq!(matrix_table("self_attn.q_proj"), Quant::Bf16Pass);
        assert_eq!(matrix_table("self_attn.k_proj"), Quant::Bf16Pass);
        assert_eq!(matrix_table("self_attn.v_proj"), Quant::Bf16Pass);
        assert_eq!(matrix_table("self_attn.o_proj"), Quant::Bf16Pass);
        assert_eq!(matrix_table("mlp.gate"), Quant::Bf16Pass);
        assert_eq!(matrix_table("lm_head"), Quant::Bf16Pass);
        assert_eq!(matrix_table("attn.qkv"), Quant::Bf16Pass);
        assert_eq!(matrix_table("attn.proj"), Quant::Bf16Pass);
        assert_eq!(matrix_table("patch_embed.proj"), Quant::Bf16Pass);
        assert_eq!(matrix_table("pos_embed"), Quant::Bf16Pass);
    }

    #[test]
    fn test_matrix_table_unknown_blocks() {
        assert_eq!(matrix_table("mlp.switch_mlp.gate_up_proj"), Quant::Int4);
        assert_eq!(matrix_table("mlp.switch_mlp.down_proj"), Quant::Int4);
        assert_eq!(matrix_table("mlp.shared_expert.gate_proj"), Quant::Int4);
        assert_eq!(matrix_table("embed_tokens"), Quant::Int4);
        assert_eq!(matrix_table("linear_attn.in_proj_q"), Quant::Int4);
    }

    #[test]
    fn test_bq4_attention_projection() {
        let q = bq4(
            "language_model.model.layers.3.self_attn.q_proj.weight",
            &[8192, 2048],
        );
        assert_eq!(q, Quant::Bf16Pass);
    }

    #[test]
    fn test_bq4_expert() {
        let q = bq4(
            "language_model.model.layers.0.mlp.switch_mlp.gate_up_proj.weight",
            &[256, 2048],
        );
        assert_eq!(q, Quant::Int4);
    }

    #[test]
    fn test_bq4_norm_vector() {
        let q = bq4(
            "language_model.model.layers.0.input_layernorm.weight",
            &[2048],
        );
        assert_eq!(q, Quant::Bf16);
    }

    #[test]
    fn test_bq4_a_log() {
        let q = bq4(
            "language_model.model.layers.0.linear_attn.A_log",
            &[128],
        );
        assert_eq!(q, Quant::Fp32);
    }

    #[test]
    fn test_bq4_scales() {
        let q = bq4(
            "language_model.model.embed_tokens.scales",
            &[248320, 32],
        );
        assert_eq!(q, Quant::Bf16);
    }

    #[test]
    fn test_bq4_lm_head() {
        let q = bq4(
            "language_model.lm_head.weight",
            &[248320, 2048],
        );
        assert_eq!(q, Quant::Bf16Pass);
    }

    #[test]
    fn test_f32_to_bf16_roundtrip() {
        let vals: Vec<f32> = vec![0.0, 1.0, -1.0, 3.14159, 0.001, 1000.0];
        let bf16 = f32_to_bf16_u16(&vals);
        let back: Vec<f32> = bf16.iter().map(|&v| bf16_to_f32(v)).collect();
        for (orig, &recon) in vals.iter().zip(back.iter()) {
            let err = (orig - recon).abs();
            let rel = if *orig == 0.0 { err } else { err / orig.abs() };
            assert!(rel < 0.01, "orig={}, recon={}, rel_err={}", orig, recon, rel);
        }
    }

    #[test]
    fn test_int4_roundtrip_small() {
        // 2 rows, 128 cols (2 groups per row)
        let out_dim = 2;
        let in_dim = 128;
        let vals: Vec<f32> = (0..(out_dim * in_dim))
            .map(|i| (i as f32).sin())
            .collect();

        let (packed, scales, biases) = quant_f32_to_int4(&vals, out_dim, in_dim);
        let recon = int4_to_f32(&packed, &scales, &biases, out_dim, in_dim);

        // Check within-group reconstruction
        for g in 0..(in_dim / GROUP_SIZE) {
            let row = 0;
            let base = row * in_dim + g * GROUP_SIZE;
            let group_orig = &vals[base..base + GROUP_SIZE];
            let group_recon = &recon[base..base + GROUP_SIZE];

            // INT4 should recover values well within each group
            let max_err = group_orig
                .iter()
                .zip(group_recon.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let range = group_orig.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
                - group_orig.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            if range > 0.001 {
                assert!(max_err / range < 0.1, "max_err={} range={}", max_err, range);
            }
        }
    }

    #[test]
    fn test_int4_degenerate_group() {
        // All-constant input: max == min for each group
        let out_dim = 1;
        let in_dim = 128;
        let vals = vec![5.0f32; out_dim * in_dim];

        let (packed, scales, biases) = quant_f32_to_int4(&vals, out_dim, in_dim);
        let recon = int4_to_f32(&packed, &scales, &biases, out_dim, in_dim);

        // Degenerate case: scale = (1.0) / 15, bias = 5.0, nibble = 0 → recon = 5.0
        for (i, &r) in recon.iter().enumerate() {
            assert!(
                (r - 5.0).abs() < 0.001,
                "idx {}: expected ~5.0, got {}", i, r
            );
        }
    }

    #[test]
    fn test_classify_weight() {
        assert_eq!(
            classify_weight(
                "language_model.model.layers.0.self_attn.q_proj.weight",
                &[8192, 2048]
            ),
            "bf16"
        );
        assert_eq!(
            classify_weight(
                "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                &[256, 2048]
            ),
            "u32"
        );
        assert_eq!(
            classify_weight(
                "language_model.model.layers.0.linear_attn.A_log",
                &[128]
            ),
            "f32"
        );
    }
}

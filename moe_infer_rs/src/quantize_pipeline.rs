// Full A→Z quantization pipeline: read HF safetensors → classify → encode → write BQ4.
//
// Called from Python via a single `moe_infer.quantize()` function.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::quantize::{self, bq4, Quant, GROUP_SIZE};

const ALIGN: u64 = 64;

/// Extract layer index from tensor name.  Looks for `layers.{N}.` pattern.
fn extract_layer(name: &str) -> Option<usize> {
    let haystack = name;
    let mut start = 0;
    while let Some(pos) = haystack[start..].find("layers.") {
        let after = start + pos + 7; // after "layers."
        let digits_end = haystack[after..]
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(haystack.len() - after);
        if digits_end > 0 {
            if let Ok(n) = haystack[after..after + digits_end].parse::<usize>() {
                // Check there's a dot after the digits (e.g. "layers.0." not "layers.0abc")
                if after + digits_end < haystack.len()
                    && haystack.as_bytes()[after + digits_end] == b'.'
                {
                    return Some(n);
                }
            }
        }
        start = after;
    }
    None
}

// ─── Name mapping ────────────────────────────────────────────────────────────

type NameMap = HashMap<String, String>;

fn load_name_mapping(
    path: &Path,
    num_layers: usize,
    num_vision_blocks: usize,
) -> Result<NameMap, String> {
    let json_str = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mapping: HashMap<String, String> =
        serde_json::from_str(&json_str).map_err(|e| e.to_string())?;

    let mut flat = HashMap::new();
    for (hf_pat, mlx_pat) in &mapping {
        if hf_pat.contains("{L}") {
            for l in 0..num_layers {
                flat.insert(
                    hf_pat.replace("{L}", &l.to_string()),
                    mlx_pat.replace("{L}", &l.to_string()),
                );
            }
        } else if hf_pat.contains("{B}") {
            for b in 0..num_vision_blocks {
                flat.insert(
                    hf_pat.replace("{B}", &b.to_string()),
                    mlx_pat.replace("{B}", &b.to_string()),
                );
            }
        } else {
            flat.insert(hf_pat.clone(), mlx_pat.clone());
        }
    }
    Ok(flat)
}

// ─── HF config ───────────────────────────────────────────────────────────────

struct ModelParams {
    hidden_dim: usize,
    num_layers: usize,
    moe_intermediate: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    shared_intermediate: usize,
    mtp_num_layers: usize,
    full_attn_interval: usize,
    vocab_size: usize,
    num_attn_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    linear_num_value_heads: usize,
    linear_num_key_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel_dim: usize,
    partial_rotary_factor: f64,
    rope_theta: f64,
    hf_config_raw: serde_json::Value,
}

fn load_config(model_path: &Path) -> Result<ModelParams, String> {
    let json_str =
        fs::read_to_string(model_path.join("config.json")).map_err(|e| e.to_string())?;
    let root: serde_json::Value =
        serde_json::from_str(&json_str).map_err(|e| e.to_string())?;
    let tc = root.get("text_config").unwrap_or(&root);

    let get = |key: &str, default: usize| -> usize {
        tc.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    };
    let get_f = |key: &str, default: f64| -> f64 {
        tc.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    };
    let rope = tc.get("rope_parameters");

    Ok(ModelParams {
        hidden_dim: get("hidden_size", 0),
        num_layers: get("num_hidden_layers", 0),
        moe_intermediate: get("moe_intermediate_size", 0),
        num_experts: get("num_experts", 0),
        num_experts_per_tok: get("num_experts_per_tok", 0),
        shared_intermediate: get("shared_expert_intermediate_size", 0),
        mtp_num_layers: get("mtp_num_hidden_layers", 0),
        full_attn_interval: get("full_attention_interval", 4),
        vocab_size: get("vocab_size", 0),
        num_attn_heads: get("num_attention_heads", 0),
        num_kv_heads: get("num_key_value_heads", 0),
        head_dim: get("head_dim", 0),
        rms_norm_eps: get_f("rms_norm_eps", 1e-6),
        linear_num_value_heads: get("linear_num_value_heads", 32),
        linear_num_key_heads: get("linear_num_key_heads", 16),
        linear_key_head_dim: get("linear_key_head_dim", 128),
        linear_value_head_dim: get("linear_value_head_dim", 128),
        linear_conv_kernel_dim: get("linear_conv_kernel_dim", 4),
        partial_rotary_factor: get_f(
            "partial_rotary_factor",
            rope.and_then(|r| r.get("partial_rotary_factor"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.25),
        ),
        rope_theta: get_f(
            "rope_theta",
            rope.and_then(|r| r.get("rope_theta"))
                .and_then(|v| v.as_f64())
                .unwrap_or(10_000_000.0),
        ),
        hf_config_raw: root,
    })
}

// ─── Safetensors I/O ─────────────────────────────────────────────────────────

struct TensorMeta {
    shape: Vec<usize>,
    dtype: String,
    data_offsets: [u64; 2],
}

struct ShardHeader {
    data_start: u64,
    tensors: HashMap<String, TensorMeta>,
}

fn parse_safetensors(path: &Path) -> Result<ShardHeader, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hdr_len_buf = [0u8; 8];
    f.read_exact(&mut hdr_len_buf).map_err(|e| e.to_string())?;
    let hdr_len = u64::from_le_bytes(hdr_len_buf) as usize;

    let mut hdr_json = vec![0u8; hdr_len];
    f.read_exact(&mut hdr_json).map_err(|e| e.to_string())?;
    let root: serde_json::Value =
        serde_json::from_slice(&hdr_json).map_err(|e| e.to_string())?;

    let mut tensors = HashMap::new();
    if let Some(obj) = root.as_object() {
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = meta["dtype"].as_str().unwrap_or("").to_string();
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
                .unwrap_or_default();
            let off = meta["data_offsets"].as_array().map_or([0, 0], |a| {
                [a[0].as_u64().unwrap_or(0), a[1].as_u64().unwrap_or(0)]
            });
            tensors.insert(name.clone(), TensorMeta { shape, dtype, data_offsets: off });
        }
    }
    Ok(ShardHeader {
        data_start: 8 + hdr_len as u64,
        tensors,
    })
}

fn read_tensor_bytes(
    shard_path: &Path,
    header: &ShardHeader,
    name: &str,
) -> Result<Vec<u8>, String> {
    let meta = header
        .tensors
        .get(name)
        .ok_or_else(|| format!("tensor '{}' not found in {}", name, shard_path.display()))?;
    let off = meta.data_offsets;
    let len = (off[1] - off[0]) as usize;

    let mut f = fs::File::open(shard_path).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start(header.data_start + off[0]))
        .map_err(|e| e.to_string())?;
    let mut data = vec![0u8; len];
    f.read_exact(&mut data).map_err(|e| e.to_string())?;
    Ok(data)
}

// ─── Source dtype → f32 conversion ───────────────────────────────────────────

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        // Zero / subnormal
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: normalize
            let m = mant;
            let e = 1i32 - 15 - 10; // min exponent for f16, minus mantissa bits
            let mut m2 = m;
            while m2 < 0x400 {
                m2 <<= 1;
            }
            let actual_exp = e + (m2.leading_zeros() as i32 - 21);
            let actual_mant = (m2 & 0x3FF) << 13;
            f32::from_bits((sign << 31) | (((actual_exp + 127) as u32) << 23) | actual_mant)
        }
    } else if exp == 0x1F {
        // Infinity / NaN
        f32::from_bits((sign << 31) | 0x7F80_0000 | (mant << 13))
    } else {
        // Normal
        let e = (exp as i32) - 15 + 127;
        f32::from_bits((sign << 31) | ((e as u32) << 23) | (mant << 13))
    }
}

fn bytes_to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F32" => {
            let n = data.len() / 4;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let b = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
                out.push(f32::from_le_bytes(b));
            }
            out
        }
        "F16" => {
            let n = data.len() / 2;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let lo = data[i * 2] as u16;
                let hi = data[i * 2 + 1] as u16;
                out.push(f16_to_f32(lo | (hi << 8)));
            }
            out
        }
        _ => {
            // BF16 (default for Qwen HF models)
            let n = data.len() / 2;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let lo = data[i * 2] as u16;
                let hi = data[i * 2 + 1] as u16;
                out.push(quantize::bf16_to_f32(lo | (hi << 8)));
            }
            out
        }
    }
}

// ─── Sanitization ────────────────────────────────────────────────────────────

const MLX_NORM_KEYS: &[&str] = &[
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
];
const MTP_NORM_KEYS: &[&str] = &[
    ".hnorm.weight", ".enorm.weight", ".shared_head.norm.weight",
    ".norm1.weight", ".norm2.weight",
];

fn is_norm_key(mlx_name: &str) -> bool {
    MLX_NORM_KEYS
        .iter()
        .chain(MTP_NORM_KEYS)
        .any(|k| mlx_name.ends_with(k))
}

/// np.moveaxis(arr, 2, 1) for 3D f32 arrays: [C, K, S] → [C, S, K]
fn moveaxis_2_to_1(vals: &mut [f32], shape: &mut Vec<usize>) {
    let (c, k, s) = (shape[0], shape[1], shape[2]);
    let orig = vals.to_vec();
    for ci in 0..c {
        for ki in 0..k {
            for si in 0..s {
                let old_idx = ci * (k * s) + ki * s + si;
                let new_idx = ci * (s * k) + si * k + ki;
                vals[new_idx] = orig[old_idx];
            }
        }
    }
    *shape = vec![c, s, k];
}

// ─── Expert detection ────────────────────────────────────────────────────────

fn is_expert_tensor(mlx_name: &str) -> bool {
    mlx_name.contains(".switch_mlp.gate_proj.")
        || mlx_name.contains(".switch_mlp.gate_up_proj.")
        || mlx_name.contains(".switch_mlp.up_proj.")
        || mlx_name.contains(".switch_mlp.down_proj.")
}

// ─── Expert pack size ────────────────────────────────────────────────────────

fn expert_pack_size(hd: usize, mi: usize) -> usize {
    let gs = GROUP_SIZE;
    let gate_w = mi * hd / 2;
    let gate_sb = mi * (hd / gs) * 2;
    let up_w = mi * hd / 2;
    let up_sb = mi * (hd / gs) * 2;
    let down_w = hd * mi / 2;
    let down_sb = hd * (mi / gs) * 2;
    gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w + 2 * down_sb
}

// ─── Main pipeline ───────────────────────────────────────────────────────────

pub fn run(
    model_path: &str,
    output_dir: &str,
    name_mapping_path: &str,
    qwen36: bool,
    strip_layers: usize,
    strip_experts: usize,
) -> Result<(), String> {
    let model_path = Path::new(model_path);
    let output_dir = Path::new(output_dir);
    let mapping_path = Path::new(name_mapping_path);

    // ── 1. Load config ──────────────────────────────────────────────────
    let params = load_config(model_path)?;
    let hd = params.hidden_dim;
    let num_layers = params.num_layers;
    let mi = params.moe_intermediate;
    let num_experts = params.num_experts;
    let num_experts_per_tok = params.num_experts_per_tok;
    let shared_inter = params.shared_intermediate;
    let mtp_layers = params.mtp_num_layers;
    let full_attn_interval = params.full_attn_interval;
    let vocab_size = params.vocab_size;
    let num_attn_heads = params.num_attn_heads;
    let num_kv_heads = params.num_kv_heads;
    let head_dim = params.head_dim;
    let num_main_layers = num_layers - mtp_layers;

    eprintln!("Model config:");
    eprintln!("  hidden_dim={}, vocab_size={}", hd, vocab_size);
    eprintln!(
        "  num_layers={} (main={}, mtp={})",
        num_layers, num_main_layers, mtp_layers
    );
    eprintln!(
        "  num_experts={}, experts_per_tok={}",
        num_experts, num_experts_per_tok
    );
    eprintln!("  moe_intermediate={}, shared_intermediate={}", mi, shared_inter);

    // ── 2. Load name mapping ────────────────────────────────────────────
    let name_map = load_name_mapping(mapping_path, num_layers, 27)?;
    eprintln!("  Name mapping entries: {}", name_map.len());

    // ── 3. Load weight map ──────────────────────────────────────────────
    let index_path = model_path.join("model.safetensors.index.json");
    let weight_map = if index_path.exists() {
        let idx_str = fs::read_to_string(&index_path).map_err(|e| e.to_string())?;
        let idx: serde_json::Value =
            serde_json::from_str(&idx_str).map_err(|e| e.to_string())?;
        let mut wm = HashMap::new();
        if let Some(map) = idx["weight_map"].as_object() {
            for (k, v) in map {
                wm.insert(k.clone(), v.as_str().unwrap_or("").to_string());
            }
        }
        wm
    } else {
        eprintln!("  No safetensors index found, scanning shards...");
        let mut wm = HashMap::new();
        let mut shards: Vec<PathBuf> = Vec::new();
        if let Ok(entries) = fs::read_dir(model_path) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    shards.push(entry.path());
                }
            }
        }
        shards.sort();
        for shard_path in &shards {
            let header = parse_safetensors(shard_path)?;
            for k in header.tensors.keys() {
                wm.insert(
                    k.clone(),
                    shard_path.file_name().unwrap().to_string_lossy().to_string(),
                );
            }
        }
        wm
    };
    eprintln!("  Total tensors: {}", weight_map.len());

    // ── 4. Classify tensors ─────────────────────────────────────────────
    let mut non_expert: Vec<(String, String)> = Vec::new(); // (hf_name, shard)
    let mut expert: Vec<(String, String)> = Vec::new();
    let mut unmapped: Vec<String> = Vec::new();

    for (hf_name, shard) in weight_map.iter() {
        match name_map.get(hf_name) {
            Some(mlx_name) => {
                if is_expert_tensor(mlx_name) {
                    expert.push((hf_name.clone(), shard.clone()));
                } else {
                    non_expert.push((hf_name.clone(), shard.clone()));
                }
            }
            None => unmapped.push(hf_name.clone()),
        }
    }

    if !unmapped.is_empty() {
        return Err(format!(
            "{} tensors not in name mapping:\n  {}",
            unmapped.len(),
            unmapped.iter().take(20).cloned().collect::<Vec<_>>().join("\n  ")
        ));
    }

    non_expert.sort_by(|a, b| a.0.cmp(&b.0));
    expert.sort_by(|a, b| a.0.cmp(&b.0));

    eprintln!(
        "  Non-expert: {}, Expert: {}",
        non_expert.len(),
        expert.len()
    );

    // ── 4b. Strip mode ──────────────────────────────────────────────────
    let (eff_num_layers, eff_num_experts) = if strip_layers > 0 {
        let sl = strip_layers;
        let se = if strip_experts > 0 {
            strip_experts
        } else {
            num_experts
        };
        eprintln!("  [strip] layers={}, experts={}", sl, se);

        // Filter non-expert
        non_expert.retain(|(hf_name, _)| {
            match extract_layer(hf_name) {
                Some(n) => n < sl,
                None => true, // non-layered tensors
            }
        });
        expert.retain(|(hf_name, _)| {
            match extract_layer(hf_name) {
                Some(n) => n < sl,
                None => true,
            }
        });

        eprintln!(
            "  [strip] non-expert: {}, expert: {}",
            non_expert.len(),
            expert.len()
        );
        (sl, se)
    } else {
        (num_layers, num_experts)
    };

    // ── 5. Cache shard headers ──────────────────────────────────────────
    let mut shard_set: HashSet<String> = HashSet::new();
    for (_, shard) in non_expert.iter().chain(expert.iter()) {
        shard_set.insert(shard.clone());
    }
    let mut header_cache: HashMap<String, ShardHeader> = HashMap::new();
    for shard_name in &shard_set {
        let shard_path = model_path.join(shard_name);
        if shard_path.exists() {
            eprintln!("  Caching header: {}", shard_name);
            header_cache.insert(shard_name.clone(), parse_safetensors(&shard_path)?);
        }
    }

    // ── 6. Create output directory ──────────────────────────────────────
    fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    let experts_dir = output_dir.join("packed_experts");
    fs::create_dir_all(&experts_dir).map_err(|e| e.to_string())?;

    // ── 7. Quantize non-expert → model_weights.bin + .json ──────────────
    eprintln!("\n============================================================");
    eprintln!("Quantizing non-expert weights (BQ4)...");
    eprintln!("============================================================");

    let mut manifest: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    manifest.insert("model".into(), serde_json::Value::String(model_path.to_string_lossy().into()));

    // Build config
    let mut cfg = serde_json::Map::new();
    macro_rules! ins {
        ($k:expr, $v:expr) => {
            cfg.insert($k.into(), serde_json::Value::from($v));
        };
    }
    ins!("hidden_size", hd);
    ins!("num_hidden_layers", eff_num_layers);
    ins!("num_attention_heads", num_attn_heads);
    ins!("num_key_value_heads", num_kv_heads);
    ins!("head_dim", head_dim);
    ins!("vocab_size", vocab_size);
    ins!("rms_norm_eps", params.rms_norm_eps);
    ins!("num_experts", eff_num_experts);
    ins!("num_experts_per_tok", num_experts_per_tok);
    ins!("moe_intermediate_size", mi);
    ins!("shared_expert_intermediate_size", shared_inter);
    ins!("full_attention_interval", full_attn_interval);
    ins!("linear_num_value_heads", params.linear_num_value_heads);
    ins!("linear_num_key_heads", params.linear_num_key_heads);
    ins!("linear_key_head_dim", params.linear_key_head_dim);
    ins!("linear_value_head_dim", params.linear_value_head_dim);
    ins!("linear_conv_kernel_dim", params.linear_conv_kernel_dim);
    ins!("partial_rotary_factor", params.partial_rotary_factor);
    ins!("rope_theta", params.rope_theta);
    ins!("mtp_num_hidden_layers", if strip_layers > 0 { 0 } else { mtp_layers });

    // Layer types
    let layer_types: Vec<String> = (0..eff_num_layers.min(num_main_layers))
        .map(|i| {
            if (i + 1) % full_attn_interval == 0 {
                "full_attention".to_string()
            } else {
                "linear_attention".to_string()
            }
        })
        .collect();
    cfg.insert("layer_types".into(), serde_json::Value::Array(
        layer_types.iter().map(|s| serde_json::Value::String(s.clone())).collect()
    ));

    manifest.insert("config".into(), serde_json::Value::Object(cfg));
    manifest.insert("num_tensors".into(), serde_json::Value::from(0));

    let bin_path = output_dir.join("model_weights.bin");
    let mut out_f =
        fs::File::create(&bin_path).map_err(|e| format!("cannot create {}: {}", bin_path.display(), e))?;
    let mut offset: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut tensor_count: u64 = 0;
    let mut tensors_map = serde_json::Map::new();
    let mut quant_summary: HashMap<String, usize> = HashMap::new();

    let t0 = std::time::Instant::now();

    for (hf_name, shard_name) in &non_expert {
        let header = header_cache
            .get(shard_name)
            .ok_or_else(|| format!("header not cached for {}", shard_name))?;

        if !header.tensors.contains_key(hf_name) {
            eprintln!("  WARNING: {} not in {}, skipping", hf_name, shard_name);
            continue;
        }

        let meta = &header.tensors[hf_name];
        let shape = meta.shape.clone();
        let dtype = meta.dtype.clone();
        let mlx_name = name_map[hf_name].clone();

        // Read raw bytes
        let raw_data = read_tensor_bytes(&model_path.join(shard_name), header, hf_name)?;

        // Classify
        let q = bq4(&mlx_name, &shape);
        let q_str = q.as_str().to_string();
        *quant_summary.entry(q_str.clone()).or_insert(0) += 1;

        // Read source as F32
        let mut f32_vals = bytes_to_f32(&raw_data, &dtype);
        let mut out_shape = shape.clone();

        // ── Sanitize (BF16 only) ────────────────────────────────────
        if q.needs_sanitization() {
            if qwen36 && is_norm_key(&mlx_name) {
                for v in &mut f32_vals {
                    *v += 1.0;
                }
            }
            if mlx_name.contains("conv1d.weight") && out_shape.len() == 3 {
                moveaxis_2_to_1(&mut f32_vals, &mut out_shape);
            }
        }

        // ── Pad inner dim for INT4 ──────────────────────────────────
        let out_dim = out_shape[0];
        let in_dim = if out_shape.len() >= 2 { out_shape[1] } else { 0 };
        let (padded_in, f32_padded) = if q == Quant::Int4 {
            let pi = (in_dim + GROUP_SIZE - 1) / GROUP_SIZE * GROUP_SIZE;
            if pi != in_dim {
                let mut p = vec![0.0f32; out_dim * pi];
                for r in 0..out_dim {
                    let src = r * in_dim;
                    let dst = r * pi;
                    p[dst..dst + in_dim].copy_from_slice(&f32_vals[src..src + in_dim]);
                }
                (pi, p)
            } else {
                (in_dim, f32_vals)
            }
        } else {
            (in_dim, f32_vals)
        };

        // ── Align ──────────────────────────────────────────────────
        if offset % ALIGN != 0 {
            let pad = ALIGN - (offset % ALIGN);
            out_f
                .write_all(&vec![0u8; pad as usize])
                .map_err(|e| e.to_string())?;
            offset += pad;
        }

        // ── Encode ─────────────────────────────────────────────────
        let base = if mlx_name.ends_with(".weight") {
            mlx_name[..mlx_name.len() - 7].to_string()
        } else {
            mlx_name.clone()
        };

        match q {
            Quant::Int4 => {
                let (packed, scales, biases) =
                    quantize::quant_f32_to_int4(&f32_padded, out_dim, padded_in);
                let num_groups = padded_in / GROUP_SIZE;

                let packed_bytes: Vec<u8> =
                    packed.iter().flat_map(|v| v.to_le_bytes()).collect();
                let scales_bytes: Vec<u8> =
                    scales.iter().flat_map(|v| v.to_le_bytes()).collect();
                let biases_bytes: Vec<u8> =
                    biases.iter().flat_map(|v| v.to_le_bytes()).collect();

                for (data, suffix, data_shape, data_dtype) in [
                    (packed_bytes, ".weight", vec![out_dim, padded_in / 8], "u32"),
                    (scales_bytes, ".scales", vec![out_dim, num_groups], "bf16"),
                    (biases_bytes, ".biases", vec![out_dim, num_groups], "bf16"),
                ] {
                    let tname = format!("{}{}", base, suffix);
                    let dlen = data.len() as u64;
                    out_f.write_all(&data).map_err(|e| e.to_string())?;
                    let mut entry = serde_json::Map::new();
                    entry.insert("offset".into(), serde_json::Value::from(offset));
                    entry.insert("size".into(), serde_json::Value::from(dlen));
                    entry.insert(
                        "shape".into(),
                        serde_json::Value::Array(
                            data_shape
                                .iter()
                                .map(|&n| serde_json::Value::from(n as u64))
                                .collect(),
                        ),
                    );
                    entry.insert("dtype".into(), serde_json::Value::String(data_dtype.into()));
                    tensors_map.insert(tname, serde_json::Value::Object(entry));
                    offset += dlen;
                    total_bytes += dlen;
                    tensor_count += 1;
                }
            }
            Quant::Fp32 => {
                let data = f32_padded
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>();
                let dlen = data.len() as u64;
                out_f.write_all(&data).map_err(|e| e.to_string())?;
                let mut entry = serde_json::Map::new();
                entry.insert("offset".into(), serde_json::Value::from(offset));
                entry.insert("size".into(), serde_json::Value::from(dlen));
                entry.insert(
                    "shape".into(),
                    serde_json::Value::Array(
                        out_shape
                            .iter()
                            .map(|&n| serde_json::Value::from(n as u64))
                            .collect(),
                    ),
                );
                entry.insert("dtype".into(), serde_json::Value::String("f32".into()));
                tensors_map.insert(base, serde_json::Value::Object(entry));
                offset += dlen;
                total_bytes += dlen;
                tensor_count += 1;
            }
            Quant::Bf16 | Quant::Bf16Pass => {
                let bf16 = quantize::f32_to_bf16_u16(&f32_padded);
                let data: Vec<u8> = bf16.iter().flat_map(|v| v.to_le_bytes()).collect();
                let dlen = data.len() as u64;
                out_f.write_all(&data).map_err(|e| e.to_string())?;
                let mut entry = serde_json::Map::new();
                entry.insert("offset".into(), serde_json::Value::from(offset));
                entry.insert("size".into(), serde_json::Value::from(dlen));
                entry.insert(
                    "shape".into(),
                    serde_json::Value::Array(
                        out_shape
                            .iter()
                            .map(|&n| serde_json::Value::from(n as u64))
                            .collect(),
                    ),
                );
                entry.insert("dtype".into(), serde_json::Value::String("bf16".into()));
                tensors_map.insert(mlx_name.clone(), serde_json::Value::Object(entry));
                offset += dlen;
                total_bytes += dlen;
                tensor_count += 1;
            }
        }
    }

    manifest.insert("num_tensors".into(), serde_json::Value::from(tensor_count));
    manifest.insert("tensors".into(), serde_json::Value::Object(tensors_map));

    let elapsed = t0.elapsed();
    eprintln!(
        "  {} tensors, {:.2} GB",
        tensor_count,
        total_bytes as f64 / 1e9
    );
    eprintln!(
        "  Written in {:.1}s ({:.1} GB/s)",
        elapsed.as_secs_f64(),
        total_bytes as f64 / elapsed.as_secs_f64() / 1e9
    );
    eprintln!("  By dtype: {:?}", quant_summary);

    // Write manifest
    let json_path = output_dir.join("model_weights.json");
    let json_str = serde_json::to_string_pretty(&serde_json::Value::Object(manifest))
        .map_err(|e| e.to_string())?;
    fs::write(&json_path, json_str).map_err(|e| e.to_string())?;
    eprintln!("  Manifest: {}", json_path.display());

    // ── 8. Repack experts ──────────────────────────────────────────────
    eprintln!("\n============================================================");
    eprintln!("Quantizing expert weights (int4)...");
    eprintln!("============================================================");

    let t1 = std::time::Instant::now();

    // Group expert tensors by layer index
    let mut expert_layers: BTreeMap<usize, (String, String)> = BTreeMap::new();
    // layer_idx → (gate_up_proj_hf, down_proj_hf)

    for (hf_name, _shard) in &expert {
        if let Some(layer) = extract_layer(hf_name) {
            let entry = expert_layers.entry(layer).or_insert_with(|| (String::new(), String::new()));
            if hf_name.contains("gate_up_proj") || hf_name.contains("gate_.biases") || hf_name.contains("gate_.scales") {
                // The actual weight key for gate_up_proj
                if !hf_name.contains("biases") && !hf_name.contains("scales") {
                    entry.0 = hf_name.clone();
                }
            } else if hf_name.contains("down_proj") && !hf_name.contains("biases") && !hf_name.contains("scales") {
                entry.1 = hf_name.clone();
            }
        }
    }

    let mut expert_layers_done = 0usize;

    for (layer_idx, (gate_up_key, down_key)) in &expert_layers {
        if gate_up_key.is_empty() || down_key.is_empty() {
            eprintln!("  Layer {} SKIPPED (missing keys)", layer_idx);
            continue;
        }

        // Determine which shard has these tensors
        let gu_shard = weight_map.get(gate_up_key).ok_or("shard not found")?;
        let down_shard = weight_map.get(down_key).ok_or("shard not found")?;

        let gu_header = header_cache.get(gu_shard).ok_or("header not cached")?;
        let down_header = header_cache.get(down_shard).ok_or("header not cached")?;

        // Read gate_up_proj (fused [E, 2*I, H] as BF16)
        let gu_raw = read_tensor_bytes(&model_path.join(gu_shard), gu_header, gate_up_key)?;
        let gu_f32 = bytes_to_f32(&gu_raw, &gu_header.tensors[gate_up_key].dtype);
        // gu_f32 shape: [E, 2*I, H]

        // Read down_proj ([E, H, I] as BF16)
        let down_raw =
            read_tensor_bytes(&model_path.join(down_shard), down_header, down_key)?;
        let down_f32 = bytes_to_f32(&down_raw, &down_header.tensors[down_key].dtype);
        // down_f32 shape: [E, H, I]

        let inter = mi;
        let hidden = hd;
        let gs = GROUP_SIZE;

        // Layout per expert
        let gate_w_bytes = inter * (hidden / 8) * 4;
        let gate_s_bytes = inter * (hidden / gs) * 2;
        let gate_b_bytes = gate_s_bytes;
        let up_w_bytes = gate_w_bytes;
        let up_s_bytes = gate_s_bytes;
        let up_b_bytes = gate_b_bytes;
        let down_w_bytes = hidden * (inter / 8) * 4;
        let down_s_bytes = hidden * (inter / gs) * 2;
        let down_b_bytes = down_s_bytes;
        let expert_size = gate_w_bytes + gate_s_bytes + gate_b_bytes
            + up_w_bytes + up_s_bytes + up_b_bytes
            + down_w_bytes + down_s_bytes + down_b_bytes;

        let mut buf = vec![0u8; eff_num_experts * expert_size];

        for e in 0..eff_num_experts {
            // Extract gate [I, H] and up [I, H] from fused gate_up [2*I, H]
            let gu_base = e * (2 * inter * hidden);
            let gate_f32: Vec<f32> = gu_f32[gu_base..gu_base + inter * hidden].to_vec();
            let up_f32: Vec<f32> =
                gu_f32[gu_base + inter * hidden..gu_base + 2 * inter * hidden].to_vec();

            let down_base = e * (hidden * inter);
            let down_f32_e: Vec<f32> = down_f32[down_base..down_base + hidden * inter].to_vec();

            let (gate_p, gate_s, gate_b) = quantize::quant_f32_to_int4(&gate_f32, inter, hidden);
            let (up_p, up_s, up_b) = quantize::quant_f32_to_int4(&up_f32, inter, hidden);
            let (down_p, down_s, down_b) =
                quantize::quant_f32_to_int4(&down_f32_e, hidden, inter);

            let base = e * expert_size;

            // Gate weight (u32)
            let bytes: Vec<u8> = gate_p.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[base..base + gate_w_bytes].copy_from_slice(&bytes);
            let mut pos = base + gate_w_bytes;
            // Gate scales (u16)
            let bytes: Vec<u8> = gate_s.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + gate_s_bytes].copy_from_slice(&bytes);
            pos += gate_s_bytes;
            // Gate biases (u16)
            let bytes: Vec<u8> = gate_b.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + gate_b_bytes].copy_from_slice(&bytes);
            pos += gate_b_bytes;
            // Up weight (u32)
            let bytes: Vec<u8> = up_p.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + up_w_bytes].copy_from_slice(&bytes);
            pos += up_w_bytes;
            // Up scales (u16)
            let bytes: Vec<u8> = up_s.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + up_s_bytes].copy_from_slice(&bytes);
            pos += up_s_bytes;
            // Up biases (u16)
            let bytes: Vec<u8> = up_b.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + up_b_bytes].copy_from_slice(&bytes);
            pos += up_b_bytes;
            // Down weight (u32)
            let bytes: Vec<u8> = down_p.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + down_w_bytes].copy_from_slice(&bytes);
            pos += down_w_bytes;
            // Down scales (u16)
            let bytes: Vec<u8> = down_s.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + down_s_bytes].copy_from_slice(&bytes);
            pos += down_s_bytes;
            // Down biases (u16)
            let bytes: Vec<u8> = down_b.iter().flat_map(|v| v.to_le_bytes()).collect();
            buf[pos..pos + down_b_bytes].copy_from_slice(&bytes);
        }

        let out_path = experts_dir.join(format!("layer_{:02}.bin", layer_idx));
        fs::write(&out_path, &buf).map_err(|e| e.to_string())?;
        eprintln!(
            "  Layer {:02}: {:.1} MB → {}",
            layer_idx,
            buf.len() as f64 / 1e6,
            out_path.file_name().unwrap().to_string_lossy()
        );
        expert_layers_done += 1;
    }

    let t2 = t1.elapsed();
    eprintln!(
        "\n  {} expert layers in {:.1}s",
        expert_layers_done,
        t2.as_secs_f64()
    );

    // ── 9. Write config.json ────────────────────────────────────────────
    let dst_config = output_dir.join("config.json");
    let config_str = serde_json::to_string_pretty(&params.hf_config_raw)
        .map_err(|e| e.to_string())?;
    fs::write(&dst_config, config_str).map_err(|e| e.to_string())?;

    // ── 10. Summary ─────────────────────────────────────────────────────
    let total_time = t0.elapsed();
    let bin_size = fs::metadata(&bin_path)
        .map(|m| m.len())
        .unwrap_or(0);
    eprintln!("\n============================================================");
    eprintln!("Done!");
    eprintln!(
        "  model_weights.bin : {:.2} GB",
        bin_size as f64 / 1e9
    );
    eprintln!("  model_weights.json: {}", json_path.display());
    eprintln!("  packed_experts    : {} layers", expert_layers_done);
    eprintln!(
        "  Total time        : {:.1}s",
        total_time.as_secs_f64()
    );
    eprintln!("============================================================");

    Ok(())
}

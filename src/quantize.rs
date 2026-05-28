// Generic quantization pipeline: download → classify → encode → write.
//
// Works with any model once a QuantScheme is provided.  The scheme owns all
// model-specific knowledge: name mapping, dtype classification, expert tensor
// processing, sanitization, and manifest config.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::dtype::{DType, GROUP_SIZE};
use crate::hf_util::HfRepo;
use crate::safetensors::{bytes_to_f32, parse_safetensors, read_tensor_bytes};

const ALIGN: u64 = 64;

/// What to do with a given weight tensor.
#[derive(Clone, Debug)]
pub enum WeightKind {
    /// Process during shard pass, write to model_weights.bin.
    Normal,
    /// Defer to expert pass, write to packed_experts/layer_{N}.bin.
    Expert(usize),
    /// Ignore entirely (e.g. vision encoder tensors).
    Skip,
}

/// Everything the pipeline needs to know about a weight tensor.
#[derive(Clone)]
pub struct WeightClass {
    pub name: String,
    pub quant: DType,
    pub kind: WeightKind,
}

/// Model-specific quantization logic.
pub trait QuantScheme {
    /// Hidden dimension — used for INT4 padding.
    fn hidden_dim(&self) -> usize;
    /// Number of layers (after strip).
    fn num_layers(&self) -> usize;
    /// Number of experts (after strip).
    fn num_experts(&self) -> usize;

    /// Classify a single weight tensor by its raw name and shape.
    fn classify(&self, name: &str, shape: &[usize]) -> WeightClass;

    /// Optional pre-encode transform (norm shift, conv1d moveaxis, etc.).
    fn sanitize(&self, _name: &str, _values: &mut [f32], _shape: &mut Vec<usize>) {}

    /// Expert pass: scheme-specific processing for all expert tensors.
    /// Called after the shard pass.  Returns the number of expert layers processed.
    fn process_experts(
        &self,
        repo: &HfRepo,
        weight_map: &HashMap<String, String>,
        classified: &[(String, WeightClass)],
        output_dir: &Path,
    ) -> Result<usize, String>;

    /// Populate the `config` section of the manifest JSON.
    fn write_manifest_config(
        &self,
        cfg: &mut serde_json::Map<String, serde_json::Value>,
    );
}

// ─── Pipeline runner ─────────────────────────────────────────────────────────

pub fn run(
    input: &str,
    output: &str,
    scheme: &dyn QuantScheme,
) -> Result<(), String> {
    // ── 0. Local directory or HF repo? ────────────────────────────────
    let repo = if Path::new(input).is_dir() {
        HfRepo::from_local(Path::new(input).to_path_buf())
    } else {
        HfRepo::from_hf(input)?
    };
    let output_dir = Path::new(output);
    let experts_dir = output_dir.join("packed_experts");
    fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    fs::create_dir_all(&experts_dir).map_err(|e| e.to_string())?;

    let _hd = scheme.hidden_dim();

    // ── 1. Load weight map from index.json ─────────────────────────
    repo.ensure("config.json")?;
    let index_path = repo.ensure("model.safetensors.index.json")?;
    let weight_map = {
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
    };
    eprintln!("  Total tensors: {}", weight_map.len());

    // ── 2. Classify as we process each shard (need shapes from headers) ──
    let mut classified: HashMap<String, WeightClass> = HashMap::new();

    // ── 3. Group by shard and classify on the fly ───────────────────
    let shard_order: Vec<String> = {
        let mut set: HashSet<String> = HashSet::new();
        for (_hf_name, shard) in weight_map.iter() {
            set.insert(shard.clone());
        }
        let mut v: Vec<String> = set.into_iter().collect();
        v.sort();
        v
    };

    // Gather expert tensor info for phase 2
    let mut expert_by_layer: BTreeMap<usize, (String, String)> = BTreeMap::new();
    // (gate_up_key, down_key) — HF convention, replaced by scheme.process_experts()

    // ── 4. Build manifest config ────────────────────────────────────
    let mut manifest: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    manifest.insert("model".into(), serde_json::Value::String(input.to_owned()));
    manifest.insert("num_tensors".into(), serde_json::Value::from(0));

    let mut cfg = serde_json::Map::new();
    scheme.write_manifest_config(&mut cfg);
    manifest.insert("config".into(), serde_json::Value::Object(cfg));

    // ── 5. Shard pass: non-expert tensors ────────────────────────────
    eprintln!("\n============================================================");
    eprintln!("Quantizing non-expert weights...");
    eprintln!("============================================================");

    let bin_path = output_dir.join("model_weights.bin");
    let mut out_f = fs::File::create(&bin_path)
        .map_err(|e| format!("cannot create {}: {}", bin_path.display(), e))?;
    let mut offset: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut tensor_count: u64 = 0;
    let mut tensors_map = serde_json::Map::new();
    let mut quant_summary: HashMap<String, usize> = HashMap::new();

    let t0 = std::time::Instant::now();

    for shard_name in &shard_order {
        check_interrupt()?;

        let shard_path = repo.ensure(shard_name)?;
        eprintln!("  Caching header: {}", shard_name);
        let header = parse_safetensors(&shard_path)?;

        // Find tensors in this shard and classify them now (we have shapes from header)
        for (hf_name, shard) in weight_map.iter() {
            if *shard != *shard_name { continue; }

            let meta = match header.tensors.get(hf_name) {
                Some(m) => m,
                None => continue,
            };
            let shape = meta.shape.clone();

            let cls = scheme.classify(hf_name, &shape);
            let mlx_name = cls.name.clone();

            // Store classification for expert pass
            classified.insert(hf_name.clone(), cls.clone());

            match cls.kind {
                WeightKind::Skip => continue,
                WeightKind::Expert(layer) => {
                    let entry = expert_by_layer.entry(layer)
                        .or_insert_with(|| (String::new(), String::new()));
                    if mlx_name.contains("gate_up_proj") || mlx_name.contains("gate_proj") {
                        entry.0 = hf_name.clone();
                    } else if mlx_name.contains("down_proj") {
                        entry.1 = hf_name.clone();
                    }
                    continue;
                }
                WeightKind::Normal => {}
            }

            // ── Non-expert processing ──────────────────────────
            let q = cls.quant;
            let q_str = q.as_str().to_string();
            *quant_summary.entry(q_str.clone()).or_insert(0) += 1;

            let raw_data = read_tensor_bytes(&shard_path, &header, hf_name)?;
            let mut f32_vals = bytes_to_f32(&raw_data, &meta.dtype);
            let mut out_shape = shape.clone();

            // Sanitize
            scheme.sanitize(&mlx_name, &mut f32_vals, &mut out_shape);

            // Pad inner dim for INT4
            let out_dim = out_shape[0];
            let in_dim = if out_shape.len() >= 2 { out_shape[1] } else { 0 };
            let (padded_in, f32_padded) = if q == DType::Int4 {
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

            // Align
            if offset % ALIGN != 0 {
                let pad = ALIGN - (offset % ALIGN);
                out_f.write_all(&vec![0u8; pad as usize]).map_err(|e| e.to_string())?;
                offset += pad;
            }

            // Base name for the entry (strip .weight suffix if present)
            let base = if mlx_name.ends_with(".weight") {
                mlx_name[..mlx_name.len() - 7].to_string()
            } else {
                mlx_name.clone()
            };

            // Encode and write
            let encoded = q.encode(&f32_padded, out_dim, padded_in);
            for et in &encoded {
                let tname = if et.suffix.is_empty() {
                    base.clone()
                } else {
                    format!("{}{}", base, et.suffix)
                };
                let dlen = et.data.len() as u64;
                out_f.write_all(&et.data).map_err(|e| e.to_string())?;

                let mut entry = serde_json::Map::new();
                entry.insert("offset".into(), serde_json::Value::from(offset));
                entry.insert("size".into(), serde_json::Value::from(dlen));
                entry.insert("shape".into(), serde_json::Value::Array(
                    et.shape.iter().map(|&n| serde_json::Value::from(n as u64)).collect(),
                ));
                entry.insert("dtype".into(), serde_json::Value::String(et.dtype.into()));
                tensors_map.insert(tname, serde_json::Value::Object(entry));

                offset += dlen;
                total_bytes += dlen;
                tensor_count += 1;
            }
        }

        // Done with this shard
        if repo.is_hf() {
            repo.remove(shard_name);
            eprintln!("  Deleted {}", shard_name);
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

    // ── 6. Expert pass ────────────────────────────────────────────────
    eprintln!("\n============================================================");
    eprintln!("Quantizing expert weights...");
    eprintln!("============================================================");

    let t1 = std::time::Instant::now();
    let classified_vec: Vec<(String, WeightClass)> = classified.into_iter().collect();
    let expert_layers_done = scheme.process_experts(
        &repo,
        &weight_map,
        &classified_vec,
        output_dir,
    )?;

    let t2 = t1.elapsed();
    eprintln!(
        "\n  {} expert layers in {:.1}s",
        expert_layers_done,
        t2.as_secs_f64()
    );

    // ── 7. Write config.json ──────────────────────────────────────────
    // Config is already in the model directory from ensure("config.json")
    let src_config = repo.path().join("config.json");
    let dst_config = output_dir.join("config.json");
    if src_config.exists() && src_config != dst_config {
        fs::copy(&src_config, &dst_config).map_err(|e| e.to_string())?;
    }

    // ── 8. Summary ────────────────────────────────────────────────────
    let total_time = t0.elapsed();
    let bin_size = fs::metadata(&bin_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("\n============================================================");
    eprintln!("Done!");
    eprintln!("  model_weights.bin : {:.2} GB", bin_size as f64 / 1e9);
    eprintln!("  model_weights.json: {}", json_path.display());
    eprintln!("  packed_experts    : {} layers", expert_layers_done);
    eprintln!("  Total time        : {:.1}s", total_time.as_secs_f64());
    eprintln!("============================================================");

    Ok(())
}

fn check_interrupt() -> Result<(), String> {
    #[cfg(feature = "python-bindings")]
    pyo3::Python::with_gil(|py| py.check_signals())
        .map_err(|e| format!("interrupted: {}", e))?;
    Ok(())
}

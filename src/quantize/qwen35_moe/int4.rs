// Int4Scheme: all 2D weight matrices → INT4.  Qwen3.5/Qwen3.6-specific.

use std::collections::HashMap;
use std::path::Path;

use crate::dtype::DType;
use crate::hf_util::HfRepo;
use crate::quantize::{QuantScheme, WeightClass, WeightKind};

pub use crate::qwen35_moe_common::{
    QwenVersion, ModelParams, load_config,
};

use crate::qwen35_moe_common::{
    NameMap, NAME_MAPPING_JSON, load_name_mapping,
    split_on_last_dot,
    extract_layer, is_expert_tensor, is_vision_tensor,
    is_norm_key, moveaxis_2_to_1,
    process_experts_common, write_manifest_config_common,
};

// ─── INT4 dtype selection ────────────────────────────────────────────────────

fn int4_quant(mlx_name: &str, shape: &[usize]) -> DType {
    let (_, kind) = split_on_last_dot(mlx_name);
    match kind {
        "A_log" => { debug_assert!(shape.len() <= 1); DType::Fp32 }
        "scales" | "biases" | "bias" | "dt_bias" => { debug_assert!(shape.len() <= 2); DType::Bf16 }
        "weight" => {
            if shape.len() != 2 { DType::Bf16 }
            else { DType::Int4 }
        }
        _ => DType::Bf16,
    }
}

// ─── Int4Scheme ──────────────────────────────────────────────────────────────

pub struct Int4Scheme {
    version: QwenVersion,
    params: ModelParams,
    name_map: NameMap,
    eff_num_layers: usize,
    eff_num_experts: usize,
}

impl Int4Scheme {
    pub fn new(model_path: &Path, version: QwenVersion) -> Result<Self, String> {
        let params = load_config(model_path)?;
        let num_layers = params.num_layers;
        let num_experts = params.num_experts;

        let eff_num_layers = num_layers;
        let eff_num_experts = num_experts;

        let name_map = load_name_mapping(NAME_MAPPING_JSON, num_layers)?;

        eprintln!("Model config:");
        eprintln!("  hidden_dim={}, vocab_size={}", params.hidden_dim, params.vocab_size);
        eprintln!("  num_layers={} (main={}, mtp={})",
            num_layers, num_layers - params.mtp_num_layers, params.mtp_num_layers);
        eprintln!("  num_experts={}, experts_per_tok={}", num_experts, params.num_experts_per_tok);
        eprintln!("  moe_intermediate={}, shared_intermediate={}",
            params.moe_intermediate, params.shared_intermediate);
        eprintln!("  Name mapping entries: {}", name_map.len());

        Ok(Self { version, params, name_map, eff_num_layers, eff_num_experts })
    }
}

impl QuantScheme for Int4Scheme {
    fn hidden_dim(&self) -> usize { self.params.hidden_dim }
    fn num_layers(&self) -> usize { self.eff_num_layers }
    fn num_experts(&self) -> usize { self.eff_num_experts }

    fn classify(&self, hf_name: &str, shape: &[usize]) -> WeightClass {
        let mlx_name = self.name_map.get(hf_name)
            .cloned()
            .unwrap_or_else(|| hf_name.to_string());

        if is_vision_tensor(&mlx_name) {
            return WeightClass { name: mlx_name, quant: DType::Bf16, kind: WeightKind::Skip };
        }

        if is_expert_tensor(&mlx_name) {
            if let Some(layer) = extract_layer(hf_name) {
                let q = int4_quant(&mlx_name, shape);
                return WeightClass { name: mlx_name, quant: q, kind: WeightKind::Expert(layer) };
            }
        }

        let q = int4_quant(&mlx_name, shape);
        WeightClass { name: mlx_name, quant: q, kind: WeightKind::Normal }
    }

    fn sanitize(&self, mlx_name: &str, values: &mut [f32], shape: &mut Vec<usize>) {
        if self.version.is_qwen36() && is_norm_key(mlx_name) {
            for v in &mut *values { *v += 1.0; }
        }
        if mlx_name.contains("conv1d.weight") && shape.len() == 3 {
            moveaxis_2_to_1(values, shape);
        }
    }

    fn process_experts(
        &self,
        repo: &HfRepo,
        weight_map: &HashMap<String, String>,
        classified: &[(String, WeightClass)],
        output_dir: &Path,
    ) -> Result<usize, String> {
        process_experts_common(
            self.params.moe_intermediate,
            self.params.hidden_dim,
            self.eff_num_experts,
            classified,
            repo,
            weight_map,
            output_dir,
        )
    }

    fn write_manifest_config(
        &self,
        cfg: &mut serde_json::Map<String, serde_json::Value>,
    ) {
        write_manifest_config_common(&self.params, self.eff_num_layers, self.eff_num_experts, cfg)
    }
}

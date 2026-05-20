use serde::Deserialize;
use std::path::Path;

const GROUP_SIZE: usize = 64;

/// Raw HuggingFace config.json (with optional text_config for multimodal models).
#[derive(Debug, Clone, Deserialize)]
struct HfConfig {
    text_config: Option<HfTextConfig>,
}

#[derive(Debug, Clone, Deserialize)]
struct HfTextConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    num_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    shared_expert_intermediate_size: usize,
    linear_num_value_heads: usize,
    linear_num_key_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_conv_kernel_dim: Option<usize>,
    full_attention_interval: Option<usize>,
    #[serde(default)]
    rms_norm_eps: Option<f32>,
    rope_parameters: Option<HfRopeParams>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
struct HfRopeParams {
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    partial_rotary_factor: f32,
}

fn default_rope_theta() -> f64 { 10000.0 }
fn default_partial_rotary() -> f32 { 0.25 }

/// Runtime model configuration — derived from HF config.json.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_attn_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate: usize,
    pub shared_intermediate: usize,
    pub linear_num_v_heads: usize,
    pub linear_num_k_heads: usize,
    pub rotary_dim: usize,
    pub rope_theta: f64,
    pub linear_total_key: usize,
    pub linear_total_value: usize,
    pub linear_conv_dim: usize,
    pub num_full_attn_layers: usize,
    pub num_linear_layers: usize,
    pub expert_size_4bit: usize,
    pub expert_size_2bit: usize,
    pub expert_layout_4bit: ExpertLayout,
    pub expert_layout_2bit: ExpertLayout,
    pub group_size: usize,
    pub bits: usize,
    pub model_path: String,
}

/// Expert packed binary layout — offsets and sizes for each component.
#[derive(Debug, Clone)]
pub struct ExpertLayout {
    pub gate_w_off: usize,
    pub gate_s_off: usize,
    pub gate_b_off: usize,
    pub up_w_off: usize,
    pub up_s_off: usize,
    pub up_b_off: usize,
    pub down_w_off: usize,
    pub down_s_off: usize,
    pub down_b_off: usize,
    pub gate_w_size: usize,
    pub gate_s_size: usize,
    pub gate_b_size: usize,
    pub up_w_size: usize,
    pub up_s_size: usize,
    pub up_b_size: usize,
    pub down_w_size: usize,
    pub down_s_size: usize,
    pub down_b_size: usize,
}

fn compute_expert_layout(hidden_dim: usize, inter_dim: usize) -> (ExpertLayout, ExpertLayout, usize, usize) {
    let gs = GROUP_SIZE;

    let gate_w = inter_dim * hidden_dim / 2;
    let gate_sb = inter_dim * (hidden_dim / gs) * 2;
    let up_w = gate_w;
    let up_sb = gate_sb;
    let down_w = hidden_dim * inter_dim / 2;
    let down_sb = hidden_dim * (inter_dim / gs) * 2;

    let gate_w_off = 0;
    let gate_s_off = gate_w;
    let gate_b_off = gate_w + gate_sb;
    let up_w_off = gate_w + 2 * gate_sb;
    let up_s_off = up_w_off + up_w;
    let up_b_off = up_s_off + up_sb;
    let down_w_off = up_b_off + up_sb;
    let down_s_off = down_w_off + down_w;
    let down_b_off = down_s_off + down_sb;
    let expert_size_4bit = down_b_off + down_sb;

    let gate_w2 = inter_dim * hidden_dim / 4;
    let up_w2 = gate_w2;
    let down_w2 = hidden_dim * inter_dim / 4;

    let gate_w_off_2 = 0;
    let gate_s_off_2 = gate_w2;
    let gate_b_off_2 = gate_w2 + gate_sb;
    let up_w_off_2 = gate_w2 + 2 * gate_sb;
    let up_s_off_2 = up_w_off_2 + up_w2;
    let up_b_off_2 = up_s_off_2 + up_sb;
    let down_w_off_2 = up_b_off_2 + up_sb;
    let down_s_off_2 = down_w_off_2 + down_w2;
    let down_b_off_2 = down_s_off_2 + down_sb;
    let expert_size_2bit = down_b_off_2 + down_sb;

    let layout_4bit = ExpertLayout {
        gate_w_off, gate_s_off, gate_b_off, up_w_off, up_s_off, up_b_off,
        down_w_off, down_s_off, down_b_off,
        gate_w_size: gate_w, gate_s_size: gate_sb, gate_b_size: gate_sb,
        up_w_size: up_w, up_s_size: up_sb, up_b_size: up_sb,
        down_w_size: down_w, down_s_size: down_sb, down_b_size: down_sb,
    };
    let layout_2bit = ExpertLayout {
        gate_w_off: gate_w_off_2, gate_s_off: gate_s_off_2, gate_b_off: gate_b_off_2,
        up_w_off: up_w_off_2, up_s_off: up_s_off_2, up_b_off: up_b_off_2,
        down_w_off: down_w_off_2, down_s_off: down_s_off_2, down_b_off: down_b_off_2,
        gate_w_size: gate_w2, gate_s_size: gate_sb, gate_b_size: gate_sb,
        up_w_size: up_w2, up_s_size: up_sb, up_b_size: up_sb,
        down_w_size: down_w2, down_s_size: down_sb, down_b_size: down_sb,
    };

    (layout_4bit, layout_2bit, expert_size_4bit, expert_size_2bit)
}

/// Load model configuration from an HF config.json file.
pub fn load_model_config(model_path: &Path) -> anyhow::Result<ModelConfig> {
    let config_path = model_path.join("config.json");
    let content = std::fs::read_to_string(&config_path)?;
    let hf: HfConfig = serde_json::from_str(&content)?;

    let tc = hf.text_config.as_ref().unwrap_or_else(|| {
        // If no text_config key, try parsing the whole file as text config
        panic!("config.json missing text_config; multimodal wrapper expected")
    });

    let full_attn_interval = tc.full_attention_interval.unwrap_or(4);
    let rp = tc.rope_parameters.as_ref();
    let rope_theta = rp.map_or(10000.0, |r| r.rope_theta);
    let partial_rotary = rp.map_or(0.25, |r| r.partial_rotary_factor);
    let rotary_dim = (tc.head_dim as f32 * partial_rotary) as usize;

    let linear_key_dim = tc.linear_key_head_dim;
    let linear_value_dim = tc.linear_value_head_dim;
    let linear_total_key = tc.linear_num_key_heads * linear_key_dim;
    let linear_total_value = tc.linear_num_value_heads * linear_value_dim;
    let linear_conv_dim = linear_total_key * 2 + linear_total_value;

    let num_full_attn_layers = tc.num_hidden_layers / full_attn_interval;
    let num_linear_layers = tc.num_hidden_layers - num_full_attn_layers;

    let (layout_4bit, layout_2bit, expert_size_4bit, expert_size_2bit) =
        compute_expert_layout(tc.hidden_size, tc.moe_intermediate_size);

    Ok(ModelConfig {
        hidden_dim: tc.hidden_size,
        num_layers: tc.num_hidden_layers,
        num_attn_heads: tc.num_attention_heads,
        num_kv_heads: tc.num_key_value_heads,
        head_dim: tc.head_dim,
        vocab_size: tc.vocab_size,
        num_experts: tc.num_experts,
        num_experts_per_tok: tc.num_experts_per_tok,
        moe_intermediate: tc.moe_intermediate_size,
        shared_intermediate: tc.shared_expert_intermediate_size,
        linear_num_v_heads: tc.linear_num_value_heads,
        linear_num_k_heads: tc.linear_num_key_heads,
        rotary_dim,
        rope_theta,
        linear_total_key,
        linear_total_value,
        linear_conv_dim,
        num_full_attn_layers,
        num_linear_layers,
        expert_size_4bit,
        expert_size_2bit,
        expert_layout_4bit: layout_4bit,
        expert_layout_2bit: layout_2bit,
        group_size: GROUP_SIZE,
        bits: 4,
        model_path: model_path.to_string_lossy().to_string(),
    })
}

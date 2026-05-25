/// Compile-time model-dimenssion trait for Qwen3.6 MoE inference.
///
/// Implemented by marker types in this module so engine code can be generic
/// over `C: ModelConfig` instead of duplicating files per variant.

#[allow(non_snake_case)]
pub trait ModelConfig: 'static {
    const HIDDEN_DIM: usize;
    const NUM_LAYERS: usize;
    const NUM_ATTN_HEADS: usize;
    const NUM_KV_HEADS: usize;
    const HEAD_DIM: usize;
    const VOCAB_SIZE: usize;
    const NUM_EXPERTS: usize;
    const NUM_EXPERTS_PER_TOK: usize;
    const MOE_INTERMEDIATE: usize;
    const SHARED_INTERMEDIATE: usize;

    const LINEAR_NUM_V_HEADS: usize;
    const LINEAR_NUM_K_HEADS: usize;
    const LINEAR_KEY_DIM: usize;
    const LINEAR_VALUE_DIM: usize;
    const LINEAR_TOTAL_KEY: usize;
    const LINEAR_TOTAL_VALUE: usize;
    const LINEAR_CONV_DIM: usize;

    const ROPE_THETA: f64;
    const ROTARY_DIM: usize;

    const NUM_FULL_ATTN_LAYERS: usize;
    const NUM_LINEAR_LAYERS: usize;

    const KV_DIM: usize;

    // Expert 4-bit packed layout
    const EXPERT_SIZE_4BIT: usize;
    const GATE_W_OFF: usize;
    const GATE_S_OFF: usize;
    const GATE_B_OFF: usize;
    const UP_W_OFF: usize;
    const UP_S_OFF: usize;
    const UP_B_OFF: usize;
    const DOWN_W_OFF: usize;
    const DOWN_S_OFF: usize;
    const DOWN_B_OFF: usize;
    const GATE_W_SIZE: usize;
    const GATE_S_SIZE: usize;
    const GATE_B_SIZE: usize;
    const UP_W_SIZE: usize;
    const UP_S_SIZE: usize;
    const UP_B_SIZE: usize;
    const DOWN_W_SIZE: usize;
    const DOWN_S_SIZE: usize;
    const DOWN_B_SIZE: usize;
    const EXPECTED_ARCHITECTURE: &'static str;

    fn validate_config(c: &crate::model::config::ModelConfig) -> Result<(), String> {
        let mut errs = Vec::new();
        let get = |k| c.get_usize(k).unwrap_or(0);

        let archs: Vec<&str> = c.resolve("architectures")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        if !archs.iter().any(|a| *a == Self::EXPECTED_ARCHITECTURE) {
            errs.push(format!("architecture mismatch: found={:?}, expected=\"{}\"",
                archs, Self::EXPECTED_ARCHITECTURE));
        }

        let lnum_k = get("linear_num_key_heads");
        let lkey_dim = get("linear_key_head_dim");
        let lnum_v = get("linear_num_value_heads");
        let lval_dim = get("linear_value_head_dim");

        if get("hidden_size") != Self::HIDDEN_DIM { errs.push(format!("hidden_size: config={}, const={}", get("hidden_size"), Self::HIDDEN_DIM)); }
        if get("num_hidden_layers") != Self::NUM_LAYERS { errs.push(format!("num_hidden_layers: config={}, const={}", get("num_hidden_layers"), Self::NUM_LAYERS)); }
        if get("num_experts") != Self::NUM_EXPERTS { errs.push(format!("num_experts: config={}, const={}", get("num_experts"), Self::NUM_EXPERTS)); }
        if get("num_experts_per_tok") != Self::NUM_EXPERTS_PER_TOK { errs.push(format!("num_experts_per_tok: config={}, const={}", get("num_experts_per_tok"), Self::NUM_EXPERTS_PER_TOK)); }
        if get("moe_intermediate_size") != Self::MOE_INTERMEDIATE { errs.push(format!("moe_intermediate_size: config={}, const={}", get("moe_intermediate_size"), Self::MOE_INTERMEDIATE)); }
        if get("shared_expert_intermediate_size") != Self::SHARED_INTERMEDIATE { errs.push(format!("shared_expert_intermediate_size: config={}, const={}", get("shared_expert_intermediate_size"), Self::SHARED_INTERMEDIATE)); }
        if get("num_attention_heads") != Self::NUM_ATTN_HEADS { errs.push(format!("num_attention_heads: config={}, const={}", get("num_attention_heads"), Self::NUM_ATTN_HEADS)); }
        if get("num_key_value_heads") != Self::NUM_KV_HEADS { errs.push(format!("num_key_value_heads: config={}, const={}", get("num_key_value_heads"), Self::NUM_KV_HEADS)); }
        if get("head_dim") != Self::HEAD_DIM { errs.push(format!("head_dim: config={}, const={}", get("head_dim"), Self::HEAD_DIM)); }
        if get("vocab_size") != Self::VOCAB_SIZE { errs.push(format!("vocab_size: config={}, const={}", get("vocab_size"), Self::VOCAB_SIZE)); }
        if lnum_v != Self::LINEAR_NUM_V_HEADS { errs.push(format!("linear_num_value_heads: config={}, const={}", lnum_v, Self::LINEAR_NUM_V_HEADS)); }
        if lnum_k != Self::LINEAR_NUM_K_HEADS { errs.push(format!("linear_num_key_heads: config={}, const={}", lnum_k, Self::LINEAR_NUM_K_HEADS)); }
        if lnum_k * lkey_dim != Self::LINEAR_TOTAL_KEY { errs.push(format!("linear_total_key: config={}, const={}", lnum_k * lkey_dim, Self::LINEAR_TOTAL_KEY)); }
        if lnum_v * lval_dim != Self::LINEAR_TOTAL_VALUE { errs.push(format!("linear_total_value: config={}, const={}", lnum_v * lval_dim, Self::LINEAR_TOTAL_VALUE)); }
        if errs.is_empty() { Ok(()) } else { Err(errs.join("; ")) }
    }
}

// ─── Expert layout helper ─────────────────────────────────────────────

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
    pub expert_size_4bit: usize,
}

pub const fn expert_layout(hd: usize, mi: usize, gs: usize) -> ExpertLayout {
    let gate_w = mi * hd / 2;
    let gate_sb = mi * (hd / gs) * 2;
    let up_w = mi * hd / 2;
    let up_sb = mi * (hd / gs) * 2;
    let down_w = hd * mi / 2;
    let down_sb = hd * (mi / gs) * 2;
    ExpertLayout {
        gate_w_off: 0,
        gate_s_off: gate_w,
        gate_b_off: gate_w + gate_sb,
        up_w_off: gate_w + 2 * gate_sb,
        up_s_off: gate_w + 2 * gate_sb + up_w,
        up_b_off: gate_w + 2 * gate_sb + up_w + up_sb,
        down_w_off: gate_w + 2 * gate_sb + up_w + 2 * up_sb,
        down_s_off: gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w,
        down_b_off: gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w + down_sb,
        gate_w_size: gate_w, gate_s_size: gate_sb, gate_b_size: gate_sb,
        up_w_size: up_w, up_s_size: up_sb, up_b_size: up_sb,
        down_w_size: down_w, down_s_size: down_sb, down_b_size: down_sb,
        expert_size_4bit: gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w + 2 * down_sb,
    }
}

const L: ExpertLayout = expert_layout(2048, 512, 64);

// ─── Marker types ─────────────────────────────────────────────────────

/// Full model: 40 layers, 256 experts, 8 experts-per-tok.
pub struct FullModel;

impl ModelConfig for FullModel {
    const HIDDEN_DIM: usize = 2048;
    const NUM_LAYERS: usize = 40;
    const NUM_ATTN_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const VOCAB_SIZE: usize = 248320;
    const NUM_EXPERTS: usize = 256;
    const NUM_EXPERTS_PER_TOK: usize = 8;
    const MOE_INTERMEDIATE: usize = 512;
    const SHARED_INTERMEDIATE: usize = 512;
    const LINEAR_NUM_V_HEADS: usize = 32;
    const LINEAR_NUM_K_HEADS: usize = 16;
    const LINEAR_KEY_DIM: usize = 128;
    const LINEAR_VALUE_DIM: usize = 128;
    const LINEAR_TOTAL_KEY: usize = 2048;
    const LINEAR_TOTAL_VALUE: usize = 4096;
    const LINEAR_CONV_DIM: usize = 8192;
    const ROPE_THETA: f64 = 10_000_000.0;
    const ROTARY_DIM: usize = 64;
    const NUM_FULL_ATTN_LAYERS: usize = 10;
    const NUM_LINEAR_LAYERS: usize = 30;
    const KV_DIM: usize = 512;

    const EXPERT_SIZE_4BIT: usize = L.expert_size_4bit;
    const GATE_W_OFF: usize = L.gate_w_off;
    const GATE_S_OFF: usize = L.gate_s_off;
    const GATE_B_OFF: usize = L.gate_b_off;
    const UP_W_OFF: usize = L.up_w_off;
    const UP_S_OFF: usize = L.up_s_off;
    const UP_B_OFF: usize = L.up_b_off;
    const DOWN_W_OFF: usize = L.down_w_off;
    const DOWN_S_OFF: usize = L.down_s_off;
    const DOWN_B_OFF: usize = L.down_b_off;
    const GATE_W_SIZE: usize = L.gate_w_size;
    const GATE_S_SIZE: usize = L.gate_s_size;
    const GATE_B_SIZE: usize = L.gate_b_size;
    const UP_W_SIZE: usize = L.up_w_size;
    const UP_S_SIZE: usize = L.up_s_size;
    const UP_B_SIZE: usize = L.up_b_size;
    const DOWN_W_SIZE: usize = L.down_w_size;
    const DOWN_S_SIZE: usize = L.down_s_size;
    const DOWN_B_SIZE: usize = L.down_b_size;
    const EXPECTED_ARCHITECTURE: &'static str = "Qwen3_5MoeForConditionalGeneration";
}

/// Stripped model: 4 layers, 4 experts, 4 experts-per-tok (test model).
pub struct StrippedModel;

impl ModelConfig for StrippedModel {
    const HIDDEN_DIM: usize = 2048;
    const NUM_LAYERS: usize = 4;
    const NUM_ATTN_HEADS: usize = 16;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 256;
    const VOCAB_SIZE: usize = 248320;
    const NUM_EXPERTS: usize = 4;
    const NUM_EXPERTS_PER_TOK: usize = 4;
    const MOE_INTERMEDIATE: usize = 512;
    const SHARED_INTERMEDIATE: usize = 512;
    const LINEAR_NUM_V_HEADS: usize = 32;
    const LINEAR_NUM_K_HEADS: usize = 16;
    const LINEAR_KEY_DIM: usize = 128;
    const LINEAR_VALUE_DIM: usize = 128;
    const LINEAR_TOTAL_KEY: usize = 2048;
    const LINEAR_TOTAL_VALUE: usize = 4096;
    const LINEAR_CONV_DIM: usize = 8192;
    const ROPE_THETA: f64 = 10_000_000.0;
    const ROTARY_DIM: usize = 64;
    const NUM_FULL_ATTN_LAYERS: usize = 1;
    const NUM_LINEAR_LAYERS: usize = 3;
    const KV_DIM: usize = 512;

    const EXPERT_SIZE_4BIT: usize = L.expert_size_4bit;
    const GATE_W_OFF: usize = L.gate_w_off;
    const GATE_S_OFF: usize = L.gate_s_off;
    const GATE_B_OFF: usize = L.gate_b_off;
    const UP_W_OFF: usize = L.up_w_off;
    const UP_S_OFF: usize = L.up_s_off;
    const UP_B_OFF: usize = L.up_b_off;
    const DOWN_W_OFF: usize = L.down_w_off;
    const DOWN_S_OFF: usize = L.down_s_off;
    const DOWN_B_OFF: usize = L.down_b_off;
    const GATE_W_SIZE: usize = L.gate_w_size;
    const GATE_S_SIZE: usize = L.gate_s_size;
    const GATE_B_SIZE: usize = L.gate_b_size;
    const UP_W_SIZE: usize = L.up_w_size;
    const UP_S_SIZE: usize = L.up_s_size;
    const UP_B_SIZE: usize = L.up_b_size;
    const DOWN_W_SIZE: usize = L.down_w_size;
    const DOWN_S_SIZE: usize = L.down_s_size;
    const DOWN_B_SIZE: usize = L.down_b_size;
    const EXPECTED_ARCHITECTURE: &'static str = "Qwen3_5MoeForConditionalGeneration_Stripped";
}

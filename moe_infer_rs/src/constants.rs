/// Compile-time constants.

/// Number of output rows per threadgroup in the v3/v5 shaders.
pub const ROWS_PER_TG: u32 = 8;

/// Threadgroup size for optimized kernels.
pub const TG_SIZE: u32 = 256;

// ─── Shared architecture constants ──────────────────────────────────────

/// Maximum sequence length (controls KV cache allocation).
pub const MAX_SEQ: usize = 4096;

/// Epsilon for RMS normalization.
pub const RMS_NORM_EPS: f32 = 1e-6;

/// Interval at which full (self) attention layers appear.
pub const FULL_ATTN_INTERVAL: usize = 4;

/// Group size for 4-bit quantization (64 weights → 1 scale + 1 bias).
pub const GROUP_SIZE: usize = 64;

/// Convolution kernel size for the linear attention conv1d step.
pub const CONV_KERNEL_SIZE: usize = 4;

// ─── Qwen3.6-35B-A3B-4bit model constants ──────────────────────────────
/// Mirrors the #define constants in moe_infer_c/bench.m.
/// Import as `use crate::constants::qwen35_35b::*;` to use bare names.
pub mod qwen35_35b {
    use super::{GROUP_SIZE, FULL_ATTN_INTERVAL};

    /// Validate that a loaded model config matches these compile-time constants.
    /// Returns Ok(()) on match, Err(String) describing mismatches.
    pub fn validate_config(hidden_dim: usize, num_layers: usize, num_experts: usize,
                           num_experts_per_tok: usize, moe_intermediate: usize,
                           shared_intermediate: usize, num_attn_heads: usize,
                           num_kv_heads: usize, head_dim: usize, vocab_size: usize,
                           linear_num_v_heads: usize, linear_num_k_heads: usize,
                           linear_total_key: usize, linear_total_value: usize,
                           ) -> Result<(), String> {
        let mut errs = Vec::new();
        if hidden_dim != HIDDEN_DIM { errs.push(format!("hidden_dim: config={}, const={}", hidden_dim, HIDDEN_DIM)); }
        if num_layers != NUM_LAYERS { errs.push(format!("num_layers: config={}, const={}", num_layers, NUM_LAYERS)); }
        if num_experts != NUM_EXPERTS { errs.push(format!("num_experts: config={}, const={}", num_experts, NUM_EXPERTS)); }
        if num_experts_per_tok != NUM_EXPERTS_PER_TOK { errs.push(format!("num_experts_per_tok: config={}, const={}", num_experts_per_tok, NUM_EXPERTS_PER_TOK)); }
        if moe_intermediate != MOE_INTERMEDIATE { errs.push(format!("moe_intermediate: config={}, const={}", moe_intermediate, MOE_INTERMEDIATE)); }
        if shared_intermediate != SHARED_INTERMEDIATE { errs.push(format!("shared_intermediate: config={}, const={}", shared_intermediate, SHARED_INTERMEDIATE)); }
        if num_attn_heads != NUM_ATTN_HEADS { errs.push(format!("num_attn_heads: config={}, const={}", num_attn_heads, NUM_ATTN_HEADS)); }
        if num_kv_heads != NUM_KV_HEADS { errs.push(format!("num_kv_heads: config={}, const={}", num_kv_heads, NUM_KV_HEADS)); }
        if head_dim != HEAD_DIM { errs.push(format!("head_dim: config={}, const={}", head_dim, HEAD_DIM)); }
        if vocab_size != VOCAB_SIZE { errs.push(format!("vocab_size: config={}, const={}", vocab_size, VOCAB_SIZE)); }
        if linear_num_v_heads != LINEAR_NUM_V_HEADS { errs.push(format!("linear_num_v_heads: config={}, const={}", linear_num_v_heads, LINEAR_NUM_V_HEADS)); }
        if linear_num_k_heads != LINEAR_NUM_K_HEADS { errs.push(format!("linear_num_k_heads: config={}, const={}", linear_num_k_heads, LINEAR_NUM_K_HEADS)); }
        if linear_total_key != LINEAR_TOTAL_KEY { errs.push(format!("linear_total_key: config={}, const={}", linear_total_key, LINEAR_TOTAL_KEY)); }
        if linear_total_value != LINEAR_TOTAL_VALUE { errs.push(format!("linear_total_value: config={}, const={}", linear_total_value, LINEAR_TOTAL_VALUE)); }
        if errs.is_empty() { Ok(()) } else { Err(errs.join("; ")) }
    }

    pub const HIDDEN_DIM: usize            = 2048;
    pub const NUM_LAYERS: usize            = 40;
    pub const NUM_ATTN_HEADS: usize        = 16;
    pub const NUM_KV_HEADS: usize          = 2;
    pub const HEAD_DIM: usize              = 256;
    pub const VOCAB_SIZE: usize            = 248320;
    pub const NUM_EXPERTS: usize           = 256;
    pub const NUM_EXPERTS_PER_TOK: usize   = 8;
    pub const MOE_INTERMEDIATE: usize      = 512;
    pub const SHARED_INTERMEDIATE: usize   = 512;

    pub const LINEAR_NUM_V_HEADS: usize    = 32;
    pub const LINEAR_NUM_K_HEADS: usize    = 16;
    pub const LINEAR_KEY_DIM: usize        = 128;
    pub const LINEAR_VALUE_DIM: usize      = 128;
    pub const LINEAR_TOTAL_KEY: usize      = LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM;   // 2048
    pub const LINEAR_TOTAL_VALUE: usize    = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM; // 4096
    pub const LINEAR_CONV_DIM: usize       = LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE; // 8192

    pub const ROPE_THETA: f64              = 10_000_000.0;
    pub const PARTIAL_ROTARY: f32          = 0.25;
    pub const ROTARY_DIM: usize            = (HEAD_DIM as f32 * PARTIAL_ROTARY) as usize; // 64

    pub const NUM_FULL_ATTN_LAYERS: usize  = NUM_LAYERS / FULL_ATTN_INTERVAL;  // 10
    pub const NUM_LINEAR_LAYERS: usize     = NUM_LAYERS - NUM_FULL_ATTN_LAYERS; // 30

    // ─── Expert 4-bit packed layout ─────────────────────────────────────

    const GS: usize = GROUP_SIZE;
    const HD: usize = HIDDEN_DIM;
    const MI: usize = MOE_INTERMEDIATE;

    const GATE_W: usize   = MI * HD / 2;              // 524288
    const GATE_SB: usize  = MI * (HD / GS) * 2;       // 32768
    const UP_W: usize     = MI * HD / 2;              // 524288
    const UP_SB: usize    = MI * (HD / GS) * 2;       // 32768
    const DOWN_W: usize   = HD * MI / 2;              // 524288
    const DOWN_SB: usize  = HD * (MI / GS) * 2;       // 32768

    pub const GATE_W_OFF: usize   = 0;                                    // 0
    pub const GATE_S_OFF: usize   = GATE_W;                               // 524288
    pub const GATE_B_OFF: usize   = GATE_W + GATE_SB;                     // 557056
    pub const UP_W_OFF: usize     = GATE_W + 2 * GATE_SB;                 // 589824
    pub const UP_S_OFF: usize     = UP_W_OFF + UP_W;                      // 1114112
    pub const UP_B_OFF: usize     = UP_S_OFF + UP_SB;                     // 1146880
    pub const DOWN_W_OFF: usize   = UP_B_OFF + UP_SB;                     // 1179648
    pub const DOWN_S_OFF: usize   = DOWN_W_OFF + DOWN_W;                  // 1703936
    pub const DOWN_B_OFF: usize   = DOWN_S_OFF + DOWN_SB;                 // 1736704
    pub const EXPERT_SIZE_4BIT: usize = DOWN_B_OFF + DOWN_SB;             // 1769472

    pub const GATE_W_SIZE: usize   = GATE_W;
    pub const GATE_S_SIZE: usize   = GATE_SB;
    pub const GATE_B_SIZE: usize   = GATE_SB;
    pub const UP_W_SIZE: usize     = UP_W;
    pub const UP_S_SIZE: usize     = UP_SB;
    pub const UP_B_SIZE: usize     = UP_SB;
    pub const DOWN_W_SIZE: usize   = DOWN_W;
    pub const DOWN_S_SIZE: usize   = DOWN_SB;
    pub const DOWN_B_SIZE: usize   = DOWN_SB;
}

/// Same as qwen35_35b but with reduced layers/experts for testing (stripped model).
/// Import as `use crate::constants::qwen35_35b_stripped::*;` to use bare names.
pub mod qwen35_35b_stripped {
    use super::{GROUP_SIZE, FULL_ATTN_INTERVAL};

    pub fn validate_config(hidden_dim: usize, num_layers: usize, num_experts: usize,
                           num_experts_per_tok: usize, moe_intermediate: usize,
                           shared_intermediate: usize, num_attn_heads: usize,
                           num_kv_heads: usize, head_dim: usize, vocab_size: usize,
                           linear_num_v_heads: usize, linear_num_k_heads: usize,
                           linear_total_key: usize, linear_total_value: usize,
                           ) -> Result<(), String> {
        let mut errs = Vec::new();
        if hidden_dim != HIDDEN_DIM { errs.push(format!("hidden_dim: config={}, const={}", hidden_dim, HIDDEN_DIM)); }
        if num_layers != NUM_LAYERS { errs.push(format!("num_layers: config={}, const={}", num_layers, NUM_LAYERS)); }
        if num_experts != NUM_EXPERTS { errs.push(format!("num_experts: config={}, const={}", num_experts, NUM_EXPERTS)); }
        if num_experts_per_tok != NUM_EXPERTS_PER_TOK { errs.push(format!("num_experts_per_tok: config={}, const={}", num_experts_per_tok, NUM_EXPERTS_PER_TOK)); }
        if moe_intermediate != MOE_INTERMEDIATE { errs.push(format!("moe_intermediate: config={}, const={}", moe_intermediate, MOE_INTERMEDIATE)); }
        if shared_intermediate != SHARED_INTERMEDIATE { errs.push(format!("shared_intermediate: config={}, const={}", shared_intermediate, SHARED_INTERMEDIATE)); }
        if num_attn_heads != NUM_ATTN_HEADS { errs.push(format!("num_attn_heads: config={}, const={}", num_attn_heads, NUM_ATTN_HEADS)); }
        if num_kv_heads != NUM_KV_HEADS { errs.push(format!("num_kv_heads: config={}, const={}", num_kv_heads, NUM_KV_HEADS)); }
        if head_dim != HEAD_DIM { errs.push(format!("head_dim: config={}, const={}", head_dim, HEAD_DIM)); }
        if vocab_size != VOCAB_SIZE { errs.push(format!("vocab_size: config={}, const={}", vocab_size, VOCAB_SIZE)); }
        if linear_num_v_heads != LINEAR_NUM_V_HEADS { errs.push(format!("linear_num_v_heads: config={}, const={}", linear_num_v_heads, LINEAR_NUM_V_HEADS)); }
        if linear_num_k_heads != LINEAR_NUM_K_HEADS { errs.push(format!("linear_num_k_heads: config={}, const={}", linear_num_k_heads, LINEAR_NUM_K_HEADS)); }
        if linear_total_key != LINEAR_TOTAL_KEY { errs.push(format!("linear_total_key: config={}, const={}", linear_total_key, LINEAR_TOTAL_KEY)); }
        if linear_total_value != LINEAR_TOTAL_VALUE { errs.push(format!("linear_total_value: config={}, const={}", linear_total_value, LINEAR_TOTAL_VALUE)); }
        if errs.is_empty() { Ok(()) } else { Err(errs.join("; ")) }
    }

    pub const HIDDEN_DIM: usize            = 2048;
    pub const NUM_LAYERS: usize            = 4;
    pub const NUM_ATTN_HEADS: usize        = 16;
    pub const NUM_KV_HEADS: usize          = 2;
    pub const HEAD_DIM: usize              = 256;
    pub const VOCAB_SIZE: usize            = 248320;
    pub const NUM_EXPERTS: usize           = 4;
    pub const NUM_EXPERTS_PER_TOK: usize   = 4;
    pub const MOE_INTERMEDIATE: usize      = 512;
    pub const SHARED_INTERMEDIATE: usize   = 512;

    pub const LINEAR_NUM_V_HEADS: usize    = 32;
    pub const LINEAR_NUM_K_HEADS: usize    = 16;
    pub const LINEAR_KEY_DIM: usize        = 128;
    pub const LINEAR_VALUE_DIM: usize      = 128;
    pub const LINEAR_TOTAL_KEY: usize      = LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM;   // 2048
    pub const LINEAR_TOTAL_VALUE: usize    = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM; // 4096
    pub const LINEAR_CONV_DIM: usize       = LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE; // 8192

    pub const ROPE_THETA: f64              = 10_000_000.0;
    pub const PARTIAL_ROTARY: f32          = 0.25;
    pub const ROTARY_DIM: usize            = (HEAD_DIM as f32 * PARTIAL_ROTARY) as usize; // 64

    pub const NUM_FULL_ATTN_LAYERS: usize  = NUM_LAYERS / FULL_ATTN_INTERVAL;  // 1
    pub const NUM_LINEAR_LAYERS: usize     = NUM_LAYERS - NUM_FULL_ATTN_LAYERS; // 3

    // ─── Expert 4-bit packed layout ─────────────────────────────────────

    const GS: usize = GROUP_SIZE;
    const HD: usize = HIDDEN_DIM;
    const MI: usize = MOE_INTERMEDIATE;

    const GATE_W: usize   = MI * HD / 2;              // 524288
    const GATE_SB: usize  = MI * (HD / GS) * 2;       // 32768
    const UP_W: usize     = MI * HD / 2;              // 524288
    const UP_SB: usize    = MI * (HD / GS) * 2;       // 32768
    const DOWN_W: usize   = HD * MI / 2;              // 524288
    const DOWN_SB: usize  = HD * (MI / GS) * 2;       // 32768

    pub const GATE_W_OFF: usize   = 0;                                    // 0
    pub const GATE_S_OFF: usize   = GATE_W;                               // 524288
    pub const GATE_B_OFF: usize   = GATE_W + GATE_SB;                     // 557056
    pub const UP_W_OFF: usize     = GATE_W + 2 * GATE_SB;                 // 589824
    pub const UP_S_OFF: usize     = UP_W_OFF + UP_W;                      // 1114112
    pub const UP_B_OFF: usize     = UP_S_OFF + UP_SB;                     // 1146880
    pub const DOWN_W_OFF: usize   = UP_B_OFF + UP_SB;                     // 1179648
    pub const DOWN_S_OFF: usize   = DOWN_W_OFF + DOWN_W;                  // 1703936
    pub const DOWN_B_OFF: usize   = DOWN_S_OFF + DOWN_SB;                 // 1736704
    pub const EXPERT_SIZE_4BIT: usize = DOWN_B_OFF + DOWN_SB;             // 1769472

    pub const GATE_W_SIZE: usize   = GATE_W;
    pub const GATE_S_SIZE: usize   = GATE_SB;
    pub const GATE_B_SIZE: usize   = GATE_SB;
    pub const UP_W_SIZE: usize     = UP_W;
    pub const UP_S_SIZE: usize     = UP_SB;
    pub const UP_B_SIZE: usize     = UP_SB;
    pub const DOWN_W_SIZE: usize   = DOWN_W;
    pub const DOWN_S_SIZE: usize   = DOWN_SB;
    pub const DOWN_B_SIZE: usize   = DOWN_SB;
}

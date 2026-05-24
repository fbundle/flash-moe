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

// ─── Backward-compat re-exports ─────────────────────────────────────────
/// Mirrors the #define constants in moe_infer_c/bench.m.
/// These modules exist so external code that still references
/// `qwen35_35b::HIDDEN_DIM` continues to compile.
#[allow(dead_code)]
pub mod qwen35_35b {
    pub use crate::engine::qwen35_moe::constants::{FullModel, ModelConfig};
    pub const HIDDEN_DIM: usize = 2048;
    pub const NUM_LAYERS: usize = 40;
    pub const NUM_ATTN_HEADS: usize = 16;
    pub const NUM_KV_HEADS: usize = 2;
    pub const HEAD_DIM: usize = 256;
    pub const VOCAB_SIZE: usize = 248320;
    pub const NUM_EXPERTS: usize = 256;
    pub const NUM_EXPERTS_PER_TOK: usize = 8;
    pub const MOE_INTERMEDIATE: usize = 512;
    pub const SHARED_INTERMEDIATE: usize = 512;
    pub const LINEAR_NUM_V_HEADS: usize = 32;
    pub const LINEAR_NUM_K_HEADS: usize = 16;
    pub const LINEAR_KEY_DIM: usize = 128;
    pub const LINEAR_VALUE_DIM: usize = 128;
    pub const LINEAR_TOTAL_KEY: usize = 2048;
    pub const LINEAR_TOTAL_VALUE: usize = 4096;
    pub const LINEAR_CONV_DIM: usize = 8192;
    pub const ROPE_THETA: f64 = 10_000_000.0;
    pub const ROTARY_DIM: usize = 64;
    pub const NUM_FULL_ATTN_LAYERS: usize = 10;
    pub const NUM_LINEAR_LAYERS: usize = 30;
    pub const KV_DIM: usize = 512;
    pub const EXPERT_SIZE_4BIT: usize = <FullModel as ModelConfig>::EXPERT_SIZE_4BIT;
    pub const GATE_W_OFF: usize = <FullModel as ModelConfig>::GATE_W_OFF;
    pub const GATE_S_OFF: usize = <FullModel as ModelConfig>::GATE_S_OFF;
    pub const GATE_B_OFF: usize = <FullModel as ModelConfig>::GATE_B_OFF;
    pub const UP_W_OFF: usize = <FullModel as ModelConfig>::UP_W_OFF;
    pub const UP_S_OFF: usize = <FullModel as ModelConfig>::UP_S_OFF;
    pub const UP_B_OFF: usize = <FullModel as ModelConfig>::UP_B_OFF;
    pub const DOWN_W_OFF: usize = <FullModel as ModelConfig>::DOWN_W_OFF;
    pub const DOWN_S_OFF: usize = <FullModel as ModelConfig>::DOWN_S_OFF;
    pub const DOWN_B_OFF: usize = <FullModel as ModelConfig>::DOWN_B_OFF;
    pub const GATE_W_SIZE: usize = <FullModel as ModelConfig>::GATE_W_SIZE;
    pub const GATE_S_SIZE: usize = <FullModel as ModelConfig>::GATE_S_SIZE;
    pub const GATE_B_SIZE: usize = <FullModel as ModelConfig>::GATE_B_SIZE;
    pub const UP_W_SIZE: usize = <FullModel as ModelConfig>::UP_W_SIZE;
    pub const UP_S_SIZE: usize = <FullModel as ModelConfig>::UP_S_SIZE;
    pub const UP_B_SIZE: usize = <FullModel as ModelConfig>::UP_B_SIZE;
    pub const DOWN_W_SIZE: usize = <FullModel as ModelConfig>::DOWN_W_SIZE;
    pub const DOWN_S_SIZE: usize = <FullModel as ModelConfig>::DOWN_S_SIZE;
    pub const DOWN_B_SIZE: usize = <FullModel as ModelConfig>::DOWN_B_SIZE;
    pub fn validate_config(hidden_dim: usize, num_layers: usize, num_experts: usize,
                           num_experts_per_tok: usize, moe_intermediate: usize,
                           shared_intermediate: usize, num_attn_heads: usize,
                           num_kv_heads: usize, head_dim: usize, vocab_size: usize,
                           linear_num_v_heads: usize, linear_num_k_heads: usize,
                           linear_total_key: usize, linear_total_value: usize,
    ) -> Result<(), String> {
        <FullModel as ModelConfig>::validate_config(hidden_dim, num_layers, num_experts,
            num_experts_per_tok, moe_intermediate, shared_intermediate,
            num_attn_heads, num_kv_heads, head_dim, vocab_size,
            linear_num_v_heads, linear_num_k_heads, linear_total_key, linear_total_value)
    }
}

#[allow(dead_code)]
pub mod qwen35_35b_stripped {
    pub use crate::engine::qwen35_moe::constants::{StrippedModel, ModelConfig};
    pub const HIDDEN_DIM: usize = 2048;
    pub const NUM_LAYERS: usize = 4;
    pub const NUM_ATTN_HEADS: usize = 16;
    pub const NUM_KV_HEADS: usize = 2;
    pub const HEAD_DIM: usize = 256;
    pub const VOCAB_SIZE: usize = 248320;
    pub const NUM_EXPERTS: usize = 4;
    pub const NUM_EXPERTS_PER_TOK: usize = 4;
    pub const MOE_INTERMEDIATE: usize = 512;
    pub const SHARED_INTERMEDIATE: usize = 512;
    pub const LINEAR_NUM_V_HEADS: usize = 32;
    pub const LINEAR_NUM_K_HEADS: usize = 16;
    pub const LINEAR_KEY_DIM: usize = 128;
    pub const LINEAR_VALUE_DIM: usize = 128;
    pub const LINEAR_TOTAL_KEY: usize = 2048;
    pub const LINEAR_TOTAL_VALUE: usize = 4096;
    pub const LINEAR_CONV_DIM: usize = 8192;
    pub const ROPE_THETA: f64 = 10_000_000.0;
    pub const ROTARY_DIM: usize = 64;
    pub const NUM_FULL_ATTN_LAYERS: usize = 1;
    pub const NUM_LINEAR_LAYERS: usize = 3;
    pub const KV_DIM: usize = 512;
    pub const EXPERT_SIZE_4BIT: usize = <StrippedModel as ModelConfig>::EXPERT_SIZE_4BIT;
    pub const GATE_W_OFF: usize = <StrippedModel as ModelConfig>::GATE_W_OFF;
    pub const GATE_S_OFF: usize = <StrippedModel as ModelConfig>::GATE_S_OFF;
    pub const GATE_B_OFF: usize = <StrippedModel as ModelConfig>::GATE_B_OFF;
    pub const UP_W_OFF: usize = <StrippedModel as ModelConfig>::UP_W_OFF;
    pub const UP_S_OFF: usize = <StrippedModel as ModelConfig>::UP_S_OFF;
    pub const UP_B_OFF: usize = <StrippedModel as ModelConfig>::UP_B_OFF;
    pub const DOWN_W_OFF: usize = <StrippedModel as ModelConfig>::DOWN_W_OFF;
    pub const DOWN_S_OFF: usize = <StrippedModel as ModelConfig>::DOWN_S_OFF;
    pub const DOWN_B_OFF: usize = <StrippedModel as ModelConfig>::DOWN_B_OFF;
    pub const GATE_W_SIZE: usize = <StrippedModel as ModelConfig>::GATE_W_SIZE;
    pub const GATE_S_SIZE: usize = <StrippedModel as ModelConfig>::GATE_S_SIZE;
    pub const GATE_B_SIZE: usize = <StrippedModel as ModelConfig>::GATE_B_SIZE;
    pub const UP_W_SIZE: usize = <StrippedModel as ModelConfig>::UP_W_SIZE;
    pub const UP_S_SIZE: usize = <StrippedModel as ModelConfig>::UP_S_SIZE;
    pub const UP_B_SIZE: usize = <StrippedModel as ModelConfig>::UP_B_SIZE;
    pub const DOWN_W_SIZE: usize = <StrippedModel as ModelConfig>::DOWN_W_SIZE;
    pub const DOWN_S_SIZE: usize = <StrippedModel as ModelConfig>::DOWN_S_SIZE;
    pub const DOWN_B_SIZE: usize = <StrippedModel as ModelConfig>::DOWN_B_SIZE;
    pub fn validate_config(hidden_dim: usize, num_layers: usize, num_experts: usize,
                           num_experts_per_tok: usize, moe_intermediate: usize,
                           shared_intermediate: usize, num_attn_heads: usize,
                           num_kv_heads: usize, head_dim: usize, vocab_size: usize,
                           linear_num_v_heads: usize, linear_num_k_heads: usize,
                           linear_total_key: usize, linear_total_value: usize,
    ) -> Result<(), String> {
        <StrippedModel as ModelConfig>::validate_config(hidden_dim, num_layers, num_experts,
            num_experts_per_tok, moe_intermediate, shared_intermediate,
            num_attn_heads, num_kv_heads, head_dim, vocab_size,
            linear_num_v_heads, linear_num_k_heads, linear_total_key, linear_total_value)
    }
}

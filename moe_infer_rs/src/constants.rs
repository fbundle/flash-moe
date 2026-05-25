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

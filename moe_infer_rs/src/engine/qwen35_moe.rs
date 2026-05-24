#[path = "qwen35_moe/constants.rs"]
pub mod constants;
#[path = "qwen35_moe/cpu.rs"]
pub mod cpu;
#[path = "qwen35_moe/fused_4bit.rs"]
pub mod fused_4bit;
#[path = "qwen35_moe/metal_context.rs"]
pub mod metal_context;
#[path = "qwen35_moe/metal_kernels.rs"]
pub mod metal_kernels;

pub use constants::{ModelConfig, FullModel, StrippedModel};
pub use cpu::CpuEngine;
pub use fused_4bit::Fused4bit;

/// Type alias for stripped model variant.
pub type Fused4bitStripped<'a> = Fused4bit<'a, StrippedModel>;

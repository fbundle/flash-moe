#[path = "qwen35_moe/constants.rs"]
pub mod constants;
#[path = "qwen35_moe/cpu.rs"]
pub mod cpu;
#[path = "qwen35_moe/fused_4bit.rs"]
pub mod fused_4bit;
#[path = "qwen35_moe/fused_4bit_exp1.rs"]
pub mod fused_4bit_exp1;
#[path = "qwen35_moe/fused_4bit_exp2.rs"]
pub mod fused_4bit_exp2;
#[path = "qwen35_moe/fused_4bit_exp3.rs"]
pub mod fused_4bit_exp3;
#[path = "qwen35_moe/metal_context.rs"]
pub mod metal_context;
#[path = "qwen35_moe/metal_kernels.rs"]
pub mod metal_kernels;

pub use metal_context::MetalContext as Qwen35MoEMetalContext;

pub use constants::{ModelConfig, FullModel, StrippedModel};
pub use constants::{ModelConfig as Qwen35MoEModelConfig, FullModel as Qwen35MoEFullModel, StrippedModel as Qwen35MoEStrippedModel};

pub use cpu::CpuEngine;
pub use cpu::CpuEngine as Qwen35MoECpuEngine;
pub use fused_4bit::Fused4bit;
pub use fused_4bit::Fused4bit as Qwen35MoEFused4bit;
pub use fused_4bit_exp1::Fused4bitExp1;
pub use fused_4bit_exp1::Fused4bitExp1 as Qwen35MoEFused4bitExp1;
pub use fused_4bit_exp2::Fused4bitExp2;
pub use fused_4bit_exp2::Fused4bitExp2 as Qwen35MoEFused4bitExp2;
pub use fused_4bit_exp3::Fused4bitExp3;
pub use fused_4bit_exp3::Fused4bitExp3 as Qwen35MoEFused4bitExp3;

pub type Qwen35MoEFused4bitStripped = Fused4bit<StrippedModel>;
pub type Qwen35MoEFused4bitExp1Stripped = Fused4bitExp1<StrippedModel>;
pub type Qwen35MoEFused4bitExp2Stripped = Fused4bitExp2<StrippedModel>;
pub type Qwen35MoEFused4bitExp3Stripped = Fused4bitExp3<StrippedModel>;

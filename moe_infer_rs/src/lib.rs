pub mod config;
pub mod constants;
pub mod error;
pub mod expert;
pub mod full_forward;
pub mod gpu_forward;
pub mod kernels;
pub mod metal_context;
pub mod moe;
pub mod pipeline_common;
pub mod pipeline_cpu;
pub mod pipeline_fusedwoods;
pub mod pipeline_fusedexp;
pub mod quant;
pub mod timer;
pub mod weights;

#[cfg(feature = "python-bindings")]
mod python_bindings;

// Re-export key types
pub use config::{load_model_config, ExpertLayout, ModelConfig};
pub use constants::*;
pub use error::MoEError;
pub use expert::{run_expert_forward, run_expert_forward_fast, ExpertTiming};
pub use full_forward::{run_full_forward, FullForwardTiming};
pub use gpu_forward::{moe_layer_forward, linear_attention_forward, full_attention_forward};
pub use pipeline_common::{LinearAttnState, FullAttnCache, FullAttnCmd2State, DeferredExperts, PipelineMode, LinearAttnFusedWoodsState};
pub use metal_context::{MetalContext, GpuWeightCtx, ExpertIOState, ExpertCache, metal_buf_shared};
pub use moe::{run_moe_forward, run_moe_forward_fused, MoETiming};
pub use quant::{bf16_to_f32, cpu_dequant_matvec_4bit, cpu_swiglu};
pub use timer::now_ms;
pub use weights::WeightFile;

#[cfg(feature = "python-bindings")]
#[pyo3::pymodule]
fn moe_infer(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    m.add_class::<python_bindings::Cache>()?;
    m.add_class::<python_bindings::Context>()?;
    Ok(())
}

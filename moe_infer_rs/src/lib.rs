#[path = "model/model.rs"] mod model;
#[path = "model/model_config.rs"] mod model_config;
#[path = "model/model_weights.rs"] mod model_weights;
mod cache;
mod constants;
#[path = "engine/engine.rs"] pub mod engine;
#[path = "engine/engine_cpu.rs"] mod engine_cpu;
#[path = "engine/engine_gpu.rs"] mod engine_gpu;
#[path = "engine/engine_fusedexp.rs"] mod engine_fusedexp;
#[path = "engine/engine_fusedwoods.rs"] mod engine_fusedwoods;
mod error;
mod metal_kernels;
mod metal_context;
#[path = "math/math.rs"] mod math;
mod generate;
mod timer;

#[cfg(feature = "python-bindings")]
mod python_bindings;

#[cfg(feature = "python-bindings")]
#[pyo3::pymodule]
fn moe_infer(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    m.add_class::<python_bindings::Model>()?;
    m.add_class::<python_bindings::Engine>()?;
    m.add_class::<python_bindings::Cache>()?;
    Ok(())
}

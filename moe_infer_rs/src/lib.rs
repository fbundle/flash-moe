mod model_config;
mod constants;
pub mod engine;
mod error;
mod engine_gpu;
mod metal_kernels;
mod metal_context;
mod math;
mod engine_cpu;
mod engine_fusedwoods;
mod engine_fusedexp;
mod generate;
mod timer;
mod model_weights;

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

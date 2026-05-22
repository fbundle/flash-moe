mod config;
mod constants;
pub mod engine;
mod error;
mod pipeline_gpu;
mod metal_kernels;
mod metal_context;
mod pipeline_common;
mod pipeline_cpu;
mod pipeline_fusedwoods;
mod pipeline_fusedexp;
mod timer;
mod weights;

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

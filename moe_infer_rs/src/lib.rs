#[path = "math/util.rs"] mod math;
#[path = "math/full_attention.rs"] mod math_full_attention;
#[path = "math/linear_attention.rs"] mod math_linear_attention;
#[path = "math/lm_head.rs"] mod math_lm_head;
#[path = "math/moe.rs"] mod math_moe;
#[path = "math/sample.rs"] mod math_sample;

pub mod engine;
pub mod model;

mod cache;
mod constants;
mod error;
mod metal_kernels;
mod metal_context;
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

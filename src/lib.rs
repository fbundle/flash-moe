#[path = "math_util.rs"] mod math;
#[path = "engine.rs"] pub mod engine;
#[path = "model.rs"] pub mod model;

mod cache;
mod constants;
mod error;
mod quant;
mod safetensors;
#[path = "quantize/qwen35_moe/bq4.rs"] pub mod bq4;
mod timer;

#[cfg(feature = "python-bindings")]
mod python_bindings;

#[cfg(feature = "python-bindings")]
#[pyo3::pymodule]
fn _moe_infer_rs(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;
    use pyo3::wrap_pyfunction;
    m.add_class::<python_bindings::Model>()?;
    m.add_class::<python_bindings::Engine>()?;
    m.add_class::<python_bindings::Cache>()?;
    m.add_function(wrap_pyfunction!(python_bindings::record_engine_telemetry, m)?)?;
    m.add_function(wrap_pyfunction!(python_bindings::qwen35_moe_bq4_quantize, m)?)?;
    Ok(())
}

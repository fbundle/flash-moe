/// Thin PyO3 bindings for the MoE-Infer inference engine.
use std::collections::BTreeMap;
use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::cache::Cache as CoreCache;
use crate::model::Model as CoreModel;
use crate::error::MoEError;
use crate::engine::{SignalCheckFn, TelemetryValue, set_record_telemetry, EngineEnum, DynEngine};
use crate::engine::qwen35_moe::metal_context::{ExpertBuffer, WeightBuffer, MetalContext};

// ─── Module-level functions ──────────────────────────────────────────────────

/// Enable or disable engine-level telemetry recording globally.
#[pyfunction]
pub fn record_engine_telemetry(on: bool) {
    set_record_telemetry(on);
}

// ─── Model (thin wrapper) ───────────────────────────────────────────────────

#[pyclass(unsendable)]
pub struct Model {
    inner: Arc<CoreModel>,
}

#[pymethods]
impl Model {
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        CoreModel::load(model_path)
            .map(|m| Model { inner: Arc::new(m) })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("Model({} layers, hidden={})",
            self.inner.config.get_usize("num_layers").unwrap_or(0),
            self.inner.config.get_usize("hidden_dim").unwrap_or(0))
    }
}

// ─── Cache (thin wrapper) ───────────────────────────────────────────────────

#[pyclass(unsendable)]
pub struct Cache {
    inner: CoreCache,
}

#[pymethods]
impl Cache {
    #[new]
    fn new(model: &Model) -> Self {
        Cache { inner: CoreCache::new(&model.inner.config) }
    }

    #[getter]
    fn pos(&self) -> usize { self.inner.pos }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        CoreCache::load(path)
            .map(|c| Cache { inner: c })
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("Cache(pos={})", self.inner.pos)
    }
}

// ─── Engine (owns GPU resources, holds the type-erased inner engine) ─────────

#[pyclass(unsendable)]
pub struct Engine {
    engine: Option<DynEngine>,
    model: Arc<CoreModel>,
    ctx: MetalContext,
    weight_buffer: WeightBuffer,
    expert_buffer: Option<ExpertBuffer>,
    engine_type: EngineEnum,
    k: usize,
    /// Engine-level telemetry: only populated when record_engine_telemetry(true).
    pub telemetry: BTreeMap<String, TelemetryValue>,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (model, pipeline_mode="FusedExp", k=0))]
    fn new(model: &Model, pipeline_mode: &str, k: usize) -> PyResult<Self> {
        let engine_type = match pipeline_mode {
            "FusedExp" => EngineEnum::FusedExp,
            "FusedExpStripped" => EngineEnum::FusedExpStripped,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown pipeline_mode: {}. Use FusedExp|FusedExpStripped", pipeline_mode
            ))),
        };

        let (ctx, weight_buffer, expert_buffer) = engine_type.init_gpu(&model.inner, k)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Engine {
            engine: None,
            model: model.inner.clone(),
            ctx,
            weight_buffer,
            expert_buffer: Some(expert_buffer),
            engine_type,
            k,
            telemetry: BTreeMap::new(),
        })
    }

    fn forward(&mut self, py: Python<'_>, input_ids: &Bound<PyArray1<i64>>,
        cache: &mut Cache,
    ) -> PyResult<PyObject> {
        let ids = input_ids.readonly();
        let ids = ids.as_slice()?;
        let pos = cache.inner.pos;
        let start = if pos < ids.len() { pos } else { 0 };
        let new_ids = &ids[start..];
        let n = new_ids.len();
        let vs = self.model.config.get_usize("vocab_size").unwrap();

        let logits = self.forward_impl(new_ids, &mut cache.inner, &mut || py.check_signals().is_err())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let arr = PyArray2::<f32>::from_owned_array(py,
            numpy::ndarray::Array2::from_shape_vec((n, vs), logits)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("shape error: {}", e)))?);
        Ok(arr.into_pyobject(py)?.into_any().into())
    }

    /// Expose upload/download for callers that manage cache persistence.
    fn upload_cache(&self, cache: &Cache) {
        if let Some(ref eng) = self.engine {
            eng.upload_cache(&cache.inner);
        }
    }

    fn download_cache(&self, cache: &mut Cache) {
        if let Some(ref eng) = self.engine {
            eng.download_cache(&mut cache.inner);
        }
    }

    /// Engine-level telemetry (only populated when record_engine_telemetry(true)).
    fn telemetry(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in &self.telemetry {
            match v {
                TelemetryValue::Scalar(val) => { dict.set_item(k, *val)?; }
                TelemetryValue::List(vals) => {
                    let py_list = PyList::new(py, vals.iter().map(|&x| x))?;
                    dict.set_item(k, py_list)?;
                }
            }
        }
        Ok(dict.into_pyobject(py)?.into_any().into())
    }

    fn __repr__(&self) -> String {
        format!("Engine(loaded: {} layers, hidden={})",
            self.model.config.get_usize("num_layers").unwrap_or(0),
            self.model.config.get_usize("hidden_dim").unwrap_or(0))
    }
}

// ─── Internal forward impl (lazy-inits the inner engine) ────────────────────

impl Engine {
    fn forward_impl(
        &mut self,
        input_ids: &[i64],
        cache: &mut CoreCache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError> {
        if self.engine.is_none() {
            // SAFETY: Engine is on the PyO3 heap (stable address).
            self.engine = Some(unsafe {
                DynEngine::new(
                    &self.model, &self.ctx, &self.weight_buffer,
                    self.expert_buffer.as_mut(), self.k, self.engine_type,
                )?
            });
        }
        let eng = self.engine.as_mut().unwrap();
        eng.upload_cache(cache);
        let logits = eng.forward(input_ids, check_signal)?;
        eng.download_cache(cache);
        self.telemetry = eng.telemetry();
        Ok(logits)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Drop engine before the fields it references (model, ctx, weight_buffer).
        let _ = self.engine.take();
    }
}

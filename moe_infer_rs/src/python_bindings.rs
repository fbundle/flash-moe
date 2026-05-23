/// Thin PyO3 bindings for the MoE-Infer inference engine.
use std::collections::BTreeMap;
use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::cache::Cache as CoreCache;
use crate::model::Model as CoreModel;
use crate::engine::cpu::EngineCPU;
use crate::engine::fusedexp::EngineFusedExp;
use crate::engine::fusedwoods::EngineFusedWoods;
use crate::error::MoEError;
use crate::engine::{Engine as EngineTrait, SignalCheckFn, TelemetryValue, set_record_telemetry};
use crate::metal_context::{ExpertBuffer, WeightBuffer, MetalContext};

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
            self.inner.config.num_layers, self.inner.config.hidden_dim)
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

    fn __repr__(&self) -> String { format!("Cache(pos={})", self.inner.pos) }
}

// ─── Engine (owns GPU resources, implements Engine trait) ──────────────────

#[pyclass(unsendable)]
pub struct Engine {
    model: Arc<CoreModel>,
    ctx: MetalContext,
    gpu_wf: WeightBuffer,
    expert_gpu_buffer: Option<ExpertBuffer>,
    mode: String,
    k: usize,
    /// Engine-level telemetry: only populated when record_engine_telemetry(true).
    pub telemetry: BTreeMap<String, TelemetryValue>,
}

impl EngineTrait for Engine {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut CoreCache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError> {
        let result = match self.mode.as_str() {
            "Cpu" | "CpuOnly" => {
                let mut engine = EngineCPU { model: &self.model };
                let logits = engine.forward(input_ids, cache, check_signal);
                self.telemetry = engine.telemetry();
                logits
            }
            "FusedExp" => {
                let mut engine = EngineFusedExp {
                    model: &self.model,
                    ctx: &self.ctx,
                    gpu_wf: &self.gpu_wf,
                    expert_gpu_buffer: self.expert_gpu_buffer.as_mut(),
                    k: self.k,
                    timing: BTreeMap::new(),
                };
                let logits = engine.forward(input_ids, cache, check_signal);
                self.telemetry = engine.telemetry();
                logits
            }
            "FusedWoods" => {
                let mut engine = EngineFusedWoods {
                    model: &self.model,
                    ctx: &self.ctx,
                    gpu_wf: &self.gpu_wf,
                    expert_gpu_buffer: self.expert_gpu_buffer.as_mut(),
                    norm_cache: std::collections::HashMap::new(),
                };
                let logits = engine.forward(input_ids, cache, check_signal);
                self.telemetry = engine.telemetry();
                logits
            }
            _ => return Err(MoEError::Config(format!("Unknown pipeline mode: {}", self.mode))),
        };
        result
    }
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (model, pipeline_mode="FusedExp", k=0))]
    fn new(model: &Model, pipeline_mode: &str, k: usize) -> PyResult<Self> {
        match pipeline_mode {
            "Cpu" | "CpuOnly" | "FusedExp" | "FusedWoods" => {}
            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown pipeline_mode: {}. Use Cpu|FusedExp|FusedWoods", pipeline_mode
            ))),
        }
        let config = &model.inner.config;
        let k = if k == 0 { config.num_experts_per_tok } else { k };
        if k > config.num_experts_per_tok {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "k ({}) must not exceed model's num_experts_per_tok ({})", k, config.num_experts_per_tok
            )));
        }
        let mut ctx = MetalContext::init()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("metal: {}", e)))?;
        let key_dim = config.linear_total_key / config.linear_num_k_heads;
        let value_dim = config.linear_total_value / config.linear_num_v_heads;
        ctx.init_linear_attn_buffers(
            config.num_linear_layers,
            config.linear_conv_dim,
            config.linear_num_v_heads,
            config.linear_total_value,
            key_dim,
            value_dim,
            config.hidden_dim,
            config.num_experts,
            config.shared_intermediate,
        );
        let expert_gpu_buffer = Some(ctx.init_expert_buffers(
            config.expert_size_4bit,
            config.hidden_dim,
            config.moe_intermediate,
            config.shared_intermediate,
        ));
        let gpu_wf = WeightBuffer::new(&ctx.device, &model.inner.wf);

        eprintln!(
            "[engine] {} layers hidden={} experts={} mode={}",
            config.num_layers, config.hidden_dim, config.num_experts, pipeline_mode
        );
        Ok(Engine {
            model: model.inner.clone(),
            ctx,
            gpu_wf,
            expert_gpu_buffer,
            mode: pipeline_mode.to_string(),
            k,
            telemetry: BTreeMap::new(),
        })
    }

    fn forward(&mut self, py: Python<'_>, input_ids: &Bound<PyArray1<i64>>,
        cache: &mut Cache,
    ) -> PyResult<PyObject> {
        let ids = input_ids.readonly();
        let ids = ids.as_slice()?;
        let start = if cache.inner.pos < ids.len() { cache.inner.pos } else { 0 };
        let new_ids = &ids[start..];
        let n = new_ids.len();
        let vs = self.model.config.vocab_size;

        let logits = EngineTrait::forward(
            self, new_ids, &mut cache.inner,
            &mut || py.check_signals().is_err(),
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let arr = PyArray2::<f32>::from_owned_array(py,
            numpy::ndarray::Array2::from_shape_vec((n, vs), logits)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("shape error: {}", e)))?);
        Ok(arr.into_pyobject(py)?.into_any().into())
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
            self.model.config.num_layers, self.model.config.hidden_dim)
    }
}

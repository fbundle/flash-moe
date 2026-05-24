/// Thin PyO3 bindings for the MoE-Infer inference engine.
use std::collections::BTreeMap;
use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::cache::Cache as CoreCache;
use crate::model::Model as CoreModel;
use crate::constants::{qwen35_35b, qwen35_35b_stripped};
use crate::error::MoEError;
use crate::engine::{SignalCheckFn, TelemetryValue, set_record_telemetry, PipelineMode, ErasedEngine};
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

    fn __repr__(&self) -> String { format!("Cache(pos={})", self.inner.pos) }
}

// ─── Engine (owns GPU resources, holds the type-erased inner engine) ─────────

#[pyclass(unsendable)]
pub struct Engine {
    engine: Option<ErasedEngine>,
    model: Arc<CoreModel>,
    ctx: MetalContext,
    gpu_wf: WeightBuffer,
    expert_gpu_buffer: Option<ExpertBuffer>,
    mode: PipelineMode,
    k: usize,
    /// Engine-level telemetry: only populated when record_engine_telemetry(true).
    pub telemetry: BTreeMap<String, TelemetryValue>,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (model, pipeline_mode="FusedExp", k=0))]
    fn new(model: &Model, pipeline_mode: &str, k: usize) -> PyResult<Self> {
        let mode = match pipeline_mode {
            "FusedExp" => PipelineMode::FusedExp,
            "FusedWoods" => PipelineMode::FusedWoods,
            "FusedExpStripped" => PipelineMode::FusedExpStripped,
            "FusedWoodsStripped" => PipelineMode::FusedWoodsStripped,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown pipeline_mode: {}. Use FusedExp|FusedWoods|FusedExpStripped|FusedWoodsStripped", pipeline_mode
            ))),
        };

        let is_stripped = matches!(mode, PipelineMode::FusedExpStripped | PipelineMode::FusedWoodsStripped);
        let (num_layers, num_experts, num_experts_per_tok, num_linear_layers, linear_conv_dim,
             linear_num_v_heads, linear_total_value, linear_key_dim, linear_value_dim,
             hidden_dim, shared_intermediate, moe_intermediate, expert_size_4bit,
             num_full_attn_layers, kv_dim, num_attn_heads, head_dim) =
            if is_stripped {
                (qwen35_35b_stripped::NUM_LAYERS, qwen35_35b_stripped::NUM_EXPERTS, qwen35_35b_stripped::NUM_EXPERTS_PER_TOK,
                 qwen35_35b_stripped::NUM_LINEAR_LAYERS, qwen35_35b_stripped::LINEAR_CONV_DIM,
                 qwen35_35b_stripped::LINEAR_NUM_V_HEADS, qwen35_35b_stripped::LINEAR_TOTAL_VALUE,
                 qwen35_35b_stripped::LINEAR_KEY_DIM, qwen35_35b_stripped::LINEAR_VALUE_DIM,
                 qwen35_35b_stripped::HIDDEN_DIM, qwen35_35b_stripped::SHARED_INTERMEDIATE,
                 qwen35_35b_stripped::MOE_INTERMEDIATE, qwen35_35b_stripped::EXPERT_SIZE_4BIT,
                 qwen35_35b_stripped::NUM_FULL_ATTN_LAYERS,
                 qwen35_35b_stripped::NUM_KV_HEADS * qwen35_35b_stripped::HEAD_DIM,
                 qwen35_35b_stripped::NUM_ATTN_HEADS, qwen35_35b_stripped::HEAD_DIM)
            } else {
                (qwen35_35b::NUM_LAYERS, qwen35_35b::NUM_EXPERTS, qwen35_35b::NUM_EXPERTS_PER_TOK,
                 qwen35_35b::NUM_LINEAR_LAYERS, qwen35_35b::LINEAR_CONV_DIM,
                 qwen35_35b::LINEAR_NUM_V_HEADS, qwen35_35b::LINEAR_TOTAL_VALUE,
                 qwen35_35b::LINEAR_KEY_DIM, qwen35_35b::LINEAR_VALUE_DIM,
                 qwen35_35b::HIDDEN_DIM, qwen35_35b::SHARED_INTERMEDIATE,
                 qwen35_35b::MOE_INTERMEDIATE, qwen35_35b::EXPERT_SIZE_4BIT,
                 qwen35_35b::NUM_FULL_ATTN_LAYERS,
                 qwen35_35b::NUM_KV_HEADS * qwen35_35b::HEAD_DIM,
                 qwen35_35b::NUM_ATTN_HEADS, qwen35_35b::HEAD_DIM)
            };

        let k = if k == 0 { num_experts_per_tok } else { k };
        if k > num_experts_per_tok {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "k ({}) must not exceed model's num_experts_per_tok ({})", k, num_experts_per_tok
            )));
        }
        let mut ctx = MetalContext::init()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("metal: {}", e)))?;
        ctx.init_linear_attn_buffers(
            num_linear_layers, linear_conv_dim, linear_num_v_heads,
            linear_total_value, linear_key_dim, linear_value_dim,
            hidden_dim, num_experts, shared_intermediate,
            num_full_attn_layers, kv_dim,
            num_attn_heads, head_dim,
            num_attn_heads * 2 * head_dim,
        );
        let expert_gpu_buffer = Some(ctx.init_expert_buffers(
            expert_size_4bit, hidden_dim, moe_intermediate, shared_intermediate,
        ));
        let gpu_wf = WeightBuffer::new(&ctx.device, &model.inner.wf);

        eprintln!(
            "[engine] {} layers hidden={} experts={} mode={}",
            num_layers, hidden_dim, num_experts, pipeline_mode
        );
        Ok(Engine {
            engine: None,
            model: model.inner.clone(),
            ctx,
            gpu_wf,
            expert_gpu_buffer,
            mode,
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
        let vs = self.model.config.get_usize("vocab_size").unwrap();

        let logits = self.forward_impl(new_ids, &mut cache.inner, &mut || py.check_signals().is_err())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

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
                ErasedEngine::new(
                    &self.model, &self.ctx, &self.gpu_wf,
                    self.expert_gpu_buffer.as_mut(), self.k, self.mode,
                )?
            });
        }
        let eng = self.engine.as_mut().unwrap();
        let logits = eng.forward(input_ids, cache, check_signal)?;
        self.telemetry = eng.telemetry();
        Ok(logits)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Drop engine before the fields it references (model, ctx, gpu_wf).
        let _ = self.engine.take();
    }
}

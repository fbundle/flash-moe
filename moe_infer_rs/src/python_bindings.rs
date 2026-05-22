/// Thin PyO3 bindings for the MoE-Infer inference engine.
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand::Rng;

use crate::cache::Cache as CoreCache;
use crate::model::Model as CoreModel;
use crate::engine::cpu::EngineCPU;
use crate::engine::fusedexp::EngineFusedExp;
use crate::engine::fusedwoods::EngineFusedWoods;
use crate::engine::{Engine as EngineTrait, SignalCheckFn, TelemetryValue, set_record_telemetry as set_telemetry};
use crate::math::softmax;
use crate::metal_context::{ExpertBuffer, WeightBuffer, MetalContext};

// ─── Module-level functions ──────────────────────────────────────────────────

/// Enable or disable engine-level telemetry recording globally.
#[pyfunction]
pub fn record_telemetry(on: bool) {
    set_telemetry(on);
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
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
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

// ─── Telemetry ──────────────────────────────────────────────────────────────

pub struct Telemetry {
    pub prefill_ms: f64,
    pub total_ms: f64,
    pub tokens_generated: usize,
}

// ─── Sampling ──────────────────────────────────────────────────────────────

pub fn sample(logits: &mut [f32], temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> usize {
    let n = logits.len();
    if (temperature - 1.0).abs() > 1e-7 {
        let inv = 1.0 / temperature.max(1e-8);
        for v in logits.iter_mut() { *v *= inv; }
    }
    if temperature < 0.01 {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
    }
    softmax(logits);

    if top_k > 0 && top_k < n {
        let mut v: Vec<f32> = logits.to_vec();
        v.select_nth_unstable_by(top_k, |a, b| b.partial_cmp(a).unwrap());
        let t = v[top_k - 1];
        for x in logits.iter_mut() { if *x < t { *x = 0.0; } }
    }
    if top_p < 1.0 {
        let mut s: Vec<f32> = logits.iter().copied().filter(|&x| x > 0.0).collect();
        s.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let total: f32 = s.iter().sum();
        let mut cum = 0.0;
        let mut cut = 0.0;
        for v in s {
            cum += v;
            if cum / total >= top_p { cut = v; break; }
        }
        for x in logits.iter_mut() { if *x < cut { *x = 0.0; } }
    }
    if min_p > 0.0 {
        let max_p = logits.iter().fold(0.0f32, |a, &b| a.max(b));
        let t = max_p * min_p;
        for x in logits.iter_mut() { if *x < t { *x = 0.0; } }
    }

    let sum: f32 = logits.iter().sum();
    if sum <= 0.0 { return 0; }
    let inv = 1.0 / sum;
    let r: f32 = rand::thread_rng().gen();
    let mut cum = 0.0;
    for (i, &v) in logits.iter().enumerate() {
        cum += v * inv;
        if r <= cum { return i; }
    }
    n - 1
}

fn pick_token(logits: &[f32], temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> usize {
    if temperature < 0.01 {
        logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        let mut copy = logits.to_vec();
        sample(&mut copy, temperature, top_k, top_p, min_p)
    }
}

// ─── Engine (owns GPU resources, implements Engine trait) ──────────────────

#[pyclass(unsendable)]
pub struct Engine {
    model: Arc<CoreModel>,
    ctx: MetalContext,
    gpu_wf: WeightBuffer,
    expert_gpu_buffer: Option<ExpertBuffer>,
    mode: String,
    pub telemetry: Telemetry,
    engine_telemetry: BTreeMap<String, TelemetryValue>,
}

impl EngineTrait for Engine {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut CoreCache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, String> {
        let result = match self.mode.as_str() {
            "Cpu" | "CpuOnly" => {
                let mut engine = EngineCPU { model: &self.model };
                let logits = engine.forward(input_ids, cache, check_signal);
                self.engine_telemetry = engine.telemetry();
                logits
            }
            "FusedExp" => {
                let mut engine = EngineFusedExp {
                    model: &self.model,
                    ctx: &self.ctx,
                    gpu_wf: &self.gpu_wf,
                    expert_gpu_buffer: self.expert_gpu_buffer.as_mut(),
                    timing: BTreeMap::new(),
                };
                let logits = engine.forward(input_ids, cache, check_signal);
                self.engine_telemetry = engine.telemetry();
                logits
            }
            "FusedWoods" => {
                let mut engine = EngineFusedWoods {
                    model: &self.model,
                    ctx: &self.ctx,
                    gpu_wf: &self.gpu_wf,
                    expert_gpu_buffer: self.expert_gpu_buffer.as_mut(),
                };
                let logits = engine.forward(input_ids, cache, check_signal);
                self.engine_telemetry = engine.telemetry();
                logits
            }
            _ => return Err(format!("Unknown pipeline mode: {}", self.mode)),
        };
        result
    }
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (model, pipeline_mode="FusedExp"))]
    fn new(model: &Model, pipeline_mode: &str) -> PyResult<Self> {
        match pipeline_mode {
            "Cpu" | "CpuOnly" | "FusedExp" | "FusedWoods" => {}
            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown pipeline_mode: {}. Use Cpu|FusedExp|FusedWoods", pipeline_mode
            ))),
        }
        let config = &model.inner.config;
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
            telemetry: Telemetry { prefill_ms: 0.0, total_ms: 0.0, tokens_generated: 0 },
            engine_telemetry: BTreeMap::new(),
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
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        let arr = PyArray2::<f32>::from_owned_array(py,
            numpy::ndarray::Array2::from_shape_vec((n, vs), logits).unwrap());
        Ok(arr.into_pyobject(py).unwrap().into_any().into())
    }

    #[pyo3(signature = (input_ids, cache, max_tokens=256, temperature=0.0,
                        top_k=0, top_p=1.0, min_p=0.0, eos_token_ids=None))]
    fn stream_generate(&mut self, py: Python<'_>, input_ids: &Bound<PyArray1<i64>>,
        cache: &mut Cache,
        max_tokens: usize, temperature: f32, top_k: usize, top_p: f32, min_p: f32,
        eos_token_ids: Option<&Bound<PyArray1<i64>>>,
    ) -> PyResult<PyObject> {
        let gen_t0 = Instant::now();
        let eos: HashSet<usize> = match eos_token_ids {
            Some(a) => a.readonly().to_vec()?.into_iter().map(|x| x as usize).collect(),
            None => [248046usize, 248044].into(),
        };

        let logits_obj = self.forward(py, input_ids, cache)?;
        let la = logits_obj.downcast_bound::<PyArray2<f32>>(py).map_err(|_|
            pyo3::exceptions::PyRuntimeError::new_err("expected ndarray"))?;
        let ls = unsafe { la.as_slice() }.map_err(|e|
            pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

        let vs = self.model.config.vocab_size;
        let logits_last = &ls[ls.len() - vs..];

        self.telemetry = Telemetry { prefill_ms: gen_t0.elapsed().as_secs_f64() * 1000.0,
            total_ms: 0.0, tokens_generated: 0 };

        let first_token = pick_token(logits_last, temperature, top_k, top_p, min_p);

        Ok(StreamGenIterator {
            engine_ptr: self as *mut Engine as *mut dyn EngineTrait,
            cache_ptr: cache as *mut Cache as *mut CoreCache,
            telemetry_ptr: &mut self.telemetry as *mut Telemetry,
            gen_t0,
            next_token: first_token,
            logits: logits_last.to_vec(),
            remaining: max_tokens.saturating_sub(1),
            done: false,
            eos,
            temperature, top_k, top_p, min_p,
        }.into_pyobject(py).unwrap().into_any().into())
    }

    fn telemetry(&self, py: Python<'_>) -> PyResult<PyObject> {
        let t = &self.telemetry;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("prefill_ms", t.prefill_ms)?;
        dict.set_item("total_ms", t.total_ms)?;
        dict.set_item("tokens_generated", t.tokens_generated)?;
        let tps = if t.total_ms > 0.0 && t.tokens_generated > 1 {
            let gen_ms = t.total_ms - t.prefill_ms;
            if gen_ms > 0.0 {
                (t.tokens_generated - 1) as f64 / (gen_ms / 1000.0)
            } else { 0.0 }
        } else { 0.0 };
        dict.set_item("tokens_per_sec", tps)?;
        for (k, v) in &self.engine_telemetry {
            match v {
                TelemetryValue::Scalar(val) => { dict.set_item(k, *val)?; }
                TelemetryValue::List(vals) => {
                    let py_list = PyList::new(py, vals.iter().map(|&x| x))?;
                    dict.set_item(k, py_list)?;
                }
            }
        }
        Ok(dict.into_pyobject(py).unwrap().into_any().into())
    }

    fn __repr__(&self) -> String {
        format!("Engine(loaded: {} layers, hidden={})",
            self.model.config.num_layers, self.model.config.hidden_dim)
    }
}

// ─── Streaming iterator ─────────────────────────────────────────────────────

#[pyclass(unsendable)]
pub struct StreamGenIterator {
    engine_ptr: *mut dyn EngineTrait,
    cache_ptr: *mut CoreCache,
    telemetry_ptr: *mut Telemetry,
    gen_t0: Instant,
    next_token: usize,
    logits: Vec<f32>,
    remaining: usize,
    done: bool,
    eos: HashSet<usize>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
}

#[pymethods]
impl StreamGenIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<(i64, PyObject)>> {
        if self.done {
            return Ok(None);
        }
        let token = self.next_token as i64;
        let logits = std::mem::take(&mut self.logits);
        unsafe { &mut *self.telemetry_ptr }.tokens_generated += 1;

        if self.remaining == 0 || self.eos.contains(&self.next_token) {
            self.done = true;
            let t = unsafe { &mut *self.telemetry_ptr };
            t.total_ms = self.gen_t0.elapsed().as_secs_f64() * 1000.0;
            let obj = PyArray1::<f32>::from_vec(py, logits)
                .into_pyobject(py).unwrap().into_any().into();
            return Ok(Some((token, obj)));
        }

        self.remaining -= 1;
        let engine = unsafe { &mut *self.engine_ptr };
        let cache = unsafe { &mut *self.cache_ptr };
        self.logits = engine
            .forward(&[token], cache, &mut || py.check_signals().is_err())
            .unwrap_or_else(|_| {
                self.done = true;
                vec![]
            });
        if !self.done {
            self.next_token = pick_token(&self.logits, self.temperature,
                self.top_k, self.top_p, self.min_p);
        }

        let obj = PyArray1::<f32>::from_vec(py, logits)
            .into_pyobject(py).unwrap().into_any().into();
        Ok(Some((token, obj)))
    }
}

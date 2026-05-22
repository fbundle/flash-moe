/// Thin PyO3 bindings for the MoE-Infer inference engine.
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::engine::{
    Cache as CoreCache, Engine as EngineTrait, Model as CoreModel,
};
use crate::engine_cpu::EngineCPU;
use crate::engine_fusedexp::{process_token_fusedexp_pipelined, EngineFusedExp};
use crate::engine_fusedwoods::EngineFusedWoods;
use crate::engine_common::process_token_inner;
use crate::engine_gpu::EngineGPU;
use crate::generate::{SampleParams, Telemetry};
use crate::engine_common::{
    embed_lookup, final_norm, gpu_lm_head, sample,
    ExecCtxGpu, SignalCheckFn,
};
use crate::metal_context::{ExpertBuffer, GpuWeightCtx, MetalContext};

// ─── Engine name (selects which engine impl to use) ───────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineName {
    Cpu,
    Gpu,
    FusedExp,
    FusedWoods,
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

// ─── Engine (owns GPU resources, implements Engine trait) ──────────────────

#[pyclass(unsendable)]
pub struct Engine {
    model: Arc<CoreModel>,
    ctx: MetalContext,
    gpu_wf: GpuWeightCtx,
    expert_gpu_buffer: Option<ExpertBuffer>,
    pipeline_mode: EngineName,
    pub telemetry: Telemetry,
}

fn pipeline_mode_from_str(s: &str) -> PyResult<EngineName> {
    match s {
        "Cpu" | "CpuOnly" => Ok(EngineName::Cpu),
        "Gpu" => Ok(EngineName::Gpu),
        "FusedExp" => Ok(EngineName::FusedExp),
        "FusedWoods" => Ok(EngineName::FusedWoods),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown pipeline_mode: {}. Use Cpu|Gpu|FusedExp|FusedWoods", s
        ))),
    }
}

impl Engine {
    fn make_exec_ctx(&mut self) -> ExecCtxGpu<'_> {
        ExecCtxGpu {
            wf: &self.model.wf,
            ctx: &self.ctx,
            gpu_wf: &self.gpu_wf,
            config: &self.model.config,
            expert_fds: &self.model.expert_fds,
            expert_gpu_buffer: self.expert_gpu_buffer.as_mut(),
        }
    }
}

impl EngineTrait for Engine {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut CoreCache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, String> {
        let mode = self.pipeline_mode;
        match mode {
            EngineName::Cpu => {
                let mut engine = EngineCPU { model: &self.model };
                engine.forward(input_ids, cache, check_signal)
            }
            EngineName::Gpu => {
                let mut engine = EngineGPU {
                    model: &self.model,
                    ctx: &self.ctx,
                    gpu_wf: &self.gpu_wf,
                    expert_gpu_buffer: self.expert_gpu_buffer.as_mut(),
                };
                engine.forward(input_ids, cache, check_signal)
            }
            EngineName::FusedExp => {
                let mut engine = EngineFusedExp {
                    model: &self.model,
                    ctx: &self.ctx,
                    gpu_wf: &self.gpu_wf,
                    expert_gpu_buffer: self.expert_gpu_buffer.as_mut(),
                };
                engine.forward(input_ids, cache, check_signal)
            }
            EngineName::FusedWoods => {
                let mut engine = EngineFusedWoods {
                    model: &self.model,
                    ctx: &self.ctx,
                    gpu_wf: &self.gpu_wf,
                    expert_gpu_buffer: self.expert_gpu_buffer.as_mut(),
                };
                engine.forward(input_ids, cache, check_signal)
            }
        }
    }
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (model, pipeline_mode="FusedExp"))]
    fn new(model: &Model, pipeline_mode: &str) -> PyResult<Self> {
        let mode = pipeline_mode_from_str(pipeline_mode)?;
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
        let gpu_wf = GpuWeightCtx::new(&ctx.device, &model.inner.wf);

        eprintln!(
            "[engine] {} layers hidden={} experts={} mode={:?}",
            config.num_layers, config.hidden_dim, config.num_experts, mode
        );
        Ok(Engine {
            model: model.inner.clone(),
            ctx,
            gpu_wf,
            expert_gpu_buffer,
            pipeline_mode: mode,
            telemetry: Telemetry { prefill_ms: 0.0, total_ms: 0.0, tokens_generated: 0 },
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
        Ok(arr.into_py(py))
    }

    fn forward_debug(&mut self, py: Python<'_>, input_ids: &Bound<PyArray1<i64>>,
        cache: &mut Cache,
    ) -> PyResult<(PyObject, PyObject)> {
        let t0 = Instant::now();
        let ids = input_ids.readonly();
        let ids = ids.as_slice()?;
        let n = ids.len();
        let hd = self.model.config.hidden_dim;
        let vs = self.model.config.vocab_size;
        let num_layers = self.model.config.num_layers;

        let mut logits = vec![0.0f32; n * vs];
        if n == 0 {
            let arr = PyArray2::<f32>::from_owned_array(py,
                numpy::ndarray::Array2::from_shape_vec((n, vs), logits).unwrap());
            return Ok((arr.into_py(py), PyList::empty(py).into_py(py)));
        }

        let mut embed = vec![0.0f32; n * hd];
        let wf_ref = &self.model.wf;
        for (i, &id) in ids.iter().enumerate() {
            embed_lookup(wf_ref, id as usize, &mut embed[i * hd..(i + 1) * hd], hd);
        }

        let is_fused_exp = self.pipeline_mode == EngineName::FusedExp;
        let is_fused_woods = self.pipeline_mode == EngineName::FusedWoods;

        let mut hidden = vec![0.0f32; hd];
        let mut all_layer_outputs: Vec<Vec<f32>> = Vec::new();

        for (ti, _) in ids.iter().enumerate() {
            hidden.copy_from_slice(&embed[ti * hd..(ti + 1) * hd]);
            let mut layer_outputs = Vec::new();
            let mut exec = self.make_exec_ctx();
            if is_fused_exp {
                process_token_fusedexp_pipelined(
                    &mut exec, &mut hidden,
                    cache.inner.pos, &mut cache.inner.kv, &mut cache.inner.lin,
                    &mut || py.check_signals().is_err(),
                    true, &mut layer_outputs,
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            } else {
                process_token_inner(
                    &mut exec, &mut hidden,
                    cache.inner.pos, &mut cache.inner.kv, &mut cache.inner.lin,
                    &mut || py.check_signals().is_err(),
                    true, &mut layer_outputs,
                    is_fused_woods,
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            }
            cache.inner.pos += 1;
            all_layer_outputs.push(hidden.to_vec());
            all_layer_outputs.extend(layer_outputs);
            final_norm(exec.wf, &mut hidden, hd);
            gpu_lm_head(exec.wf, &hidden,
                &mut logits[ti * vs..(ti + 1) * vs],
                exec.gpu_wf, exec.ctx);
        }

        self.telemetry.prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let per_token_entries = 1 + num_layers;
        let last_token_start = (n - 1) * per_token_entries;
        let py_list = PyList::empty(py);
        for li in 0..num_layers {
            let layer_hidden = &all_layer_outputs[last_token_start + 1 + li];
            let arr = PyArray1::<f32>::from_owned_array(py,
                numpy::ndarray::Array1::from_vec(layer_hidden.clone()));
            py_list.append(arr)?;
        }

        let arr = PyArray2::<f32>::from_owned_array(py,
            numpy::ndarray::Array2::from_shape_vec((n, vs), logits).unwrap());
        Ok((arr.into_py(py), py_list.into_py(py)))
    }

    #[pyo3(signature = (input_ids, cache, max_tokens=256, temperature=0.0,
                        top_k=0, top_p=1.0, min_p=0.0, eos_token_ids=None))]
    fn generate(&mut self, py: Python<'_>, input_ids: &Bound<PyArray1<i64>>,
        cache: &mut Cache,
        max_tokens: usize, temperature: f32, top_k: usize, top_p: f32, min_p: f32,
        eos_token_ids: Option<&Bound<PyArray1<i64>>>,
    ) -> PyResult<PyObject> {
        let ids = input_ids.readonly();
        let ids = ids.as_slice()?;
        let start = if cache.inner.pos < ids.len() { cache.inner.pos } else { 0 };
        let new_ids = &ids[start..];
        let eos: HashSet<usize> = match eos_token_ids {
            Some(a) => a.readonly().to_vec()?.into_iter().map(|x| x as usize).collect(),
            None => [248046usize, 248044].into(),
        };

        let params = SampleParams {
            max_tokens, temperature, top_k, top_p, min_p, eos,
        };

        let mut telemetry = Telemetry { prefill_ms: 0.0, total_ms: 0.0, tokens_generated: 0 };
        let (tokens, _logits_last) = crate::generate::generate(
            self,
            new_ids,
            &mut cache.inner,
            &params,
            &mut || py.check_signals().is_err(),
            &mut telemetry,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        self.telemetry = telemetry;

        Ok(PyArray1::<i64>::from_vec(py, tokens).into_py(py))
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

        let hd = self.model.config.hidden_dim;
        let vs = self.model.config.vocab_size;
        let mut logits = ls[ls.len() - vs..].to_vec();

        let next = if temperature < 0.01 {
            logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap_or(0)
        } else { sample(&mut logits, temperature, top_k, top_p, min_p) };

        let is_fused_exp = self.pipeline_mode == EngineName::FusedExp;
        let is_fused_woods = self.pipeline_mode == EngineName::FusedWoods;

        let iter = StreamGenIterator {
            model_ptr: self as *mut Engine,
            cache_ptr: cache as *mut Cache,
            hd,
            vs,
            hidden: vec![0.0f32; hd],
            logits,
            next_token: next,
            remaining: max_tokens.saturating_sub(1),
            temperature,
            top_k,
            top_p,
            min_p,
            eos,
            gen_t0,
            tokens_generated: 0,
            done: false,
            telemetry_ptr: &mut self.telemetry as *mut Telemetry,
            is_fused_exp,
            is_fused_woods,
        };

        Ok(iter.into_py(py))
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
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> String {
        format!("Engine(loaded: {} layers, hidden={})",
            self.model.config.num_layers, self.model.config.hidden_dim)
    }
}

// ─── Streaming iterator ─────────────────────────────────────────────────────

#[pyclass(unsendable)]
pub struct StreamGenIterator {
    model_ptr: *mut Engine,
    cache_ptr: *mut Cache,
    hd: usize,
    vs: usize,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    next_token: usize,
    remaining: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    eos: HashSet<usize>,
    gen_t0: Instant,
    tokens_generated: usize,
    done: bool,
    telemetry_ptr: *mut Telemetry,
    is_fused_exp: bool,
    is_fused_woods: bool,
}

#[pymethods]
impl StreamGenIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<(i64, PyObject)>> {
        if self.done {
            return Ok(None);
        }

        let token = self.next_token as i64;
        let logits_obj = PyArray1::<f32>::from_vec(py, self.logits.clone()).into_py(py);
        self.tokens_generated += 1;

        if self.remaining == 0 || self.eos.contains(&self.next_token) {
            self.done = true;
            let t = unsafe { &mut *self.telemetry_ptr };
            t.total_ms = self.gen_t0.elapsed().as_secs_f64() * 1000.0;
            t.tokens_generated = self.tokens_generated;
            return Ok(Some((token, logits_obj)));
        }

        self.remaining -= 1;
        let eng = unsafe { &mut *self.model_ptr };
        let cache = unsafe { &mut *self.cache_ptr };

        {
            let model = &eng.model;
            embed_lookup(&model.wf, self.next_token, &mut self.hidden, self.hd);
        }

        {
            let mut exec = eng.make_exec_ctx();
            if self.is_fused_exp {
                process_token_fusedexp_pipelined(
                    &mut exec, &mut self.hidden,
                    cache.inner.pos, &mut cache.inner.kv, &mut cache.inner.lin,
                    &mut || py.check_signals().is_err(),
                    false, &mut Vec::new(),
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            } else {
                process_token_inner(
                    &mut exec, &mut self.hidden,
                    cache.inner.pos, &mut cache.inner.kv, &mut cache.inner.lin,
                    &mut || py.check_signals().is_err(),
                    false, &mut Vec::new(),
                    self.is_fused_woods,
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            }
        }
        cache.inner.pos += 1;

        {
            let model = &eng.model;
            final_norm(&model.wf, &mut self.hidden, self.hd);
            self.logits.fill(0.0);
            gpu_lm_head(&model.wf, &self.hidden, &mut self.logits,
                &eng.gpu_wf, &eng.ctx);
        }

        self.next_token = if self.temperature < 0.01 {
            self.logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap_or(0)
        } else {
            sample(&mut self.logits.clone(), self.temperature, self.top_k,
                self.top_p, self.min_p)
        };

        Ok(Some((token, logits_obj)))
    }
}

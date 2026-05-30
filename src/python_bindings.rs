/// Thin PyO3 bindings for the MoE-Infer inference engine.
use std::collections::BTreeMap;
use std::sync::Arc;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::cache::Cache as CoreCache;
use crate::model::Model as CoreModel;
use crate::error::MoEError;
use crate::engine::{SignalCheckFn, TelemetryValue, set_record_telemetry, DynEngine};
use crate::bq4::{BQ4Scheme, QwenVersion};
use crate::hf_util::HfRepo;
use crate::int4::Int4Scheme;

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
            self.inner.config.get_usize("num_hidden_layers").unwrap_or(0),
            self.inner.config.get_usize("hidden_size").unwrap_or(0))
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

    fn save(&self, bin_path: &str, json_path: &str) -> PyResult<()> {
        if self.inner.gpu_dirty {
            eprintln!(
                "[cache] WARNING: saving cache with gpu_dirty=true — GPU state \
                 hasn't been downloaded.  Call engine.download_cache(cache) first \
                 to capture mid-conversation K/V state."
            );
        }
        self.inner.save(std::path::Path::new(bin_path), std::path::Path::new(json_path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn load(bin_path: &str, json_path: &str) -> PyResult<Self> {
        CoreCache::load(std::path::Path::new(bin_path), std::path::Path::new(json_path))
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
    engine: DynEngine,
    model: Arc<CoreModel>,
    pub telemetry: BTreeMap<String, TelemetryValue>,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (model, pipeline_mode="Qwen35MoEFusedExp2", num_active_experts=0, *, expert_cache_count=0))]
    fn new(model: &Model, pipeline_mode: &str, num_active_experts: usize, expert_cache_count: usize) -> PyResult<Self> {
        let engine = DynEngine::new(pipeline_mode, model.inner.clone(), num_active_experts, expert_cache_count)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Engine {
            engine,
            model: model.inner.clone(),
            telemetry: BTreeMap::new(),
        })
    }

    /// Convert token IDs to embeddings. Returns [n, hidden_dim] float32 array.
    fn embed_lookup(&self, py: Python<'_>, input_ids: &Bound<PyArray1<i64>>) -> PyResult<PyObject> {
        let ids = input_ids.readonly();
        let ids = ids.as_slice()?;
        let n = ids.len();
        let hd = self.model.config.get_usize("hidden_size").unwrap();
        let mut embed = vec![0.0f32; n * hd];
        self.engine.embed_lookup(ids, &mut embed);
        let arr = PyArray2::<f32>::from_owned_array(py,
            numpy::ndarray::Array2::from_shape_vec((n, hd), embed)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("shape error: {}", e)))?);
        Ok(arr.into_pyobject(py)?.into_any().into())
    }

    /// Process pre-computed embeddings through the LM. Returns [n, vocab_size] logits.
    #[pyo3(signature = (embeddings, cache, *, mtp=false))]
    fn forward_hidden(&mut self, py: Python<'_>, embeddings: &Bound<PyArray2<f32>>,
        cache: &mut Cache,
        mtp: bool,
    ) -> PyResult<PyObject> {
        let emb = embeddings.readonly();
        let emb_slice = emb.as_slice()?;
        let shape = emb.shape();
        let n = shape[0];
        let vs = self.model.config.get_usize("vocab_size").unwrap();

        let logits = self.forward_hidden_impl(emb_slice, &mut cache.inner, &mut || py.check_signals().is_err(), mtp)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let arr = PyArray2::<f32>::from_owned_array(py,
            numpy::ndarray::Array2::from_shape_vec((n, vs), logits)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("shape error: {}", e)))?);
        Ok(arr.into_pyobject(py)?.into_any().into())
    }

    /// Embed *tokens* and run the LM on them, advancing the cache by
    /// ``tokens.len()`` positions.  Pass ONLY the new tokens to consume —
    /// not the full sequence so far.  The engine appends to its existing
    /// KV cache; previously processed tokens stay where they were.
    ///
    /// This is the fast path for text generation: it does embed lookup and
    /// forward in a single pyo3 call, saving a numpy allocation and a
    /// Python round-trip per token compared to ``embed_lookup`` +
    /// ``forward_hidden``.  Use ``forward_hidden`` only when the caller has
    /// to supply custom embeddings (e.g. spliced vision features).
    #[pyo3(signature = (tokens, cache, *, mtp=false))]
    fn forward(&mut self, py: Python<'_>, tokens: &Bound<PyArray1<i64>>,
        cache: &mut Cache,
        mtp: bool,
    ) -> PyResult<PyObject> {
        let toks = tokens.readonly();
        let toks = toks.as_slice()?;
        let n = toks.len();
        let hd = self.model.config.get_usize("hidden_size").unwrap();
        let vs = self.model.config.get_usize("vocab_size").unwrap();

        if n == 0 {
            let arr = PyArray2::<f32>::from_owned_array(py,
                numpy::ndarray::Array2::from_shape_vec((0, vs), Vec::new())
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("shape error: {}", e)))?);
            return Ok(arr.into_pyobject(py)?.into_any().into());
        }

        let mut embed = vec![0.0f32; n * hd];
        self.engine.embed_lookup(toks, &mut embed);

        let logits = self.forward_hidden_impl(&embed, &mut cache.inner, &mut || py.check_signals().is_err(), mtp)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let arr = PyArray2::<f32>::from_owned_array(py,
            numpy::ndarray::Array2::from_shape_vec((n, vs), logits)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("shape error: {}", e)))?);
        Ok(arr.into_pyobject(py)?.into_any().into())
    }

    /// Batched-prefill variant of forward. Same return shape, different
    /// internal loop strategy: layer-batched instead of token-serial.
    /// On engines that don't have a batched path yet (default trait impl),
    /// this just delegates to forward() — A/B-safe.
    #[pyo3(signature = (tokens, cache, *, mtp=false))]
    fn forward_batched(&mut self, py: Python<'_>, tokens: &Bound<PyArray1<i64>>,
        cache: &mut Cache,
        mtp: bool,
    ) -> PyResult<PyObject> {
        let toks = tokens.readonly();
        let toks = toks.as_slice()?;
        let n = toks.len();
        let hd = self.model.config.get_usize("hidden_size").unwrap();
        let vs = self.model.config.get_usize("vocab_size").unwrap();

        if n == 0 {
            let arr = PyArray2::<f32>::from_owned_array(py,
                numpy::ndarray::Array2::from_shape_vec((0, vs), Vec::new())
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                        format!("shape error: {}", e)))?);
            return Ok(arr.into_pyobject(py)?.into_any().into());
        }

        let mut embed = vec![0.0f32; n * hd];
        self.engine.embed_lookup(toks, &mut embed);

        if cache.inner.cpu_dirty {
            self.engine.upload_cache(&cache.inner);
            cache.inner.cpu_dirty = false;
        }
        let logits = self.engine.forward_hidden_batched(&embed, &mut || py.check_signals().is_err(), mtp)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        cache.inner.gpu_dirty = true;
        self.telemetry = self.engine.telemetry();

        let arr = PyArray2::<f32>::from_owned_array(py,
            numpy::ndarray::Array2::from_shape_vec((n, vs), logits)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("shape error: {}", e)))?);
        Ok(arr.into_pyobject(py)?.into_any().into())
    }

    /// Expose upload/download for callers that manage cache persistence.
    /// Hot-path forward_hidden no longer round-trips the cache, so callers
    /// that want to persist GPU state to disk must download_cache first.
    fn upload_cache(&self, cache: &mut Cache) {
        self.engine.upload_cache(&cache.inner);
        cache.inner.cpu_dirty = false;
    }

    fn download_cache(&self, cache: &mut Cache) {
        self.engine.download_cache(&mut cache.inner);
        cache.inner.gpu_dirty = false;
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

    /// H pre-norm from the last forward pass (needed by MTP draft).
    fn last_h_pre_norm(&self, py: Python<'_>) -> PyResult<PyObject> {
        let h = self.engine.last_h_pre_norm();
        let arr = PyArray1::<f32>::from_owned_array(py,
            numpy::ndarray::Array1::from_vec(h.to_vec()));
        Ok(arr.into_pyobject(py)?.into_any().into())
    }

    /// MTP draft step: (last_h_pre_norm, token_id) → logits.
    fn mtp_forward(&mut self, py: Python<'_>, token_id: i64) -> PyResult<PyObject> {
        let logits = self.engine.mtp_forward(token_id as usize);
        let vs = logits.len();
        let arr = PyArray1::<f32>::from_owned_array(py,
            numpy::ndarray::Array1::from_vec(logits));
        Ok(arr.into_pyobject(py)?.into_any().into())
    }

    /// Reset MTP KV cache (call when starting new sequence or after verification).
    fn mtp_reset(&mut self) {
        self.engine.mtp_reset();
    }

    /// Roll back MTP KV cache to a specific position (after partial accept).
    fn mtp_rollback(&mut self, pos: usize) {
        self.engine.mtp_rollback(pos);
    }

    fn __repr__(&self) -> String {
        format!("Engine(loaded: {} layers, hidden={})",
            self.model.config.get_usize("num_hidden_layers").unwrap_or(0),
            self.model.config.get_usize("hidden_size").unwrap_or(0))
    }
}

// ─── Quantize function ────────────────────────────────────────────────────────

/// Full quantization pipeline: HF safetensors → BQ4 format.
///
/// Reads HuggingFace BF16 safetensors, classifies each weight tensor with
/// BQ4 rules, quantizes, and writes ``model_weights.bin``,
/// ``model_weights.json``, and ``packed_experts/layer_XX.bin``.
#[pyfunction]
#[pyo3(signature = (model_path, output_dir, *, version, scheme="bq4"))]
pub fn qwen35_moe_quantize(
    model_path: &str,
    output_dir: &str,
    version: &str,
    scheme: &str,
) -> PyResult<()> {
    let qwen_version = match version {
        "3.5" => QwenVersion::V35,
        "3.6" => QwenVersion::V36,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown version: {}. Expected '3.5' or '3.6'.", version)
        )),
    };

    // Determine directory containing config.json (local or HF staging)
    let config_dir = if std::path::Path::new(model_path).is_dir() {
        std::path::PathBuf::from(model_path)
    } else {
        let repo = crate::hf_util::HfRepo::from_hf(model_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        repo.ensure("config.json")
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
        repo.path().to_path_buf()
    };

    let config_json = std::fs::read_to_string(config_dir.join("config.json"))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let arch = {
        let v: serde_json::Value = serde_json::from_str(&config_json)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        v["architectures"][0].as_str()
            .unwrap_or("Qwen3_5MoeForConditionalGeneration")
            .to_string()
    };

    if arch != "Qwen3_5MoeForConditionalGeneration" {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown architecture: {}", arch)
        ));
    };

    match scheme {
        "bq4" => {
            let s = BQ4Scheme::new(&config_dir, qwen_version)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            crate::quantize::run(model_path, output_dir, &s)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        }
        "int4" => {
            let s = Int4Scheme::new(&config_dir, qwen_version)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            crate::quantize::run(model_path, output_dir, &s)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown scheme: {}. Expected 'bq4' or 'int4'.", scheme)
        )),
    }
}

// ─── HfRepo (HF downloader exposed to Python) ────────────────────────────

#[pyclass(unsendable)]
pub struct PyHfRepo {
    repo: HfRepo,
}

#[pymethods]
impl PyHfRepo {
    #[new]
    #[pyo3(signature = (repo_id))]
    fn new(repo_id: &str) -> PyResult<Self> {
        HfRepo::from_hf(repo_id)
            .map(|repo| PyHfRepo { repo })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Download a file and return its local path.
    fn ensure(&self, filename: &str) -> PyResult<String> {
        self.repo.ensure(filename)
            .map(|p| p.to_string_lossy().to_string())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Download multiple files in parallel (Rust-side threading).
    /// Returns local paths in the same order.
    fn ensure_batch(&self, filenames: Vec<String>) -> PyResult<Vec<String>> {
        self.repo.ensure_batch(&filenames)
            .map(|paths| paths.iter().map(|p| p.to_string_lossy().to_string()).collect())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Get a file's expected size from HF (or local fs metadata).
    fn file_size(&self, filename: &str) -> PyResult<u64> {
        self.repo.file_size(filename)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Get (filename, size_bytes) for all files in the repo.
    fn file_sizes(&self) -> PyResult<Vec<(String, u64)>> {
        self.repo.file_sizes()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Delete a cached file.
    fn remove(&self, filename: &str) {
        self.repo.remove(filename)
    }

    /// List immediate children of *dir* (defaults to root).  Behaves like UNIX
    /// ``ls``: returns names of files and directories at that level.
    #[pyo3(signature = (dir=None))]
    fn ls(&self, dir: Option<&str>) -> PyResult<Vec<String>> {
        self.repo.ls(dir)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    #[getter]
    fn path(&self) -> String {
        self.repo.path().to_string_lossy().to_string()
    }

    #[getter]
    fn is_hf(&self) -> bool {
        self.repo.is_hf()
    }
}

// ─── Internal forward impl ─────────────────────────────────────────────────

impl Engine {
    fn forward_hidden_impl(
        &mut self,
        embeddings: &[f32],
        cache: &mut CoreCache,
        check_signal: SignalCheckFn<'_>,
        mtp: bool,
    ) -> Result<Vec<f32>, MoEError> {
        // Cache sync optimization:
        //   - Upload only when CPU has changes the GPU hasn't seen (cpu_dirty).
        //     In steady-state autoregressive generation this is always false,
        //     so we skip the K/V copy entirely.
        //   - Skip the full download — the GPU is the source of truth.
        //   - Skip syncing cache.pos here.  Nothing in the chat loop reads it
        //     between forwards; the only consumer (`Engine.forward`) calls
        //     `engine_pos()` directly.  cache.pos lags GPU until the next
        //     explicit `engine.download_cache(cache)` (or save() warning).
        if cache.cpu_dirty {
            self.engine.upload_cache(cache);
            cache.cpu_dirty = false;
        }
        let logits = self.engine.forward_hidden(embeddings, check_signal, mtp)?;
        cache.gpu_dirty = true;
        self.telemetry = self.engine.telemetry();
        Ok(logits)
    }
}

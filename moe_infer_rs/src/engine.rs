use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::cache::Cache;
use crate::engine::qwen35_moe::{Qwen35MoEFullModel, Qwen35MoEStrippedModel};
use qwen35_moe::{Qwen35MoEFused4bit, Qwen35MoEFused4bitExp1, Qwen35MoEFused4bitExp2, Qwen35MoEFused4bitExp3};

use crate::error::MoEError;
use crate::model::Model;

#[path = "engine/qwen35_moe.rs"]
pub mod qwen35_moe;

/// Signal check callback: returns true if processing should abort (e.g. Ctrl-C).
pub type SignalCheckFn<'a> = &'a mut dyn FnMut() -> bool;

/// Global toggle for engine-level telemetry recording.
static RECORD_TELEMETRY: AtomicBool = AtomicBool::new(false);

/// Enable or disable engine-level telemetry globally.
pub fn set_record_telemetry(on: bool) {
    RECORD_TELEMETRY.store(on, Ordering::Relaxed);
}

/// Check whether engine-level telemetry is enabled.
pub fn record_telemetry() -> bool {
    RECORD_TELEMETRY.load(Ordering::Relaxed)
}

/// A telemetry value: either a scalar or a list of per-invocation measurements.
#[derive(Clone)]
pub enum TelemetryValue {
    Scalar(f64),
    List(Vec<f64>),
}

pub trait Engine {
    /// Upload CPU cache → GPU buffers before forward. No-op if pos == 0.
    fn upload_cache(&self, cache: &Cache);
    /// Download GPU buffers → CPU cache after forward.
    fn download_cache(&self, cache: &mut Cache);

    /// Process `input_ids` through all layers. Returns logits [n, vocab_size].
    fn forward(
        &mut self,
        input_ids: &[i64],
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError>;

    /// Per-engine telemetry. Keys are like `engine.*`.
    /// Values can be scalars or per-invocation lists.
    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        BTreeMap::new()
    }
}

// ─── Engine type ──────────────────────────────────────────────────────────────

// ─── Type-erased engine ─────────────────────────────────────────────────────

/// Type-erased engine holding one of the engine variants via trait object.
pub struct DynEngine {
    inner: Box<dyn Engine>,
}

impl DynEngine {
    pub fn new(
        engine_type: &str,
        model: Arc<Model>,
        k: usize,
    ) -> Result<Self, MoEError> {
        let arch = model.config.resolve("architectures")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let inner: Box<dyn Engine> = match (engine_type, arch) {
            ("Qwen35MoEFused4bit", "Qwen3_5MoeForConditionalGeneration") =>
                Box::new(Qwen35MoEFused4bit::<Qwen35MoEFullModel>::new(model, k)?),
            ("Qwen35MoEFused4bit", "Qwen3_5MoeForConditionalGeneration_Stripped") =>
                Box::new(Qwen35MoEFused4bit::<Qwen35MoEStrippedModel>::new(model, k)?),
            ("Qwen35MoEFused4bitExp1", "Qwen3_5MoeForConditionalGeneration") =>
                Box::new(Qwen35MoEFused4bitExp1::<Qwen35MoEFullModel>::new(model, k)?),
            ("Qwen35MoEFused4bitExp1", "Qwen3_5MoeForConditionalGeneration_Stripped") =>
                Box::new(Qwen35MoEFused4bitExp1::<Qwen35MoEStrippedModel>::new(model, k)?),
            ("Qwen35MoEFused4bitExp2", "Qwen3_5MoeForConditionalGeneration") =>
                Box::new(Qwen35MoEFused4bitExp2::<Qwen35MoEFullModel>::new(model, k)?),
            ("Qwen35MoEFused4bitExp2", "Qwen3_5MoeForConditionalGeneration_Stripped") =>
                Box::new(Qwen35MoEFused4bitExp2::<Qwen35MoEStrippedModel>::new(model, k)?),
            ("Qwen35MoEFused4bitExp3", "Qwen3_5MoeForConditionalGeneration") =>
                Box::new(Qwen35MoEFused4bitExp3::<Qwen35MoEFullModel>::new(model, k)?),
            ("Qwen35MoEFused4bitExp3", "Qwen3_5MoeForConditionalGeneration_Stripped") =>
                Box::new(Qwen35MoEFused4bitExp3::<Qwen35MoEStrippedModel>::new(model, k)?),
            _ => return Err(MoEError::Config(format!(
                "Unknown engine: engine_type={:?}, arch={:?}", engine_type, arch
            ))),
        };
        Ok(DynEngine { inner })
    }

    pub fn upload_cache(&self, cache: &Cache) {
        self.inner.upload_cache(cache);
    }

    pub fn download_cache(&self, cache: &mut Cache) {
        self.inner.download_cache(cache);
    }

    pub fn forward(&mut self, input_ids: &[i64], check_signal: SignalCheckFn<'_>) -> Result<Vec<f32>, MoEError> {
        self.inner.forward(input_ids, check_signal)
    }

    pub fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        self.inner.telemetry()
    }
}

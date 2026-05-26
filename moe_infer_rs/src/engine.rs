use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::cache::Cache;
use crate::error::MoEError;
use crate::model::Model;

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

    /// Convert token IDs to embeddings. Writes into `embeddings` [n, hidden_dim].
    fn embed_lookup(&self, token_ids: &[i64], embeddings: &mut [f32]);

    /// Process pre-computed embeddings through all layers.
    /// `embeddings` shape: [n_tokens, hidden_dim]. Returns logits [n, vocab_size].
    fn forward_hidden(
        &mut self,
        embeddings: &[f32],
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError>;

    /// Per-engine telemetry. Keys are like `engine.*`.
    /// Values can be scalars or per-invocation lists.
    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        BTreeMap::new()
    }
}

// ─── Type-erased engine ─────────────────────────────────────────────────────
#[path = "engine/qwen35_moe/constants.rs"]
mod qwen35_constants;
#[path = "engine/qwen35_moe/cpu.rs"]
mod cpu;
#[path = "engine/qwen35_moe/fused_bq4_exp1.rs"]
mod fused_bq4_exp1;
#[path = "engine/qwen35_moe/fused_bq4_exp2.rs"]
mod fused_bq4_exp2;
#[path = "engine/qwen35_moe/metal_context.rs"]
pub mod metal_context;
#[path = "engine/qwen35_moe/metal_kernels.rs"]
mod metal_kernels;

use crate::engine::qwen35_constants::{FullModel, StrippedModel};
use crate::engine::fused_bq4_exp1::FusedBq4Exp1;
use crate::engine::fused_bq4_exp2::FusedBq4Exp2;

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
            ("Qwen35MoEBq4Exp1", "Qwen3_5MoeForConditionalGeneration") =>
                Box::new(FusedBq4Exp1::<FullModel>::new(model, k)?),
            ("Qwen35MoEBq4Exp1", "Qwen3_5MoeForConditionalGeneration_Stripped") =>
                Box::new(FusedBq4Exp1::<StrippedModel>::new(model, k)?),
            ("Qwen35MoEBq4Exp2", "Qwen3_5MoeForConditionalGeneration") =>
                Box::new(FusedBq4Exp2::<FullModel>::new(model, k)?),
            ("Qwen35MoEBq4Exp2", "Qwen3_5MoeForConditionalGeneration_Stripped") =>
                Box::new(FusedBq4Exp2::<StrippedModel>::new(model, k)?),
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

    pub fn embed_lookup(&self, token_ids: &[i64], embeddings: &mut [f32]) {
        self.inner.embed_lookup(token_ids, embeddings);
    }

    pub fn forward_hidden(&mut self, embeddings: &[f32], check_signal: SignalCheckFn<'_>) -> Result<Vec<f32>, MoEError> {
        self.inner.forward_hidden(embeddings, check_signal)
    }

    pub fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        self.inner.telemetry()
    }
}

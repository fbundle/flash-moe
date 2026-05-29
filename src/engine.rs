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
    /// Current engine-tracked sequence position (cheap — just reads a Cell).
    /// Used by the hot path to update CPU cache.pos without copying K/V data.
    fn engine_pos(&self) -> usize { 0 }

    /// Convert token IDs to embeddings. Writes into `embeddings` [n, hidden_dim].
    fn embed_lookup(&self, token_ids: &[i64], embeddings: &mut [f32]);

    /// Process pre-computed embeddings through all layers.
    /// `embeddings` shape: [n_tokens, hidden_dim]. Returns logits [n, vocab_size].
    fn forward_hidden(
        &mut self,
        embeddings: &[f32],
        check_signal: SignalCheckFn<'_>,
        mtp: bool,
    ) -> Result<Vec<f32>, MoEError>;

    /// H pre-norm from the last forward pass (before final norm + lm_head).
    /// Only populated by engines that support MTP.
    fn last_h_pre_norm(&self) -> &[f32] { &[] }

    /// MTP draft step: (last_h_pre_norm, token_id) → logits.
    /// Returns empty vec if MTP is not supported.
    fn mtp_forward(&mut self, _token_id: usize) -> Vec<f32> { Vec::new() }

    /// Reset MTP KV cache.
    fn mtp_reset(&mut self) {}

    /// Roll back MTP KV cache to a specific position.
    fn mtp_rollback(&mut self, _pos: usize) {}

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
#[path = "engine/qwen35_moe/fused_exp1.rs"]
mod fused_exp1;
#[path = "engine/qwen35_moe/fused_exp2.rs"]
mod fused_exp2;
#[path = "engine/qwen35_moe/metal_context.rs"]
pub mod metal_context;
#[path = "engine/qwen35_moe/metal_kernels.rs"]
mod metal_kernels;
#[path = "engine/qwen35_moe/mtp.rs"]
pub mod mtp;

use crate::engine::qwen35_constants::{FullModel, StrippedModel};
use crate::engine::fused_exp1::FusedExp1;
use crate::engine::fused_exp2::FusedExp2;

/// Type-erased engine holding one of the engine variants via trait object.
pub struct DynEngine {
    inner: Box<dyn Engine>,
}

impl DynEngine {
    pub fn new(
        engine_type: &str,
        model: Arc<Model>,
        k: usize,
        expert_cache: usize,
    ) -> Result<Self, MoEError> {
        let arch = model.config.resolve("architectures")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let inner: Box<dyn Engine> = match (engine_type, arch) {
            ("Qwen35MoEFusedExp1", "Qwen3_5MoeForConditionalGeneration") =>
                Box::new(FusedExp1::<FullModel>::new(model, k, expert_cache)?),
            ("Qwen35MoEFusedExp1", "Qwen3_5MoeForConditionalGeneration_Stripped") =>
                Box::new(FusedExp1::<StrippedModel>::new(model, k, expert_cache)?),
            ("Qwen35MoEFusedExp2", "Qwen3_5MoeForConditionalGeneration") =>
                Box::new(FusedExp2::<FullModel>::new(model, k, expert_cache)?),
            ("Qwen35MoEFusedExp2", "Qwen3_5MoeForConditionalGeneration_Stripped") =>
                Box::new(FusedExp2::<StrippedModel>::new(model, k, expert_cache)?),
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

    pub fn engine_pos(&self) -> usize {
        self.inner.engine_pos()
    }

    pub fn embed_lookup(&self, token_ids: &[i64], embeddings: &mut [f32]) {
        self.inner.embed_lookup(token_ids, embeddings);
    }

    pub fn forward_hidden(&mut self, embeddings: &[f32], check_signal: SignalCheckFn<'_>, mtp: bool) -> Result<Vec<f32>, MoEError> {
        self.inner.forward_hidden(embeddings, check_signal, mtp)
    }

    pub fn last_h_pre_norm(&self) -> &[f32] {
        self.inner.last_h_pre_norm()
    }

    pub fn mtp_forward(&mut self, token_id: usize) -> Vec<f32> {
        self.inner.mtp_forward(token_id)
    }

    pub fn mtp_reset(&mut self) {
        self.inner.mtp_reset()
    }

    pub fn mtp_rollback(&mut self, pos: usize) {
        self.inner.mtp_rollback(pos)
    }

    pub fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        self.inner.telemetry()
    }
}

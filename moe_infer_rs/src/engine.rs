use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::cache::Cache;
use crate::error::MoEError;

pub mod cpu;
pub mod qwen35_35b;
pub mod qwen35_35b_stripped;

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
    /// Process `input_ids` through all layers. Returns logits [n, vocab_size].
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, MoEError>;

    /// Per-engine telemetry. Keys are like `engine.*`.
    /// Values can be scalars or per-invocation lists.
    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> {
        BTreeMap::new()
    }
}

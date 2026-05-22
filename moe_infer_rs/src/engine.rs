use std::os::fd::RawFd;

use crate::cache::Cache;
use crate::metal_context::{ExpertBuffer, WeightBuffer, MetalContext};
use crate::model::config::ModelConfig;
use crate::model::weights::WeightFile;

pub mod cpu;
pub mod fusedexp;
pub mod fusedwoods;

/// Signal check callback: returns true if processing should abort (e.g. Ctrl-C).
pub type SignalCheckFn<'a> = &'a mut dyn FnMut() -> bool;

/// GPU execution context — includes Metal device, GPU weight buffers, and expert I/O.
pub struct ExecCtxGpu<'a> {
    pub wf: &'a WeightFile,
    pub ctx: &'a MetalContext,
    pub gpu_wf: &'a WeightBuffer,
    pub config: &'a ModelConfig,
    pub expert_fds: &'a [RawFd],
    pub expert_gpu_buffer: Option<&'a mut ExpertBuffer>,
}

pub trait Engine {
    /// Process `input_ids` through all layers. Returns logits [n, vocab_size].
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, String>;
}

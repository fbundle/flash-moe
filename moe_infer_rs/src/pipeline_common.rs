/// Shared types, constants, and CPU helpers used across pipeline modules.
use std::os::fd::RawFd;

use metal::{Buffer, CommandBuffer};

use crate::config::ModelConfig;
use crate::metal_context::{ExpertIOState, GpuWeightCtx, MetalContext};
use crate::quant::bf16_to_f32;
use crate::weights::WeightFile;

// ─── Constants ───────────────────────────────────────────────────────────

pub(crate) const MAX_SEQ: usize = 4096;
pub const RMS_NORM_EPS: f32 = 1e-6;
pub const FULL_ATTN_INTERVAL: usize = 4;
pub const GROUP_SIZE: usize = 64;
pub const LINEAR_KEY_DIM: usize = 128;
pub const LINEAR_VALUE_DIM: usize = 128;
pub const CONV_KERNEL_SIZE: usize = 4;

// ─── Pipeline mode ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    Cpu,
    Gpu,
    FusedExp,
    FusedWoods,
}

// ─── Execution context (borrowed view of Engine for pipeline fns) ────────

/// Borrowed execution context bundling model data + GPU state for inner fns.
pub struct ExecCtx<'a> {
    pub wf: &'a WeightFile,
    pub ctx: &'a MetalContext,
    pub gpu_wf: &'a GpuWeightCtx,
    pub config: &'a ModelConfig,
    pub expert_fds: &'a [RawFd],
    pub pipeline_mode: PipelineMode,
    pub expert_io: Option<&'a mut ExpertIOState>,
}

/// Signal check callback: returns true if processing should abort (e.g. Ctrl-C).
pub type SignalCheckFn<'a> = &'a mut dyn FnMut() -> bool;

// ─── CPU helper functions ────────────────────────────────────────────────

pub fn cpu_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn cpu_silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

pub fn cpu_softmax(x: &mut [f32]) {
    let max_val = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

pub fn cpu_topk(scores: &[f32], k: usize, indices: &mut [usize], values: &mut [f32]) {
    for (i, &score) in scores.iter().enumerate() {
        if i < k {
            let mut pos = i;
            while pos > 0 && values[(pos - 1) / 2] > score {
                values[pos] = values[(pos - 1) / 2];
                indices[pos] = indices[(pos - 1) / 2];
                pos = (pos - 1) / 2;
            }
            values[pos] = score;
            indices[pos] = i;
        } else if score > values[0] {
            values[0] = score;
            indices[0] = i;
            let mut pos = 0;
            loop {
                let left = 2 * pos + 1;
                let right = 2 * pos + 2;
                let mut smallest = pos;
                if left < k && values[left] < values[smallest] { smallest = left; }
                if right < k && values[right] < values[smallest] { smallest = right; }
                if smallest == pos { break; }
                values.swap(pos, smallest);
                indices.swap(pos, smallest);
                pos = smallest;
            }
        }
    }
}

pub fn cpu_normalize_weights(weights: &mut [f32]) {
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for w in weights.iter_mut() { *w *= inv; }
    }
}

pub fn cpu_rms_norm_bare(x: &[f32], out: &mut [f32], dim: usize, eps: f32) {
    let sum_sq: f32 = x[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms;
    }
}

pub fn cpu_rms_norm_gated(
    x: &[f32], z: &[f32], w_bf16: &[u16],
    out: &mut [f32], dim: usize, eps: f32,
) {
    let sum_sq: f32 = x[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        let w = bf16_to_f32(w_bf16[i]);
        let silu_z = z[i] / (1.0 + (-z[i]).exp());
        out[i] = x[i] * inv_rms * w * silu_z;
    }
}

pub fn cpu_conv1d_step(
    conv_state: &[f32],
    new_input: &[f32],
    weight_bf16: &[u16],
    out: &mut [f32],
    channels: usize,
    kernel_size: usize,
) {
    for c in 0..channels {
        let mut acc = 0.0f32;
        for k in 0..kernel_size - 1 {
            let w = bf16_to_f32(weight_bf16[c * kernel_size + k]);
            acc += conv_state[k * channels + c] * w;
        }
        let w = bf16_to_f32(weight_bf16[c * kernel_size + (kernel_size - 1)]);
        acc += new_input[c] * w;
        out[c] = acc;
    }
    cpu_silu(&mut out[..channels]);
}

// ─── Full attention KV cache ─────────────────────────────────────────────

pub struct FullAttnCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub len: usize,
}

impl FullAttnCache {
    pub fn new(max_seq: usize, kv_dim: usize) -> Self {
        FullAttnCache {
            k_cache: vec![0.0f32; max_seq * kv_dim],
            v_cache: vec![0.0f32; max_seq * kv_dim],
            len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

// ─── Linear attention state ──────────────────────────────────────────────

pub struct LinearAttnState {
    pub conv_state: Vec<f32>,
    pub ssm_state: Vec<f32>,
    pub ssm_state_gpu: Option<Buffer>,
}

impl LinearAttnState {
    pub fn new(num_v_heads: usize, key_dim: usize, value_dim: usize, qkv_dim: usize) -> Self {
        LinearAttnState {
            conv_state: vec![0.0f32; (CONV_KERNEL_SIZE - 1) * qkv_dim],
            ssm_state: vec![0.0f32; num_v_heads * value_dim * key_dim],
            ssm_state_gpu: None,
        }
    }
}

// ─── GPU state passed from full-attention forward to MoE for CMD2 fusion ─

pub struct FullAttnCmd2State {
    pub q_buf: Buffer,
    pub q_gate_buf: Buffer,
    pub kc_buf: Buffer,
    pub vc_buf: Buffer,
    pub scores_buf: Buffer,
    pub out_buf: Buffer,
    pub hidden_buf: Buffer,
    pub seq_len: u32,
    pub seq_stride: u32,
    pub num_attn_heads: u32,
    pub head_dim: u32,
    pub kv_dim: u32,
    pub heads_per_kv: u32,
    pub scale: f32,
    pub q_dim: u32,
    pub o_prefix: String,
}

// ─── GPU/CPU state from linear attention CMD1 for FusedWoods ─────────────────

pub struct LinearAttnFusedWoodsState {
    pub gated_buf: Buffer,
    pub h_mid: Vec<f32>,
    pub total_value: usize,
    pub o_prefix: String,
    pub post_norm_name: String,
}

// ─── Deferred expert results (CMD3 async dispatch) ───────────────────────

pub struct DeferredExperts {
    pub(crate) cmd_buf: Option<CommandBuffer>,
    pub(crate) out_buf: Option<Buffer>,
    pub(crate) _keep_alive: Vec<Buffer>,
    pub gpu_combined: bool,
}

impl DeferredExperts {
    pub fn new() -> Self {
        DeferredExperts {
            cmd_buf: None,
            out_buf: None,
            _keep_alive: Vec::new(),
            gpu_combined: false,
        }
    }

    pub fn is_active(&self) -> bool {
        self.cmd_buf.is_some()
    }

    pub fn complete(&mut self, hidden: &mut [f32], hidden_dim: usize) {
        if let Some(ref cmd_buf) = self.cmd_buf {
            cmd_buf.wait_until_completed();
        }
        if let Some(ref out_buf) = self.out_buf {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_buf.contents() as *const f32,
                    hidden.as_mut_ptr(),
                    hidden_dim,
                );
            }
        }
        self.cmd_buf = None;
        self.out_buf = None;
        self._keep_alive.clear();
    }

    pub fn complete_fast(&mut self, hidden: &mut [f32], hidden_dim: usize) {
        // Wait on CMD3's own command buffer for CPU cache coherence.
        // Even though a later CMD1 on the same serial queue has completed
        // (guaranteeing CMD3 finished first), Metal requires waiting on the
        // specific command buffer that wrote the data for CPU visibility.
        if let Some(ref cmd_buf) = self.cmd_buf {
            cmd_buf.wait_until_completed();
        }
        if let Some(ref out_buf) = self.out_buf {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_buf.contents() as *const f32,
                    hidden.as_mut_ptr(),
                    hidden_dim,
                );
            }
        }
        self.cmd_buf = None;
        self.out_buf = None;
        self._keep_alive.clear();
    }

    pub fn discard(&mut self) {
        self.cmd_buf = None;
        self.out_buf = None;
        self._keep_alive.clear();
    }
}

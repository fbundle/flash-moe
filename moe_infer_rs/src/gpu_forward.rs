/// GPU-accelerated MoE forward and linear attention (GatedDeltaNet).
///
/// Port of moe_forward, linear_attention_forward, and fused_layer_forward_debug
/// from moe_infer/core_src/layer_forward.h and attention.h.
use std::os::fd::RawFd;
use std::ffi::c_void;

use metal::{Buffer, MTLSize};
use crate::config::ModelConfig;
use crate::error::MoEError;
use crate::kernels;

const MAX_SEQ: usize = 4096;
use crate::metal_context::{metal_buf_shared, ExpertIOState, GpuWeightCtx, MetalContext, MAX_K};
use crate::quant::{bf16_to_f32, cpu_dequant_matvec_4bit, cpu_rms_norm};
use crate::weights::WeightFile;

const RMS_NORM_EPS: f32 = 1e-6;
const GROUP_SIZE: usize = 64;
pub const LINEAR_KEY_DIM: usize = 128;
pub const LINEAR_VALUE_DIM: usize = 128;
const CONV_KERNEL_SIZE: usize = 4;

/// Pipeline execution mode — controls how GPU command buffers are batched.
///
/// The C engine uses 3 command buffers per layer, split by CPU-side routing
/// which must happen between the gate projection and expert dispatch:
///
///   CMD1: attention projs → conv1d → SSM
///   CMD2: o_proj → residual → norm → routing gate
///   CPU:  softmax + top-K + expert I/O  (inherently serial)
///   CMD3: expert forward → combine → residual → norm
///
/// Since the CPU routing step is unavoidable, the minimum is 2 CMDs (pre-routing
/// and post-routing). The 3-CMD approach isolates attention from the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineMode {
    /// CPU-only path: all matvecs, activations, and routing on CPU (no Metal).
    CpuOnly,
    /// GPU path: individual Metal dispatches per operation (no command-buffer fusion).
    Gpu,
    /// Fused experiment: what we currently have — fused CMD1 (linear attention qkv/z/b/a),
    /// but MoE experts are dispatched individually (no CMD3 batching, no GPU combine).
    FusedExp,
    /// 3-CMD fused: CMD1 (attention) + CMD2 (o_proj/routing) + CMD3 (async experts + GPU combine).
    /// Matches the original C engine architecture: 2 sync points for all 40 layers.
    Fused3,
}

// ─── CPU helper functions ──────────────────────────────────────────────────

fn cpu_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn cpu_silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

fn cpu_softmax(x: &mut [f32]) {
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

fn cpu_topk(scores: &[f32], k: usize, indices: &mut [usize], values: &mut [f32]) {
    // Min-heap of K smallest
    for (i, &score) in scores.iter().enumerate() {
        if i < k {
            // Insert into heap
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

fn cpu_normalize_weights(weights: &mut [f32]) {
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for w in weights.iter_mut() { *w *= inv; }
    }
}

fn cpu_rms_norm_bare(x: &[f32], out: &mut [f32], dim: usize, eps: f32) {
    let sum_sq: f32 = x[..dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
    for i in 0..dim {
        out[i] = x[i] * inv_rms;
    }
}

fn cpu_rms_norm_gated(
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

fn cpu_conv1d_step(
    conv_state: &[f32],   // [(kernel_size-1) * channels]
    new_input: &[f32],    // [channels]
    weight_bf16: &[u16],  // [channels * kernel_size]
    out: &mut [f32],      // [channels]
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

// ─── Full attention KV cache ──────────────────────────────────────────────

/// CPU-side KV cache for a full-attention layer.
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

// ─── Linear attention state ────────────────────────────────────────────────

pub struct LinearAttnState {
    /// Conv1d state: [(kernel_size-1) * qkv_dim] ring buffer (CPU)
    pub conv_state: Vec<f32>,
    /// SSM state: [num_v_heads * value_dim * key_dim] — the S matrix per v-head
    pub ssm_state: Vec<f32>,
    /// GPU persistent SSM state buffer (created lazily)
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

// ─── RoPE ─────────────────────────────────────────────────────────────────

fn apply_rope(
    q: &mut [f32], k: &mut [f32], pos: usize,
    num_q_heads: usize, num_kv_heads: usize,
    head_dim: usize, rotary_dim: usize, rope_theta: f64,
) {
    let pos_f = pos as f32;
    for h in 0..num_q_heads {
        let qh = &mut q[h * head_dim..];
        for d in (0..rotary_dim).step_by(2) {
            let theta = pos_f as f64 * rope_theta.powf(-2.0 * (d as f64) / rotary_dim as f64);
            let cos = theta.cos() as f32;
            let sin = theta.sin() as f32;
            let (q0, q1) = (qh[d], qh[d + 1]);
            qh[d] = q0 * cos - q1 * sin;
            qh[d + 1] = q0 * sin + q1 * cos;
        }
    }
    for h in 0..num_kv_heads {
        let kh = &mut k[h * head_dim..];
        for d in (0..rotary_dim).step_by(2) {
            let theta = pos_f as f64 * rope_theta.powf(-2.0 * (d as f64) / rotary_dim as f64);
            let cos = theta.cos() as f32;
            let sin = theta.sin() as f32;
            let (k0, k1) = (kh[d], kh[d + 1]);
            kh[d] = k0 * cos - k1 * sin;
            kh[d + 1] = k0 * sin + k1 * cos;
        }
    }
}

// ─── GPU state passed from full-attention forward to MoE for CMD2 fusion ──

/// GPU-resident attention state for fusing batched attention + o_proj + gate
/// into a single CMD2 (matching the C engine architecture).
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

/// GPU/CPU state from linear attention CMD1 for Fused3 (matching C engine exactly).
///
/// C does gated_norm on CPU after CMD1, then out_proj + residual + norm + gate
/// in CMD2. This struct carries the gated attention output and pre-attention
/// hidden state (h_mid) into moe_layer_forward's CMD2.
pub struct LinearAttnFused3State {
    pub gated_out: Vec<f32>,       // gated_norm output (total_value floats, CPU)
    pub h_mid: Vec<f32>,           // pre-attention hidden (hidden_dim floats)
    pub total_value: usize,
    pub o_prefix: String,
    pub post_norm_name: String,
}

// ─── Full attention forward ───────────────────────────────────────────────

/// Single-token full (self) attention forward: QKV proj, Q/K norms, RoPE,
/// KV cache append.
///
/// Returns `FullAttnCmd2State` for GPU-fused CMD2 (batched attn + o_proj + gate).
/// When GPU is available, skips batched attention/o_proj/residual — those are
/// deferred to `moe_layer_forward`'s CMD2. When GPU unavailable, computes
/// everything on CPU/separate CMDs and returns None.
pub fn full_attention_forward(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    kv: &mut FullAttnCache,
    pos: usize,
    config: &ModelConfig,
    gpu_wf: Option<&GpuWeightCtx>,
    ctx: Option<&MetalContext>,
    mode: PipelineMode,
) -> Option<FullAttnCmd2State> {
    let hidden_dim = config.hidden_dim;
    let num_attn_heads = config.num_attn_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let rotary_dim = config.rotary_dim;
    let rope_theta = config.rope_theta;

    let q_proj_dim = num_attn_heads * head_dim * 2;
    let q_dim = num_attn_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Input RMS norm
    let norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
    let nw = wf.get_tensor_u16(&norm_name);
    let mut normed = vec![0.0f32; hidden_dim];
    if let Some(nw) = nw {
        let nw_f32: Vec<f32> = nw.iter().map(|&v| bf16_to_f32(v)).collect();
        cpu_rms_norm(hidden, &nw_f32, &mut normed, hidden_dim, RMS_NORM_EPS);
    } else {
        normed.copy_from_slice(hidden);
    }

    // QKV projections (GPU)
    let mut q_proj_out = vec![0.0f32; q_proj_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];
    if let (Some(gw), Some(c)) = (gpu_wf, ctx) {
        let x_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        unsafe { let dst = x_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(normed.as_ptr(), dst, hidden_dim); }
        let qbuf = metal_buf_shared(&c.device, q_proj_dim * 4);
        let kbuf = metal_buf_shared(&c.device, kv_dim * 4);
        let vbuf = metal_buf_shared(&c.device, kv_dim * 4);
        let cm = c.queue.new_command_buffer();
        let enc = cm.new_compute_command_encoder();
        let q_name = format!("model.layers.{}.self_attn.q_proj", layer_idx);
        let k_name = format!("model.layers.{}.self_attn.k_proj", layer_idx);
        let v_name = format!("model.layers.{}.self_attn.v_proj", layer_idx);
        gw.encode_matvec_into(wf, c, &enc, &q_name, &x_buf, 0, &qbuf, 0, q_proj_dim, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &k_name, &x_buf, 0, &kbuf, 0, kv_dim, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &v_name, &x_buf, 0, &vbuf, 0, kv_dim, hidden_dim);
        enc.end_encoding(); cm.commit(); cm.wait_until_completed();
        unsafe {
            std::ptr::copy_nonoverlapping(qbuf.contents() as *const f32, q_proj_out.as_mut_ptr(), q_proj_dim);
            std::ptr::copy_nonoverlapping(kbuf.contents() as *const f32, k.as_mut_ptr(), kv_dim);
            std::ptr::copy_nonoverlapping(vbuf.contents() as *const f32, v.as_mut_ptr(), kv_dim);
        }
    }

    // Split Q and Q-gate from concatenated output
    let mut q = vec![0.0f32; q_dim];
    let mut q_gate = vec![0.0f32; q_dim];
    for h in 0..num_attn_heads {
        let src = &q_proj_out[h * 2 * head_dim..];
        q[h * head_dim..h * head_dim + head_dim].copy_from_slice(&src[..head_dim]);
        q_gate[h * head_dim..h * head_dim + head_dim].copy_from_slice(&src[head_dim..2 * head_dim]);
    }

    // Q/K norms
    let qn_name = format!("model.layers.{}.self_attn.q_norm.weight", layer_idx);
    let kn_name = format!("model.layers.{}.self_attn.k_norm.weight", layer_idx);
    if let Some(qnw) = wf.get_tensor_u16(&qn_name) {
        for h in 0..num_attn_heads {
            let qh = &mut q[h * head_dim..(h + 1) * head_dim];
            let sum_sq: f32 = qh.iter().map(|&x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / head_dim as f32 + RMS_NORM_EPS).sqrt();
            for i in 0..head_dim.min(qnw.len()) { qh[i] = qh[i] * inv_rms * bf16_to_f32(qnw[i]); }
        }
    }
    if let Some(knw) = wf.get_tensor_u16(&kn_name) {
        for h in 0..num_kv_heads {
            let kh = &mut k[h * head_dim..(h + 1) * head_dim];
            let sum_sq: f32 = kh.iter().map(|&x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / head_dim as f32 + RMS_NORM_EPS).sqrt();
            for i in 0..head_dim.min(knw.len()) { kh[i] = kh[i] * inv_rms * bf16_to_f32(knw[i]); }
        }
    }

    // RoPE
    apply_rope(&mut q, &mut k, pos, num_attn_heads, num_kv_heads, head_dim, rotary_dim, rope_theta);

    // Append K, V to cache
    let cache_pos = kv.len;
    kv.k_cache[cache_pos * kv_dim..(cache_pos + 1) * kv_dim].copy_from_slice(&k);
    kv.v_cache[cache_pos * kv_dim..(cache_pos + 1) * kv_dim].copy_from_slice(&v);
    kv.len += 1;

    // GPU batched attention (scores + softmax + values + sigmoid gate)
    let heads_per_kv = num_attn_heads / num_kv_heads;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let seq_len = kv.len;
    let seq_stride = MAX_SEQ;
    // o_proj output (filled by GPU fused path or CPU fallback below)
    let mut o_out = vec![0.0f32; hidden_dim];

    let use_gpu_attn = mode != PipelineMode::CpuOnly
        && ctx.is_some()
        && gpu_wf.is_some()
        && ctx.unwrap().attn_scores_batched.is_some()
        && ctx.unwrap().attn_softmax_batched.is_some()
        && ctx.unwrap().attn_values_batched.is_some();

    if use_gpu_attn {
        let c = ctx.unwrap();
        // Upload Q, K_cache, V_cache, Q_gate, hidden → returned for CMD2 fusion
        let q_buf = metal_buf_shared(&c.device, q_dim * 4);
        let kc_buf = metal_buf_shared(&c.device, seq_stride * kv_dim * 4);
        let vc_buf = metal_buf_shared(&c.device, seq_stride * kv_dim * 4);
        let scores_buf = metal_buf_shared(&c.device, num_attn_heads * seq_stride * 4);
        let out_buf = metal_buf_shared(&c.device, q_dim * 4);
        let q_gate_buf = metal_buf_shared(&c.device, q_dim * 4);
        let hidden_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(q.as_ptr(), q_buf.contents() as *mut f32, q_dim);
            std::ptr::copy_nonoverlapping(kv.k_cache.as_ptr(), kc_buf.contents() as *mut f32, seq_len * kv_dim);
            std::ptr::copy_nonoverlapping(kv.v_cache.as_ptr(), vc_buf.contents() as *mut f32, seq_len * kv_dim);
            std::ptr::copy_nonoverlapping(q_gate.as_ptr(), q_gate_buf.contents() as *mut f32, q_dim);
            std::ptr::copy_nonoverlapping(hidden.as_ptr(), hidden_buf.contents() as *mut f32, hidden_dim);
        }

        let o_prefix = format!("model.layers.{}.self_attn.o_proj", layer_idx);
        return Some(FullAttnCmd2State {
            q_buf, q_gate_buf, kc_buf, vc_buf, scores_buf, out_buf, hidden_buf,
            seq_len: seq_len as u32,
            seq_stride: seq_stride as u32,
            num_attn_heads: num_attn_heads as u32,
            head_dim: head_dim as u32,
            kv_dim: kv_dim as u32,
            heads_per_kv: heads_per_kv as u32,
            scale,
            q_dim: q_dim as u32,
            o_prefix,
        });
    }
    // CPU fallback — no GPU attention available, do everything on CPU
    {
        let mut attn_out = vec![0.0f32; q_dim];
        for h in 0..num_attn_heads {
            let kv_h = h / heads_per_kv;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = vec![0.0f32; seq_len];
            for p in 0..seq_len {
                let kp = &kv.k_cache[p * kv_dim + kv_h * head_dim..p * kv_dim + (kv_h + 1) * head_dim];
                scores[p] = qh.iter().zip(kp.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale;
            }
            let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = scores.iter().map(|&s| (s - max_val).exp()).sum();
            let inv_sum = 1.0 / sum;
            let oh = &mut attn_out[h * head_dim..(h + 1) * head_dim];
            for p in 0..seq_len {
                let weight = (scores[p] - max_val).exp() * inv_sum;
                let vp = &kv.v_cache[p * kv_dim + kv_h * head_dim..p * kv_dim + (kv_h + 1) * head_dim];
                for d in 0..head_dim { oh[d] += weight * vp[d]; }
            }
        }
        for i in 0..q_dim { attn_out[i] *= 1.0f32 / (1.0f32 + (-q_gate[i]).exp()); }

        // o_proj
        let o_prefix = format!("model.layers.{}.self_attn.o_proj", layer_idx);
        if mode != PipelineMode::CpuOnly {
            if let (Some(gw), Some(c)) = (gpu_wf, ctx) {
                let attn_buf = metal_buf_shared(&c.device, q_dim * 4);
                unsafe { let dst = attn_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(attn_out.as_ptr(), dst, q_dim); }
                let buf = metal_buf_shared(&c.device, hidden_dim * 4);
                let cm = c.queue.new_command_buffer();
                let enc = cm.new_compute_command_encoder();
                gw.encode_matvec_into(wf, c, &enc, &o_prefix, &attn_buf, 0, &buf, 0, hidden_dim, q_dim);
                enc.end_encoding(); cm.commit(); cm.wait_until_completed();
                unsafe { std::ptr::copy_nonoverlapping(buf.contents() as *const f32, o_out.as_mut_ptr(), hidden_dim); }
            }
        } else {
            if let (Some(ow), Some(os), Some(ob)) = (
                wf.get_tensor_u32(&format!("{}.weight", o_prefix)),
                wf.get_tensor_u16(&format!("{}.scales", o_prefix)),
                wf.get_tensor_u16(&format!("{}.biases", o_prefix)),
            ) { cpu_dequant_matvec_4bit(ow, os, ob, &attn_out, &mut o_out, hidden_dim, q_dim, GROUP_SIZE); }
        }
    }

    // Residual add
    for i in 0..hidden_dim { hidden[i] += o_out[i]; }
    None
}

// ─── Deferred expert results (CMD3 async dispatch) ─────────────────────────

/// Holds a committed but not-yet-completed CMD3 (expert forward + GPU combine).
///
/// In Fused3 mode, CMD3 is committed without `wait_until_completed()`, then
/// results are collected at the start of the *next* layer — overlapping GPU work.
pub struct DeferredExperts {
    /// Committed command buffer (CMD3). Wait on next layer.
    cmd_buf: Option<metal::CommandBuffer>,
    /// moe_combine_residual output — final hidden state for this layer.
    out_buf: Option<metal::Buffer>,
    /// All intermediate GPU buffers kept alive until CMD completes.
    _keep_alive: Vec<metal::Buffer>,
}

impl DeferredExperts {
    pub fn new() -> Self {
        DeferredExperts {
            cmd_buf: None,
            out_buf: None,
            _keep_alive: Vec::new(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.cmd_buf.is_some()
    }

    /// Complete deferred CMD3: wait for GPU, copy result into hidden.
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

    /// Discard deferred results without CPU readback.
    pub fn discard(&mut self) {
        self.cmd_buf = None;
        self.out_buf = None;
        self._keep_alive.clear();
    }
}

// ─── Linear attention forward (GatedDeltaNet) ─────────────────────────────

/// Full linear attention forward (GatedDeltaNet) for single-token incremental inference.
/// Port of fused_layer_forward from layer_forward.h (CMD1 linear attention pipeline).
pub fn linear_attention_forward(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    state: &mut LinearAttnState,
    hidden_dim: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    total_key: usize,
    total_value: usize,
    qkv_dim: usize,
    gpu_wf: Option<&GpuWeightCtx>,
    ctx: Option<&MetalContext>,
    linear_idx: usize,  // index into persistent GPU state buffers
    mode: PipelineMode,
) -> Option<LinearAttnFused3State> {
    let use_gpu = mode != PipelineMode::CpuOnly
        && gpu_wf.is_some()
        && ctx.is_some();

    // Input RMS norm
    let norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
    let nw = wf.get_tensor_u16(&norm_name);
    let mut normed = vec![0.0f32; hidden_dim];
    let mut residual = vec![0.0f32; hidden_dim];
    residual.copy_from_slice(hidden);

    if let Some(nw) = nw {
        let nw_f32: Vec<f32> = nw.iter().map(|&v| bf16_to_f32(v)).collect();
        cpu_rms_norm(hidden, &nw_f32, &mut normed, hidden_dim, RMS_NORM_EPS);
    } else {
        normed.copy_from_slice(hidden);
    }

    // Batch projections: QKV, Z, B, A
    let mut qkv = vec![0.0f32; qkv_dim];
    let mut z = vec![0.0f32; total_value];
    let mut beta = vec![0.0f32; num_v_heads];
    let mut alpha = vec![0.0f32; num_v_heads];

    let prefix = format!("model.layers.{}.linear_attn", layer_idx);

    let key_dim = total_key / num_k_heads;
    let value_dim = total_value / num_v_heads;
    let inv_scale = 1.0 / (key_dim as f32).sqrt();
    let k_heads_per_v = num_v_heads / num_k_heads;

    let debug_this_layer = layer_idx == 0;

    let mut gated_out = vec![0.0f32; total_value];

    // ── Fused GPU path (CMD1): attention projections + conv1d + SSM in ONE command buffer ──
    let gpu_compatible = key_dim == 128 && value_dim == 128 && use_gpu;
    let use_fused_gpu = (mode == PipelineMode::FusedExp)
        && gpu_compatible
        && ctx.is_some()
        && ctx.unwrap().buf_conv_output.is_some()
        && linear_idx < ctx.unwrap().buf_conv_state.len()
        && linear_idx < ctx.unwrap().buf_delta_state.len()
        && ctx.unwrap().batch_out.len() >= 4
        && ctx.unwrap().residual_add.is_some();

    if use_fused_gpu {
        let c = ctx.unwrap();
        let gw = gpu_wf.unwrap();
        let prefix_std = format!("{}.in_proj_qkv", prefix);
        let prefix_z = format!("{}.in_proj_z", prefix);
        let prefix_b = format!("{}.in_proj_b", prefix);
        let prefix_a = format!("{}.in_proj_a", prefix);

        // Upload normed input + residual once
        let x_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let residual_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        unsafe {
            let dst = x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(normed.as_ptr(), dst, hidden_dim);
            let dst_r = residual_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(residual.as_ptr(), dst_r, hidden_dim);
        }

        // CMD1: Single command buffer — attention projs + full linear attn pipeline
        let cmd_buf = c.queue.new_command_buffer();

        // ── Encoder 1: 4 attention projections → batch_out[0..3] ──
        {
            let enc = cmd_buf.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, &prefix_std, &x_buf, 0, &c.batch_out[0], 0, qkv_dim, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_z, &x_buf, 0, &c.batch_out[1], 0, total_value, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_b, &x_buf, 0, &c.batch_out[2], 0, num_v_heads, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_a, &x_buf, 0, &c.batch_out[3], 0, num_v_heads, hidden_dim);
            enc.end_encoding();
        }

        // ── Encoder 2: conv1d_step (reads qkv from batch_out[0], writes buf_conv_output, updates buf_conv_state) ──
        if let Some(conv_w_ptr) = wf.get_tensor_ptr(&format!("{}.conv1d.weight", prefix)) {
            let conv_w_off = (conv_w_ptr as usize - gw.base as usize) as u64;
            let enc = cmd_buf.new_compute_command_encoder();
            kernels::encode_conv1d_step(c, &enc,
                &c.buf_conv_state[linear_idx],      // persistent conv state
                &c.batch_out[0],                     // input = QKV projection
                &gw.buf, conv_w_off,                 // weights from wf_buf with offset
                c.buf_conv_output.as_ref().unwrap(),  // output
                qkv_dim as u32);
            enc.end_encoding();
        }

        // ── Encoder 3: rms_norm_qk (reads q/k from buf_conv_output at offsets) ──
        {
            let enc = cmd_buf.new_compute_command_encoder();
            kernels::encode_rms_norm_qk(c, &enc,
                c.buf_conv_output.as_ref().unwrap(), 0,                             // q at offset 0
                c.buf_conv_output.as_ref().unwrap(), (total_key * 4) as u64,         // k at offset total_key*f32
                num_k_heads as u32, key_dim as u32, inv_scale);
            enc.end_encoding();
        }

        // ── Encoder 4: compute_decay_beta (reads alpha/beta from batch_out[3]/[2], A_log/dt_bias from wf_buf) ──
        {
            let a_log_ptr = wf.get_tensor_ptr(&format!("{}.A_log", prefix));
            let dt_bias_ptr = wf.get_tensor_ptr(&format!("{}.dt_bias", prefix));
            let a_log_off = a_log_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
            let dt_bias_off = dt_bias_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
            let enc = cmd_buf.new_compute_command_encoder();
            kernels::encode_compute_decay_beta(c, &enc,
                &c.batch_out[3],                             // alpha
                &c.batch_out[2],                             // beta
                if a_log_ptr.is_some() { &gw.buf } else { &c.batch_out[3] }, a_log_off,   // A_log (or dummy)
                if dt_bias_ptr.is_some() { &gw.buf } else { &c.batch_out[2] }, dt_bias_off, // dt_bias (or dummy)
                c.buf_delta_g_decay.as_ref().unwrap(),
                c.buf_delta_beta.as_ref().unwrap(),
                num_v_heads as u32);
            enc.end_encoding();
        }

        // ── Encoder 5: gated_delta_net_step (reads q/k/v from buf_conv_output at offsets, updates buf_delta_state) ──
        {
            let q_off = 0u64;
            let k_off = (total_key * 4) as u64;
            let v_off = (2 * total_key * 4) as u64;
            let conv_out = c.buf_conv_output.as_ref().unwrap();
            let enc = cmd_buf.new_compute_command_encoder();
            kernels::encode_gated_delta_net_step(c, &enc,
                &c.buf_delta_state[linear_idx],   // persistent SSM state
                conv_out, q_off,                   // q at offset 0
                conv_out, k_off,                   // k at offset total_key*4
                conv_out, v_off,                   // v at offset 2*total_key*4
                c.buf_delta_g_decay.as_ref().unwrap(),
                c.buf_delta_beta.as_ref().unwrap(),
                c.buf_delta_output.as_ref().unwrap(),
                num_v_heads as u32, k_heads_per_v as u32, key_dim as u32, value_dim as u32);
            enc.end_encoding();
        }

        // ── Encoder 6: gated_rms_norm (reads buf_delta_output, z from batch_out[1], weight from wf_buf) ──
        let gated_gpu = metal_buf_shared(&c.device, total_value * 4);
        {
            let gnw_ptr = wf.get_tensor_ptr(&format!("{}.norm.weight", prefix));
            let enc = cmd_buf.new_compute_command_encoder();
            if let Some(gnw_p) = gnw_ptr {
                let gnw_off = (gnw_p as usize - gw.base as usize) as u64;
                kernels::encode_gated_rms_norm(c, &enc,
                    c.buf_delta_output.as_ref().unwrap(),
                    &c.batch_out[1],                // z
                    &gw.buf, gnw_off,               // norm weight from wf_buf
                    &gated_gpu,
                    num_v_heads as u32, value_dim as u32);
            }
            enc.end_encoding();
        }

        // ── Encoder 7: out_proj matvec (gated_out → hidden_dim) ──
        let o_proj_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        {
            let enc = cmd_buf.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.out_proj", prefix),
                &gated_gpu, 0, &o_proj_buf, 0, hidden_dim, total_value);
            enc.end_encoding();
        }

        // ── Encoder 8: residual_add (o_proj_out + residual → hidden_out) ──
        let hidden_out = metal_buf_shared(&c.device, hidden_dim * 4);
        {
            let enc = cmd_buf.new_compute_command_encoder();
            let pipe = c.residual_add.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&o_proj_buf), 0);
            enc.set_buffer(1, Some(&residual_buf), 0);
            enc.set_buffer(2, Some(&hidden_out), 0);
            unsafe { enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void); }
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
        }

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read final hidden (already has residual + attn_out)
        unsafe {
            std::ptr::copy_nonoverlapping(hidden_out.contents() as *const f32,
                hidden.as_mut_ptr(), hidden_dim);
            // Also keep gated_out for potential fallback use
            std::ptr::copy_nonoverlapping(gated_gpu.contents() as *const f32,
                gated_out.as_mut_ptr(), total_value);
        }

        // Update CPU conv_state for non-fused fallback / debugging
        let state_off = (CONV_KERNEL_SIZE - 2) * qkv_dim;
        state.conv_state.copy_within(qkv_dim.., 0);
        state.conv_state[state_off..state_off + qkv_dim].fill(0.0);

        // Skip separate out_proj + residual (already done in CMD1)
        return None;
    }

    // ── Fused3 path (matching C exactly): CMD1 without gated_norm/out_proj/residual ──
    // C does: CMD1(qkvz+ba+conv1d+SSM) → CPU(gated_norm) → CMD2(out_proj+residual+norm+gate+shared)
    let use_fused3_gpu = mode == PipelineMode::Fused3
        && gpu_compatible
        && ctx.is_some()
        && ctx.unwrap().buf_conv_output.is_some()
        && linear_idx < ctx.unwrap().buf_conv_state.len()
        && linear_idx < ctx.unwrap().buf_delta_state.len()
        && ctx.unwrap().batch_out.len() >= 4;

    if use_fused3_gpu {
        let c = ctx.unwrap();
        let gw = gpu_wf.unwrap();
        let prefix_std = format!("{}.in_proj_qkv", prefix);
        let prefix_z = format!("{}.in_proj_z", prefix);
        let prefix_b = format!("{}.in_proj_b", prefix);
        let prefix_a = format!("{}.in_proj_a", prefix);

        let x_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        unsafe { let dst = x_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(normed.as_ptr(), dst, hidden_dim); }

        // CMD1: encoders 1-5 only (projections + conv1d + rms_norm_qk + decay_beta + SSM)
        let cmd_buf = c.queue.new_command_buffer();

        // Encoder 1: 4 attention projections
        {
            let enc = cmd_buf.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, &prefix_std, &x_buf, 0, &c.batch_out[0], 0, qkv_dim, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_z, &x_buf, 0, &c.batch_out[1], 0, total_value, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_b, &x_buf, 0, &c.batch_out[2], 0, num_v_heads, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &prefix_a, &x_buf, 0, &c.batch_out[3], 0, num_v_heads, hidden_dim);
            enc.end_encoding();
        }

        // Encoder 2: conv1d_step
        if let Some(conv_w_ptr) = wf.get_tensor_ptr(&format!("{}.conv1d.weight", prefix)) {
            let conv_w_off = (conv_w_ptr as usize - gw.base as usize) as u64;
            let enc = cmd_buf.new_compute_command_encoder();
            kernels::encode_conv1d_step(c, &enc,
                &c.buf_conv_state[linear_idx],
                &c.batch_out[0],
                &gw.buf, conv_w_off,
                c.buf_conv_output.as_ref().unwrap(),
                qkv_dim as u32);
            enc.end_encoding();
        }

        // Encoder 3: rms_norm_qk
        {
            let enc = cmd_buf.new_compute_command_encoder();
            kernels::encode_rms_norm_qk(c, &enc,
                c.buf_conv_output.as_ref().unwrap(), 0,
                c.buf_conv_output.as_ref().unwrap(), (total_key * 4) as u64,
                num_k_heads as u32, key_dim as u32, inv_scale);
            enc.end_encoding();
        }

        // Encoder 4: compute_decay_beta
        {
            let a_log_ptr = wf.get_tensor_ptr(&format!("{}.A_log", prefix));
            let dt_bias_ptr = wf.get_tensor_ptr(&format!("{}.dt_bias", prefix));
            let a_log_off = a_log_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
            let dt_bias_off = dt_bias_ptr.map_or(0, |p| (p as usize - gw.base as usize) as u64);
            let enc = cmd_buf.new_compute_command_encoder();
            kernels::encode_compute_decay_beta(c, &enc,
                &c.batch_out[3],
                &c.batch_out[2],
                if a_log_ptr.is_some() { &gw.buf } else { &c.batch_out[3] }, a_log_off,
                if dt_bias_ptr.is_some() { &gw.buf } else { &c.batch_out[2] }, dt_bias_off,
                c.buf_delta_g_decay.as_ref().unwrap(),
                c.buf_delta_beta.as_ref().unwrap(),
                num_v_heads as u32);
            enc.end_encoding();
        }

        // Encoder 5: gated_delta_net_step
        {
            let q_off = 0u64;
            let k_off = (total_key * 4) as u64;
            let v_off = (2 * total_key * 4) as u64;
            let conv_out = c.buf_conv_output.as_ref().unwrap();
            let enc = cmd_buf.new_compute_command_encoder();
            kernels::encode_gated_delta_net_step(c, &enc,
                &c.buf_delta_state[linear_idx],
                conv_out, q_off,
                conv_out, k_off,
                conv_out, v_off,
                c.buf_delta_g_decay.as_ref().unwrap(),
                c.buf_delta_beta.as_ref().unwrap(),
                c.buf_delta_output.as_ref().unwrap(),
                num_v_heads as u32, k_heads_per_v as u32, key_dim as u32, value_dim as u32);
            enc.end_encoding();
        }

        // NO encoders 6-8 (gated_norm, out_proj, residual) — those go in CMD2

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read back SSM output and z for CPU gated_norm (matching C's CPU-side _precise_swiglu)
        let mut ssm_out = vec![0.0f32; total_value];
        let mut z_cpu = vec![0.0f32; total_value];
        unsafe {
            std::ptr::copy_nonoverlapping(
                c.buf_delta_output.as_ref().unwrap().contents() as *const f32,
                ssm_out.as_mut_ptr(), total_value);
            std::ptr::copy_nonoverlapping(
                c.batch_out[1].contents() as *const f32,
                z_cpu.as_mut_ptr(), total_value);
        }

        // CPU gated_norm: sigmoid(z) * rms_norm(ssm_out)
        let gnw = wf.get_tensor_u16(&format!("{}.norm.weight", prefix));
        let gnw_f32: Vec<f32> = gnw.map_or(vec![1.0f32; value_dim], |w| w.iter().map(|&v| bf16_to_f32(v)).collect());
        for hv in 0..num_v_heads {
            let v_start = hv * value_dim;
            let v_end = v_start + value_dim;
            // rms_norm
            let sum_sq: f32 = ssm_out[v_start..v_end].iter().map(|&x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / value_dim as f32 + RMS_NORM_EPS).sqrt();
            for i in 0..value_dim {
                ssm_out[v_start + i] = ssm_out[v_start + i] * inv_rms * gnw_f32[i];
            }
            // silu gate: silu(z) * rms_norm(x) = z * sigmoid(z) * rms_norm(x)
            for i in 0..value_dim {
                let z_val = z_cpu[hv * value_dim + i];
                let gate_i = z_val / (1.0 + (-z_val).exp());  // silu(z) = z * sigmoid(z)
                ssm_out[v_start + i] *= gate_i;
            }
        }

        // Update CPU conv_state for consistency
        let state_off = (CONV_KERNEL_SIZE - 2) * qkv_dim;
        state.conv_state.copy_within(qkv_dim.., 0);
        state.conv_state[state_off..state_off + qkv_dim].fill(0.0);

        return Some(LinearAttnFused3State {
            gated_out: ssm_out,
            h_mid: residual,  // pre-attention hidden (saved before norm)
            total_value,
            o_prefix: format!("{}.out_proj", prefix),
            post_norm_name: format!("model.layers.{}.post_attention_layernorm.weight", layer_idx),
        });
    }

    {
        // ── Non-fused or CPU path ──
        if debug_this_layer {
            let rms_normed: f32 = (normed.iter().map(|&x| x*x).sum::<f32>() / normed.len() as f32).sqrt();
            eprintln!("[RUST-CPU-L0] normed_rms={:.6} normed_first5=[{:.8},{:.8},{:.8},{:.8},{:.8}]", rms_normed, normed[0], normed[1], normed[2], normed[3], normed[4]);
            let rms_hidden: f32 = (hidden.iter().map(|&x| x*x).sum::<f32>() / hidden.len() as f32).sqrt();
            eprintln!("[RUST-CPU-L0] hidden_in_rms={:.6} hidden_in_first5=[{:.8},{:.8},{:.8},{:.8},{:.8}]", rms_hidden, hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
            // Dump full hidden array as comma-separated values for Python consumption
            eprintln!("[RUST-CPU-L0-HIDDEN] {}", hidden.iter().map(|x| format!("{:.8}", x)).collect::<Vec<_>>().join(","));
        }
        // CPU: attention projections
        if let (Some(qw), Some(qs), Some(qb)) = (
            wf.get_tensor_u32(&format!("{}.in_proj_qkv.weight", prefix)),
            wf.get_tensor_u16(&format!("{}.in_proj_qkv.scales", prefix)),
            wf.get_tensor_u16(&format!("{}.in_proj_qkv.biases", prefix)),
        ) { cpu_dequant_matvec_4bit(qw, qs, qb, &normed, &mut qkv, qkv_dim, hidden_dim, GROUP_SIZE); }
        if let (Some(zw), Some(zs), Some(zb)) = (
            wf.get_tensor_u32(&format!("{}.in_proj_z.weight", prefix)),
            wf.get_tensor_u16(&format!("{}.in_proj_z.scales", prefix)),
            wf.get_tensor_u16(&format!("{}.in_proj_z.biases", prefix)),
        ) { cpu_dequant_matvec_4bit(zw, zs, zb, &normed, &mut z, total_value, hidden_dim, GROUP_SIZE); }
        if let (Some(bw), Some(bs), Some(bb)) = (
            wf.get_tensor_u32(&format!("{}.in_proj_b.weight", prefix)),
            wf.get_tensor_u16(&format!("{}.in_proj_b.scales", prefix)),
            wf.get_tensor_u16(&format!("{}.in_proj_b.biases", prefix)),
        ) { cpu_dequant_matvec_4bit(bw, bs, bb, &normed, &mut beta, num_v_heads, hidden_dim, GROUP_SIZE); }
        if let (Some(aw), Some(ass), Some(ab)) = (
            wf.get_tensor_u32(&format!("{}.in_proj_a.weight", prefix)),
            wf.get_tensor_u16(&format!("{}.in_proj_a.scales", prefix)),
            wf.get_tensor_u16(&format!("{}.in_proj_a.biases", prefix)),
        ) { cpu_dequant_matvec_4bit(aw, ass, ab, &normed, &mut alpha, num_v_heads, hidden_dim, GROUP_SIZE); }

        // Conv1d step (CPU)
        let mut conv_out = vec![0.0f32; qkv_dim];
        if let Some(conv_w) = wf.get_tensor_u16(&format!("{}.conv1d.weight", prefix)) {
            cpu_conv1d_step(&state.conv_state, &qkv, conv_w, &mut conv_out, qkv_dim, CONV_KERNEL_SIZE);
        } else {
            conv_out.copy_from_slice(&qkv);
        }
        let state_off = (CONV_KERNEL_SIZE - 2) * qkv_dim;
        state.conv_state.copy_within(qkv_dim.., 0);
        state.conv_state[state_off..state_off + qkv_dim].copy_from_slice(&qkv);
        let lin_q = conv_out[..total_key].to_vec();
        let lin_k = conv_out[total_key..2 * total_key].to_vec();
        let lin_v = conv_out[2 * total_key..].to_vec();

        // Try non-fused GPU SSM (or CPU fallback)
        let gpu_ssm_ok = gpu_compatible && ctx.is_some();
        if gpu_ssm_ok {
            let c = ctx.unwrap();
            let ssm_size = num_v_heads * value_dim * key_dim;
            let ssm_gpu = state.ssm_state_gpu.get_or_insert_with(|| {
                metal_buf_shared(&c.device, ssm_size * 4)
            });
            unsafe { let dst = ssm_gpu.contents() as *mut f32; std::ptr::copy_nonoverlapping(state.ssm_state.as_ptr(), dst, ssm_size); }

            let q_gpu = metal_buf_shared(&c.device, total_key * 4);
            let k_gpu = metal_buf_shared(&c.device, total_key * 4);
            let v_gpu = metal_buf_shared(&c.device, total_value * 4);
            let z_gpu = metal_buf_shared(&c.device, total_value * 4);
            let alpha_gpu = metal_buf_shared(&c.device, num_v_heads * 4);
            let beta_gpu = metal_buf_shared(&c.device, num_v_heads * 4);
            let out_gpu = metal_buf_shared(&c.device, total_value * 4);
            unsafe {
                std::ptr::copy_nonoverlapping(lin_q.as_ptr(), q_gpu.contents() as *mut f32, total_key);
                std::ptr::copy_nonoverlapping(lin_k.as_ptr(), k_gpu.contents() as *mut f32, total_key);
                std::ptr::copy_nonoverlapping(lin_v.as_ptr(), v_gpu.contents() as *mut f32, total_value);
                std::ptr::copy_nonoverlapping(z.as_ptr(), z_gpu.contents() as *mut f32, total_value);
                std::ptr::copy_nonoverlapping(alpha.as_ptr(), alpha_gpu.contents() as *mut f32, num_v_heads);
                std::ptr::copy_nonoverlapping(beta.as_ptr(), beta_gpu.contents() as *mut f32, num_v_heads);
            }
            let a_log_ptr = wf.get_tensor_ptr(&format!("{}.A_log", prefix));
            let dt_bias_ptr = wf.get_tensor_ptr(&format!("{}.dt_bias", prefix));
            let a_log_gpu = metal_buf_shared(&c.device, num_v_heads * 4);
            let dt_bias_gpu = metal_buf_shared(&c.device, num_v_heads * 2);
            if let Some(p) = a_log_ptr {
                unsafe { std::ptr::copy_nonoverlapping(p as *const f32, a_log_gpu.contents() as *mut f32, num_v_heads); }
            }
            if let Some(p) = dt_bias_ptr {
                unsafe { std::ptr::copy_nonoverlapping(p as *const u16, dt_bias_gpu.contents() as *mut u16, num_v_heads); }
            }
            let g_decay_gpu = metal_buf_shared(&c.device, num_v_heads * 4);
            let beta_gate_gpu = metal_buf_shared(&c.device, num_v_heads * 4);
            let gated_gpu2 = metal_buf_shared(&c.device, total_value * 4);
            let gnw_ptr = wf.get_tensor_u16(&format!("{}.norm.weight", prefix));
            let gnw_gpu = gnw_ptr.map(|gnw| {
                let buf = metal_buf_shared(&c.device, gnw.len() * 2);
                unsafe { std::ptr::copy_nonoverlapping(gnw.as_ptr(), buf.contents() as *mut u16, gnw.len()); }
                buf
            });

            let cmd_buf = c.queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();
            kernels::encode_rms_norm_qk(c, &enc, &q_gpu, 0, &k_gpu, 0, num_k_heads as u32, key_dim as u32, inv_scale);
            kernels::encode_compute_decay_beta(c, &enc, &alpha_gpu, &beta_gpu, &a_log_gpu, 0, &dt_bias_gpu, 0, &g_decay_gpu, &beta_gate_gpu, num_v_heads as u32);
            kernels::encode_gated_delta_net_step(c, &enc, ssm_gpu, &q_gpu, 0, &k_gpu, 0, &v_gpu, 0, &g_decay_gpu, &beta_gate_gpu, &out_gpu, num_v_heads as u32, k_heads_per_v as u32, key_dim as u32, value_dim as u32);
            if let Some(ref gnw_buf) = gnw_gpu {
                kernels::encode_gated_rms_norm(c, &enc, &out_gpu, &z_gpu, gnw_buf, 0, &gated_gpu2, num_v_heads as u32, value_dim as u32);
            }
            enc.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
            if gnw_gpu.is_some() {
                unsafe { std::ptr::copy_nonoverlapping(gated_gpu2.contents() as *const f32, gated_out.as_mut_ptr(), total_value); }
            } else {
                unsafe { std::ptr::copy_nonoverlapping(out_gpu.contents() as *const f32, gated_out.as_mut_ptr(), total_value); }
            }
            unsafe { std::ptr::copy_nonoverlapping(ssm_gpu.contents() as *const f32, state.ssm_state.as_mut_ptr(), ssm_size); }
        } else {
        // RMS norm q and k (bare, no weights) then scale
        if debug_this_layer {
            let rms_qkv: f32 = (qkv.iter().map(|&x| x*x).sum::<f32>() / qkv.len() as f32).sqrt();
            let rms_z: f32 = (z.iter().map(|&x| x*x).sum::<f32>() / z.len() as f32).sqrt();
            let rms_b: f32 = (beta.iter().map(|&x| x*x).sum::<f32>() / beta.len() as f32).sqrt();
            let rms_a: f32 = (alpha.iter().map(|&x| x*x).sum::<f32>() / alpha.len() as f32).sqrt();
            eprintln!("[RUST-CPU-L0] qkv_rms={:.6} z_rms={:.6} beta_rms={:.6} alpha_rms={:.6}", rms_qkv, rms_z, rms_b, rms_a);
            eprintln!("[RUST-CPU-L0] qkv_first5=[{:.8},{:.8},{:.8},{:.8},{:.8}]", qkv[0], qkv[1], qkv[2], qkv[3], qkv[4]);
            eprintln!("[RUST-CPU-L0] z_first5=[{:.8},{:.8},{:.8},{:.8},{:.8}]", z[0], z[1], z[2], z[3], z[4]);
            let rms_conv: f32 = (conv_out.iter().map(|&x| x*x).sum::<f32>() / conv_out.len() as f32).sqrt();
            eprintln!("[RUST-CPU-L0] conv_out_rms={:.6} conv_first5=[{:.8},{:.8},{:.8},{:.8},{:.8}]", rms_conv, conv_out[0], conv_out[1], conv_out[2], conv_out[3], conv_out[4]);
            let rms_q: f32 = (lin_q.iter().map(|&x| x*x).sum::<f32>() / lin_q.len() as f32).sqrt();
            let rms_k: f32 = (lin_k.iter().map(|&x| x*x).sum::<f32>() / lin_k.len() as f32).sqrt();
            let rms_v: f32 = (lin_v.iter().map(|&x| x*x).sum::<f32>() / lin_v.len() as f32).sqrt();
            eprintln!("[RUST-CPU-L0] lin_q_rms={:.6} lin_k_rms={:.6} lin_v_rms={:.6}", rms_q, rms_k, rms_v);
        }
        let mut q_normed = vec![0.0f32; total_key];
        let mut k_normed = vec![0.0f32; total_key];
        for h in 0..num_k_heads {
            let qh = &lin_q[h * key_dim..(h + 1) * key_dim];
            let qh_out = &mut q_normed[h * key_dim..(h + 1) * key_dim];
            cpu_rms_norm_bare(qh, qh_out, key_dim, 1e-6);
            let q_scale = inv_scale * inv_scale;
            for d in qh_out.iter_mut() { *d *= q_scale; }
        }
        for h in 0..num_k_heads {
            let kh = &lin_k[h * key_dim..(h + 1) * key_dim];
            let kh_out = &mut k_normed[h * key_dim..(h + 1) * key_dim];
            cpu_rms_norm_bare(kh, kh_out, key_dim, 1e-6);
            for d in kh_out.iter_mut() { *d *= inv_scale; }
        }

        if debug_this_layer {
            let rms_qn: f32 = (q_normed.iter().map(|&x| x*x).sum::<f32>() / q_normed.len() as f32).sqrt();
            let rms_kn: f32 = (k_normed.iter().map(|&x| x*x).sum::<f32>() / k_normed.len() as f32).sqrt();
            eprintln!("[RUST-CPU-L0] q_normed_rms={:.6} qn_first5=[{:.8},{:.8},{:.8},{:.8},{:.8}]", rms_qn, q_normed[0], q_normed[1], q_normed[2], q_normed[3], q_normed[4]);
            eprintln!("[RUST-CPU-L0] k_normed_rms={:.6} kn_first5=[{:.8},{:.8},{:.8},{:.8},{:.8}]", rms_kn, k_normed[0], k_normed[1], k_normed[2], k_normed[3], k_normed[4]);
        }

        let a_log = wf.get_tensor_f32(&format!("{}.A_log", prefix));
        let dt_bias = wf.get_tensor_u16(&format!("{}.dt_bias", prefix));

        let mut out_values = vec![0.0f32; total_value];

        for vh in 0..num_v_heads {
            let kh = vh / k_heads_per_v;
            let a_val = a_log.map_or(1.0, |al| al[vh]);
            let dt_b = dt_bias.map_or(0.0, |db| bf16_to_f32(db[vh]));
            let softplus_val = (1.0 + (alpha[vh] + dt_b).exp()).ln();
            let g_decay = (-a_val.exp() * softplus_val).exp();
            let beta_gate = cpu_sigmoid(beta[vh]);
            let s_off = vh * value_dim * key_dim;
            let ssm = &mut state.ssm_state[s_off..s_off + value_dim * key_dim];
            let v_h = &lin_v[vh * value_dim..(vh + 1) * value_dim];
            let k_h = &k_normed[kh * key_dim..(kh + 1) * key_dim];
            for vi in 0..value_dim {
                for ki in 0..key_dim { ssm[vi * key_dim + ki] *= g_decay; }
            }
            for vi in 0..value_dim {
                let mut kv_mem = 0.0f32;
                for ki in 0..key_dim { kv_mem += ssm[vi * key_dim + ki] * k_h[ki]; }
                let delta = (v_h[vi] - kv_mem) * beta_gate;
                for ki in 0..key_dim { ssm[vi * key_dim + ki] += k_h[ki] * delta; }
            }
            let q_h = &q_normed[kh * key_dim..(kh + 1) * key_dim];
            let o_h = &mut out_values[vh * value_dim..(vh + 1) * value_dim];
            for vi in 0..value_dim {
                let mut sum = 0.0f32;
                for ki in 0..key_dim { sum += ssm[vi * key_dim + ki] * q_h[ki]; }
                o_h[vi] = sum;
            }
        }

        // RMSNormGated
        if debug_this_layer {
            let rms_out: f32 = (out_values.iter().map(|&x| x*x).sum::<f32>() / out_values.len() as f32).sqrt();
            eprintln!("[RUST-CPU-L0] ssm_out_rms={:.6} ssm_out_first5=[{:.8},{:.8},{:.8},{:.8},{:.8}]", rms_out, out_values[0], out_values[1], out_values[2], out_values[3], out_values[4]);
        }
        if let Some(gnw) = wf.get_tensor_u16(&format!("{}.norm.weight", prefix)) {
            for vh in 0..num_v_heads {
                let oh = &out_values[vh * value_dim..(vh + 1) * value_dim];
                let zh = &z[vh * value_dim..(vh + 1) * value_dim];
                let gh = &mut gated_out[vh * value_dim..(vh + 1) * value_dim];
                cpu_rms_norm_gated(oh, zh, gnw, gh, value_dim, RMS_NORM_EPS);
            }
        } else {
            gated_out.copy_from_slice(&out_values);
        }
        if debug_this_layer {
            let rms_gated: f32 = (gated_out.iter().map(|&x| x*x).sum::<f32>() / gated_out.len() as f32).sqrt();
            eprintln!("[RUST-CPU-L0] gated_out_rms={:.6} gated_first5=[{:.8},{:.8},{:.8},{:.8},{:.8}]", rms_gated, gated_out[0], gated_out[1], gated_out[2], gated_out[3], gated_out[4]);
        }
    }
    }

    // Output projection
    let mut attn_out = vec![0.0f32; hidden_dim];
    if use_gpu {
        let gw = gpu_wf.unwrap();
        let c = ctx.unwrap();
        gw.matvec(wf, c, &format!("{}.out_proj", prefix), &gated_out, &mut attn_out, hidden_dim, total_value);
    } else if let (Some(ow), Some(os), Some(ob)) = (
        wf.get_tensor_u32(&format!("{}.out_proj.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.out_proj.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.out_proj.biases", prefix)),
    ) {
        cpu_dequant_matvec_4bit(ow, os, ob, &gated_out, &mut attn_out, hidden_dim, total_value, GROUP_SIZE);
    }
    // Residual add
    if debug_this_layer {
        let rms_attn: f32 = (attn_out.iter().map(|&x| x*x).sum::<f32>() / attn_out.len() as f32).sqrt();
        eprintln!("[RUST-CPU-L0] attn_out_rms={:.6} attn_first5=[{:.8},{:.8},{:.8},{:.8},{:.8}]", rms_attn, attn_out[0], attn_out[1], attn_out[2], attn_out[3], attn_out[4]);
    }
    for i in 0..hidden_dim {
        hidden[i] = residual[i] + attn_out[i];
    }
    if debug_this_layer {
        let rms_out: f32 = (hidden.iter().map(|&x| x*x).sum::<f32>() / hidden.len() as f32).sqrt();
        eprintln!("[RUST-CPU-L0] hidden_out_rms={:.6} hidden_out_first5=[{:.8},{:.8},{:.8},{:.8},{:.8}]", rms_out, hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }
    None
}

// ─── MoE layer forward ─────────────────────────────────────────────────────

/// Run the full MoE block for a single layer: routing, shared expert, K routed experts, combine.
///
/// When `attn_state` is `Some`, fuses batched attention + o_proj + residual + norm + gate
/// into a single CMD2 (matching the C engine's 3-CMD architecture).
/// When `attn_state` is `None` and `lin_attn` is `Some`, fuses out_proj + residual + norm + gate
/// into a single CMD2 for linear attention layers (Fused3 mode, matching C exactly).
///
/// Returns `Some(DeferredExperts)` when GPU expert dispatch is used (async CMD3).
pub fn moe_layer_forward(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    packed_fd: RawFd,
    ctx: Option<&MetalContext>,
    gpu_wf: Option<&GpuWeightCtx>,
    config: &ModelConfig,
    mode: PipelineMode,
    attn_state: Option<FullAttnCmd2State>,
    lin_attn: Option<LinearAttnFused3State>,
    mut expert_io: Option<&mut ExpertIOState>,
) -> Result<Option<DeferredExperts>, MoEError> {
    let hidden_dim = config.hidden_dim;
    let num_experts = config.num_experts;
    let moe_inter = config.moe_intermediate;
    let shared_inter = config.shared_intermediate;
    let expert_size = config.expert_size_4bit;
    let layout = &config.expert_layout_4bit;
    let k = config.num_experts_per_tok;

    let use_gpu = mode != PipelineMode::CpuOnly
        && ctx.is_some()
        && gpu_wf.is_some();

    // Save h_mid (residual) — already completed by caller before this layer
    let h_mid = hidden.to_vec();

    let post_norm_name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
    let mut h_post = vec![0.0f32; hidden_dim];

    // ── Router gate + shared expert projections ──
    let mut gate_scores = vec![0.0f32; num_experts];
    let mut shared_gate = vec![0.0f32; shared_inter];
    let mut shared_up = vec![0.0f32; shared_inter];
    let mut shared_gate_score = 0.0f32;

    let prefix = format!("model.layers.{}.mlp", layer_idx);

    // GPU buffers (preserved for expert dispatch combine)
    let mut sg_buf_gpu: Option<Buffer> = None;
    let mut su_buf_gpu: Option<Buffer> = None;
    // When set, CMD3 uses this instead of uploading h_mid from CPU
    let mut hmid_gpu_override: Option<Buffer> = None;

    // ── CMD2 fusion path: batched attn + o_proj + residual + norm + gate ──
    let use_cmd2_fusion = attn_state.is_some()
        && use_gpu
        && ctx.is_some()
        && ctx.unwrap().attn_scores_batched.is_some()
        && ctx.unwrap().attn_softmax_batched.is_some()
        && ctx.unwrap().attn_values_batched.is_some()
        && ctx.unwrap().sigmoid_gate.is_some()
        && ctx.unwrap().residual_add.is_some()
        && ctx.unwrap().rms_norm_apply_bf16.is_some();

    if use_cmd2_fusion {
        let c = ctx.unwrap();
        let gw = gpu_wf.unwrap();
        let attn = attn_state.unwrap();

        // Allocate intermediate buffers
        let o_proj_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let temp_buf = metal_buf_shared(&c.device, hidden_dim * 4);  // residual_add output
        let sum_sq_buf = metal_buf_shared(&c.device, 4);
        let normed_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let gate_buf = metal_buf_shared(&c.device, num_experts * 4);
        let sg_buf = metal_buf_shared(&c.device, shared_inter * 4);
        let su_buf = metal_buf_shared(&c.device, shared_inter * 4);
        let sge_buf = metal_buf_shared(&c.device, 4);

        let cmd_buf = c.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        // Step 1: attn_scores_batched
        {
            let pipe = c.attn_scores_batched.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&attn.q_buf), 0);
            enc.set_buffer(1, Some(&attn.kc_buf), 0);
            enc.set_buffer(2, Some(&attn.scores_buf), 0);
            unsafe {
                enc.set_bytes(3, 4, &attn.head_dim as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &attn.kv_dim as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &attn.seq_len as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &attn.seq_stride as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &attn.scale as *const f32 as *const c_void);
                enc.set_bytes(8, 4, &attn.heads_per_kv as *const u32 as *const c_void);
                enc.set_bytes(9, 4, &attn.seq_len as *const u32 as *const c_void);
            }
            enc.dispatch_thread_groups(
                MTLSize::new((attn.num_attn_heads * attn.seq_len) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 2: attn_softmax_batched
        {
            let pipe = c.attn_softmax_batched.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&attn.scores_buf), 0);
            unsafe {
                enc.set_bytes(1, 4, &attn.seq_len as *const u32 as *const c_void);
                enc.set_bytes(2, 4, &attn.seq_stride as *const u32 as *const c_void);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(attn.num_attn_heads as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 3: attn_values_batched
        {
            let pipe = c.attn_values_batched.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&attn.scores_buf), 0);
            enc.set_buffer(1, Some(&attn.vc_buf), 0);
            enc.set_buffer(2, Some(&attn.out_buf), 0);
            unsafe {
                enc.set_bytes(3, 4, &attn.head_dim as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &attn.kv_dim as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &attn.seq_len as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &attn.seq_stride as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &attn.heads_per_kv as *const u32 as *const c_void);
            }
            let total_threads = attn.num_attn_heads * attn.head_dim;
            enc.dispatch_thread_groups(
                MTLSize::new(((total_threads + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 4: sigmoid_gate (attn_out *= sigmoid(q_gate))
        {
            let pipe = c.sigmoid_gate.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&attn.out_buf), 0);
            enc.set_buffer(1, Some(&attn.q_gate_buf), 0);
            unsafe { enc.set_bytes(2, 4, &attn.q_dim as *const u32 as *const c_void); }
            enc.dispatch_thread_groups(
                MTLSize::new(((attn.q_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 5: o_proj matvec (gated attention → hidden_dim)
        gw.encode_matvec_into(wf, c, &enc, &attn.o_prefix, &attn.out_buf, 0, &o_proj_buf, 0, hidden_dim, attn.q_dim as usize);

        // Step 6: residual_add (o_proj_out + h_mid → temp_buf)
        {
            let pipe = c.residual_add.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&o_proj_buf), 0);
            enc.set_buffer(1, Some(&attn.hidden_buf), 0);
            enc.set_buffer(2, Some(&temp_buf), 0);
            unsafe { enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void); }
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 7: post_attention_layernorm (rms_norm_sum + rms_norm_apply_bf16)
        {
            enc.set_compute_pipeline_state(&c.rms_norm_sum);
            enc.set_buffer(0, Some(&temp_buf), 0);
            enc.set_buffer(1, Some(&sum_sq_buf), 0);
            unsafe { enc.set_bytes(2, 4, &(hidden_dim as u32) as *const u32 as *const c_void); }
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        }
        {
            let pipe = c.rms_norm_apply_bf16.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&temp_buf), 0);
            let pnw_ptr = wf.get_tensor_ptr(&post_norm_name).unwrap();
            let pnw_off = (pnw_ptr as usize - gw.base as usize) as u64;
            enc.set_buffer(1, Some(&gw.buf), pnw_off);
            enc.set_buffer(2, Some(&sum_sq_buf), 0);
            enc.set_buffer(3, Some(&normed_buf), 0);
            unsafe {
                enc.set_bytes(4, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 8-11: gate, shared_gate, shared_up, shared_expert_gate matvecs (on normed hidden)
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.gate", prefix), &normed_buf, 0, &gate_buf, 0, num_experts, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.gate_proj", prefix), &normed_buf, 0, &sg_buf, 0, shared_inter, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.up_proj", prefix), &normed_buf, 0, &su_buf, 0, shared_inter, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert_gate", prefix), &normed_buf, 0, &sge_buf, 0, 1, hidden_dim);

        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read back results
        unsafe {
            std::ptr::copy_nonoverlapping(gate_buf.contents() as *const f32, gate_scores.as_mut_ptr(), num_experts);
            std::ptr::copy_nonoverlapping(sg_buf.contents() as *const f32, shared_gate.as_mut_ptr(), shared_inter);
            std::ptr::copy_nonoverlapping(su_buf.contents() as *const f32, shared_up.as_mut_ptr(), shared_inter);
            shared_gate_score = *(sge_buf.contents() as *const f32);
            // Update hidden to post-normed value (h_post for expert input)
            std::ptr::copy_nonoverlapping(normed_buf.contents() as *const f32, hidden.as_mut_ptr(), hidden_dim);
        }

        // Keep shared gate/up on GPU for CMD3 SwiGLU
        sg_buf_gpu = Some(sg_buf);
        su_buf_gpu = Some(su_buf);

        // h_post already set from normed_buf readback above (stored in hidden[])
        h_post.copy_from_slice(hidden);
        // temp_buf = h_mid + attn_out — use as hmid_gpu in CMD3 combine
        hmid_gpu_override = Some(temp_buf);
    } else if lin_attn.is_some() && use_gpu
        && ctx.is_some()
        && ctx.unwrap().residual_add.is_some()
        && ctx.unwrap().rms_norm_apply_bf16.is_some()
    {
        // ── Fused3 linear CMD2: out_proj + residual_add + rms_norm + gate + shared ──
        // Matches C engine exactly: CMD1(SSM) → CPU(gated_norm) → CMD2(this block) → CMD3
        let c = ctx.unwrap();
        let gw = gpu_wf.unwrap();
        let la = lin_attn.unwrap();

        // Upload gated_out and h_mid to GPU
        let gated_buf = metal_buf_shared(&c.device, la.total_value * 4);
        let hmid_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(la.gated_out.as_ptr(), gated_buf.contents() as *mut f32, la.total_value);
            std::ptr::copy_nonoverlapping(la.h_mid.as_ptr(), hmid_buf.contents() as *mut f32, hidden_dim);
        }

        let o_proj_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let temp_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let sum_sq_buf = metal_buf_shared(&c.device, 4);
        let normed_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let gate_buf = metal_buf_shared(&c.device, num_experts * 4);
        let sg_buf = metal_buf_shared(&c.device, shared_inter * 4);
        let su_buf = metal_buf_shared(&c.device, shared_inter * 4);
        let sge_buf = metal_buf_shared(&c.device, 4);

        let cmd_buf = c.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        // Step 1: out_proj (gated_out → hidden_dim)
        gw.encode_matvec_into(wf, c, &enc, &la.o_prefix, &gated_buf, 0, &o_proj_buf, 0, hidden_dim, la.total_value);

        // Step 2: residual_add (o_proj_out + h_mid → temp_buf)
        {
            let pipe = c.residual_add.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&o_proj_buf), 0);
            enc.set_buffer(1, Some(&hmid_buf), 0);
            enc.set_buffer(2, Some(&temp_buf), 0);
            unsafe { enc.set_bytes(3, 4, &(hidden_dim as u32) as *const u32 as *const c_void); }
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 3: post_attention_layernorm (rms_norm_sum + rms_norm_apply_bf16)
        {
            enc.set_compute_pipeline_state(&c.rms_norm_sum);
            enc.set_buffer(0, Some(&temp_buf), 0);
            enc.set_buffer(1, Some(&sum_sq_buf), 0);
            unsafe { enc.set_bytes(2, 4, &(hidden_dim as u32) as *const u32 as *const c_void); }
            enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        }
        {
            let pipe = c.rms_norm_apply_bf16.as_ref().unwrap();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&temp_buf), 0);
            let pnw_ptr = wf.get_tensor_ptr(&la.post_norm_name).unwrap();
            let pnw_off = (pnw_ptr as usize - gw.base as usize) as u64;
            enc.set_buffer(1, Some(&gw.buf), pnw_off);
            enc.set_buffer(2, Some(&sum_sq_buf), 0);
            enc.set_buffer(3, Some(&normed_buf), 0);
            unsafe {
                enc.set_bytes(4, 4, &(hidden_dim as u32) as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &RMS_NORM_EPS as *const f32 as *const c_void);
            }
            enc.dispatch_thread_groups(
                MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        // Step 4: gate, shared_gate, shared_up, shared_expert_gate matvecs (on normed hidden)
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.gate", prefix), &normed_buf, 0, &gate_buf, 0, num_experts, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.gate_proj", prefix), &normed_buf, 0, &sg_buf, 0, shared_inter, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.up_proj", prefix), &normed_buf, 0, &su_buf, 0, shared_inter, hidden_dim);
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert_gate", prefix), &normed_buf, 0, &sge_buf, 0, 1, hidden_dim);

        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read back results
        unsafe {
            std::ptr::copy_nonoverlapping(gate_buf.contents() as *const f32, gate_scores.as_mut_ptr(), num_experts);
            std::ptr::copy_nonoverlapping(sg_buf.contents() as *const f32, shared_gate.as_mut_ptr(), shared_inter);
            std::ptr::copy_nonoverlapping(su_buf.contents() as *const f32, shared_up.as_mut_ptr(), shared_inter);
            shared_gate_score = *(sge_buf.contents() as *const f32);
            // Update hidden to post-normed value
            std::ptr::copy_nonoverlapping(normed_buf.contents() as *const f32, hidden.as_mut_ptr(), hidden_dim);
        }

        sg_buf_gpu = Some(sg_buf);
        su_buf_gpu = Some(su_buf);
        h_post.copy_from_slice(hidden);
        // temp_buf = h_mid + out_proj(attn_out) — use as hmid_gpu in CMD3 combine
        hmid_gpu_override = Some(temp_buf);
    } else {
        // ── Non-fused path: post-norm on CPU, router CMD separately ──
        let pnw = wf.get_tensor_u16(&post_norm_name);
        if let Some(pnw) = pnw {
            let pnw_f32: Vec<f32> = pnw.iter().map(|&v| bf16_to_f32(v)).collect();
            cpu_rms_norm(hidden, &pnw_f32, &mut h_post, hidden_dim, RMS_NORM_EPS);
        } else {
            h_post.copy_from_slice(hidden);
        }

        // Router gate + shared expert projections: all independent (same input) → batch
        if use_gpu {
            let gw = gpu_wf.unwrap();
            let c = ctx.unwrap();
            let x_buf = metal_buf_shared(&c.device, hidden_dim * 4);
            unsafe { let dst = x_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(h_post.as_ptr(), dst, hidden_dim); }
            let gate_buf = metal_buf_shared(&c.device, num_experts * 4);
            let sg_buf = metal_buf_shared(&c.device, shared_inter * 4);
            let su_buf = metal_buf_shared(&c.device, shared_inter * 4);
            let sge_buf = metal_buf_shared(&c.device, 4);

            let cmd_buf = c.queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.gate", prefix), &x_buf, 0, &gate_buf, 0, num_experts, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.gate_proj", prefix), &x_buf, 0, &sg_buf, 0, shared_inter, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.up_proj", prefix), &x_buf, 0, &su_buf, 0, shared_inter, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert_gate", prefix), &x_buf, 0, &sge_buf, 0, 1, hidden_dim);
            enc.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            unsafe {
                std::ptr::copy_nonoverlapping(gate_buf.contents() as *const f32, gate_scores.as_mut_ptr(), num_experts);
                std::ptr::copy_nonoverlapping(sg_buf.contents() as *const f32, shared_gate.as_mut_ptr(), shared_inter);
                std::ptr::copy_nonoverlapping(su_buf.contents() as *const f32, shared_up.as_mut_ptr(), shared_inter);
                let tmp = sge_buf.contents() as *const f32;
                shared_gate_score = *tmp;
            }
            sg_buf_gpu = Some(sg_buf);
            su_buf_gpu = Some(su_buf);
        } else {
            // CPU fallback
            if let (Some(gw_p), Some(gs), Some(gb)) = (
                wf.get_tensor_u32(&format!("{}.gate.weight", prefix)),
                wf.get_tensor_u16(&format!("{}.gate.scales", prefix)),
                wf.get_tensor_u16(&format!("{}.gate.biases", prefix)),
            ) { cpu_dequant_matvec_4bit(gw_p, gs, gb, &h_post, &mut gate_scores, num_experts, hidden_dim, GROUP_SIZE); }
            if let (Some(sgw), Some(sgs), Some(sgb)) = (
                wf.get_tensor_u32(&format!("{}.shared_expert.gate_proj.weight", prefix)),
                wf.get_tensor_u16(&format!("{}.shared_expert.gate_proj.scales", prefix)),
                wf.get_tensor_u16(&format!("{}.shared_expert.gate_proj.biases", prefix)),
            ) { cpu_dequant_matvec_4bit(sgw, sgs, sgb, &h_post, &mut shared_gate, shared_inter, hidden_dim, GROUP_SIZE); }
            if let (Some(suw), Some(sus), Some(sub)) = (
                wf.get_tensor_u32(&format!("{}.shared_expert.up_proj.weight", prefix)),
                wf.get_tensor_u16(&format!("{}.shared_expert.up_proj.scales", prefix)),
                wf.get_tensor_u16(&format!("{}.shared_expert.up_proj.biases", prefix)),
            ) { cpu_dequant_matvec_4bit(suw, sus, sub, &h_post, &mut shared_up, shared_inter, hidden_dim, GROUP_SIZE); }
            if let (Some(segw), Some(segs), Some(segb)) = (
                wf.get_tensor_u32(&format!("{}.shared_expert_gate.weight", prefix)),
                wf.get_tensor_u16(&format!("{}.shared_expert_gate.scales", prefix)),
                wf.get_tensor_u16(&format!("{}.shared_expert_gate.biases", prefix)),
            ) {
                let mut tmp = [0.0f32];
                cpu_dequant_matvec_4bit(segw, segs, segb, &h_post, &mut tmp, 1, hidden_dim, GROUP_SIZE);
                shared_gate_score = tmp[0];
            }
        }
    }

    // ── Routing: softmax + topk ──
    cpu_softmax(&mut gate_scores);

    let mut expert_indices = vec![0usize; k];
    let mut expert_weights = vec![0.0f32; k];
    cpu_topk(&gate_scores, k, &mut expert_indices, &mut expert_weights);
    cpu_normalize_weights(&mut expert_weights);

    // ── Routed expert computation ──
    let mut moe_out = vec![0.0f32; hidden_dim];

    if use_gpu {
        let ctx = ctx.unwrap();
        let gw = gpu_wf.unwrap();
        let k = expert_indices.len();
        let actual_k = k.min(MAX_K);

        let hidden_u32 = hidden_dim as u32;
        let inter_u32 = moe_inter as u32;
        let gs_u32 = GROUP_SIZE as u32;

        // ── Phase 1: Parallel pread (cache hits skip I/O) ──
        let mut valid = [false; MAX_K];
        let has_io = expert_io.is_some();
        let mut fallback_expert_bufs: Vec<Buffer> = Vec::new();

        // Mutable phase: check cache, insert misses, parallel pread
        if let Some(ref mut io) = expert_io {
            let mut miss_ei = [0usize; MAX_K];
            let mut miss_k_slot = [0usize; MAX_K];
            let mut miss_count = 0;

            for ki in 0..actual_k {
                let eidx = expert_indices[ki];
                if let Some(buf) = io.cache.lookup(layer_idx, eidx) {
                    io.expert_data[ki] = buf;
                    valid[ki] = true;
                } else {
                    miss_ei[miss_count] = eidx;
                    miss_k_slot[miss_count] = ki;
                    miss_count += 1;
                }
            }

            // Insert cache entries for misses BEFORE rayon::scope
            for m in 0..miss_count {
                let ki = miss_k_slot[m];
                let eidx = miss_ei[m];
                let buf = io.cache.insert_get_buf(layer_idx, eidx);
                io.expert_data[ki] = buf;
            }

            // Parallel pread cache misses — raw pointers only, no &io in scope
            // Use usize to transmit pointers across threads (raw pointers aren't Send)
            if miss_count > 0 {
                let mut pread_tasks: Vec<(RawFd, usize, usize, i64)> = Vec::with_capacity(miss_count);
                for m in 0..miss_count {
                    let ki = miss_k_slot[m];
                    let eidx = miss_ei[m];
                    let ptr = io.expert_data[ki].contents() as usize;
                    pread_tasks.push((packed_fd, ptr, expert_size, (eidx as i64) * (expert_size as i64)));
                }
                rayon::scope(|s| {
                    for (fd, dst, sz, off) in pread_tasks {
                        s.spawn(move |_| {
                            unsafe { libc::pread(fd, dst as *mut std::ffi::c_void, sz, off); }
                        });
                    }
                });
            }
            for m in 0..miss_count {
                valid[miss_k_slot[m]] = true;
            }
        } else {
            // Fallback: sequential pread into ad-hoc buffers
            for ki in 0..actual_k {
                let eidx = expert_indices[ki];
                let buf = metal_buf_shared(&ctx.device, expert_size);
                let nread = unsafe {
                    let ptr = buf.contents() as *mut u8;
                    let slice = std::slice::from_raw_parts_mut(ptr, expert_size);
                    libc::pread(packed_fd, slice.as_mut_ptr() as *mut std::ffi::c_void, expert_size, (eidx as i64) * (expert_size as i64))
                };
                if nread == expert_size as isize {
                    valid[ki] = true;
                }
                fallback_expert_bufs.push(buf);
            }
        }
        // Mutable borrow of expert_io ends here

        let any_valid = valid.iter().take(actual_k).any(|&v| v);

        if any_valid {
            // ── Phase 2: GPU dispatch using pre-allocated or ad-hoc buffers ──
            let io_ref = expert_io.as_deref();  // Option<&ExpertIOState>

            let (x_buf, gate_out, up_out, act_out, out_bufs,
                 shared_act_gpu, shared_down_gpu, hidden_out, params_buf)
                = if let Some(io) = io_ref {
                // Pre-allocated path: reuse persistent Metal buffers
                unsafe { let dst = io.input_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(h_post.as_ptr(), dst, hidden_dim); }
                let ob: Vec<Buffer> = io.expert_out.iter().take(actual_k).cloned().collect();
                (io.input_buf.clone(), io.scratch_gate.clone(), io.scratch_up.clone(),
                 io.scratch_act.clone(), ob,
                 io.shared_act.clone(), io.shared_down.clone(),
                 io.combine_out.clone(), io.combine_params.clone())
            } else {
                // Legacy path: allocate per-layer
                let x = metal_buf_shared(&ctx.device, hidden_dim * 4);
                unsafe { let dst = x.contents() as *mut f32; std::ptr::copy_nonoverlapping(h_post.as_ptr(), dst, hidden_dim); }
                (x,
                 metal_buf_shared(&ctx.device, moe_inter * 4),
                 metal_buf_shared(&ctx.device, moe_inter * 4),
                 metal_buf_shared(&ctx.device, moe_inter * 4),
                 (0..actual_k).map(|_| metal_buf_shared(&ctx.device, hidden_dim * 4)).collect(),
                 metal_buf_shared(&ctx.device, shared_inter * 4),
                 metal_buf_shared(&ctx.device, hidden_dim * 4),
                 metal_buf_shared(&ctx.device, hidden_dim * 4),
                 metal_buf_shared(&ctx.device, 40))
            };

            // h_mid on GPU for moe_combine_residual
            let hmid_gpu = if let Some(buf) = hmid_gpu_override.take() {
                buf
            } else {
                let buf = metal_buf_shared(&ctx.device, hidden_dim * 4);
                unsafe { let dst = buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(h_mid.as_ptr(), dst, hidden_dim); }
                buf
            };

            // ── FUSED CMD: K experts + shared SwiGLU + shared down_proj + moe_combine_residual ──
            let cmd_buf = ctx.queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();

            for ki in 0..actual_k {
                if !valid[ki] { continue; }
                // Expert data: from pre-allocated io.expert_data[ki] or fallback
                let expert_buf: &Buffer = if let Some(io) = io_ref {
                    &io.expert_data[ki]
                } else if ki < fallback_expert_bufs.len() {
                    &fallback_expert_bufs[ki]
                } else {
                    continue;
                };
                kernels::encode_matvec_offset(ctx, &enc,
                    expert_buf, layout.gate_w_off as u64,
                    expert_buf, layout.gate_s_off as u64,
                    expert_buf, layout.gate_b_off as u64,
                    &x_buf, 0, &gate_out, 0,
                    inter_u32, hidden_u32, gs_u32, 3);

                kernels::encode_matvec_offset(ctx, &enc,
                    expert_buf, layout.up_w_off as u64,
                    expert_buf, layout.up_s_off as u64,
                    expert_buf, layout.up_b_off as u64,
                    &x_buf, 0, &up_out, 0,
                    inter_u32, hidden_u32, gs_u32, 3);

                kernels::encode_swiglu(ctx, &enc, &gate_out, 0, &up_out, 0, &act_out, 0, inter_u32);

                kernels::encode_matvec_offset(ctx, &enc,
                    expert_buf, layout.down_w_off as u64,
                    expert_buf, layout.down_s_off as u64,
                    expert_buf, layout.down_b_off as u64,
                    &act_out, 0, &out_bufs[ki], 0,
                    hidden_u32, inter_u32, gs_u32, 3);
            }

            // Shared expert SwiGLU on GPU
            if let (Some(ref sg), Some(ref su)) = (sg_buf_gpu.as_ref(), su_buf_gpu.as_ref()) {
                kernels::encode_swiglu(ctx, &enc, sg, 0, su, 0, &shared_act_gpu, 0, shared_inter as u32);
            }

            gw.encode_matvec_into(wf, ctx, &enc,
                &format!("{}.shared_expert.down_proj", prefix),
                &shared_act_gpu, 0, &shared_down_gpu, 0, hidden_dim, shared_inter);

            // ── moe_combine_residual ──
            {
                let mcr_pipe = ctx.moe_combine_residual.as_ref().unwrap();
                enc.set_compute_pipeline_state(mcr_pipe);
                enc.set_buffer(0, Some(&hmid_gpu), 0);
                enc.set_buffer(1, Some(&shared_down_gpu), 0);
                enc.set_buffer(2, Some(&hidden_out), 0);
                for ei in 0..MAX_K {
                    if ei < actual_k && valid[ei] {
                        enc.set_buffer(3 + ei as u64, Some(&out_bufs[ei]), 0);
                    } else {
                        enc.set_buffer(3 + ei as u64, Some(&hidden_out), 0);
                    }
                }
                let mut params = [0.0f32; 10];
                for (i, &w) in expert_weights.iter().enumerate() { params[i] = w; }
                params[8] = shared_gate_score;
                unsafe { std::ptr::copy_nonoverlapping(params.as_ptr(), params_buf.contents() as *mut f32, 10); }
                enc.set_buffer(11, Some(&params_buf), 0);
                unsafe {
                    enc.set_bytes(12, 4, &hidden_u32 as *const u32 as *const c_void);
                    let ku = actual_k as u32;
                    enc.set_bytes(13, 4, &ku as *const u32 as *const c_void);
                }
                enc.dispatch_thread_groups(
                    MTLSize::new(((hidden_dim + 255) / 256) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }

            enc.end_encoding();
            cmd_buf.commit();

            // Only per-layer-allocated buffers need keep_alive; pre-allocated live in ExpertIOState
            let mut keep_alive = Vec::with_capacity(4);
            keep_alive.push(hmid_gpu);
            if io_ref.is_none() {
                keep_alive.push(shared_act_gpu);
                keep_alive.push(shared_down_gpu);
                keep_alive.push(params_buf);
                keep_alive.push(x_buf);
                keep_alive.push(gate_out);
                keep_alive.push(up_out);
                keep_alive.push(act_out);
                keep_alive.extend(out_bufs);
                keep_alive.extend(fallback_expert_bufs);
            }
            if let Some(b) = sg_buf_gpu.take() { keep_alive.push(b); }
            if let Some(b) = su_buf_gpu.take() { keep_alive.push(b); }

            return Ok(Some(DeferredExperts {
                cmd_buf: Some(cmd_buf.to_owned()),
                out_buf: Some(hidden_out),
                _keep_alive: keep_alive,
            }));
        }
        // No experts loaded — fall through to CPU below
    }

    let gpu_done = !moe_out.iter().all(|&v| v == 0.0);
    if !gpu_done {
        // ── CPU fallback: compute everything synchronously ──
        let mut expert_data = vec![0u8; expert_size];
        let mut gate_tmp = vec![0.0f32; moe_inter];
        let mut up_tmp = vec![0.0f32; moe_inter];
        let mut act_tmp = vec![0.0f32; moe_inter];
        let mut eout = vec![0.0f32; hidden_dim];

        for (&eidx, &ew) in expert_indices.iter().zip(expert_weights.iter()) {
            let expert_offset = (eidx as i64) * (expert_size as i64);
            let nread = unsafe {
                libc::pread(
                    packed_fd,
                    expert_data.as_mut_ptr() as *mut std::ffi::c_void,
                    expert_size,
                    expert_offset,
                )
            };
            if nread != expert_size as isize {
                continue;
            }

            // gate_proj
            let gw = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.gate_w_off) as *const u32, layout.gate_w_size / 4) };
            let gs = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.gate_s_off) as *const u16, layout.gate_s_size / 2) };
            let gb = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.gate_b_off) as *const u16, layout.gate_b_size / 2) };
            cpu_dequant_matvec_4bit(gw, gs, gb, &h_post, &mut gate_tmp, moe_inter, hidden_dim, GROUP_SIZE);

            // up_proj
            let uw = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.up_w_off) as *const u32, layout.up_w_size / 4) };
            let us = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.up_s_off) as *const u16, layout.up_s_size / 2) };
            let ub = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.up_b_off) as *const u16, layout.up_b_size / 2) };
            cpu_dequant_matvec_4bit(uw, us, ub, &h_post, &mut up_tmp, moe_inter, hidden_dim, GROUP_SIZE);

            // SwiGLU
            for i in 0..moe_inter {
                let g = gate_tmp[i];
                let silu_g = g / (1.0 + (-g).exp());
                act_tmp[i] = silu_g * up_tmp[i];
            }

            // down_proj
            let dw = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.down_w_off) as *const u32, layout.down_w_size / 4) };
            let ds = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.down_s_off) as *const u16, layout.down_s_size / 2) };
            let db = unsafe { std::slice::from_raw_parts(expert_data.as_ptr().add(layout.down_b_off) as *const u16, layout.down_b_size / 2) };
            cpu_dequant_matvec_4bit(dw, ds, db, &act_tmp, &mut eout, hidden_dim, moe_inter, GROUP_SIZE);

            for d in 0..hidden_dim {
                moe_out[d] += eout[d] * ew;
            }
        }
    }

    // ── Shared expert SwiGLU + down_proj ──
    let mut shared_out = vec![0.0f32; hidden_dim];
    let mut shared_act = vec![0.0f32; shared_inter];

    // SwiGLU on shared gate/up
    for i in 0..shared_inter {
        let g = shared_gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        shared_act[i] = silu_g * shared_up[i];
    }

    // Shared expert down_proj
    if use_gpu {
        let gw = gpu_wf.unwrap();
        let c = ctx.unwrap();
        let sa_buf = metal_buf_shared(&c.device, shared_inter * 4);
        unsafe { let dst = sa_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(shared_act.as_ptr(), dst, shared_inter); }
        let so_buf = metal_buf_shared(&c.device, hidden_dim * 4);
        let cmd_buf = c.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        gw.encode_matvec_into(wf, c, &enc, &format!("{}.shared_expert.down_proj", prefix), &sa_buf, 0, &so_buf, 0, hidden_dim, shared_inter);
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        unsafe { std::ptr::copy_nonoverlapping(so_buf.contents() as *const f32, shared_out.as_mut_ptr(), hidden_dim); }
    } else if let (Some(sdw), Some(sds), Some(sdb)) = (
        wf.get_tensor_u32(&format!("{}.shared_expert.down_proj.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.shared_expert.down_proj.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.shared_expert.down_proj.biases", prefix)),
    ) {
        cpu_dequant_matvec_4bit(sdw, sds, sdb, &shared_act, &mut shared_out, hidden_dim, shared_inter, GROUP_SIZE);
    }

    let shared_weight = cpu_sigmoid(shared_gate_score);

    // ── Final combine: hidden = h_mid + moe_out + shared_weight * shared_out ──
    for i in 0..hidden_dim {
        hidden[i] = h_mid[i] + moe_out[i] + shared_weight * shared_out[i];
    }

    Ok(None)
}

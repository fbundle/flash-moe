/// Metal context management: device, queue, shader library, pipeline states.
///
/// Port of metal_init() / MetalContext from main.m:172-322.
use metal::*;
use objc::rc::autoreleasepool;
use std::collections::HashMap;
use std::ffi::c_void;

use crate::error::MoEError;
use crate::kernels;
use crate::weights::WeightFile;

// ─── Expert I/O pre-allocation & LRU cache ───────────────────────────────────

pub const MAX_K: usize = 8;
const CACHE_SIZE: usize = 32;

/// LRU cache for expert Metal buffers. Maps (layer, expert_id) → pre-allocated buffer.
/// Cache hits skip pread entirely; misses evict the least-recently-used entry.
pub struct ExpertCache {
    entries: Vec<CacheEntry>,
    map: HashMap<(usize, usize), usize>,
    access_counter: u64,
    pub hits: u64,
    pub misses: u64,
}

struct CacheEntry {
    buffer: Buffer,
    layer_idx: i32,   // -1 = unused
    expert_idx: i32,
    last_used: u64,
}

impl ExpertCache {
    pub fn new(device: &Device, max_entries: usize, expert_size: usize) -> Self {
        let mut entries = Vec::with_capacity(max_entries);
        for _ in 0..max_entries {
            entries.push(CacheEntry {
                buffer: metal_buf_shared(device, expert_size),
                layer_idx: -1,
                expert_idx: -1,
                last_used: 0,
            });
        }
        ExpertCache {
            entries,
            map: HashMap::with_capacity(max_entries),
            access_counter: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a cached Metal buffer. Returns None on miss.
    pub fn lookup(&mut self, layer: usize, expert: usize) -> Option<Buffer> {
        if let Some(&idx) = self.map.get(&(layer, expert)) {
            self.entries[idx].last_used = self.access_counter;
            self.access_counter += 1;
            self.hits += 1;
            Some(self.entries[idx].buffer.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert/evict: returns a buffer to pread expert data into.
    /// The returned buffer may be newly unused or LRU-evicted — its old contents
    /// are invalid and must be overwritten with pread.
    pub fn insert_get_buf(&mut self, layer: usize, expert: usize) -> Buffer {
        self.access_counter += 1;

        // Already cached? Just update LRU
        if let Some(&idx) = self.map.get(&(layer, expert)) {
            self.entries[idx].last_used = self.access_counter;
            return self.entries[idx].buffer.clone();
        }

        // Find slot: unused first, then LRU
        let target = if self.map.len() < self.entries.len() {
            self.map.len()
        } else {
            // Evict LRU
            let mut lru = 0;
            let mut min_used = u64::MAX;
            for (i, e) in self.entries.iter().enumerate() {
                if e.last_used < min_used {
                    min_used = e.last_used;
                    lru = i;
                }
            }
            let old_layer = self.entries[lru].layer_idx;
            let old_expert = self.entries[lru].expert_idx;
            if old_layer >= 0 && old_expert >= 0 {
                self.map.remove(&(old_layer as usize, old_expert as usize));
            }
            lru
        };

        self.entries[target].layer_idx = layer as i32;
        self.entries[target].expert_idx = expert as i32;
        self.entries[target].last_used = self.access_counter;
        self.map.insert((layer, expert), target);
        self.entries[target].buffer.clone()
    }
}

/// Pre-allocated GPU buffers for expert dispatch — allocated once, reused across
/// all layers and tokens.  Matches C's `buf_multi_expert_*` and scratch buffers.
pub struct ExpertIOState {
    /// Expert data buffers [MAX_K] — pread'd expert weights land here
    pub expert_data: Vec<Buffer>,
    /// Per-expert output [MAX_K] — each expert's down_proj result
    pub expert_out: Vec<Buffer>,
    /// Scratch buffers (shared across sequential expert dispatches within one CMD)
    pub scratch_gate: Buffer,
    pub scratch_up: Buffer,
    pub scratch_act: Buffer,
    /// Shared input buffer (h_post copy, reused by all experts)
    pub input_buf: Buffer,
    /// Shared expert intermediate + output
    pub shared_act: Buffer,
    pub shared_down: Buffer,
    /// moe_combine_residual output — final hidden state
    pub combine_out: Buffer,
    /// Combine params [10 f32]: expert_weights[8] + shared_gate_score + padding
    pub combine_params: Buffer,
    /// LRU cache for expert data (avoids pread for repeated experts)
    pub cache: ExpertCache,
}

impl ExpertIOState {
    pub fn new(
        device: &Device,
        expert_size: usize,
        hidden_dim: usize,
        moe_inter: usize,
        shared_inter: usize,
        cache_entries: usize,
    ) -> Self {
        let mut expert_data = Vec::with_capacity(MAX_K);
        let mut expert_out = Vec::with_capacity(MAX_K);
        for _ in 0..MAX_K {
            expert_data.push(metal_buf_shared(device, expert_size));
            expert_out.push(metal_buf_shared(device, hidden_dim * 4));
        }
        ExpertIOState {
            expert_data,
            expert_out,
            scratch_gate: metal_buf_shared(device, moe_inter * 4),
            scratch_up: metal_buf_shared(device, moe_inter * 4),
            scratch_act: metal_buf_shared(device, moe_inter * 4),
            input_buf: metal_buf_shared(device, hidden_dim * 4),
            shared_act: metal_buf_shared(device, shared_inter * 4),
            shared_down: metal_buf_shared(device, hidden_dim * 4),
            combine_out: metal_buf_shared(device, hidden_dim * 4),
            combine_params: metal_buf_shared(device, 40),
            cache: ExpertCache::new(device, cache_entries, expert_size),
        }
    }
}

/// Holds all Metal device state and compute pipeline handles.
pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub library: Library,

    // Pipeline states for each kernel
    pub matvec_naive: ComputePipelineState,
    pub matvec_fast: ComputePipelineState,
    pub matvec_v3: ComputePipelineState,
    pub matvec_v4: ComputePipelineState,
    pub matvec_batched: ComputePipelineState,
    pub matvec_v5: Option<ComputePipelineState>,
    pub matvec_2bit: Option<ComputePipelineState>,
    pub swiglu: ComputePipelineState,
    pub swiglu_vec4: Option<ComputePipelineState>,
    pub swiglu_batched: Option<ComputePipelineState>,
    pub weighted_sum: ComputePipelineState,
    pub rms_norm_sum: ComputePipelineState,
    pub rms_norm_apply: ComputePipelineState,
    pub rms_norm_apply_bf16: Option<ComputePipelineState>,
    pub fused_gate_up: Option<ComputePipelineState>,
    pub residual_add: Option<ComputePipelineState>,
    pub sigmoid_gate: Option<ComputePipelineState>,
    pub moe_combine_residual: Option<ComputePipelineState>,
    pub attn_scores_batched: Option<ComputePipelineState>,
    pub attn_softmax_batched: Option<ComputePipelineState>,
    pub attn_values_batched: Option<ComputePipelineState>,
    pub gated_delta_net_step: Option<ComputePipelineState>,
    pub conv1d_step: Option<ComputePipelineState>,
    pub rms_norm_qk: Option<ComputePipelineState>,
    pub compute_decay_beta: Option<ComputePipelineState>,
    pub gated_rms_norm: Option<ComputePipelineState>,

    // ── Persistent GPU buffers for fused forward ──
    /// Per linear-layer conv state: [(kernel_size-1) * qkv_dim] f32
    pub buf_conv_state: Vec<Buffer>,
    /// Per linear-layer SSM state: [num_v_heads * value_dim * key_dim] f32
    pub buf_delta_state: Vec<Buffer>,
    /// Conv output buffer: [qkv_dim] f32 (q+k+v concatenated)
    pub buf_conv_output: Option<Buffer>,
    /// Gated delta intermediate: [num_v_heads] f32
    pub buf_delta_g_decay: Option<Buffer>,
    /// Gated delta intermediate: [num_v_heads] f32
    pub buf_delta_beta: Option<Buffer>,
    /// Delta net output: [total_value] f32
    pub buf_delta_output: Option<Buffer>,
    /// Batch projection outputs (for fused CMD1): QKV, Z, B, A
    pub batch_out: Vec<Buffer>,
}

impl MetalContext {
    /// Allocate persistent GPU buffers for fused linear attention.
    /// Must be called after model config is loaded.
    pub fn init_linear_attn_buffers(
        &mut self,
        num_linear_layers: usize,
        qkv_dim: usize,
        num_v_heads: usize,
        total_value: usize,
        key_dim: usize,
        value_dim: usize,
    ) {
        self.buf_conv_state.clear();
        self.buf_delta_state.clear();
        for _ in 0..num_linear_layers {
            self.buf_conv_state.push(metal_buf_shared(&self.device, 3 * qkv_dim * 4));
            self.buf_delta_state.push(metal_buf_shared(&self.device, num_v_heads * value_dim * key_dim * 4));
        }
        self.buf_conv_output = Some(metal_buf_shared(&self.device, qkv_dim * 4));
        self.buf_delta_g_decay = Some(metal_buf_shared(&self.device, num_v_heads * 4));
        self.buf_delta_beta = Some(metal_buf_shared(&self.device, num_v_heads * 4));
        self.buf_delta_output = Some(metal_buf_shared(&self.device, total_value * 4));
        // 4 batch outputs for QKV, Z, B, A projections
        self.batch_out.clear();
        self.batch_out.push(metal_buf_shared(&self.device, qkv_dim * 4));
        self.batch_out.push(metal_buf_shared(&self.device, total_value * 4));
        self.batch_out.push(metal_buf_shared(&self.device, num_v_heads * 4));
        self.batch_out.push(metal_buf_shared(&self.device, num_v_heads * 4));
    }

    /// Allocate persistent GPU buffers for expert I/O. Returns the state which
    /// should be stored separately (in ModelState) to allow independent borrowing.
    pub fn init_expert_buffers(
        &self,
        expert_size: usize,
        hidden_dim: usize,
        moe_inter: usize,
        shared_inter: usize,
    ) -> ExpertIOState {
        let io = ExpertIOState::new(
            &self.device,
            expert_size,
            hidden_dim,
            moe_inter,
            shared_inter,
            CACHE_SIZE,
        );
        eprintln!(
            "[expert-io] Pre-allocated {}x data bufs ({} MB each), {}x scratch, LRU cache={} entries",
            MAX_K,
            expert_size / (1024 * 1024),
            MAX_K,
            CACHE_SIZE,
        );
        io
    }
}

/// Embed the shaders.metal source at compile time.
const SHADER_SOURCE: &str = include_str!("../shaders/shaders.metal");

impl MetalContext {
    /// Initialize Metal: create device, queue, compile shaders, build all pipelines.
    pub fn init() -> Result<Self, MoEError> {
        autoreleasepool(|| {
            let device = Device::system_default()
                .ok_or_else(|| MoEError::Metal("No Metal device found".into()))?;

            eprintln!("[metal] Device: {}", device.name());
            eprintln!("[metal] Unified memory: {}", if device.has_unified_memory() { "YES" } else { "NO" });
            eprintln!("[metal] Max buffer size: {:.0} MB", device.max_buffer_length() as f64 / (1024.0 * 1024.0));

            let queue = device.new_command_queue();

            // Compile shaders from source at runtime
            let compile_opts = CompileOptions::new();
            compile_opts.set_fast_math_enabled(true);
            compile_opts.set_language_version(MTLLanguageVersion::V3_1);

            eprintln!("[metal] Compiling shaders from source...");
            let t_compile = crate::timer::now_ms();
            let library = device
                .new_library_with_source(SHADER_SOURCE, &compile_opts)
                .map_err(|e| MoEError::Shader(format!("Shader compilation failed: {:?}", e)))?;
            eprintln!("[metal] Shader compilation: {:.0} ms", crate::timer::now_ms() - t_compile);

            // Helper to create a pipeline
            let make_pipeline = |name: &str| -> Result<ComputePipelineState, MoEError> {
                let function = library
                    .get_function(name, None)
                    .map_err(|e| MoEError::Shader(format!("Shader function '{}' not found: {}", name, e)))?;
                device.new_compute_pipeline_state_with_function(&function)
                    .map_err(|e| MoEError::Shader(format!("Pipeline '{}' creation failed: {:?}", name, e)))
            };

            let matvec_naive = make_pipeline("dequant_matvec_4bit")?;
            let matvec_v3 = make_pipeline("dequant_matvec_4bit_v3")?;
            let swiglu = make_pipeline("swiglu_fused")?;
            let weighted_sum = make_pipeline("weighted_sum")?;
            let rms_norm_sum = make_pipeline("rms_norm_sum_sq")?;
            let rms_norm_apply = make_pipeline("rms_norm_apply")?;

            // Optional pipelines
            let matvec_fast = make_pipeline("dequant_matvec_4bit_fast").ok();
            let matvec_v4 = make_pipeline("dequant_matvec_4bit_v4").ok();
            let matvec_batched = make_pipeline("dequant_matvec_4bit_batched").ok();
            let matvec_v5 = make_pipeline("dequant_matvec_4bit_v5").ok();
            let matvec_2bit = make_pipeline("dequant_matvec_2bit").ok();
            let swiglu_vec4 = make_pipeline("swiglu_fused_vec4").ok();
            let swiglu_batched = make_pipeline("swiglu_fused_batched").ok();
            let rms_norm_apply_bf16 = make_pipeline("rms_norm_apply_bf16").ok();
            let fused_gate_up = make_pipeline("fused_gate_up_swiglu").ok();
            let residual_add = make_pipeline("residual_add").ok();
            let sigmoid_gate = make_pipeline("sigmoid_gate").ok();
            let moe_combine_residual = make_pipeline("moe_combine_residual").ok();
            let attn_scores_batched = make_pipeline("attn_scores_batched").ok();
            let attn_softmax_batched = make_pipeline("attn_softmax_batched").ok();
            let attn_values_batched = make_pipeline("attn_values_batched").ok();
            let gated_delta_net_step = make_pipeline("gated_delta_net_step").ok();
            let conv1d_step = make_pipeline("conv1d_step").ok();
            let rms_norm_qk = make_pipeline("rms_norm_qk").ok();
            let compute_decay_beta = make_pipeline("compute_decay_beta").ok();
            let gated_rms_norm = make_pipeline("gated_rms_norm").ok();

            // Validate required pipelines exist
            let matvec_fast = matvec_fast.ok_or_else(|| MoEError::Shader("dequant_matvec_4bit_fast not found".into()))?;
            let matvec_v4 = matvec_v4.ok_or_else(|| MoEError::Shader("dequant_matvec_4bit_v4 not found".into()))?;
            let _ = matvec_batched;

            eprintln!("[metal] All pipelines created successfully");

            Ok(MetalContext {
                device,
                queue,
                library,
                matvec_naive,
                matvec_fast,
                matvec_v3: matvec_v3.clone(),
                matvec_v4,
                matvec_batched: matvec_batched.unwrap_or_else(|| matvec_v3),
                matvec_v5,
                matvec_2bit,
                swiglu,
                swiglu_vec4,
                swiglu_batched,
                weighted_sum,
                rms_norm_sum,
                rms_norm_apply,
                rms_norm_apply_bf16,
                fused_gate_up,
                residual_add,
                sigmoid_gate,
                moe_combine_residual,
                attn_scores_batched,
                attn_softmax_batched,
                attn_values_batched,
                gated_delta_net_step,
                conv1d_step,
                rms_norm_qk,
                compute_decay_beta,
                gated_rms_norm,
                // Persistent GPU buffers — initialized later via init_linear_attn_buffers()
                buf_conv_state: Vec::new(),
                buf_delta_state: Vec::new(),
                buf_conv_output: None,
                buf_delta_g_decay: None,
                buf_delta_beta: None,
                buf_delta_output: None,
                batch_out: Vec::new(),
            })
        })
    }
}

/// Create a shared-memory Metal buffer (CPU and GPU see the same memory).
pub fn metal_buf_shared(device: &Device, size: usize) -> Buffer {
    device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared)
}

/// Create a shared buffer filled from a file descriptor using pread.
pub fn metal_buf_pread(device: &Device, fd: std::os::fd::RawFd, size: usize, offset: i64) -> Result<Buffer, MoEError> {
    let buf = metal_buf_shared(device, size);
    let ptr = buf.contents() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, size) };

    let nread = unsafe {
        libc::pread(fd, slice.as_mut_ptr() as *mut c_void, size, offset)
    };
    if nread != size as isize {
        let err = std::io::Error::last_os_error();
        return Err(MoEError::Io(std::io::Error::new(
            err.kind(),
            format!("pread returned {}, expected {} (err={})", nread, size, err),
        )));
    }
    Ok(buf)
}

// ─── GPU weight buffer wrapper ─────────────────────────────────────────────

const GPU_MATVEC_GROUP_SIZE: u32 = 64;

/// Wraps the entire model weight file in a Metal buffer (zero-copy via mmap).
/// Tensor matvecs dispatch on GPU using byte offsets within this buffer.
pub struct GpuWeightCtx {
    pub buf: Buffer,
    pub base: *const u8,
}

impl GpuWeightCtx {
    /// Create a Metal buffer wrapping the weight file mmap.
    pub fn new(device: &Device, wf: &WeightFile) -> Self {
        let data = wf.data_ptr();
        let size = wf.size;
        let buf = device.new_buffer_with_bytes_no_copy(
            data as *mut std::ffi::c_void,
            size as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        eprintln!("[gpu-weight] Wrapped {:.2} GB weight file in Metal buffer", size as f64 / 1e9);
        GpuWeightCtx { buf, base: data }
    }

    /// Encode a GPU dequant matvec into an existing encoder (for batched dispatch).
    /// Returns false if tensor not found. Caller must end_encoding, commit, and wait.
    pub fn encode_matvec_into(
        &self,
        wf: &WeightFile,
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        prefix: &str,
        x_buf: &BufferRef,
        x_offset: u64,
        out_buf: &BufferRef,
        out_offset: u64,
        out_dim: usize,
        in_dim: usize,
    ) -> bool {
        let w_ptr = match wf.get_tensor_ptr(&format!("{}.weight", prefix)) {
            Some(p) => p, None => return false,
        };
        let s_ptr = match wf.get_tensor_ptr(&format!("{}.scales", prefix)) {
            Some(p) => p, None => return false,
        };
        let b_ptr = match wf.get_tensor_ptr(&format!("{}.biases", prefix)) {
            Some(p) => p, None => return false,
        };

        let w_off = (w_ptr as usize - self.base as usize) as u64;
        let s_off = (s_ptr as usize - self.base as usize) as u64;
        let b_off = (b_ptr as usize - self.base as usize) as u64;

        kernels::encode_matvec_offset(
            ctx, encoder,
            &self.buf, w_off, &self.buf, s_off, &self.buf, b_off,
            x_buf, x_offset, out_buf, out_offset,
            out_dim as u32, in_dim as u32, GPU_MATVEC_GROUP_SIZE, 3,
        );
        true
    }

    /// Dispatch a single GPU dequant matvec (convenience — creates command buffer).
    pub fn matvec(
        &self,
        wf: &WeightFile,
        ctx: &MetalContext,
        prefix: &str,
        x: &[f32],
        out: &mut [f32],
        out_dim: usize,
        in_dim: usize,
    ) -> bool {
        let x_buf = metal_buf_shared(&ctx.device, in_dim * 4);
        unsafe { let dst = x_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(x.as_ptr(), dst, in_dim); }
        let out_buf = metal_buf_shared(&ctx.device, out_dim * 4);

        let cmd_buf = ctx.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        let ok = self.encode_matvec_into(wf, ctx, encoder, prefix, &x_buf, 0, &out_buf, 0, out_dim, in_dim);
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        if ok {
            unsafe { let src = out_buf.contents() as *const f32; std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out_dim); }
        }
        ok
    }
}

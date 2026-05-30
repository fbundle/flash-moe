/// Metal context management: device, queue, shader library, pipeline states.
///
/// Port of metal_init() / MetalContext from main.m:172-322.
use metal::*;
use objc::rc::autoreleasepool;
use std::collections::HashMap;
use crate::cache::Cache;
use crate::constants::FULL_ATTN_INTERVAL;
use crate::error::MoEError;
use crate::engine::metal_kernels;
use crate::model::weights::WeightFile;
use crate::dtype::{DType, string_to_dtype};

// ─── Expert I/O pre-allocation ───────────────────────────────────────────────

pub const MAX_K: usize = 8;

/// Single shared LRU cache for expert Metal buffers, keyed by
/// ``(layer, expert_id)``.  Sharing across layers lets one entry serve any
/// hot expert wherever it gets routed — the router has structural overlap
/// across layers, so cross-layer reuse is the common pattern.
///
/// Cache hits skip the pread entirely; misses evict the least-recently-used
/// entry via a zero-copy buffer swap (no memcpy).
pub struct ExpertCache {
    entries: Vec<CacheEntry>,
    map: HashMap<(usize, usize), usize>,
    access_counter: u64,
    pub hits: u64,
    pub misses: u64,
}

struct CacheEntry {
    buffer: Buffer,
    layer_idx: i32,
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

    /// Look up a cached Metal buffer.  Returns the buffer on hit, None on miss.
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

    /// Swap the given buffer into the cache for ``(layer, expert)``, evicting
    /// the LRU entry if full.  Zero-copy: the cache takes ownership of the
    /// loaded data buffer and the caller's slot now holds the old cache
    /// buffer, ready for reuse as the next pread destination.
    ///
    /// Returns a clone of the now-cached buffer so the caller can use it
    /// directly without a follow-up `lookup` (which would inflate hit stats).
    pub fn insert_swap(&mut self, layer: usize, expert: usize, buf: &mut Buffer) -> Buffer {
        self.access_counter += 1;

        if let Some(&idx) = self.map.get(&(layer, expert)) {
            self.entries[idx].last_used = self.access_counter;
            return self.entries[idx].buffer.clone();
        }

        let target = if self.map.len() < self.entries.len() {
            self.map.len()
        } else {
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

        std::mem::swap(&mut self.entries[target].buffer, buf);
        self.entries[target].layer_idx = layer as i32;
        self.entries[target].expert_idx = expert as i32;
        self.entries[target].last_used = self.access_counter;
        self.map.insert((layer, expert), target);
        self.entries[target].buffer.clone()
    }

    pub fn len(&self) -> usize { self.map.len() }

    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }
}

/// Pre-allocated GPU buffers for expert dispatch — allocated once, reused across
/// all layers and tokens.  Matches C's `buf_multi_expert_*` and scratch buffers.
pub struct ExpertBuffer {
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
    /// Single shared GPU-side LRU cache for expert weights (keyed by (layer, idx)).
    pub cache: Option<ExpertCache>,
}

impl ExpertBuffer {
    pub fn new(
        device: &Device,
        expert_size: usize,
        hidden_dim: usize,
        moe_inter: usize,
        shared_inter: usize,
    ) -> Self {
        let mut expert_data = Vec::with_capacity(MAX_K);
        let mut expert_out = Vec::with_capacity(MAX_K);
        for _ in 0..MAX_K {
            expert_data.push(metal_buf_shared(device, expert_size));
            expert_out.push(metal_buf_shared(device, hidden_dim * 4));
        }
        ExpertBuffer {
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
            cache: None,  // populated by init_expert_buffers if expert_cache_count > 0
        }
    }

    /// Initialize a single shared LRU cache with `size` entries.
    pub fn init_cache(&mut self, device: &Device, size: usize, expert_size: usize) {
        eprintln!("[expert-io] LRU cache: {} entries shared across layers (~{} MB)",
            size, size * expert_size / (1024 * 1024));
        self.cache = Some(ExpertCache::new(device, size, expert_size));
    }
}

/// Report LRU cache statistics.
pub fn report_cache_stats(label: &str, expert_buffer: &ExpertBuffer) {
    if let Some(ref cache) = expert_buffer.cache {
        let total = cache.hits + cache.misses;
        let rate = if total > 0 { 100.0 * cache.hits as f64 / total as f64 } else { 0.0 };
        eprintln!("[cache-{label}] hits={} misses={} hit_rate={:.1}%",
            cache.hits, cache.misses, rate);
    }
}

/// Holds all Metal device state and compute pipeline handles.
pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    #[allow(dead_code)]
    library: Library,

    // Pipeline states for each kernel
    pub matvec_naive: ComputePipelineState,
    pub matvec_fast: ComputePipelineState,
    pub matvec_v3: ComputePipelineState,
    pub matvec_bf16: ComputePipelineState,
    pub matvec_int8: ComputePipelineState,
    pub matvec_fp4_e2m1: Option<ComputePipelineState>,
    pub matvec_fp8_e4m3: Option<ComputePipelineState>,
    pub swiglu: ComputePipelineState,
    pub swiglu_vec4: Option<ComputePipelineState>,
    pub rms_norm_sum: ComputePipelineState,
    pub rms_norm_fused_bf16: Option<ComputePipelineState>,   // single-pass fused RMS norm
    pub rms_norm_apply_bf16: Option<ComputePipelineState>,
    pub residual_add: Option<ComputePipelineState>,
    pub sigmoid_gate: Option<ComputePipelineState>,
    pub moe_combine_residual: Option<ComputePipelineState>,
    pub attn_sdpa_fused: Option<ComputePipelineState>,        // fused online-softmax SDPA
    pub attn_sdpa_block: Option<ComputePipelineState>,       // 2-pass SDPA: block pass
    pub attn_sdpa_reduce: Option<ComputePipelineState>,      // 2-pass SDPA: reduce pass
    pub gated_delta_net_step: Option<ComputePipelineState>,
    pub conv1d_step: Option<ComputePipelineState>,
    pub rms_norm_qk: Option<ComputePipelineState>,
    pub compute_decay_beta: Option<ComputePipelineState>,
    pub gated_rms_norm: Option<ComputePipelineState>,
    pub q_head_norm_rope: Option<ComputePipelineState>,
    pub k_head_norm_rope: Option<ComputePipelineState>,
    pub kv_cache_append: Option<ComputePipelineState>,
    // Batched (`_n`) variants for batched-prefill path
    pub matvec_bf16_n: Option<ComputePipelineState>,
    pub matvec_int8_n: Option<ComputePipelineState>,
    pub dequant_matvec_4bit_n: Option<ComputePipelineState>,
    pub attn_sdpa_causal_n: Option<ComputePipelineState>,
    pub kv_cache_append_n: Option<ComputePipelineState>,

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
    /// Batch projection outputs (for fused CMD1): 8 slots matching C
    pub batch_out: Vec<Buffer>,
    /// CMD3 → next-layer CMD1: normed hidden [hidden_dim] f32
    pub buf_input: Option<Buffer>,
    /// CMD3 RMS norm sum-of-squares scratch [4 bytes]
    pub buf_cmd3_sum_sq: Option<Buffer>,
    /// CMD3 moe_combine_residual output — next layer reads hidden from here (FAST PATH)
    pub buf_moe_hidden: Option<Buffer>,
    // ── Pipelined Fused4bit persistent GPU buffers ──
    /// Gate projection output [num_experts] f32 — read by CPU for routing
    pub buf_gate_scores: Option<Buffer>,
    /// Shared expert gate scalar [1] f32
    pub buf_shared_gate_score: Option<Buffer>,
    /// Post-attn normed hidden [hidden_dim] f32 — expert input for next cmd buf
    pub buf_post_normed: Option<Buffer>,
    /// Shared gate proj [shared_inter] f32 — stays on GPU for next cmd buf
    pub buf_shared_gate: Option<Buffer>,
    /// Shared up proj [shared_inter] f32 — stays on GPU for next cmd buf
    pub buf_shared_up: Option<Buffer>,
    /// out_proj output [hidden_dim] f32 — scratch for pre_expert
    pub buf_out_proj: Option<Buffer>,
    /// residual_add output [hidden_dim] f32 — scratch for pre_expert
    pub buf_temp_residual: Option<Buffer>,
    /// Post-attn norm sum_sq [4] f32 — scratch for pre_expert
    pub buf_post_sum_sq: Option<Buffer>,
    /// Residual (h_mid) upload buffer [hidden_dim] f32 — used in CMD2 for residual_add
    pub buf_residual: Option<Buffer>,
    // ── Persistent KV cache buffers for full attention (one per full-attn layer) ──
    /// K cache buffers: [MAX_SEQ * kv_dim] f32 each, one per full-attention layer
    pub buf_kv_k: Vec<Buffer>,
    /// V cache buffers: [MAX_SEQ * kv_dim] f32 each
    pub buf_kv_v: Vec<Buffer>,
    // ── Pre-allocated full-attention buffers (match C's buf_attn_*) ──
    /// Q buffer [num_attn_heads * head_dim] f32
    pub buf_attn_q: Option<Buffer>,
    /// Q-gate buffer [num_attn_heads * head_dim] f32
    pub buf_attn_q_gate: Option<Buffer>,
    /// Attention output [num_attn_heads * head_dim] f32
    pub buf_attn_out: Option<Buffer>,
    /// 2-pass SDPA partials [num_attn_heads * ceil(MAX_SEQ/32) * (2 + head_dim)] f32
    pub buf_attn_partials: Option<Buffer>,
    // ── Pre-allocated QKV projection buffers (match C's cs->x_buf/qbuf/kbuf/vbuf) ──
    /// Input normed hidden [hidden_dim] f32
    pub buf_qkv_x: Option<Buffer>,
    /// Q projection output [q_proj_dim] f32
    pub buf_qkv_q: Option<Buffer>,
    /// K projection output [kv_dim] f32
    pub buf_qkv_k: Option<Buffer>,
    /// V projection output [kv_dim] f32
    pub buf_qkv_v: Option<Buffer>,
    // ── Cache sync metadata ──
    pub pos: std::cell::Cell<usize>,
    kv_dim: usize,
    num_layers: usize,
}

impl MetalContext {
    /// Allocate persistent GPU buffers for fused linear attention.
    /// Must be called after model config is loaded.
    #[allow(clippy::too_many_arguments)]
    pub fn init_linear_attn_buffers(
        &mut self,
        num_linear_layers: usize,
        qkv_dim: usize,
        num_v_heads: usize,
        total_value: usize,
        key_dim: usize,
        value_dim: usize,
        hidden_dim: usize,
        num_experts: usize,
        shared_intermediate: usize,
        num_full_attn_layers: usize,
        kv_dim: usize,
        num_attn_heads: usize,
        head_dim: usize,
        q_proj_dim: usize,
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
        // 8 batch outputs matching C: [0]=qkv, [1]=z, [2]=beta, [3]=alpha,
        // [4..5]=unused, [6]=gated_rms_norm output, [7]=unused
        self.batch_out.clear();
        self.batch_out.push(metal_buf_shared(&self.device, qkv_dim * 4));       // 0: qkv
        self.batch_out.push(metal_buf_shared(&self.device, total_value * 4));   // 1: z
        self.batch_out.push(metal_buf_shared(&self.device, num_v_heads * 4));   // 2: beta
        self.batch_out.push(metal_buf_shared(&self.device, num_v_heads * 4));   // 3: alpha
        self.batch_out.push(metal_buf_shared(&self.device, 4));                 // 4: unused
        self.batch_out.push(metal_buf_shared(&self.device, 4));                 // 5: unused
        self.batch_out.push(metal_buf_shared(&self.device, total_value * 4));   // 6: gated_rms_norm output
        self.batch_out.push(metal_buf_shared(&self.device, 4));                 // 7: unused
        // CMD3 GPU-side combine + input_norm buffers (matches C buf_cmd3_sum_sq, buf_input, buf_moe_hidden)
        self.buf_cmd3_sum_sq = Some(metal_buf_shared(&self.device, 4));
        self.buf_input = Some(metal_buf_shared(&self.device, hidden_dim * 4));
        self.buf_moe_hidden = Some(metal_buf_shared(&self.device, hidden_dim * 4));
        // Pipelined Fused4bit buffers
        self.buf_gate_scores = Some(metal_buf_shared(&self.device, num_experts * 4));
        self.buf_shared_gate_score = Some(metal_buf_shared(&self.device, 4));
        self.buf_post_normed = Some(metal_buf_shared(&self.device, hidden_dim * 4));
        self.buf_shared_gate = Some(metal_buf_shared(&self.device, shared_intermediate * 4));
        self.buf_shared_up = Some(metal_buf_shared(&self.device, shared_intermediate * 4));
        self.buf_out_proj = Some(metal_buf_shared(&self.device, hidden_dim * 4));
        self.buf_temp_residual = Some(metal_buf_shared(&self.device, hidden_dim * 4));
        self.buf_post_sum_sq = Some(metal_buf_shared(&self.device, 4));
        self.buf_residual = Some(metal_buf_shared(&self.device, hidden_dim * 4));
        // Persistent KV cache buffers for full attention — avoids per-call
        // allocation and full-history copy.  Matches C's buf_kv_k / buf_kv_v.
        self.buf_kv_k.clear();
        self.buf_kv_v.clear();
        let kv_buf_size = crate::constants::MAX_SEQ * kv_dim * 4;
        self.kv_dim = kv_dim;
        self.num_layers = num_full_attn_layers + num_linear_layers;
        eprintln!("[metal] Allocating {} full-attn KV buffers ({} MB each)",
            num_full_attn_layers, kv_buf_size / (1024 * 1024));
        for _ in 0..num_full_attn_layers {
            self.buf_kv_k.push(metal_buf_shared(&self.device, kv_buf_size));
            self.buf_kv_v.push(metal_buf_shared(&self.device, kv_buf_size));
        }
        // Pre-allocated full-attention GPU buffers (match C's buf_attn_*).
        // Shared across all full-attention layers within a token since processing is sequential.
        let q_dim = num_attn_heads * head_dim;
        self.buf_attn_q = Some(metal_buf_shared(&self.device, q_dim * 4));
        self.buf_attn_q_gate = Some(metal_buf_shared(&self.device, q_dim * 4));
        self.buf_attn_out = Some(metal_buf_shared(&self.device, q_dim * 4));
        // 2-pass SDPA partials — sized for the worst case (MAX_SEQ tokens) so
        // we don't allocate a fresh Metal buffer per full-attn layer per token.
        let max_blocks = (crate::constants::MAX_SEQ + 31) / 32;
        let partials_stride = 2 + head_dim;
        let partials_size = num_attn_heads * max_blocks * partials_stride * 4;
        self.buf_attn_partials = Some(metal_buf_shared(&self.device, partials_size));
        // Pre-allocated QKV projection buffers — reused across all full-attention layers.
        self.buf_qkv_x = Some(metal_buf_shared(&self.device, hidden_dim * 4));
        self.buf_qkv_q = Some(metal_buf_shared(&self.device, q_proj_dim * 4));
        self.buf_qkv_k = Some(metal_buf_shared(&self.device, kv_dim * 4));
        self.buf_qkv_v = Some(metal_buf_shared(&self.device, kv_dim * 4));
    }

    /// Allocate persistent GPU buffers for expert I/O. Returns the state which
    /// should be stored separately (in ModelState) to allow independent borrowing.
    pub fn init_expert_buffers(
        &self,
        expert_size: usize,
        hidden_dim: usize,
        moe_inter: usize,
        shared_inter: usize,
        _num_layers: usize,
        expert_cache_count: usize,
    ) -> ExpertBuffer {
        let mut io = ExpertBuffer::new(
            &self.device,
            expert_size,
            hidden_dim,
            moe_inter,
            shared_inter,
        );
        if expert_cache_count > 0 {
            io.init_cache(&self.device, expert_cache_count, expert_size);
        }
        eprintln!(
            "[expert-io] Pre-allocated {}x data bufs ({} MB each), {}x scratch",
            MAX_K,
            expert_size / (1024 * 1024),
            MAX_K,
        );
        io
    }
}

impl MetalContext {
    /// One-shot initialization: create MetalContext, init buffers, wrap weight file.
    /// Returns (ctx, weight_buffer, expert_buffer) ready for engine construction.
    pub fn new<C: crate::engine::qwen35_constants::ModelConfig>(
        weight_file: &WeightFile,
        num_active_experts: usize,
        label: &str,
        expert_cache_count: usize,
    ) -> Result<(Self, WeightBuffer, ExpertBuffer), MoEError> {
        let num_active = if num_active_experts == 0 { C::NUM_EXPERTS_PER_TOK } else { num_active_experts };
        if num_active > C::NUM_EXPERTS_PER_TOK {
            return Err(MoEError::Config(format!(
                "num_active_experts ({}) must not exceed model's num_experts_per_tok ({})",
                num_active, C::NUM_EXPERTS_PER_TOK
            )));
        }
        if num_active > MAX_K {
            return Err(MoEError::Config(format!(
                "num_active_experts ({}) exceeds engine MAX_K ({}); raise MAX_K and the \
                 moe_combine_residual shader's expert-buffer count to support more",
                num_active, MAX_K
            )));
        }

        let mut ctx = Self::init()?;
        ctx.init_linear_attn_buffers(
            C::NUM_LINEAR_LAYERS, C::LINEAR_CONV_DIM, C::LINEAR_NUM_V_HEADS,
            C::LINEAR_TOTAL_VALUE, C::LINEAR_KEY_DIM, C::LINEAR_VALUE_DIM,
            C::HIDDEN_DIM, C::NUM_EXPERTS, C::SHARED_INTERMEDIATE,
            C::NUM_FULL_ATTN_LAYERS, C::KV_DIM,
            C::NUM_ATTN_HEADS, C::HEAD_DIM,
            C::NUM_ATTN_HEADS * 2 * C::HEAD_DIM,
        );
        let expert_buffer = ctx.init_expert_buffers(
            C::EXPERT_SIZE_4BIT, C::HIDDEN_DIM, C::MOE_INTERMEDIATE, C::SHARED_INTERMEDIATE,
            C::NUM_LAYERS, expert_cache_count,
        );
        let weight_buffer = WeightBuffer::new(&ctx.device, weight_file);

        eprintln!(
            "[engine] {} layers hidden={} experts={} mode={}",
            C::NUM_LAYERS, C::HIDDEN_DIM, C::NUM_EXPERTS, label
        );

        Ok((ctx, weight_buffer, expert_buffer))
    }
}

/// Embed the shaders.metal source at compile time.
const SHADER_SOURCE: &str = include_str!("shaders.metal");

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
            let t_compile = crate::timer::now();
            let library = device
                .new_library_with_source(SHADER_SOURCE, &compile_opts)
                .map_err(|e| MoEError::Shader(format!("Shader compilation failed: {:?}", e)))?;
            eprintln!("[metal] Shader compilation: {:.0} ms", crate::timer::now().duration_since(t_compile).as_secs_f64() * 1000.0);

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
            let matvec_bf16 = make_pipeline("matvec_bf16")?;
            let matvec_int8 = make_pipeline("matvec_int8")?;
            let matvec_fp4_e2m1 = make_pipeline("dequant_matvec_fp4_e2m1").ok();
            let matvec_fp8_e4m3 = make_pipeline("matvec_fp8_e4m3").ok();
            let swiglu = make_pipeline("swiglu_fused")?;
            let rms_norm_sum = make_pipeline("rms_norm_sum_sq")?;

            // Optional pipelines
            let matvec_fast = make_pipeline("dequant_matvec_4bit_fast").ok();
            let swiglu_vec4 = make_pipeline("swiglu_fused_vec4").ok();
            let rms_norm_fused_bf16 = make_pipeline("rms_norm_fused_bf16").ok();
            let rms_norm_apply_bf16 = make_pipeline("rms_norm_apply_bf16").ok();
            let residual_add = make_pipeline("residual_add").ok();
            let sigmoid_gate = make_pipeline("sigmoid_gate").ok();
            let moe_combine_residual = make_pipeline("moe_combine_residual").ok();
            let attn_sdpa_fused = make_pipeline("attn_sdpa_fused").ok();
            let attn_sdpa_block  = make_pipeline("attn_sdpa_block").ok();
            let attn_sdpa_reduce = make_pipeline("attn_sdpa_reduce").ok();
            let gated_delta_net_step = make_pipeline("gated_delta_net_step").ok();
            let conv1d_step = make_pipeline("conv1d_step").ok();
            let rms_norm_qk = make_pipeline("rms_norm_qk").ok();
            let compute_decay_beta = make_pipeline("compute_decay_beta").ok();
            let gated_rms_norm = make_pipeline("gated_rms_norm").ok();
            let q_head_norm_rope = make_pipeline("q_head_norm_rope").ok();
            let k_head_norm_rope = make_pipeline("k_head_norm_rope").ok();
            let kv_cache_append = make_pipeline("kv_cache_append").ok();
            let matvec_bf16_n = make_pipeline("matvec_bf16_n").ok();
            let matvec_int8_n = make_pipeline("matvec_int8_n").ok();
            let dequant_matvec_4bit_n = make_pipeline("dequant_matvec_4bit_n").ok();
            let attn_sdpa_causal_n = make_pipeline("attn_sdpa_causal_n").ok();
            let kv_cache_append_n = make_pipeline("kv_cache_append_n").ok();

            // Validate required pipelines exist
            let matvec_fast = matvec_fast.ok_or_else(|| MoEError::Shader("dequant_matvec_4bit_fast not found".into()))?;

            eprintln!("[metal] All pipelines created successfully");

            Ok(MetalContext {
                device,
                queue,
                library,
                matvec_naive,
                matvec_fast,
                matvec_v3: matvec_v3.clone(),
                matvec_bf16,
                matvec_int8,
                matvec_fp4_e2m1,
                matvec_fp8_e4m3,
                swiglu,
                swiglu_vec4,
                rms_norm_sum,
                rms_norm_fused_bf16,
                rms_norm_apply_bf16,
                residual_add,
                sigmoid_gate,
                moe_combine_residual,
                attn_sdpa_fused,
                attn_sdpa_block,
                attn_sdpa_reduce,
                gated_delta_net_step,
                conv1d_step,
                rms_norm_qk,
                compute_decay_beta,
                gated_rms_norm,
                q_head_norm_rope,
                k_head_norm_rope,
                kv_cache_append,
                matvec_bf16_n,
                matvec_int8_n,
                dequant_matvec_4bit_n,
                attn_sdpa_causal_n,
                kv_cache_append_n,
                // Persistent GPU buffers — initialized later via init_linear_attn_buffers()
                buf_conv_state: Vec::new(),
                buf_delta_state: Vec::new(),
                buf_conv_output: None,
                buf_delta_g_decay: None,
                buf_delta_beta: None,
                buf_delta_output: None,
                batch_out: Vec::new(),
                buf_input: None,
                buf_cmd3_sum_sq: None,
                buf_moe_hidden: None,
                buf_gate_scores: None,
                buf_shared_gate_score: None,
                buf_post_normed: None,
                buf_shared_gate: None,
                buf_shared_up: None,
                buf_out_proj: None,
                buf_temp_residual: None,
                buf_post_sum_sq: None,
                buf_residual: None,
                buf_kv_k: Vec::new(),
                buf_kv_v: Vec::new(),
                buf_attn_q: None,
                buf_attn_q_gate: None,
                buf_attn_out: None,
                buf_attn_partials: None,
                buf_qkv_x: None,
                buf_qkv_q: None,
                buf_qkv_k: None,
                buf_qkv_v: None,
                pos: std::cell::Cell::new(0),
                kv_dim: 0,
                num_layers: 0,
            })
        })
    }
}

// ─── Cache ↔ GPU sync ─────────────────────────────────────────────────────

impl MetalContext {
    /// Upload CPU cache state → GPU buffers (restoring from persistent state).
    pub fn upload_cache(&self, cache: &Cache) {
        assert!(self.num_layers > 0, "upload_cache called before init_linear_attn_buffers");
        self.pos.set(cache.pos);
        for layer in 0..self.num_layers {
            if (layer + 1) % FULL_ATTN_INTERVAL == 0 {
                let fa_idx = layer / FULL_ATTN_INTERVAL;
                let kv = cache.full(layer);
                let n = kv.len * self.kv_dim;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        kv.k_cache.as_ptr(),
                        self.buf_kv_k[fa_idx].contents() as *mut f32,
                        n,
                    );
                    std::ptr::copy_nonoverlapping(
                        kv.v_cache.as_ptr(),
                        self.buf_kv_v[fa_idx].contents() as *mut f32,
                        n,
                    );
                }
            } else {
                let li = layer - (layer + 1) / FULL_ATTN_INTERVAL;
                let lin = cache.lin(layer);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        lin.ssm_state.as_ptr(),
                        self.buf_delta_state[li].contents() as *mut f32,
                        lin.ssm_state.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        lin.conv_state.as_ptr(),
                        self.buf_conv_state[li].contents() as *mut f32,
                        lin.conv_state.len(),
                    );
                }
            }
        }
    }

    /// Download GPU state → CPU cache (for persistence). Sets cache.pos.
    pub fn download_cache(&self, cache: &mut Cache) {
        assert!(self.num_layers > 0, "download_cache called before init_linear_attn_buffers");
        let pos = self.pos.get();
        cache.set_pos(pos);
        for layer in 0..self.num_layers {
            if (layer + 1) % FULL_ATTN_INTERVAL == 0 {
                let fa_idx = layer / FULL_ATTN_INTERVAL;
                unsafe {
                    let kv = cache.full_mut(layer);
                    let n = kv.len * self.kv_dim;
                    std::ptr::copy_nonoverlapping(
                        self.buf_kv_k[fa_idx].contents() as *const f32,
                        kv.k_cache.as_mut_ptr(),
                        n,
                    );
                    std::ptr::copy_nonoverlapping(
                        self.buf_kv_v[fa_idx].contents() as *const f32,
                        kv.v_cache.as_mut_ptr(),
                        n,
                    );
                }
            } else {
                let li = layer - (layer + 1) / FULL_ATTN_INTERVAL;
                unsafe {
                    let lin = cache.lin_mut(layer);
                    std::ptr::copy_nonoverlapping(
                        self.buf_delta_state[li].contents() as *const f32,
                        lin.ssm_state.as_mut_ptr(),
                        lin.ssm_state.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        self.buf_conv_state[li].contents() as *const f32,
                        lin.conv_state.as_mut_ptr(),
                        lin.conv_state.len(),
                    );
                }
            }
        }
    }
}

/// Create a shared-memory Metal buffer (CPU and GPU see the same memory).
pub fn metal_buf_shared(device: &Device, size: usize) -> Buffer {
    device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared)
}

// ─── GPU weight buffer wrapper ─────────────────────────────────────────────

const GPU_MATVEC_GROUP_SIZE: u32 = 64;

/// Wraps the entire model weight file in a Metal buffer (zero-copy via mmap).
/// Tensor matvecs dispatch on GPU using byte offsets within this buffer.
pub struct WeightBuffer {
    pub buf: Buffer,
    pub base: *const u8,
}

impl WeightBuffer {
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
        WeightBuffer { buf, base: data }
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
        let weight_name = format!("{}.weight", prefix);
        let w_ptr = match wf.get_tensor_ptr(&weight_name) {
            Some(p) => p,
            None => {
                eprintln!("[encode_matvec_into] WARNING: tensor not found: {}.weight", prefix);
                return false;
            }
        };
        let w_off = (w_ptr as usize - self.base as usize) as u64;

        // BQ4 dispatch: parse dtype string → Quant → choose kernel
        let dtype = wf.get_tensor_info(&weight_name)
            .map(|info| info.dtype.as_str())
            .unwrap_or("u32");
        let q = string_to_dtype(dtype);

        match q {
            Some(DType::Bf16) => {
                metal_kernels::encode_matvec_bf16_offset(
                    ctx, encoder,
                    &self.buf, w_off,
                    x_buf, x_offset, out_buf, out_offset,
                    out_dim as u32, in_dim as u32,
                );
                return true;
            }
            Some(DType::Int8) => {
                let s_ptr = match wf.get_tensor_ptr(&format!("{}.scales", prefix)) {
                    Some(p) => p,
                    None => {
                        eprintln!("[encode_matvec_into] WARNING: tensor not found: {}.scales", prefix);
                        return false;
                    }
                };
                let s_off = (s_ptr as usize - self.base as usize) as u64;
                metal_kernels::encode_matvec_int8_offset(
                    ctx, encoder,
                    &self.buf, w_off, &self.buf, s_off,
                    x_buf, x_offset, out_buf, out_offset,
                    out_dim as u32, in_dim as u32,
                );
                return true;
            }
            Some(DType::Fp8E4m3) => {
                let s_ptr = match wf.get_tensor_ptr(&format!("{}.scales", prefix)) {
                    Some(p) => p,
                    None => {
                        eprintln!("[encode_matvec_into] WARNING: tensor not found: {}.scales", prefix);
                        return false;
                    }
                };
                let s_off = (s_ptr as usize - self.base as usize) as u64;
                metal_kernels::encode_matvec_fp8_e4m3_offset(
                    ctx, encoder,
                    &self.buf, w_off, &self.buf, s_off,
                    x_buf, x_offset, out_buf, out_offset,
                    out_dim as u32, in_dim as u32, crate::dtype::FP8_GROUP_SIZE as u32,
                );
                return true;
            }
            Some(DType::Fp4E2m1) => {
                let s_ptr = match wf.get_tensor_ptr(&format!("{}.scales", prefix)) {
                    Some(p) => p,
                    None => {
                        eprintln!("[encode_matvec_into] WARNING: tensor not found: {}.scales", prefix);
                        return false;
                    }
                };
                let s_off = (s_ptr as usize - self.base as usize) as u64;
                metal_kernels::encode_matvec_fp4_e2m1_offset(
                    ctx, encoder,
                    &self.buf, w_off, &self.buf, s_off,
                    x_buf, x_offset, out_buf, out_offset,
                    out_dim as u32, in_dim as u32, GPU_MATVEC_GROUP_SIZE,
                );
                return true;
            }
            Some(DType::Int4) => {
                let s_ptr = match wf.get_tensor_ptr(&format!("{}.scales", prefix)) {
                    Some(p) => p,
                    None => {
                        eprintln!("[encode_matvec_into] WARNING: tensor not found: {}.scales", prefix);
                        return false;
                    }
                };
                let b_ptr = match wf.get_tensor_ptr(&format!("{}.biases", prefix)) {
                    Some(p) => p,
                    None => {
                        eprintln!("[encode_matvec_into] WARNING: tensor not found: {}.biases", prefix);
                        return false;
                    }
                };

                let s_off = (s_ptr as usize - self.base as usize) as u64;
                let b_off = (b_ptr as usize - self.base as usize) as u64;

                metal_kernels::encode_matvec_offset(
                    ctx, encoder,
                    &self.buf, w_off, &self.buf, s_off, &self.buf, b_off,
                    x_buf, x_offset, out_buf, out_offset,
                    out_dim as u32, in_dim as u32, GPU_MATVEC_GROUP_SIZE, 3,
                );
                return true;
            }
            q => {
                let d = q.map(|q| q.as_str()).unwrap_or("unknown");
                eprintln!("[encode_matvec_into] ERROR: unsupported dtype '{}' for tensor {}", d, weight_name);
                return false;
            }
        }
    }

}

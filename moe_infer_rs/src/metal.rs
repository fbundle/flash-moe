// Metal GPU context setup — mirrors metal_setup.h
// Compiles embedded Metal shaders, creates compute pipelines, allocates buffers.

use metal::*;
pub use metal::MTLResourceOptions;
use std::ffi::c_void;

use crate::constants::{MAX_K, USE_KV_CACHE_BF16};
use crate::types::ModelConfig;

const MAX_BATCH_SLOTS: usize = 8;

/// All Metal state for GPU-accelerated inference.
pub struct MetalCtx {
    pub device: Device,
    pub queue: CommandQueue,

    // Compute pipelines
    pub matvec_v3: ComputePipelineState,
    pub matvec_v5: ComputePipelineState,
    pub matvec_fast: ComputePipelineState,
    pub matvec_2bit: Option<ComputePipelineState>,
    pub rms_norm_sum: ComputePipelineState,
    pub rms_norm_apply: ComputePipelineState,
    pub rms_norm_apply_bf16: ComputePipelineState,
    pub residual_add: ComputePipelineState,
    pub swiglu: ComputePipelineState,
    pub attn_scores_pipe: Option<ComputePipelineState>,
    pub attn_softmax_pipe: Option<ComputePipelineState>,
    pub attn_values_pipe: Option<ComputePipelineState>,
    pub sigmoid_gate_pipe: Option<ComputePipelineState>,
    pub moe_combine_residual: Option<ComputePipelineState>,
    pub delta_net_step: Option<ComputePipelineState>,
    pub conv1d_step: Option<ComputePipelineState>,
    pub rms_norm_qk: Option<ComputePipelineState>,
    pub compute_decay_beta: Option<ComputePipelineState>,
    pub gated_rms_norm: Option<ComputePipelineState>,

    // Working buffers
    pub buf_input: Buffer,
    pub buf_output: Buffer,
    pub wf_buf: Option<Buffer>,
    pub batch_out: [Buffer; MAX_BATCH_SLOTS],

    // Expert computation buffers
    pub buf_expert_data: Buffer,
    pub buf_expert_input: Buffer,
    pub buf_expert_gate: Buffer,
    pub buf_expert_up: Buffer,
    pub buf_expert_act: Buffer,
    pub buf_expert_out: Buffer,

    // Multi-expert buffers
    pub buf_multi_expert_data: [Buffer; MAX_K],
    pub buf_multi_expert_data_b: [Buffer; MAX_K],
    pub buf_multi_expert_gate: [Buffer; MAX_K],
    pub buf_multi_expert_up: [Buffer; MAX_K],
    pub buf_multi_expert_act: [Buffer; MAX_K],
    pub buf_multi_expert_out: [Buffer; MAX_K],
    pub buf_multi_expert_input: Buffer,

    // Shared expert
    pub buf_shared_gate: Buffer,
    pub buf_shared_up: Buffer,
    pub buf_shared_act: Buffer,
    pub buf_shared_out: Buffer,

    // Fused CMD2 buffers
    pub buf_residual: Buffer,
    pub buf_h_mid: Buffer,
    pub buf_sum_sq: Buffer,

    // GPU-side combine (CMD3)
    pub buf_moe_hidden: Buffer,
    pub buf_combine_params: Buffer,
    pub buf_cmd3_sum_sq: Buffer,

    // GPU attention
    pub buf_kv_k: Vec<Buffer>,
    pub buf_kv_v: Vec<Buffer>,
    pub buf_attn_q: Buffer,
    pub buf_attn_scores: Buffer,
    pub buf_attn_out: Buffer,
    pub buf_attn_gate: Buffer,

    // Delta-net persistent state
    pub buf_delta_state: Vec<Buffer>,
    pub buf_conv_state: Vec<Buffer>,

    // Delta-net scratch
    pub buf_delta_q: Buffer,
    pub buf_delta_k: Buffer,
    pub buf_delta_v: Buffer,
    pub buf_delta_g_decay: Buffer,
    pub buf_delta_beta: Buffer,
    pub buf_delta_output: Buffer,
    pub buf_conv_input: Buffer,
    pub buf_conv_output: Buffer,

    // Event pipeline (MTLSharedEvent for non-blocking GPU completion check)
    pub pipeline_event: SharedEvent,
}

// Embedded Metal Shading Language source.
// Kept as a const string — the C version uses gen_shaders.py to embed it.
// For now, this is a placeholder; the actual shaders would be set at build time.
const SHADER_SOURCE: &str = include_str!("../../moe_infer_mlx/core_src/shaders.metal");

fn make_pipe(library: &Library, device: &Device, name: &str) -> Option<ComputePipelineState> {
    let func = library.get_function(name, None).ok()?;
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&func));
    match device.new_compute_pipeline_state(&desc) {
        Ok(ps) => Some(ps),
        Err(e) => {
            eprintln!("[metal] WARNING: pipeline '{}' failed: {:?}", name, e);
            None
        }
    }
}

impl MetalCtx {
    pub fn new(cfg: &ModelConfig) -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "ERROR: No Metal device found".to_string())?;
        println!("[metal] Device: {}", device.name());

        let queue = device.new_command_queue();

        // Compile shaders
        let opts = CompileOptions::new();
        opts.set_language_version(MTLLanguageVersion::V3_1);
        opts.set_fast_math_enabled(true);

        let library = device
            .new_library_with_source(SHADER_SOURCE, &opts)
            .map_err(|e| format!("ERROR: Shader compile failed: {:?}", e))?;

        // Create pipelines
        let matvec_v3 = make_pipe(&library, &device, "dequant_matvec_4bit_v3")
            .ok_or_else(|| "ERROR: Required pipeline missing: dequant_matvec_4bit_v3".to_string())?;
        let matvec_v5 = make_pipe(&library, &device, "dequant_matvec_4bit_v5")
            .unwrap_or_else(|| matvec_v3.clone());
        let matvec_fast = make_pipe(&library, &device, "dequant_matvec_4bit_fast")
            .ok_or_else(|| "ERROR: Required pipeline missing: dequant_matvec_4bit_fast".to_string())?;
        let matvec_2bit = make_pipe(&library, &device, "dequant_matvec_2bit");
        let rms_norm_sum = make_pipe(&library, &device, "rms_norm_sum_sq")
            .ok_or_else(|| "ERROR: Required pipeline missing: rms_norm_sum_sq".to_string())?;
        let rms_norm_apply = make_pipe(&library, &device, "rms_norm_apply")
            .ok_or_else(|| "ERROR: Required pipeline missing: rms_norm_apply".to_string())?;
        let rms_norm_apply_bf16 = make_pipe(&library, &device, "rms_norm_apply_bf16")
            .ok_or_else(|| "ERROR: Required pipeline missing: rms_norm_apply_bf16".to_string())?;
        let residual_add = make_pipe(&library, &device, "residual_add")
            .ok_or_else(|| "ERROR: Required pipeline missing: residual_add".to_string())?;
        let swiglu = make_pipe(&library, &device, "swiglu_fused")
            .ok_or_else(|| "ERROR: Required pipeline missing: swiglu_fused".to_string())?;

        let attn_scores_pipe = make_pipe(&library, &device, "attn_scores_batched");
        let attn_softmax_pipe = make_pipe(&library, &device, "attn_softmax_batched");
        let attn_values_pipe = make_pipe(&library, &device, "attn_values_batched");
        let sigmoid_gate_pipe = make_pipe(&library, &device, "sigmoid_gate");
        let moe_combine_residual = make_pipe(&library, &device, "moe_combine_residual");
        let delta_net_step = make_pipe(&library, &device, "gated_delta_net_step");
        let conv1d_step = make_pipe(&library, &device, "conv1d_step");
        let rms_norm_qk = make_pipe(&library, &device, "rms_norm_qk");
        let compute_decay_beta = make_pipe(&library, &device, "compute_decay_beta");
        let gated_rms_norm = make_pipe(&library, &device, "gated_rms_norm");

        // Allocate buffers
        let hd = cfg.hidden_dim as u64;
        let max_out = cfg.vocab_size as u64 * std::mem::size_of::<f32>() as u64;
        let max_in = (cfg.linear_total_value as u64)
            .max(cfg.num_attn_heads as u64 * cfg.head_dim as u64)
            * std::mem::size_of::<f32>() as u64;

        let buf_input = device.new_buffer(max_in, MTLResourceOptions::StorageModeShared);
        let buf_output = device.new_buffer(max_out, MTLResourceOptions::StorageModeShared);

        // Batch output slots
        let slot_size = ((cfg.num_attn_heads * cfg.head_dim * 2) as u64)
            .max(cfg.linear_conv_dim as u64)
            * std::mem::size_of::<f32>() as u64;
        let batch_out = std::array::from_fn(|_| {
            device.new_buffer(slot_size, MTLResourceOptions::StorageModeShared)
        });

        // Expert buffers
        let buf_expert_data = device.new_buffer(cfg.expert_size_4bit as u64, MTLResourceOptions::StorageModeShared);
        let buf_expert_input = device.new_buffer(hd * 4, MTLResourceOptions::StorageModeShared);
        let mi = cfg.moe_intermediate as u64 * 4;
        let buf_expert_gate = device.new_buffer(mi, MTLResourceOptions::StorageModeShared);
        let buf_expert_up = device.new_buffer(mi, MTLResourceOptions::StorageModeShared);
        let buf_expert_act = device.new_buffer(mi, MTLResourceOptions::StorageModeShared);
        let buf_expert_out = device.new_buffer(hd * 4, MTLResourceOptions::StorageModeShared);

        // Multi-expert buffers (8 slots)
        let expert_alloc_size = ((cfg.expert_size_4bit as u64 + 2 * 1024 * 1024 - 1) / (2 * 1024 * 1024)) * (2 * 1024 * 1024);
        let buf_multi_expert_data = std::array::from_fn(|_| {
            device.new_buffer(expert_alloc_size, MTLResourceOptions::StorageModeShared)
        });
        let buf_multi_expert_data_b = std::array::from_fn(|_| {
            device.new_buffer(expert_alloc_size, MTLResourceOptions::StorageModeShared)
        });
        let buf_multi_expert_gate = std::array::from_fn(|_| {
            device.new_buffer(mi, MTLResourceOptions::StorageModeShared)
        });
        let buf_multi_expert_up = std::array::from_fn(|_| {
            device.new_buffer(mi, MTLResourceOptions::StorageModeShared)
        });
        let buf_multi_expert_act = std::array::from_fn(|_| {
            device.new_buffer(mi, MTLResourceOptions::StorageModeShared)
        });
        let buf_multi_expert_out = std::array::from_fn(|_| {
            device.new_buffer(hd * 4, MTLResourceOptions::StorageModeShared)
        });
        let buf_multi_expert_input = device.new_buffer(hd * 4, MTLResourceOptions::StorageModeShared);

        // Shared expert
        let si = cfg.shared_intermediate as u64 * 4;
        let buf_shared_gate = device.new_buffer(si, MTLResourceOptions::StorageModeShared);
        let buf_shared_up = device.new_buffer(si, MTLResourceOptions::StorageModeShared);
        let buf_shared_act = device.new_buffer(si, MTLResourceOptions::StorageModeShared);
        let buf_shared_out = device.new_buffer(hd * 4, MTLResourceOptions::StorageModeShared);

        // Fused CMD2 buffers
        let buf_residual = device.new_buffer(hd * 4, MTLResourceOptions::StorageModeShared);
        let buf_h_mid = device.new_buffer(hd * 4, MTLResourceOptions::StorageModeShared);
        let buf_sum_sq = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        // GPU combine buffers
        let buf_moe_hidden = device.new_buffer(hd * 4, MTLResourceOptions::StorageModeShared);
        let buf_combine_params = device.new_buffer(10 * 4, MTLResourceOptions::StorageModeShared);
        let buf_cmd3_sum_sq = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        // GPU attention KV caches
        let kv_dim = (cfg.num_kv_heads * cfg.head_dim) as u64;
        let elem_size = if USE_KV_CACHE_BF16 { 2u64 } else { 4u64 };
        let kv_cache_size = cfg.gpu_kv_seq as u64 * kv_dim * elem_size;

        let mut buf_kv_k = Vec::with_capacity(cfg.num_full_attn_layers as usize);
        let mut buf_kv_v = Vec::with_capacity(cfg.num_full_attn_layers as usize);
        for _ in 0..cfg.num_full_attn_layers {
            buf_kv_k.push(device.new_buffer(kv_cache_size, MTLResourceOptions::StorageModeShared));
            buf_kv_v.push(device.new_buffer(kv_cache_size, MTLResourceOptions::StorageModeShared));
        }

        let q_dim = (cfg.num_attn_heads * cfg.head_dim) as u64 * 4;
        let buf_attn_q = device.new_buffer(q_dim, MTLResourceOptions::StorageModeShared);
        let scores_size = cfg.num_attn_heads as u64 * cfg.gpu_kv_seq as u64 * 4;
        let buf_attn_scores = device.new_buffer(scores_size, MTLResourceOptions::StorageModeShared);
        let buf_attn_out = device.new_buffer(q_dim, MTLResourceOptions::StorageModeShared);
        let buf_attn_gate = device.new_buffer(q_dim, MTLResourceOptions::StorageModeShared);

        // Delta-net state buffers
        let delta_state_sz = cfg.linear_num_v_heads as u64 * cfg.linear_value_dim as u64 * cfg.linear_key_dim as u64 * 4;
        let conv_state_sz = 3 * cfg.linear_conv_dim as u64 * 4;
        let mut buf_delta_state = Vec::with_capacity(cfg.num_linear_layers as usize);
        let mut buf_conv_state = Vec::with_capacity(cfg.num_linear_layers as usize);
        for _ in 0..cfg.num_linear_layers {
            buf_delta_state.push(device.new_buffer(delta_state_sz, MTLResourceOptions::StorageModeShared));
            buf_conv_state.push(device.new_buffer(conv_state_sz, MTLResourceOptions::StorageModeShared));
        }

        // Delta-net scratch
        let buf_delta_q = device.new_buffer(cfg.linear_total_key as u64 * 4, MTLResourceOptions::StorageModeShared);
        let buf_delta_k = device.new_buffer(cfg.linear_total_key as u64 * 4, MTLResourceOptions::StorageModeShared);
        let buf_delta_v = device.new_buffer(cfg.linear_total_value as u64 * 4, MTLResourceOptions::StorageModeShared);
        let buf_delta_g_decay = device.new_buffer(cfg.linear_num_v_heads as u64 * 4, MTLResourceOptions::StorageModeShared);
        let buf_delta_beta = device.new_buffer(cfg.linear_num_v_heads as u64 * 4, MTLResourceOptions::StorageModeShared);
        let buf_delta_output = device.new_buffer(cfg.linear_total_value as u64 * 4, MTLResourceOptions::StorageModeShared);
        let buf_conv_input = device.new_buffer(cfg.linear_conv_dim as u64 * 4, MTLResourceOptions::StorageModeShared);
        let buf_conv_output = device.new_buffer(cfg.linear_conv_dim as u64 * 4, MTLResourceOptions::StorageModeShared);

        // Shared event for non-blocking GPU completion check
        let pipeline_event = device.new_shared_event();

        Ok(Self {
            device,
            queue,
            matvec_v3,
            matvec_v5,
            matvec_fast,
            matvec_2bit,
            rms_norm_sum,
            rms_norm_apply,
            rms_norm_apply_bf16,
            residual_add,
            swiglu,
            attn_scores_pipe,
            attn_softmax_pipe,
            attn_values_pipe,
            sigmoid_gate_pipe,
            moe_combine_residual,
            delta_net_step,
            conv1d_step,
            rms_norm_qk,
            compute_decay_beta,
            gated_rms_norm,
            buf_input,
            buf_output,
            wf_buf: None,
            batch_out,
            buf_expert_data,
            buf_expert_input,
            buf_expert_gate,
            buf_expert_up,
            buf_expert_act,
            buf_expert_out,
            buf_multi_expert_data,
            buf_multi_expert_data_b,
            buf_multi_expert_gate,
            buf_multi_expert_up,
            buf_multi_expert_act,
            buf_multi_expert_out,
            buf_multi_expert_input,
            buf_shared_gate,
            buf_shared_up,
            buf_shared_act,
            buf_shared_out,
            buf_residual,
            buf_h_mid,
            buf_sum_sq,
            buf_moe_hidden,
            buf_combine_params,
            buf_cmd3_sum_sq,
            buf_kv_k,
            buf_kv_v,
            buf_attn_q,
            buf_attn_scores,
            buf_attn_out,
            buf_attn_gate,
            buf_delta_state,
            buf_conv_state,
            buf_delta_q,
            buf_delta_k,
            buf_delta_v,
            buf_delta_g_decay,
            buf_delta_beta,
            buf_delta_output,
            buf_conv_input,
            buf_conv_output,
            pipeline_event,
        })
    }

    /// Wrap the weight file data as a zero-copy Metal buffer.
    pub fn set_weights(&mut self, data: *mut u8, size: usize) {
        let page_size = 16384;
        let aligned_size = (size + page_size - 1) & !(page_size - 1);

        let buf = self.device.new_buffer_with_bytes_no_copy(
            data as *const c_void,
            aligned_size as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        self.wf_buf = Some(buf);
        println!(
            "[metal] Weight file wrapped as Metal buffer ({:.2} GB)",
            aligned_size as f64 / 1e9
        );
    }

    /// Reset delta-net and conv GPU state buffers.
    pub fn reset_delta_net_state(&self, cfg: &ModelConfig) {
        for i in 0..cfg.num_linear_layers as usize {
            unsafe {
                let ds_ptr = self.buf_delta_state[i].contents() as *mut u8;
                let ds_len = self.buf_delta_state[i].length() as usize;
                std::ptr::write_bytes(ds_ptr, 0, ds_len);

                let cs_ptr = self.buf_conv_state[i].contents() as *mut u8;
                let cs_len = self.buf_conv_state[i].length() as usize;
                std::ptr::write_bytes(cs_ptr, 0, cs_len);
            }
        }
    }
}

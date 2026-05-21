/// Metal context management: device, queue, shader library, pipeline states.
///
/// Port of metal_init() / MetalContext from main.m:172-322.
use metal::*;
use objc::rc::autoreleasepool;
use std::ffi::c_void;

use crate::error::MoEError;
use crate::kernels;
use crate::weights::WeightFile;

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

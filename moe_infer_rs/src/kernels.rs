/// GPU kernel dispatch wrappers.
///
/// Port of metal_dequant_matvec, metal_swiglu, encode_matvec_v3, etc. from main.m.
use metal::*;
use std::ffi::c_void;

use crate::constants::{ROWS_PER_TG, TG_SIZE};
use crate::metal_context::MetalContext;

/// Helper: set a u32 constant buffer value at the given index.
unsafe fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    let val: *const u32 = &value;
    encoder.set_bytes(index, 4, val as *const c_void);
}

/// Helper: set an f32 constant buffer value at the given index.
unsafe fn set_f32(encoder: &ComputeCommandEncoderRef, index: u64, value: f32) {
    let val: *const f32 = &value;
    encoder.set_bytes(index, 4, val as *const c_void);
}

// ---------------------------------------------------------------------------
// Dequant matvec (v3 / fast / naive)
// ---------------------------------------------------------------------------

/// Encode a dequant matvec dispatch using the v3 (tiled SIMD) shader.
pub fn encode_matvec_v3(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    w_packed: &BufferRef, w_offset: u64,
    scales: &BufferRef, s_offset: u64,
    biases: &BufferRef, b_offset: u64,
    x: &BufferRef, x_offset: u64,
    out: &BufferRef, o_offset: u64,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
) {
    encoder.set_compute_pipeline_state(&ctx.matvec_v3);
    encoder.set_buffer(0, Some(w_packed), w_offset);
    encoder.set_buffer(1, Some(scales), s_offset);
    encoder.set_buffer(2, Some(biases), b_offset);
    encoder.set_buffer(3, Some(x), x_offset);
    encoder.set_buffer(4, Some(out), o_offset);
    unsafe {
        set_u32(encoder, 5, out_dim);
        set_u32(encoder, 6, in_dim);
        set_u32(encoder, 7, group_size);
    }

    let num_tgs = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs as u64, 1, 1),
        MTLSize::new(TG_SIZE as u64, 1, 1),
    );
}

/// Dispatch a dequant matvec (standalone — creates a new command buffer).
pub fn metal_dequant_matvec(
    ctx: &MetalContext,
    w_packed: &Buffer,
    scales: &Buffer,
    biases: &Buffer,
    x: &Buffer,
    out: &Buffer,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    use_fast: i32,
) {
    let cmd_buf = ctx.queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();

    let pipeline = if use_fast >= 3 {
        &ctx.matvec_v3
    } else if use_fast >= 1 {
        &ctx.matvec_fast
    } else {
        &ctx.matvec_naive
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(w_packed), 0);
    encoder.set_buffer(1, Some(scales), 0);
    encoder.set_buffer(2, Some(biases), 0);
    encoder.set_buffer(3, Some(x), 0);
    encoder.set_buffer(4, Some(out), 0);
    unsafe {
        set_u32(encoder, 5, out_dim);
        set_u32(encoder, 6, in_dim);
        set_u32(encoder, 7, group_size);
    }

    if use_fast >= 3 {
        let num_tgs = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(TG_SIZE as u64, 1, 1),
        );
    } else {
        let tg_size: u64 = if use_fast >= 1 { 64 } else { 256 };
        encoder.dispatch_thread_groups(
            MTLSize::new(out_dim as u64, 1, 1),
            MTLSize::new(tg_size, 1, 1),
        );
    }
    encoder.end_encoding();

    cmd_buf.commit();
    cmd_buf.wait_until_completed();
}

/// Encode a dequant matvec with buffer offsets (called within an existing encoder).
pub fn encode_matvec_offset(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    w_packed: &BufferRef, w_offset: u64,
    scales: &BufferRef, s_offset: u64,
    biases: &BufferRef, b_offset: u64,
    x: &BufferRef, x_offset: u64,
    out: &BufferRef, o_offset: u64,
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    use_fast: i32,
) {
    let pipeline = if use_fast >= 3 {
        &ctx.matvec_v3
    } else if use_fast >= 1 {
        &ctx.matvec_fast
    } else {
        &ctx.matvec_naive
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(w_packed), w_offset);
    encoder.set_buffer(1, Some(scales), s_offset);
    encoder.set_buffer(2, Some(biases), b_offset);
    encoder.set_buffer(3, Some(x), x_offset);
    encoder.set_buffer(4, Some(out), o_offset);
    unsafe {
        set_u32(encoder, 5, out_dim);
        set_u32(encoder, 6, in_dim);
        set_u32(encoder, 7, group_size);
    }

    if use_fast >= 3 {
        let num_tgs = (out_dim + ROWS_PER_TG - 1) / ROWS_PER_TG;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(TG_SIZE as u64, 1, 1),
        );
    } else {
        let tg_size: u64 = if use_fast >= 1 { 64 } else { 256 };
        encoder.dispatch_thread_groups(
            MTLSize::new(out_dim as u64, 1, 1),
            MTLSize::new(tg_size, 1, 1),
        );
    }
}

// ---------------------------------------------------------------------------
// SwiGLU
// ---------------------------------------------------------------------------

/// Encode SwiGLU into an existing encoder.
pub fn encode_swiglu(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    gate: &BufferRef, gate_offset: u64,
    up: &BufferRef, up_offset: u64,
    out: &BufferRef, out_offset: u64,
    dim: u32,
) {
    if let Some(ref swiglu_vec4) = ctx.swiglu_vec4 {
        if dim % 4 == 0 {
            encoder.set_compute_pipeline_state(swiglu_vec4);
            encoder.set_buffer(0, Some(gate), gate_offset);
            encoder.set_buffer(1, Some(up), up_offset);
            encoder.set_buffer(2, Some(out), out_offset);
            unsafe { set_u32(encoder, 3, dim); }
            let vec_dim = dim / 4;
            let num_tgs = (vec_dim + 255) / 256;
            encoder.dispatch_thread_groups(
                MTLSize::new(num_tgs as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            return;
        }
    }

    encoder.set_compute_pipeline_state(&ctx.swiglu);
    encoder.set_buffer(0, Some(gate), gate_offset);
    encoder.set_buffer(1, Some(up), up_offset);
    encoder.set_buffer(2, Some(out), out_offset);
    unsafe { set_u32(encoder, 3, dim); }
    let num_tgs = (dim + 255) / 256;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Standalone SwiGLU dispatch.
pub fn metal_swiglu(
    ctx: &MetalContext,
    gate: &Buffer,
    up: &Buffer,
    out: &Buffer,
    dim: u32,
) {
    let cmd_buf = ctx.queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encode_swiglu(ctx, encoder, gate, 0, up, 0, out, 0, dim);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
}

// ---------------------------------------------------------------------------
// Weighted sum
// ---------------------------------------------------------------------------

/// Encode weighted sum into an existing encoder.
pub fn encode_weighted_sum(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    stacked: &BufferRef,
    weights: &BufferRef,
    out: &BufferRef,
    k: u32,
    dim: u32,
) {
    encoder.set_compute_pipeline_state(&ctx.weighted_sum);
    encoder.set_buffer(0, Some(stacked), 0);
    encoder.set_buffer(1, Some(weights), 0);
    encoder.set_buffer(2, Some(out), 0);
    unsafe {
        set_u32(encoder, 3, k);
        set_u32(encoder, 4, dim);
    }
    let num_tgs = (dim + 255) / 256;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Standalone weighted sum dispatch.
pub fn metal_weighted_sum(
    ctx: &MetalContext,
    stacked: &Buffer,
    weights: &Buffer,
    out: &Buffer,
    k: u32,
    dim: u32,
) {
    let cmd_buf = ctx.queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encode_weighted_sum(ctx, encoder, stacked, weights, out, k, dim);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
}

// ---------------------------------------------------------------------------
// RMS Normalization
// ---------------------------------------------------------------------------

/// Encode RMS norm sum-of-squares reduction.
pub fn encode_rms_norm_sum(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    x: &BufferRef,
    sum_sq: &BufferRef,
    dim: u32,
) {
    encoder.set_compute_pipeline_state(&ctx.rms_norm_sum);
    encoder.set_buffer(0, Some(x), 0);
    encoder.set_buffer(1, Some(sum_sq), 0);
    unsafe { set_u32(encoder, 2, dim); }
    encoder.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Encode RMS norm apply.
pub fn encode_rms_norm_apply(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    x: &BufferRef,
    weight: &BufferRef,
    sum_sq: &BufferRef,
    out: &BufferRef,
    dim: u32,
    eps: f32,
) {
    encoder.set_compute_pipeline_state(&ctx.rms_norm_apply);
    encoder.set_buffer(0, Some(x), 0);
    encoder.set_buffer(1, Some(weight), 0);
    encoder.set_buffer(2, Some(sum_sq), 0);
    encoder.set_buffer(3, Some(out), 0);
    unsafe {
        set_u32(encoder, 4, dim);
        set_f32(encoder, 5, eps);
    }
    let num_tgs = (dim + 255) / 256;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

/// Standalone RMS norm.
pub fn metal_rms_norm(
    ctx: &MetalContext,
    x: &Buffer,
    weight: &Buffer,
    out: &Buffer,
    dim: u32,
    eps: f32,
) {
    let sum_sq = ctx.device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    let cmd_buf = ctx.queue.new_command_buffer();

    {
        let encoder = cmd_buf.new_compute_command_encoder();
        encode_rms_norm_sum(ctx, encoder, x, &sum_sq, dim);
        encoder.end_encoding();
    }

    {
        let encoder = cmd_buf.new_compute_command_encoder();
        encode_rms_norm_apply(ctx, encoder, x, weight, &sum_sq, out, dim, eps);
        encoder.end_encoding();
    }

    cmd_buf.commit();
    cmd_buf.wait_until_completed();
}

// ─── Linear attention GPU kernels ─────────────────────────────────────────

/// Encode gated delta net step — SSM recurrence (one threadgroup per v-head).
pub fn encode_gated_delta_net_step(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    state: &BufferRef,
    q: &BufferRef,
    k: &BufferRef,
    v: &BufferRef,
    g_decay: &BufferRef,
    beta_gate: &BufferRef,
    output: &BufferRef,
    num_v_heads: u32,
    k_heads_per_v: u32,
    _key_dim: u32,   // kernel hardcodes key_dim=128
    value_dim: u32,
) {
    debug_assert!(_key_dim == 128, "gated_delta_net_step kernel hardcodes key_dim=128");
    debug_assert!(value_dim == 128, "gated_delta_net_step kernel hardcodes value_dim=128");
    let pipeline = ctx.gated_delta_net_step.as_ref().expect("gated_delta_net_step kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(state), 0);
    encoder.set_buffer(1, Some(q), 0);
    encoder.set_buffer(2, Some(k), 0);
    encoder.set_buffer(3, Some(v), 0);
    encoder.set_buffer(4, Some(g_decay), 0);
    encoder.set_buffer(5, Some(beta_gate), 0);
    encoder.set_buffer(6, Some(output), 0);
    unsafe { set_u32(encoder, 7, k_heads_per_v); }
    encoder.dispatch_thread_groups(
        MTLSize::new(num_v_heads as u64, 1, 1),
        MTLSize::new(value_dim as u64, 1, 1),
    );
}

/// Encode compute_decay_beta — computes g_decay and beta_gate from alpha, beta, A_log, dt_bias.
pub fn encode_compute_decay_beta(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    alpha: &BufferRef,
    beta: &BufferRef,
    a_log: &BufferRef,
    dt_bias: &BufferRef,
    g_decay: &BufferRef,
    beta_gate: &BufferRef,
    num_v_heads: u32,
) {
    let pipeline = ctx.compute_decay_beta.as_ref().expect("compute_decay_beta kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(alpha), 0);
    encoder.set_buffer(1, Some(beta), 0);
    encoder.set_buffer(2, Some(a_log), 0);
    encoder.set_buffer(3, Some(dt_bias), 0);
    encoder.set_buffer(4, Some(g_decay), 0);
    encoder.set_buffer(5, Some(beta_gate), 0);
    encoder.dispatch_thread_groups(
        MTLSize::new(1, 1, 1),
        MTLSize::new(num_v_heads as u64, 1, 1),
    );
}

/// Encode RMS norm for q/k (per-head, bare norm with scale).
pub fn encode_rms_norm_qk(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    q: &BufferRef,
    k: &BufferRef,
    num_heads: u32,
    key_dim: u32,
    inv_scale: f32,
) {
    let pipeline = ctx.rms_norm_qk.as_ref().expect("rms_norm_qk kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(q), 0);
    encoder.set_buffer(1, Some(k), 0);
    unsafe {
        set_u32(encoder, 2, key_dim);
        set_f32(encoder, 3, inv_scale);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(num_heads as u64, 1, 1),
        MTLSize::new(key_dim as u64, 1, 1),
    );
}

/// Encode gated RMS norm (z-gated output normalization, per v-head).
pub fn encode_gated_rms_norm(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    values: &BufferRef,
    z: &BufferRef,
    weight: &BufferRef,   // bf16 u16 weight, value_dim elements, shared across heads
    output: &BufferRef,
    num_v_heads: u32,
    value_dim: u32,
) {
    let pipeline = ctx.gated_rms_norm.as_ref().expect("gated_rms_norm kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(values), 0);
    encoder.set_buffer(1, Some(z), 0);
    encoder.set_buffer(2, Some(weight), 0);
    encoder.set_buffer(3, Some(output), 0);
    unsafe {
        set_u32(encoder, 4, value_dim);
        set_f32(encoder, 5, 1e-6);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(num_v_heads as u64, 1, 1),
        MTLSize::new(value_dim as u64, 1, 1),
    );
}

/// Encode depthwise conv1d step (with SiLU activation and state update).
pub fn encode_conv1d_step(
    ctx: &MetalContext,
    encoder: &ComputeCommandEncoderRef,
    conv_state: &BufferRef,   // [(kernel_size-1) * conv_dim] = [3 * conv_dim]
    input: &BufferRef,        // [conv_dim]
    weights: &BufferRef,      // bf16 u16 weight, [conv_dim * 4]
    output: &BufferRef,       // [conv_dim]
    conv_dim: u32,
) {
    let pipeline = ctx.conv1d_step.as_ref().expect("conv1d_step kernel missing");
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(conv_state), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(weights), 0);
    encoder.set_buffer(3, Some(output), 0);
    unsafe { set_u32(encoder, 4, conv_dim); }
    let num_tgs = (conv_dim + 255) / 256;
    encoder.dispatch_thread_groups(
        MTLSize::new(num_tgs as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

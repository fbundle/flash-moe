// GPU encode/dispatch for matvec, MoE experts, attention, delta-net.
// Port of moe_infer_mlx/core_src/gpu_ops.h — adapted for Rust metal crate API.

use metal::*;
use crate::constants::MAX_K;
use crate::metal::MetalCtx;
use crate::types::*;

// ---- Batch matvec spec ----

pub struct BatchMatvecSpec {
    pub w: *const u32,
    pub scales: *const u16,
    pub biases: *const u16,
    pub out_cpu: *mut f32,
    pub out_dim: u32,
    pub in_dim: u32,
    pub group_size: u32,
    pub batch_slot: usize,
}

// ---- Fast dequant matvec (GPU if available, CPU fallback) ----

pub unsafe fn fast_dequant_matvec(
    ctx: &MetalCtx,
    w: *const u32,
    scales: *const u16,
    biases: *const u16,
    x: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) {
    if ctx.wf_buf.is_some() {
        gpu_dequant_matvec(ctx, w, scales, biases, x, out, out_dim, in_dim, group_size);
    } else {
        crate::kernels::cpu_dequant_matvec(
            unsafe { std::slice::from_raw_parts(w, out_dim * in_dim / 8) },
            unsafe { std::slice::from_raw_parts(scales, out_dim * in_dim / group_size) },
            unsafe { std::slice::from_raw_parts(biases, out_dim * in_dim / group_size) },
            x, out, out_dim, in_dim, group_size,
        );
    }
}

/// GPU dequant matvec: out = W_4bit @ x
fn gpu_dequant_matvec(
    ctx: &MetalCtx,
    w: *const u32,
    scales: *const u16,
    biases: *const u16,
    x: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) {
    let wf = ctx.wf_buf.as_ref().unwrap();
    let wf_ptr = wf.contents() as *const u8;

    // Copy input
    unsafe {
        let input_ptr = ctx.buf_input.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(x.as_ptr(), input_ptr, in_dim);
    }

    let w_off = unsafe { (w as *const u8).offset_from(wf_ptr) as u64 };
    let s_off = unsafe { (scales as *const u8).offset_from(wf_ptr) as u64 };
    let b_off = unsafe { (biases as *const u8).offset_from(wf_ptr) as u64 };

    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();

    let use_v3 = in_dim <= 4096;
    let pipe = if use_v3 { &ctx.matvec_v3 } else { &ctx.matvec_fast };
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(wf), w_off);
    enc.set_buffer(1, Some(wf), s_off);
    enc.set_buffer(2, Some(wf), b_off);
    enc.set_buffer(3, Some(&ctx.buf_input), 0);
    enc.set_buffer(4, Some(&ctx.buf_output), 0);
    enc.set_bytes(5, 4, &(out_dim as u32) as *const u32 as *const _);
    enc.set_bytes(6, 4, &(in_dim as u32) as *const u32 as *const _);
    enc.set_bytes(7, 4, &(group_size as u32) as *const u32 as *const _);

    if use_v3 {
        let tgs = ((out_dim as u32 + 7) / 8) as u64;
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
    } else {
        enc.dispatch_thread_groups(
            MTLSize::new(out_dim as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }
    enc.end_encoding();

    cmd.commit();
    cmd.wait_until_completed();

    // Read result
    unsafe {
        let out_ptr = ctx.buf_output.contents() as *const f32;
        std::ptr::copy_nonoverlapping(out_ptr, out.as_mut_ptr(), out_dim);
    }
}

// ---- Batched matvec encode (into existing command buffer) ----

pub fn gpu_encode_batch_matvec(
    ctx: &MetalCtx,
    cmd: &CommandBufferRef,
    specs: &[BatchMatvecSpec],
) {
    let wf = match ctx.wf_buf.as_ref() {
        Some(b) => b,
        None => return,
    };
    let wf_ptr = wf.contents() as *const u8;

    for s in specs {
        let w_off = unsafe { (s.w as *const u8).offset_from(wf_ptr) as u64 };
        let s_off = unsafe { (s.scales as *const u8).offset_from(wf_ptr) as u64 };
        let b_off = unsafe { (s.biases as *const u8).offset_from(wf_ptr) as u64 };

        let enc = cmd.new_compute_command_encoder();
        let use_v3 = s.in_dim <= 4096;
        let pipe = if use_v3 { &ctx.matvec_v3 } else { &ctx.matvec_fast };
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(wf), w_off);
        enc.set_buffer(1, Some(wf), s_off);
        enc.set_buffer(2, Some(wf), b_off);
        enc.set_buffer(3, Some(&ctx.buf_input), 0);
        enc.set_buffer(4, Some(&ctx.batch_out[s.batch_slot]), 0);
        enc.set_bytes(5, 4, &s.out_dim as *const u32 as *const _);
        enc.set_bytes(6, 4, &s.in_dim as *const u32 as *const _);
        enc.set_bytes(7, 4, &s.group_size as *const u32 as *const _);

        if use_v3 {
            let tgs = ((s.out_dim + 7) / 8) as u64;
            enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
        } else {
            enc.dispatch_thread_groups(
                MTLSize::new(s.out_dim as u64, 1, 1),
                MTLSize::new(64, 1, 1),
            );
        }
        enc.end_encoding();
    }
}

pub fn gpu_flush_batch_results(ctx: &MetalCtx, specs: &[BatchMatvecSpec]) {
    for s in specs {
        unsafe {
            let src = ctx.batch_out[s.batch_slot].contents() as *const f32;
            std::ptr::copy_nonoverlapping(src, s.out_cpu, s.out_dim as usize);
        }
    }
}

// ---- Encode expert forward into command buffer (multi-expert slot K) ----

pub fn gpu_encode_expert_forward_slot(
    cfg: &ModelConfig,
    ctx: &MetalCtx,
    cmd: &CommandBufferRef,
    k: usize,
    use_2bit: bool,
    data_buf: Option<&Buffer>,  // if Some, use this instead of buf_multi_expert_data[k]
) {
    let layout = if use_2bit { &cfg.layout_2bit } else { &cfg.layout_4bit };
    let pipe = if use_2bit {
        ctx.matvec_2bit.as_ref().unwrap_or(&ctx.matvec_v3)
    } else {
        &ctx.matvec_v3
    };

    let go = cfg.moe_intermediate as u32;
    let gi = cfg.hidden_dim as u32;
    let dout = cfg.hidden_dim as u32;
    let din = cfg.moe_intermediate as u32;
    let gs = cfg.group_size as u32;

    let expert_data = data_buf.unwrap_or(&ctx.buf_multi_expert_data[k]);

    // gate_proj
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(expert_data), layout.gate_w_off as u64);
        enc.set_buffer(1, Some(expert_data), layout.gate_s_off as u64);
        enc.set_buffer(2, Some(expert_data), layout.gate_b_off as u64);
        enc.set_buffer(3, Some(&ctx.buf_multi_expert_input), 0);
        enc.set_buffer(4, Some(&ctx.buf_multi_expert_gate[k]), 0);
        enc.set_bytes(5, 4, &go as *const u32 as *const _);
        enc.set_bytes(6, 4, &gi as *const u32 as *const _);
        enc.set_bytes(7, 4, &gs as *const u32 as *const _);
        let tgs = ((go + 7) / 8) as u64;
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
    // up_proj
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(expert_data), layout.up_w_off as u64);
        enc.set_buffer(1, Some(expert_data), layout.up_s_off as u64);
        enc.set_buffer(2, Some(expert_data), layout.up_b_off as u64);
        enc.set_buffer(3, Some(&ctx.buf_multi_expert_input), 0);
        enc.set_buffer(4, Some(&ctx.buf_multi_expert_up[k]), 0);
        enc.set_bytes(5, 4, &go as *const u32 as *const _);
        enc.set_bytes(6, 4, &gi as *const u32 as *const _);
        enc.set_bytes(7, 4, &gs as *const u32 as *const _);
        let tgs = ((go + 7) / 8) as u64;
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
    // SwiGLU
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.swiglu);
        enc.set_buffer(0, Some(&ctx.buf_multi_expert_gate[k]), 0);
        enc.set_buffer(1, Some(&ctx.buf_multi_expert_up[k]), 0);
        enc.set_buffer(2, Some(&ctx.buf_multi_expert_act[k]), 0);
        enc.set_bytes(3, 4, &go as *const u32 as *const _);
        let tgs = ((go + 255) / 256) as u64;
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
    // down_proj
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipe);
        enc.set_buffer(0, Some(expert_data), layout.down_w_off as u64);
        enc.set_buffer(1, Some(expert_data), layout.down_s_off as u64);
        enc.set_buffer(2, Some(expert_data), layout.down_b_off as u64);
        enc.set_buffer(3, Some(&ctx.buf_multi_expert_act[k]), 0);
        enc.set_buffer(4, Some(&ctx.buf_multi_expert_out[k]), 0);
        enc.set_bytes(5, 4, &dout as *const u32 as *const _);
        enc.set_bytes(6, 4, &din as *const u32 as *const _);
        enc.set_bytes(7, 4, &gs as *const u32 as *const _);
        let tgs = ((dout + 7) / 8) as u64;
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
}

// ---- Encode shared expert down+swiglu into command buffer ----

pub fn gpu_encode_shared_down_swiglu(
    cfg: &ModelConfig,
    ctx: &MetalCtx,
    cmd: &CommandBufferRef,
    sgw: *const u32, sgs: *const u16, sgb: *const u16,
    suw: *const u32, sus: *const u16, sub: *const u16,
    sdw: *const u32, sds: *const u16, sdb: *const u16,
) {
    if ctx.wf_buf.is_none() { return; }
    let wf = ctx.wf_buf.as_ref().unwrap();
    let wf_ptr = wf.contents() as *const u8;

    let si = cfg.shared_intermediate as u32;
    let hd = cfg.hidden_dim as u32;
    let gs = cfg.group_size as u32;

    // Gate proj: buf_shared_gate = sg_weight @ h_post
    {
        let w_off = unsafe { (sgw as *const u8).offset_from(wf_ptr) as u64 };
        let s_off = unsafe { (sgs as *const u8).offset_from(wf_ptr) as u64 };
        let b_off = unsafe { (sgb as *const u8).offset_from(wf_ptr) as u64 };
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.matvec_v3);
        enc.set_buffer(0, Some(wf), w_off);
        enc.set_buffer(1, Some(wf), s_off);
        enc.set_buffer(2, Some(wf), b_off);
        enc.set_buffer(3, Some(&ctx.buf_multi_expert_input), 0);
        enc.set_buffer(4, Some(&ctx.buf_shared_gate), 0);
        enc.set_bytes(5, 4, &si as *const u32 as *const _);
        enc.set_bytes(6, 4, &hd as *const u32 as *const _);
        enc.set_bytes(7, 4, &gs as *const u32 as *const _);
        let tgs = ((si + 7) / 8) as u64;
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }

    // Up proj: buf_shared_up = su_weight @ h_post
    {
        let w_off = unsafe { (suw as *const u8).offset_from(wf_ptr) as u64 };
        let s_off = unsafe { (sus as *const u8).offset_from(wf_ptr) as u64 };
        let b_off = unsafe { (sub as *const u8).offset_from(wf_ptr) as u64 };
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.matvec_v3);
        enc.set_buffer(0, Some(wf), w_off);
        enc.set_buffer(1, Some(wf), s_off);
        enc.set_buffer(2, Some(wf), b_off);
        enc.set_buffer(3, Some(&ctx.buf_multi_expert_input), 0);
        enc.set_buffer(4, Some(&ctx.buf_shared_up), 0);
        enc.set_bytes(5, 4, &si as *const u32 as *const _);
        enc.set_bytes(6, 4, &hd as *const u32 as *const _);
        enc.set_bytes(7, 4, &gs as *const u32 as *const _);
        let tgs = ((si + 7) / 8) as u64;
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }

    // SwiGLU: buf_shared_act = swiglu(buf_shared_gate, buf_shared_up)
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.swiglu);
        enc.set_buffer(0, Some(&ctx.buf_shared_gate), 0);
        enc.set_buffer(1, Some(&ctx.buf_shared_up), 0);
        enc.set_buffer(2, Some(&ctx.buf_shared_act), 0);
        enc.set_bytes(3, 4, &si as *const u32 as *const _);
        let tgs = ((si + 255) / 256) as u64;
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }

    // Down proj: buf_shared_out = sd_weight @ buf_shared_act
    {
        let w_off = unsafe { (sdw as *const u8).offset_from(wf_ptr) as u64 };
        let s_off = unsafe { (sds as *const u8).offset_from(wf_ptr) as u64 };
        let b_off = unsafe { (sdb as *const u8).offset_from(wf_ptr) as u64 };
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.matvec_v3);
        enc.set_buffer(0, Some(wf), w_off);
        enc.set_buffer(1, Some(wf), s_off);
        enc.set_buffer(2, Some(wf), b_off);
        enc.set_buffer(3, Some(&ctx.buf_shared_act), 0);
        enc.set_buffer(4, Some(&ctx.buf_shared_out), 0);
        enc.set_bytes(5, 4, &hd as *const u32 as *const _);
        enc.set_bytes(6, 4, &si as *const u32 as *const _);
        enc.set_bytes(7, 4, &gs as *const u32 as *const _);
        let tgs = ((hd + 7) / 8) as u64;
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
}

// ---- GPU combine dispatch (moe_combine_residual + rms_norm -> buf_input) ----

pub fn gpu_encode_combine(
    ctx: &MetalCtx,
    cmd: &CommandBufferRef,
    actual_k: u32,
    shared_gate_score: f32,
    expert_weights: &[f32; MAX_K],
    next_norm_w: *const u16,
    hidden_dim: u32,
    rms_norm_eps: f32,
) {
    let combine_pipe = match &ctx.moe_combine_residual {
        Some(p) => p,
        None => return,
    };
    if ctx.wf_buf.is_none() { return; }
    let wf = ctx.wf_buf.as_ref().unwrap();
    let wf_ptr = wf.contents() as *const u8;

    // Prepare combine params
    unsafe {
        let params = ctx.buf_combine_params.contents() as *mut f32;
        std::ptr::write_bytes(params as *mut u8, 0, 10 * 4);
        for k in 0..actual_k as usize {
            *params.add(k) = expert_weights[k];
        }
        *params.add(8) = shared_gate_score;
    }

    // moe_combine_residual
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(combine_pipe);
        enc.set_buffer(0, Some(&ctx.buf_h_mid), 0);
        enc.set_buffer(1, Some(&ctx.buf_shared_out), 0);
        enc.set_buffer(2, Some(&ctx.buf_moe_hidden), 0);
        for k in 0..MAX_K {
            enc.set_buffer((3 + k) as u64, Some(&ctx.buf_multi_expert_out[k]), 0);
        }
        enc.set_buffer(11, Some(&ctx.buf_combine_params), 0);
        enc.set_bytes(12, 4, &hidden_dim as *const u32 as *const _);
        enc.set_bytes(13, 4, &actual_k as *const u32 as *const _);
        let tgs = ((hidden_dim + 255) / 256) as u64;
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }

    // rms_norm_sum_sq
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.rms_norm_sum);
        enc.set_buffer(0, Some(&ctx.buf_moe_hidden), 0);
        enc.set_buffer(1, Some(&ctx.buf_cmd3_sum_sq), 0);
        enc.set_bytes(2, 4, &hidden_dim as *const u32 as *const _);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }

    // rms_norm_apply_bf16
    let norm_off = unsafe { (next_norm_w as *const u8).offset_from(wf_ptr) as u64 };
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&ctx.rms_norm_apply_bf16);
        enc.set_buffer(0, Some(&ctx.buf_moe_hidden), 0);
        enc.set_buffer(1, Some(wf), norm_off);
        enc.set_buffer(2, Some(&ctx.buf_cmd3_sum_sq), 0);
        enc.set_buffer(3, Some(&ctx.buf_input), 0);
        enc.set_bytes(4, 4, &hidden_dim as *const u32 as *const _);
        enc.set_bytes(5, 4, &rms_norm_eps as *const f32 as *const _);
        let tgs = ((hidden_dim + 255) / 256) as u64;
        enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
    }
}

// ---- Rotary embedding (CPU) ----

pub fn apply_rotary_emb(
    q: &mut [f32],
    k: &mut [f32],
    pos: i32,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
) {
    let pos_f = pos as f32;
    let half = rotary_dim / 2;
    for h in 0..num_heads {
        let qh = &mut q[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            let freq = 1.0 / rope_theta.powf((2 * i) as f32 / rotary_dim as f32);
            let angle = pos_f * freq;
            let cos = angle.cos();
            let sin = angle.sin();
            let q0 = qh[i];
            let q1 = qh[i + half];
            qh[i]        = q0 * cos - q1 * sin;
            qh[i + half] = q0 * sin + q1 * cos;
        }
    }
    for h in 0..num_kv_heads {
        let kh = &mut k[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            let freq = 1.0 / rope_theta.powf((2 * i) as f32 / rotary_dim as f32);
            let angle = pos_f * freq;
            let cos = angle.cos();
            let sin = angle.sin();
            let k0 = kh[i];
            let k1 = kh[i + half];
            kh[i]        = k0 * cos - k1 * sin;
            kh[i + half] = k0 * sin + k1 * cos;
        }
    }
}

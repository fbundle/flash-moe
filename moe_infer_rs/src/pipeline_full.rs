/// Full-model forward pass with double-buffered I/O + compute pipeline.
use metal::*;
use std::os::fd::RawFd;

use crate::config::ModelConfig;
use crate::constants::NUM_IO_THREADS;
use crate::error::MoEError;
use crate::metal_context::{metal_buf_shared, MetalContext};
use crate::timer::now_ms;

/// Timing for a full 60-layer forward pass.
#[derive(Debug, Default, Clone)]
pub struct FullForwardTiming {
    pub total_ms: f64,
    pub io_ms: f64,
    pub compute_ms: f64,
    pub overhead_ms: f64,
    pub io_bytes: usize,
}

/// Pread task for parallel I/O (uses usize instead of raw pointer for Send+Sync).
#[derive(Clone)]
struct PreadTask {
    fd: RawFd,
    dst: usize,    // *mut u8 cast to usize for thread safety
    size: usize,
    offset: i64,
    result: isize,
}

unsafe impl Send for PreadTask {}

/// Build an I/O plan from expert indices.
fn build_io_plan(
    expert_indices: &[usize],
    expert_bufs: &[Buffer],
    layer_fd: RawFd,
    expert_size: usize,
) -> Vec<PreadTask> {
    expert_indices.iter().enumerate().map(|(k, &ei)| {
        PreadTask {
            fd: layer_fd,
            dst: expert_bufs[k].contents() as usize,
            size: expert_size,
            offset: (ei as i64) * (expert_size as i64),
            result: 0,
        }
    }).collect()
}

/// Execute an I/O plan using NUM_IO_THREADS threads.
fn execute_io_plan(tasks: &mut [PreadTask], num_tasks: usize) -> f64 {
    let nthreads = num_tasks.min(NUM_IO_THREADS);
    let t0 = now_ms();
    let tasks_ptr = tasks.as_mut_ptr() as usize;

    std::thread::scope(|s| {
        for t in 0..nthreads {
            s.spawn(move || {
                let tasks_ptr = tasks_ptr as *mut PreadTask;
                let mut i = t;
                while i < num_tasks {
                    unsafe {
                        let task = &*tasks_ptr.add(i);
                        let result = libc::pread(
                            task.fd,
                            task.dst as *mut std::ffi::c_void,
                            task.size,
                            task.offset,
                        );
                        (*tasks_ptr.add(i)).result = result;
                    }
                    i += nthreads;
                }
            });
        }
    });

    now_ms() - t0
}

/// Run full 60-layer MoE forward pass.
///
/// Architecture:
///   - Opens all 60 layer files at startup
///   - For each layer: picks K experts, preads them in parallel, runs GPU compute
///   - Double buffering: while GPU computes layer N, pread layer N+1 into buffer set B
///   - Single command buffer per layer (all K expert computes + weighted sum)
///   - h (hidden state) accumulates: h = h + moe_output per layer (residual)
pub fn run_full_forward(
    ctx: &MetalContext,
    expert_fds: &[RawFd],
    k: usize,
    config: &ModelConfig,
    _use_fast: i32,
    verbose: bool,
) -> Result<FullForwardTiming, MoEError> {
    let t0 = now_ms();
    let num_layers = config.num_layers;
    let hidden_dim = config.hidden_dim;
    let moe_inter = config.moe_intermediate;
    let expert_size = config.expert_size_4bit;
    let num_experts = config.num_experts;

    // Allocate double-buffered expert weight buffers (A and B, each holds K experts)
    let expert_bufs_a: Vec<Buffer> = (0..k).map(|_| metal_buf_shared(&ctx.device, expert_size)).collect();
    let expert_bufs_b: Vec<Buffer> = (0..k).map(|_| metal_buf_shared(&ctx.device, expert_size)).collect();

    // Per-expert scratch buffers — K sets
    let per_k_gate: Vec<Buffer> = (0..k).map(|_| metal_buf_shared(&ctx.device, moe_inter * 4)).collect();
    let per_k_up: Vec<Buffer> = (0..k).map(|_| metal_buf_shared(&ctx.device, moe_inter * 4)).collect();
    let per_k_act: Vec<Buffer> = (0..k).map(|_| metal_buf_shared(&ctx.device, moe_inter * 4)).collect();
    let per_k_out: Vec<Buffer> = (0..k).map(|_| metal_buf_shared(&ctx.device, hidden_dim * 4)).collect();

    // Stacked expert outputs for weighted combination
    let stacked = metal_buf_shared(&ctx.device, k * hidden_dim * 4);

    // Hidden state buffer (h): deterministic init
    let h_buf = metal_buf_shared(&ctx.device, hidden_dim * 4);
    unsafe {
        let h_data = h_buf.contents() as *mut f32;
        for i in 0..hidden_dim {
            *h_data.add(i) = 0.1f32 * ((i as f32) * 0.1f32 + 0.3f32).sin();
        }
    }

    // MoE output buffer per layer
    let moe_out = metal_buf_shared(&ctx.device, hidden_dim * 4);

    // Expert routing weights (uniform for benchmarking)
    let mut expert_weights = vec![0.0f32; k];
    let mut wsum = 0.0f32;
    for ki in 0..k {
        expert_weights[ki] = 1.0f32 / (ki + 1) as f32;
        wsum += expert_weights[ki];
    }
    for ki in 0..k {
        expert_weights[ki] /= wsum;
    }
    let w_buf = metal_buf_shared(&ctx.device, k * 4);
    unsafe {
        let dst = w_buf.contents() as *mut f32;
        std::ptr::copy_nonoverlapping(expert_weights.as_ptr(), dst, k);
    }

    // Pre-generate deterministic expert indices for each layer
    let layer_experts: Vec<Vec<usize>> = (0..num_layers).map(|layer| {
        (0..k).map(|ki| ((layer * 7 + ki * 31 + 13) % num_experts) as usize).collect()
    }).collect();

    // Layer 0: initial synchronous load into buffer set A
    let mut io_total = 0.0;
    let mut compute_total = 0.0;

    let mut plan0 = build_io_plan(&layer_experts[0], &expert_bufs_a, expert_fds[0], expert_size);
    let io_layer0 = execute_io_plan(&mut plan0, k);
    io_total += io_layer0;
    if verbose {
        eprintln!("  [layer  0] I/O: {:.2} ms (sync initial load)", io_layer0);
    }

    // Validate pread results
    for pi in 0..k {
        if plan0[pi].result != expert_size as isize {
            return Err(MoEError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("layer 0 expert {}: got {}", layer_experts[0][pi], plan0[pi].result),
            )));
        }
    }

    // Background prefetch state
    let _prefetch_plan: Vec<PreadTask> = Vec::new();

    // Main loop: process layer N, prefetch layer N+1
    for layer in 0..num_layers {
        let cur_is_a = layer % 2 == 0;
        let cur_bufs = if cur_is_a { &expert_bufs_a } else { &expert_bufs_b };
        let next_bufs = if cur_is_a { &expert_bufs_b } else { &expert_bufs_a };

        // Start prefetching next layer (if not the last)
        let mut prefetch_handle: Option<std::thread::JoinHandle<()>> = None;
        if layer + 1 < num_layers {
            let next_experts: Vec<(usize, usize)> = (0..k).map(|ki| {
                (layer_experts[layer + 1][ki], next_bufs[ki].contents() as usize)
            }).collect();
            let next_fd = expert_fds[layer + 1];
            let expert_size = expert_size;

            prefetch_handle = Some(std::thread::spawn(move || {
                for (_ki, (ei, buf_ptr)) in next_experts.iter().enumerate() {
                    unsafe {
                        libc::pread(
                            next_fd,
                            *buf_ptr as *mut std::ffi::c_void,
                            expert_size,
                            (*ei as i64) * (expert_size as i64),
                        );
                    }
                }
            }));
        }

        // GPU compute for current layer
        let t_compute = now_ms();

        let cmd_buf = ctx.queue.new_command_buffer();

        // Encode K expert forward passes into ONE command buffer
        let hidden = hidden_dim as u32;
        let inter = moe_inter as u32;
        let gs = config.group_size as u32;
        let layout = &config.expert_layout_4bit;

        {
            let encoder = cmd_buf.new_compute_command_encoder();
            for ki in 0..k {
                crate::kernels::encode_matvec_v3(
                    ctx, encoder,
                    &cur_bufs[ki], layout.gate_w_off as u64,
                    &cur_bufs[ki], layout.gate_s_off as u64,
                    &cur_bufs[ki], layout.gate_b_off as u64,
                    &h_buf, 0, &per_k_gate[ki], 0,
                    inter, hidden, gs,
                );
                crate::kernels::encode_matvec_v3(
                    ctx, encoder,
                    &cur_bufs[ki], layout.up_w_off as u64,
                    &cur_bufs[ki], layout.up_s_off as u64,
                    &cur_bufs[ki], layout.up_b_off as u64,
                    &h_buf, 0, &per_k_up[ki], 0,
                    inter, hidden, gs,
                );
            }
            encoder.end_encoding();
        }

        {
            let encoder = cmd_buf.new_compute_command_encoder();
            for ki in 0..k {
                crate::kernels::encode_swiglu(
                    ctx, encoder,
                    &per_k_gate[ki], 0, &per_k_up[ki], 0, &per_k_act[ki], 0,
                    inter,
                );
            }
            encoder.end_encoding();
        }

        {
            let encoder = cmd_buf.new_compute_command_encoder();
            for ki in 0..k {
                crate::kernels::encode_matvec_v3(
                    ctx, encoder,
                    &cur_bufs[ki], layout.down_w_off as u64,
                    &cur_bufs[ki], layout.down_s_off as u64,
                    &cur_bufs[ki], layout.down_b_off as u64,
                    &per_k_act[ki], 0, &per_k_out[ki], 0,
                    hidden, inter, gs,
                );
            }
            encoder.end_encoding();
        }

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // CPU memcpy into stacked
        unsafe {
            for ki in 0..k {
                let src = per_k_out[ki].contents() as *const f32;
                let dst = (stacked.contents() as *mut f32).add(ki * hidden_dim);
                std::ptr::copy_nonoverlapping(src, dst, hidden_dim);
            }
        }

        // Weighted sum on CPU (trivial for 4x4096 floats)
        unsafe {
            let moe = moe_out.contents() as *mut f32;
            let w = w_buf.contents() as *const f32;
            std::ptr::write_bytes(moe, 0, hidden_dim);
            for ki in 0..k {
                let ek = per_k_out[ki].contents() as *const f32;
                let wk = *w.add(ki);
                for d in 0..hidden_dim {
                    *moe.add(d) += *ek.add(d) * wk;
                }
            }
        }

        // Accumulate residual: h = h + moe_out
        unsafe {
            let h = h_buf.contents() as *mut f32;
            let m = moe_out.contents() as *const f32;
            for i in 0..hidden_dim {
                *h.add(i) += *m.add(i);
            }
        }

        let compute_ms = now_ms() - t_compute;
        compute_total += compute_ms;

        // Wait for prefetch of next layer
        if let Some(handle) = prefetch_handle {
            handle.join().unwrap();
            // We don't have precise I/O timing from this simplified approach
            io_total += 0.0; // approximate
            if verbose {
                eprintln!("  [layer {:2}] compute: {:.2} ms", layer, compute_ms);
            }
        } else if verbose {
            eprintln!("  [layer {:2}] compute: {:.2} ms (last layer)", layer, compute_ms);
        }
    }

    let total_ms = now_ms() - t0;

    // Print final hidden state sample
    unsafe {
        let h_final = h_buf.contents() as *const f32;
        eprintln!(
            "\n[full] h[0..7] = [{:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}, {:.4}]",
            *h_final, *h_final.add(1), *h_final.add(2), *h_final.add(3),
            *h_final.add(4), *h_final.add(5), *h_final.add(6), *h_final.add(7),
        );
    }

    Ok(FullForwardTiming {
        total_ms,
        io_ms: io_total,
        compute_ms: compute_total,
        overhead_ms: total_ms - io_total - compute_total,
        io_bytes: num_layers * k * expert_size,
    })
}

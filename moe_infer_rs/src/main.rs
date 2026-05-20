/// Flash-MoE CLI — Pure Rust/Metal MoE inference engine.
///
/// Port of main.m from vendor/flash-moe/metal_infer.
///
/// Usage:
///   moe-infer --layer 0 --expert 0                   # Single expert forward
///   moe-infer --layer 0 --expert 0 --verify            # Verify Metal vs CPU
///   moe-infer --benchmark                              # Benchmark (10 iterations)
///   moe-infer --moe --k 4                             # Full MoE on single layer
///   moe-infer --full --k 4                            # Full 60-layer MoE forward
///   moe-infer --full --k 4 --benchmark                # Benchmark full forward
use std::os::fd::IntoRawFd;
use std::path::{Path, PathBuf};

use clap::Parser;
use moe_infer::*;

/// CLI arguments for moe-infer.
#[derive(Parser, Debug)]
#[command(name = "moe-infer", version, about = "Rust Metal MoE inference engine")]
struct Args {
    /// Layer index (default: 0)
    #[arg(long, default_value = "0")]
    layer: usize,

    /// Expert index (default: 0)
    #[arg(long, default_value = "0")]
    expert: usize,

    /// Run timing benchmark (10 iterations)
    #[arg(long)]
    benchmark: bool,

    /// Run full MoE with K experts on one layer
    #[arg(long)]
    moe: bool,

    /// Run full 60-layer MoE forward pass
    #[arg(long)]
    full: bool,

    /// Number of active experts per layer (default: 4)
    #[arg(long, default_value = "4")]
    k: usize,

    /// Verify Metal output against CPU reference
    #[arg(long)]
    verify: bool,

    /// Use threadgroup-optimized v3 shader
    #[arg(long)]
    fast: bool,

    /// Model path (default: built-in)
    #[arg(long)]
    model: Option<String>,

    /// Model config path (model_config.json)
    #[arg(long)]
    config: Option<String>,

    /// Run HTTP server (OpenAI-compatible API)
    #[arg(long, default_value = "0")]
    serve: u16,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    let k = args.k.clamp(1, MAX_ACTIVE_EXPERTS);
    let use_fast: i32 = if args.fast { 3 } else { 0 };
    let shader_name = if use_fast >= 3 { "v3-tiled" } else if use_fast >= 1 { "fast-simd" } else { "naive" };

    eprintln!("=== moe-infer: 4-bit dequant MoE engine (Rust/Metal) ===");

    // Load model config at runtime from model_config.json
    let (config, model_path) = if let Some(config_path) = &args.config {
        let cfg = load_model_config(Path::new(config_path))?;
        let mp = PathBuf::from(cfg.model_path.clone());
        (cfg, mp)
    } else if let Some(model_path) = &args.model {
        let mp = PathBuf::from(model_path);
        // If not an existing path and not absolute/relative, look in data/
        let mp = if mp.exists() {
            mp
        } else if mp.is_relative() && !mp.starts_with(".") {
            let data_mp = PathBuf::from("data").join(&mp);
            if data_mp.exists() {
                data_mp
            } else {
                mp
            }
        } else {
            mp
        };
        let cfg = load_model_config(&mp)?;
        (cfg, mp)
    } else {
        anyhow::bail!("Either --model <path> or --config <config_path> is required. The model directory must contain model_config.json.");
    };

    eprintln!("[config] Model path: {}", model_path.display());
    eprintln!("[config] Hidden dim: {}, Layers: {}, Experts: {}", config.hidden_dim, config.num_layers, config.num_experts);
    if args.full {
        eprintln!(
            "Mode: FULL {}-layer forward, K={}, Shader: {}, Benchmark: {}",
            config.num_layers, k, shader_name,
            if args.benchmark { "YES" } else { "NO" },
        );
    } else {
        eprintln!(
            "Layer: {}, Expert: {}, Shader: {}, Benchmark: {}, MoE: {}, Verify: {}",
            args.layer, args.expert, shader_name,
            if args.benchmark { "YES" } else { "NO" },
            if args.moe { "YES" } else { "NO" },
            if args.verify { "YES" } else { "NO" },
        );
    }

    // ========== Serve mode ==========
    if args.serve > 0 {
        run_server(args.serve, &model_path, &config)?;
        return Ok(());
    }

    // Initialize Metal
    eprintln!("[init] Initializing Metal...");
    let ctx = MetalContext::init()?;

    // ========== Full 60-layer forward pass mode ==========
    if args.full {
        let num_layers = config.num_layers;
        let expert_size = config.expert_size_4bit;

        // Open all layer files
        let mut layer_fds = Vec::with_capacity(num_layers);
        eprintln!("\n[io] Opening all {} layer files...", num_layers);
        let t_open = now_ms();
        for i in 0..num_layers {
            let path = model_path.join("packed_experts").join(format!("layer_{:02}.bin", i));
            let file = std::fs::File::open(&path)
                .map_err(|e| anyhow::anyhow!("Cannot open {}: {}", path.display(), e))?;
            layer_fds.push(file.into_raw_fd());
        }
        eprintln!("[io] Opened {} layer files in {:.1} ms", num_layers, now_ms() - t_open);

        let total_expert_bytes = num_layers * k * expert_size;
        eprintln!("\n=== Full {}-layer MoE forward (K={}) ===", num_layers, k);
        eprintln!(
            "[config] {} layers x {} experts x {:.2} MB = {:.1} MB total I/O",
            num_layers, k, expert_size as f64 / (1024.0 * 1024.0),
            total_expert_bytes as f64 / (1024.0 * 1024.0),
        );
        eprintln!("[config] Double-buffered I/O + compute pipeline");
        eprintln!("[config] {} threads for parallel pread", NUM_IO_THREADS);

        let ft = run_full_forward(
            &ctx, &layer_fds, k, &config, use_fast,
            !args.benchmark,
        )?;

        eprintln!("\nFull {}-layer MoE (K={}):", num_layers, k);
        eprintln!("  Total:   {:.1} ms ({:.2} tok/s)", ft.total_ms, 1000.0 / ft.total_ms);
        eprintln!("  I/O:     {:.1} ms ({:.1} GB/s)", ft.io_ms, ft.io_bytes as f64 / (ft.io_ms * 1e6));
        eprintln!("  Compute: {:.1} ms", ft.compute_ms);
        let overhead = ft.total_ms - ft.io_ms - ft.compute_ms;
        eprintln!("  Overhead: {:.1} ms", overhead);

        // Benchmark
        if args.benchmark {
            let n = 3;
            eprintln!("\n--- Full Forward Benchmark ({} iterations) ---", n);
            let mut best_total = f64::MAX;
            let mut sum_total = 0.0;
            let mut sum_io = 0.0;
            let mut sum_compute = 0.0;

            for i in 0..n {
                let bt = run_full_forward(&ctx, &layer_fds, k, &config, use_fast, false)?;
                sum_total += bt.total_ms;
                sum_io += bt.io_ms;
                sum_compute += bt.compute_ms;
                if bt.total_ms < best_total { best_total = bt.total_ms; }

                eprintln!(
                    "  [{}] total={:.1} ms, io={:.1} ms, compute={:.1} ms, {:.2} tok/s",
                    i, bt.total_ms, bt.io_ms, bt.compute_ms, 1000.0 / bt.total_ms,
                );
            }

            eprintln!("\n[bench] Average:");
            eprintln!("  Total:   {:.1} ms ({:.2} tok/s)", sum_total / n as f64, 1000.0 / (sum_total / n as f64));
            eprintln!("  I/O:     {:.1} ms ({:.1} GB/s)", sum_io / n as f64,
                (total_expert_bytes as f64 * n as f64) / (sum_io * 1e6));
            eprintln!("  Compute: {:.1} ms", sum_compute / n as f64);
            eprintln!("[bench] Best: {:.1} ms ({:.2} tok/s)", best_total, 1000.0 / best_total);
        }

        // Cleanup
        for fd in layer_fds {
            unsafe { libc::close(fd); }
        }
        eprintln!("\nDone.");
        return Ok(());
    }

    // ========== Single-layer modes ==========
    let layer_idx = args.layer;
    let path = model_path.join("packed_experts").join(format!("layer_{:02}.bin", layer_idx));
    eprintln!("[io] Opening: {}", path.display());
    let file = std::fs::File::open(&path)
        .map_err(|e| anyhow::anyhow!("Cannot open {}: {}", path.display(), e))?;
    let packed_fd = file.into_raw_fd();

    // Create input vector with deterministic values
    let hidden_dim = config.hidden_dim;
    let x_buf = metal_context::metal_buf_shared(&ctx.device, hidden_dim * 4);
    let x_data = unsafe { std::slice::from_raw_parts_mut(x_buf.contents() as *mut f32, hidden_dim) };
    for i in 0..hidden_dim {
        x_data[i] = 0.1f32 * ((i as f32) * 0.1f32 + 0.3f32).sin();
    }
    eprintln!(
        "[init] Input vector: x[0..3] = [{:.6}, {:.6}, {:.6}, {:.6}]",
        x_data[0], x_data[1], x_data[2], x_data[3],
    );

    let out_buf = metal_context::metal_buf_shared(&ctx.device, hidden_dim * 4);

    // ========== Single expert forward ==========
    if !args.moe {
        let expert_idx = args.expert;
        eprintln!("\n--- Single expert forward (expert {}) ---", expert_idx);

        let et = run_expert_forward(&ctx, packed_fd, expert_idx, &x_buf, &out_buf, &config, use_fast)?;

        let out_data = unsafe { std::slice::from_raw_parts(out_buf.contents() as *const f32, 8) };
        eprintln!(
            "[result] out[0..7] = [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
            out_data[0], out_data[1], out_data[2], out_data[3],
            out_data[4], out_data[5], out_data[6], out_data[7],
        );
        eprintln!(
            "[timing] I/O: {:.2} ms ({:.1} GB/s), Compute: {:.2} ms, Total: {:.2} ms",
            et.io_ms, et.io_bytes as f64 / (et.io_ms * 1e6), et.compute_ms, et.total_ms,
        );

        // Verify against CPU
        if args.verify {
            verify_expert(
                packed_fd, expert_idx, x_data,
                unsafe { std::slice::from_raw_parts(out_buf.contents() as *const f32, hidden_dim) },
                &config, et.compute_ms,
            );
        }

        // Benchmark
        if args.benchmark {
            eprintln!("\n--- Benchmark (10 iterations) ---");
            let n = 10;
            let mut io_sum = 0.0;
            let mut compute_sum = 0.0;
            let mut total_sum = 0.0;
            for i in 0..n {
                let bt = run_expert_forward(&ctx, packed_fd, expert_idx, &x_buf, &out_buf, &config, use_fast)?;
                io_sum += bt.io_ms;
                compute_sum += bt.compute_ms;
                total_sum += bt.total_ms;
                eprintln!("  [{}] io={:.2} ms, compute={:.2} ms, total={:.2} ms", i, bt.io_ms, bt.compute_ms, bt.total_ms);
            }
            eprintln!(
                "[bench] Average: io={:.2} ms, compute={:.2} ms, total={:.2} ms",
                io_sum / n as f64, compute_sum / n as f64, total_sum / n as f64,
            );
            eprintln!(
                "[bench] I/O throughput: {:.1} GB/s",
                config.expert_size_4bit as f64 * n as f64 / (io_sum * 1e6),
            );
        }
    }

    // ========== Full MoE forward (K experts, single layer) ==========
    if args.moe {
        eprintln!(
            "\n--- Full MoE forward ({} experts, {}) ---",
            k, if use_fast >= 3 { "FUSED v3" } else { "legacy" },
        );

        // Simulated routing
        let moe_experts: Vec<usize> = (0..k).map(|ki| (ki * (config.num_experts / k)) % config.num_experts).collect();
        let mut moe_weights: Vec<f32> = moe_experts.iter().enumerate().map(|(ki, _)| 1.0f32 / (ki + 1) as f32).collect();
        let wsum: f32 = moe_weights.iter().sum();
        for w in &mut moe_weights { *w /= wsum; }

        eprint!("[moe] Experts: ");
        for ki in 0..k {
            eprint!("{}({:.3}) ", moe_experts[ki], moe_weights[ki]);
        }
        eprintln!();

        let moe_out = metal_context::metal_buf_shared(&ctx.device, hidden_dim * 4);

        let mt = if use_fast >= 3 && k <= MAX_K_FUSED {
            run_moe_forward_fused(
                &ctx, packed_fd, &moe_experts, &moe_weights,
                &x_buf, &moe_out, &config,
            )?
        } else {
            run_moe_forward(
                &ctx, packed_fd, &moe_experts, &moe_weights,
                &x_buf, &moe_out, &config, use_fast,
            )?
        };

        let moe_data = unsafe { std::slice::from_raw_parts(moe_out.contents() as *const f32, 8) };
        eprintln!(
            "[result] out[0..7] = [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
            moe_data[0], moe_data[1], moe_data[2], moe_data[3],
            moe_data[4], moe_data[5], moe_data[6], moe_data[7],
        );
        eprintln!(
            "[timing] I/O: {:.2} ms ({:.1} GB/s)",
            mt.io_ms, (config.expert_size_4bit * k) as f64 / (mt.io_ms * 1e6),
        );
        eprintln!("[timing] Compute: {:.2} ms (all {} experts)", mt.compute_ms, k);
        eprintln!("[timing] Total: {:.2} ms", mt.total_ms);
        eprintln!("[timing] Experts/sec: {:.0}", k as f64 / (mt.total_ms / 1000.0));
        eprintln!("[timing] Per-expert compute: {:.3} ms", mt.compute_ms / k as f64);

        if args.benchmark {
            eprintln!("\n--- MoE Benchmark (10 iterations, {}) ---", if use_fast >= 3 { "FUSED v3" } else { "legacy" });
            let n = 10;
            let mut total_time = 0.0;
            let mut io_time = 0.0;
            let mut compute_time = 0.0;
            for i in 0..n {
                let bt = if use_fast >= 3 && k <= MAX_K_FUSED {
                    run_moe_forward_fused(&ctx, packed_fd, &moe_experts, &moe_weights, &x_buf, &moe_out, &config)?
                } else {
                    run_moe_forward(&ctx, packed_fd, &moe_experts, &moe_weights, &x_buf, &moe_out, &config, use_fast)?
                };
                total_time += bt.total_ms;
                io_time += bt.io_ms;
                compute_time += bt.compute_ms;
                eprintln!("  [{}] io={:.2} compute={:.2} total={:.2} ms", i, bt.io_ms, bt.compute_ms, bt.total_ms);
            }
            eprintln!(
                "[bench] Average: io={:.2} ms, compute={:.2} ms, total={:.2} ms",
                io_time / n as f64, compute_time / n as f64, total_time / n as f64,
            );
            eprintln!("[bench] Per-expert compute: {:.3} ms", compute_time / (n as f64 * k as f64));
            eprintln!("[bench] If this were the whole token: {:.1} tok/s", 1000.0 / (total_time / n as f64));

            // Legacy comparison
            if use_fast >= 3 && k <= MAX_K_FUSED {
                eprintln!("\n--- Legacy sequential comparison (10 iter) ---");
                total_time = 0.0;
                for i in 0..n {
                    let bt = run_moe_forward(&ctx, packed_fd, &moe_experts, &moe_weights, &x_buf, &moe_out, &config, use_fast)?;
                    total_time += bt.total_ms;
                    eprintln!("  [{}] total={:.2} ms", i, bt.total_ms);
                }
                eprintln!(
                    "[bench-legacy] Average: {:.2} ms ({:.1} tok/s)",
                    total_time / n as f64, 1000.0 / (total_time / n as f64),
                );
            }
        }
    }

    unsafe { libc::close(packed_fd); }
    eprintln!("\nDone.");
    Ok(())
}

/// Verify Metal output against CPU reference computation.
fn verify_expert(
    packed_fd: std::os::fd::RawFd,
    expert_idx: usize,
    x_data: &[f32],
    gpu_out: &[f32],
    config: &ModelConfig,
    gpu_compute_ms: f64,
) {
    eprintln!("\n--- CPU verification ---");
    let hidden = config.hidden_dim;
    let mut cpu_out = vec![0.0f32; hidden];

    let expert_size = config.expert_size_4bit;
    let expert_offset = (expert_idx as i64) * (expert_size as i64);

    // Read all expert components
    let mut w_packed = vec![0u32; expert_size / 4];
    let _scales = vec![0u16; expert_size / 2];
    let _biases = vec![0u16; expert_size / 2];

    let t_cpu = now_ms();
    unsafe {
        let ptr = w_packed.as_mut_ptr() as *mut u8;
        let buf = std::slice::from_raw_parts_mut(ptr, expert_size);
        libc::pread(packed_fd, buf.as_mut_ptr() as *mut std::ffi::c_void, expert_size, expert_offset);
    }

    // Extract components
    let layout = &config.expert_layout_4bit;
    let w_packed_u32 = &w_packed;
    // Scales are uint16 at specific offset
    let scales_u16 = unsafe {
        std::slice::from_raw_parts(
            (w_packed.as_ptr() as *const u8).add(layout.gate_s_off) as *const u16,
            (layout.gate_s_size + layout.gate_b_size + layout.up_s_size + layout.up_b_size
             + layout.down_s_size + layout.down_b_size) / 2,
        )
    };
    let biases_u16 = unsafe {
        std::slice::from_raw_parts(
            (w_packed.as_ptr() as *const u8).add(layout.gate_b_off) as *const u16,
            layout.gate_b_size / 2,
        )
    };

    cpu_dequant_matvec_4bit(
        &w_packed_u32[layout.gate_w_off / 4..],
        scales_u16, biases_u16,
        x_data, &mut cpu_out,
        hidden, hidden, config.group_size,
    );

    let cpu_ms = now_ms() - t_cpu;
    eprintln!("[cpu] Time: {:.2} ms", cpu_ms);
    // Note: this is a simplified verification — full expert forward would need all steps

    let mut max_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    let mut worst_idx = 0;
    for i in 0..hidden {
        let diff = (gpu_out[i] - cpu_out[i]).abs();
        let rel = if cpu_out[i].abs() > 1e-6 { diff / cpu_out[i].abs() } else { diff };
        if diff > max_diff {
            max_diff = diff;
            worst_idx = i;
        }
        if rel > max_rel_diff { max_rel_diff = rel; }
    }
    eprintln!(
        "[verify] Max abs diff: {:.6} at index {} (GPU={:.6}, CPU={:.6})",
        max_diff, worst_idx, gpu_out[worst_idx], cpu_out[worst_idx],
    );
    eprintln!("[verify] Max rel diff: {:.6}", max_rel_diff);
    eprintln!("[verify] {} (threshold: 0.01)", if max_rel_diff < 0.01 { "PASS" } else { "FAIL" });
    eprintln!("[verify] GPU speedup: {:.1}x vs CPU", cpu_ms / gpu_compute_ms);
}

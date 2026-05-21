/// Bench binary — pure Rust token generation benchmark (no HTTP).
/// Generates 100+ tokens and reports stable tok/s.
use std::io::Write;
use std::os::fd::IntoRawFd;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use moe_infer::*;

#[derive(Parser, Debug)]
#[command(name = "bench", about = "Pure Rust token generation benchmark")]
struct Args {
    /// Model directory (e.g., "data/Qwen3.5-35B-A3B-4bit")
    #[arg(long)]
    model: String,

    /// Number of tokens to generate (default: 100)
    #[arg(long, default_value = "100")]
    tokens: usize,

    /// Prompt text (default: "Hello, how are you?")
    #[arg(long, default_value = "Hello, how are you?")]
    prompt: String,

    /// Verbose output (shows generated text)
    #[arg(long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let model_dir = PathBuf::from(&args.model);
    if !model_dir.exists() {
        anyhow::bail!("Model directory not found: {}", model_dir.display());
    }

    // ── Load model config ──
    let config = load_model_config(&model_dir)
        .map_err(|e| anyhow::anyhow!("Failed to load config: {}", e))?;

    eprintln!("[config] {} hidden_dim={} layers={} experts={}",
        model_dir.file_name().unwrap_or_default().to_string_lossy(),
        config.hidden_dim, config.num_layers, config.num_experts);

    // ── Load weight file ──
    let bin_path = model_dir.join("model_weights.bin");
    let json_path = model_dir.join("model_weights.json");
    let wf = WeightFile::open(&bin_path, &json_path)?;

    // ── Load tokenizer ──
    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = BpeTokenizer::load(&tok_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Build vocabulary
    let mut entries: Vec<(u32, String)> = tokenizer
        .vocab
        .iter()
        .map(|e| (e.id, String::from_utf8_lossy(&e.str_bytes).to_string()))
        .collect();
    for added in &tokenizer.added {
        entries.push((added.id, String::from_utf8_lossy(&added.str_bytes).to_string()));
    }
    entries.sort_by_key(|(id, _)| *id);
    let vocab_tokens: Vec<String> = entries.into_iter().map(|(_, s)| s).collect();

    fn decode(tokens: &[String], token_id: usize) -> &str {
        tokens.get(token_id).map(|s| s.as_str()).unwrap_or("<unk>")
    }

    // ── Initialize Metal ──
    let mut ctx = MetalContext::init()?;
    let key_dim = config.linear_total_key / config.linear_num_k_heads;
    let value_dim = config.linear_total_value / config.linear_num_v_heads;
    ctx.init_linear_attn_buffers(
        config.num_linear_layers,
        config.linear_conv_dim,
        config.linear_num_v_heads,
        config.linear_total_value,
        key_dim,
        value_dim,
    );
    let gpu_wf = GpuWeightCtx::new(&ctx.device, &wf);

    // ── Open layer expert files ──
    let packed_dir = model_dir.join("packed_experts");
    let mut layer_fds: Vec<std::os::fd::RawFd> = Vec::with_capacity(config.num_layers);
    for layer in 0..config.num_layers {
        let path = packed_dir.join(format!("layer_{:02}.bin", layer));
        let file = std::fs::File::open(&path)
            .map_err(|e| anyhow::anyhow!("Cannot open {}: {}", path.display(), e))?;
        layer_fds.push(file.into_raw_fd());
    }

    // ── Tokenize prompt ──
    let prompt_str = format!("<|im_start|>system\nYou are a helpful assistant. /think<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n", args.prompt);
    let prompt_ids: Vec<usize> = tokenizer
        .encode(&prompt_str, 8192)
        .into_iter()
        .map(|id| id as usize)
        .collect();

    eprintln!("[bench] Prompt tokens: {}, target gen: {}", prompt_ids.len(), args.tokens);

    const FULL_ATTN_INTERVAL: usize = 4;
    const RMS_NORM_EPS: f32 = 1e-6;
    const EOS_1: usize = 248046;
    const EOS_2: usize = 248044;

    // ── Allocate layer states ──
    enum LayerState {
        FullAttention(KVCache),
        LinearAttention(LinearAttnState),
    }

    struct KVCache {
        k_cache: Vec<f32>,
        v_cache: Vec<f32>,
        len: usize,
    }

    let max_seq = 4096;
    let mut layer_states: Vec<LayerState> = (0..config.num_layers)
        .map(|layer| {
            if (layer + 1) % FULL_ATTN_INTERVAL == 0 {
                let kv_dim = config.num_kv_heads * config.head_dim;
                LayerState::FullAttention(KVCache {
                    k_cache: vec![0.0f32; max_seq * kv_dim],
                    v_cache: vec![0.0f32; max_seq * kv_dim],
                    len: 0,
                })
            } else {
                LayerState::LinearAttention(LinearAttnState::new(
                    config.linear_num_v_heads,
                    config.linear_total_key / config.linear_num_k_heads,
                    config.linear_total_value / config.linear_num_v_heads,
                    config.linear_conv_dim,
                ))
            }
        })
        .collect();

    let hidden_dim = config.hidden_dim;
    let mut hidden = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        hidden[i] = (i as f32 * 0.1f32 + 0.3f32).sin() * 0.1f32;
    }

    // CPU helper functions (copied from server.rs)
    fn em_lookup(wf: &WeightFile, token_id: usize, out: &mut [f32], hidden_dim: usize) {
        let w_data = wf.get_tensor_u32("model.embed_tokens.weight");
        let s_data = wf.get_tensor_u16("model.embed_tokens.scales");
        let b_data = wf.get_tensor_u16("model.embed_tokens.biases");
        let (Some(w), Some(s), Some(b)) = (w_data, s_data, b_data) else { out.fill(0.0); return; };
        let w_info = wf.get_tensor_info("model.embed_tokens.weight").unwrap();
        let packed_cols = w_info.shape[1];
        let s_info = wf.get_tensor_info("model.embed_tokens.scales").unwrap();
        let num_groups = s_info.shape[1];
        let group_size = hidden_dim / num_groups;
        let packed_per_group = group_size / 8;
        let w_row = &w[token_id * packed_cols..];
        let s_row = &s[token_id * num_groups..];
        let b_row = &b[token_id * num_groups..];
        for g in 0..num_groups {
            let scale = bf16_to_f32(s_row[g]);
            let bias = bf16_to_f32(b_row[g]);
            let base = g * group_size;
            for p in 0..packed_per_group {
                let packed = w_row[g * packed_per_group + p];
                for n in 0..8 {
                    let nibble = (packed >> (n * 4)) & 0xF;
                    out[base + p * 8 + n] = (nibble as f32) * scale + bias;
                }
            }
        }
    }

    fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize,
                   num_q_heads: usize, num_kv_heads: usize,
                   head_dim: usize, rotary_dim: usize, rope_theta: f64) {
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

    fn cpu_norm(hidden: &[f32], nw_f32: &[f32], normed: &mut [f32], dim: usize, eps: f32) {
        let sum_sq: f32 = hidden[..dim].iter().map(|v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / dim as f32 + eps).sqrt();
        for i in 0..dim { normed[i] = hidden[i] * inv_rms * nw_f32[i]; }
    }

    fn full_attn(
        wf: &WeightFile, layer_idx: usize,
        hidden: &mut [f32], kv: &mut KVCache, pos: usize,
        hidden_dim: usize, num_attn_heads: usize, num_kv_heads: usize,
        head_dim: usize, rotary_dim: usize,
        rope_theta: f64,
        gpu_wf: Option<&GpuWeightCtx>, ctx: Option<&MetalContext>,
    ) {
        let q_proj_dim = num_attn_heads * head_dim * 2;
        let q_dim = num_attn_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
        let nw = wf.get_tensor_u16(&norm_name);
        let mut normed = vec![0.0f32; hidden_dim];
        if let Some(nw) = nw {
            let nw_f32: Vec<f32> = nw.iter().map(|&v| bf16_to_f32(v)).collect();
            cpu_norm(hidden, &nw_f32, &mut normed, hidden_dim, RMS_NORM_EPS);
        } else { normed.copy_from_slice(hidden); }
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
            gw.encode_matvec_into(wf, c, &enc, &format!("model.layers.{}.self_attn.q_proj", layer_idx), &x_buf, 0, &qbuf, 0, q_proj_dim, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &format!("model.layers.{}.self_attn.k_proj", layer_idx), &x_buf, 0, &kbuf, 0, kv_dim, hidden_dim);
            gw.encode_matvec_into(wf, c, &enc, &format!("model.layers.{}.self_attn.v_proj", layer_idx), &x_buf, 0, &vbuf, 0, kv_dim, hidden_dim);
            enc.end_encoding(); cm.commit(); cm.wait_until_completed();
            unsafe {
                std::ptr::copy_nonoverlapping(qbuf.contents() as *const f32, q_proj_out.as_mut_ptr(), q_proj_dim);
                std::ptr::copy_nonoverlapping(kbuf.contents() as *const f32, k.as_mut_ptr(), kv_dim);
                std::ptr::copy_nonoverlapping(vbuf.contents() as *const f32, v.as_mut_ptr(), kv_dim);
            }
        }
        let mut q = vec![0.0f32; q_dim];
        let mut q_gate = vec![0.0f32; q_dim];
        for h in 0..num_attn_heads {
            let src = &q_proj_out[h * 2 * head_dim..];
            q[h * head_dim..h * head_dim + head_dim].copy_from_slice(&src[..head_dim]);
            q_gate[h * head_dim..h * head_dim + head_dim].copy_from_slice(&src[head_dim..2 * head_dim]);
        }
        if let Some(qnw) = wf.get_tensor_u16(&format!("model.layers.{}.self_attn.q_norm.weight", layer_idx)) {
            for h in 0..num_attn_heads {
                let qh = &mut q[h * head_dim..];
                let sum_sq: f32 = qh.iter().map(|&x| x * x).sum();
                let inv_rms = 1.0 / (sum_sq / head_dim as f32 + RMS_NORM_EPS).sqrt();
                for i in 0..qh.len().min(qnw.len()) { qh[i] = qh[i] * inv_rms * bf16_to_f32(qnw[i]); }
            }
        }
        if let Some(knw) = wf.get_tensor_u16(&format!("model.layers.{}.self_attn.k_norm.weight", layer_idx)) {
            for h in 0..num_kv_heads {
                let kh = &mut k[h * head_dim..];
                let sum_sq: f32 = kh.iter().map(|&x| x * x).sum();
                let inv_rms = 1.0 / (sum_sq / head_dim as f32 + RMS_NORM_EPS).sqrt();
                for i in 0..kh.len().min(knw.len()) { kh[i] = kh[i] * inv_rms * bf16_to_f32(knw[i]); }
            }
        }
        apply_rope(&mut q, &mut k, pos, num_attn_heads, num_kv_heads, head_dim, rotary_dim, rope_theta);
        let cache_pos = kv.len;
        let start = cache_pos * kv_dim;
        kv.k_cache[start..start + kv_dim].copy_from_slice(&k);
        kv.v_cache[start..start + kv_dim].copy_from_slice(&v);
        kv.len += 1;
        let heads_per_kv = num_attn_heads / num_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; q_dim];
        for h in 0..num_attn_heads {
            let kv_h = h / heads_per_kv;
            let qh = &q[h * head_dim..];
            let seq_len = kv.len;
            let mut scores = vec![0.0f32; seq_len];
            for p in 0..seq_len {
                let kp = &kv.k_cache[p * kv_dim + kv_h * head_dim..];
                scores[p] = qh.iter().zip(kp.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale;
            }
            let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = scores.iter().map(|&s| (s - max_val).exp()).sum();
            let inv_sum = 1.0 / sum;
            let oh = &mut attn_out[h * head_dim..];
            for p in 0..seq_len {
                let weight = (scores[p] - max_val).exp() * inv_sum;
                let vp = &kv.v_cache[p * kv_dim + kv_h * head_dim..];
                for d in 0..head_dim { oh[d] += weight * vp[d]; }
            }
        }
        for i in 0..q_dim { attn_out[i] *= 1.0f32 / (1.0f32 + (-q_gate[i]).exp()); }
        let o_prefix = format!("model.layers.{}.self_attn.o_proj", layer_idx);
        let mut o_out = vec![0.0f32; hidden_dim];
        if let (Some(gw), Some(c)) = (gpu_wf, ctx) {
            let attn_buf = metal_buf_shared(&c.device, q_dim * 4);
            unsafe { let dst = attn_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(attn_out.as_ptr(), dst, q_dim); }
            let out_buf = metal_buf_shared(&c.device, hidden_dim * 4);
            let cm = c.queue.new_command_buffer();
            let enc = cm.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, &o_prefix, &attn_buf, 0, &out_buf, 0, hidden_dim, q_dim);
            enc.end_encoding(); cm.commit(); cm.wait_until_completed();
            unsafe { std::ptr::copy_nonoverlapping(out_buf.contents() as *const f32, o_out.as_mut_ptr(), hidden_dim); }
        }
        for i in 0..hidden_dim { hidden[i] += o_out[i]; }
    }

    // ── Prefill ──
    eprintln!("[bench] Prefilling {} tokens...", prompt_ids.len());
    let t_prefill = Instant::now();
    let mut pos: usize = 0;

    let mut embed_batch = vec![0.0f32; prompt_ids.len() * hidden_dim];
    for (i, &id) in prompt_ids.iter().enumerate() {
        em_lookup(&wf, id, &mut embed_batch[i * hidden_dim..(i + 1) * hidden_dim], hidden_dim);
    }

    let n_prefill = prompt_ids.len().saturating_sub(1);
    for i in 0..n_prefill {
        hidden.copy_from_slice(&embed_batch[i * hidden_dim..(i + 1) * hidden_dim]);
        let mut deferred = None;
        for layer in 0..config.num_layers {
            let is_full_attn = (layer + 1) % FULL_ATTN_INTERVAL == 0;
            if is_full_attn {
                if let LayerState::FullAttention(ref mut kv) = layer_states[layer] {
                    full_attn(&wf, layer, &mut hidden, kv, pos,
                        config.hidden_dim, config.num_attn_heads, config.num_kv_heads,
                        config.head_dim, config.rotary_dim, config.rope_theta,
                        Some(&gpu_wf), Some(&ctx));
                }
            } else {
                if let LayerState::LinearAttention(ref mut state) = layer_states[layer] {
                    let linear_idx = layer - (layer + 1) / FULL_ATTN_INTERVAL;
                    linear_attention_forward(&wf, layer, &mut hidden, state,
                        config.hidden_dim, config.linear_num_k_heads, config.linear_num_v_heads,
                        config.linear_total_key, config.linear_total_value, config.linear_conv_dim,
                        Some(&gpu_wf), Some(&ctx), linear_idx);
                }
            }
            let _ = moe_layer_forward(&wf, layer, &mut hidden, layer_fds[layer],
                Some(&ctx), Some(&gpu_wf), &config, &mut deferred);
        }
        pos += 1;
    }

    // Last prefill token
    if !prompt_ids.is_empty() {
        let last_i = prompt_ids.len() - 1;
        hidden.copy_from_slice(&embed_batch[last_i * hidden_dim..(last_i + 1) * hidden_dim]);
        let mut deferred = None;
        for layer in 0..config.num_layers {
            let is_full_attn = (layer + 1) % FULL_ATTN_INTERVAL == 0;
            if is_full_attn {
                if let LayerState::FullAttention(ref mut kv) = layer_states[layer] {
                    full_attn(&wf, layer, &mut hidden, kv, pos,
                        config.hidden_dim, config.num_attn_heads, config.num_kv_heads,
                        config.head_dim, config.rotary_dim, config.rope_theta,
                        Some(&gpu_wf), Some(&ctx));
                }
            } else {
                if let LayerState::LinearAttention(ref mut state) = layer_states[layer] {
                    let linear_idx = layer - (layer + 1) / FULL_ATTN_INTERVAL;
                    linear_attention_forward(&wf, layer, &mut hidden, state,
                        config.hidden_dim, config.linear_num_k_heads, config.linear_num_v_heads,
                        config.linear_total_key, config.linear_total_value, config.linear_conv_dim,
                        Some(&gpu_wf), Some(&ctx), linear_idx);
                }
            }
            let _ = moe_layer_forward(&wf, layer, &mut hidden, layer_fds[layer],
                Some(&ctx), Some(&gpu_wf), &config, &mut deferred);
        }
        if let Some(ref mut def) = deferred {
            def.complete(&mut hidden, hidden_dim);
        }
        pos += 1;
    }

    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[bench] Prefill: {:.0} ms ({} tokens)", prefill_ms, prompt_ids.len());

    // ── Final norm + LM head for first token ──
    fn apply_final_norm(wf: &WeightFile, hidden: &mut [f32], hidden_dim: usize) {
        if let Some(fnw) = wf.get_tensor_u16("model.norm.weight") {
            let fnw_f32: Vec<f32> = fnw.iter().map(|&v| bf16_to_f32(v)).collect();
            let sum_sq: f32 = hidden[..hidden_dim].iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / hidden_dim as f32 + RMS_NORM_EPS).sqrt();
            for i in 0..hidden_dim { hidden[i] = hidden[i] * inv_rms * fnw_f32[i]; }
        }
    }
    fn lm_head(wf: &WeightFile, hidden: &[f32], logits: &mut [f32],
               gpu_wf: Option<&GpuWeightCtx>, ctx: Option<&MetalContext>) {
        if let (Some(gw), Some(c)) = (gpu_wf, ctx) {
            let x_buf = metal_buf_shared(&c.device, hidden.len() * 4);
            unsafe { let dst = x_buf.contents() as *mut f32; std::ptr::copy_nonoverlapping(hidden.as_ptr(), dst, hidden.len()); }
            let out_buf = metal_buf_shared(&c.device, logits.len() * 4);
            let cm = c.queue.new_command_buffer();
            let enc = cm.new_compute_command_encoder();
            gw.encode_matvec_into(wf, c, &enc, "lm_head", &x_buf, 0, &out_buf, 0, logits.len(), hidden.len());
            enc.end_encoding(); cm.commit(); cm.wait_until_completed();
            unsafe { std::ptr::copy_nonoverlapping(out_buf.contents() as *const f32, logits.as_mut_ptr(), logits.len()); }
        }
    }
    fn argmax(x: &[f32]) -> usize {
        x.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0)
    }

    let mut logits = vec![0.0f32; config.vocab_size];
    apply_final_norm(&wf, &mut hidden, hidden_dim);
    lm_head(&wf, &hidden, &mut logits, Some(&gpu_wf), Some(&ctx));
    let mut next_token = argmax(&logits);

    // ── Generation loop ──
    let mut gen_count = 0usize;
    let mut output_text = String::new();
    let t_gen_start = Instant::now();

    for _gen in 0..args.tokens {
        if next_token == EOS_1 || next_token == EOS_2 {
            // Process EOS token through layers for state update
            em_lookup(&wf, next_token, &mut hidden, hidden_dim);
            let mut deferred = None;
            for layer in 0..config.num_layers {
                let is_full_attn = (layer + 1) % FULL_ATTN_INTERVAL == 0;
                if is_full_attn {
                    if let LayerState::FullAttention(ref mut kv) = layer_states[layer] {
                        full_attn(&wf, layer, &mut hidden, kv, pos,
                            config.hidden_dim, config.num_attn_heads, config.num_kv_heads,
                            config.head_dim, config.rotary_dim, config.rope_theta,
                            Some(&gpu_wf), Some(&ctx));
                    }
                } else {
                    if let LayerState::LinearAttention(ref mut state) = layer_states[layer] {
                        let linear_idx = layer - (layer + 1) / FULL_ATTN_INTERVAL;
                        linear_attention_forward(&wf, layer, &mut hidden, state,
                            config.hidden_dim, config.linear_num_k_heads, config.linear_num_v_heads,
                            config.linear_total_key, config.linear_total_value, config.linear_conv_dim,
                            Some(&gpu_wf), Some(&ctx), linear_idx);
                    }
                }
                let _ = moe_layer_forward(&wf, layer, &mut hidden, layer_fds[layer],
                    Some(&ctx), Some(&gpu_wf), &config, &mut deferred);
            }
            break;
        }

        let tok_str = decode(&vocab_tokens, next_token);
        if args.verbose {
            print!("{}", tok_str);
            let _ = std::io::stdout().flush();
        }
        output_text.push_str(tok_str);
        gen_count += 1;

        em_lookup(&wf, next_token, &mut hidden, hidden_dim);
        let mut deferred = None;
        for layer in 0..config.num_layers {
            let is_full_attn = (layer + 1) % FULL_ATTN_INTERVAL == 0;
            if is_full_attn {
                if let LayerState::FullAttention(ref mut kv) = layer_states[layer] {
                    full_attn(&wf, layer, &mut hidden, kv, pos,
                        config.hidden_dim, config.num_attn_heads, config.num_kv_heads,
                        config.head_dim, config.rotary_dim, config.rope_theta,
                        Some(&gpu_wf), Some(&ctx));
                }
            } else {
                if let LayerState::LinearAttention(ref mut state) = layer_states[layer] {
                    let linear_idx = layer - (layer + 1) / FULL_ATTN_INTERVAL;
                    linear_attention_forward(&wf, layer, &mut hidden, state,
                        config.hidden_dim, config.linear_num_k_heads, config.linear_num_v_heads,
                        config.linear_total_key, config.linear_total_value, config.linear_conv_dim,
                        Some(&gpu_wf), Some(&ctx), linear_idx);
                }
            }
            let _ = moe_layer_forward(&wf, layer, &mut hidden, layer_fds[layer],
                Some(&ctx), Some(&gpu_wf), &config, &mut deferred);
        }
        if let Some(ref mut def) = deferred {
            def.complete(&mut hidden, hidden_dim);
        }
        pos += 1;

        apply_final_norm(&wf, &mut hidden, hidden_dim);
        logits.fill(0.0);
        lm_head(&wf, &hidden, &mut logits, Some(&gpu_wf), Some(&ctx));
        next_token = argmax(&logits);
    }

    let gen_elapsed_ms = t_gen_start.elapsed().as_secs_f64() * 1000.0;
    let tok_s = if gen_count > 0 { gen_count as f64 * 1000.0 / gen_elapsed_ms } else { 0.0 };

    if args.verbose {
        println!();
    }
    eprintln!("\n[bench] Results:");
    eprintln!("[bench]   Generated: {} tokens", gen_count);
    eprintln!("[bench]   Time:      {:.0} ms", gen_elapsed_ms);
    eprintln!("[bench]   Speed:     {:.2} tok/s", tok_s);
    eprintln!("[bench]   Output:    {}", output_text.chars().take(200).collect::<String>());

    // Cleanup
    for fd in layer_fds {
        unsafe { libc::close(fd); }
    }

    Ok(())
}

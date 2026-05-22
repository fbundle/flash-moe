/// MoE-Infer core engine: Model data, GPU execution, and helper logic.
///
/// This module is PyO3-free. python_bindings.rs wraps the public types as pyclasses.
use std::collections::HashSet;
use std::os::fd::{IntoRawFd, RawFd};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rand::Rng;

use crate::config::{load_model_config, ModelConfig};
use crate::metal_context::{metal_buf_shared, GpuWeightCtx, MetalContext};
use crate::pipeline_common::{
    DeferredExperts, ExecCtx, FullAttnCache, FullAttnCmd2State, LinearAttnFusedWoodsState,
    LinearAttnState, PipelineMode, SignalCheckFn, FULL_ATTN_INTERVAL, MAX_SEQ, RMS_NORM_EPS,
};
use crate::pipeline_fusedexp::process_token_fusedexp_pipelined;
use crate::pipeline_gpu::{full_attention_forward, linear_attention_forward, moe_layer_forward};
use crate::quant::bf16_to_f32;
use crate::weights::WeightFile;

// ─── Model (data only) ──────────────────────────────────────────────────────

pub struct Model {
    pub config: ModelConfig,
    pub wf: WeightFile,
    pub expert_fds: Vec<RawFd>,
}

impl Model {
    pub fn load(model_path: &str) -> Result<Self, String> {
        let dir = PathBuf::from(model_path);
        if !dir.exists() {
            return Err(format!("not found: {}", dir.display()));
        }
        let config = load_model_config(&dir).map_err(|e| format!("config: {}", e))?;
        let wf = WeightFile::open(
            &dir.join("model_weights.bin"),
            &dir.join("model_weights.json"),
        )
        .map_err(|e| format!("weights: {}", e))?;

        let packed_dir = dir.join("packed_experts");
        let mut expert_fds = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let f = std::fs::File::open(packed_dir.join(format!("layer_{:02}.bin", layer)))
                .map_err(|e| format!("expert {}: {}", layer, e))?;
            expert_fds.push(f.into_raw_fd());
        }

        eprintln!(
            "[model] {} layers hidden={} experts={}",
            config.num_layers, config.hidden_dim, config.num_experts
        );
        Ok(Model { config, wf, expert_fds })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        for fd in &self.expert_fds {
            unsafe { libc::close(*fd); }
        }
    }
}

// ─── Cache (data only) ──────────────────────────────────────────────────────

pub struct Cache {
    pub pos: usize,
    pub kv: Vec<Option<FullAttnCache>>,
    pub lin: Vec<Option<LinearAttnState>>,
}

impl Cache {
    pub fn new(config: &ModelConfig) -> Self {
        let mut kv = Vec::with_capacity(config.num_layers);
        let mut lin = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            if (layer + 1) % FULL_ATTN_INTERVAL == 0 {
                let kv_dim = config.num_kv_heads * config.head_dim;
                kv.push(Some(FullAttnCache::new(MAX_SEQ, kv_dim)));
                lin.push(None);
            } else {
                kv.push(None);
                lin.push(Some(LinearAttnState::new(
                    config.linear_num_v_heads,
                    config.linear_total_key / config.linear_num_k_heads,
                    config.linear_total_value / config.linear_num_v_heads,
                    config.linear_conv_dim,
                )));
            }
        }
        Cache { pos: 0, kv, lin }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        for kv in self.kv.iter_mut().flatten() {
            kv.reset();
        }
        for s in self.lin.iter_mut().flatten() {
            s.conv_state.fill(0.0);
            s.ssm_state.fill(0.0);
        }
    }
}

// ─── Engine (GPU resources + model) ─────────────────────────────────────────

pub struct Telemetry {
    pub prefill_ms: f64,
    pub total_ms: f64,
    pub tokens_generated: usize,
}

pub struct Engine {
    pub model: Arc<Model>,
    pub ctx: MetalContext,
    pub gpu_wf: GpuWeightCtx,
    pub expert_io: Option<crate::metal_context::ExpertIOState>,
    pub pipeline_mode: PipelineMode,
    pub telemetry: Telemetry,
}

impl Engine {
    pub fn new(model: Arc<Model>, pipeline_mode: PipelineMode) -> Result<Self, String> {
        let config = &model.config;
        let mut ctx = MetalContext::init().map_err(|e| format!("metal: {}", e))?;
        let key_dim = config.linear_total_key / config.linear_num_k_heads;
        let value_dim = config.linear_total_value / config.linear_num_v_heads;
        ctx.init_linear_attn_buffers(
            config.num_linear_layers,
            config.linear_conv_dim,
            config.linear_num_v_heads,
            config.linear_total_value,
            key_dim,
            value_dim,
            config.hidden_dim,
            config.num_experts,
            config.shared_intermediate,
        );
        let expert_io = Some(ctx.init_expert_buffers(
            config.expert_size_4bit,
            config.hidden_dim,
            config.moe_intermediate,
            config.shared_intermediate,
        ));
        let gpu_wf = GpuWeightCtx::new(&ctx.device, &model.wf);

        eprintln!(
            "[engine] {} layers hidden={} experts={} mode={:?}",
            config.num_layers, config.hidden_dim, config.num_experts, pipeline_mode
        );
        Ok(Engine {
            model,
            ctx,
            gpu_wf,
            expert_io,
            pipeline_mode,
            telemetry: Telemetry { prefill_ms: 0.0, total_ms: 0.0, tokens_generated: 0 },
        })
    }

    pub fn exec_ctx(&mut self) -> ExecCtx<'_> {
        ExecCtx {
            wf: &self.model.wf,
            ctx: &self.ctx,
            gpu_wf: &self.gpu_wf,
            config: &self.model.config,
            expert_fds: &self.model.expert_fds,
            pipeline_mode: self.pipeline_mode,
            expert_io: self.expert_io.as_mut(),
        }
    }

    // ─── Forward ──────────────────────────────────────────────────────────

    /// Run forward pass for input token IDs. Returns logits for all positions.
    /// All tokens in input_ids are processed as new tokens from cache.pos.
    /// Callers must trim the prefix (input_ids[cache.pos:]) before calling.
    pub fn forward(
        &mut self, input_ids: &[i64], cache: &mut Cache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, String> {
        let t0 = Instant::now();
        let n = input_ids.len();
        let hd = self.model.config.hidden_dim;
        let vs = self.model.config.vocab_size;

        let mut logits = vec![0.0f32; n * vs];
        if n == 0 {
            return Ok(logits);
        }

        let mut embed = vec![0.0f32; n * hd];
        for (i, &id) in input_ids.iter().enumerate() {
            embed_lookup(&self.model.wf, id as usize, &mut embed[i * hd..(i + 1) * hd], hd);
        }

        let mut hidden = vec![0.0f32; hd];
        for (ti, _) in input_ids.iter().enumerate() {
            hidden.copy_from_slice(&embed[ti * hd..(ti + 1) * hd]);
            let mut exec = self.exec_ctx();
            process_token_inner(
                &mut exec, &mut hidden,
                cache.pos, &mut cache.kv, &mut cache.lin,
                check_signal, false, &mut Vec::new(),
            )?;
            cache.pos += 1;
            final_norm(exec.wf, &mut hidden, hd);
            lm_head(exec.wf, &hidden,
                &mut logits[ti * vs..(ti + 1) * vs],
                exec.gpu_wf, exec.ctx);
        }

        self.telemetry.prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.telemetry.total_ms = 0.0;
        self.telemetry.tokens_generated = 0;

        Ok(logits)
    }

    // ─── Generate ─────────────────────────────────────────────────────────

    /// Run autoregressive generation starting from input_ids.
    /// Returns (token_ids, last_logits).
    pub fn generate(
        &mut self, input_ids: &[i64], cache: &mut Cache,
        max_tokens: usize, temperature: f32, top_k: usize, top_p: f32, min_p: f32,
        eos: &HashSet<usize>,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<(Vec<i64>, Vec<f32>), String> {
        let gen_t0 = Instant::now();
        let logits = self.forward(input_ids, cache, check_signal)?;
        let hd = self.model.config.hidden_dim;
        let vs = self.model.config.vocab_size;
        let mut logits_last = logits[logits.len() - vs..].to_vec();

        let mut next = if temperature < 0.01 {
            logits_last.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i).unwrap_or(0)
        } else { sample(&mut logits_last, temperature, top_k, top_p, min_p) };

        let mut output = Vec::with_capacity(max_tokens);
        let mut hidden = vec![0.0f32; hd];
        let wf_ptr = &self.model.wf as *const WeightFile;
        let ctx_ptr = &self.ctx as *const MetalContext;
        let gpu_wf_ptr = &self.gpu_wf as *const GpuWeightCtx;

        for _ in 0..max_tokens {
            if eos.contains(&next) { break; }
            output.push(next as i64);
            embed_lookup(unsafe { &*wf_ptr }, next, &mut hidden, hd);

            {
                let mut exec = self.exec_ctx();
                process_token_inner(
                    &mut exec, &mut hidden,
                    cache.pos, &mut cache.kv, &mut cache.lin,
                    check_signal, false, &mut Vec::new(),
                )?;
            }
            cache.pos += 1;

            final_norm(unsafe { &*wf_ptr }, &mut hidden, hd);
            logits_last.fill(0.0);
            lm_head(unsafe { &*wf_ptr }, &hidden, &mut logits_last,
                unsafe { &*gpu_wf_ptr }, unsafe { &*ctx_ptr });

            next = if temperature < 0.01 {
                logits_last.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i).unwrap_or(0)
            } else { sample(&mut logits_last, temperature, top_k, top_p, min_p) };
        }
        self.telemetry.total_ms = gen_t0.elapsed().as_secs_f64() * 1000.0;
        self.telemetry.tokens_generated = output.len();
        Ok((output, logits_last))
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

pub fn embed_lookup(wf: &WeightFile, token_id: usize, out: &mut [f32], hidden_dim: usize) {
    let (Some(w), Some(s), Some(b)) = (
        wf.get_tensor_u32("model.embed_tokens.weight"),
        wf.get_tensor_u16("model.embed_tokens.scales"),
        wf.get_tensor_u16("model.embed_tokens.biases"),
    ) else {
        out.fill(0.0);
        return;
    };
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

pub fn final_norm(wf: &WeightFile, hidden: &mut [f32], hidden_dim: usize) {
    let Some(fnw_u16) = wf.get_tensor_u16("model.norm.weight") else { return };
    let fnw_f32: Vec<f32> = fnw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
    let sum_sq: f32 = hidden[..hidden_dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / hidden_dim as f32 + RMS_NORM_EPS).sqrt();
    for i in 0..hidden_dim {
        hidden[i] *= inv_rms * fnw_f32[i];
    }
}

pub fn lm_head(
    wf: &WeightFile, hidden: &[f32], logits: &mut [f32],
    gpu_wf: &GpuWeightCtx, ctx: &MetalContext,
) {
    let x_buf = metal_buf_shared(&ctx.device, hidden.len() * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(hidden.as_ptr(), x_buf.contents() as *mut f32, hidden.len());
    }
    let out_buf = metal_buf_shared(&ctx.device, logits.len() * 4);
    let cm = ctx.queue.new_command_buffer();
    let enc = cm.new_compute_command_encoder();
    gpu_wf.encode_matvec_into(wf, ctx, &enc, "lm_head", &x_buf, 0, &out_buf, 0, logits.len(), hidden.len());
    enc.end_encoding();
    cm.commit();
    cm.wait_until_completed();
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents() as *const f32, logits.as_mut_ptr(), logits.len());
    }
}

// ─── Sampling ───────────────────────────────────────────────────────────────

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = x.iter_mut().map(|v| { *v = (*v - max).exp(); *v }).sum();
    for v in x { *v /= sum; }
}

pub fn sample(logits: &mut [f32], temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> usize {
    let n = logits.len();
    if (temperature - 1.0).abs() > 1e-7 {
        let inv = 1.0 / temperature.max(1e-8);
        for v in logits.iter_mut() { *v *= inv; }
    }
    if temperature < 0.01 {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
    }
    softmax(logits);

    if top_k > 0 && top_k < n {
        let mut v: Vec<f32> = logits.to_vec();
        v.select_nth_unstable_by(top_k, |a, b| b.partial_cmp(a).unwrap());
        let t = v[top_k - 1];
        for x in logits.iter_mut() { if *x < t { *x = 0.0; } }
    }
    if top_p < 1.0 {
        let mut s: Vec<f32> = logits.iter().copied().filter(|&x| x > 0.0).collect();
        s.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let total: f32 = s.iter().sum();
        let mut cum = 0.0;
        let mut cut = 0.0;
        for v in s {
            cum += v;
            if cum / total >= top_p { cut = v; break; }
        }
        for x in logits.iter_mut() { if *x < cut { *x = 0.0; } }
    }
    if min_p > 0.0 {
        let max_p = logits.iter().fold(0.0f32, |a, &b| a.max(b));
        let t = max_p * min_p;
        for x in logits.iter_mut() { if *x < t { *x = 0.0; } }
    }

    let sum: f32 = logits.iter().sum();
    if sum <= 0.0 { return 0; }
    let inv = 1.0 / sum;
    let r: f32 = rand::thread_rng().gen();
    let mut cum = 0.0;
    for (i, &v) in logits.iter().enumerate() {
        cum += v * inv;
        if r <= cum { return i; }
    }
    n - 1
}

// ─── Core processing (no PyO3 dependency) ───────────────────────────────────

pub fn process_token_inner(
    exec: &mut ExecCtx<'_>,
    hidden: &mut [f32],
    pos: usize,
    kv: &mut [Option<FullAttnCache>],
    lin: &mut [Option<LinearAttnState>],
    check_signal: SignalCheckFn<'_>,
    capture_per_layer: bool,
    layer_outputs: &mut Vec<Vec<f32>>,
) -> Result<(), String> {
    if exec.pipeline_mode == PipelineMode::FusedExp {
        return process_token_fusedexp_pipelined(
            exec, hidden, pos, kv, lin, check_signal, capture_per_layer, layer_outputs);
    }

    let mut deferred: Option<DeferredExperts> = None;
    let mode = exec.pipeline_mode;
    let hd = exec.config.hidden_dim;
    for layer in 0..exec.config.num_layers {
        if layer % 4 == 0 && check_signal() {
            return Err("interrupted".into());
        }
        let prev_gpu_combined = deferred.as_ref().map_or(false, |d| d.gpu_combined);
        if !prev_gpu_combined {
            if let Some(ref mut def) = deferred.take() {
                def.complete(hidden, hd);
            }
        }
        let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;
        let mut attn_state: Option<FullAttnCmd2State> = None;
        let mut lin_state: Option<LinearAttnFusedWoodsState> = None;
        let mut h_mid_saved: Option<Vec<f32>> = None;
        if is_full {
            if prev_gpu_combined {
                if let Some(ref mut def) = deferred.take() {
                    def.complete_fast(hidden, hd);
                }
                h_mid_saved = Some(hidden.to_vec());
            }
            if let Some(ref mut kv) = kv[layer] {
                attn_state = full_attention_forward(
                    exec.wf, layer, hidden, kv, pos, exec.config,
                    Some(exec.gpu_wf), Some(exec.ctx), mode);
            }
        } else if let Some(ref mut s) = lin[layer] {
            let li = layer - (layer + 1) / FULL_ATTN_INTERVAL;
            if prev_gpu_combined && mode == PipelineMode::FusedExp {
                if let Some(ref mut def) = deferred.take() {
                    def.complete_fast(hidden, hd);
                }
            }
            if mode == PipelineMode::FusedWoods && !prev_gpu_combined {
                h_mid_saved = Some(hidden.to_vec());
            }
            lin_state = linear_attention_forward(
                exec.wf, layer, hidden, s,
                hd,
                exec.config.linear_num_k_heads, exec.config.linear_num_v_heads,
                exec.config.linear_total_key, exec.config.linear_total_value,
                exec.config.linear_conv_dim,
                Some(exec.gpu_wf), Some(exec.ctx), li, mode, prev_gpu_combined,
            );
            if prev_gpu_combined {
                if let Some(ref mut def) = deferred.take() {
                    def.complete_fast(hidden, hd);
                }
                if let Some(ref mut ls) = lin_state {
                    ls.h_mid.copy_from_slice(hidden);
                }
                h_mid_saved = Some(hidden.to_vec());
            }
            if let Some(ref hmid) = h_mid_saved {
                hidden.copy_from_slice(hmid);
            }
        }
        let r = moe_layer_forward(
            exec.wf, layer, hidden, exec.expert_fds[layer],
            Some(exec.ctx), Some(exec.gpu_wf), exec.config,
            mode, attn_state, lin_state,
            exec.expert_io.as_mut().map(|x| &mut **x),
        );
        deferred = r.unwrap_or(None);
        if capture_per_layer {
            layer_outputs.push(hidden.to_vec());
        }
    }
    if let Some(ref mut def) = deferred {
        def.complete(hidden, hd);
    }
    Ok(())
}

pub fn process_token(
    exec: &mut ExecCtx<'_>,
    hidden: &mut [f32],
    pos: usize,
    kv: &mut [Option<FullAttnCache>],
    lin: &mut [Option<LinearAttnState>],
    check_signal: SignalCheckFn<'_>,
) -> Result<(), String> {
    process_token_inner(exec, hidden, pos, kv, lin, check_signal, false, &mut Vec::new())
}

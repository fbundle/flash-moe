/// Thin PyO3 bindings for the Flash-MoE inference engine.
///
/// Two classes:
///   Cache   — pure data: KV caches + linear attention states (no Metal resources)
///   Context — resource manager: holds 0–1 loaded model, provides forward/generate
use std::collections::HashSet;
use std::os::fd::{IntoRawFd, RawFd};
use std::path::PathBuf;
use std::time::Instant;

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::config::{load_model_config, ModelConfig};
use crate::gpu_forward::{full_attention_forward, linear_attention_forward, moe_layer_forward};
use crate::pipeline_common::{DeferredExperts, FullAttnCache, FullAttnCmd2State, LinearAttnFusedWoodsState, LinearAttnState, PipelineMode};
use crate::metal_context::{metal_buf_shared, ExpertIOState, GpuWeightCtx, MetalContext};
use crate::quant::bf16_to_f32;
use crate::weights::WeightFile;

const FULL_ATTN_INTERVAL: usize = 4;
const RMS_NORM_EPS: f32 = 1e-6;
const MAX_SEQ: usize = 4096;

// ─── ModelState (held by Context, loaded/unloaded) ──────────────────────────

struct ModelState {
    config: ModelConfig,
    wf: WeightFile,
    ctx: MetalContext,
    gpu_wf: GpuWeightCtx,
    layer_fds: Vec<RawFd>,
    pipeline_mode: PipelineMode,
    expert_io: Option<ExpertIOState>,
}

impl ModelState {
    fn load(model_path: &str, pipeline_mode: PipelineMode) -> PyResult<Self> {
        let dir = PathBuf::from(model_path);
        if !dir.exists() {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
                "not found: {}", dir.display()
            )));
        }
        let config = load_model_config(&dir).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("config: {}", e))
        })?;
        let wf = WeightFile::open(
            &dir.join("model_weights.bin"),
            &dir.join("model_weights.json"),
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("weights: {}", e)))?;

        let mut ctx = MetalContext::init()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("metal: {}", e)))?;
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
        let gpu_wf = GpuWeightCtx::new(&ctx.device, &wf);

        let packed_dir = dir.join("packed_experts");
        let mut layer_fds = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let f = std::fs::File::open(packed_dir.join(format!("layer_{:02}.bin", layer)))
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("expert {}: {}", layer, e))
                })?;
            layer_fds.push(f.into_raw_fd());
        }

        eprintln!(
            "[model] {} layers hidden={} experts={} mode={:?}",
            config.num_layers, config.hidden_dim, config.num_experts, pipeline_mode
        );
        Ok(ModelState { config, wf, ctx, gpu_wf, layer_fds, pipeline_mode, expert_io })
    }
}

impl Drop for ModelState {
    fn drop(&mut self) {
        for fd in &self.layer_fds {
            unsafe { libc::close(*fd); }
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn embed_lookup(wf: &WeightFile, token_id: usize, out: &mut [f32], hidden_dim: usize) {
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

fn final_norm(wf: &WeightFile, hidden: &mut [f32], hidden_dim: usize) {
    let Some(fnw_u16) = wf.get_tensor_u16("model.norm.weight") else { return };
    let fnw_f32: Vec<f32> = fnw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
    let sum_sq: f32 = hidden[..hidden_dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / hidden_dim as f32 + RMS_NORM_EPS).sqrt();
    for i in 0..hidden_dim {
        hidden[i] *= inv_rms * fnw_f32[i];
    }
}

fn lm_head(
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
        std::ptr::copy_nonoverlapping(out_buf.contents() as *const f32, logits.as_mut_ptr(), logits.len());
    }
}

/// Pipelined FusedExp: N+1 command buffers for N consecutive linear layers.
///
/// CMD layout for a group of M linear layers [L0..L_{M-1}]:
///   CMD 0: pre_expert(L0)
///   CMD i (1..M-1): post_expert(L_{i-1}) + pre_expert(L_i)
///   CMD M: post_expert(L_{M-1})
///
/// Full attention layers break the pipeline and are handled with the existing
/// CMD2 fusion approach. Consecutive linear layers are pipelined.
fn process_token_fusedexp_pipelined(
    m: &mut ModelState,
    hidden: &mut [f32],
    pos: usize,
    kv: &mut [Option<FullAttnCache>],
    lin: &mut [Option<LinearAttnState>],
    py: Python<'_>,
    capture_per_layer: bool,
    layer_outputs: &mut Vec<Vec<f32>>,
) -> PyResult<()> {
    use crate::pipeline_common::{cpu_softmax, cpu_topk, cpu_normalize_weights, CONV_KERNEL_SIZE};
    use crate::metal_context::{ExpertIOState, MAX_K};

    let hd = m.config.hidden_dim;
    let num_layers = m.config.num_layers;
    let num_experts = m.config.num_experts;
    let moe_inter = m.config.moe_intermediate;
    let shared_inter = m.config.shared_intermediate;
    let k = m.config.num_experts_per_tok;
    let expert_size = m.config.expert_size_4bit;
    let layout = &m.config.expert_layout_4bit;
    let qkv_dim = m.config.linear_conv_dim;
    let total_key = m.config.linear_total_key;
    let total_val = m.config.linear_total_value;
    let num_k_heads = m.config.linear_num_k_heads;
    let num_v_heads = m.config.linear_num_v_heads;
    let key_dim = total_key / num_k_heads;
    let val_dim = total_val / num_v_heads;
    let inv_scale = 1.0 / (key_dim as f32).sqrt();
    let k_heads_per_v = num_v_heads / num_k_heads;

    // Helper: upload hidden to GPU buffers for the first layer of a pipeline group.
    // buf_moe_hidden = hidden (residual for residual_add in pre_expert)
    // buf_input = CPU input_norm(hidden, layer's input_layernorm weight)
    let upload_first_layer = |ctx: &MetalContext, wf: &WeightFile, hidden: &[f32], layer_idx: usize| {
        let buf_moe = ctx.buf_moe_hidden.as_ref().unwrap();
        unsafe { std::ptr::copy_nonoverlapping(hidden.as_ptr(), buf_moe.contents() as *mut f32, hd); }
        let norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
        let buf_in = ctx.buf_input.as_ref().unwrap();
        if let Some(nw_u16) = wf.get_tensor_u16(&norm_name) {
            let nw: Vec<f32> = nw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
            let sum_sq: f32 = hidden[..hd].iter().map(|v| v * v).sum();
            let inv_rms = 1.0 / (sum_sq / hd as f32 + RMS_NORM_EPS).sqrt();
            unsafe {
                let dst = buf_in.contents() as *mut f32;
                for i in 0..hd {
                    *dst.add(i) = hidden[i] * inv_rms * nw[i];
                }
            }
        } else {
            unsafe { std::ptr::copy_nonoverlapping(hidden.as_ptr(), buf_in.contents() as *mut f32, hd); }
        }
    };

    // Helper: route + pread for a layer after its pre_expert completed.
    // Reads gate_scores from GPU, does softmax+topk, preads experts into expert_io.
    let route_and_pread = |ctx: &MetalContext,
                           expert_io: &mut ExpertIOState,
                           layer_idx: usize,
                           layer_fd: RawFd|
        -> (Vec<usize>, Vec<f32>, f32)
    {
        let gate_buf = ctx.buf_gate_scores.as_ref().unwrap();
        let mut gate_scores = vec![0.0f32; num_experts];
        unsafe { std::ptr::copy_nonoverlapping(gate_buf.contents() as *const f32, gate_scores.as_mut_ptr(), num_experts); }
        let shared_gate_score = unsafe { *(ctx.buf_shared_gate_score.as_ref().unwrap().contents() as *const f32) };

        cpu_softmax(&mut gate_scores);
        let mut expert_indices = vec![0usize; k];
        let mut expert_weights = vec![0.0f32; k];
        cpu_topk(&gate_scores, k, &mut expert_indices, &mut expert_weights);
        cpu_normalize_weights(&mut expert_weights);

        let actual_k = k.min(MAX_K);
        let mut miss_ei = [0usize; MAX_K];
        let mut miss_k_slot = [0usize; MAX_K];
        let mut miss_count = 0;
        for ki in 0..actual_k {
            let eidx = expert_indices[ki];
            if let Some(buf) = expert_io.cache.lookup(layer_idx, eidx) {
                expert_io.expert_data[ki] = buf;
            } else {
                miss_ei[miss_count] = eidx;
                miss_k_slot[miss_count] = ki;
                miss_count += 1;
            }
        }
        for m in 0..miss_count {
            let ki = miss_k_slot[m];
            let eidx = miss_ei[m];
            let buf = expert_io.cache.insert_get_buf(layer_idx, eidx);
            expert_io.expert_data[ki] = buf;
        }
        if miss_count > 0 {
            let mut pread_tasks: Vec<(RawFd, usize, usize, i64)> = Vec::with_capacity(miss_count);
            for m in 0..miss_count {
                let ki = miss_k_slot[m];
                let eidx = miss_ei[m];
                let ptr = expert_io.expert_data[ki].contents() as usize;
                pread_tasks.push((layer_fd, ptr, expert_size, (eidx as i64) * (expert_size as i64)));
            }
            rayon::scope(|s| {
                for (fd, dst, sz, off) in pread_tasks {
                    s.spawn(move |_| {
                        unsafe { libc::pread(fd, dst as *mut std::ffi::c_void, sz, off); }
                    });
                }
            });
        }
        (expert_indices, expert_weights, shared_gate_score)
    };

    let mut layer = 0;
    let mut pending_deferred: Option<DeferredExperts> = None;
    while layer < num_layers {
        py.check_signals()?;

        let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;

        if is_full {
            // ── Full attention layer: use existing approach, defer CMD3 ──
            // Complete pending deferred from previous layer first (if any)
            if let Some(ref mut def) = pending_deferred.take() {
                def.complete(hidden, hd);
                if capture_per_layer {
                    layer_outputs.push(hidden.to_vec());
                }
            }

            let mut attn_state: Option<FullAttnCmd2State> = None;
            if let Some(ref mut kv_entry) = kv[layer] {
                attn_state = full_attention_forward(
                    &m.wf, layer, hidden, kv_entry, pos, &m.config,
                    Some(&m.gpu_wf), Some(&m.ctx), m.pipeline_mode,
                );
            }
            let r = moe_layer_forward(
                &m.wf, layer, hidden, m.layer_fds[layer],
                Some(&m.ctx), Some(&m.gpu_wf), &m.config,
                m.pipeline_mode, attn_state, None, m.expert_io.as_mut(),
            );
            // Defer CMD3 completion — will be completed before next pipeline group
            pending_deferred = r.unwrap_or(None);
            // Capture deferred until completion
            layer += 1;
            continue;
        }

        // ── Gather consecutive linear layers into a pipeline group ──
        let group_start = layer;
        while layer < num_layers && (layer + 1) % FULL_ATTN_INTERVAL != 0 {
            layer += 1;
        }
        let group_layers: Vec<usize> = (group_start..layer).collect();
        let m_layers = group_layers.len();
        if m_layers == 0 { continue; }

        let first_layer = group_layers[0];
        let last_layer = group_layers[m_layers - 1];

        // Complete pending deferred from previous full-attn layer before upload
        if let Some(ref mut def) = pending_deferred.take() {
            def.complete(hidden, hd);
            if capture_per_layer {
                layer_outputs.push(hidden.to_vec());
            }
        }

        // Upload initial hidden for first layer of the group
        upload_first_layer(&m.ctx, &m.wf, hidden, first_layer);

        // ── CMD 0: pre_expert(first_layer) ──
        {
            let li = first_layer - (first_layer + 1) / FULL_ATTN_INTERVAL;
            let cmd = m.ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            crate::pipeline_fusedexp::encode_pre_expert(
                &m.wf, &m.gpu_wf, &m.ctx, &enc, first_layer, li,
                hd, num_k_heads, num_v_heads, total_key, total_val, qkv_dim,
                key_dim, val_dim, k_heads_per_v, inv_scale, num_experts, shared_inter,
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Update conv_state on CPU
            if let Some(ref mut s) = lin[first_layer] {
                crate::pipeline_fusedexp::update_conv_state(s, qkv_dim);
            }
        }

        // Route for first layer
        let (expert_indices_0, expert_weights_0, shared_gate_score_0) =
            route_and_pread(&m.ctx, m.expert_io.as_mut().unwrap(), first_layer, m.layer_fds[first_layer]);

        // ── Middle CMDs: post_expert(L_{i-1}) + pre_expert(L_i) ──
        // Keep track of the previous layer's routing results for the current CMD
        let mut prev_expert_indices = expert_indices_0;
        let mut prev_expert_weights = expert_weights_0;
        let mut prev_shared_gate_score = shared_gate_score_0;

        for gi in 1..m_layers {
            let prev_layer = group_layers[gi - 1];
            let curr_layer = group_layers[gi];
            let curr_li = curr_layer - (curr_layer + 1) / FULL_ATTN_INTERVAL;

            // Next layer's norm weight for GPU input_norm in post_expert
            let next_norm = m.wf.get_tensor_ptr(
                &format!("model.layers.{}.input_layernorm.weight", curr_layer));
            let next_norm_info = next_norm.map(|p| (p as *const std::ffi::c_void, m.gpu_wf.base as usize));

            let cmd = m.ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();

            // post_expert(prev_layer)
            {
                let io = m.expert_io.as_ref().unwrap();
                crate::pipeline_fusedexp::encode_post_expert(
                    &m.wf, &m.gpu_wf, &m.ctx, &enc, prev_layer,
                    &prev_expert_weights, prev_shared_gate_score,
                    &io.expert_data, &io.scratch_gate, &io.scratch_up, &io.scratch_act,
                    &io.expert_out, &io.shared_act, &io.shared_down, &io.combine_params,
                    next_norm_info,
                    hd, moe_inter, shared_inter, k, layout,
                );
            }

            // pre_expert(curr_layer)
            crate::pipeline_fusedexp::encode_pre_expert(
                &m.wf, &m.gpu_wf, &m.ctx, &enc, curr_layer, curr_li,
                hd, num_k_heads, num_v_heads, total_key, total_val, qkv_dim,
                key_dim, val_dim, k_heads_per_v, inv_scale, num_experts, shared_inter,
            );

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Update conv_state for curr_layer
            if let Some(ref mut s) = lin[curr_layer] {
                crate::pipeline_fusedexp::update_conv_state(s, qkv_dim);
            }

            // Capture hidden for prev_layer (buf_moe_hidden has post_expert(prev_layer) output)
            if capture_per_layer {
                let mut h = vec![0.0f32; hd];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        m.ctx.buf_moe_hidden.as_ref().unwrap().contents() as *const f32,
                        h.as_mut_ptr(), hd);
                }
                layer_outputs.push(h);
            }

            // Route for curr_layer (needed by next iteration's post_expert or last CMD)
            {
                let (indices, weights, gate_score) =
                    route_and_pread(&m.ctx, m.expert_io.as_mut().unwrap(), curr_layer, m.layer_fds[curr_layer]);
                prev_expert_indices = indices;
                prev_expert_weights = weights;
                prev_shared_gate_score = gate_score;
            }
        }

        // ── Last CMD: post_expert(last_layer) only ──
        {
            // Next norm may be None (last layer of model) or point to next layer
            let next_norm_info = if last_layer + 1 < num_layers {
                m.wf.get_tensor_ptr(
                    &format!("model.layers.{}.input_layernorm.weight", last_layer + 1))
                    .map(|p| (p as *const std::ffi::c_void, m.gpu_wf.base as usize))
            } else {
                None
            };

            let cmd = m.ctx.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            {
                let io = m.expert_io.as_ref().unwrap();
                crate::pipeline_fusedexp::encode_post_expert(
                    &m.wf, &m.gpu_wf, &m.ctx, &enc, last_layer,
                    &prev_expert_weights, prev_shared_gate_score,
                    &io.expert_data, &io.scratch_gate, &io.scratch_up, &io.scratch_act,
                    &io.expert_out, &io.shared_act, &io.shared_down, &io.combine_params,
                    next_norm_info,
                    hd, moe_inter, shared_inter, k, layout,
                );
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // Read final hidden from buf_moe_hidden
            unsafe {
                std::ptr::copy_nonoverlapping(
                    m.ctx.buf_moe_hidden.as_ref().unwrap().contents() as *const f32,
                    hidden.as_mut_ptr(), hd);
            }

            // Capture per-layer output for last_layer
            if capture_per_layer {
                layer_outputs.push(hidden.to_vec());
            }
        }
    }

    // Complete any remaining deferred (last layer was full attention)
    if let Some(ref mut def) = pending_deferred.take() {
        def.complete(hidden, hd);
        if capture_per_layer {
            layer_outputs.push(hidden.to_vec());
        }
    }

    Ok(())
}

fn process_token_inner(
    m: &mut ModelState,
    hidden: &mut [f32],
    pos: usize,
    kv: &mut [Option<FullAttnCache>],
    lin: &mut [Option<LinearAttnState>],
    py: Python<'_>,
    capture_per_layer: bool,
    layer_outputs: &mut Vec<Vec<f32>>,
) -> PyResult<()> {
    if m.pipeline_mode == PipelineMode::FusedExp {
        return process_token_fusedexp_pipelined(m, hidden, pos, kv, lin, py, capture_per_layer, layer_outputs);
    }

    let mut deferred: Option<DeferredExperts> = None;
    let mode = m.pipeline_mode;
    for layer in 0..m.config.num_layers {
        // Check for Ctrl-C every 4 layers (each layer ~5-10ms)
        if layer % 4 == 0 {
            py.check_signals()?;
        }
        // FAST PATH: previous CMD3 wrote input_norm to buf_input. Skip CPU rms_norm
        // and submit CMD1 immediately — GPU queue serializes CMD3(N-1) then CMD1(N).
        let prev_gpu_combined = deferred.as_ref().map_or(false, |d| d.gpu_combined);
        if !prev_gpu_combined {
            // SLOW PATH: complete previous layer's async MoE (CPU wait + readback)
            if let Some(ref mut def) = deferred.take() {
                def.complete(hidden, m.config.hidden_dim);
            }
        }
        let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;
        let mut attn_state: Option<FullAttnCmd2State> = None;
        let mut lin_state: Option<LinearAttnFusedWoodsState> = None;
        let mut h_mid_saved: Option<Vec<f32>> = None;
        if is_full {
            // FAST PATH: complete previous layer's MoE before full attention.
            // CMD1 for linear layers calls complete_fast() after CMD1; full attention
            // must do it before the forward so the correct hidden (MoE output) is used
            // for input_norm and residual_add.
            if prev_gpu_combined {
                if let Some(ref mut def) = deferred.take() {
                    def.complete_fast(hidden, m.config.hidden_dim);
                }
                h_mid_saved = Some(hidden.to_vec());
            }
            if let Some(ref mut kv) = kv[layer] {
                attn_state = full_attention_forward(&m.wf, layer, hidden, kv, pos, &m.config, Some(&m.gpu_wf), Some(&m.ctx), mode);
            }
        } else if let Some(ref mut s) = lin[layer] {
            let li = layer - (layer + 1) / FULL_ATTN_INTERVAL;
            // FusedExp: CMD1 does out_proj + residual_add internally, so it needs
            // the correct hidden for both input_norm and residual. Complete the
            // deferred before CMD1 to get the correct MoE output from prev layer.
            if prev_gpu_combined && mode == PipelineMode::FusedExp {
                if let Some(ref mut def) = deferred.take() {
                    def.complete_fast(hidden, m.config.hidden_dim);
                }
            }
            // In FusedWoods, h_mid must be saved before linear_attention_forward because
            // it doesn't modify hidden (matching C: CMD2 handles residual_add).
            // FAST PATH: hidden is stale (CMD3(N-1) not yet completed), so defer
            // h_mid capture until after CMD1 + complete_fast().
            if mode == PipelineMode::FusedWoods && !prev_gpu_combined {
                h_mid_saved = Some(hidden.to_vec());
            }
            lin_state = linear_attention_forward(
                &m.wf, layer, hidden, s,
                m.config.hidden_dim,
                m.config.linear_num_k_heads, m.config.linear_num_v_heads,
                m.config.linear_total_key, m.config.linear_total_value, m.config.linear_conv_dim,
                Some(&m.gpu_wf), Some(&m.ctx), li, mode, prev_gpu_combined,
            );

            if prev_gpu_combined {
                if let Some(ref mut def) = deferred.take() {
                    def.complete_fast(hidden, m.config.hidden_dim);
                }
                // Fix h_mid in the state — was set from stale hidden during CMD1
                if let Some(ref mut ls) = lin_state {
                    ls.h_mid.copy_from_slice(hidden);
                }
                h_mid_saved = Some(hidden.to_vec());
            }
            // In FusedWoods: restore hidden to h_mid so moe_layer_forward uses pre-attention state
            if let Some(ref hmid) = h_mid_saved {
                hidden.copy_from_slice(hmid);
            }
        }
        let r = moe_layer_forward(
            &m.wf, layer, hidden, m.layer_fds[layer],
            Some(&m.ctx), Some(&m.gpu_wf), &m.config, mode, attn_state, lin_state,
            m.expert_io.as_mut(),
        );
        deferred = r.unwrap_or(None);
        // Capture hidden state after this decoder layer completes.
        // The MoE output is already in hidden (via complete() for GPU path or
        // via the final combine in CPU path). This matches MLX layer.__call__ output.
        if capture_per_layer {
            layer_outputs.push(hidden.to_vec());
        }
    }
    // Complete last layer's deferred
    if let Some(ref mut def) = deferred {
        def.complete(hidden, m.config.hidden_dim);
    }
    Ok(())
}

fn process_token(m: &mut ModelState, hidden: &mut [f32], pos: usize,
    kv: &mut [Option<FullAttnCache>], lin: &mut [Option<LinearAttnState>],
    py: Python<'_>,
) -> PyResult<()> {
    process_token_inner(m, hidden, pos, kv, lin, py, false, &mut Vec::new())
}

// ─── Sampling ───────────────────────────────────────────────────────────────

fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = x.iter_mut().map(|v| { *v = (*v - max).exp(); *v }).sum();
    for v in x { *v /= sum; }
}

fn sample(logits: &mut [f32], temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> usize {
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

    // Top-k
    if top_k > 0 && top_k < n {
        let mut v: Vec<f32> = logits.to_vec();
        v.select_nth_unstable_by(top_k, |a, b| b.partial_cmp(a).unwrap());
        let t = v[top_k - 1];
        for x in logits.iter_mut() { if *x < t { *x = 0.0; } }
    }
    // Top-p
    if top_p < 1.0 {
        let mut s: Vec<f32> = logits.iter().copied().filter(|&x| x > 0.0).collect();
        s.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let total: f32 = s.iter().sum();
        let mut cum = 0.0;
        let mut cut = 0.0;
        for v in s { cum += v; if cum / total >= top_p { cut = v; break; } }
        for x in logits.iter_mut() { if *x < cut { *x = 0.0; } }
    }
    // Min-p
    if min_p > 0.0 {
        let max_p = logits.iter().fold(0.0f32, |a, &b| a.max(b));
        let t = max_p * min_p;
        for x in logits.iter_mut() { if *x < t { *x = 0.0; } }
    }

    let sum: f32 = logits.iter().sum();
    if sum <= 0.0 { return 0; }
    let inv = 1.0 / sum;
    use rand::Rng;
    let r: f32 = rand::thread_rng().gen();
    let mut cum = 0.0;
    for (i, &v) in logits.iter().enumerate() {
        cum += v * inv;
        if r <= cum { return i; }
    }
    n - 1
}

// ─── Python classes ─────────────────────────────────────────────────────────

#[pyclass(unsendable)]
pub struct Cache {
    pos: usize,
    kv: Vec<Option<FullAttnCache>>,
    lin: Vec<Option<LinearAttnState>>,
}

#[pymethods]
impl Cache {
    #[getter]
    fn pos(&self) -> usize { self.pos }

    /// Reset all state: zero position, clear KV caches, reset linear states.
    fn reset(&mut self) {
        self.pos = 0;
        for kv in self.kv.iter_mut().flatten() { kv.reset(); }
        for s in self.lin.iter_mut().flatten() {
            s.conv_state.fill(0.0);
            s.ssm_state.fill(0.0);
        }
    }

    fn __repr__(&self) -> String { format!("Cache(pos={})", self.pos) }
}

/// Lightweight telemetry snapshot returned to Python.
#[derive(Clone)]
struct Telemetry {
    prefill_ms: f64,
    total_ms: f64,
    tokens_generated: usize,
}

#[pyclass(unsendable)]
pub struct Context {
    model: Option<ModelState>,
    config: Option<ModelConfig>,  // cached for new_cache() even after unload
    telemetry: Telemetry,
}

#[pymethods]
impl Context {
    #[new]
    fn new() -> Self {
        Context {
            model: None,
            config: None,
            telemetry: Telemetry { prefill_ms: 0.0, total_ms: 0.0, tokens_generated: 0 },
        }
    }

    /// Load a model. Must be called before forward/generate.
    #[pyo3(signature = (model_path, pipeline_mode="FusedExp"))]
    fn load_model(&mut self, model_path: &str, pipeline_mode: &str) -> PyResult<()> {
        let mode = match pipeline_mode {
            "Cpu" | "CpuOnly" => PipelineMode::Cpu,
            "Gpu" => PipelineMode::Gpu,
            "FusedExp" => PipelineMode::FusedExp,
            "FusedWoods" => PipelineMode::FusedWoods,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown pipeline_mode: {}. Use Cpu|Gpu|FusedExp|FusedWoods", pipeline_mode)
            )),
        };
        let ms = ModelState::load(model_path, mode)?;
        self.config = Some(ms.config.clone());
        self.model = Some(ms);
        Ok(())
    }

    /// Unload the current model, freeing Metal resources and closing expert files.
    fn unload_model(&mut self) {
        self.model = None;
    }

    /// Create a new Cache sized for the loaded model (or a given model_path).
    #[pyo3(signature = (model_path=None))]
    fn new_cache(&self, model_path: Option<&str>) -> PyResult<Cache> {
        let config: ModelConfig = if let Some(ref m) = self.model {
            m.config.clone()
        } else if let Some(ref c) = self.config {
            c.clone()
        } else if let Some(path) = model_path {
            load_model_config(&PathBuf::from(path)).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("config: {}", e))
            })?
        } else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No model loaded and no model_path given"
            ));
        };
        Cache::from_config(&config)
    }

    /// forward(input_ids: [n]int64, cache: Cache) -> [n, d]float32
    fn forward(&mut self, py: Python<'_>, input_ids: &Bound<PyArray1<i64>>, cache: &mut Cache) -> PyResult<PyObject> {
        let t0 = Instant::now();
        let m = self.model.as_mut().ok_or_else(||
            pyo3::exceptions::PyRuntimeError::new_err("No model loaded"))?;
        let ids = input_ids.readonly();
        let ids = ids.as_slice()?;
        let n = ids.len();
        let start = cache.pos;
        let new_tokens = &ids[start..];
        let n_new = new_tokens.len();
        let (hd, vs) = (m.config.hidden_dim, m.config.vocab_size);

        let mut logits = vec![0.0f32; n * vs];
        if n_new == 0 {
            let arr = unsafe { PyArray2::<f32>::from_owned_array(py,
                numpy::ndarray::Array2::from_shape_vec((n, vs), logits).unwrap()) };
            return Ok(arr.into_py(py));
        }

        let mut embed = vec![0.0f32; n_new * hd];
        for (i, &id) in new_tokens.iter().enumerate() {
            embed_lookup(&m.wf, id as usize, &mut embed[i * hd..(i + 1) * hd], hd);
        }

        let mut hidden = vec![0.0f32; hd];
        for (ti, _) in new_tokens.iter().enumerate() {
            hidden.copy_from_slice(&embed[ti * hd..(ti + 1) * hd]);
            process_token(m, &mut hidden, cache.pos, &mut cache.kv, &mut cache.lin, py)?;
            cache.pos += 1;
            final_norm(&m.wf, &mut hidden, hd);
            lm_head(&m.wf, &hidden, &mut logits[(start + ti) * vs..(start + ti + 1) * vs], &m.gpu_wf, &m.ctx);
        }

        self.telemetry.prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.telemetry.total_ms = 0.0;
        self.telemetry.tokens_generated = 0;

        let arr = unsafe { PyArray2::<f32>::from_owned_array(py,
            numpy::ndarray::Array2::from_shape_vec((n, vs), logits).unwrap()) };
        Ok(arr.into_py(py))
    }

    /// forward_debug(input_ids: [n]int64, cache: Cache) -> (logits: [n, d]float32, layers: [[d]float32])
    ///
    /// Like forward() but also returns the hidden state after each decoder layer
    /// for the LAST token in the batch. Useful for comparing against MLX per-layer outputs.
    fn forward_debug(&mut self, py: Python<'_>, input_ids: &Bound<PyArray1<i64>>, cache: &mut Cache) -> PyResult<(PyObject, PyObject)> {
        let t0 = Instant::now();
        let m = self.model.as_mut().ok_or_else(||
            pyo3::exceptions::PyRuntimeError::new_err("No model loaded"))?;
        let ids = input_ids.readonly();
        let ids = ids.as_slice()?;
        let n = ids.len();
        let start = cache.pos;
        let new_tokens = &ids[start..];
        let n_new = new_tokens.len();
        let (hd, vs) = (m.config.hidden_dim, m.config.vocab_size);
        let num_layers = m.config.num_layers;

        let mut logits = vec![0.0f32; n * vs];
        if n_new == 0 {
            let arr = unsafe { PyArray2::<f32>::from_owned_array(py,
                numpy::ndarray::Array2::from_shape_vec((n, vs), logits).unwrap()) };
            let empty_list = PyList::empty(py);
            return Ok((arr.into_py(py), empty_list.into_py(py)));
        }

        let mut embed = vec![0.0f32; n_new * hd];
        for (i, &id) in new_tokens.iter().enumerate() {
            embed_lookup(&m.wf, id as usize, &mut embed[i * hd..(i + 1) * hd], hd);
        }

        let mut hidden = vec![0.0f32; hd];
        // Per-layer hidden states for ALL tokens (n_new * num_layers * hd)
        let mut all_layer_outputs: Vec<Vec<f32>> = Vec::new();

        for (ti, _) in new_tokens.iter().enumerate() {
            hidden.copy_from_slice(&embed[ti * hd..(ti + 1) * hd]);
            let mut layer_outputs = Vec::new();
            process_token_inner(m, &mut hidden, cache.pos, &mut cache.kv, &mut cache.lin, py, true, &mut layer_outputs)?;
            cache.pos += 1;
            // Append layer outputs BEFORE final_norm (hidden at this point is the MoE output)
            all_layer_outputs.push(hidden.to_vec());  // For convenience, also save the pre-norm hidden
            all_layer_outputs.extend(layer_outputs);
            final_norm(&m.wf, &mut hidden, hd);
            lm_head(&m.wf, &hidden, &mut logits[(start + ti) * vs..(start + ti + 1) * vs], &m.gpu_wf, &m.ctx);
        }

        self.telemetry.prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.telemetry.total_ms = 0.0;
        self.telemetry.tokens_generated = 0;

        // Build Python list of per-layer states for the LAST token only
        // all_layer_outputs layout: for each token: [final_hidden, layer_0, layer_1, ..., layer_N-1]
        // We want the last token's per-layer outputs (layer_0 through layer_N-1)
        let per_token_entries = 1 + num_layers;  // final_hidden + per-layer outputs
        let last_token_start = (n_new - 1) * per_token_entries;
        let py_list = PyList::empty(py);
        // Skip the "final_hidden" entry (index last_token_start), use per-layer outputs
        for li in 0..num_layers {
            let layer_hidden = &all_layer_outputs[last_token_start + 1 + li];
            let arr = unsafe { PyArray1::<f32>::from_owned_array(py,
                numpy::ndarray::Array1::from_vec(layer_hidden.clone())) };
            py_list.append(arr)?;
        }

        let arr = unsafe { PyArray2::<f32>::from_owned_array(py,
            numpy::ndarray::Array2::from_shape_vec((n, vs), logits).unwrap()) };
        Ok((arr.into_py(py), py_list.into_py(py)))
    }

    #[pyo3(signature = (input_ids, cache, max_tokens=256, temperature=0.0,
                        top_k=0, top_p=1.0, min_p=0.0, eos_token_ids=None))]
    fn generate(&mut self, py: Python<'_>, input_ids: &Bound<PyArray1<i64>>, cache: &mut Cache,
        max_tokens: usize, temperature: f32, top_k: usize, top_p: f32, min_p: f32,
        eos_token_ids: Option<&Bound<PyArray1<i64>>>,
    ) -> PyResult<PyObject> {
        let gen_t0 = Instant::now();
        let eos: HashSet<usize> = match eos_token_ids {
            Some(a) => a.readonly().to_vec()?.into_iter().map(|x| x as usize).collect(),
            None => [248046usize, 248044].into(),
        };

        // Prefill → get last logits
        let logits_obj = self.forward(py, input_ids, cache)?;
        let la = logits_obj.downcast_bound::<PyArray2<f32>>(py).map_err(|_|
            pyo3::exceptions::PyRuntimeError::new_err("expected ndarray"))?;
        let ls = unsafe { la.as_slice() }.map_err(|e|
            pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

        let m = self.model.as_mut().ok_or_else(||
            pyo3::exceptions::PyRuntimeError::new_err("No model loaded"))?;
        let (hd, vs) = (m.config.hidden_dim, m.config.vocab_size);
        let mut logits = ls[ls.len() - vs..].to_vec();

        let mut next = if temperature < 0.01 {
            logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0)
        } else { sample(&mut logits, temperature, top_k, top_p, min_p) };

        let mut output = Vec::with_capacity(max_tokens);
        let mut hidden = vec![0.0f32; hd];
        for _ in 0..max_tokens {
            if eos.contains(&next) { break; }
            output.push(next as i64);
            embed_lookup(&m.wf, next, &mut hidden, hd);
            process_token(m, &mut hidden, cache.pos, &mut cache.kv, &mut cache.lin, py)?;
            cache.pos += 1;
            final_norm(&m.wf, &mut hidden, hd);
            logits.fill(0.0);
            lm_head(&m.wf, &hidden, &mut logits, &m.gpu_wf, &m.ctx);
            next = if temperature < 0.01 {
                logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0)
            } else { sample(&mut logits, temperature, top_k, top_p, min_p) };
        }
        self.telemetry.total_ms = gen_t0.elapsed().as_secs_f64() * 1000.0;
        self.telemetry.tokens_generated = output.len();
        Ok(PyArray1::<i64>::from_vec(py, output).into_py(py))
    }

    #[pyo3(signature = (input_ids, cache, max_tokens=256, temperature=0.0,
                        top_k=0, top_p=1.0, min_p=0.0, eos_token_ids=None))]
    fn stream_generate(&mut self, py: Python<'_>, input_ids: &Bound<PyArray1<i64>>, cache: &mut Cache,
        max_tokens: usize, temperature: f32, top_k: usize, top_p: f32, min_p: f32,
        eos_token_ids: Option<&Bound<PyArray1<i64>>>,
    ) -> PyResult<PyObject> {
        let gen_t0 = Instant::now();
        let eos: HashSet<usize> = match eos_token_ids {
            Some(a) => a.readonly().to_vec()?.into_iter().map(|x| x as usize).collect(),
            None => [248046usize, 248044].into(),
        };

        // Prefill → get last logits
        let logits_obj = self.forward(py, input_ids, cache)?;
        let la = logits_obj.downcast_bound::<PyArray2<f32>>(py).map_err(|_|
            pyo3::exceptions::PyRuntimeError::new_err("expected ndarray"))?;
        let ls = unsafe { la.as_slice() }.map_err(|e|
            pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;

        let m = self.model.as_mut().ok_or_else(||
            pyo3::exceptions::PyRuntimeError::new_err("No model loaded"))?;
        let (hd, vs) = (m.config.hidden_dim, m.config.vocab_size);
        let mut logits = ls[ls.len() - vs..].to_vec();

        let mut next = if temperature < 0.01 {
            logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0)
        } else { sample(&mut logits, temperature, top_k, top_p, min_p) };

        let mut results: Vec<(i64, PyObject)> = Vec::with_capacity(max_tokens);
        let mut hidden = vec![0.0f32; hd];

        // Yield first token + its logits
        results.push((next as i64, PyArray1::<f32>::from_vec(py, logits.clone()).into_py(py)));

        for _ in 1..max_tokens {
            if eos.contains(&next) { break; }
            embed_lookup(&m.wf, next, &mut hidden, hd);
            process_token(m, &mut hidden, cache.pos, &mut cache.kv, &mut cache.lin, py)?;
            cache.pos += 1;
            final_norm(&m.wf, &mut hidden, hd);
            logits.fill(0.0);
            lm_head(&m.wf, &hidden, &mut logits, &m.gpu_wf, &m.ctx);
            next = if temperature < 0.01 {
                logits.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap_or(0)
            } else { sample(&mut logits, temperature, top_k, top_p, min_p) };
            results.push((next as i64, PyArray1::<f32>::from_vec(py, logits.clone()).into_py(py)));
        }
        self.telemetry.total_ms = gen_t0.elapsed().as_secs_f64() * 1000.0;
        self.telemetry.tokens_generated = results.len();
        Ok(results.into_py(py))
    }

    /// Return telemetry from the last forward/generate/stream_generate call.
    /// Keys: ttft_ms, prefill_ms, total_ms, tokens_generated, tokens_per_sec.
    /// tokens_per_sec excludes prefill and the first token.
    fn telemetry(&self, py: Python<'_>) -> PyResult<PyObject> {
        let t = &self.telemetry;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("prefill_ms", t.prefill_ms)?;
        dict.set_item("total_ms", t.total_ms)?;
        dict.set_item("tokens_generated", t.tokens_generated)?;
        let tps = if t.total_ms > 0.0 && t.tokens_generated > 1 {
            let gen_ms = t.total_ms - t.prefill_ms;  // exclude prefill
            if gen_ms > 0.0 {
                (t.tokens_generated - 1) as f64 / (gen_ms / 1000.0)  // exclude first token
            } else { 0.0 }
        } else { 0.0 };
        dict.set_item("tokens_per_sec", tps)?;
        Ok(dict.into_py(py))
    }

    fn __repr__(&self) -> String {
        match &self.model {
            Some(m) => format!("Context(loaded: {} layers, hidden={})", m.config.num_layers, m.config.hidden_dim),
            None => "Context(no model loaded)".into(),
        }
    }
}

impl Cache {
    fn from_config(config: &ModelConfig) -> PyResult<Self> {
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
        Ok(Cache { pos: 0, kv, lin })
    }
}

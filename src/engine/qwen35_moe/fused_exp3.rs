//! FusedExp3 — batched-prefill engine, layered on top of FusedExp2.
//!
//! Same per-token forward as FusedExp2 (delegated). Overrides:
//!   - `forward_hidden_batched` — layer-batched prefill: op1 batched for
//!     full-attn, per-token in one cmd buffer for linear-attn (DeltaNet
//!     recurrence is sequential), single-commit op2 with a unique-expert
//!     pool, batched lm_head. ~1.5-2.7× faster than the token-serial path
//!     on Apple M4 for prompt lengths 8..128. See `batched.rs`.
//!   - `snapshot` / `restore` — capture/restore mutable runtime state
//!     (pos, MTP pos, DeltaNet recurrent state, last_h_pre_norm). Needed
//!     for speculative-decoding rollback.
//!
//! Opt-in via `pipeline_mode="Qwen35MoEFusedExp3"`. The default
//! `Qwen35MoEFusedExp2` remains the production engine, byte-for-byte
//! unchanged.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;
use std::marker::PhantomData;

use metal::*;

use crate::cache::Cache;
use crate::constants::{FULL_ATTN_INTERVAL, MAX_SEQ, RMS_NORM_EPS};
use crate::engine::{Engine, EngineSnapshot, SignalCheckFn, TelemetryValue};
use crate::engine::metal_context::{MAX_K, metal_buf_shared};
use crate::engine::batched::{BatchedFullBuffers, ExpertPool, op1_full_batched, op1_linear_batched, encode_post_expert_at};
use crate::engine::fused_exp2::FusedExp2;
use crate::engine::qwen35_constants::ModelConfig;
use crate::error::MoEError;
use crate::math::{bf16_to_f32, normalize_weights, softmax, topk};
use crate::model::Model;
use crate::model::weights::WeightFile;

pub struct FusedExp3<C: ModelConfig> {
    inner: FusedExp2<C>,
    _phantom: PhantomData<C>,
}

impl<C: ModelConfig> FusedExp3<C> {
    pub fn new(model: Arc<Model>, num_active_experts: usize, expert_cache_count: usize) -> Result<Self, MoEError> {
        let inner = FusedExp2::<C>::new(model, num_active_experts, expert_cache_count)?;
        Ok(Self { inner, _phantom: PhantomData })
    }
}

// ─── Local helpers ──────────────────────────────────────────────────────────

/// Same RMSNorm at the model's final-norm position as FusedExp2's private
/// final_norm — duplicated here to keep fused_exp2.rs untouched.
fn final_norm(wf: &WeightFile, hidden: &mut [f32], hidden_dim: usize) {
    let Some(fnw_u16) = wf.get_tensor_u16("language_model.model.norm.weight") else { return };
    let fnw_f32: Vec<f32> = fnw_u16.iter().map(|&v| bf16_to_f32(v)).collect();
    let sum_sq: f32 = hidden[..hidden_dim].iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / hidden_dim as f32 + RMS_NORM_EPS).sqrt();
    for i in 0..hidden_dim {
        hidden[i] *= inv_rms * fnw_f32[i];
    }
}

/// Batched lm_head for N tokens. One Metal command buffer + one batched
/// matvec instead of N separate per-token (allocate, copy, dispatch, wait)
/// cycles.
fn gpu_lm_head_n<C: ModelConfig>(
    inner: &FusedExp2<C>,
    hiddens: &[f32], logits: &mut [f32],
    n: usize, hidden_dim: usize, vocab_size: usize,
) {
    debug_assert_eq!(hiddens.len(), n * hidden_dim);
    debug_assert_eq!(logits.len(), n * vocab_size);
    let ctx = &inner.ctx;
    let wf = &inner.model.weight_file;
    let wb = &inner.weight_buffer;
    let x_buf = metal_buf_shared(&ctx.device, n * hidden_dim * 4);
    unsafe {
        std::ptr::copy_nonoverlapping(hiddens.as_ptr(), x_buf.contents() as *mut f32, n * hidden_dim);
    }
    let out_buf = metal_buf_shared(&ctx.device, n * vocab_size * 4);
    let cm = ctx.queue.new_command_buffer();
    let enc = cm.new_compute_command_encoder();
    let ok = wb.encode_matvec_n_into(
        wf, ctx, &enc, "language_model.lm_head",
        &x_buf, 0, &out_buf, 0, vocab_size, hidden_dim, n as u32,
    );
    assert!(ok, "encode_matvec_n_into failed for language_model.lm_head");
    enc.end_encoding();
    cm.commit();
    cm.wait_until_completed();
    unsafe {
        std::ptr::copy_nonoverlapping(
            out_buf.contents() as *const f32, logits.as_mut_ptr(), n * vocab_size);
    }
}

/// CPU-only routing for one token: softmax + topk + normalize. No I/O.
fn route_cpu_only(num_active_experts: usize, scores: &mut [f32]) -> (Vec<usize>, Vec<f32>) {
    softmax(scores);
    let mut indices = vec![0usize; num_active_experts];
    let mut weights = vec![0.0f32; num_active_experts];
    topk(scores, num_active_experts, &mut indices, &mut weights);
    normalize_weights(&mut weights);
    (indices, weights)
}

/// Parallel pread the given UNIQUE expert IDs from this layer's expert file
/// into pool.slot(expert_id). Each unique expert is loaded ONCE per layer
/// regardless of how many tokens picked it — the major pread-bandwidth
/// saving over the per-token pool design.
fn pread_unique_experts<C: ModelConfig>(inner: &FusedExp2<C>, layer: usize, unique_ids: &[usize], pool: &ExpertPool) {
    let expert_file = &inner.model.expert_files[layer];
    let expert_size = expert_file.expert_size();
    rayon::scope(|s| {
        for &eid in unique_ids {
            let dst_slot = pool.slot(eid);
            let dst = unsafe {
                std::slice::from_raw_parts_mut(dst_slot.contents() as *mut u8, expert_size)
            };
            s.spawn(move |_| {
                expert_file.read_expert(eid, dst).unwrap();
            });
        }
    });
}

/// Plain-old-data routing record (avoids depending on FusedExp2's private Routing).
struct TokenRouting {
    expert_weights: Vec<f32>,
    shared_gate_score: f32,
    expert_data: Vec<Buffer>,
}

// ─── Engine impl: delegate non-batched ops to inner FusedExp2 ──────────────

impl<C: ModelConfig> Engine for FusedExp3<C> {
    fn upload_cache(&self, cache: &Cache) { self.inner.upload_cache(cache); }
    fn download_cache(&self, cache: &mut Cache) { self.inner.download_cache(cache); }
    fn engine_pos(&self) -> usize { self.inner.engine_pos() }
    fn embed_lookup(&self, token_ids: &[i64], embeddings: &mut [f32]) {
        self.inner.embed_lookup(token_ids, embeddings);
    }
    fn forward_hidden(
        &mut self, embeddings: &[f32], check_signal: SignalCheckFn<'_>, mtp: bool,
    ) -> Result<Vec<f32>, MoEError> {
        self.inner.forward_hidden(embeddings, check_signal, mtp)
    }
    fn last_h_pre_norm(&self) -> &[f32] { self.inner.last_h_pre_norm() }
    fn mtp_forward(&mut self, token_id: usize) -> Vec<f32> { self.inner.mtp_forward(token_id) }
    fn mtp_reset(&mut self) { self.inner.mtp_reset() }
    fn mtp_rollback(&mut self, pos: usize) { self.inner.mtp_rollback(pos) }
    fn telemetry(&self) -> BTreeMap<String, TelemetryValue> { self.inner.telemetry() }

    fn forward_hidden_batched(
        &mut self, embeddings: &[f32], check_signal: SignalCheckFn<'_>, mtp: bool,
    ) -> Result<Vec<f32>, MoEError> {
        let t0 = Instant::now();
        let hidden_dim = C::HIDDEN_DIM;
        let n = embeddings.len() / hidden_dim;
        let vocab_size = C::VOCAB_SIZE;
        let num_layers = C::NUM_LAYERS;
        let num_experts = C::NUM_EXPERTS;

        let mut logits = vec![0.0f32; n * vocab_size];
        if n == 0 { return Ok(logits); }
        // For N=1 (chat generation), fall back to the token-serial path —
        // batched overhead exceeds the win at a single token.
        if n == 1 {
            return self.inner.forward_hidden(embeddings, check_signal, mtp);
        }

        let past_pos = self.inner.ctx.pos.get();
        assert!(past_pos + n <= MAX_SEQ, "past_pos + n ({} + {}) exceeds MAX_SEQ ({})",
                past_pos, n, MAX_SEQ);

        // Per-call buffers + per-layer unique-expert pool.
        let bufs = BatchedFullBuffers::new::<C>(&self.inner.ctx.device, n);
        let expert_size = self.inner.model.expert_files[0].expert_size();
        let pool = ExpertPool::new(&self.inner.ctx.device, num_experts, expert_size);

        // Seed bufs.hidden_n with the input embeddings.
        unsafe {
            std::ptr::copy_nonoverlapping(
                embeddings.as_ptr(),
                bufs.hidden_n.contents() as *mut f32,
                n * hidden_dim,
            );
        }

        for layer in 0..num_layers {
            if check_signal() {
                return Err(MoEError::Metal("interrupted".into()));
            }
            let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;

            // ── Op1: produce per-token (post_normed, temp_residual, shared_*, gate_scores) ──
            if is_full {
                let fa_idx = layer / FULL_ATTN_INTERVAL;
                let cm = op1_full_batched::<C>(
                    &self.inner.model.weight_file, &self.inner.weight_buffer, &self.inner.ctx,
                    layer, fa_idx, past_pos, n, &bufs,
                );
                cm.commit();
                cm.wait_until_completed();
            } else {
                let linear_idx = layer - (layer + 1) / FULL_ATTN_INTERVAL;
                let cm = op1_linear_batched::<C>(
                    &self.inner.model.weight_file, &self.inner.weight_buffer, &self.inner.ctx,
                    layer, linear_idx, n, &bufs,
                );
                cm.commit();
                cm.wait_until_completed();
            }

            // ── Batched MoE op2: one cmd buffer per layer covers all N tokens ──
            // Phase 1: CPU routing (softmax/topk/normalize) per token.
            let k = self.inner.num_active_experts;
            let mut all_indices: Vec<Vec<usize>> = Vec::with_capacity(n);
            let mut all_weights: Vec<Vec<f32>>   = Vec::with_capacity(n);
            let mut sg_scores: Vec<f32>          = Vec::with_capacity(n);
            for ti in 0..n {
                let mut gs_vec = vec![0.0f32; num_experts];
                let sg_score: f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (bufs.gate_scores_n.contents() as *const f32).add(ti * num_experts),
                        gs_vec.as_mut_ptr(), num_experts);
                    sg_score = *((bufs.shared_gate_score_n.contents() as *const f32).add(ti));
                }
                let (idx, w) = route_cpu_only(k, &mut gs_vec);
                all_indices.push(idx);
                all_weights.push(w);
                sg_scores.push(sg_score);
            }
            // Phase 2: pread each UNIQUE expert ONCE (the big bandwidth win).
            let unique_ids: Vec<usize> = {
                let mut seen = vec![false; num_experts];
                let mut ids = Vec::new();
                for indices in &all_indices {
                    for &eid in indices {
                        if !seen[eid] { seen[eid] = true; ids.push(eid); }
                    }
                }
                ids
            };
            pread_unique_experts(&self.inner, layer, &unique_ids, &pool);
            // Phase 3: build Routing structs with refs into the unique pool.
            let actual_k = k.min(MAX_K);
            let routings: Vec<TokenRouting> = (0..n).map(|ti| {
                let expert_data: Vec<Buffer> = (0..actual_k)
                    .map(|ki| pool.slot(all_indices[ti][ki]).clone())
                    .collect();
                TokenRouting {
                    expert_weights: all_weights[ti].clone(),
                    shared_gate_score: sg_scores[ti],
                    expert_data,
                }
            }).collect();
            // Phase 4: single command buffer, all N op2 dispatches.
            {
                let cm = self.inner.ctx.queue.new_command_buffer().to_owned();
                let enc = cm.new_compute_command_encoder();
                for ti in 0..n {
                    encode_post_expert_at::<C>(
                        &self.inner.model.weight_file, &self.inner.weight_buffer, &self.inner.ctx, &enc,
                        layer, ti,
                        &routings[ti].expert_weights, routings[ti].shared_gate_score,
                        &routings[ti].expert_data, &self.inner.expert_buffer,
                        self.inner.num_active_experts,
                        &bufs,
                    );
                }
                enc.end_encoding();
                cm.commit();
                cm.wait_until_completed();
            }
            // MTP: stash the last token's hidden state (pre-norm) at the last layer.
            if mtp && layer + 1 == num_layers {
                let mut last = vec![0.0f32; hidden_dim];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (bufs.hidden_n.contents() as *const f32).add((n - 1) * hidden_dim),
                        last.as_mut_ptr(),
                        hidden_dim,
                    );
                }
                self.inner.last_h_pre_norm = last;
            }
        }

        self.inner.ctx.pos.set(past_pos + n);

        // Batched final_norm + lm_head.
        let mut hiddens_flat = vec![0.0f32; n * hidden_dim];
        unsafe {
            std::ptr::copy_nonoverlapping(
                bufs.hidden_n.contents() as *const f32,
                hiddens_flat.as_mut_ptr(),
                n * hidden_dim,
            );
        }
        for ti in 0..n {
            final_norm(
                &self.inner.model.weight_file,
                &mut hiddens_flat[ti * hidden_dim..(ti + 1) * hidden_dim],
                hidden_dim,
            );
        }
        gpu_lm_head_n(&self.inner, &hiddens_flat, &mut logits, n, hidden_dim, vocab_size);

        // Stash timing for telemetry.
        let total = t0.elapsed().as_secs_f64() * 1000.0;
        if crate::engine::record_telemetry() {
            self.inner.timing.entry("engine.total_ms".into())
                .and_modify(|v| if let TelemetryValue::Scalar(ref mut x) = v { *x += total; })
                .or_insert(TelemetryValue::Scalar(total));
        }
        Ok(logits)
    }

    fn snapshot(&self) -> EngineSnapshot {
        let copy_buf = |b: &Buffer| -> Vec<u8> {
            let len = b.length() as usize;
            let mut v = vec![0u8; len];
            unsafe {
                std::ptr::copy_nonoverlapping(b.contents() as *const u8, v.as_mut_ptr(), len);
            }
            v
        };
        EngineSnapshot {
            pos: self.inner.ctx.pos.get(),
            mtp_pos: self.inner.mtp_state.as_ref().map(|s| s.pos).unwrap_or(0),
            last_h_pre_norm: self.inner.last_h_pre_norm.clone(),
            conv_state:  self.inner.ctx.buf_conv_state.iter().map(copy_buf).collect(),
            delta_state: self.inner.ctx.buf_delta_state.iter().map(copy_buf).collect(),
        }
    }

    fn restore(&mut self, snap: &EngineSnapshot) {
        self.inner.ctx.pos.set(snap.pos);
        if let Some(ref mut s) = self.inner.mtp_state {
            s.pos = snap.mtp_pos;
        }
        self.inner.last_h_pre_norm = snap.last_h_pre_norm.clone();
        for (i, src) in snap.conv_state.iter().enumerate() {
            if i < self.inner.ctx.buf_conv_state.len() {
                let dst = &self.inner.ctx.buf_conv_state[i];
                debug_assert_eq!(src.len(), dst.length() as usize);
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst.contents() as *mut u8, src.len());
                }
            }
        }
        for (i, src) in snap.delta_state.iter().enumerate() {
            if i < self.inner.ctx.buf_delta_state.len() {
                let dst = &self.inner.ctx.buf_delta_state[i];
                debug_assert_eq!(src.len(), dst.length() as usize);
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), dst.contents() as *mut u8, src.len());
                }
            }
        }
    }
}

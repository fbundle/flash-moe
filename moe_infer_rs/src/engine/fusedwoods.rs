/// FusedWoods pipeline mode: 3-CMD architecture matching the C engine.
///
/// CMD1: attention projections + conv1d + SSM + gated_rms_norm (no out_proj/residual)
/// CMD2: out_proj + residual_add + rms_norm + gate + shared (1 fused encoder)
/// CMD3: experts + combine + GPU-side input_norm (async, deferred commit)
use crate::cache::{Cache, FullAttnCache, LinearAttnState};
use crate::constants::FULL_ATTN_INTERVAL;
use crate::engine::Engine;
use crate::metal_context::{ExpertBuffer, WeightBuffer, MetalContext};
use crate::model::Model;
use crate::engine::{ExecCtxGpu, SignalCheckFn};
use crate::math::{embed_lookup, final_norm};
use crate::math_full_attention::FullAttnGpuOut;
use crate::math_full_attention::mixed_full_attention_forward;
use crate::math_linear_attention::{self, LinearAttnGpuOut};
use crate::math_lm_head::gpu_lm_head;
use crate::math_moe::{DeferredExperts, moe_layer_forward};

// ─── General-purpose token processing ──────────────────────────────────────────

pub fn process_token_inner(
    exec: &mut ExecCtxGpu<'_>,
    hidden: &mut [f32],
    pos: usize,
    kv: &mut [Option<FullAttnCache>],
    lin: &mut [Option<LinearAttnState>],
    check_signal: SignalCheckFn<'_>,
    capture_per_layer: bool,
    layer_outputs: &mut Vec<Vec<f32>>,
    use_fusedwoods: bool,
) -> Result<(), String> {
    let mut deferred: Option<DeferredExperts> = None;
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
        let mut attn_state: Option<FullAttnGpuOut> = None;
        let mut lin_state: Option<LinearAttnGpuOut> = None;
        let mut h_mid_saved: Option<Vec<f32>> = None;
        if is_full {
            if prev_gpu_combined {
                if let Some(ref mut def) = deferred.take() {
                    def.complete_fast(hidden, hd);
                }
            }
            if let Some(ref mut kv) = kv[layer] {
                attn_state = mixed_full_attention_forward(
                    exec.wf, layer, hidden, kv, pos, exec.config,
                    Some(exec.gpu_wf), Some(exec.ctx));
            }
        } else if let Some(ref mut s) = lin[layer] {
            let li = layer - (layer + 1) / FULL_ATTN_INTERVAL;
            if use_fusedwoods && !prev_gpu_combined {
                h_mid_saved = Some(hidden.to_vec());
            }
            lin_state = math_linear_attention::gpu_linear_attention(
                exec.wf, layer, hidden, s,
                hd,
                exec.config.linear_num_k_heads, exec.config.linear_num_v_heads,
                exec.config.linear_total_key, exec.config.linear_total_value,
                exec.config.linear_conv_dim,
                Some(exec.gpu_wf), Some(exec.ctx), li,
                false, use_fusedwoods, prev_gpu_combined,
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
            attn_state, lin_state,
            exec.expert_gpu_buffer.as_mut().map(|x| &mut **x),
            use_fusedwoods,
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

// ─── EngineFusedWoods ─────────────────────────────────────────────────────

pub struct EngineFusedWoods<'a> {
    pub model: &'a Model,
    pub ctx: &'a MetalContext,
    pub gpu_wf: &'a WeightBuffer,
    pub expert_gpu_buffer: Option<&'a mut ExpertBuffer>,
}

impl<'a> EngineFusedWoods<'a> {
    fn make_exec_ctx(&mut self) -> ExecCtxGpu<'_> {
        ExecCtxGpu {
            wf: &self.model.wf,
            ctx: self.ctx,
            gpu_wf: self.gpu_wf,
            config: &self.model.config,
            expert_fds: &self.model.expert_fds,
            expert_gpu_buffer: self.expert_gpu_buffer.as_deref_mut(),
        }
    }
}

impl<'a> Engine for EngineFusedWoods<'a> {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, String> {
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
            let mut exec = self.make_exec_ctx();
            process_token_inner(
                &mut exec, &mut hidden,
                cache.pos, &mut cache.kv, &mut cache.lin,
                &mut || check_signal(), false, &mut Vec::new(),
                true,
            )?;
            cache.pos += 1;
            final_norm(exec.wf, &mut hidden, hd);
            gpu_lm_head(exec.wf, &hidden,
                &mut logits[ti * vs..(ti + 1) * vs],
                exec.gpu_wf, exec.ctx);
        }

        Ok(logits)
    }
}

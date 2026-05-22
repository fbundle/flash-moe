use crate::cache::Cache;
use crate::constants::{FULL_ATTN_INTERVAL, RMS_NORM_EPS};
use crate::engine::Engine;
use crate::model::Model;
use crate::math_linear_attention;
use crate::engine::SignalCheckFn;
use crate::math::{
    bf16_to_f32, embed_lookup, final_norm,
    rms_norm,
};
use crate::math_full_attention::full_attention_forward;
use crate::math_moe::moe_layer_forward;
use crate::math_lm_head::lm_head;

/// CPU-only engine: no GPU resources required.
pub struct EngineCPU<'a> {
    pub model: &'a Model,
}

impl<'a> Engine for EngineCPU<'a> {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, String> {
        let n = input_ids.len();
        let hd = self.model.config.hidden_dim;
        let vs = self.model.config.vocab_size;
        let num_layers = self.model.config.num_layers;

        let mut logits = vec![0.0f32; n * vs];
        if n == 0 {
            return Ok(logits);
        }

        let mut embed = vec![0.0f32; n * hd];
        for (i, &id) in input_ids.iter().enumerate() {
            embed_lookup(&self.model.wf, id as usize, &mut embed[i * hd..(i + 1) * hd], hd);
        }

        let num_k_heads = self.model.config.linear_num_k_heads;
        let num_v_heads = self.model.config.linear_num_v_heads;
        let total_key = self.model.config.linear_total_key;
        let total_value = self.model.config.linear_total_value;
        let qkv_dim = self.model.config.linear_conv_dim;
        let key_dim = total_key / num_k_heads;
        let value_dim = total_value / num_v_heads;
        let inv_scale = 1.0 / (key_dim as f32).sqrt();
        let k_heads_per_v = num_v_heads / num_k_heads;

        let mut hidden = vec![0.0f32; hd];
        for (ti, _) in input_ids.iter().enumerate() {
            hidden.copy_from_slice(&embed[ti * hd..(ti + 1) * hd]);

            for layer in 0..num_layers {
                if layer % 4 == 0 && check_signal() {
                    return Err("interrupted".into());
                }
                let is_full = (layer + 1) % FULL_ATTN_INTERVAL == 0;

                // Input norm + save residual
                let norm_name = format!("model.layers.{}.input_layernorm.weight", layer);
                let nw_u16 = self.model.wf.get_tensor_u16(&norm_name);
                let residual = hidden.to_vec();
                let mut normed = vec![0.0f32; hd];
                if let Some(nw) = nw_u16 {
                    let nw_f32: Vec<f32> = nw.iter().map(|&v| bf16_to_f32(v)).collect();
                    rms_norm(&hidden, &nw_f32, &mut normed, hd, RMS_NORM_EPS);
                } else {
                    normed.copy_from_slice(&hidden);
                }

                let attn_state = if is_full {
                    if let Some(ref mut kv) = cache.kv[layer] {
                        full_attention_forward(
                            &self.model.wf, layer, &mut hidden, kv, cache.pos,
                            &self.model.config,
                        )
                    } else {
                        None
                    }
                } else {
                    if let Some(ref mut state) = cache.lin[layer] {
                        math_linear_attention::linear_attention(
                            &self.model.wf, layer, &mut hidden, &normed, &residual, state,
                            num_k_heads, num_v_heads, total_key, total_value, qkv_dim,
                            hd, key_dim, value_dim, inv_scale, k_heads_per_v,
                        );
                    }
                    None
                };

                let _ = moe_layer_forward(
                    &self.model.wf, layer, &mut hidden,
                    self.model.expert_fds[layer],
                    None, None, &self.model.config,
                    attn_state, None, None, false,
                );
            }

            cache.pos += 1;
            final_norm(&self.model.wf, &mut hidden, hd);
            lm_head(&self.model.wf, &hidden, &mut logits[ti * vs..(ti + 1) * vs]);
        }

        Ok(logits)
    }
}

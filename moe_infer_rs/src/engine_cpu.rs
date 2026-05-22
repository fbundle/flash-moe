/// CPU linear attention path.
///
/// Runs attention projections, conv1d, SSM, gated_rms_norm, out_proj, and residual
/// add entirely on CPU. Used when GPU fused pipelines are unavailable or when mode
/// is CpuOnly.
use crate::constants::{CONV_KERNEL_SIZE, FULL_ATTN_INTERVAL, GROUP_SIZE, RMS_NORM_EPS};
use crate::engine::{Cache, Engine, LinearAttnState, Model};
use crate::engine_common::{
    bf16_to_f32, conv1d_step, dequant_matvec_4bit,
    embed_lookup, final_norm, full_attention_forward, lm_head,
    moe_layer_forward, rms_norm, rms_norm_bare, rms_norm_gated, sigmoid,
    SignalCheckFn,
};
use crate::model_weights::WeightFile;

/// Run CPU linear attention: dequant projections → conv1d → SSM → gated_norm → out_proj → residual.
///
/// Writes final hidden (with residual) into `hidden`.
pub fn cpu_linear_attention(
    wf: &WeightFile,
    layer_idx: usize,
    hidden: &mut [f32],
    normed: &[f32],
    residual: &[f32],
    state: &mut LinearAttnState,
    num_k_heads: usize,
    num_v_heads: usize,
    total_key: usize,
    total_value: usize,
    qkv_dim: usize,
    hidden_dim: usize,
    key_dim: usize,
    value_dim: usize,
    inv_scale: f32,
    k_heads_per_v: usize,
    use_gpu: bool,
    gpu_wf: Option<&crate::metal_context::GpuWeightCtx>,
    ctx: Option<&crate::metal_context::MetalContext>,
) {
    let prefix = format!("model.layers.{}.linear_attn", layer_idx);

    // CPU: attention projections
    let mut qkv = vec![0.0f32; qkv_dim];
    let mut z = vec![0.0f32; total_value];
    let mut beta = vec![0.0f32; num_v_heads];
    let mut alpha = vec![0.0f32; num_v_heads];

    if let (Some(qw), Some(qs), Some(qb)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_qkv.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_qkv.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_qkv.biases", prefix)),
    ) { dequant_matvec_4bit(qw, qs, qb, normed, &mut qkv, qkv_dim, hidden_dim, GROUP_SIZE); }
    if let (Some(zw), Some(zs), Some(zb)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_z.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_z.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_z.biases", prefix)),
    ) { dequant_matvec_4bit(zw, zs, zb, normed, &mut z, total_value, hidden_dim, GROUP_SIZE); }
    if let (Some(bw), Some(bs), Some(bb)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_b.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_b.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_b.biases", prefix)),
    ) { dequant_matvec_4bit(bw, bs, bb, normed, &mut beta, num_v_heads, hidden_dim, GROUP_SIZE); }
    if let (Some(aw), Some(ass), Some(ab)) = (
        wf.get_tensor_u32(&format!("{}.in_proj_a.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_a.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.in_proj_a.biases", prefix)),
    ) { dequant_matvec_4bit(aw, ass, ab, normed, &mut alpha, num_v_heads, hidden_dim, GROUP_SIZE); }

    // Conv1d step (CPU)
    let mut conv_out = vec![0.0f32; qkv_dim];
    if let Some(conv_w) = wf.get_tensor_u16(&format!("{}.conv1d.weight", prefix)) {
        conv1d_step(&state.conv_state, &qkv, conv_w, &mut conv_out, qkv_dim, CONV_KERNEL_SIZE);
    } else {
        conv_out.copy_from_slice(&qkv);
    }
    let state_off = (CONV_KERNEL_SIZE - 2) * qkv_dim;
    state.conv_state.copy_within(qkv_dim.., 0);
    state.conv_state[state_off..state_off + qkv_dim].copy_from_slice(&qkv);
    let lin_q = conv_out[..total_key].to_vec();
    let lin_k = conv_out[total_key..2 * total_key].to_vec();
    let lin_v = conv_out[2 * total_key..].to_vec();

    // Try non-fused GPU SSM (or CPU fallback)
    let gpu_compatible = key_dim == 128 && value_dim == 128 && use_gpu;
    let gpu_ssm_ok = gpu_compatible && ctx.is_some();
    let mut gated_out = vec![0.0f32; total_value];

    if gpu_ssm_ok {
        let c = ctx.unwrap();
        let ssm_size = num_v_heads * value_dim * key_dim;
        let ssm_gpu = state.ssm_state_gpu.get_or_insert_with(|| {
            crate::metal_context::metal_buf_shared(&c.device, ssm_size * 4)
        });
        unsafe { let dst = ssm_gpu.contents() as *mut f32; std::ptr::copy_nonoverlapping(state.ssm_state.as_ptr(), dst, ssm_size); }

        let q_gpu = crate::metal_context::metal_buf_shared(&c.device, total_key * 4);
        let k_gpu = crate::metal_context::metal_buf_shared(&c.device, total_key * 4);
        let v_gpu = crate::metal_context::metal_buf_shared(&c.device, total_value * 4);
        let z_gpu = crate::metal_context::metal_buf_shared(&c.device, total_value * 4);
        let alpha_gpu = crate::metal_context::metal_buf_shared(&c.device, num_v_heads * 4);
        let beta_gpu = crate::metal_context::metal_buf_shared(&c.device, num_v_heads * 4);
        let out_gpu = crate::metal_context::metal_buf_shared(&c.device, total_value * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(lin_q.as_ptr(), q_gpu.contents() as *mut f32, total_key);
            std::ptr::copy_nonoverlapping(lin_k.as_ptr(), k_gpu.contents() as *mut f32, total_key);
            std::ptr::copy_nonoverlapping(lin_v.as_ptr(), v_gpu.contents() as *mut f32, total_value);
            std::ptr::copy_nonoverlapping(z.as_ptr(), z_gpu.contents() as *mut f32, total_value);
            std::ptr::copy_nonoverlapping(alpha.as_ptr(), alpha_gpu.contents() as *mut f32, num_v_heads);
            std::ptr::copy_nonoverlapping(beta.as_ptr(), beta_gpu.contents() as *mut f32, num_v_heads);
        }
        let a_log_ptr = wf.get_tensor_ptr(&format!("{}.A_log", prefix));
        let dt_bias_ptr = wf.get_tensor_ptr(&format!("{}.dt_bias", prefix));
        let a_log_gpu = crate::metal_context::metal_buf_shared(&c.device, num_v_heads * 4);
        let dt_bias_gpu = crate::metal_context::metal_buf_shared(&c.device, num_v_heads * 2);
        if let Some(p) = a_log_ptr {
            unsafe { std::ptr::copy_nonoverlapping(p as *const f32, a_log_gpu.contents() as *mut f32, num_v_heads); }
        }
        if let Some(p) = dt_bias_ptr {
            unsafe { std::ptr::copy_nonoverlapping(p as *const u16, dt_bias_gpu.contents() as *mut u16, num_v_heads); }
        }
        let g_decay_gpu = crate::metal_context::metal_buf_shared(&c.device, num_v_heads * 4);
        let beta_gate_gpu = crate::metal_context::metal_buf_shared(&c.device, num_v_heads * 4);
        let gated_gpu2 = crate::metal_context::metal_buf_shared(&c.device, total_value * 4);
        let gnw_ptr = wf.get_tensor_u16(&format!("{}.norm.weight", prefix));
        let gnw_gpu = gnw_ptr.map(|gnw| {
            let buf = crate::metal_context::metal_buf_shared(&c.device, gnw.len() * 2);
            unsafe { std::ptr::copy_nonoverlapping(gnw.as_ptr(), buf.contents() as *mut u16, gnw.len()); }
            buf
        });

        // Single command buffer: all GPU kernels batched together
        let cmd_buf = c.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        crate::metal_kernels::encode_rms_norm_qk(c, &enc, &q_gpu, 0, &k_gpu, 0, num_k_heads as u32, key_dim as u32, inv_scale);
        crate::metal_kernels::encode_compute_decay_beta(c, &enc, &alpha_gpu, &beta_gpu, &a_log_gpu, 0, &dt_bias_gpu, 0, &g_decay_gpu, &beta_gate_gpu, num_v_heads as u32);
        crate::metal_kernels::encode_gated_delta_net_step(c, &enc, ssm_gpu, &q_gpu, 0, &k_gpu, 0, &v_gpu, 0, &g_decay_gpu, &beta_gate_gpu, &out_gpu, num_v_heads as u32, k_heads_per_v as u32, key_dim as u32, value_dim as u32);
        if let Some(ref gnw_buf) = gnw_gpu {
            crate::metal_kernels::encode_gated_rms_norm(c, &enc, &out_gpu, &z_gpu, gnw_buf, 0, &gated_gpu2, num_v_heads as u32, value_dim as u32);
        }
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        if gnw_gpu.is_some() {
            unsafe { std::ptr::copy_nonoverlapping(gated_gpu2.contents() as *const f32, gated_out.as_mut_ptr(), total_value); }
        } else {
            unsafe { std::ptr::copy_nonoverlapping(out_gpu.contents() as *const f32, gated_out.as_mut_ptr(), total_value); }
        }
        unsafe { std::ptr::copy_nonoverlapping(ssm_gpu.contents() as *const f32, state.ssm_state.as_mut_ptr(), ssm_size); }
    } else {
        // CPU SSM
        let mut q_normed = vec![0.0f32; total_key];
        let mut k_normed = vec![0.0f32; total_key];
        for h in 0..num_k_heads {
            let qh = &lin_q[h * key_dim..(h + 1) * key_dim];
            let qh_out = &mut q_normed[h * key_dim..(h + 1) * key_dim];
            rms_norm_bare(qh, qh_out, key_dim, 1e-6);
            let q_scale = inv_scale * inv_scale;
            for d in qh_out.iter_mut() { *d *= q_scale; }
        }
        for h in 0..num_k_heads {
            let kh = &lin_k[h * key_dim..(h + 1) * key_dim];
            let kh_out = &mut k_normed[h * key_dim..(h + 1) * key_dim];
            rms_norm_bare(kh, kh_out, key_dim, 1e-6);
            for d in kh_out.iter_mut() { *d *= inv_scale; }
        }


        let a_log = wf.get_tensor_f32(&format!("{}.A_log", prefix));
        let dt_bias = wf.get_tensor_u16(&format!("{}.dt_bias", prefix));

        let mut out_values = vec![0.0f32; total_value];

        for vh in 0..num_v_heads {
            let kh = vh / k_heads_per_v;
            let a_val = a_log.map_or(1.0, |al| al[vh]);
            let dt_b = dt_bias.map_or(0.0, |db| bf16_to_f32(db[vh]));
            let softplus_val = (1.0 + (alpha[vh] + dt_b).exp()).ln();
            let g_decay = (-a_val.exp() * softplus_val).exp();
            let beta_gate = sigmoid(beta[vh]);
            let s_off = vh * value_dim * key_dim;
            let ssm = &mut state.ssm_state[s_off..s_off + value_dim * key_dim];
            let v_h = &lin_v[vh * value_dim..(vh + 1) * value_dim];
            let k_h = &k_normed[kh * key_dim..(kh + 1) * key_dim];
            for vi in 0..value_dim {
                for ki in 0..key_dim { ssm[vi * key_dim + ki] *= g_decay; }
            }
            for vi in 0..value_dim {
                let mut kv_mem = 0.0f32;
                for ki in 0..key_dim { kv_mem += ssm[vi * key_dim + ki] * k_h[ki]; }
                let delta = (v_h[vi] - kv_mem) * beta_gate;
                for ki in 0..key_dim { ssm[vi * key_dim + ki] += k_h[ki] * delta; }
            }
            let q_h = &q_normed[kh * key_dim..(kh + 1) * key_dim];
            let o_h = &mut out_values[vh * value_dim..(vh + 1) * value_dim];
            for vi in 0..value_dim {
                let mut sum = 0.0f32;
                for ki in 0..key_dim { sum += ssm[vi * key_dim + ki] * q_h[ki]; }
                o_h[vi] = sum;
            }
        }

        // RMSNormGated
        if let Some(gnw) = wf.get_tensor_u16(&format!("{}.norm.weight", prefix)) {
            for vh in 0..num_v_heads {
                let oh = &out_values[vh * value_dim..(vh + 1) * value_dim];
                let zh = &z[vh * value_dim..(vh + 1) * value_dim];
                let gh = &mut gated_out[vh * value_dim..(vh + 1) * value_dim];
                rms_norm_gated(oh, zh, gnw, gh, value_dim, RMS_NORM_EPS);
            }
        } else {
            gated_out.copy_from_slice(&out_values);
        }
    }

    // Output projection
    let mut attn_out = vec![0.0f32; hidden_dim];
    if use_gpu {
        let gw = gpu_wf.unwrap();
        let c = ctx.unwrap();
        gw.matvec(wf, c, &format!("{}.out_proj", prefix), &gated_out, &mut attn_out, hidden_dim, total_value);
    } else if let (Some(ow), Some(os), Some(ob)) = (
        wf.get_tensor_u32(&format!("{}.out_proj.weight", prefix)),
        wf.get_tensor_u16(&format!("{}.out_proj.scales", prefix)),
        wf.get_tensor_u16(&format!("{}.out_proj.biases", prefix)),
    ) {
        dequant_matvec_4bit(ow, os, ob, &gated_out, &mut attn_out, hidden_dim, total_value, GROUP_SIZE);
    }

    // Residual add
    for i in 0..hidden_dim {
        hidden[i] = residual[i] + attn_out[i];
    }

}

// ─── EngineCPU ────────────────────────────────────────────────────────────

/// CPU-only engine: no GPU resources required.
pub struct EngineCPU<'a> {
    pub model: &'a Model,
}

impl<'a> Engine for EngineCPU<'a> {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        mut check_signal: SignalCheckFn<'_>,
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

        let kv_dim = self.model.config.num_kv_heads * self.model.config.head_dim;
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
                        cpu_linear_attention(
                            &self.model.wf, layer, &mut hidden, &normed, &residual, state,
                            num_k_heads, num_v_heads, total_key, total_value, qkv_dim,
                            hd, key_dim, value_dim, inv_scale, k_heads_per_v,
                            false, None, None,
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

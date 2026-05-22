/// EngineGPU: GPU-accelerated per-token processing using process_token_inner.
use crate::engine::{Cache, Engine, Model};
use crate::engine_common::{
    embed_lookup, final_norm, gpu_lm_head, process_token_inner,
    ExecCtxGpu, SignalCheckFn,
};
use crate::metal_context::{ExpertBuffer, GpuWeightCtx, MetalContext};

// ─── EngineGPU ────────────────────────────────────────────────────────────

pub struct EngineGPU<'a> {
    pub model: &'a Model,
    pub ctx: &'a MetalContext,
    pub gpu_wf: &'a GpuWeightCtx,
    pub expert_gpu_buffer: Option<&'a mut ExpertBuffer>,
}

impl<'a> EngineGPU<'a> {
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

impl<'a> Engine for EngineGPU<'a> {
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        mut check_signal: SignalCheckFn<'_>,
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
                false,
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

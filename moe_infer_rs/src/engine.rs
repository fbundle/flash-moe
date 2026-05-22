/// Model, Cache, Engine trait — the public API surface.
use std::os::fd::{IntoRawFd, RawFd};
use std::path::PathBuf;

use metal::Buffer;

use crate::constants::{
    CONV_KERNEL_SIZE, FULL_ATTN_INTERVAL, MAX_SEQ,
};
use crate::math::SignalCheckFn;
use crate::model_config::{load_model_config, ModelConfig};
use crate::model_weights::WeightFile;

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

// ─── Full attention cache ───────────────────────────────────────────────────

pub struct FullAttnCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub len: usize,
}

impl FullAttnCache {
    pub fn new(max_seq: usize, kv_dim: usize) -> Self {
        FullAttnCache {
            k_cache: vec![0.0f32; max_seq * kv_dim],
            v_cache: vec![0.0f32; max_seq * kv_dim],
            len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

// ─── Linear attention state ─────────────────────────────────────────────────

pub struct LinearAttnState {
    pub conv_state: Vec<f32>,
    pub ssm_state: Vec<f32>,
    pub ssm_state_gpu: Option<Buffer>,
}

impl LinearAttnState {
    pub fn new(num_v_heads: usize, key_dim: usize, value_dim: usize, qkv_dim: usize) -> Self {
        LinearAttnState {
            conv_state: vec![0.0f32; (CONV_KERNEL_SIZE - 1) * qkv_dim],
            ssm_state: vec![0.0f32; num_v_heads * value_dim * key_dim],
            ssm_state_gpu: None,
        }
    }
}

// ─── Engine trait ───────────────────────────────────────────────────────────

pub trait Engine {
    /// Process `input_ids` through all layers. Returns logits [n, vocab_size].
    fn forward(
        &mut self,
        input_ids: &[i64],
        cache: &mut Cache,
        check_signal: SignalCheckFn<'_>,
    ) -> Result<Vec<f32>, String>;
}

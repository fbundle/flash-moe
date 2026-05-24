use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::constants::{CONV_KERNEL_SIZE, FULL_ATTN_INTERVAL, MAX_SEQ};
use crate::model::config::ModelConfig;

// ─── Per-layer state ────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub enum State {
    Full(FullState),
    Linear(LinearState),
}

impl State {
    pub fn as_full(&self) -> &FullState {
        match self { State::Full(s) => s, _ => panic!("expected Full state") }
    }
    pub fn as_full_mut(&mut self) -> &mut FullState {
        match self { State::Full(s) => s, _ => panic!("expected Full state") }
    }
    pub fn as_linear(&self) -> &LinearState {
        match self { State::Linear(s) => s, _ => panic!("expected Linear state") }
    }
    pub fn as_linear_mut(&mut self) -> &mut LinearState {
        match self { State::Linear(s) => s, _ => panic!("expected Linear state") }
    }
}

// ─── Cache — a list of per-layer state + sequence position ──────────────────

#[derive(Serialize, Deserialize)]
pub struct Cache {
    pub pos: usize,
    pub states: Vec<State>,
}

impl Cache {
    pub fn new(config: &ModelConfig) -> Self {
        let num_layers = config.get_usize("num_layers").unwrap();
        let num_kv_heads = config.get_usize("num_kv_heads").unwrap();
        let head_dim = config.get_usize("head_dim").unwrap();
        let kv_dim = num_kv_heads * head_dim;
        let lnum_v_heads = config.get_usize("linear_num_v_heads").unwrap();
        let ltotal_key = config.get_usize("linear_total_key").unwrap();
        let lnum_k_heads = config.get_usize("linear_num_k_heads").unwrap();
        let ltotal_value = config.get_usize("linear_total_value").unwrap();
        let lconv_dim = config.get_usize("linear_conv_dim").unwrap();

        let mut states = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            if (layer + 1) % FULL_ATTN_INTERVAL == 0 {
                states.push(State::Full(FullState::new(MAX_SEQ, kv_dim)));
            } else {
                states.push(State::Linear(LinearState::new(
                    lnum_v_heads,
                    ltotal_key / lnum_k_heads,
                    ltotal_value / lnum_v_heads,
                    lconv_dim,
                )));
            }
        }
        Cache { pos: 0, states }
    }

    /// Accessors panicking if the layer has the wrong attention type.
    pub fn full(&self, layer: usize) -> &FullState { self.states[layer].as_full() }
    pub fn full_mut(&mut self, layer: usize) -> &mut FullState { self.states[layer].as_full_mut() }
    pub fn lin(&self, layer: usize) -> &LinearState { self.states[layer].as_linear() }
    pub fn lin_mut(&mut self, layer: usize) -> &mut LinearState { self.states[layer].as_linear_mut() }

    pub fn reset(&mut self) {
        self.set_pos(0);
        for s in &mut self.states {
            if let State::Linear(ls) = s {
                ls.conv_state.fill(0.0);
                ls.ssm_state.fill(0.0);
            }
        }
    }

    /// Set position and sync all full-attention layer lengths.
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
        for s in &mut self.states {
            if let State::Full(kv) = s {
                kv.len = pos;
            }
        }
    }

    /// Serialize cache to a JSON file (for persistence across restarts).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Deserialize cache from a JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let cache: Cache = serde_json::from_reader(reader)?;
        Ok(cache)
    }
}

// ─── Full-attention state ───────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct FullState {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub len: usize,
}

impl FullState {
    pub fn new(max_seq: usize, kv_dim: usize) -> Self {
        FullState {
            k_cache: vec![0.0f32; max_seq * kv_dim],
            v_cache: vec![0.0f32; max_seq * kv_dim],
            len: 0,
        }
    }
}

// ─── Linear-attention state ─────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct LinearState {
    pub conv_state: Vec<f32>,
    pub ssm_state: Vec<f32>,
    #[serde(skip, default = "default_ssm_gpu")]
    pub ssm_state_gpu: Option<metal::Buffer>,
}

fn default_ssm_gpu() -> Option<metal::Buffer> { None }

impl LinearState {
    pub fn new(num_v_heads: usize, key_dim: usize, value_dim: usize, qkv_dim: usize) -> Self {
        LinearState {
            conv_state: vec![0.0f32; (CONV_KERNEL_SIZE - 1) * qkv_dim],
            ssm_state: vec![0.0f32; num_v_heads * value_dim * key_dim],
            ssm_state_gpu: None,
        }
    }
}

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::constants::{CONV_KERNEL_SIZE, FULL_ATTN_INTERVAL, MAX_SEQ};
use crate::model::config::ModelConfig;

// ─── Per-layer state ────────────────────────────────────────────────────────

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

pub struct Cache {
    pub pos: usize,
    pub states: Vec<State>,
}

impl Cache {
    pub fn new(config: &ModelConfig) -> Self {
        let num_layers = config.get_usize("num_hidden_layers").unwrap();
        let num_kv_heads = config.get_usize("num_key_value_heads").unwrap();
        let head_dim = config.get_usize("head_dim").unwrap();
        let kv_dim = num_kv_heads * head_dim;
        let lnum_v_heads = config.get_usize("linear_num_value_heads").unwrap();
        let lnum_k_heads = config.get_usize("linear_num_key_heads").unwrap();
        let ltotal_key = lnum_k_heads * config.get_usize("linear_key_head_dim").unwrap();
        let ltotal_value = lnum_v_heads * config.get_usize("linear_value_head_dim").unwrap();
        let lconv_dim = ltotal_key * 2 + ltotal_value;

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

    /// Copy state vectors from another cache (requires matching layer count).
    pub fn copy_from(&mut self, other: &Cache) {
        self.pos = other.pos;
        for (s, o) in self.states.iter_mut().zip(other.states.iter()) {
            match (s, o) {
                (State::Full(s), State::Full(o)) => {
                    s.k_cache.copy_from_slice(&o.k_cache);
                    s.v_cache.copy_from_slice(&o.v_cache);
                    s.len = o.len;
                }
                (State::Linear(s), State::Linear(o)) => {
                    s.conv_state.copy_from_slice(&o.conv_state);
                    s.ssm_state.copy_from_slice(&o.ssm_state);
                }
                _ => panic!("cache layer type mismatch"),
            }
        }
    }

    // ── Binary save / load (same format as model_weights: flat .bin + .json manifest) ─

    /// Save cache as `bin_path` (flat binary) + `json_path` (manifest).
    pub fn save(&self, bin_path: &Path, json_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let mut manifest_tensors = serde_json::Map::new();
        let mut offset: u64 = 0;

        let file = File::create(bin_path)?;
        let mut writer = BufWriter::new(file);

        for (i, state) in self.states.iter().enumerate() {
            match state {
                State::Full(s) => {
                    write_f32_tensor(&mut writer, &mut manifest_tensors, &mut offset,
                        &format!("cache.layer_{}.k_cache", i), &s.k_cache,
                        &[MAX_SEQ, s.k_cache.len() / MAX_SEQ])?;
                    write_f32_tensor(&mut writer, &mut manifest_tensors, &mut offset,
                        &format!("cache.layer_{}.v_cache", i), &s.v_cache,
                        &[MAX_SEQ, s.v_cache.len() / MAX_SEQ])?;
                    write_u32_scalar(&mut writer, &mut manifest_tensors, &mut offset,
                        &format!("cache.layer_{}.len", i), s.len as u32)?;
                }
                State::Linear(s) => {
                    write_f32_tensor(&mut writer, &mut manifest_tensors, &mut offset,
                        &format!("cache.layer_{}.conv_state", i), &s.conv_state,
                        &[CONV_KERNEL_SIZE - 1, s.conv_state.len() / (CONV_KERNEL_SIZE - 1)])?;
                    write_f32_tensor(&mut writer, &mut manifest_tensors, &mut offset,
                        &format!("cache.layer_{}.ssm_state", i), &s.ssm_state,
                        &[s.ssm_state.len()])?;
                }
            }
        }

        // Write pos last so its offset is last (simpler backward compat).
        write_u32_scalar(&mut writer, &mut manifest_tensors, &mut offset,
            "cache.pos", self.pos as u32)?;

        writer.flush()?;

        let manifest = serde_json::json!({ "tensors": manifest_tensors });
        let json_str = serde_json::to_string_pretty(&manifest)?;
        fs::write(json_path, &json_str)?;

        Ok(())
    }

    /// Load cache from `bin_path` (flat binary) + `json_path` (manifest).
    pub fn load(bin_path: &Path, json_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json_str = fs::read_to_string(json_path)?;
        let manifest: serde_json::Value = serde_json::from_str(&json_str)?;
        let tensors = manifest["tensors"].as_object()
            .ok_or("No 'tensors' key in manifest")?;

        let mmap = unsafe { memmap2::Mmap::map(&File::open(bin_path)?)? };
        let data = mmap.as_ptr();

        let read_u32 = |name: &str| -> Result<u32, Box<dyn std::error::Error>> {
            let t = tensors.get(name).ok_or(format!("missing tensor: {}", name))?;
            let off = t["offset"].as_u64().ok_or("missing offset")? as usize;
            let ptr = unsafe { data.add(off) as *const u32 };
            Ok(unsafe { *ptr })
        };

        let read_f32_vec = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let t = tensors.get(name).ok_or(format!("missing tensor: {}", name))?;
            let off = t["offset"].as_u64().ok_or("missing offset")? as usize;
            let size = t["size"].as_u64().ok_or("missing size")? as usize;
            let count = size / 4;
            let ptr = unsafe { data.add(off) as *const f32 };
            let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
            Ok(slice.to_vec())
        };

        let pos = read_u32("cache.pos")? as usize;
        // Count layers by checking what's in the manifest.
        let mut layer_count = 0usize;
        while tensors.contains_key(&format!("cache.layer_{}.k_cache", layer_count))
           || tensors.contains_key(&format!("cache.layer_{}.conv_state", layer_count))
        {
            layer_count += 1;
        }

        let mut states = Vec::with_capacity(layer_count);
        for i in 0..layer_count {
            if tensors.contains_key(&format!("cache.layer_{}.k_cache", i)) {
                let k_cache = read_f32_vec(&format!("cache.layer_{}.k_cache", i))?;
                let v_cache = read_f32_vec(&format!("cache.layer_{}.v_cache", i))?;
                let len = read_u32(&format!("cache.layer_{}.len", i))? as usize;
                states.push(State::Full(FullState { k_cache, v_cache, len }));
            } else {
                let conv_state = read_f32_vec(&format!("cache.layer_{}.conv_state", i))?;
                let ssm_state = read_f32_vec(&format!("cache.layer_{}.ssm_state", i))?;
                states.push(State::Linear(LinearState {
                    conv_state,
                    ssm_state,
                    ssm_state_gpu: None,
                }));
            }
        }

        Ok(Cache { pos, states })
    }
}

// ─── Binary format helpers ─────────────────────────────────────────────────

fn write_f32_tensor(
    w: &mut BufWriter<File>, manifest: &mut serde_json::Map<String, serde_json::Value>,
    offset: &mut u64, name: &str, data: &[f32], shape: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    w.write_all(bytes)?;
    let size = bytes.len() as u64;
    manifest.insert(name.into(), serde_json::json!({
        "offset": *offset,
        "size": size,
        "shape": shape,
        "dtype": "float32",
    }));
    *offset += size;
    Ok(())
}

fn write_u32_scalar(
    w: &mut BufWriter<File>, manifest: &mut serde_json::Map<String, serde_json::Value>,
    offset: &mut u64, name: &str, val: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = val.to_ne_bytes();
    w.write_all(&bytes)?;
    manifest.insert(name.into(), serde_json::json!({
        "offset": *offset,
        "size": 4,
        "shape": [],
        "dtype": "uint32",
    }));
    *offset += 4;
    Ok(())
}

// ─── Full-attention state ───────────────────────────────────────────────────

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

pub struct LinearState {
    pub conv_state: Vec<f32>,
    pub ssm_state: Vec<f32>,
    #[allow(dead_code)]
    pub ssm_state_gpu: Option<metal::Buffer>,
}

impl LinearState {
    pub fn new(num_v_heads: usize, key_dim: usize, value_dim: usize, qkv_dim: usize) -> Self {
        LinearState {
            conv_state: vec![0.0f32; (CONV_KERNEL_SIZE - 1) * qkv_dim],
            ssm_state: vec![0.0f32; num_v_heads * value_dim * key_dim],
            ssm_state_gpu: None,
        }
    }
}

use std::os::fd::{IntoRawFd, RawFd};
use std::path::PathBuf;

use crate::model_config::{load_model_config, ModelConfig};
use crate::model_weights::WeightFile;

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

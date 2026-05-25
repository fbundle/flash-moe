#[path = "model/config.rs"]
pub mod config;
#[path = "model/expert.rs"]
pub mod expert;
#[path = "model/weights.rs"]
pub mod weights;

use std::io;
use std::os::fd::IntoRawFd;
use std::path::PathBuf;

use crate::error::MoEError;
use self::config::{load_model_config, ModelConfig};
use self::expert::ExpertFile;
use self::weights::WeightFile;

pub use self::expert::ExpertFile as ExpertFileType;

pub struct Model {
    pub config: ModelConfig,
    pub weight_file: WeightFile,
    pub expert_files: Vec<ExpertFile>,
}

impl Model {
    pub fn load(model_path: &str) -> Result<Self, MoEError> {
        let dir = PathBuf::from(model_path);
        if !dir.exists() {
            return Err(MoEError::Config(format!("not found: {}", dir.display())));
        }
        let config = load_model_config(&dir)
            .map_err(|e| MoEError::Config(format!("config: {}", e)))?;
        let wf = WeightFile::open(
            &dir.join("model_weights.bin"),
            &dir.join("model_weights.json"),
        )
        .map_err(|e| MoEError::Config(format!("weights: {}", e)))?;

        let packed_dir = dir.join("packed_experts");
        let lz4_dir = dir.join("packed_experts_lz4");
        let hd = config.get_usize("hidden_size").unwrap();
        let mi = config.get_usize("moe_intermediate_size").unwrap();
        let expert_size = config::expert_size_4bit(hd, mi, 64);
        let num_layers = config.get_usize("num_hidden_layers").unwrap();

        let mut expert_files = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let lz4_path = lz4_dir.join(format!("layer_{:02}.bin", layer));
            if lz4_path.exists() {
                use std::io::Read;
                // Header: [u32 num_experts][u32 off_0]...[u32 off_{N-1}][u32 total_size]
                // Guard against corrupted files (expert count must match config).
                let mut f = std::fs::File::open(&lz4_path)
                    .map_err(|e| MoEError::Io(io::Error::new(io::ErrorKind::Other,
                        format!("lz4 expert {}: {}", layer, e))))?;
                let mut hdr4 = [0u8; 4];
                f.read_exact(&mut hdr4)
                    .map_err(|e| MoEError::Io(io::Error::new(io::ErrorKind::Other,
                        format!("lz4 hdr {}: {}", layer, e))))?;
                let n = u32::from_le_bytes(hdr4) as usize;
                let num_experts = config.get_usize("num_experts").unwrap();
                if n != num_experts {
                    return Err(MoEError::Io(io::Error::new(io::ErrorKind::InvalidData,
                        format!("lz4 expert {}: header says {} experts, config says {} — file may be corrupted",
                            layer, n, num_experts))));
                }
                let off_len = n + 1;
                let mut off = vec![0u32; off_len];
                let mut off_buf = vec![0u8; off_len * 4];
                f.read_exact(&mut off_buf)
                    .map_err(|e| MoEError::Io(io::Error::new(io::ErrorKind::Other,
                        format!("lz4 off {}: {}", layer, e))))?;
                for i in 0..off_len {
                    off[i] = u32::from_le_bytes([
                        off_buf[i*4], off_buf[i*4+1], off_buf[i*4+2], off_buf[i*4+3]
                    ]);
                }
                let fd = f.into_raw_fd();
                expert_files.push(ExpertFile::Lz4 { fd, offsets: off, expert_size });
            } else {
                let f = std::fs::File::open(packed_dir.join(format!("layer_{:02}.bin", layer)))
                    .map_err(|e| MoEError::Io(io::Error::new(io::ErrorKind::Other,
                        format!("expert {}: {}", layer, e))))?;
                expert_files.push(ExpertFile::Raw { fd: f.into_raw_fd(), expert_size });
            }
        }

        let lz4_count = expert_files.iter().filter(|e| matches!(e, ExpertFile::Lz4 { .. })).count();
        eprintln!(
            "[model] {} layers hidden={} experts={} lz4_layers={}",
            num_layers, hd,
            config.get_usize("num_experts").unwrap_or(0), lz4_count
        );
        Ok(Model { config, weight_file: wf, expert_files })
    }
}

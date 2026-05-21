/// Non-expert weight loading from model_weights.bin (mmap) + model_weights.json (manifest).
///
/// Port of WeightFile / TensorManifest from infer.m:388-569.
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

use memmap2::Mmap;

use crate::error::MoEError;

/// Describes one tensor in the weight file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub ndim: usize,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Mmap'd weight file + tensor manifest.
///
/// The mmap lives for the lifetime of this struct. Tensor data is accessed
/// via `get_tensor_ptr()` which returns a pointer into the mapped region.
pub struct WeightFile {
    _mmap: Mmap,
    data_ptr: *const u8,
    pub size: usize,
    tensors: HashMap<String, TensorInfo>,
}

// SAFETY: WeightFile holds a read-only mmap, so it's safe to Send/Sync.
unsafe impl Send for WeightFile {}
unsafe impl Sync for WeightFile {}

impl WeightFile {
    /// Open a weight file: mmap the binary blob + parse the JSON manifest.
    pub fn open(bin_path: &Path, json_path: &Path) -> Result<Self, MoEError> {
        // mmap the binary file
        let file = fs::File::open(bin_path)
            .map_err(|e| {
                MoEError::Io(io::Error::new(
                    e.kind(),
                    format!("Cannot open {}: {}", bin_path.display(), e),
                ))
            })?;

        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                MoEError::Io(io::Error::new(
                    io::ErrorKind::Other,
                    format!("mmap failed: {}", e),
                ))
            })?
        };

        let size = mmap.len();
        let data_ptr = mmap.as_ptr();

        // Parse manifest JSON
        let json_str = fs::read_to_string(json_path).map_err(|e| {
            MoEError::Io(io::Error::new(
                e.kind(),
                format!("Cannot read {}: {}", json_path.display(), e),
            ))
        })?;

        let manifest: serde_json::Value =
            serde_json::from_str(&json_str).map_err(|e| {
                MoEError::Config(format!(
                    "JSON parse error in {}: {}",
                    json_path.display(),
                    e
                ))
            })?;

        let tensors_obj = manifest["tensors"].as_object().ok_or_else(|| {
            MoEError::Config("No 'tensors' key in manifest".into())
        })?;

        let mut tensor_map: HashMap<String, TensorInfo> = HashMap::new();
        for (name, info) in tensors_obj {
            let t = TensorInfo {
                name: name.clone(),
                offset: info["offset"].as_u64().unwrap_or(0),
                size: info["size"].as_u64().unwrap_or(0),
                ndim: info["shape"]
                    .as_array()
                    .map(|a| a.len())
                    .unwrap_or(0),
                shape: info["shape"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|v| v.as_u64().unwrap_or(0) as usize)
                            .collect()
                    })
                    .unwrap_or_default(),
                dtype: info["dtype"].as_str().unwrap_or("").to_string(),
            };
            tensor_map.insert(name.clone(), t);
        }

        eprintln!(
            "[weights] mmap'd {:.2} GB from {} ({} tensors)",
            size as f64 / 1e9,
            bin_path.display(),
            tensor_map.len()
        );

        Ok(WeightFile {
            _mmap: mmap,
            data_ptr,
            size,
            tensors: tensor_map,
        })
    }

    /// Get raw pointer to tensor data (valid while WeightFile lives).
    #[inline]
    pub fn get_tensor_ptr(&self, name: &str) -> Option<*const u8> {
        self.tensors
            .get(name)
            .map(|t| unsafe { self.data_ptr.add(t.offset as usize) })
    }

    /// Get tensor info (metadata).
    #[inline]
    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Get a tensor as a slice of u32 (for packed 4-bit weights).
    #[inline]
    pub fn get_tensor_u32(&self, name: &str) -> Option<&[u32]> {
        let t = self.tensors.get(name)?;
        let ptr = unsafe { self.data_ptr.add(t.offset as usize) } as *const u32;
        let len = t.size as usize / 4;
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Get a tensor as a slice of u16 (for scales/biases/norm weights).
    #[inline]
    pub fn get_tensor_u16(&self, name: &str) -> Option<&[u16]> {
        let t = self.tensors.get(name)?;
        let ptr = unsafe { self.data_ptr.add(t.offset as usize) } as *const u16;
        let len = t.size as usize / 2;
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Get a tensor as a slice of f32.
    #[inline]
    pub fn get_tensor_f32(&self, name: &str) -> Option<&[f32]> {
        let t = self.tensors.get(name)?;
        let ptr = unsafe { self.data_ptr.add(t.offset as usize) } as *const f32;
        let len = t.size as usize / 4;
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Raw base pointer to the mmap'd data (for GPU buffer wrapping).
    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.data_ptr
    }

    /// Byte offset of a tensor within the mmap (for GPU buffer dispatch).
    #[inline]
    pub fn tensor_offset(&self, name: &str) -> Option<usize> {
        self.tensors.get(name).map(|t| t.offset as usize)
    }
}

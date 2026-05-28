// Safe tensors format: header parsing, reading, and dtype conversion.
//
// This module handles reading tensors from HuggingFace safetensors shards,
// parsing the JSON header, and converting raw bytes to f32 for any of the
// common storage formats (F32, F16, BF16).

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::dtype::bf16_to_f32;

/// Metadata for one tensor within a safetensors shard.
pub struct TensorMeta {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub data_offsets: [u64; 2],
}

/// Parsed header of a safetensors shard file.
pub struct ShardHeader {
    pub data_start: u64,
    pub tensors: HashMap<String, TensorMeta>,
}

/// Parse a safetensors file header, returning the shard's tensor index.
pub fn parse_safetensors(path: &Path) -> Result<ShardHeader, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hdr_len_buf = [0u8; 8];
    f.read_exact(&mut hdr_len_buf).map_err(|e| e.to_string())?;
    let hdr_len = u64::from_le_bytes(hdr_len_buf) as usize;

    let mut hdr_json = vec![0u8; hdr_len];
    f.read_exact(&mut hdr_json).map_err(|e| e.to_string())?;
    let root: serde_json::Value =
        serde_json::from_slice(&hdr_json).map_err(|e| e.to_string())?;

    let mut tensors = HashMap::new();
    if let Some(obj) = root.as_object() {
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = meta["dtype"].as_str().unwrap_or("").to_string();
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
                .unwrap_or_default();
            let off = meta["data_offsets"].as_array().map_or([0, 0], |a| {
                [a[0].as_u64().unwrap_or(0), a[1].as_u64().unwrap_or(0)]
            });
            tensors.insert(name.clone(), TensorMeta { shape, dtype, data_offsets: off });
        }
    }
    Ok(ShardHeader {
        data_start: 8 + hdr_len as u64,
        tensors,
    })
}

/// Read the raw bytes of a tensor from a safetensors shard.
pub fn read_tensor_bytes(
    shard_path: &Path,
    header: &ShardHeader,
    name: &str,
) -> Result<Vec<u8>, String> {
    let meta = header
        .tensors
        .get(name)
        .ok_or_else(|| format!("tensor '{}' not found in {}", name, shard_path.display()))?;
    let off = meta.data_offsets;
    let len = (off[1] - off[0]) as usize;

    let mut f = fs::File::open(shard_path).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start(header.data_start + off[0]))
        .map_err(|e| e.to_string())?;
    let mut data = vec![0u8; len];
    f.read_exact(&mut data).map_err(|e| e.to_string())?;
    Ok(data)
}

/// Convert an IEEE 754 half-precision (F16) value to f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: normalize
            let mut m2 = mant;
            while m2 < 0x400 {
                m2 <<= 1;
            }
            let shift = m2.leading_zeros() as i32 - 21;
            let actual_mant = (m2 & 0x3FF) << 13;
            f32::from_bits((sign << 31) | ((((-14 - 10 + 127) + shift) as u32) << 23) | actual_mant)
        }
    } else if exp == 0x1F {
        f32::from_bits((sign << 31) | 0x7F80_0000 | (mant << 13))
    } else {
        let e = (exp as i32) - 15 + 127;
        f32::from_bits((sign << 31) | ((e as u32) << 23) | (mant << 13))
    }
}

/// Convert raw safetensors bytes to f32, dispatching on the storage dtype.
pub fn bytes_to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F32" => {
            let n = data.len() / 4;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let b = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
                out.push(f32::from_le_bytes(b));
            }
            out
        }
        "F16" => {
            let n = data.len() / 2;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let lo = data[i * 2] as u16;
                let hi = data[i * 2 + 1] as u16;
                out.push(f16_to_f32(lo | (hi << 8)));
            }
            out
        }
        _ => {
            // BF16 (default for Qwen HF models)
            let n = data.len() / 2;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let lo = data[i * 2] as u16;
                let hi = data[i * 2 + 1] as u16;
                out.push(bf16_to_f32(lo | (hi << 8)));
            }
            out
        }
    }
}

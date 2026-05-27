// HuggingFace Hub downloader — one file at a time.
//
// Thin wrapper around the `hf-hub` crate.  Downloads model files individually
// through HuggingFace's official API with caching, resume, and retries built-in.
//
// Typical workflow:
//   1. `HfApi::new("hub/")` — point to local hub cache
//   2. `api.download("config.json")` — fetch one file
//   3. `api.weight_map()` — download + parse index
//   4. `api.download_shards_for(&tensor_names)` — only needed shards

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder};

/// File paths exposed by the weight map (index.json).
#[derive(Clone, Debug)]
pub struct ModelFiles {
    /// Path to the local index file.
    pub index_path: PathBuf,
    /// Maps tensor names → safetensors shard filenames.
    pub weight_map: HashMap<String, String>,
    /// Index metadata (e.g. total_size).
    pub metadata: HashMap<String, serde_json::Value>,
}

/// HuggingFace Hub downloader wrapping `hf-hub`.
pub struct HfApi {
    api: Api,
}

impl HfApi {
    /// Create a new API pointing at `hub_path` (e.g. `/path/to/hub/`).
    ///
    /// `repo_id` is like `"Qwen/Qwen3.6-35B-A3B"`.
    /// Files are cached under `{hub_path}/models--{org}--{name}/`.
    pub fn new(hub_path: &Path, _repo_id: &str) -> Result<Self, String> {
        let api = ApiBuilder::new()
            .with_cache_dir(hub_path.to_path_buf())
            .with_progress(false)
            .build()
            .map_err(|e| format!("hf-hub init: {e}"))?;

        Ok(HfApi { api })
    }

    /// The underlying `hf-hub` API client, for lower-level access.
    pub fn inner(&self) -> &Api {
        &self.api
    }

    /// Fetch a single file from the HF Hub, downloading if not cached.
    ///
    /// Returns the local path of the file.
    pub fn download(&self, repo_id: &str, filename: &str) -> Result<PathBuf, String> {
        self.api
            .model(repo_id.to_string())
            .download(filename)
            .map_err(|e| format!("download {filename}: {e}"))
    }

    /// Like `download` but returns the cached path without downloading if present.
    pub fn get(&self, repo_id: &str, filename: &str) -> Result<PathBuf, String> {
        self.api
            .model(repo_id.to_string())
            .get(filename)
            .map_err(|e| format!("get {filename}: {e}"))
    }

    /// List all files in the model repo via the Hub API.
    pub fn list_files(&self, repo_id: &str) -> Result<Vec<String>, String> {
        let info = self
            .api
            .model(repo_id.to_string())
            .info()
            .map_err(|e| format!("list files: {e}"))?;

        Ok(info.siblings.into_iter().map(|s| s.rfilename).collect())
    }

    /// Download only the small metadata files: config.json, index.json, etc.
    pub fn download_meta(&self, repo_id: &str) -> Result<ModelFiles, String> {
        // Download config & index first
        self.download(repo_id, "config.json")?;
        let index_path = self.download(repo_id, "model.safetensors.index.json")?;

        // Parse index
        let json_str =
            std::fs::read_to_string(&index_path).map_err(|e| format!("read index: {e}"))?;
        let root: serde_json::Value =
            serde_json::from_str(&json_str).map_err(|e| format!("parse index: {e}"))?;

        let metadata = root
            .get("metadata")
            .and_then(|v| v.as_object())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        let weight_map: HashMap<String, String> = root
            .get("weight_map")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                    .collect()
            })
            .ok_or_else(|| "no weight_map in index".to_string())?;

        Ok(ModelFiles {
            index_path,
            weight_map,
            metadata,
        })
    }

    /// Download a specific set of safetensors shards determined by the weight map.
    ///
    /// Only downloads shards that are needed to cover the given tensor names.
    /// Each file is downloaded individually (one at a time).
    pub fn download_shards_for(
        &self,
        repo_id: &str,
        weight_map: &HashMap<String, String>,
        tensor_names: &[String],
    ) -> Result<Vec<PathBuf>, String> {
        // Collect unique shards
        let mut needed: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for name in tensor_names {
            if let Some(shard) = weight_map.get(name) {
                needed.insert(shard.clone());
            }
        }

        let mut paths = Vec::new();
        for shard in &needed {
            let path = self.download(repo_id, shard)?;
            paths.push(path);
        }

        Ok(paths)
    }

    /// Check if a model's index and all its shards are cached locally.
    pub fn is_cached(&self, repo_id: &str) -> bool {
        let f = match self.list_files(repo_id) {
            Ok(files) => files,
            Err(_) => return false,
        };
        f.iter()
            .filter(|name| name.ends_with(".safetensors"))
            .all(|name| {
                let _ = self.api.model(repo_id.to_string()).get(name);
                // get() returns Ok if cached, Err if not — but Err includes
                // "not found" case vs "network" case.  Just check the file exists.
                let cache = self.api.model(repo_id.to_string());
                let local = cache.get(name);
                local.is_ok()
            })
    }
}

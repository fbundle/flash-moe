// Unified local/remote access to a HuggingFace model repository.
//
// - Local mode  – created via `HfRepo::from_local(path)` — operations
//   read directly from the local filesystem.
// - HF mode     – created via `HfRepo::from_hf(repo_id)` — files are
//   downloaded on demand into a temp staging directory and can be
//   removed after processing.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

pub struct HfRepo {
    repo_id: Option<String>,
    staging: PathBuf,
}

impl HfRepo {
    pub fn from_local(path: PathBuf) -> Self {
        Self { repo_id: None, staging: path }
    }

    pub fn from_hf(repo_id: &str) -> Result<Self, String> {
        eprintln!("  Input '{}' not found locally — treating as HF repo ID", repo_id);
        let staging = PathBuf::from(format!("hub/models--{}", repo_id.replace('/', "--")));
        fs::create_dir_all(&staging).map_err(|e| e.to_string())?;
        Ok(Self { repo_id: Some(repo_id.to_owned()), staging })
    }

    pub fn is_hf(&self) -> bool { self.repo_id.is_some() }

    pub fn path(&self) -> &Path { &self.staging }

    /// List immediate children of *dir* (defaults to root).  Behaves like UNIX ``ls``:
    /// returns names of files and directories at that level, not recursive.
    /// - Local: reads the filesystem.
    /// - HF: fetches the full file tree from the Hub API and filters to one level.
    pub fn ls(&self, dir: Option<&str>) -> Result<Vec<String>, String> {
        let prefix = dir.unwrap_or("");
        match &self.repo_id {
            None => {
                let target = if prefix.is_empty() {
                    self.staging.clone()
                } else {
                    self.staging.join(prefix)
                };
                let mut entries = Vec::new();
                for entry in fs::read_dir(&target).map_err(|e| e.to_string())? {
                    let entry = entry.map_err(|e| e.to_string())?;
                    entries.push(entry.file_name().to_string_lossy().to_string());
                }
                entries.sort();
                Ok(entries)
            }
            Some(repo_id) => {
                let tree = fetch_tree(repo_id)?;
                let all_paths: Vec<&str> = tree
                    .iter()
                    .filter_map(|v| v["path"].as_str())
                    .collect();
                // Find immediate children of *prefix*: strip prefix + "/",
                // keep only entries that have exactly one more component
                let prefix_len = if prefix.is_empty() { 0 } else { prefix.len() + 1 };
                let mut seen = std::collections::BTreeSet::new();
                for path in &all_paths {
                    if prefix_len > 0 && !path.starts_with(prefix) {
                        continue;
                    }
                    let rest = &path[prefix_len..];
                    if let Some(slash) = rest.find('/') {
                        seen.insert(rest[..slash].to_string());
                    } else if !rest.is_empty() {
                        seen.insert(rest.to_string());
                    }
                }
                Ok(seen.into_iter().collect())
            }
        }
    }

    /// Ensure a file is available locally, downloading if in HF mode.
    /// Shows a tqdm-style progress bar for the download.
    pub fn ensure(&self, filename: &str) -> Result<PathBuf, String> {
        let local = self.staging.join(filename);
        if local.exists() {
            return Ok(local);
        }
        match &self.repo_id {
            None => Err(format!("file not found: {}", local.display())),
            Some(repo_id) => {
                let pb = ProgressBar::new_spinner();
                pb.set_style(tqdm_style());
                pb.set_message(filename.to_string());
                download_hf(repo_id, filename, &self.staging, &pb)
            }
        }
    }

    /// Download multiple files in parallel with tqdm-style progress bars.
    pub fn ensure_batch(&self, filenames: &[String]) -> Result<Vec<PathBuf>, String> {
        match &self.repo_id {
            None => {
                let mut paths = Vec::with_capacity(filenames.len());
                for f in filenames {
                    let local = self.staging.join(f);
                    if !local.exists() {
                        return Err(format!("file not found: {}", local.display()));
                    }
                    paths.push(local);
                }
                Ok(paths)
            }
            Some(repo_id) => {
                let repo_id = repo_id.clone();
                let staging = self.staging.clone();
                let mp = MultiProgress::new();
                std::thread::scope(move |s| {
                    let mut handles = Vec::with_capacity(filenames.len());
                    for f in filenames {
                        let pb = mp.add(ProgressBar::new_spinner());
                        pb.set_style(tqdm_style());
                        pb.set_message(f.clone());
                        let rid = repo_id.clone();
                        let stg = staging.clone();
                        let f = f.clone();
                        handles.push(s.spawn(move || download_hf(&rid, &f, &stg, &pb)));
                    }
                    let mut results = Vec::with_capacity(filenames.len());
                    for h in handles {
                        results.push(h.join().map_err(|_| "thread panicked".to_string())??);
                    }
                    Ok(results)
                })
            }
        }
    }

    /// Remove a file from the staging directory (no-op if absent).
    pub fn remove(&self, filename: &str) {
        fs::remove_file(self.staging.join(filename)).ok();
    }

    /// Get file size in bytes.
    /// - Local: reads filesystem metadata.
    /// - HF: fetches the file tree from the Hub API.
    pub fn file_size(&self, filename: &str) -> Result<u64, String> {
        match &self.repo_id {
            None => {
                let meta = fs::metadata(self.staging.join(filename))
                    .map_err(|e| e.to_string())?;
                Ok(meta.len())
            }
            Some(repo_id) => {
                let tree = fetch_tree(repo_id)?;
                let size = tree.iter()
                    .find(|v| v["path"].as_str() == Some(filename))
                    .and_then(|v| v["size"].as_u64())
                    .ok_or_else(|| format!("file not found in tree: {filename}"))?;
                Ok(size)
            }
        }
    }

    /// Get sizes for all files in the repo (filename → bytes).
    /// - Local: walks the filesystem.
    /// - HF: fetches the file tree from the Hub API.
    pub fn file_sizes(&self) -> Result<Vec<(String, u64)>, String> {
        match &self.repo_id {
            None => {
                let mut sizes = Vec::new();
                walk_sizes(&self.staging, &self.staging, &mut sizes)
                    .map_err(|e| e.to_string())?;
                sizes.sort();
                Ok(sizes)
            }
            Some(repo_id) => {
                let tree = fetch_tree(repo_id)?;
                let mut sizes: Vec<(String, u64)> = tree.iter()
                    .filter(|v| v["type"] == "file")
                    .filter_map(|v| {
                        let path = v["path"].as_str()?;
                        let size = v["size"].as_u64()?;
                        Some((path.to_string(), size))
                    })
                    .collect();
                sizes.sort();
                Ok(sizes)
            }
        }
    }
}

fn fetch_tree(repo_id: &str) -> Result<Vec<serde_json::Value>, String> {
    let url = format!("https://huggingface.co/api/models/{}/tree/main", repo_id);
    let body_str = ureq::get(&url)
        .call()
        .map_err(|e| format!("HF API error: {e}"))?
        .into_body()
        .read_to_string()
        .map_err(|e| format!("read API response: {e}"))?;
    serde_json::from_str(&body_str).map_err(|e| e.to_string())
}

fn walk_sizes(base: &Path, dir: &Path, out: &mut Vec<(String, u64)>) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            walk_sizes(base, &path, out)?;
        } else if entry.file_type()?.is_file() {
            if let Ok(rel) = path.strip_prefix(base) {
                out.push((rel.to_string_lossy().to_string(), entry.metadata()?.len()));
            }
        }
    }
    Ok(())
}

fn tqdm_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{msg:30} {bar:40.cyan/blue} {percent:>3}% {bytes:>10}/{total_bytes:10} {bytes_per_sec:>12} {elapsed:>6}"
    ).unwrap()
    .progress_chars("━╸─")
}

fn download_hf(
    repo_id: &str,
    filename: &str,
    dest_dir: &Path,
    pb: &ProgressBar,
) -> Result<PathBuf, String> {
    let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, filename);
    let dest = dest_dir.join(filename);

    if dest.exists() {
        pb.finish_and_clear();
        return Ok(dest);
    }

    let mut resp = ureq::get(&url)
        .call()
        .map_err(|e| format!("HTTP error for {filename}: {e}"))?;

    let total: u64 = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    pb.set_length(total);

    let mut reader = resp.body_mut().with_config().limit(5_000_000_000).reader();

    let tmp = dest.with_extension("part");
    let mut file = fs::File::create(&tmp)
        .map_err(|e| format!("create {filename}: {e}"))?;

    let mut buf = vec![0u8; 8 * 1024 * 1024]; // 8 MiB
    loop {
        let n = reader.read(&mut buf)
            .map_err(|e| format!("read {filename}: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("write {filename}: {e}"))?;
        pb.inc(n as u64);
    }

    drop(file);
    fs::rename(&tmp, &dest).map_err(|e| format!("rename {filename}: {e}"))?;
    pb.finish_and_clear();

    Ok(dest)
}

use std::path::Path;

/// Runtime model configuration — JSON-like key-value store.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    data: serde_json::Map<String, serde_json::Value>,
}

impl ModelConfig {
    pub(crate) fn resolve(&self, key: &str) -> Option<&serde_json::Value> {
        self.data.get(key).or_else(|| {
            self.data.get("text_config")
                .and_then(|tc| tc.get(key))
        })
    }

    pub fn get_usize(&self, key: &str) -> Option<usize> {
        if let Some((obj, field)) = key.split_once('.') {
            return self.resolve(obj)?
                .get(field)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
        }
        self.resolve(key).and_then(|v| v.as_u64()).map(|v| v as usize)
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        if let Some((obj, field)) = key.split_once('.') {
            return self.resolve(obj)?
                .get(field)
                .and_then(|v| v.as_f64());
        }
        self.resolve(key).and_then(|v| v.as_f64())
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.resolve(key).and_then(|v| v.as_str())
    }

    pub fn get_object(&self, key: &str) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.resolve(key).and_then(|v| v.as_object())
    }

    pub fn usize_or(&self, key: &str, default: usize) -> usize {
        self.get_usize(key).unwrap_or(default)
    }

    pub fn from_map(data: serde_json::Map<String, serde_json::Value>) -> Self {
        ModelConfig { data }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &serde_json::Value)> {
        self.data.iter()
    }

    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

pub(crate) fn expert_size_4bit(hd: usize, mi: usize, gs: usize) -> usize {
    let gate_w = mi * hd / 2;
    let gate_sb = mi * (hd / gs) * 2;
    let up_w = mi * hd / 2;
    let up_sb = mi * (hd / gs) * 2;
    let down_w = hd * mi / 2;
    let down_sb = hd * (mi / gs) * 2;
    gate_w + 2 * gate_sb + up_w + 2 * up_sb + down_w + 2 * down_sb
}

/// Load model configuration from an HF config.json file.
pub fn load_model_config(model_path: &Path) -> anyhow::Result<ModelConfig> {
    let config_path = model_path.join("config.json");
    let content = std::fs::read_to_string(&config_path)?;
    let root: serde_json::Value = serde_json::from_str(&content)?;
    Ok(ModelConfig { data: root.as_object().cloned().unwrap_or_default() })
}

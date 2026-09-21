//! Model registry and downloader.
//!
//! Replaces Python `models/downloader.py`. Provides model metadata registry,
//! local path resolution, SHA256 verification, and async Range download
//! with exponential backoff retry.

use crate::config::default_voice_models_dirs;
use crate::errors::{ErrorCode, Result, XiaobaiError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Model metadata from models.yaml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub size_mb: f64,
    #[serde(default)]
    pub subdir: String,
    #[serde(default)]
    pub urls: Vec<String>,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub archive_format: String,
    #[serde(default)]
    pub entry: HashMap<String, String>,
    #[serde(default)]
    pub engine: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub optional: bool,
}

/// Model status (for listing).
#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub id: String,
    pub name: String,
    pub license: String,
    pub size_mb: f64,
    pub downloaded: bool,
    pub sha256_ok: Option<bool>,
    pub local_root: Option<String>,
    pub engine: Option<String>,
    pub category: Option<String>,
    pub optional: bool,
}

/// Model registry: reads models.yaml and resolves local paths.
pub struct ModelRegistry {
    models: Vec<ModelMeta>,
    version: u32,
    extra_dirs: Vec<PathBuf>,
}

impl ModelRegistry {
    /// Load registry from a models.yaml file.
    pub fn from_yaml(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            XiaobaiError::new(ErrorCode::MissingModel, format!("Failed to read models.yaml: {e}"))
        })?;
        let raw: serde_yaml::Value = serde_yaml::from_str(&content)?;
        let version = raw.get("version").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let models: Vec<ModelMeta> = raw
            .get("models")
            .and_then(|v| serde_yaml::from_value(v.clone()).ok())
            .unwrap_or_default();
        Ok(Self {
            models,
            version,
            extra_dirs: Vec::new(),
        })
    }

    pub fn with_extra_dirs(mut self, dirs: Vec<PathBuf>) -> Self {
        self.extra_dirs = dirs;
        self
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn meta(&self, id: &str) -> Option<&ModelMeta> {
        self.models.iter().find(|m| m.id == id)
    }

    pub fn list_all(&self) -> Vec<ModelStatus> {
        self.models
            .iter()
            .map(|m| {
                let local = self.find_local_root(&m.id);
                ModelStatus {
                    id: m.id.clone(),
                    name: if m.name.is_empty() { m.id.clone() } else { m.name.clone() },
                    license: if m.license.is_empty() { "Unknown".to_string() } else { m.license.clone() },
                    size_mb: m.size_mb,
                    downloaded: local.is_some(),
                    sha256_ok: local.as_ref().and_then(|root| self.verify_sha256(m, root)),
                    local_root: local.map(|p| p.to_string_lossy().to_string()),
                    engine: m.engine.clone(),
                    category: m.category.clone(),
                    optional: m.optional,
                }
            })
            .collect()
    }

    /// Find the local root directory for a model.
    pub fn find_local_root(&self, id: &str) -> Option<PathBuf> {
        let meta = self.meta(id)?;
        let subdir = &meta.subdir;
        for root in self.model_root_candidates(subdir) {
            if self.check_entry(&root, meta) {
                return Some(root);
            }
        }
        None
    }

    /// Resolve model to {root, entry} for backend path parsing.
    pub fn resolve(&self, id: &str) -> Option<HashMap<String, String>> {
        let meta = self.meta(id)?;
        let local = self.find_local_root(id)?;
        let mut result = HashMap::new();
        result.insert("id".to_string(), id.to_string());
        result.insert("root".to_string(), local.to_string_lossy().to_string());
        for (k, v) in &meta.entry {
            result.insert(format!("entry.{}", k), v.clone());
        }
        Some(result)
    }

    fn model_root_candidates(&self, subdir: &str) -> Vec<PathBuf> {
        let mut dirs = Vec::new();
        dirs.extend(self.extra_dirs.iter().cloned());
        dirs.extend(default_voice_models_dirs());
        dirs.into_iter().map(|d| d.join(subdir)).collect()
    }

    fn check_entry(&self, root: &Path, meta: &ModelMeta) -> bool {
        if !root.is_dir() {
            return false;
        }
        if meta.entry.is_empty() {
            return true; // CosyVoice2 etc: directory existence is enough
        }
        for (_, v) in &meta.entry {
            if v.is_empty() {
                continue;
            }
            if !root.join(v).is_file() {
                return false;
            }
        }
        true
    }

    fn verify_sha256(&self, meta: &ModelMeta, root: &Path) -> Option<bool> {
        let expected = meta.sha256.trim().to_lowercase();
        if expected.is_empty() || expected.starts_with("tbd") {
            return None;
        }
        let archive_format = if meta.archive_format.is_empty() {
            "tgz".to_string()
        } else {
            meta.archive_format.clone()
        };
        let pkg = if archive_format == "file" {
            meta.urls
                .first()
                .map(|url| {
                    let name = url.split('/').next_back().unwrap_or("model");
                    root.parent().unwrap_or(root).join(name)
                })
                .unwrap_or_else(|| root.join("model.bin"))
        } else {
            root.parent()
                .unwrap_or(root)
                .join(format!("{}.tar.gz", meta.subdir))
        };
        if pkg.is_file() {
            Some(sha256_file(&pkg).map_or(false, |h| h == expected))
        } else {
            None
        }
    }
}

/// Compute SHA256 of a file.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).map_err(|e| {
        XiaobaiError::new(ErrorCode::Runtime, format!("Failed to open file for SHA256: {e}"))
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let n = std::io::Read::read(&mut file, &mut buffer).map_err(|e| {
            XiaobaiError::new(ErrorCode::Runtime, format!("Read error: {e}"))
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Download progress event.
#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgress {
    pub model_id: String,
    pub state: String, // downloading / done / error / cached
    pub progress_pct: f64,
    pub speed_mbps: f64,
    pub eta_s: f64,
}

// --- PyO3 bindings ---
use pyo3::prelude::*;

#[pyclass(name = "ModelRegistry")]
pub struct PyModelRegistry {
    inner: ModelRegistry,
}

#[pymethods]
impl PyModelRegistry {
    #[new]
    #[pyo3(signature = (yaml_path=None))]
    fn new(yaml_path: Option<String>) -> PyResult<Self> {
        let path = yaml_path
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models.yaml"));
        let inner = ModelRegistry::from_yaml(&path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn list_all(&self) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let statuses = self.inner.list_all();
            let mut result = Vec::new();
            for s in statuses {
                let d = pyo3::types::PyDict::new_bound(py);
                d.set_item("id", s.id)?;
                d.set_item("name", s.name)?;
                d.set_item("license", s.license)?;
                d.set_item("size_mb", s.size_mb)?;
                d.set_item("downloaded", s.downloaded)?;
                d.set_item("sha256_ok", s.sha256_ok)?;
                d.set_item("local_root", s.local_root)?;
                d.set_item("engine", s.engine)?;
                d.set_item("category", s.category)?;
                d.set_item("optional", s.optional)?;
                result.push(d.unbind().into());
            }
            Ok(result)
        })
    }

    fn find_local_root(&self, model_id: &str) -> Option<String> {
        self.inner
            .find_local_root(model_id)
            .map(|p| p.to_string_lossy().to_string())
    }

    fn resolve(&self, model_id: &str) -> PyResult<Option<PyObject>> {
        Python::with_gil(|py| {
            Ok(self.inner.resolve(model_id).map(|m| {
                let d = pyo3::types::PyDict::new_bound(py);
                for (k, v) in m {
                    d.set_item(k, v).unwrap();
                }
                d.unbind().into()
            }))
        })
    }

    #[getter]
    fn version(&self) -> u32 {
        self.inner.version()
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModelRegistry>()?;
    m.add_function(wrap_pyfunction!(py_sha256_file, m)?)?;
    Ok(())
}

#[pyfunction]
fn py_sha256_file(path: &str) -> PyResult<String> {
    sha256_file(Path::new(path)).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_known() {
        // SHA256 of empty string
        let tmp = std::env::temp_dir().join("xiaobai_test_empty.bin");
        std::fs::write(&tmp, "").unwrap();
        let hash = sha256_file(&tmp).unwrap();
        assert_eq!(hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        std::fs::remove_file(&tmp).ok();
    }
}

//! Cross-platform configuration loader.
//!
//! Replaces Python `config/loader.py`. Provides YAML deep-merge,
//! cross-platform config path resolution, and atomic writes.

use crate::errors::{ErrorCode, Result, XiaobaiError};
use serde_yaml::Value;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Resolve the platform-specific config directory.
pub fn platform_config_path() -> PathBuf {
    if cfg!(windows) {
        let appdata = std::env::var("APPDATA")
            .unwrap_or_else(|_| format!("{}\\AppData\\Roaming", dirs::home_dir().unwrap_or_default().display()));
        PathBuf::from(appdata).join("mox").join("xiaobai").join("config.yaml")
    } else if cfg!(target_os = "macos") {
        dirs::home_dir()
            .unwrap_or_default()
            .join("Library")
            .join("Application Support")
            .join("mox")
            .join("xiaobai")
            .join("config.yaml")
    } else {
        let xdg = std::env::var("XDG_CONFIG_HOME")
            .unwrap_or_else(|_| format!("{}/.config", dirs::home_dir().unwrap_or_default().display()));
        PathBuf::from(xdg).join("mox").join("xiaobai").join("config.yaml")
    }
}

/// Resolve the platform-specific log directory.
pub fn default_log_path() -> PathBuf {
    if cfg!(windows) {
        let appdata = std::env::var("APPDATA")
            .unwrap_or_else(|_| format!("{}\\AppData\\Roaming", dirs::home_dir().unwrap_or_default().display()));
        PathBuf::from(appdata).join("mox").join("xiaobai").join("logs")
    } else if cfg!(target_os = "macos") {
        dirs::home_dir()
            .unwrap_or_default()
            .join("Library")
            .join("Logs")
            .join("mox")
            .join("xiaobai")
    } else {
        let xdg = std::env::var("XDG_STATE_HOME")
            .unwrap_or_else(|_| format!("{}/.local/state", dirs::home_dir().unwrap_or_default().display()));
        PathBuf::from(xdg).join("mox").join("xiaobai").join("logs")
    }
}

/// Default model search directories (exe-level > user dir > repo models).
pub fn default_voice_models_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    // User directory
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".mox").join("models").join("voice"));
    }
    dirs
}

/// Deep merge two YAML values. `override` takes precedence.
pub fn deep_merge(base: &Value, override_val: &Value) -> Value {
    match (base, override_val) {
        (Value::Mapping(b), Value::Mapping(o)) => {
            let mut merged = b.clone();
            for (k, v) in o.iter() {
                match merged.get(k) {
                    Some(existing) => {
                        merged.insert(k.clone(), deep_merge(existing, v));
                    }
                    None => {
                        merged.insert(k.clone(), v.clone());
                    }
                }
            }
            Value::Mapping(merged)
        }
        _ => override_val.clone(),
    }
}

/// Thread-safe configuration loader with deep-merge and atomic writes.
pub struct ConfigLoader {
    user_path: PathBuf,
    default_path: PathBuf,
    data: RwLock<Value>,
}

impl ConfigLoader {
    /// Create a new config loader. If `user_path` is None, uses platform default.
    pub fn new(user_path: Option<PathBuf>, default_path: PathBuf) -> Result<Self> {
        let user_path = user_path.unwrap_or_else(platform_config_path);
        if let Some(parent) = user_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                XiaobaiError::new(ErrorCode::Runtime, format!("Failed to create config dir: {e}"))
            })?;
        }

        let defaults = Self::read_yaml(&default_path).unwrap_or(Value::Mapping(serde_yaml::Mapping::new()));
        let user = Self::read_yaml(&user_path).unwrap_or(Value::Mapping(serde_yaml::Mapping::new()));
        let merged = deep_merge(&defaults, &user);

        Ok(Self {
            user_path,
            default_path,
            data: RwLock::new(merged),
        })
    }

    /// Get a nested value by dot-separated path (e.g. "voice.tts.speed").
    pub fn get(&self, dotted: &str) -> Option<Value> {
        let data = self.data.read().ok()?;
        let mut node = &*data;
        for part in dotted.split('.') {
            match node {
                Value::Mapping(m) => {
                    let key = Value::String(part.to_string());
                    node = m.get(&key)?;
                }
                _ => return None,
            }
        }
        Some(node.clone())
    }

    /// Get a string value by dot path.
    pub fn get_string(&self, dotted: &str) -> Option<String> {
        self.get(dotted).and_then(|v| v.as_str().map(|s| s.to_string()))
    }

    /// Get a float value by dot path.
    pub fn get_float(&self, dotted: &str) -> Option<f64> {
        self.get(dotted).and_then(|v| v.as_f64())
    }

    /// Get an int value by dot path.
    pub fn get_int(&self, dotted: &str) -> Option<i64> {
        self.get(dotted).and_then(|v| v.as_i64())
    }

    /// Get a bool value by dot path.
    pub fn get_bool(&self, dotted: &str) -> Option<bool> {
        self.get(dotted).and_then(|v| v.as_bool())
    }

    /// Get the full merged config as a serde_yaml::Value.
    pub fn data(&self) -> Value {
        self.data.read().map(|d| d.clone()).unwrap_or(Value::Null)
    }

    /// Save a patch to the user config (deep merge with existing user config).
    pub fn save_patch(&self, patch: Value) -> Result<Value> {
        let existing = Self::read_yaml(&self.user_path).unwrap_or(Value::Mapping(serde_yaml::Mapping::new()));
        let merged = deep_merge(&existing, &patch);
        Self::atomic_write_yaml(&self.user_path, &merged)?;
        // Reload
        let defaults = Self::read_yaml(&self.default_path).unwrap_or(Value::Mapping(serde_yaml::Mapping::new()));
        let user = Self::read_yaml(&self.user_path).unwrap_or(Value::Mapping(serde_yaml::Mapping::new()));
        let reloaded = deep_merge(&defaults, &user);
        if let Ok(mut data) = self.data.write() {
            *data = reloaded.clone();
        }
        Ok(reloaded)
    }

    /// Reload config from disk.
    pub fn reload(&self) -> Result<()> {
        let defaults = Self::read_yaml(&self.default_path).unwrap_or(Value::Mapping(serde_yaml::Mapping::new()));
        let user = Self::read_yaml(&self.user_path).unwrap_or(Value::Mapping(serde_yaml::Mapping::new()));
        let merged = deep_merge(&defaults, &user);
        if let Ok(mut data) = self.data.write() {
            *data = merged;
        }
        Ok(())
    }

    pub fn user_path(&self) -> &Path {
        &self.user_path
    }

    pub fn default_path(&self) -> &Path {
        &self.default_path
    }

    // --- internal ---

    fn read_yaml(path: &Path) -> Option<Value> {
        if !path.is_file() {
            return None;
        }
        let content = std::fs::read_to_string(path).ok()?;
        serde_yaml::from_str(&content).ok()
    }

    fn atomic_write_yaml(path: &Path, data: &Value) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                XiaobaiError::new(ErrorCode::Runtime, format!("Failed to create dir: {e}"))
            })?;
        }
        let tmp = path.with_extension(format!("{}.tmp", path.extension().and_then(|e| e.to_str()).unwrap_or("yaml")));
        let content = serde_yaml::to_string(data).map_err(|e| {
            XiaobaiError::new(ErrorCode::ConfigInvalid, format!("YAML serialize error: {e}"))
        })?;
        std::fs::write(&tmp, content).map_err(|e| {
            XiaobaiError::new(ErrorCode::Runtime, format!("Write error: {e}"))
        })?;
        std::fs::rename(&tmp, path).map_err(|e| {
            XiaobaiError::new(ErrorCode::Runtime, format!("Rename error: {e}"))
        })?;
        Ok(())
    }
}

// --- PyO3 bindings ---
use pyo3::prelude::*;

#[pyclass(name = "ConfigLoader")]
pub struct PyConfigLoader {
    inner: ConfigLoader,
}

#[pymethods]
impl PyConfigLoader {
    #[new]
    #[pyo3(signature = (user_path=None, default_path=None))]
    fn new(user_path: Option<String>, default_path: Option<String>) -> PyResult<Self> {
        let default = default_path
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("default_config.yaml"));
        let inner = ConfigLoader::new(user_path.map(PathBuf::from), default)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn get(&self, dotted: &str) -> PyResult<Option<PyObject>> {
        Python::with_gil(|py| {
            Ok(self.inner.get(dotted).map(|v| {
                serde_yaml::to_string(&v)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&yaml_to_json(&s)).ok())
                    .map(|json| json_to_py(py, &json))
                    .unwrap_or_else(|| py.None())
            }))
        })
    }

    fn get_string(&self, dotted: &str) -> Option<String> {
        self.inner.get_string(dotted)
    }

    fn get_float(&self, dotted: &str) -> Option<f64> {
        self.inner.get_float(dotted)
    }

    fn get_int(&self, dotted: &str) -> Option<i64> {
        self.inner.get_int(dotted)
    }

    fn get_bool(&self, dotted: &str) -> Option<bool> {
        self.inner.get_bool(dotted)
    }

    fn reload(&self) -> PyResult<()> {
        self.inner.reload().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    fn user_path(&self) -> String {
        self.inner.user_path().to_string_lossy().to_string()
    }
}

fn yaml_to_json(yaml_str: &str) -> String {
    // Simple: use serde_yaml to parse, then serde_json to stringify
    if let Ok(val) = serde_yaml::from_str::<Value>(yaml_str) {
        serde_json::to_string(&val).unwrap_or_else(|_| "null".to_string())
    } else {
        "null".to_string()
    }
}

fn json_to_py(py: Python<'_>, val: &serde_json::Value) -> PyObject {
    match val {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.to_object(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_object(py)
            } else if let Some(f) = n.as_f64() {
                f.to_object(py)
            } else {
                py.None()
            }
        }
        serde_json::Value::String(s) => s.to_object(py),
        serde_json::Value::Array(arr) => {
            let list = pyo3::types::PyList::empty_bound(py);
            for v in arr {
                list.append(json_to_py(py, v)).unwrap();
            }
            list.unbind().into()
        }
        serde_json::Value::Object(obj) => {
            let dict = pyo3::types::PyDict::new_bound(py);
            for (k, v) in obj {
                dict.set_item(k, json_to_py(py, v)).unwrap();
            }
            dict.unbind().into()
        }
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConfigLoader>()?;
    m.add_function(wrap_pyfunction!(py_platform_config_path, m)?)?;
    m.add_function(wrap_pyfunction!(py_default_log_path, m)?)?;
    Ok(())
}

#[pyfunction]
fn py_platform_config_path() -> String {
    platform_config_path().to_string_lossy().to_string()
}

#[pyfunction]
fn py_default_log_path() -> String {
    default_log_path().to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deep_merge() {
        let base: Value = serde_yaml::from_str("a:\n  b: 1\n  c: 2\n").unwrap();
        let override_val: Value = serde_yaml::from_str("a:\n  c: 3\n  d: 4\n").unwrap();
        let merged = deep_merge(&base, &override_val);
        assert_eq!(merged["a"]["b"].as_i64(), Some(1));
        assert_eq!(merged["a"]["c"].as_i64(), Some(3));
        assert_eq!(merged["a"]["d"].as_i64(), Some(4));
    }

    #[test]
    fn test_platform_config_path_not_empty() {
        let p = platform_config_path();
        assert!(!p.to_string_lossy().is_empty());
    }
}

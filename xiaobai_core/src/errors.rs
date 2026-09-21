//! Unified error types matching Python's ErrorCode enum.
//!
//! Provides a Rust-native error hierarchy with PyO3 conversion,
//! replacing the Python `errors.py` module for core logic.

use pyo3::exceptions::PyException;
use pyo3::{create_exception, PyErr};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Error codes aligned 1:1 with Python `ErrorCode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    Ok,
    MissingDep,
    MissingModel,
    DllLoadFail,
    GpuOom,
    LicenseGate,
    ConfigInvalid,
    Runtime,
    PermissionDenied,
    OperatorUnsupported,
    OperatorFailed,
    BridgeDisconnected,
    IntentAmbiguous,
    IntentUnknown,
    HotwordsFormat,
    HotwordsReinstantiateFail,
    Unknown,
}

impl ErrorCode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCode::Ok => "OK",
            ErrorCode::MissingDep => "MISSING_DEP",
            ErrorCode::MissingModel => "MISSING_MODEL",
            ErrorCode::DllLoadFail => "DLL_LOAD_FAIL",
            ErrorCode::GpuOom => "GPU_OOM",
            ErrorCode::LicenseGate => "LICENSE_GATE",
            ErrorCode::ConfigInvalid => "CONFIG_INVALID",
            ErrorCode::Runtime => "RUNTIME",
            ErrorCode::PermissionDenied => "PERMISSION_DENIED",
            ErrorCode::OperatorUnsupported => "OPERATOR_UNSUPPORTED",
            ErrorCode::OperatorFailed => "OPERATOR_FAILED",
            ErrorCode::BridgeDisconnected => "BRIDGE_DISCONNECTED",
            ErrorCode::IntentAmbiguous => "INTENT_AMBIGUOUS",
            ErrorCode::IntentUnknown => "INTENT_UNKNOWN",
            ErrorCode::HotwordsFormat => "HOTWORDS_FORMAT",
            ErrorCode::HotwordsReinstantiateFail => "HOTWORDS_REINSTANTIATE_FAIL",
            ErrorCode::Unknown => "UNKNOWN",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "OK" => ErrorCode::Ok,
            "MISSING_DEP" => ErrorCode::MissingDep,
            "MISSING_MODEL" => ErrorCode::MissingModel,
            "DLL_LOAD_FAIL" => ErrorCode::DllLoadFail,
            "GPU_OOM" => ErrorCode::GpuOom,
            "LICENSE_GATE" => ErrorCode::LicenseGate,
            "CONFIG_INVALID" => ErrorCode::ConfigInvalid,
            "RUNTIME" => ErrorCode::Runtime,
            "PERMISSION_DENIED" => ErrorCode::PermissionDenied,
            "OPERATOR_UNSUPPORTED" => ErrorCode::OperatorUnsupported,
            "OPERATOR_FAILED" => ErrorCode::OperatorFailed,
            "BRIDGE_DISCONNECTED" => ErrorCode::BridgeDisconnected,
            "INTENT_AMBIGUOUS" => ErrorCode::IntentAmbiguous,
            "INTENT_UNKNOWN" => ErrorCode::IntentUnknown,
            "HOTWORDS_FORMAT" => ErrorCode::HotwordsFormat,
            "HOTWORDS_REINSTANTIATE_FAIL" => ErrorCode::HotwordsReinstantiateFail,
            _ => ErrorCode::Unknown,
        }
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Core error type matching Python's `XiaobaiError`.
#[derive(Debug, Error)]
pub struct XiaobaiError {
    pub code: ErrorCode,
    pub message: String,
    pub cause: Option<String>,
    pub details: HashMap<String, String>,
}

impl XiaobaiError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            cause: None,
            details: HashMap::new(),
        }
    }

    pub fn with_cause(mut self, cause: impl Into<String>) -> Self {
        self.cause = Some(cause.into());
        self
    }

    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }

    pub fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("code".to_string(), self.code.as_str().to_string());
        m.insert("message".to_string(), self.message.clone());
        if let Some(c) = &self.cause {
            m.insert("cause".to_string(), c.clone());
        }
        for (k, v) in &self.details {
            m.insert(k.clone(), v.clone());
        }
        m
    }
}

impl std::fmt::Display for XiaobaiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

// --- PyO3 exception type ---
create_exception!(xiaobai_core, PyXiaobaiError, PyException);

impl From<XiaobaiError> for PyErr {
    fn from(e: XiaobaiError) -> Self {
        let msg = format!("[{}] {}", e.code, e.message);
        PyXiaobaiError::new_err(msg)
    }
}

impl From<anyhow::Error> for XiaobaiError {
    fn from(e: anyhow::Error) -> Self {
        XiaobaiError::new(ErrorCode::Runtime, format!("{e}"))
    }
}

impl From<std::io::Error> for XiaobaiError {
    fn from(e: std::io::Error) -> Self {
        XiaobaiError::new(ErrorCode::Runtime, format!("IO error: {e}"))
    }
}

impl From<serde_json::Error> for XiaobaiError {
    fn from(e: serde_json::Error) -> Self {
        XiaobaiError::new(ErrorCode::ConfigInvalid, format!("JSON error: {e}"))
    }
}

impl From<serde_yaml::Error> for XiaobaiError {
    fn from(e: serde_yaml::Error) -> Self {
        XiaobaiError::new(ErrorCode::ConfigInvalid, format!("YAML error: {e}"))
    }
}

impl From<regex::Error> for XiaobaiError {
    fn from(e: regex::Error) -> Self {
        XiaobaiError::new(ErrorCode::ConfigInvalid, format!("Regex error: {e}"))
    }
}

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, XiaobaiError>;

//! xiaobai_core — Rust core library for Xiaobai voice assistant.
//!
//! Provides high-performance DSP, config management, intent routing,
//! model registry, and RBAC-gated system operators via PyO3 bindings.
//!
//! Python usage:
//! ```python
//! from xiaobai_core import dsp, config, intent, models, operators
//! processed = dsp.process_tts_audio(samples, 22050, 16000, 1.03, -18.0, True)
//! ```

use pyo3::prelude::*;

pub mod config;
pub mod dsp;
pub mod errors;
pub mod intent;
pub mod models;
pub mod operators;

/// Version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[pymodule]
fn xiaobai_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Version
    m.add("__version__", VERSION)?;

    // Errors
    m.add(
        "XiaobaiError",
        m.py().get_type_bound::<errors::PyXiaobaiError>(),
    )?;

    // DSP submodule
    let dsp_mod = PyModule::new_bound(m.py(), "dsp")?;
    dsp::register_module(&dsp_mod)?;
    m.add_submodule(&dsp_mod)?;

    // Config submodule
    let config_mod = PyModule::new_bound(m.py(), "config")?;
    config::register_module(&config_mod)?;
    m.add_submodule(&config_mod)?;

    // Intent submodule
    let intent_mod = PyModule::new_bound(m.py(), "intent")?;
    intent::register_module(&intent_mod)?;
    m.add_submodule(&intent_mod)?;

    // Models submodule
    let models_mod = PyModule::new_bound(m.py(), "models")?;
    models::register_module(&models_mod)?;
    m.add_submodule(&models_mod)?;

    // Operators submodule
    let operators_mod = PyModule::new_bound(m.py(), "operators")?;
    operators::register_module(&operators_mod)?;
    m.add_submodule(&operators_mod)?;

    Ok(())
}

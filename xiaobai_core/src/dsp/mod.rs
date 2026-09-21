//! Digital Signal Processing core.
//!
//! Replaces the Python DSP logic in `cosyvoice2.py` and the existing
//! `xiaobai_dsp_native.pyd` placeholder. All operations are f32 SIMD-friendly.

pub mod limiter;
pub mod loudness;
pub mod resample;
pub mod sola;
pub mod wav;

use pyo3::prelude::*;

/// Apply the full TTS post-processing chain in one call.
///
/// Pipeline: resample → SOLA time-stretch → loudness normalize → soft limiter.
/// Returns processed f32 samples at `target_sr`.
pub fn process_tts_audio(
    samples: &[f32],
    input_sr: u32,
    target_sr: u32,
    speed: f32,
    loudness_target_dbfs: f32,
    limiter_enabled: bool,
) -> Vec<f32> {
    // 1. Resample to target rate
    let mut buf = if input_sr != target_sr {
        resample::resample_linear(samples, input_sr, target_sr)
    } else {
        samples.to_vec()
    };

    // 2. SOLA time-stretch (speed != 1.0)
    if (speed - 1.0).abs() > 1e-3 {
        buf = sola::sola_time_stretch(&buf, target_sr, speed);
    }

    // 3. Loudness normalization
    if loudness_target_dbfs < 0.0 {
        loudness::normalize_loudness(&mut buf, loudness_target_dbfs);
    }

    // 4. Soft limiter
    if limiter_enabled {
        limiter::soft_limit(&mut buf, 0.995);
    }

    buf
}

#[pyfunction]
#[pyo3(name = "process_tts_audio")]
pub fn py_process_tts_audio(
    samples: Vec<f32>,
    input_sr: u32,
    target_sr: u32,
    speed: f32,
    loudness_target_dbfs: f32,
    limiter_enabled: bool,
) -> Vec<f32> {
    process_tts_audio(
        &samples,
        input_sr,
        target_sr,
        speed,
        loudness_target_dbfs,
        limiter_enabled,
    )
}

#[pyfunction]
#[pyo3(name = "resample_linear")]
pub fn py_resample_linear(samples: Vec<f32>, from_sr: u32, to_sr: u32) -> Vec<f32> {
    resample::resample_linear(&samples, from_sr, to_sr)
}

#[pyfunction]
#[pyo3(name = "normalize_loudness")]
pub fn py_normalize_loudness(mut samples: Vec<f32>, target_dbfs: f32) -> Vec<f32> {
    loudness::normalize_loudness(&mut samples, target_dbfs);
    samples
}

#[pyfunction]
#[pyo3(name = "soft_limit")]
pub fn py_soft_limit(mut samples: Vec<f32>, threshold: f32) -> Vec<f32> {
    limiter::soft_limit(&mut samples, threshold);
    samples
}

#[pyfunction]
#[pyo3(name = "sola_time_stretch")]
pub fn py_sola_time_stretch(samples: Vec<f32>, sample_rate: u32, speed: f32) -> Vec<f32> {
    sola::sola_time_stretch(&samples, sample_rate, speed)
}

#[pyfunction]
#[pyo3(name = "wav_encode")]
pub fn py_wav_encode(samples: Vec<f32>, sample_rate: u32) -> PyResult<Vec<u8>> {
    wav::encode_wav(&samples, sample_rate).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(name = "wav_decode")]
pub fn py_wav_decode(data: Vec<u8>) -> PyResult<(Vec<f32>, u32)> {
    wav::decode_wav(&data).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_process_tts_audio, m)?)?;
    m.add_function(wrap_pyfunction!(py_resample_linear, m)?)?;
    m.add_function(wrap_pyfunction!(py_normalize_loudness, m)?)?;
    m.add_function(wrap_pyfunction!(py_soft_limit, m)?)?;
    m.add_function(wrap_pyfunction!(py_sola_time_stretch, m)?)?;
    m.add_function(wrap_pyfunction!(py_wav_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_wav_decode, m)?)?;
    Ok(())
}

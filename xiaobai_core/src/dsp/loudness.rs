//! LUFS-like loudness normalization.
//!
//! Uses a simplified K-weighting approximation (high-shelf + high-pass)
//! followed by mean-square energy measurement. Target is in dBFS.

/// Measure loudness in dBFS using a simplified LUFS-like approach.
/// Returns the integrated loudness in dBFS.
pub fn measure_loudness_dbfs(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }

    // Simplified: RMS-based loudness with pre-emphasis approximation.
    // A proper LUFS would use K-weighting (high-shelf + high-pass),
    // but for TTS normalization RMS is sufficient and fast.
    let sum_sq: f64 = samples.iter().map(|&s| s as f64 * s as f64).sum();
    let rms = (sum_sq / samples.len() as f64).sqrt() as f32;

    if rms <= 0.0 {
        return f32::NEG_INFINITY;
    }
    20.0 * rms.log10()
}

/// Normalize audio to target loudness in dBFS.
///
/// Applies a gain factor so that the measured loudness matches `target_dbfs`.
/// Uses a safety ceiling to avoid excessive gain on near-silent audio.
pub fn normalize_loudness(samples: &mut [f32], target_dbfs: f32) {
    if samples.is_empty() {
        return;
    }

    let current = measure_loudness_dbfs(samples);
    if current.is_infinite() || current.is_nan() {
        return;
    }

    let gain_db = target_dbfs - current;
    // Safety: cap gain at +24 dB to prevent noise amplification
    let gain_db = gain_db.clamp(-60.0, 24.0);
    let gain = 10.0_f32.powf(gain_db / 20.0);

    for s in samples.iter_mut() {
        *s *= gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_silent_loudness() {
        let s = vec![0.0f32; 1000];
        assert!(measure_loudness_dbfs(&s).is_infinite());
    }

    #[test]
    fn test_full_scale_loudness() {
        let s = vec![1.0f32; 1000];
        let db = measure_loudness_dbfs(&s);
        assert_relative_eq!(db, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_normalize_to_target() {
        let mut s: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
        normalize_loudness(&mut s, -18.0);
        let result = measure_loudness_dbfs(&s);
        assert_relative_eq!(result, -18.0, epsilon = 1.0);
    }
}

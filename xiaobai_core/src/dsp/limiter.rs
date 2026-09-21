//! Soft limiter using tanh saturation.
//!
//! Prevents peak clipping by applying smooth saturation above `threshold`.
//! Replaces Python limiter logic in cosyvoice2.py.

/// Apply soft limiting to audio samples.
///
/// Samples with absolute value below `threshold` pass through unchanged.
/// Samples above `threshold` are compressed using a tanh-based curve
/// that asymptotically approaches ±1.0.
pub fn soft_limit(samples: &mut [f32], threshold: f32) {
    let threshold = threshold.clamp(0.1, 1.0);
    let inv_threshold = 1.0 / threshold;

    for s in samples.iter_mut() {
        let abs_s = s.abs();
        if abs_s > threshold {
            // Normalize to threshold-relative, apply tanh, scale back
            let normalized = (abs_s - threshold) * inv_threshold;
            let compressed = threshold + (1.0 - threshold) * normalized.tanh();
            *s = s.signum() * compressed;
        }
    }
}

/// Find the peak absolute value in samples.
pub fn peak_amplitude(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |max, &s| max.max(s.abs()))
}

/// Check if any sample exceeds the given threshold.
pub fn has_clipping(samples: &[f32], threshold: f32) -> bool {
    samples.iter().any(|&s| s.abs() > threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_clipping_below_threshold() {
        let mut s = vec![0.1f32, -0.2, 0.3, -0.4];
        let original = s.clone();
        soft_limit(&mut s, 0.5);
        for (a, b) in s.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_clipping_above_threshold() {
        let mut s = vec![1.5f32, -1.5, 0.8];
        soft_limit(&mut s, 0.995);
        for &v in &s {
            assert!(v.abs() <= 1.0, "sample {} exceeds 1.0", v);
        }
    }

    #[test]
    fn test_peak() {
        let s = vec![0.1f32, -0.9, 0.5, -0.3];
        assert_eq!(peak_amplitude(&s), 0.9);
    }
}

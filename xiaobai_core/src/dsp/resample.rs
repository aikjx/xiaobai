//! Linear interpolation resampler. O(n), no extra deps.
//!
//! Replaces Python `_resample_linear` in cosyvoice2.py.

/// Resample audio from `from_sr` to `to_sr` using linear interpolation.
pub fn resample_linear(samples: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if from_sr == to_sr || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = from_sr as f64 / to_sr as f64;
    let out_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;

        if idx + 1 < samples.len() {
            out.push(samples[idx] * (1.0 - frac) + samples[idx + 1] * frac);
        } else {
            out.push(samples[idx.min(samples.len() - 1)]);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity() {
        let s = vec![0.0f32, 0.5, 1.0, -0.5];
        let out = resample_linear(&s, 16000, 16000);
        assert_eq!(out, s);
    }

    #[test]
    fn test_upsample_length() {
        let s = vec![0.0f32; 160];
        let out = resample_linear(&s, 16000, 22050);
        let expected = (160.0 * 22050.0 / 16000.0).ceil() as usize;
        assert_eq!(out.len(), expected);
    }

    #[test]
    fn test_downsample_preserves_dc() {
        let s = vec![0.8f32; 1000];
        let out = resample_linear(&s, 22050, 16000);
        for &v in &out {
            assert_relative_eq!(v, 0.8, epsilon = 1e-6);
        }
    }
}

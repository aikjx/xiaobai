//! SOLA-like time-domain pitch-synchronous overlap-add for speed control.
//!
//! Changes playback speed without altering pitch. Frame length 20ms,
//! overlap 10ms, matching the Python implementation in cosyvoice2.py.

/// Apply SOLA-like time stretching.
///
/// `speed > 1.0` makes audio faster (shorter), `speed < 1.0` slower (longer).
/// Pitch is preserved. Uses 20ms frames with 10ms overlap.
pub fn sola_time_stretch(samples: &[f32], sample_rate: u32, speed: f32) -> Vec<f32> {
    if samples.is_empty() || (speed - 1.0).abs() < 1e-3 {
        return samples.to_vec();
    }

    let sr = sample_rate as f32;
    let frame_ms = 20.0;
    let overlap_ms = 10.0;
    let frame_len = (sr * frame_ms / 1000.0) as usize;
    let overlap_len = (sr * overlap_ms / 1000.0) as usize;
    let hop_len = frame_len - overlap_len;

    if frame_len == 0 || hop_len == 0 {
        return samples.to_vec();
    }

    // Number of output frames
    let out_duration = samples.len() as f32 / speed;
    let num_frames = (out_duration / hop_len as f32).ceil() as usize;

    let mut output = Vec::with_capacity(out_duration as usize + frame_len);

    // Hann window for crossfade
    let window: Vec<f32> = (0..overlap_len)
        .map(|i| {
            let t = i as f32 / (overlap_len - 1) as f32;
            0.5 - 0.5 * (2.0 * std::f32::consts::PI * t).cos()
        })
        .collect();

    for frame_idx in 0..num_frames {
        // Source position: advance by speed * hop_len each frame
        let src_start = (frame_idx as f32 * hop_len as f32 * speed) as usize;
        if src_start >= samples.len() {
            break;
        }

        let src_end = (src_start + frame_len).min(samples.len());
        let frame = &samples[src_start..src_end];

        let out_start = frame_idx * hop_len;

        // Ensure output buffer is large enough
        while output.len() < out_start + frame.len() {
            output.push(0.0);
        }

        if frame_idx == 0 {
            // First frame: copy directly
            for (i, &s) in frame.iter().enumerate() {
                output[out_start + i] = s;
            }
        } else {
            // Overlap-add with crossfade
            let crossfade_len = overlap_len.min(frame.len());
            for i in 0..crossfade_len {
                let w = window[i];
                let existing = output[out_start + i];
                let incoming = frame[i];
                output[out_start + i] = existing * (1.0 - w) + incoming * w;
            }
            // Copy the rest of the frame
            for i in crossfade_len..frame.len() {
                output[out_start + i] = frame[i];
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_speed() {
        let s: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = sola_time_stretch(&s, 16000, 1.0);
        assert_eq!(out, s);
    }

    #[test]
    fn test_faster_is_shorter() {
        let s: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = sola_time_stretch(&s, 16000, 1.5);
        assert!(out.len() < s.len(), "faster should be shorter: {} vs {}", out.len(), s.len());
    }

    #[test]
    fn test_slower_is_longer() {
        let s: Vec<f32> = (0..10000).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = sola_time_stretch(&s, 16000, 0.7);
        assert!(out.len() > s.len(), "slower should be longer: {} vs {}", out.len(), s.len());
    }

    #[test]
    fn test_empty_input() {
        let out = sola_time_stretch(&[], 16000, 1.2);
        assert!(out.is_empty());
    }
}

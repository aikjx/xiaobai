//! Minimal WAV file I/O (16-bit PCM, mono).
//!
//! Replaces Python's `wave` / `soundfile` for simple TTS output encoding.
//! Supports reading and writing standard RIFF WAV files.

use crate::errors::{ErrorCode, Result, XiaobaiError};

const WAV_HEADER_SIZE: usize = 44;

/// Encode f32 mono samples to 16-bit PCM WAV bytes.
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = samples.len() as u32 * 2;
    let chunk_size = 36 + data_size;

    let mut buf = Vec::with_capacity(WAV_HEADER_SIZE + data_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&chunk_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // subchunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    // Convert f32 to i16
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let val = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&val.to_le_bytes());
    }

    Ok(buf)
}

/// Decode 16-bit PCM WAV bytes to f32 mono samples.
/// Returns (samples, sample_rate).
pub fn decode_wav(data: &[u8]) -> Result<(Vec<f32>, u32)> {
    if data.len() < WAV_HEADER_SIZE {
        return Err(XiaobaiError::new(
            ErrorCode::ConfigInvalid,
            format!("WAV data too short: {} bytes", data.len()),
        ));
    }

    // Verify RIFF header
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(XiaobaiError::new(
            ErrorCode::ConfigInvalid,
            "Not a valid RIFF WAV file",
        ));
    }

    let num_channels = u16::from_le_bytes([data[22], data[23]]);
    let sample_rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let bits_per_sample = u16::from_le_bytes([data[34], data[35]]);

    if bits_per_sample != 16 {
        return Err(XiaobaiError::new(
            ErrorCode::ConfigInvalid,
            format!("Only 16-bit PCM supported, got {} bits", bits_per_sample),
        ));
    }

    // Find data chunk (skip fmt chunk, handle extra chunks)
    let mut offset = 12;
    let mut samples_data: &[u8] = &[];

    while offset + 8 <= data.len() {
        let chunk_id = &data[offset..offset + 4];
        let chunk_size = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]) as usize;

        if chunk_id == b"data" {
            let data_start = offset + 8;
            let data_end = (data_start + chunk_size).min(data.len());
            samples_data = &data[data_start..data_end];
            break;
        }

        offset += 8 + chunk_size + (chunk_size % 2); // padding
    }

    if samples_data.is_empty() {
        return Err(XiaobaiError::new(
            ErrorCode::ConfigInvalid,
            "No data chunk found in WAV file",
        ));
    }

    // Convert i16 to f32, downmix to mono if stereo
    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let total_samples = samples_data.len() / bytes_per_sample;
    let frame_size = num_channels as usize;
    let num_frames = total_samples / frame_size.max(1);

    let mut samples = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let mut sum: f32 = 0.0;
        for ch in 0..frame_size {
            let idx = (i * frame_size + ch) * bytes_per_sample;
            if idx + 1 < samples_data.len() {
                let val = i16::from_le_bytes([samples_data[idx], samples_data[idx + 1]]);
                sum += val as f32 / 32768.0;
            }
        }
        samples.push(sum / frame_size as f32);
    }

    Ok((samples, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let original: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let wav = encode_wav(&original, 22050).unwrap();
        let (decoded, sr) = decode_wav(&wav).unwrap();
        assert_eq!(sr, 22050);
        assert_eq!(decoded.len(), original.len());
        for (a, b) in original.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.001, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_header_size() {
        let wav = encode_wav(&[0.0f32; 100], 16000).unwrap();
        assert_eq!(wav.len(), WAV_HEADER_SIZE + 200);
    }

    #[test]
    fn test_invalid_header() {
        let result = decode_wav(&[0u8; 10]);
        assert!(result.is_err());
    }
}

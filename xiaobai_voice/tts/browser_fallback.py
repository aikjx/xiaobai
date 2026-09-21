"""Browser Fallback TTS：不生成音频字节，只在响应头标记 X-TTS-Fallback=browser，前端切回 SpeechSynthesis。"""
from __future__ import annotations

from collections.abc import Generator

from .base import TTSBackend, TTSOptions


class BrowserFallbackBackend(TTSBackend):
    name = "browser"

    def synthesize(self, opts: TTSOptions) -> Generator[bytes, None, None]:
        # 生成 0.5 s 静音作为 placeholder 音频（16 kHz × 16 bit × 1 channel = 16000 B/s）
        sr = 16000
        dur = 0.5
        data_len = int(sr * dur * 2)
        header = _make_wav_header(sr=sr, channels=1, bits=16, data_len=data_len)
        yield header
        yield b"\x00" * data_len


def _make_wav_header(*, sr: int, channels: int, bits: int, data_len: int) -> bytes:
    import struct

    byte_rate = sr * channels * bits // 8
    block_align = channels * bits // 8
    riff = (
        b"RIFF"
        + struct.pack("<I", 36 + data_len)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sr, byte_rate, block_align, bits)
        + b"data"
        + struct.pack("<I", data_len)
    )
    return riff

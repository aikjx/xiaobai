"""
xiaobai_voice.core — Rust core integration layer.

Provides a unified Python API over the `xiaobai_core` Rust extension
(PyO3). All performance-critical operations (DSP, config, intent routing,
model registry, RBAC operators) are delegated to Rust when available,
with graceful fallback to pure-Python implementations.

Usage:
    from xiaobai_voice.core import dsp, config, intent, operators, models

    # DSP: process TTS audio pipeline
    processed = dsp.process_tts_audio(samples, 22050, 16000, speed=1.03,
                                        loudness_target_dbfs=-18.0, limiter=True)

    # Intent routing
    router = intent.IntentRouter()
    result = router.route("打开记事本", role="Member")

    # Operators with RBAC
    engine = operators.OperatorEngine()
    result = engine.dispatch("volume", "get_volume", role="Member")
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rust extension availability probe
# ---------------------------------------------------------------------------

try:
    import xiaobai_core as _rust  # type: ignore

    RUST_AVAILABLE = True
    RUST_VERSION = getattr(_rust, "__version__", "unknown")
    logger.info("xiaobai_core v%s loaded (Rust acceleration enabled)", RUST_VERSION)
except ImportError as e:
    _rust = None  # type: ignore
    RUST_AVAILABLE = False
    RUST_VERSION = None
    logger.warning(
        "xiaobai_core Rust extension not available (%s). "
        "Falling back to pure-Python implementations. "
        "Build with: cd xiaobai_core && cargo build --release",
        e,
    )


# ---------------------------------------------------------------------------
# DSP submodule wrapper
# ---------------------------------------------------------------------------

class _DSPWrapper:
    """Digital Signal Processing — delegates to Rust when available."""

    def process_tts_audio(
        self,
        samples: list[float],
        input_sr: int,
        target_sr: int,
        speed: float = 1.0,
        loudness_target_dbfs: float = -18.0,
        limiter: bool = True,
    ) -> list[float]:
        """Full TTS post-processing pipeline: resample → SOLA → loudness → limiter."""
        if RUST_AVAILABLE:
            return _rust.dsp.process_tts_audio(
                samples, input_sr, target_sr, speed, loudness_target_dbfs, limiter
            )
        return self._py_process(samples, input_sr, target_sr, speed, loudness_target_dbfs, limiter)

    def resample(self, samples: list[float], from_sr: int, to_sr: int) -> list[float]:
        if RUST_AVAILABLE:
            return _rust.dsp.resample_linear(samples, from_sr, to_sr)
        return self._py_resample(samples, from_sr, to_sr)

    def normalize_loudness(self, samples: list[float], target_dbfs: float) -> list[float]:
        if RUST_AVAILABLE:
            return _rust.dsp.normalize_loudness(samples, target_dbfs)
        return samples  # Python fallback: no-op

    def soft_limit(self, samples: list[float], threshold: float = 0.995) -> list[float]:
        if RUST_AVAILABLE:
            return _rust.dsp.soft_limit(samples, threshold)
        return [max(-threshold, min(threshold, s)) for s in samples]

    def sola_time_stretch(self, samples: list[float], sample_rate: int, speed: float) -> list[float]:
        if RUST_AVAILABLE:
            return _rust.dsp.sola_time_stretch(samples, sample_rate, speed)
        return samples  # Python fallback: no-op

    def wav_encode(self, samples: list[float], sample_rate: int) -> bytes:
        if RUST_AVAILABLE:
            return bytes(_rust.dsp.wav_encode(samples, sample_rate))
        return self._py_wav_encode(samples, sample_rate)

    def wav_decode(self, data: bytes) -> tuple[list[float], int]:
        if RUST_AVAILABLE:
            return _rust.dsp.wav_decode(list(data))
        raise RuntimeError("Pure-Python WAV decode not implemented; install xiaobai_core")

    # --- Pure-Python fallbacks ---

    @staticmethod
    def _py_resample(samples: list[float], from_sr: int, to_sr: int) -> list[float]:
        if from_sr == to_sr or not samples:
            return samples
        ratio = from_sr / to_sr
        out_len = int(len(samples) / ratio) + 1
        out = []
        for i in range(out_len):
            src = i * ratio
            idx = int(src)
            frac = src - idx
            if idx + 1 < len(samples):
                out.append(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
            else:
                out.append(samples[min(idx, len(samples) - 1)])
        return out

    def _py_process(
        self, samples, input_sr, target_sr, speed, loudness_target_dbfs, limiter
    ):
        buf = self._py_resample(samples, input_sr, target_sr)
        if limiter:
            buf = [max(-0.995, min(0.995, s)) for s in buf]
        return buf

    @staticmethod
    def _py_wav_encode(samples: list[float], sample_rate: int) -> bytes:
        import struct
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(samples) * 2
        chunk_size = 36 + data_size
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", chunk_size, b"WAVE",
            b"fmt ", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
            b"data", data_size,
        )
        body = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
        return header + body


# ---------------------------------------------------------------------------
# Config submodule wrapper
# ---------------------------------------------------------------------------

class _ConfigWrapper:
    """Configuration management — delegates to Rust when available."""

    def ConfigLoader(self, user_path: Optional[str] = None, default_path: Optional[str] = None):
        if RUST_AVAILABLE:
            return _rust.config.ConfigLoader(user_path, default_path)
        from xiaobai_voice.config.loader import ConfigLoader as PyLoader
        return PyLoader(user_path, default_path)

    def platform_config_path(self) -> str:
        if RUST_AVAILABLE:
            return _rust.config.platform_config_path()
        from xiaobai_voice.config.loader import platform_config_path
        return str(platform_config_path())


# ---------------------------------------------------------------------------
# Intent submodule wrapper
# ---------------------------------------------------------------------------

class _IntentWrapper:
    """Intent routing — delegates to Rust when available."""

    def IntentRouter(self):
        if RUST_AVAILABLE:
            return _rust.intent.IntentRouter()
        from xiaobai_voice.intent.router import IntentRouter as PyRouter
        return PyRouter()


# ---------------------------------------------------------------------------
# Operators submodule wrapper
# ---------------------------------------------------------------------------

class _OperatorsWrapper:
    """System operators with RBAC — delegates to Rust when available."""

    def OperatorEngine(self, strategy: str = "local_first"):
        if RUST_AVAILABLE:
            return _rust.operators.OperatorEngine(strategy)
        from xiaobai_voice.operator.base import OperatorEngine as PyEngine
        return PyEngine(strategy)

    def access_level_from_role(self, role: str) -> int:
        if RUST_AVAILABLE:
            return _rust.operators.access_level_from_role(role)
        from xiaobai_voice.operator.base import AccessLevel
        return int(AccessLevel.from_role(role))


# ---------------------------------------------------------------------------
# Models submodule wrapper
# ---------------------------------------------------------------------------

class _ModelsWrapper:
    """Model registry — delegates to Rust when available."""

    def ModelRegistry(self, yaml_path: Optional[str] = None):
        if RUST_AVAILABLE:
            return _rust.models.ModelRegistry(yaml_path)
        raise RuntimeError("Pure-Python ModelRegistry not implemented; install xiaobai_core")

    def sha256_file(self, path: str) -> str:
        if RUST_AVAILABLE:
            return _rust.models.sha256_file(path)
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Public singletons
# ---------------------------------------------------------------------------

dsp = _DSPWrapper()
config = _ConfigWrapper()
intent = _IntentWrapper()
operators = _OperatorsWrapper()
models = _ModelsWrapper()

__all__ = [
    "dsp",
    "config",
    "intent",
    "operators",
    "models",
    "RUST_AVAILABLE",
    "RUST_VERSION",
]

"""TTS 抽象基类：流式字节生成（默认 WAV 16 bit PCM 头+分段）+ synthesize_full。"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from typing import Any


@dataclasses.dataclass
class TTSOptions:
    text: str
    voice: str = "xiaobai"
    emotion: str = "neutral"     # neutral | happy | sad | serious
    speed: float = 1.0           # 0.8 ~ 1.4
    sample_rate: int = 24000
    clone_reference: str | None = None   # 参考音频 hash id
    stream_chunk_ms: int = 250


class TTSBackend(ABC):
    name: str = "base"

    def __init__(self, cfg: dict, models_registry: Any | None = None) -> None:
        self.cfg = dict(cfg or {})
        self.models = models_registry
        self.sample_rate = int(self.cfg.get("sample_rate") or 24000)

    def prewarm(self) -> float:  # 默认空；引擎覆盖
        return 0.0

    def close(self) -> None:
        return None

    @abstractmethod
    def synthesize(self, opts: TTSOptions) -> Generator[bytes, None, None]:
        """同步流式字节：返回若干 chunk（每个 chunk 是 WAV 原始字节或完整 PCM chunk）。
        默认约定：**首个 chunk 必须包含 WAV RIFF header**（如 Content-Type 为 audio/wav）。
        """

    async def asynthesize(self, opts: TTSOptions) -> AsyncGenerator[bytes, None]:
        """异步包装同步生成器，便于 FastAPI StreamingResponse。"""
        import asyncio

        loop = asyncio.get_running_loop()
        it = self.synthesize(opts)
        while True:
            chunk = await loop.run_in_executor(None, next, it, None)
            if chunk is None:
                return
            yield chunk

    def synthesize_full(self, opts: TTSOptions) -> bytes:
        return b"".join(self.synthesize(opts))

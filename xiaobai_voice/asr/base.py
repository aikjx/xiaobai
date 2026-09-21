"""ASR 抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Iterable


@dataclass
class ASRPartial:
    text: str
    is_final: bool = False
    start_ms: int = 0
    end_ms: int = 0
    confidence: float = 0.0
    language: str | None = None


@dataclass
class ASRFullResult:
    text: str
    duration_ms: int = 0
    confidence: float = 0.0
    language: str | None = None
    segments: list[dict] = field(default_factory=list)


class ASRBackend(ABC):
    """所有 ASR 后端的统一接口。流式/全量/热词/预热/关闭。"""

    name: str = "base"

    def __init__(self, cfg: dict, models_registry: Any | None = None) -> None:
        self.cfg = cfg or {}
        self.models = models_registry
        self.sample_rate = int(self.cfg.get("sample_rate") or 16000)
        self.channels = int(self.cfg.get("channels") or 1)

    # ---------------------------------------------------------------- lifecycle
    @abstractmethod
    def prewarm(self) -> float:
        """启动时预热一次（防止用户首句被吞），返回预热耗时毫秒。"""

    @abstractmethod
    def close(self) -> None:
        """释放 onnxruntime / torch / GPU 资源。幂等。"""

    # ----------------------------------------------------------------- hotwords
    def set_hotwords(self, words: Iterable[dict]) -> None:
        """words = [{"word": str, "score": float}, ...]。默认空实现。"""

    # -------------------------------------------------------------- streaming
    @abstractmethod
    async def recognize_stream(
        self,
        chunks: AsyncGenerator[bytes, None],
        sample_rate: int | None = None,
    ) -> AsyncGenerator[ASRPartial, None]:
        """接受 16-bit PCM 字节流（默认 sample_rate=16000），实时吐出 partial/final。"""

    # -------------------------------------------------------------------- full
    @abstractmethod
    async def recognize_full(
        self,
        audio_bytes: bytes,
        sample_rate: int | None = None,
        fmt: str = "wav",
    ) -> ASRFullResult:
        """完整 WAV/WebM/PCM 一次性识别。"""

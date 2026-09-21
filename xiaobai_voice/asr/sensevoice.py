"""SenseVoice（可选 S1）占位后端：S1 任务实现前拒绝启动，给出明确修复路径。"""
from __future__ import annotations

from .base import ASRBackend, ASRFullResult, ASRPartial


class SenseVoiceBackend(ASRBackend):  # pragma: no cover - 占位
    name = "sensevoice"

    def prewarm(self) -> float:  # noqa: D401
        raise NotImplementedError("S1: SenseVoice backend is optional and not implemented yet.")

    def close(self) -> None:
        return None

    async def recognize_stream(self, chunks, sample_rate=None):  # noqa: D401
        raise NotImplementedError

    async def recognize_full(self, audio_bytes, sample_rate=None, fmt="wav") -> ASRFullResult:
        raise NotImplementedError

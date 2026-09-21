"""ASR backends 包入口：统一接口 build_asr_backend(config)。"""
from __future__ import annotations

from .base import ASRBackend, ASRPartial, ASRFullResult  # noqa: F401


def build_asr_backend(
    config: dict,
    license_tier: str = "auto",
    models_registry: object | None = None,
) -> ASRBackend:
    """按照 config.voice.asr.engine 选择并实例化 ASR 后端。

    顺序：sensevoice 明确指定 → 否则 sherpa_paraformer；未安装或缺模型时抛出分级错误
    （ImportError → MISSING_DEP / FileNotFoundError → MISSING_MODEL）。
    """
    from .sherpa_paraformer import SherpaParaformerBackend
    from ..errors import ErrorCode, XiaobaiError

    voice = (config or {}).get("voice", {})
    asr_cfg = voice.get("asr", {})
    engine = (asr_cfg.get("engine") or "auto").strip().lower()

    # S1 可选 SenseVoice，未完成 S1 前只作占位：如显式指定则提示未来支持
    if engine in {"sensevoice"}:
        try:
            from .sensevoice import SenseVoiceBackend  # type: ignore
            return SenseVoiceBackend(asr_cfg, models_registry)
        except Exception as exc:  # noqa: BLE001
            raise XiaobaiError(
                code=ErrorCode.MISSING_DEP,
                message=(
                    "SenseVoice 后端未就绪（S1 可选任务）。"
                    "请把 voice.asr.engine 改回 auto / sherpa_paraformer，或安装依赖。"
                ),
                cause=exc,
            ) from exc

    return SherpaParaformerBackend(asr_cfg, models_registry)

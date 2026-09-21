"""TTS backends: Fish-S2-Pro（Research 延迟 import）/ CosyVoice2（Apache2）/ BrowserFallback。"""
from __future__ import annotations

import logging
import os

from .base import TTSBackend, TTSOptions  # noqa: F401

logger = logging.getLogger("xiaobai.tts")


def build_tts_backend(
    config: dict,
    license_tier: str = "auto",
    models_registry: object | None = None,
) -> TTSBackend:
    """按 license_tier + 配置 voice.tts.engine 选择 TTS 后端。

    - license_tier=apache2 → 禁止 Fish，强制 CosyVoice2（ImportError 时 BrowserFallback）
    - license_tier=research → 优先 Fish，权重缺失或未安装 → CosyVoice2 → Browser
    - license_tier=auto → 检测 Fish 权重完整才用，否则 CosyVoice2 → Browser
    """
    from .browser_fallback import BrowserFallbackBackend
    from .cosyvoice2 import CosyVoice2Backend
    from ..errors import ErrorCode, XiaobaiError

    voice = (config or {}).get("voice", {})
    tts_cfg = dict(voice.get("tts", {}) or {})
    # 双字段兼容：default_engine 是配置文件默认项；engine 是旧版外部调用者字段；都为空则走 auto。
    engine_raw = (
        tts_cfg.get("default_engine")
        or tts_cfg.get("engine")
        or tts_cfg.get("active_engine")
        or "auto"
    )
    engine = str(engine_raw).strip().lower()

    licence_ok_fish = license_tier in {"auto", "research"}
    candidates: list[str] = []
    if engine in {"fish_s2", "fish_s2_pro", "fish", "fishes2pro"} and licence_ok_fish:
        candidates = ["fish_s2", "cosyvoice2", "browser"]
    elif engine in {"cosyvoice2", "cosy"}:
        candidates = ["cosyvoice2", "browser"]
    elif engine == "browser":
        candidates = ["browser"]
    else:  # auto
        if licence_ok_fish:
            candidates = ["fish_s2", "cosyvoice2", "browser"]
        else:
            candidates = ["cosyvoice2", "browser"]

    last_exc: Exception | None = None
    for cand in candidates:
        try:
            if cand == "fish_s2":
                # Research License 安全锁：必须显式接受 Research License 才能启用 fish_s2_pro
                if os.environ.get("XIAOBAI_ACCEPT_RESEARCH_LICENSE", "0") != "1":
                    logger.warning(
                        "Fish-S2-Pro license is Research (non-commercial). "
                        "Set XIAOBAI_ACCEPT_RESEARCH_LICENSE=1 to accept and enable."
                    )
                    # 抛 XiaobaiError 让 candidates 循环降级，不影响 cosyvoice2
                    from ..errors import ErrorCode, XiaobaiError as _XE
                    raise _XE(
                        code=ErrorCode.RUNTIME,
                        message="Fish-S2-Pro 需设置 XIAOBAI_ACCEPT_RESEARCH_LICENSE=1 接受 Research License 后方可启用。",
                    )
                from .fish_s2 import FishS2Backend  # 延迟 import 防 license 污染

                return FishS2Backend(tts_cfg, models_registry)
            if cand == "cosyvoice2":
                return CosyVoice2Backend(tts_cfg, models_registry)
            return BrowserFallbackBackend(tts_cfg)
        except XiaobaiError as exc:
            last_exc = exc
            continue
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue

    raise XiaobaiError(
        code=ErrorCode.RUNTIME,
        message="TTS 所有后端均不可用。建议安装 CosyVoice2 或改用浏览器朗读。",
        cause=last_exc,
    )

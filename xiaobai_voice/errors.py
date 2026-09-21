"""统一分级错误。ImportError → MISSING_DEP / FileNotFound → MISSING_MODEL / OSError DLL → DLL_LOAD_FAIL / GPU_OOM。"""
from __future__ import annotations

import enum
from typing import Any


class ErrorCode(str, enum.Enum):
    OK = "OK"
    MISSING_DEP = "MISSING_DEP"
    MISSING_MODEL = "MISSING_MODEL"
    DLL_LOAD_FAIL = "DLL_LOAD_FAIL"
    GPU_OOM = "GPU_OOM"
    LICENSE_GATE = "LICENSE_GATE"
    CONFIG_INVALID = "CONFIG_INVALID"
    RUNTIME = "RUNTIME"
    # FR-13: 系统算子/鉴权/桥接错误
    PERMISSION_DENIED = "PERMISSION_DENIED"      # RBAC 鉴权失败：L0/L1 用户尝试 L2/L3 操作
    OPERATOR_UNSUPPORTED = "OPERATOR_UNSUPPORTED"  # 当前平台不支持该算子（如 Linux 音量）
    OPERATOR_FAILED = "OPERATOR_FAILED"          # 算子执行异常（进程打不开/文件不存在等）
    BRIDGE_DISCONNECTED = "BRIDGE_DISCONNECTED"  # voice_proxy ↔ mox-system 链路断开
    INTENT_AMBIGUOUS = "INTENT_AMBIGUOUS"        # 意图命中多个算子，需专家联盟裁决
    INTENT_UNKNOWN = "INTENT_UNKNOWN"            # 意图路由未命中
    # FR-5: 热词注入错误
    HOTWORDS_FORMAT = "HOTWORDS_FORMAT"          # 热词格式错误（缺 word/score）
    HOTWORDS_REINSTANTIATE = "HOTWORDS_REINSTANTIATE_FAIL"  # 重建 recognizer 失败
    UNKNOWN = "UNKNOWN"


class XiaobaiError(Exception):
    def __init__(
        self,
        code: str | ErrorCode = ErrorCode.UNKNOWN,
        message: str = "",
        cause: BaseException | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code if isinstance(code, ErrorCode) else ErrorCode(str(code))
        self.message = message
        self.cause = cause
        self.details = dict(details or {})

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "message": self.message or str(self),
            "cause": None if self.cause is None else f"{type(self.cause).__name__}: {self.cause}",
            "details": self.details,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"XiaobaiError({self.code.value}, {self.message!r})"

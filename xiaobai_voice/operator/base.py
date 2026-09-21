"""算子抽象层 + RBAC 4 级鉴权装饰器。

所有系统算子必须继承 Operator，用 `@require_level(AccessLevel.L1)` 装饰每个
公开的动作方法。OperatorEngine 负责：
1. 构造时加载所有已注册算子（register_defaults）
2. dispatch(op_name, act, params, identity) → 先鉴权再执行
3. 可选绑定 voice_proxy 审计上报（审计回调 / 专家联盟裁决回调）

硬约束（与 project_memory 中的 sensitivity / RBAC 设计一致）：
- identity.role 与 AccessLevel 的映射是单向提升（高级角色>低级），
  绝不能出现"Member能做Admin操作"的降级漏洞。
- 若 identity 为 None（未登录），默认按 L0 执行——仅允许只读无害动作。
"""
from __future__ import annotations

import abc
import enum
import functools
import logging
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..errors import ErrorCode, XiaobaiError

log = logging.getLogger("xiaobai.operator")


# ---------------------------------------------------------------------------
# RBAC 4 级（与 mox-system Role 对齐）
# ---------------------------------------------------------------------------

class AccessLevel(int, enum.Enum):
    L0_PUBLIC = 0     # Auditor：只读无害（列应用/读音量）
    L1_USER = 1       # Member：非破坏性写（开应用/调音量/开文件）
    L2_POWER = 2      # Expert / Coordinator：剪贴板 / 键鼠输入 / 鼠标点击
    L3_ADMIN = 3      # MoxAdmin：破坏性（关应用 / 删文件 / 键鼠自动化）

    @classmethod
    def from_role(cls, role_name: str) -> "AccessLevel":
        """与 mox-system 的 Role 枚举对齐：MoxAdmin/Coordinator/Expert/Member/Auditor"""
        m = {
            "MoxAdmin": cls.L3_ADMIN,
            "Coordinator": cls.L2_POWER,
            "Expert": cls.L2_POWER,
            "Member": cls.L1_USER,
            "Auditor": cls.L0_PUBLIC,
        }.get(role_name)
        if m is not None:
            return m
        # 数字位直接转：字符串 "L2" / "2"
        s = role_name.strip().upper()
        if s.startswith("L") and s[1:].isdigit():
            v = int(s[1:])
            return cls(max(0, min(3, v)))
        if s.isdigit():
            return cls(max(0, min(3, int(s))))
        # 默认未登录态 L0（安全默认）
        return cls.L0_PUBLIC


@dataclass
class Identity:
    """执行主体。None 或默认 → L0 Public 未登录。"""
    user_id: str = "anon"
    role: str = "Auditor"  # 对应 AccessLevel.from_role
    tenant_id: str = "default"

    @property
    def level(self) -> AccessLevel:
        return AccessLevel.from_role(self.role)


ID_PUBLIC = Identity()


# ---------------------------------------------------------------------------
# 动作 / 返回值
# ---------------------------------------------------------------------------

@dataclass
class OperatorAction:
    """每个算子声明的可执行动作及元数据。"""
    name: str                                  # e.g. "open_app"
    level: AccessLevel                         # 所需最小等级
    description: str = ""
    params_schema: dict[str, Any] = field(default_factory=dict)  # 类型提示，未做严格校验


@dataclass
class OperatorResult:
    op: str                                     # 算子命名空间 e.g. "app"
    act: str                                    # 动作名 e.g. "open_app"
    ok: bool
    code: str = ErrorCode.OK.value
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    audit_id: str = ""

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "act": self.act,
            "ok": self.ok,
            "code": self.code,
            "message": self.message,
            "data": self.data,
            "duration_ms": round(self.duration_ms, 2),
            "audit_id": self.audit_id,
        }


# ---------------------------------------------------------------------------
# Decorator: require_level
# ---------------------------------------------------------------------------

def require_level(level: AccessLevel) -> Callable:
    """给 Operator 的动作方法打标记，由 OperatorEngine.dispatch 前置校验。"""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            # 真实鉴权发生在 Engine，此处只做装饰留痕
            return fn(self, *args, **kwargs)
        wrapper.__required_level__ = level  # type: ignore[attr-defined]
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Operator 抽象基类
# ---------------------------------------------------------------------------

class Operator(abc.ABC):
    #: 命名空间，如 "app"/"file"/"input"/"volume"（唯一）
    name: str = ""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = dict(cfg or {})
        self._platform = sys.platform  # win32 / darwin / linux
        self._actions: dict[str, OperatorAction] = {}
        self._declare_actions()

    # -------------------------------------------------------------- 子类实现
    @abc.abstractmethod
    def _declare_actions(self) -> None:
        """在子类构造时注册所有动作：
        self._actions["open_app"] = OperatorAction("open_app", AccessLevel.L1_USER, ...)
        """

    def is_supported(self) -> bool:
        """当前平台是否支持该算子（默认全支持，跨平台缺失时子类 override）。"""
        return True

    # -------------------------------------------------------------- 内省 API
    def list_actions(self) -> list[OperatorAction]:
        return list(self._actions.values())

    def action(self, name: str) -> OperatorAction | None:
        return self._actions.get(name)

    def required_level(self, act: str) -> AccessLevel:
        a = self._actions.get(act)
        if a is None:
            raise XiaobaiError(
                ErrorCode.OPERATOR_UNSUPPORTED,
                f"[{self.name}] 未声明动作: {act}",
            )
        return a.level


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

AuditCallback = Callable[["Operator", str, dict, Identity, OperatorResult | None], None]
AllianceGateCallback = Callable[
    ["Operator", str, dict, Identity], tuple[bool, str]
]  # (allowed, deny_reason) — 专家联盟事前裁决闸


class OperatorEngine:
    """桌面小白助手的系统算子权威执行入口。

    Strategy（对齐 voice.strategy）：
    - local_first ：本地直接执行，**同步**调用 audit_cb 上报日志（若已连 mox），
                    异步调用 alliance_gate 失败时不阻断执行（只打 WARN，保证可用性）；
    - cloud_only  ：强制 alliance_gate 通过后才执行；
    - cloud_fallback：优先本地执行，若本地 OPERATOR_UNSUPPORTED 则调用 alliance_gate 远程算子代理
    """

    def __init__(
        self,
        cfg: dict | None = None,
        *,
        audit_cb: AuditCallback | None = None,
        alliance_gate: AllianceGateCallback | None = None,
    ) -> None:
        self.cfg = dict(cfg or {})
        self._ops: dict[str, Operator] = {}
        self.audit_cb = audit_cb
        self.alliance_gate = alliance_gate
        self.strategy = str((cfg or {}).get("strategy") or "local_first").lower()
        # 注册默认 4 类算子（最小交付路径）
        self.register_defaults()

    # -------------------------------------------------------------- 注册
    def register_defaults(self) -> None:
        # 延迟 import 避免循环依赖
        from .app_operator import AppOperator
        from .file_operator import FileOperator
        from .input_operator import InputOperator
        from .volume_operator import VolumeOperator

        for op_cls in (AppOperator, FileOperator, InputOperator, VolumeOperator):
            try:
                self.register(op_cls(self.cfg.get(op_cls.name)))
            except Exception as exc:  # noqa: BLE001
                log.warning("算子 %s 注册失败（平台不支持？）：%s", op_cls.name, exc)

    def register(self, op: Operator) -> None:
        if not op.name:
            raise XiaobaiError(ErrorCode.CONFIG_INVALID, "算子 name 不能为空")
        if not op.is_supported():
            log.info("算子 %s 在平台 %s 不支持，已跳过", op.name, platform.system())
            return
        self._ops[op.name] = op

    def list_operators(self) -> list[Operator]:
        return list(self._ops.values())

    # -------------------------------------------------------------- 执行
    def dispatch(
        self,
        op_name: str,
        act: str,
        params: dict | None = None,
        identity: Identity | None = None,
    ) -> OperatorResult:
        t0 = time.perf_counter()
        ident = identity or ID_PUBLIC
        params = dict(params or {})
        audit_id = f"aud_{int(t0*1000):x}_{abs(hash((op_name, act, ident.user_id))) % 0x7FFFFFFF:07x}"

        # 1) 算子存在？
        op = self._ops.get(op_name)
        if op is None:
            r = OperatorResult(
                op=op_name, act=act, ok=False,
                code=ErrorCode.OPERATOR_UNSUPPORTED.value,
                message=f"未知算子: {op_name}",
                audit_id=audit_id,
            )
            self._audit(op, act, params, ident, r, t0)
            return r

        # 2) 动作声明？
        act_meta = op.action(act)
        if act_meta is None:
            r = OperatorResult(
                op=op_name, act=act, ok=False,
                code=ErrorCode.OPERATOR_UNSUPPORTED.value,
                message=f"[{op_name}] 未知动作: {act}",
                audit_id=audit_id,
            )
            self._audit(op, act, params, ident, r, t0)
            return r

        # 3) RBAC 鉴权：level 不足 → PERMISSION_DENIED
        if ident.level < act_meta.level:
            msg = (
                f"身份 {ident.user_id}@{ident.role}(L{ident.level.value}) "
                f"无权执行 [{op_name}.{act}]（需 L{act_meta.level.value}）"
            )
            r = OperatorResult(
                op=op_name, act=act, ok=False,
                code=ErrorCode.PERMISSION_DENIED.value,
                message=msg,
                audit_id=audit_id,
            )
            self._audit(op, act, params, ident, r, t0)
            return r

        # 4) 专家联盟事前裁决（cloud_only 强制；cloud_fallback/local_first 不阻断）
        if self.alliance_gate is not None:
            try:
                allowed, reason = self.alliance_gate(op, act, params, ident)
                if not allowed and (self.strategy == "cloud_only" or not self._is_local_non_destructive(act_meta)):
                    r = OperatorResult(
                        op=op_name, act=act, ok=False,
                        code=ErrorCode.PERMISSION_DENIED.value,
                        message=f"专家联盟否决: {reason}",
                        audit_id=audit_id,
                    )
                    self._audit(op, act, params, ident, r, t0)
                    return r
            except Exception as exc:  # noqa: BLE001
                if self.strategy == "cloud_only":
                    r = OperatorResult(
                        op=op_name, act=act, ok=False,
                        code=ErrorCode.BRIDGE_DISCONNECTED.value,
                        message=f"联盟裁决链路不可用(cloud_only模式阻断): {exc}",
                        audit_id=audit_id,
                    )
                    self._audit(op, act, params, ident, r, t0)
                    return r
                log.warning("联盟裁决调用失败，local_first 放行：%s", exc)

        # 5) 调用真实方法
        r: OperatorResult
        try:
            handler = getattr(op, act, None)
            if handler is None:
                raise XiaobaiError(ErrorCode.OPERATOR_UNSUPPORTED,
                                   f"[{op_name}] 缺少动作实现: {act}")
            result_data: dict | None = handler(**params)
            r = OperatorResult(
                op=op_name, act=act, ok=True,
                code=ErrorCode.OK.value,
                data=result_data or {},
                audit_id=audit_id,
            )
        except XiaobaiError as exc:
            r = OperatorResult(
                op=op_name, act=act, ok=False,
                code=exc.code.value, message=exc.message,
                details=exc.details, audit_id=audit_id,  # type: ignore[call-arg]
            )
        except NotImplementedError:
            r = OperatorResult(
                op=op_name, act=act, ok=False,
                code=ErrorCode.OPERATOR_UNSUPPORTED.value,
                message=f"[{op_name}.{act}] 当前平台未实现",
                audit_id=audit_id,
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=2)
            log.exception("算子执行异常 [%s.%s]", op_name, act)
            r = OperatorResult(
                op=op_name, act=act, ok=False,
                code=ErrorCode.OPERATOR_FAILED.value,
                message=f"{type(exc).__name__}: {exc}",
                data={"traceback": tb},
                audit_id=audit_id,
            )

        self._audit(op, act, params, ident, r, t0)
        return r

    # -------------------------------------------------------------- 内部
    @staticmethod
    def _is_local_non_destructive(meta: OperatorAction) -> bool:
        # L0/L1 动作视为非破坏性，local_first 下即使联盟断线也放行
        return meta.level <= AccessLevel.L1_USER

    def _audit(
        self,
        op: Operator | None,
        act: str,
        params: dict,
        ident: Identity,
        result: OperatorResult,
        t0: float,
    ) -> None:
        result.duration_ms = (time.perf_counter() - t0) * 1000.0
        if self.audit_cb is not None:
            try:
                self.audit_cb(op, act, params, ident, result)
            except Exception as exc:  # noqa: BLE001
                log.warning("audit_cb 异常（不影响执行结果）：%s", exc)

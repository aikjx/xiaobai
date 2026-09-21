"""系统算子层（system-operator）— 桌面小白助手控制本地电脑的权威执行入口。

8 大类算子（AIS 企业级命名空间）：
- app     ：应用启停 / 进程列表（AppOperator）
- file    ：文件打开 / 剪贴板 / 回收站（FileOperator）
- input   ：鼠标移动点击 / 键盘按键 / 文本输入（InputOperator）
- volume  ：系统音量 / 静音 / 设备枚举（VolumeOperator）
- network ：网络诊断 / 代理切换（P2，后续）
- display ：分辨率 / 壁纸 / 锁屏（P2，后续）
- browser ：Chrome/Edge 打开 URL / 标签切换（P2，后续）
- notify  ：系统通知 / Toast（P2，后续）

RBAC 4 级（对齐 mox-expert 的维度优先级与 mox-system 的 Role 继承体系）：
    L0 Public     ← Auditor（只读全局）
    L1 User       ← Member（非破坏性：开应用/调音量）
    L2 Power      ← Expert/Coordinator（剪贴板/键鼠）
    L3 Admin      ← MoxAdmin（破坏性：关应用/删文件）

所有算子动作统一 go(act, **params) → OperatorResult。
本地执行(local_first)时，同步走 OperatorEngine → Python 原生跨平台 API；
若 voice_proxy 已连 mox-system，则同步上报审计日志（INTENT → EXEC → RESULT），
由 mox-expert 专家联盟事后审计 / 事前裁决（cloud_only 模式强制走裁决后执行）。
"""
from __future__ import annotations

from .base import (
    AccessLevel,
    Operator,
    OperatorAction,
    OperatorEngine,
    OperatorResult,
    require_level,
)
from .app_operator import AppOperator
from .file_operator import FileOperator
from .input_operator import InputOperator
from .volume_operator import VolumeOperator

__all__ = [
    "AccessLevel",
    "Operator",
    "OperatorAction",
    "OperatorEngine",
    "OperatorResult",
    "require_level",
    "AppOperator",
    "FileOperator",
    "InputOperator",
    "VolumeOperator",
]

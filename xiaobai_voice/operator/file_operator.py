"""文件算子：打开文件 / 复制到剪贴板 / 丢回收站 / 预览内容。

RBAC 分级：
- file_exists / read_text_head → L0 Public（只读无害）
- open_file_with_app          → L1 User（复用 AppOperator.open_file_with_app 语义）
- copy_to_clipboard           → L2 Power（剪贴板写入涉及用户数据外流风险）
- move_to_trash               → L3 Admin（破坏性：删除文件，尽管可回收）

跨平台剪贴板：优先 pyperclip，Windows 回退 ctypes 调用 CF_UNICODETEXT。
回收站：send2trash 库（未安装 → 升级 L3 直接 os.unlink，但必须先升级鉴权）。
"""
from __future__ import annotations

import base64
import os
import shutil
import sys
from pathlib import Path

from ..errors import ErrorCode, XiaobaiError
from .base import AccessLevel, Operator, OperatorAction, require_level


class FileOperator(Operator):
    name = "file"

    def _declare_actions(self) -> None:
        self._actions["file_exists"] = OperatorAction(
            "file_exists", AccessLevel.L0_PUBLIC,
            "检查路径是否存在",
            {"path": "str"},
        )
        self._actions["read_text_head"] = OperatorAction(
            "read_text_head", AccessLevel.L0_PUBLIC,
            "读取文本文件前 N 行（默认 20 行，防止读取大文件）",
            {"path": "str", "lines": "int 默认 20"},
        )
        self._actions["open_file_with_app"] = OperatorAction(
            "open_file_with_app", AccessLevel.L1_USER,
            "用系统默认应用打开文件/目录（同 app.open_file_with_app）",
            {"path": "str"},
        )
        self._actions["copy_to_clipboard"] = OperatorAction(
            "copy_to_clipboard", AccessLevel.L2_POWER,
            "写入文本或文件内容/Base64 到系统剪贴板",
            {"text": "str | None", "source_file": "str | None（与 text 二选一）", "as_file_bytes": "bool 默认 False（source_file 时，True=写 Base64 二进制，False=写文本）"},
        )
        self._actions["move_to_trash"] = OperatorAction(
            "move_to_trash", AccessLevel.L3_ADMIN,
            "移动到回收站（优先 send2trash；未安装→要求升级显式 allow_permanent_delete）",
            {"path": "str", "allow_permanent_delete": "bool 默认 False（若 send2trash 未装且=True，则 L3 下允许真删除）"},
        )

    # ------------------------------------------------------------------ 动作
    @require_level(AccessLevel.L0_PUBLIC)
    def file_exists(self, path: str) -> dict:
        p = Path(path).expanduser()
        return {"path": str(p), "exists": p.exists(), "is_file": p.is_file(), "is_dir": p.is_dir(),
                "size_bytes": p.stat().st_size if p.is_file() else None}

    @require_level(AccessLevel.L0_PUBLIC)
    def read_text_head(self, path: str, lines: int = 20) -> dict:
        p = Path(path).expanduser()
        if not p.is_file():
            raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"不是文件：{p}")
        try:
            buf: list[str] = []
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= max(1, int(lines)):
                        break
                    buf.append(line.rstrip("\n"))
            return {"path": str(p), "lines": buf, "truncated": len(buf) == max(1, int(lines))}
        except Exception as exc:  # noqa: BLE001
            raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"读取失败：{exc}", cause=exc) from exc

    @require_level(AccessLevel.L1_USER)
    def open_file_with_app(self, path: str) -> dict:
        from .app_operator import AppOperator
        return AppOperator(self.cfg).open_file_with_app(path)

    @require_level(AccessLevel.L2_POWER)
    def copy_to_clipboard(
        self,
        text: str | None = None,
        source_file: str | None = None,
        as_file_bytes: bool = False,
    ) -> dict:
        if (text is None and source_file is None) or (text is not None and source_file is not None):
            raise XiaobaiError(ErrorCode.CONFIG_INVALID, "text 和 source_file 必须且只能传一个")

        payload: str
        payload_kind: str
        if text is not None:
            payload = str(text)
            payload_kind = "text"
        else:
            p = Path(str(source_file)).expanduser()
            if not p.is_file():
                raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"源文件不存在：{p}")
            if as_file_bytes:
                raw = p.read_bytes()
                payload = base64.b64encode(raw).decode("ascii")
                payload_kind = "base64"
            else:
                payload = p.read_text(encoding="utf-8", errors="replace")
                payload_kind = "text"

        written = self._copy(payload)
        return {"ok": written, "kind": payload_kind, "chars": len(payload)}

    @require_level(AccessLevel.L3_ADMIN)
    def move_to_trash(self, path: str, allow_permanent_delete: bool = False) -> dict:
        p = Path(path).expanduser()
        if not p.exists():
            raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"路径不存在：{p}")
        try:
            import send2trash  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001
            send2trash = None  # type: ignore[assignment]

        if send2trash is not None:
            try:
                send2trash.send2trash(str(p))
                return {"method": "send2trash", "path": str(p)}
            except Exception as exc:  # noqa: BLE001
                raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"send2trash 失败：{exc}", cause=exc) from exc

        if not allow_permanent_delete:
            raise XiaobaiError(
                ErrorCode.OPERATOR_UNSUPPORTED,
                "未安装 send2trash 库。为避免误删，需显式传 allow_permanent_delete=True 才允许 L3 真删除。"
                "安装命令：pip install send2trash",
            )
        # L3 + allow_permanent_delete → 真删除（必须记录审计日志，由 OperatorEngine audit_cb 保证）
        if p.is_dir():
            shutil.rmtree(p)
        else:
            os.unlink(p)
        return {"method": "permanent_delete", "path": str(p)}

    # -------------------------------------------------------------- 剪贴板内核
    @staticmethod
    def _copy(text: str) -> bool:
        # 1) 优先 pyperclip（跨平台通用纯 Python）
        try:
            import pyperclip  # type: ignore[import-not-found]
            pyperclip.copy(text)
            return True
        except Exception:  # noqa: BLE001
            pass

        # 2) Windows: ctypes → SetClipboardData CF_UNICODETEXT
        if sys.platform.startswith("win"):
            try:
                import ctypes
                from ctypes import wintypes
                u32 = ctypes.WinDLL("user32", use_last_error=True)
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                CF_UNICODETEXT = 13
                GMEM_MOVEABLE = 0x0002

                u32.OpenClipboard(0)
                try:
                    u32.EmptyClipboard()
                    data = ctypes.create_unicode_buffer(text)
                    size = ctypes.sizeof(data)
                    hg = k32.GlobalAlloc(GMEM_MOVEABLE, size)
                    if not hg:
                        return False
                    lock = k32.GlobalLock(hg)
                    if not lock:
                        k32.GlobalFree(hg)
                        return False
                    ctypes.memmove(lock, data, size)
                    k32.GlobalUnlock(hg)
                    if not u32.SetClipboardData(CF_UNICODETEXT, hg):
                        k32.GlobalFree(hg)
                        return False
                    return True
                finally:
                    u32.CloseClipboard()
            except Exception:  # noqa: BLE001
                return False

        # 3) macOS: pbcopy
        if sys.platform == "darwin":
            try:
                import subprocess
                p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                p.communicate(text.encode("utf-8"), timeout=5)
                return p.returncode == 0
            except Exception:  # noqa: BLE001
                return False

        # 4) Linux: xclip (X11) / wl-copy (Wayland)
        try:
            import subprocess
            for tool in (["wl-copy"], ["xclip", "-sel", "clipboard", "-i"]):
                try:
                    p = subprocess.Popen(tool, stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    p.communicate(text.encode("utf-8"), timeout=5)
                    if p.returncode == 0:
                        return True
                except FileNotFoundError:
                    continue
        except Exception:  # noqa: BLE001
            pass
        return False

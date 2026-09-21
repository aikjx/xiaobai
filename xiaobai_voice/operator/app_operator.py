"""应用算子：打开应用 / 关闭应用 / 列出运行中的应用。

跨平台（Windows/macOS/Linux）实现：
- Windows：通过 `os.startfile` / `subprocess` / `tasklist` / `taskkill`
- macOS：`open` / `ps` / `pkill`
- Linux：`xdg-open` / `ps` / `pkill`

RBAC 分级：
- list_running       → L0 Public  （仅读）
- open_app           → L1 User    （非破坏性写：启动）
- close_app          → L3 Admin   （破坏性：结束进程）
- open_file_with_app → L1 User    （系统默认程序打开）
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from ..errors import ErrorCode, XiaobaiError
from .base import AccessLevel, Operator, OperatorAction, require_level


class AppOperator(Operator):
    name = "app"

    def _declare_actions(self) -> None:
        self._actions["list_running"] = OperatorAction(
            "list_running", AccessLevel.L0_PUBLIC,
            "列出运行中的前50个进程（名称+PID）",
            {"limit": "int, 默认 50"},
        )
        self._actions["open_app"] = OperatorAction(
            "open_app", AccessLevel.L1_USER,
            "打开应用 / 命令。支持：应用名（如 'notepad'/'chrome'）、绝对路径、或命令行（参数可通过 args 传入）",
            {"target": "str（必填）", "args": "list[str] | None", "cwd": "str | None", "shell": "bool 默认 false"},
        )
        self._actions["close_app"] = OperatorAction(
            "close_app", AccessLevel.L3_ADMIN,
            "关闭应用。优先按名称关闭（taskkill/pkill），也可传 pid。",
            {"name": "str（可选，进程名如 chrome.exe）", "pid": "int（可选）", "force": "bool 默认 false"},
        )
        self._actions["open_file_with_app"] = OperatorAction(
            "open_file_with_app", AccessLevel.L1_USER,
            "用系统默认应用打开文件（ShellExecute）",
            {"path": "str（必填，本地绝对路径）"},
        )

    # ------------------------------------------------------------------ 动作
    @require_level(AccessLevel.L0_PUBLIC)
    def list_running(self, limit: int = 50) -> dict:  # 权限校验在 Engine 中完成
        if sys.platform.startswith("win"):
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Get-Process | Select-Object -First 50 Id,ProcessName,StartTime | ConvertTo-Json"]
            try:
                out = subprocess.check_output(cmd, timeout=5, text=True, stderr=subprocess.DEVNULL)
                import json as _json
                arr = _json.loads(out or "[]")
                items = []
                for p in (arr if isinstance(arr, list) else [arr]):
                    items.append({"pid": int(p.get("Id", 0)),
                                  "name": str(p.get("ProcessName", "")),
                                  "start_time": str(p.get("StartTime", "")),
                                  })
                return {"processes": items[:limit]}
            except Exception as exc:  # noqa: BLE001
                raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"查询进程列表失败：{exc}") from exc
        else:
            # POSIX: ps -eo pid,comm,lstart --no-headers
            try:
                out = subprocess.check_output(
                    ["ps", "-eo", "pid,comm,lstart", "--no-headers"],
                    timeout=5, text=True, stderr=subprocess.DEVNULL,
                )
                items = []
                for line in out.splitlines()[:limit]:
                    parts = line.split(None, 2)
                    if len(parts) >= 2:
                        items.append({
                            "pid": int(parts[0]),
                            "name": parts[1],
                            "start_time": parts[2] if len(parts) >= 3 else "",
                        })
                return {"processes": items}
            except Exception as exc:  # noqa: BLE001
                raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"查询进程列表失败：{exc}") from exc

    @require_level(AccessLevel.L1_USER)
    def open_app(
        self,
        target: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        shell: bool = False,
    ) -> dict:
        if not target:
            raise XiaobaiError(ErrorCode.CONFIG_INVALID, "open_app.target 不能为空")
        args = list(args or [])
        try:
            # Windows: 先尝试 startfile（更像用户手动双击），不占终端
            if sys.platform.startswith("win") and not args and (
                Path(target).exists() or os.path.isabs(target)
                or ("." in target and Path(target).suffix)
            ):
                try:
                    os.startfile(target)  # type: ignore[attr-defined]
                    return {"method": "startfile", "target": target}
                except OSError:
                    pass  # fallback 到 subprocess
            # 优先通过 PATH 找绝对路径
            exe = target
            if not os.path.isabs(target):
                resolved = shutil.which(target)
                if resolved:
                    exe = resolved
            popen_args = [exe, *args] if not shell else f"{shlex.quote(exe)} {' '.join(shlex.quote(a) for a in args)}"
            p = subprocess.Popen(  # noqa: S603
                popen_args, shell=shell, cwd=cwd,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            return {"method": "subprocess", "pid": p.pid, "command": popen_args if not shell else popen_args}
        except FileNotFoundError as exc:
            raise XiaobaiError(
                ErrorCode.OPERATOR_FAILED,
                f"找不到目标应用: {target}（请确认已安装并在 PATH 中，或传入绝对路径）",
                cause=exc,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"open_app 失败：{exc}", cause=exc) from exc

    @require_level(AccessLevel.L3_ADMIN)
    def close_app(
        self,
        name: str | None = None,
        pid: int | None = None,
        force: bool = False,
    ) -> dict:
        if not name and pid is None:
            raise XiaobaiError(ErrorCode.CONFIG_INVALID, "close_app 至少需要 name 或 pid 之一")
        try:
            if sys.platform.startswith("win"):
                cmd = ["taskkill"]
                if force:
                    cmd.append("/F")
                if pid is not None:
                    cmd += ["/PID", str(int(pid))]
                if name:
                    n = name if name.lower().endswith(".exe") else f"{name}.exe"
                    cmd += ["/IM", n]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)  # noqa: S603
                return {"returncode": res.returncode,
                        "stdout": res.stdout.strip(),
                        "stderr": res.stderr.strip(),
                        "command": cmd}
            else:
                # POSIX: kill / pkill
                cmd: list[str]
                if pid is not None:
                    cmd = ["kill", "-9" if force else "-15", str(int(pid))]
                else:
                    cmd = (["pkill", "-9"] if force else ["pkill", "-15"]) + [str(name)]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)  # noqa: S603
                return {"returncode": res.returncode,
                        "stdout": res.stdout.strip(),
                        "stderr": res.stderr.strip(),
                        "command": cmd}
        except Exception as exc:  # noqa: BLE001
            raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"close_app 失败：{exc}", cause=exc) from exc

    @require_level(AccessLevel.L1_USER)
    def open_file_with_app(self, path: str) -> dict:
        p = Path(path).expanduser()
        if not p.is_file() and not p.is_dir():
            raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"路径不存在或无权限：{p}")
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))  # type: ignore[attr-defined]
                return {"method": "startfile", "path": str(p)}
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return {"method": "open", "path": str(p)}
            # Linux
            subprocess.Popen(["xdg-open", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"method": "xdg-open", "path": str(p)}
        except Exception as exc:  # noqa: BLE001
            raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"open_file_with_app 失败：{exc}", cause=exc) from exc

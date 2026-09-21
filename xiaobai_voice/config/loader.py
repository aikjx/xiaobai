"""跨平台配置加载：默认值 deep-merge + 文件监听热更新 + 配置路径解析。"""
from __future__ import annotations

import copy
import logging
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("xiaobai.config")


def _platform_config_path() -> Path:
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "mox" / "xiaobai" / "config.yaml"
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "mox" / "xiaobai" / "config.yaml"
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(xdg) / "mox" / "xiaobai" / "config.yaml"


def default_log_path() -> Path:
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        root = Path(base) / "mox" / "xiaobai" / "logs"
    elif system == "Darwin":
        root = Path.home() / "Library" / "Logs" / "mox" / "xiaobai"
    else:
        xdg = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
        root = Path(xdg) / "mox" / "xiaobai" / "logs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _default_voice_models_dirs() -> list[Path]:
    """模型解析路径顺序：exe同级 > 用户目录 > 仓库 models/"""
    dirs: list[Path] = []
    if getattr(sys, "frozen", False):
        dirs.append(Path(sys.executable).resolve().parent / "models")
    dirs.append(Path.home() / ".mox" / "models" / "voice")
    dirs.append(Path(__file__).resolve().parent.parent.parent / "models")
    return [d for d in dirs if d is not None]


class ConfigLoader:
    DEFAULT_FILENAME = "default_config.yaml"

    @staticmethod
    def _resolve_default_path(filename: str) -> Path:
        """解析默认配置文件路径：frozen 时优先 cli._RESOURCE_ROOT 兜底。"""
        # 1) 先尝试 frozen 下由 cli.main() 确定的资源根
        try:
            from .. import cli
            rr = getattr(cli, "_RESOURCE_ROOT", None)
            if rr is not None:
                cand = Path(rr) / "xiaobai_voice" / "config" / filename
                if cand.is_file():
                    return cand
        except Exception:  # noqa: BLE001
            pass
        # 2) 开发态 / 兜底：按 __file__ 相对路径（PyInstaller 下通常也正确）
        return Path(__file__).resolve().parent / filename

    def __init__(
        self,
        user_path: Path | None = None,
        watch: bool = False,
        on_change=None,
    ) -> None:
        self.user_path = Path(user_path) if user_path else _platform_config_path()
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_path = self._resolve_default_path(self.DEFAULT_FILENAME)
        self._data: dict = {}
        self._mtime: float | None = None
        self._on_change = on_change
        self._lock = threading.RLock()
        self.load()
        self._watcher = None
        if watch:
            self._watcher = threading.Thread(target=self._watch_loop, name="xiaobai-config-watcher", daemon=True)
            self._watcher.start()

    # -------------------------------------------------------------------- load
    def load(self) -> dict:
        with self._lock:
            defaults = self._read_yaml(self.default_path) or {}
            user = self._read_yaml(self.user_path) or {}
            merged = _deep_merge(defaults, user)
            self._data = merged
            try:
                self._mtime = self.user_path.stat().st_mtime
            except FileNotFoundError:
                self._mtime = None
            return self._data

    @property
    def data(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._data)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return copy.deepcopy(node)

    # ------------------------------------------------------------------ persist
    def save_patch(self, patch: dict) -> dict:
        """浅写入：只改用户 config.yaml（按层级合并 patch），不直接改内存。"""
        with self._lock:
            existing = self._read_yaml(self.user_path) or {}
            merged = _deep_merge(existing, patch)
            self._atomic_write_yaml(self.user_path, merged)
            self.load()
            if callable(self._on_change):
                try:
                    self._on_change(self.data)
                except Exception:  # noqa: BLE001
                    log.exception("on_change handler raised.")
            return self.data

    # ------------------------------------------------------------------- watch
    def _watch_loop(self) -> None:  # pragma: no cover
        while True:
            time.sleep(1.5)
            try:
                mtime = self.user_path.stat().st_mtime
            except FileNotFoundError:
                mtime = None
            if mtime != self._mtime:
                log.info("config file changed, reloading.")
                self.load()
                if callable(self._on_change):
                    try:
                        self._on_change(self.data)
                    except Exception:  # noqa: BLE001
                        log.exception("on_change handler raised.")

    # --------------------------------------------------------------------- I/O
    @staticmethod
    def _read_yaml(path: Path) -> dict | None:
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:  # noqa: BLE001
            log.warning("读取配置失败 %s: %s", path, exc)
            return None

    @staticmethod
    def _atomic_write_yaml(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp, path)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ==============================================
# 对外公开别名（修复下游模块 import 契约不一致）：
# 部分调用方按「公开名」导入，实际实现用 _ 前缀的私有名。
# ==============================================
default_voice_models_dirs = _default_voice_models_dirs

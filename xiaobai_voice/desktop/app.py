"""桌面应用入口：QApplication + 悬浮球 + 主窗口 + 快捷键。"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("xiaobai.desktop")


@dataclass
class _Ctx:
    app: Any
    ball: Any
    main_window: Any
    hotkeys: Any
    loader: Any
    port: int


def run_desktop(args: Any) -> int | None:
    """桌面模式入口：初始化 Qt、加载球和主窗口、启动热键；阻塞直到退出。"""
    try:
        from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
    except ImportError as exc:
        log.error("PySide6 未安装，请 pip install PySide6：%s", exc)
        print("[desktop] 缺少 PySide6 依赖。", file=sys.stderr)
        return 4

    from ..config.loader import ConfigLoader
    loader = ConfigLoader(watch=False)
    port = int(getattr(args, "port", 0) or int(loader.get("voice.port") or 30010))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:0] if False else sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("Xiaobai")
    app.setOrganizationName("Mox")

    # 主题（深空 φ 色系）
    _apply_qt_dark_theme(app)

    # 浮窗球
    from .ball_widget import BallWidget
    size = int(loader.get("ui.float_ball_size") or 68)
    ball = BallWidget(size=size)
    ball.show()

    # 主窗口
    from .main_window import MainWindow
    ai_url = str(loader.get("ui.ai_dialog_url") or "http://localhost:3020/#/ai")
    mw = MainWindow(loader=loader, default_url=ai_url, port=port)

    # 快捷键
    from .hotkeys import HotkeyManager
    hk = HotkeyManager()
    shortcuts = loader.get("ui.shortcuts") or {}

    def on_toggle_record():
        ball.toggle_listen(mw)

    def on_read_clipboard():
        text = app.clipboard().text()
        if not text:
            ball.show_toast("剪贴板没有文本")
            return
        ball.show_toast("📣 正在朗读剪贴板")
        # 走桌面侧播放
        mw.play_text_via_voice(text)

    def on_quit():
        ball.show_toast("已退出 xiaobai")
        QtCore.QTimer.singleShot(600, _do_quit)

    def _do_quit():
        try:
            hk.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            mw.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            ball.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            stop_marker = Path(os.environ.get("TMP") or os.environ.get("TEMP") or "/tmp") / "mox_xiaobai_stop"
            stop_marker.write_text(str(int(time.time())), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        try:
            QtWidgets.QApplication.quit()
        except Exception:  # noqa: BLE001
            pass
        # 3 s 强退
        QtCore.QTimer.singleShot(3000, lambda: os._exit(0))

    hk.bind(shortcuts.get("toggle_record") or "Alt+X", on_toggle_record)
    hk.bind(shortcuts.get("read_clipboard") or "Alt+S", on_read_clipboard)
    hk.bind(shortcuts.get("quit") or "Alt+Q", on_quit)
    hk.start()

    # 信号连接
    from PySide6.QtCore import QObject, Signal
    class _Bridge(QObject):
        recognized = Signal(str)
    bridge = _Bridge()
    bridge.recognized.connect(mw.paste_recognized_text)
    ball.recognized_callback = bridge.recognized.emit

    ctx = _Ctx(app=app, ball=ball, main_window=mw, hotkeys=hk, loader=loader, port=port)
    # 全局单例，避免被 GC
    setattr(app, "_xiaobai_ctx", ctx)
    return app.exec()


def _apply_qt_dark_theme(app) -> None:
    from PySide6 import QtGui, QtWidgets  # type: ignore
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#0b1020"))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#0e1530"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#111a3c"))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor("#1a2350"))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#e6eaff"))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#e6eaff"))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#182049"))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#e6eaff"))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor("#ff5c8a"))
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor("#818cf8"))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#6366f1"))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(palette)
    try:
        QtWidgets.QToolTip.setPalette(palette)
    except Exception:  # noqa: BLE001
        pass

"""全局快捷键：Alt+X / Alt+S / Alt+Q，独立线程驱动。"""
from __future__ import annotations

import logging
import threading
from typing import Callable

log = logging.getLogger("xiaobai.hotkeys")


class HotkeyManager:
    def __init__(self) -> None:
        self._binds: dict[str, Callable[[], None]] = {}
        self._listener = None
        self._thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._qt_bridge = None

    def bind(self, combo: str, cb: Callable[[], None]) -> None:
        combo_norm = _normalize(combo)
        if not combo_norm:
            log.warning("非法快捷键：%r", combo)
            return
        self._binds[combo_norm] = _marshall_to_qt(cb)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopped.clear()
        self._thread = threading.Thread(target=self._loop, name="xiaobai-hotkeys", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        try:
            if self._listener is not None and hasattr(self._listener, "stop"):
                self._listener.stop()
        except Exception:  # noqa: BLE001
            pass
        self._thread = None

    def _loop(self) -> None:
        try:
            from pynput import keyboard  # type: ignore
        except Exception as exc:  # noqa: BLE001
            log.warning("pynput 未安装，快捷键不可用：%s", exc)
            return
        try:
            mapping = {}
            for combo, cb in self._binds.items():
                try:
                    mapping[keyboard.HotKey.parse(combo)] = cb
                except Exception as exc:  # noqa: BLE001
                    log.warning("快捷键解析失败 %s: %s", combo, exc)
            for k in mapping.keys():
                pass  # 显式持有
        except Exception as exc:  # noqa: BLE001
            log.warning("快捷键映射初始化失败：%s", exc)
            return

        def on_press(key):
            for hk, cb in mapping.items():
                try:
                    hk.press(listener.canonical(key))
                except AttributeError:
                    pass
                except Exception:  # noqa: BLE001
                    pass

        def on_release(key):
            for hk in mapping.keys():
                try:
                    fired = hk.release(listener.canonical(key))
                except Exception:  # noqa: BLE001
                    continue
                if fired:
                    try:
                        mapping[hk]()
                    except Exception:  # noqa: BLE001
                        log.exception("hotkey cb error")

        try:
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                self._listener = listener
                while not self._stopped.is_set():
                    if not listener.running:
                        break
                    self._stopped.wait(timeout=0.5)
        except Exception as exc:  # noqa: BLE001
            log.warning("快捷键监听退出：%s", exc)


def _normalize(combo: str) -> str:
    # Alt+X -> '<alt>+x'（pynput 通用写法）
    if not combo:
        return ""
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    out = []
    for p in parts:
        if p in {"alt", "ctrl", "control", "shift", "cmd", "win", "super"}:
            out.append("<" + p.replace("control", "ctrl").replace("cmd", "cmd").replace("win", "cmd").replace("super", "cmd") + ">")
        elif len(p) == 1:
            out.append(p)
        else:
            out.append("<" + p + ">")
    return "+".join(out)


def _marshall_to_qt(cb: Callable[[], None]) -> Callable[[], None]:
    """确保快捷键回调在 Qt 主线程执行，避免跨线程操作 UI 崩溃。"""
    def wrapped():
        try:
            from PySide6 import QtCore, QtWidgets  # type: ignore
            app = QtWidgets.QApplication.instance()
            if app is None:
                cb()
                return
            QtCore.QMetaObject.invokeMethod(app, cb, QtCore.Qt.QueuedConnection)
        except Exception:  # noqa: BLE001
            try:
                cb()
            except Exception:  # noqa: BLE001
                log.exception("hotkey wrapper cb error")
    return wrapped

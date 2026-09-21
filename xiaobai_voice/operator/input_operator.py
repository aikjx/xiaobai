"""键鼠输入算子：鼠标移动 / 点击 / 拖拽 / 键盘按键 / 文本输入 / 屏幕截图。

跨平台优先 pynput → Windows ctypes → macOS Quartz（未装则 NotImplementedError，
由 Engine 捕获为 OPERATOR_UNSUPPORTED）。

RBAC 分级：
- mouse_position / get_screenshot_dimensions → L0 Public（只读）
- type_text                              → L2 Power（模拟用户键盘，可改文件/填表单）
- press_key / hotkey / mouse_move / click → L2 Power
- mouse_drag / screenshot                 → L3 Admin（截图涉及隐私，拖拽风险更高）
"""
from __future__ import annotations

import base64
import io
import subprocess
import sys
import time

from ..errors import ErrorCode, XiaobaiError
from .base import AccessLevel, Operator, OperatorAction, require_level


# ---------------------------------------------------------------------------
# 可选依赖辅助
# ---------------------------------------------------------------------------

def _pynput_mouse():
    try:
        from pynput.mouse import Button, Controller  # type: ignore[import-not-found]
        return Button, Controller()
    except Exception:
        return None, None


def _pynput_keyboard():
    try:
        from pynput.keyboard import Controller, Key  # type: ignore[import-not-found]
        return Controller(), Key
    except Exception:
        return None, None


_MOUSE_BTN_MAP = {"left": 0, "middle": 1, "right": 2}
_PYNPUT_BTN_NAME = {"left": "left", "middle": "middle", "right": "right"}


class InputOperator(Operator):
    name = "input"

    def _declare_actions(self) -> None:
        self._actions["mouse_position"] = OperatorAction(
            "mouse_position", AccessLevel.L0_PUBLIC, "读取当前鼠标坐标 (x,y)")
        self._actions["mouse_move"] = OperatorAction(
            "mouse_move", AccessLevel.L2_POWER,
            "移动鼠标到 (x,y)；若 relative=True 则相对移动",
            {"x": "int", "y": "int", "relative": "bool 默认 false"},
        )
        self._actions["mouse_click"] = OperatorAction(
            "mouse_click", AccessLevel.L2_POWER,
            "鼠标点击",
            {"button": "left|middle|right 默认 left", "clicks": "int 默认 1（双击传2）"},
        )
        self._actions["mouse_drag"] = OperatorAction(
            "mouse_drag", AccessLevel.L3_ADMIN,
            "从 (x1,y1) 拖拽到 (x2,y2) ，总耗时 seconds（默认 0.3）",
            {"x1": "int", "y1": "int", "x2": "int", "y2": "int", "seconds": "float 默认 0.3"},
        )
        self._actions["type_text"] = OperatorAction(
            "type_text", AccessLevel.L2_POWER,
            "输入文本（尽力支持 UTF-8，中文场景走剪贴板回退）",
            {"text": "str（必填）", "interval": "float 默认 0.0（毫秒），建议中文传 0"},
        )
        self._actions["press_key"] = OperatorAction(
            "press_key", AccessLevel.L2_POWER,
            "单键：a/b/enter/space/esc/tab/f1~f12/up/down/left/right/home/end/pgup/pgdn/backspace/delete/printscreen",
            {"key": "str（必填）", "down": "bool 默认 True（True=按下，False=抬起）", "hold": "float 默认 0.0（秒，仅 down=True 时）"},
        )
        self._actions["hotkey"] = OperatorAction(
            "hotkey", AccessLevel.L2_POWER,
            "组合键：keys= ['ctrl','c'] 或 'ctrl+alt+del'",
            {"keys": "list[str] | str（必填）"},
        )
        self._actions["screenshot"] = OperatorAction(
            "screenshot", AccessLevel.L3_ADMIN,
            "全屏或指定区域截图，返回 base64 PNG",
            {"x": "int|None", "y": "int|None", "w": "int|None", "h": "int|None", "quality": "int 1-100 默认 90"},
        )

    # ------------------------------------------------------------------
    # 只读（L0）
    # ------------------------------------------------------------------
    @require_level(AccessLevel.L0_PUBLIC)
    def mouse_position(self) -> dict:
        _, ctrl = _pynput_mouse()
        if ctrl is not None:
            p = ctrl.position
            return {"x": int(p[0]), "y": int(p[1]), "backend": "pynput"}
        if sys.platform.startswith("win"):
            import ctypes
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            pt = POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            return {"x": int(pt.x), "y": int(pt.y), "backend": "win32"}
        raise XiaobaiError(ErrorCode.OPERATOR_UNSUPPORTED, "未装 pynput，当前平台无回退实现 mouse_position")

    # ------------------------------------------------------------------
    # 鼠标（L2 / L3）
    # ------------------------------------------------------------------
    @require_level(AccessLevel.L2_POWER)
    def mouse_move(self, x: int, y: int, relative: bool = False) -> dict:
        _, ctrl = _pynput_mouse()
        if ctrl is not None:
            if relative:
                ctrl.move(int(x), int(y))
            else:
                ctrl.position = (int(x), int(y))
            return {"x": int(ctrl.position[0]), "y": int(ctrl.position[1]), "backend": "pynput"}
        if sys.platform.startswith("win"):
            import ctypes
            u = ctypes.windll.user32
            if relative:
                MOUSEEVENTF_MOVE = 0x0001
                u.mouse_event(MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)
            else:
                u.SetCursorPos(int(x), int(y))
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            pt = POINT()
            u.GetCursorPos(ctypes.byref(pt))
            return {"x": int(pt.x), "y": int(pt.y), "backend": "win32"}
        raise XiaobaiError(ErrorCode.OPERATOR_UNSUPPORTED, "mouse_move 需要安装 pynput 或 Windows 环境")

    @require_level(AccessLevel.L2_POWER)
    def mouse_click(self, button: str = "left", clicks: int = 1) -> dict:
        Button, ctrl = _pynput_mouse()
        if ctrl is not None and Button is not None:
            b = getattr(Button, _PYNPUT_BTN_NAME.get(button, button), Button.left)
            ctrl.click(b, count=max(1, int(clicks)))
            return {"button": button, "clicks": int(clicks), "backend": "pynput"}
        if sys.platform.startswith("win"):
            import ctypes
            u = ctypes.windll.user32
            flags_by_btn = {
                "left": (0x0002, 0x0004),     # LEFTDOWN / LEFTUP
                "middle": (0x0020, 0x0040),
                "right": (0x0008, 0x0010),
            }.get(button, (0x0002, 0x0004))
            for _ in range(max(1, int(clicks))):
                u.mouse_event(flags_by_btn[0], 0, 0, 0, 0)
                u.mouse_event(flags_by_btn[1], 0, 0, 0, 0)
            return {"button": button, "clicks": int(clicks), "backend": "win32"}
        raise XiaobaiError(ErrorCode.OPERATOR_UNSUPPORTED, "mouse_click 需要安装 pynput 或 Windows 环境")

    @require_level(AccessLevel.L3_ADMIN)
    def mouse_drag(self, x1: int, y1: int, x2: int, y2: int, seconds: float = 0.3) -> dict:
        Button, ctrl = _pynput_mouse()
        if ctrl is None or Button is None:
            raise XiaobaiError(ErrorCode.OPERATOR_UNSUPPORTED, "mouse_drag 需要先安装 pynput")
        steps = max(1, int(seconds * 60))  # 60fps 插值
        ctrl.position = (int(x1), int(y1))
        ctrl.press(Button.left)
        try:
            for i in range(1, steps + 1):
                t = i / steps
                nx = int(x1 + (x2 - x1) * t)
                ny = int(y1 + (y2 - y1) * t)
                ctrl.position = (nx, ny)
                time.sleep(max(0.001, seconds / steps))
        finally:
            ctrl.release(Button.left)
        return {"backend": "pynput", "from": (x1, y1), "to": (x2, y2), "duration_sec": seconds}

    # ------------------------------------------------------------------
    # 键盘（L2）
    # ------------------------------------------------------------------
    @require_level(AccessLevel.L2_POWER)
    def type_text(self, text: str, interval: float = 0.0) -> dict:
        if not isinstance(text, str):
            raise XiaobaiError(ErrorCode.CONFIG_INVALID, "text 必须是字符串")
        k, _ = _pynput_keyboard()
        # 全 ASCII：直接 type
        if text and all(ord(c) < 128 for c in text) and k is not None:
            try:
                k.type(text)
                return {"backend": "pynput.type", "chars": len(text)}
            except Exception:  # noqa: BLE001
                pass
        # 非 ASCII（中文）：粘贴到剪贴板 + Ctrl+V
        from .file_operator import FileOperator
        fo = FileOperator(self.cfg)
        written = fo.copy_to_clipboard(text=text)  # type: ignore[call-arg]
        if not written.get("ok"):
            raise XiaobaiError(ErrorCode.OPERATOR_FAILED, "剪贴板写入失败，中文字符输入无法回退")
        # 粘贴：Ctrl+V / Cmd+V
        if sys.platform == "darwin":
            self.hotkey("cmd+v")
        else:
            self.hotkey("ctrl+v")
        if interval and k is not None:
            time.sleep(interval)
        return {"backend": "clipboard+paste", "chars": len(text), "interval": interval}

    @require_level(AccessLevel.L2_POWER)
    def press_key(self, key: str, down: bool = True, hold: float = 0.0) -> dict:
        k, Key = _pynput_keyboard()
        if k is None or Key is None:
            raise XiaobaiError(ErrorCode.OPERATOR_UNSUPPORTED, "press_key 需要先安装 pynput")
        kk = self._resolve_pynput_key(Key, key)
        if down:
            k.press(kk)
            if hold:
                time.sleep(max(0.0, float(hold)))
                k.release(kk)
                return {"key": key, "action": "press+release", "hold": hold}
            return {"key": key, "action": "press"}
        k.release(kk)
        return {"key": key, "action": "release"}

    @require_level(AccessLevel.L2_POWER)
    def hotkey(self, keys) -> dict:
        if isinstance(keys, str):
            # 支持 "ctrl+alt+del" / "ctrl + c"
            split = [p.strip() for p in keys.replace("－", "+").split("+") if p.strip()]
        else:
            split = list(keys or [])
        k, Key = _pynput_keyboard()
        if k is None or Key is None:
            raise XiaobaiError(ErrorCode.OPERATOR_UNSUPPORTED, "hotkey 需要先安装 pynput")
        resolved = [self._resolve_pynput_key(Key, s) for s in split]
        for r in resolved:
            k.press(r)
        for r in reversed(resolved):
            k.release(r)
        return {"keys": split, "backend": "pynput"}

    # ------------------------------------------------------------------
    # 截图（L3）
    # ------------------------------------------------------------------
    @require_level(AccessLevel.L3_ADMIN)
    def screenshot(
        self,
        x: int | None = None, y: int | None = None,
        w: int | None = None, h: int | None = None,
        quality: int = 90,
    ) -> dict:
        # 1) mss 优先（跨平台纯 Python）
        try:
            import mss  # type: ignore[import-not-found]
            with mss.mss() as sct:
                if x is None or y is None or w is None or h is None:
                    mon = sct.monitors[0]
                    shot = sct.grab(mon)
                else:
                    shot = sct.grab({"left": int(x), "top": int(y),
                                     "width": int(w), "height": int(h)})
                from PIL import Image  # type: ignore[import-not-found]
                img = Image.frombytes("RGB", shot.size, shot.rgb)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                return {"backend": "mss+PIL", "size": shot.size, "base64_jpeg": b64,
                        "region": {"x": x, "y": y, "w": w, "h": h}}
        except Exception:  # noqa: BLE001
            pass
        # 2) Windows: Graphics.CopyFromScreen
        if sys.platform.startswith("win"):
            try:
                from PIL import ImageGrab  # type: ignore[import-not-found]
                bbox = None
                if x is not None and y is not None and w is not None and h is not None:
                    bbox = (int(x), int(y), int(x + w), int(y + h))
                img = ImageGrab.grab(bbox=bbox, all_screens=True)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                return {"backend": "PIL.ImageGrab", "size": img.size, "base64_jpeg": b64,
                        "region": {"x": x, "y": y, "w": w, "h": h}}
            except Exception as exc:  # noqa: BLE001
                raise XiaobaiError(ErrorCode.OPERATOR_UNSUPPORTED,
                                   "截图需要安装 mss+Pillow（或仅 Pillow）： pip install Pillow mss",
                                   cause=exc) from exc
        raise XiaobaiError(ErrorCode.OPERATOR_UNSUPPORTED, "截图需要先安装 mss 和 Pillow")

    # -------------------------------------------------------------- 内部
    @staticmethod
    def _resolve_pynput_key(Key, key: str):
        s = key.strip().lower()
        special = {
            "enter": "enter", "return": "enter",
            "space": "space", " ": "space",
            "esc": "esc", "escape": "esc",
            "tab": "tab",
            "backspace": "backspace", "bs": "backspace",
            "delete": "delete", "del": "delete",
            "ins": "insert", "insert": "insert",
            "home": "home", "end": "end",
            "pgup": "page_up", "pageup": "page_up", "page_up": "page_up",
            "pgdn": "page_down", "pagedown": "page_down", "page_down": "page_down",
            "up": "up", "down": "down", "left": "left", "right": "right",
            "caps": "caps_lock", "capslock": "caps_lock",
            "shift": "shift", "ctrl": "ctrl", "control": "ctrl",
            "alt": "alt", "option": "alt", "cmd": "cmd", "win": "cmd", "super": "cmd", "command": "cmd",
            "printscreen": "print_screen", "prtsc": "print_screen",
        }
        if s in special:
            return getattr(Key, special[s], s)
        if len(s) == 3 and s.startswith("f") and s[1:].isdigit():
            return getattr(Key, f"f{s[1:]}", s)
        # 单字符
        if len(s) == 1:
            return s
        return s

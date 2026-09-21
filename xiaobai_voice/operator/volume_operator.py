"""音量算子：获取/设置系统音量 / 静音 / 枚举音频设备。

跨平台实现：
- Windows：优先 `pycaw`（COM+CoreAudio）；回退 ctypes → waveOutGetVolume
- macOS：`osascript` → AppleScript `set volume`
- Linux：`amixer`（pulseaudio / ALSA）

RBAC 分级：
- get_volume / list_devices → L0 Public（只读取）
- set_volume / mute / unmute / toggle_mute → L1 User（非破坏性，仅音量调节）
"""
from __future__ import annotations

import math
import subprocess
import sys

from ..errors import ErrorCode, XiaobaiError
from .base import AccessLevel, Operator, OperatorAction, require_level


class VolumeOperator(Operator):
    name = "volume"

    def is_supported(self) -> bool:
        # 理论三平台都有回退；但 Linux 无 amixer + Windows 无 waveOut 才会失败
        return True

    def _declare_actions(self) -> None:
        self._actions["get_volume"] = OperatorAction(
            "get_volume", AccessLevel.L0_PUBLIC,
            "读取当前主音量(0-100) + 静音状态",
            {"device_id": "str | None（默认默认设备）"},
        )
        self._actions["set_volume"] = OperatorAction(
            "set_volume", AccessLevel.L1_USER,
            "设置主音量 0-100（或相对：value='+10'/'-5'）",
            {"value": "int | str（0-100 或 '+5'/'-10'）"},
        )
        self._actions["mute"] = OperatorAction("mute", AccessLevel.L1_USER, "静音开")
        self._actions["unmute"] = OperatorAction("unmute", AccessLevel.L1_USER, "静音关")
        self._actions["toggle_mute"] = OperatorAction("toggle_mute", AccessLevel.L1_USER, "切换静音")
        self._actions["list_devices"] = OperatorAction(
            "list_devices", AccessLevel.L0_PUBLIC,
            "枚举音频输出设备（尽力而为）",
        )

    # ------------------------------------------------------------------ 动作
    @require_level(AccessLevel.L0_PUBLIC)
    def get_volume(self, device_id: str | None = None) -> dict:
        return self._impl(device=device_id, action="get")

    @require_level(AccessLevel.L1_USER)
    def set_volume(self, value: int | float | str) -> dict:
        # 相对调节：+5 / -10
        if isinstance(value, str):
            s = value.strip()
            if s.startswith("+") or s.startswith("-"):
                curr = self._impl(action="get")
                base = int(curr.get("volume_percent", 50))
                target = base + int(s)
                target = max(0, min(100, target))
                return self._impl(action="set", target=target, mode="absolute")
            value = int(float(s))
        target = max(0, min(100, int(value)))
        return self._impl(action="set", target=target, mode="absolute")

    @require_level(AccessLevel.L1_USER)
    def mute(self) -> dict:
        return self._impl(action="mute", mode="set", muted=True)

    @require_level(AccessLevel.L1_USER)
    def unmute(self) -> dict:
        return self._impl(action="mute", mode="set", muted=False)

    @require_level(AccessLevel.L1_USER)
    def toggle_mute(self) -> dict:
        curr = self._impl(action="get")
        new_muted = not bool(curr.get("muted", False))
        r = self._impl(action="mute", mode="set", muted=new_muted)
        r["data"]["toggled_from"] = curr.get("muted")
        return r

    @require_level(AccessLevel.L0_PUBLIC)
    def list_devices(self) -> dict:
        # 尽力而为：先尝试 pycaw / pulsemixer；否则返回默认设备
        if sys.platform.startswith("win"):
            try:
                from pycaw.pycaw import AudioUtilities  # type: ignore[import-not-found]
                devs = AudioUtilities.GetSpeakers()
                names: list[dict] = []
                if hasattr(devs, "__iter__"):
                    for d in devs:
                        try:
                            names.append({"id": getattr(d, "id", ""), "name": getattr(d, "FriendlyName", str(d))})
                        except Exception:  # noqa: BLE001
                            names.append({"name": str(d)})
                if not names:
                    names.append({"name": "default-speaker"})
                return {"devices": names}
            except Exception:  # noqa: BLE001
                return {"devices": [{"name": "default-speaker", "note": "pycaw未装，无法枚举"}]}
        if sys.platform.startswith("linux"):
            try:
                out = subprocess.check_output(
                    ["pactl", "list", "short", "sinks"], text=True, stderr=subprocess.DEVNULL, timeout=5,
                )
                items = []
                for line in out.splitlines():
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        items.append({"id": parts[0], "name": parts[1], "driver": parts[2] if len(parts) >= 3 else ""})
                return {"devices": items}
            except Exception:  # noqa: BLE001
                return {"devices": [{"name": "default", "note": "pactl不可用"}]}
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPAudioDataType"], text=True, stderr=subprocess.DEVNULL, timeout=10,
            )
            return {"devices": [{"name": line.strip()} for line in out.splitlines() if ":" in line.strip()][:20]}
        except Exception:  # noqa: BLE001
            return {"devices": [{"name": "default"}]}

    # -------------------------------------------------------------- 平台内核
    def _impl(
        self,
        action: str,
        *,
        target: int = 50,
        mode: str = "absolute",
        muted: bool = False,
        device: str | None = None,
    ) -> dict:
        if sys.platform.startswith("win"):
            return self._win_impl(action, target=target, mode=mode, muted=muted, device=device)
        if sys.platform == "darwin":
            return self._mac_impl(action, target=target, mode=mode, muted=muted)
        return self._linux_impl(action, target=target, mode=mode, muted=muted)

    @staticmethod
    def _win_impl(
        action: str, *, target: int, mode: str, muted: bool, device: str | None,
    ) -> dict:
        # 1) 优先 pycaw（精准，多 Session 管理）
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            vol = cast(interface, POINTER(IAudioEndpointVolume))
            if action == "get":
                level = vol.GetMasterVolumeLevelScalar()  # 0.0 ~ 1.0
                is_muted = bool(vol.GetMute())
                return {"platform": "pycaw",
                        "volume_percent": int(round(float(level) * 100)),
                        "muted": is_muted,
                        "device": device or "default"}
            if action == "set":
                vol.SetMasterVolumeLevelScalar(float(target) / 100.0, None)
                return {"platform": "pycaw", "volume_percent": target}
            # mute
            vol.SetMute(1 if muted else 0, None)
            return {"platform": "pycaw", "muted": muted}
        except Exception:  # noqa: BLE001
            pass

        # 2) ctypes waveOut 回退（仅支持主音量，不含 mute 精细）
        try:
            import ctypes
            w32 = ctypes.WinDLL("winmm", use_last_error=True)
            if action == "get":
                vol = ctypes.c_uint()
                w32.waveOutGetVolume(0, ctypes.byref(vol))
                left = vol.value & 0xFFFF
                pct = int(left / 65535.0 * 100)
                return {"platform": "waveOut", "volume_percent": pct, "muted": False,
                        "note": "waveOut 无法读 mute，建议安装 pycaw"}
            if action == "set":
                word = (target * 65535 // 100) & 0xFFFF
                combined = word | (word << 16)
                w32.waveOutSetVolume(0, combined)
                return {"platform": "waveOut", "volume_percent": target}
            # mute: waveOut 不支持 → 设 0 并标记 muted
            if action == "mute":
                if muted:
                    w32.waveOutSetVolume(0, 0)
                return {"platform": "waveOut", "muted": muted,
                        "note": "waveOut 不支持真实mute：muted=true 时置0音量，false不动作。装 pycaw。"}
        except Exception as exc:  # noqa: BLE001
            raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"Windows音量控制失败（请安装 pycaw + comtypes）：{exc}",
                               cause=exc) from exc
        return {"platform": "waveOut"}

    @staticmethod
    def _mac_impl(action: str, *, target: int, mode: str, muted: bool) -> dict:
        try:
            if action == "get":
                out = subprocess.check_output(
                    ["osascript", "-e",
                     'get output volume of (get volume settings)\n get output muted of (get volume settings)'],
                    text=True, timeout=5, stderr=subprocess.DEVNULL,
                ).strip()
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                pct = int(lines[0]) if lines and lines[0].isdigit() else 50
                is_muted = any(l.lower() == "true" for l in lines[1:])
                return {"platform": "osascript", "volume_percent": pct, "muted": is_muted}
            if action == "set":
                subprocess.run(["osascript", "-e", f'set volume output volume {target}'],
                               check=True, timeout=5, capture_output=True)
                return {"platform": "osascript", "volume_percent": target}
            # mute
            flag = "with" if muted else "without"
            subprocess.run(["osascript", "-e", f'set volume {flag} output muted'],
                           check=True, timeout=5, capture_output=True)
            return {"platform": "osascript", "muted": muted}
        except Exception as exc:  # noqa: BLE001
            raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"macOS osascript 音量控制失败：{exc}", cause=exc) from exc

    @staticmethod
    def _linux_impl(action: str, *, target: int, mode: str, muted: bool) -> dict:
        # Pulse pactl 优先；回退 amixer
        def _have_pactl() -> bool:
            try:
                subprocess.run(["pactl", "--version"], capture_output=True, timeout=3, check=False)
                return True
            except FileNotFoundError:
                return False

        try:
            if _have_pactl():
                sink = "@DEFAULT_SINK@"
                if action == "get":
                    out = subprocess.check_output(
                        ["pactl", "get-sink-volume", sink], text=True, timeout=5, stderr=subprocess.DEVNULL,
                    ).strip()
                    # Volume: front-left: 32768 /  50% / -18.06 dB,   front-right: 32768 /  50% / ...
                    pct = 50
                    for tok in out.replace("/", " ").split():
                        if tok.endswith("%"):
                            try:
                                pct = int(tok.rstrip("%"))
                                break
                            except ValueError:
                                continue
                    m = subprocess.check_output(
                        ["pactl", "get-sink-mute", sink], text=True, timeout=5, stderr=subprocess.DEVNULL,
                    ).strip().lower()
                    is_muted = "yes" in m
                    return {"platform": "pactl", "volume_percent": pct, "muted": is_muted}
                if action == "set":
                    subprocess.run(["pactl", "set-sink-volume", sink, f"{target}%"],
                                   check=True, timeout=5, capture_output=True)
                    return {"platform": "pactl", "volume_percent": target}
                subprocess.run(["pactl", "set-sink-mute", sink, "1" if muted else "0"],
                               check=True, timeout=5, capture_output=True)
                return {"platform": "pactl", "muted": muted}

            # amixer 回退
            if action == "get":
                out = subprocess.check_output(
                    ["amixer", "sget", "Master"], text=True, timeout=5, stderr=subprocess.DEVNULL,
                )
                pct = 50
                is_muted = False
                for line in out.splitlines():
                    if "%" in line:
                        s = line[line.find("[") + 1:line.find("%]")]
                        if s.isdigit():
                            pct = int(s)
                    if "[off]" in line:
                        is_muted = True
                return {"platform": "amixer", "volume_percent": pct, "muted": is_muted}
            if action == "set":
                subprocess.run(["amixer", "sset", "Master", f"{target}%"],
                               check=True, timeout=5, capture_output=True)
                return {"platform": "amixer", "volume_percent": target}
            subprocess.run(["amixer", "sset", "Master", "mute" if muted else "unmute"],
                           check=True, timeout=5, capture_output=True)
            return {"platform": "amixer", "muted": muted}
        except Exception as exc:  # noqa: BLE001
            raise XiaobaiError(ErrorCode.OPERATOR_FAILED, f"Linux 音量控制失败（请安装 pulseaudio-utils 或 alsa-utils）：{exc}",
                               cause=exc) from exc

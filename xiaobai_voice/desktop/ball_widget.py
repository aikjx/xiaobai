"""桌面悬浮球 BallWidget：φ 圆形 + 4 状态（idle/listen/think/speak）+ 拖拽吸附 + 右键菜单。"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Callable

from PySide6 import QtCore, QtGui, QtWidgets


STATE_COLORS = {
    "idle":   "#22d3ee",  # 青
    "listen": "#22c55e",  # 绿
    "think":  "#a855f7",  # 紫
    "speak":  "#6366f1",  # 靛
    "executing": "#f97316",  # 橙：FR-13 第 5 状态 — 正在执行系统算子动作（控电脑）
}
EXECUTING_STATES = frozenset({"executing"})  # 此集合内的状态下不允许二次录音触发


class BallWidget(QtWidgets.QWidget):
    def __init__(self, size: int = 68, parent: Any = None) -> None:
        super().__init__(parent, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.size_ = int(size)
        self.setFixedSize(self.size_, self.size_)
        self._state = "idle"
        self._drag_offset: QtCore.QPoint | None = None
        self._press_pos: QtCore.QPoint | None = None
        self._anim_t = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(30)  # ~33fps，φ 呼吸 1.2s 周期
        # 录音对象
        self._recorder = _LocalRecorder(ball=self)
        self.recognized_callback: Callable[[str], None] | None = None
        # FR-13：意图 → 算子执行所需引用（可在外部调用 set_operator_engine 注入）
        self._operator_engine: Any = None       # operator.OperatorEngine
        self._proxy_client: Any = None          # proxy.VoiceProxyClient（optional）
        self._exec_worker: QtCore.QThread | None = None
        self._identity: Any = None              # operator.base.Identity
        self._exec_queue_serial = QtCore.QMutex()  # 串行执行（防并发鼠标/键盘竞态）
        self._toast: "_ToastWidget | None" = None
        # 默认放到屏幕右侧 y=120 像素处
        screen = self.screen().availableGeometry() if hasattr(self, "screen") else QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 16, screen.top() + 120)
        self.setToolTip("xiaobai · Alt+X 录音 · 双击打开 AI 对话 · 右键菜单")

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> str:
        return self._state

    def set_state(self, s: str) -> None:
        if s in STATE_COLORS:
            self._state = s
            self.update()

    # --------------------------------------------------------------- FR-13 API
    def set_operator_engine(self, engine, identity=None, proxy_client=None) -> None:
        """外部（desktop/main_window.py bootstrap）注入算子执行引擎。

        Args:
            engine: xiaobai_voice.operator.OperatorEngine（必传，FR-13 最小闭环）
            identity: xiaobai_voice.operator.base.Identity（默认 Member 本地账户）
            proxy_client: xiaobai_voice.proxy.VoiceProxyClient（可选，cloud_fallback/cloud_only 时必传）
        """
        self._operator_engine = engine
        self._proxy_client = proxy_client
        if identity is None:
            from ..operator.base import Identity as _Id
            identity = _Id(role="Member", user_id=f"desktop-{os.getenv('USERNAME', 'anon')}")
        self._identity = identity

    def execute_text(self, text: str) -> None:
        """把一句话交给 PPR 路由/专家联盟 → 系统算子执行，并在状态 executing 中可视化。

        调用来源：
        1) toggle_listen 识别成功后（若已注入 engine）；
        2) 外部快捷键或 AI 对话页 TTS 完成后的"意图→控电脑"路由。
        """
        if not text:
            return
        if self._operator_engine is None and self._proxy_client is None:
            # 尚未注入：降级为只 recognized_callback（旧独立行为）
            if callable(self.recognized_callback):
                self.recognized_callback(text)
            return
        if self._exec_queue_serial.tryLock(1) is False:
            # 串行保护：上一次未结束
            self.show_toast("⏳ 仍有指令执行中，请稍候…")
            return
        try:
            worker = _ExecWorker(
                text=text,
                engine=self._operator_engine,
                proxy=self._proxy_client,
                identity=self._identity,
            )
            self._exec_worker = worker
            self.set_state("executing")
            self.show_toast(f"⚙ 执行中：{text}", ms=3000)
            worker.signals.progress.connect(self._on_exec_progress)
            worker.signals.done.connect(self._on_exec_done)
            worker.start()
        except Exception as exc:  # noqa: BLE001
            self._exec_queue_serial.unlock()
            self.set_state("idle")
            self.show_toast(f"❌ 启动执行失败：{exc}")

    def _on_exec_progress(self, payload: dict) -> None:
        stage = payload.get("stage") or ""
        route = payload.get("route") or {}
        if stage == "routed":
            self.show_toast(
                f"🧭 路由：{route.get('op') or '?'}.{route.get('act') or '?'}  "
                f"conf={(route.get('confidence') or 0):.0%}",
                ms=1600,
            )

    def _on_exec_done(self, payload: dict) -> None:
        """payload = OperatorResult.to_dict() + {text, executed:bool}"""
        try:
            ok = bool(payload.get("ok"))
            op = payload.get("op") or ""
            act = payload.get("act") or ""
            msg = payload.get("message") or ""
            code = payload.get("code") or "OK"
            if ok:
                data = payload.get("data") or {}
                short = data.get("summary") or (f"{op}.{act} 完成")
                self.show_toast(f"✅ {short}", ms=2400)
                self.set_state("idle")
            else:
                # 区分权限/桥断/未知
                if code == "PERMISSION_DENIED":
                    self.show_toast(f"🚫 权限不足：{msg}", ms=3600)
                elif code == "BRIDGE_DISCONNECTED":
                    self.show_toast(f"🔌 mox 桥离线：{msg}", ms=3600)
                elif code == "INTENT_UNKNOWN":
                    self.show_toast(f"❓ 未匹配动作：{msg}", ms=3200)
                else:
                    self.show_toast(f"⚠ 执行失败：{code or 'ERROR'} {msg}", ms=3600)
                self.set_state("idle")
            # 通知上层（AI 对话页可展示执行结果）
            if callable(self.recognized_callback):
                try:
                    self.recognized_callback(payload.get("text") or "")
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._exec_queue_serial.unlock()

    def toggle_listen(self, main_window: Any) -> None:  # noqa: D401
        """点击或 Alt+X 触发：切换录音 → 识别 → 路由→算子 或 回调 → 自动朗读回答。"""
        if self._state in EXECUTING_STATES:
            self.show_toast("⏳ 指令执行中，不接受新录音")
            return
        if self._state == "listen":
            # 停止
            self._recorder.stop()
            self.set_state("think")
            recognized = self._recorder.last_recognized or ""
            if not recognized:
                self.set_state("idle")
                self.show_toast("🛑 未识别到内容")
                return
            self.show_toast("🛑 已识别：" + recognized[:20] + ("…" if len(recognized) > 20 else ""), ms=1800)
            # FR-13：优先本地 PPR 路由 / 代理桥 dispatch，失败时回退到 recognized_callback
            if self._operator_engine is not None or self._proxy_client is not None:
                self.execute_text(recognized)
                return
            if callable(self.recognized_callback):
                self.recognized_callback(recognized)
            self.set_state("idle")
        else:
            ok = self._recorder.start(main_window)
            if ok:
                self.set_state("listen")
                self.show_toast("🎙 正在听 · 说话结束按 Alt+X 停止")
            else:
                self.set_state("idle")
                self.show_toast("麦克风不可用")

    # --------------------------------------------------------------- painting
    def paintEvent(self, _ev) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform, True)
        s = self.size_
        rect = QtCore.QRectF(0, 0, s, s)
        # 多层柔边阴影（深空 φ 风格）
        for layer, (alpha, offset, radius) in enumerate([
            (0.18, 18, 22), (0.10, 10, 14), (0.05, 4, 8),
        ]):
            shadow = QtGui.QColor(0x0b1020)
            shadow.setAlphaF(alpha)
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(shadow)
            shadow_rect = QtCore.QRectF(rect).adjusted(offset, offset, -offset, -offset)
            # 用模糊阴影近似
            effect = QtWidgets.QGraphicsDropShadowEffect()
            effect.setBlurRadius(radius)
            effect.setColor(shadow)
            effect.setOffset(0, 2)
            # QPainter 不能直接用 effect；用画多层半透明近似
            p.drawEllipse(shadow_rect)
        # 深空渐变主体
        grd = QtGui.QRadialGradient(rect.center(), rect.width() / 2)
        grd.setColorAt(0.0, QtGui.QColor("#1f2a66"))
        grd.setColorAt(0.65, QtGui.QColor("#10183f"))
        grd.setColorAt(1.0, QtGui.QColor("#080c22"))
        p.setBrush(QtGui.QBrush(grd))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(rect.adjusted(2, 2, -2, -2))
        # 状态外环
        color = QtGui.QColor(STATE_COLORS[self._state])
        pen_w = max(2, int(s / 34))
        t = self._anim_t
        if self._state == "listen":
            # 呼吸：1.2s 周期
            pulse = 0.5 + 0.5 * math.sin(t / 1.2 * 2 * math.pi)
            color.setAlphaF(0.35 + 0.55 * pulse)
            pen = QtGui.QPen(color, pen_w + 1.5 * pulse, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawEllipse(rect.adjusted(4, 4, -4, -4))
        elif self._state == "think":
            # 旋转弧
            span = int(120 * 16)
            start = int((t * 360 / 2.4) % 360) * 16
            color.setAlphaF(0.9)
            pen = QtGui.QPen(color, pen_w, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawArc(QtCore.QRectF(rect).adjusted(5, 5, -5, -5), start, span)
        elif self._state == "speak":
            # 波形：12 条竖 φ bar
            n = 12
            gap = 2.0
            bar_w = (s - gap * (n + 1) - 16) / n
            base_y = s / 2
            for i in range(n):
                phase = t * 6 + i * 0.6
                h = (s - 24) / 2 * (0.25 + 0.75 * abs(math.sin(phase)))
                r = QtCore.QRectF(8 + i * (bar_w + gap), base_y - h, bar_w, 2 * h)
                color.setAlphaF(0.55 + 0.45 * (h / max(1, (s - 24) / 2)))
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(color)
                p.drawRoundedRect(r, bar_w / 2, bar_w / 2)
        elif self._state == "executing":
            # 三段彩虹弧 + 逆时针放射齿轮，突出"正在控制电脑"的紧迫感
            arc_rect = QtCore.QRectF(rect).adjusted(5, 5, -5, -5)
            colors_seq = ["#f97316", "#f59e0b", "#eab308"]
            for i, col in enumerate(colors_seq):
                c = QtGui.QColor(col)
                c.setAlphaF(0.75 if i == 0 else (0.55 if i == 1 else 0.4))
                pen = QtGui.QPen(c, pen_w + 1.3 * (2 - i), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
                p.setPen(pen)
                p.setBrush(QtCore.Qt.NoBrush)
                base_angle = int(t * 360 / 1.6) * 16   # 1.6s 整圈
                start = (base_angle + i * 120 * 16) % (360 * 16)
                span = 100 * 16 - i * 20 * 16
                p.drawArc(arc_rect.adjusted(-i * 0.7, -i * 0.7, i * 0.7, i * 0.7), start, span)
            # 12 个放射齿轮尖点（逆时针旋转 120°/s）
            nspikes = 12
            pen = QtGui.QPen(color, max(1.2, pen_w * 0.7), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
            p.setPen(pen)
            cx = rect.center().x()
            cy = rect.center().y()
            radius_outer = rect.width() / 2 - 6
            radius_inner = rect.width() / 2 - 12
            for i in range(nspikes):
                theta = 2 * math.pi * i / nspikes + t * 2 * math.pi / 1.2
                ox = cx + radius_outer * math.cos(theta)
                oy = cy + radius_outer * math.sin(theta)
                ix = cx + radius_inner * math.cos(theta)
                iy = cy + radius_inner * math.sin(theta)
                alpha = 0.35 + 0.5 * abs(math.sin(t * 3 + i * 0.5))
                color.setAlphaF(alpha)
                pen.setColor(color)
                p.setPen(pen)
                p.drawLine(QtCore.QPointF(ix, iy), QtCore.QPointF(ox, oy))
        else:  # idle
            color.setAlphaF(0.9)
            pen = QtGui.QPen(color, pen_w, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawEllipse(rect.adjusted(6, 6, -6, -6))
        # 头像：内嵌 SVG（小白简化吉祥物：两点眼睛 + 微笑）
        self._draw_avatar(p, rect)

    def _draw_avatar(self, p: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        cx = rect.center().x()
        cy = rect.center().y()
        r = rect.width() / 2 - 14
        # 脸部椭圆
        face = QtGui.QColor("#fdf2ff")
        face.setAlphaF(0.92)
        p.setBrush(face)
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(QtCore.QPointF(cx, cy + r * 0.1), r, r * 1.08)
        # 耳朵（顶部两个小弧）
        ear_color = QtGui.QColor("#fdf2ff")
        p.setBrush(ear_color)
        p.drawEllipse(QtCore.QRectF(cx - r * 0.95, cy - r * 0.7, r * 0.5, r * 0.5))
        p.drawEllipse(QtCore.QRectF(cx + r * 0.45, cy - r * 0.7, r * 0.5, r * 0.5))
        # 眼睛（两个圆点）
        p.setBrush(QtGui.QColor("#111827"))
        ey = cy - r * 0.02
        er = max(1.6, r * 0.08)
        p.drawEllipse(QtCore.QPointF(cx - r * 0.32, ey), er, er)
        p.drawEllipse(QtCore.QPointF(cx + r * 0.32, ey), er, er)
        # 微笑
        pen = QtGui.QPen(QtGui.QColor("#111827"), max(1.2, r * 0.06), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.NoBrush)
        arc_rect = QtCore.QRectF(cx - r * 0.32, cy - r * 0.10, r * 0.64, r * 0.48)
        p.drawArc(arc_rect, 200 * 16, 140 * 16)

    # -------------------------------------------------------------- animation
    def _on_tick(self):
        self._anim_t += 0.03
        self.update()

    # ------------------------------------------------------------- drag / click
    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag_offset = ev.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self._press_pos = ev.position().toPoint()
        elif ev.button() == QtCore.Qt.RightButton:
            self._show_menu(ev.globalPosition().toPoint())

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._drag_offset is not None and ev.buttons() & QtCore.Qt.LeftButton:
            self.move(ev.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton and self._press_pos is not None:
            moved = (ev.position().toPoint() - self._press_pos).manhattanLength()
            self._drag_offset = None
            self._press_pos = None
            if moved <= QtWidgets.QApplication.startDragDistance():
                # 单点击：切换录音
                from .main_window import MainWindow
                mw = self._find_main_window()
                self.toggle_listen(mw)
            else:
                # 释放后吸附到最近左右边缘
                QtCore.QTimer.singleShot(10, self._snap_to_edge)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() != QtCore.Qt.LeftButton:
            return
        mw = self._find_main_window()
        if mw is None:
            # fallback: 用默认浏览器打开
            url = "http://localhost:3020/#/ai"
            try:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
            except Exception as e:  # noqa: BLE001
                log_warn("打开 AI 对话失败: %s", e)
            return
        mw.focus_ai_page()

    # -------------------------------------------------------- helpers / menu
    def _find_main_window(self):
        app = QtWidgets.QApplication.instance()
        for w in app.topLevelWidgets():
            from .main_window import MainWindow
            if isinstance(w, MainWindow):
                return w
        return None

    def _snap_to_edge(self):
        screen = self.screen().availableGeometry() if hasattr(self, "screen") else QtWidgets.QApplication.primaryScreen().availableGeometry()
        geom = self.frameGeometry()
        center_x = geom.center().x()
        if center_x - screen.left() < (screen.right() - center_x):
            target_x = screen.left() + 16
        else:
            target_x = screen.right() - geom.width() - 16
        target_y = max(screen.top() + 16, min(screen.bottom() - geom.height() - 16, geom.top()))
        start_geom = QtCore.QRect(geom)
        end_geom = QtCore.QRect(QtCore.QPoint(target_x, target_y), start_geom.size())
        self._anim_geom(start_geom, end_geom, 300)

    def _anim_geom(self, a: QtCore.QRect, b: QtCore.QRect, ms: int):
        self._anim_start = time.time()
        self._anim_duration_ms = ms
        self._anim_a = a
        self._anim_b = b
        tick = QtCore.QTimer(self)
        ref = {"t": tick}

        def step():
            t = min(1.0, (time.time() - self._anim_start) * 1000 / max(1, self._anim_duration_ms))
            # OutCubic：ease = 1 - (1-t)^3
            ease = 1 - (1 - t) ** 3
            def lerp_i(x, y): return int(x + (y - x) * ease)
            g = QtCore.QRect(
                lerp_i(self._anim_a.left(), self._anim_b.left()),
                lerp_i(self._anim_a.top(), self._anim_b.top()),
                self._anim_a.width(),
                self._anim_a.height(),
            )
            self.setGeometry(g)
            if t >= 1.0:
                ref["t"].stop()
                try:
                    ref["t"].deleteLater()
                except Exception:  # noqa: BLE001
                    pass
        tick.timeout.connect(step)
        tick.start(16)
        tick.startTimer(16)

    def show_toast(self, text: str, ms: int = 1800) -> None:
        if self._toast is None:
            self._toast = _ToastWidget()
        self._toast.show_text(text, ms)

    def _show_menu(self, global_pos: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(self)
        act_open = menu.addAction("打开 AI 对话")
        act_audio = menu.addAction("音色设置…")
        act_mdl = menu.addAction("模型管理…")
        act_cmp = menu.addAction("合规面板…")
        act_startup = menu.addAction("开机自启（S3 暂未启用）")
        act_startup.setEnabled(False)
        act_hk = menu.addMenu("全局快捷键")
        hk_record = act_hk.addAction("Alt+X · 切换录音")
        hk_read = act_hk.addAction("Alt+S · 朗读剪贴板")
        hk_quit = act_hk.addAction("Alt+Q · 退出")
        for a in (hk_record, hk_read, hk_quit):
            a.setEnabled(False)
        menu.addSeparator()
        act_quit = menu.addAction("退出 xiaobai")

        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen is act_open:
            mw = self._find_main_window()
            if mw:
                mw.focus_ai_page()
        elif chosen is act_audio:
            QtWidgets.QMessageBox.information(self, "音色设置",
                                              "音色、情绪、速率设置入口可在对话页“朗读”按钮下拉"
                                              "或桌面小白 T4/T9 配置面板修改。\n（此处占位）")
        elif chosen is act_mdl:
            url = "http://localhost:3020/#/ai"
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
        elif chosen is act_cmp:
            QtWidgets.QMessageBox.information(self, "合规面板",
                                              "可在 ChatView 顶栏点击“合规 φ”Chip 切换许可等级："
                                              "Auto / Research / Apache2。")
        elif chosen is act_quit:
            # 发送退出动作
            app = QtWidgets.QApplication.instance()
            try:
                ctx = getattr(app, "_xiaobai_ctx", None)
                if ctx and hasattr(ctx, "hotkeys"):
                    ctx.hotkeys.stop()
                if ctx and hasattr(ctx, "main_window"):
                    try:
                        ctx.main_window.close()
                    except Exception:  # noqa: BLE001
                        pass
                self.close()
            finally:
                QtCore.QTimer.singleShot(300, lambda: os._exit(0))


# ============================================================ _LocalRecorder


class _LocalRecorder:
    """桌面端本地录音：sounddevice → 文件 → POST /voice/asr/full → recognized。"""

    def __init__(self, ball) -> None:
        self.ball = ball
        self._stream = None
        self._buf = None
        self._sr = 16000
        self.last_recognized = ""

    def start(self, main_window) -> bool:
        try:
            import sounddevice as sd
            import numpy as np
        except Exception as exc:  # noqa: BLE001
            log_warn("sounddevice/numpy 不可用：%s", exc)
            return False
        try:
            self._buf = []
            def cb(indata, frames, t, status):
                if status:
                    return
                self._buf.append(np.array(indata, dtype=np.float32, copy=True))
            self._stream = sd.InputStream(samplerate=self._sr, channels=1, dtype="float32",
                                           blocksize=int(self._sr * 0.1), callback=cb)
            self._stream.start()
            return True
        except Exception as exc:  # noqa: BLE001
            log_warn("录音启动失败：%s", exc)
            return False

    def stop(self) -> str:
        import io
        import numpy as np
        try:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:  # noqa: BLE001
                    pass
                self._stream = None
            if not self._buf:
                self.last_recognized = ""
                return ""
            audio = np.concatenate([b.reshape(-1) for b in self._buf], axis=0).astype(np.float32)
            self._buf = None
            import soundfile as sf
            bio = io.BytesIO()
            sf.write(bio, audio, self._sr, format="WAV", subtype="PCM_16")
            wav_bytes = bio.getvalue()
            # POST /voice/asr/full（先 127.0.0.1:30010，不经过 vite proxy）
            import httpx
            port = getattr(self.ball, "_override_port", None)
            if port is None:
                try:
                    app = QtWidgets.QApplication.instance()
                    ctx = getattr(app, "_xiaobai_ctx", None)
                    port = ctx.port if ctx else 30010
                except Exception:  # noqa: BLE001
                    port = 30010
            try:
                r = httpx.post(f"http://127.0.0.1:{port}/voice/asr/full",
                               files={"file": ("rec.wav", wav_bytes, "audio/wav")}, timeout=30.0)
                r.raise_for_status()
                data = r.json() or {}
                self.last_recognized = str(data.get("text") or "").strip()
            except Exception as exc:  # noqa: BLE001
                log_warn("ASR 请求失败：%s", exc)
                self.last_recognized = ""
            return self.last_recognized or ""
        except Exception as exc:  # noqa: BLE001
            log_warn("录音停止/上传错误：%s", exc)
            self.last_recognized = ""
            return ""


# ========================================================== _ExecWorker (FR-13)

class _ExecSignals(QtCore.QObject):
    """worker 专用信号集合（QThread 不能多继承 QObject 且保留信号定义）。"""
    progress = QtCore.Signal(dict)  # {stage: "routed", route: {...}}
    done = QtCore.Signal(dict)      # OperatorResult.to_dict() + text


class _ExecWorker(QtCore.QThread):
    """桌面端异步执行 PPR 路由 + 算子执行（VoiceProxyClient.dispatch_intent
    或直接 OperatorEngine.dispatch），避免 UI 卡顿。
    """

    def __init__(self, text: str, engine, proxy, identity, parent=None) -> None:
        super().__init__(parent)
        self.text = str(text)
        self.engine = engine            # OperatorEngine 或 None
        self.proxy = proxy              # VoiceProxyClient 或 None
        self.identity = identity
        self.signals = _ExecSignals()

    def run(self) -> None:  # noqa: D401 （Qt 约定 run 是入口）
        t0 = time.perf_counter()
        try:
            result_dict = self._run_inner()
        except Exception as exc:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc(limit=2)
            log_warn("执行异常：%s\n%s", exc, tb)
            result_dict = {
                "op": "", "act": "", "ok": False,
                "code": "OPERATOR_FAILED",
                "message": f"{type(exc).__name__}: {exc}",
                "data": {"traceback": tb},
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                "audit_id": "",
            }
        # 附带原始文本字段，方便上层
        result_dict.setdefault("text", self.text)
        result_dict.setdefault("executed", True)
        self.signals.done.emit(result_dict)

    def _run_inner(self) -> dict:
        # 有 voice_proxy → 三策略路由（local_first/cloud_fallback/cloud_only）
        if self.proxy is not None:
            try:
                import asyncio
                # 在新线程里独立 event loop
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        r = loop.run_until_complete(self.proxy.dispatch_intent(self.text))
                    finally:
                        loop.close()
                finally:
                    try:
                        asyncio.set_event_loop(None)
                    except Exception:  # noqa: BLE001
                        pass
                # 构造 summary（便于 Ball toast 显示）
                d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
                self._inject_summary(d)
                return d
            except Exception as exc:  # noqa: BLE001
                # proxy 失败：若有 engine，engine 兜底
                if self.engine is None:
                    raise
                log_warn("voice_proxy 失败，降级为本地 engine：%s", exc)

        # 纯本地 engine：PPR 路由 → dispatch
        if self.engine is None:
            return {
                "op": "", "act": "", "ok": False,
                "code": "OPERATOR_UNSUPPORTED",
                "message": "未注入 engine/proxy",
                "duration_ms": 0, "audit_id": "",
            }
        from ..intent.router import IntentRouter
        router = IntentRouter()
        route = router.route(self.text, self.identity)
        self.signals.progress.emit({"stage": "routed", "route": route.as_dict()})
        if not route.op_name or not route.act:
            return {
                "op": "", "act": "", "ok": False,
                "code": "INTENT_UNKNOWN",
                "message": (
                    "本地意图未命中"
                    + (f"；候选：{route.candidates[:3]}" if route.candidates else "")
                ),
                "data": {"route": route.as_dict()},
                "duration_ms": 0, "audit_id": "",
            }
        r = self.engine.dispatch(route.op_name, route.act, route.params, identity=self.identity)
        d = r.to_dict()
        self._inject_summary(d, route=route)
        return d

    @staticmethod
    def _inject_summary(d: dict, route=None) -> None:
        if d.get("data") is None:
            d["data"] = {}
        if d["data"].get("summary"):
            return
        ok = d.get("ok")
        op = d.get("op") or (route.op_name if route else "")
        act = d.get("act") or (route.act if route else "")
        op_act = f"{op}.{act}"
        if ok:
            _SUMMARY = {
                "app.open_app": "已启动应用",
                "app.close_app": "已关闭应用",
                "app.list_running": "进程列表已获取",
                "app.open_file_with_app": "已打开文件",
                "volume.get_volume": "已读取音量",
                "volume.set_volume": "已设置音量",
                "volume.mute": "已静音",
                "volume.unmute": "已取消静音",
                "volume.toggle_mute": "已切换静音",
                "file.copy_to_clipboard": "已复制到剪贴板",
                "file.move_to_trash": "已丢入回收站",
                "file.read_text_head": "已读取文件片段",
                "input.type_text": "已输入文本",
                "input.press_key": "已按键",
                "input.hotkey": "已执行快捷键",
                "input.mouse_move": "已移动鼠标",
                "input.mouse_click": "已点击鼠标",
                "input.mouse_drag": "已拖拽鼠标",
                "input.screenshot": "已截图",
            }
            d["data"]["summary"] = _SUMMARY.get(op_act, f"{op_act} 执行完成")
        else:
            d["data"]["summary"] = f"{op_act} 失败"


# ================================================================= Toast


class _ToastWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self._label = QtWidgets.QLabel(self)
        self._label.setStyleSheet("""
            color: #e6eaff; background: rgba(10,14,40,0.92);
            border: 1px solid rgba(129,140,248,0.35);
            padding: 10px 18px; border-radius: 18px;
            font: 13px 'Microsoft YaHei','PingFang SC',sans-serif;
        """)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_text(self, text: str, ms: int) -> None:
        self._label.setText(text)
        self.resize(self._label.sizeHint())
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 32, screen.bottom() - self.height() - 48)
        self.show()
        self.raise_()
        self._timer.start(max(500, int(ms)))


def log_warn(msg: str, *args):
    try:
        logging.getLogger("xiaobai.desktop").warning(msg, *args)
    except Exception:  # noqa: BLE001
        pass

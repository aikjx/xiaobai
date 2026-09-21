"""桌面主窗口：内嵌 QWebEngineView 打开 /#/ai；4 Chip（ASR/TTS/合规/快捷键）；粘贴音频转文本；服务未启动 φ 友好启动页。"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets


log = logging.getLogger("xiaobai.desktop.main")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *, loader, default_url: str, port: int):
        super().__init__()
        self.loader = loader
        self.port = int(port or 30010)
        self.default_url = default_url
        self.setWindowTitle("小白 xiaobai · 璇玑 AI 助手")
        self.resize(1200, 780)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # 顶栏 4 Chip
        self.chip_asr = self._chip("ASR引擎", "检测中…", "#22d3ee")
        self.chip_tts = self._chip("TTS引擎", "检测中…", "#a855f7")
        self.chip_cmp = self._chip("合规", "Auto", "#6366f1")
        self.chip_hk  = self._chip("快捷键", "Alt+X / Alt+S / Alt+Q", "#22c55e")
        chip_row = QtWidgets.QHBoxLayout()
        chip_row.setSpacing(8)
        for c in (self.chip_asr, self.chip_tts, self.chip_cmp, self.chip_hk):
            chip_row.addWidget(c, 0)
        chip_row.addStretch(1)
        lay.addLayout(chip_row)

        # 核心视图栈：服务未启动页 + WebView 页
        self.stack = QtWidgets.QStackedWidget()
        lay.addWidget(self.stack, 1)

        # --- 启动页 ---
        self.page_stub = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(self.page_stub)
        sv.setContentsMargins(0, 0, 0, 0)
        phi = QtWidgets.QLabel("φ")
        phi.setStyleSheet("font: 72px 'Segoe UI'; color: #818cf8; qproperty-alignment: AlignCenter;")
        phi.setAlignment(QtCore.Qt.AlignCenter)
        title = QtWidgets.QLabel("服务未启动 φ")
        title.setStyleSheet("font: 28px 'Microsoft YaHei'; color: #e6eaff; qproperty-alignment: AlignCenter;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        tip = QtWidgets.QLabel("桌面小白需要同时运行前端（http://localhost:3020/#/ai）和语音服务（30010）。\n请选择：")
        tip.setStyleSheet("color: #9aa4cf; font: 14px 'Microsoft YaHei'; qproperty-alignment: AlignCenter;")
        tip.setAlignment(QtCore.Qt.AlignCenter)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        b_srv = QtWidgets.QPushButton("启动语音服务")
        b_srv.setMinimumHeight(42)
        b_srv.setCursor(QtCore.Qt.PointingHandCursor)
        b_browser = QtWidgets.QPushButton("打开浏览器 AI 对话")
        b_browser.setMinimumHeight(42)
        b_browser.setCursor(QtCore.Qt.PointingHandCursor)
        for b in (b_srv, b_browser):
            b.setStyleSheet("""
                QPushButton{background:#1e2862;color:#e6eaff;border:1px solid rgba(129,140,248,0.45);
                border-radius:18px;padding:6px 22px;font:14px 'Microsoft YaHei';}
                QPushButton:hover{background:#2a3680;}
            """)
        btn_row.addWidget(b_srv)
        btn_row.addSpacing(12)
        btn_row.addWidget(b_browser)
        btn_row.addStretch(1)
        sv.addSpacing(80)
        sv.addWidget(phi)
        sv.addSpacing(12)
        sv.addWidget(title)
        sv.addSpacing(16)
        sv.addWidget(tip)
        sv.addSpacing(28)
        sv.addLayout(btn_row)
        sv.addStretch(1)
        self.stack.addWidget(self.page_stub)
        b_srv.clicked.connect(self._ensure_voice_service)
        b_browser.clicked.connect(self._open_browser)

        # --- WebView 页 ---
        self.page_web = QtWidgets.QWidget()
        wl = QtWidgets.QVBoxLayout(self.page_web)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(6)
        # 尝试加载 QWebEngineView（PySide6 可选子模块）
        self.view = None
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore
            self.view = QWebEngineView(self)
            self.view.settings().setAttribute(self.view.settings().WebAttribute.LocalContentCanAccessRemoteUrls, True)
            self.view.setUrl(QtCore.QUrl(self.default_url))
            # 暴露 window.__xiaobai_paste(text, role)
            class _Bridge(QtCore.QObject):
                pass
            self._bridge = _Bridge(self)
            self.view.loadFinished.connect(self._on_web_loaded)
            self.view.installEventFilter(self)
        except Exception as exc:  # noqa: BLE001
            label = QtWidgets.QLabel(
                "未安装 PySide6 WebEngine（pip install PySide6-WebEngine）。\n请使用下面的按钮打开浏览器版本。"
            )
            label.setStyleSheet("color:#9aa4cf;padding:24px;font:14px 'Microsoft YaHei';")
            label.setAlignment(QtCore.Qt.AlignCenter)
            btn = QtWidgets.QPushButton("打开浏览器 AI 对话")
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.clicked.connect(self._open_browser)
            box = QtWidgets.QVBoxLayout()
            box.addWidget(label); box.addWidget(btn, alignment=QtCore.Qt.AlignHCenter)
            wrap = QtWidgets.QWidget()
            wrap.setLayout(box)
            self.stack_web_placeholder = wrap
            wl.addWidget(wrap)
        if self.view is not None:
            wl.addWidget(self.view)
        # 底部粘贴音频转文本提示条
        info = QtWidgets.QLabel("支持把音频文件拖到或粘贴（Ctrl+V）到此窗口，自动识别为文本并填充到 AI 对话输入框。")
        info.setStyleSheet("color:#818cf8;background:rgba(129,140,248,0.08);border:1px solid rgba(129,140,248,0.25);padding:8px 14px;border-radius:14px;font:12px 'Microsoft YaHei';")
        wl.addWidget(info)
        self.stack.addWidget(self.page_web)

        # 周期健康检测（决定显示哪个页面 + 更新 Chip）
        self._tick = QtCore.QTimer(self)
        self._tick.timeout.connect(self._health_tick)
        self._tick.start(2500)
        QtCore.QTimer.singleShot(300, self._health_tick)
        self.setAcceptDrops(True)

    # --------------------------------------------------------------------- UI
    def _chip(self, label: str, value: str, color: str) -> QtWidgets.QLabel:
        w = QtWidgets.QLabel(f"{label} · <span style='color:{color}'>{value}</span>")
        w.setTextFormat(QtCore.Qt.RichText)
        w.setStyleSheet(f"""
            QLabel {{
                background: rgba(26,34,78,0.85);
                border: 1px solid rgba(129,140,248,0.35);
                color: #c7d2fe;
                border-radius: 999px;
                padding: 8px 16px;
                font: 13px 'Microsoft YaHei','PingFang SC',sans-serif;
            }}
        """)
        return w

    # ---------------------------------------------------------- health tick
    def _health_tick(self) -> None:
        import httpx
        try:
            r = httpx.get(f"http://127.0.0.1:{self.port}/voice/health", timeout=1.5)
            r.raise_for_status()
            data = r.json() or {}
            tier = data.get("license_tier") or "auto"
            asr_ok = bool((data.get("asr") or {}).get("available"))
            tts_ok = bool((data.get("tts") or {}).get("available"))
            asr_name = (data.get("asr") or {}).get("engine") or "n/a"
            tts_name = (data.get("tts") or {}).get("engine") or "n/a"
            color_asr = "#22c55e" if asr_ok else "#ef4444"
            color_tts = "#a855f7" if tts_ok else "#ef4444"
            color_cmp = "#6366f1" if tier != "apache2" else "#22d3ee"
            self.chip_asr.setText(f"ASR引擎 · <span style='color:{color_asr}'>{asr_name}</span>")
            self.chip_tts.setText(f"TTS引擎 · <span style='color:{color_tts}'>{tts_name}</span>")
            self.chip_cmp.setText(f"合规 · <span style='color:{color_cmp}'>{tier}</span>")
            # 同时检测前端 /#/ai 端口是否在（3020）
            frontend_ok = True
            try:
                httpx.get("http://127.0.0.1:3020/", timeout=1.0)
            except Exception:  # noqa: BLE001
                frontend_ok = False
            if asr_ok and frontend_ok and self.view is not None:
                self.stack.setCurrentWidget(self.page_web)
            elif asr_ok and frontend_ok:
                self.stack.setCurrentWidget(self.page_stub)
            else:
                self.stack.setCurrentWidget(self.page_stub)
        except Exception as exc:  # noqa: BLE001
            log.debug("health tick fail: %s", exc)
            for chip, label, def_col in (
                (self.chip_asr, "ASR引擎 · <span style='color:#ef4444'>离线</span>", None),
                (self.chip_tts, "TTS引擎 · <span style='color:#ef4444'>离线</span>", None),
                (self.chip_cmp, None, None),
            ):
                if label is not None:
                    chip.setText(label)
            self.stack.setCurrentWidget(self.page_stub)

    # --------------------------------------------------------- focus / open
    def focus_ai_page(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()
        if self.view is not None:
            self.stack.setCurrentWidget(self.page_web)
            self.view.setFocus()
            self.view.setUrl(QtCore.QUrl(self.default_url))

    def _open_browser(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.default_url))

    def _ensure_voice_service(self) -> None:
        import subprocess
        import sys
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if getattr(sys, "frozen", False):
            args = [sys.executable, "serve", "--port", str(self.port)]
        else:
            args = [sys.executable, "-m", "xiaobai_voice", "serve", "--port", str(self.port)]
        try:
            subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
            QtWidgets.QMessageBox.information(self, "已启动", "语音服务正在启动，请稍候数秒。")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "启动失败", str(exc))

    # -------------------------------------------------------- paste asr file
    def eventFilter(self, obj, ev):  # noqa: N802
        if ev.type() == QtCore.QEvent.KeyPress and obj is self.view:
            if ev.key() == QtCore.Qt.Key_V and (ev.modifiers() & QtCore.Qt.ControlModifier):
                md = QtWidgets.QApplication.clipboard().mimeData()
                if md.hasUrls() or md.hasImage() or md.hasFormat("application/octet-stream"):
                    self._handle_mime(md)
                    return True
        return super().eventFilter(obj, ev)

    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev: QtGui.QDropEvent) -> None:
        if ev.mimeData().hasUrls():
            self._handle_paths([u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()])

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        if ev.key() == QtCore.Qt.Key_V and (ev.modifiers() & QtCore.Qt.ControlModifier):
            md = QtWidgets.QApplication.clipboard().mimeData()
            if md.hasUrls():
                self._handle_paths([u.toLocalFile() for u in md.urls() if u.isLocalFile()])
                return
        super().keyPressEvent(ev)

    def _handle_mime(self, md):
        if md.hasUrls():
            self._handle_paths([u.toLocalFile() for u in md.urls() if u.isLocalFile()])

    def _handle_paths(self, paths):
        audio_ext = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
        audios = [p for p in paths if p and Path(p).is_file() and Path(p).suffix.lower() in audio_ext]
        if not audios:
            return
        file = audios[0]
        import httpx
        try:
            with open(file, "rb") as f:
                r = httpx.post(f"http://127.0.0.1:{self.port}/voice/asr/full",
                               files={"file": (Path(file).name, f, "audio/wav")}, timeout=30.0)
            r.raise_for_status()
            text = (r.json() or {}).get("text") or ""
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "识别失败", str(exc))
            return
        self.paste_recognized_text(text)

    # --------------------------------------------------------- recognized
    def paste_recognized_text(self, text: str) -> None:
        if not text:
            return
        if self.view is None:
            # 把文本写到剪贴板 + 打开浏览器
            QtWidgets.QApplication.clipboard().setText(text)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(self.default_url))
            return
        role = "asr_file"
        # 安全转义
        escaped = json.dumps(text, ensure_ascii=False)
        js = (
            "(function(){"
            "if (typeof window.__xiaobai_paste === 'function') { return window.__xiaobai_paste(%s, '%s'); }"
            "var hints = document.querySelectorAll('textarea, input[type=text]');"
            "var ta = null;"
            "for (var i=hints.length-1;i>=0;i--){ if (hints[i].offsetParent && hints[i].clientWidth > 200){ ta=hints[i]; break; } }"
            "if (ta){ ta.focus(); ta.value = %s; ta.dispatchEvent(new Event('input',{bubbles:true})); return true; }"
            "return false;"
            "})()"
        ) % (escaped, role, escaped)
        try:
            self.view.page().runJavaScript(js)
        except Exception as exc:  # noqa: BLE001
            log.warning("paste js error: %s", exc)

    def play_text_via_voice(self, text: str) -> None:
        """桌面端朗读：先调 /voice/tts/stream 拿 WAV，用本地 sounddevice 播放。"""
        if not text:
            return
        try:
            import sounddevice as sd
            import soundfile as sf
            import httpx
            import io
            r = httpx.get(f"http://127.0.0.1:{self.port}/voice/tts/stream",
                          params={"text": text[:500], "voice": "xiaobai", "emotion": "neutral", "speed": 1.0},
                          timeout=60.0)
            r.raise_for_status()
            if r.headers.get("X-TTS-Fallback") == "browser":
                QtWidgets.QMessageBox.information(self, "提示",
                    "当前仅浏览器 TTS 可用，请在 AI 对话页使用朗读按钮。")
                return
            data, sr = sf.read(io.BytesIO(r.content), always_2d=False, dtype="float32")
            sd.play(data, sr)
        except Exception as exc:  # noqa: BLE001
            log.warning("桌面朗读失败：%s", exc)
            QtWidgets.QMessageBox.warning(self, "朗读失败", str(exc))

    def _on_web_loaded(self, ok: bool):
        # 注入 window.__xiaobai_paste 给桌面粘贴音频识别回填用
        if not ok or self.view is None:
            return
        js = """
        (function(){
            window.__xiaobai_paste = function(text, role){
                text = String(text || '');
                var ta = null;
                var all = document.querySelectorAll('textarea, input[type=text]');
                for (var i=all.length-1;i>=0;i--){
                    var el = all[i];
                    if (el.offsetParent && el.clientWidth > 200){ ta = el; break; }
                }
                if (!ta) return false;
                ta.focus();
                try{
                    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype || HTMLInputElement.prototype, 'value').set;
                    setter.call(ta, text);
                }catch(e){
                    ta.value = text;
                }
                ta.dispatchEvent(new Event('input', {bubbles:true}));
                ta.dispatchEvent(new Event('change', {bubbles:true}));
                if (role === 'asr_file'){
                    try{
                        if (typeof window.Event !== 'undefined'){
                            var ev = new KeyboardEvent('keydown', {key:'Enter', code:'Enter', metaKey:false, bubbles:true});
                            ta.dispatchEvent(ev);
                        }
                    }catch(e){}
                }
                return true;
            };
        })();
        """
        try:
            self.view.page().runJavaScript(js)
        except Exception:  # noqa: BLE001
            pass

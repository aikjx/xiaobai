"""FastAPI 语音服务（端口 30010 默认）。路由矩阵：/voice/health /voice/models /voice/models/download /voice/models/download/stream(SSE) /voice/ws/asr/stream /voice/asr/full /voice/tts/stream /voice/tts/clone /voice/hotwords(GET+POST) /voice/license_tier /voice/metrics。兼容别名：/voice/v1/*。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ..asr import ASRBackend, build_asr_backend
from ..asr.base import ASRFullResult, ASRPartial
from ..config.loader import ConfigLoader, default_log_path
from ..errors import ErrorCode, XiaobaiError
from ..models import ModelDownloader, ModelRegistry
from ..tts import TTSBackend, TTSOptions, build_tts_backend

log = logging.getLogger("xiaobai.service")

_lifecycle: "EngineLifecycle | None" = None
_downloader: ModelDownloader | None = None
_loader: ConfigLoader | None = None
_registry: ModelRegistry | None = None


class EngineLifecycle:
    def __init__(self, loader: ConfigLoader, registry: ModelRegistry) -> None:
        self.loader = loader
        self.registry = registry
        self.lock = threading.RLock()
        self.asr: ASRBackend | None = None
        self.tts: TTSBackend | None = None
        self.license_tier: str = "auto"
        self.started_at = time.time()
        self.stats: dict[str, Any] = {"asr_restarts": 0, "tts_restarts": 0, "asr_req": 0, "tts_req": 0}
        self._smoke_log_path: Path | None = None
        self._prepare_smoke_log()
        self.loader._on_change = self._on_config_change  # 热更新回调
        self.build_all(prewarm=True)

    def _prepare_smoke_log(self) -> None:
        log_dir = default_log_path()
        stamp = time.strftime("%Y%m%d")
        self._smoke_log_path = log_dir / f"smoke_{stamp}.jsonl"

    def _append_smoke(self, entry: dict) -> None:
        if not self._smoke_log_path:
            return
        try:
            with open(self._smoke_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def build_all(self, *, prewarm: bool = False) -> None:
        cfg = self.loader.data
        tier = str(((cfg.get("voice") or {}).get("license_tier")) or "auto").lower()
        self.license_tier = tier
        with self.lock:
            try:
                old = self.asr
                self.asr = build_asr_backend(cfg, tier, self.registry)
                if old is not None:
                    try:
                        old.close()
                    except Exception:  # noqa: BLE001
                        pass
                self.stats["asr_restarts"] += 1
            except XiaobaiError as exc:
                self.asr = None
                log.warning("ASR 初始化失败 %s：%s", exc.code.value, exc.message)
                self._append_smoke(dict(ts=time.time(), phase="asr_init", code=exc.code.value, message=exc.message))
            except Exception as exc:  # noqa: BLE001
                # 抽象类未实现 / TypeError / ImportError 等未分级错误，统一降级避免阻塞 TTS
                self.asr = None
                log.warning("ASR 初始化失败（非分级异常）%s：%s", type(exc).__name__, exc)
                self._append_smoke(dict(ts=time.time(), phase="asr_init", code="ERR",
                                        message=f"{type(exc).__name__}: {exc}"))
            try:
                old = self.tts
                self.tts = build_tts_backend(cfg, tier, self.registry)
                if old is not None:
                    try:
                        old.close()
                    except Exception:  # noqa: BLE001
                        pass
                self.stats["tts_restarts"] += 1
            except XiaobaiError as exc:
                self.tts = None
                log.warning("TTS 初始化失败 %s：%s", exc.code.value, exc.message)
                self._append_smoke(dict(ts=time.time(), phase="tts_init", code=exc.code.value, message=exc.message))
            except Exception as exc:  # noqa: BLE001
                self.tts = None
                log.warning("TTS 初始化失败（非分级异常）%s：%s", type(exc).__name__, exc)
                self._append_smoke(dict(ts=time.time(), phase="tts_init", code="ERR",
                                        message=f"{type(exc).__name__}: {exc}"))
        if prewarm:
            self._prewarm_smoke()

    def _prewarm_smoke(self) -> None:
        asr_ms = 0.0
        if self.asr:
            try:
                t0 = time.time()
                asr_ms = self.asr.prewarm()
                duration = round(max(asr_ms, (time.time() - t0) * 1000), 1)
                self._append_smoke(
                    dict(
                        ts=time.time(),
                        phase="asr_prewarm",
                        code="OK",
                        message=f"sherpa-onnx paraformer-int8 ready in {duration:.0f} ms",
                        duration_ms=duration,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._append_smoke(dict(ts=time.time(), phase="asr_prewarm", code="ERR", message=str(exc)))
        if self.tts:
            try:
                t0 = time.time()
                opts = TTSOptions(text="你好，璇玑。", voice="xiaobai", emotion="neutral", speed=1.0, sample_rate=16000)
                _ = self.tts.synthesize_full(opts)
                duration = round((time.time() - t0) * 1000, 1)
                self._append_smoke(dict(ts=time.time(), phase="tts_prewarm", code="OK", engine=self.tts.name, duration_ms=duration))
            except Exception as exc:  # noqa: BLE001
                self._append_smoke(dict(ts=time.time(), phase="tts_prewarm", code="ERR", message=str(exc)))

    def _on_config_change(self, _new_cfg: dict) -> None:
        self.build_all(prewarm=False)
        log.info("config 热更新完成，已重建 ASR/TTS。")

    def asr_engine_name(self) -> str:
        return (self.asr and self.asr.name) or "unavailable"

    def tts_engine_name(self) -> str:
        return (self.tts and self.tts.name) or "unavailable"

    def uptime_s(self) -> float:
        return time.time() - self.started_at


def get_lifecycle() -> EngineLifecycle:
    assert _lifecycle is not None
    return _lifecycle


def _temp_root() -> str:
    return os.environ.get("TMP") or os.environ.get("TEMP") or "/tmp"


def _temp_cleanup_loop() -> None:  # pragma: no cover
    import shutil

    temp = Path(os.environ.get("MOX_TEMP") or os.path.join(_temp_root(), "mox_voice"))
    temp.mkdir(parents=True, exist_ok=True)
    while True:
        time.sleep(60.0)
        try:
            mins = int((_loader and _loader.get("voice.temp_cleanup_minutes")) or 10)
            cutoff = time.time() - mins * 60.0
            for p in temp.glob("*"):
                try:
                    if p.stat().st_mtime < cutoff:
                        if p.is_dir():
                            shutil.rmtree(p, ignore_errors=True)
                        else:
                            p.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            log.exception("temp cleanup")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    global _lifecycle, _loader, _registry, _downloader
    cfg_path_env = os.environ.get("MOX_VOICE_CONFIG") or None
    _loader = ConfigLoader(Path(cfg_path_env) if cfg_path_env else None, watch=True)
    _registry = ModelRegistry()
    _downloader = ModelDownloader(_registry)
    _lifecycle = EngineLifecycle(_loader, _registry)
    threading.Thread(target=_temp_cleanup_loop, name="xiaobai-temp-cleaner", daemon=True).start()
    try:
        yield
    finally:
        if _lifecycle.asr:
            try:
                _lifecycle.asr.close()
            except Exception:  # noqa: BLE001
                pass
        if _lifecycle.tts:
            try:
                _lifecycle.tts.close()
            except Exception:  # noqa: BLE001
                pass


def create_app() -> FastAPI:
    app = FastAPI(title="Xiaobai Voice Service", version="0.1.0", lifespan=lifespan, docs_url="/voice/docs", redoc_url="/voice/redoc")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False)
    _bind_routes(app, prefix="/voice")
    _bind_routes(app, prefix="/voice/v1")
    return app


def _status_to_dict(s) -> dict:
    return dict(id=s.id, name=s.name, license=s.license, size_mb=s.size_mb, downloaded=s.downloaded, sha256_ok=s.sha256_ok, local_root=s.local_root, engine=s.engine, category=s.category, optional=s.optional)


def _dc2dict(obj: Any) -> dict:
    if is_dataclass(obj):
        return asdict(obj)
    return obj if isinstance(obj, dict) else {"value": obj}


def _partial_to_dict(p: ASRPartial) -> dict:
    return dict(type=("final" if p.is_final else "partial"), text=p.text, is_final=p.is_final, start_ms=p.start_ms, end_ms=p.end_ms, confidence=getattr(p, "confidence", 0.0), language=p.language)


_progress_listeners: dict[str, list] = {}
_pl_lock = threading.Lock()


def _add_progress_listener(model_id: str, cb) -> None:
    with _pl_lock:
        _progress_listeners.setdefault(model_id, []).append(cb)


def _remove_progress_listener(model_id: str, cb) -> None:
    with _pl_lock:
        arr = _progress_listeners.get(model_id) or []
        if cb in arr:
            arr.remove(cb)


def _broadcast_progress(event: dict) -> None:
    mid = event.get("model_id")
    if not mid:
        return
    with _pl_lock:
        cbs = list(_progress_listeners.get(mid) or [])
    for cb in cbs:
        try:
            cb(event)
        except Exception:  # noqa: BLE001
            pass


def _bind_routes(app: FastAPI, prefix: str) -> None:
    r = APIRouter(prefix=prefix)

    @r.get("/health")
    async def health():
        lc = get_lifecycle()
        fish_available = os.environ.get("XIAOBAI_ACCEPT_RESEARCH_LICENSE", "0") == "1"
        tts_name = lc.tts_engine_name()
        cosy_ok = tts_name == "cosyvoice2"
        fish_ok = tts_name in {"fish_s2", "fish_s2_pro", "fish"} and fish_available
        browser_ok = tts_name == "browser"
        # Rust DSP 探测：CosyVoice2Backend 暴露 rust_dsp_available() 函数
        rust_dsp_ok = False
        try:
            from ..tts.cosyvoice2 import rust_dsp_available as _rda
            rust_dsp_ok = bool(_rda())
        except Exception:  # noqa: BLE001
            rust_dsp_ok = False
        engines = [
            dict(name="cosyvoice2", available=cosy_ok or (lc.tts is not None and not browser_ok and not fish_ok), license="Apache-2.0", rust_dsp_available=rust_dsp_ok),
            dict(
                name="fish_s2_pro",
                available=fish_available and fish_ok,
                license="Research",
                note="默认禁用，需手动接受 Research License 后启用",
            ),
            dict(name="browser", available=browser_ok, license="Builtin"),
        ]
        return JSONResponse(dict(
            ok=True,
            asr=dict(
                ready=lc.asr is not None,
                model="paraformer-zh",
                backend="sherpa-onnx",
            ),
            tts=dict(
                ready=lc.tts is not None,
                engines=engines,
                active=tts_name,
                rust_dsp_available=rust_dsp_ok,
            ),
            endpoints=dict(
                asr_full="/voice/asr/full",
                tts_stream="/voice/tts/stream",
                ws_asr_stream="/voice/ws/asr/stream",
            ),
        ))

    @r.get("/models")
    async def list_models():
        reg = get_lifecycle().registry
        return JSONResponse(dict(version=reg.version, models=[_status_to_dict(s) for s in reg.list_all()]))

    @r.post("/models/download")
    async def download_trigger(req: Request):
        try:
            body = await req.json()
        except Exception:
            body = {}
        model_id = str(body.get("model_id") or "")
        defaults = bool(body.get("defaults"))
        if not model_id and not defaults:
            raise HTTPException(400, "需要 model_id 或 defaults=true")
        lc = get_lifecycle()
        assert _downloader is not None
        targets: list[str] = []
        if defaults:
            tier = lc.license_tier
            for m in lc.registry.list_all():
                if m.optional:
                    continue
                if m.id == "tts-fish-s2-pro" and tier == "apache2":
                    continue
                targets.append(m.id)
        else:
            targets = [model_id]
        for mid in targets:
            _downloader.download(mid, on_progress=_broadcast_progress)
        lc.build_all(prewarm=False)
        return JSONResponse(dict(status="ok", downloaded=targets))

    @r.get("/models/download/stream")
    async def download_stream(model_id: str = Query(...)):
        async def gen():
            queue: asyncio.Queue = asyncio.Queue(maxsize=32)

            def on_p(event):
                try:
                    queue.put_nowait(event)
                except Exception:  # noqa: BLE001
                    pass

            _add_progress_listener(model_id, on_p)
            try:
                while True:
                    try:
                        ev = await asyncio.wait_for(queue.get(), timeout=30.0)
                    except TimeoutError:
                        yield "event: keepalive\ndata: {}\n\n"
                        continue
                    yield "event: progress\ndata: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                    if ev.get("state") in ("done", "error"):
                        break
            finally:
                _remove_progress_listener(model_id, on_p)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @r.websocket("/ws/asr/stream")
    async def ws_asr_stream(ws: WebSocket):
        await ws.accept()
        lc = get_lifecycle()
        if lc.asr is None:
            await ws.send_json(dict(type="error", code=ErrorCode.MISSING_DEP.value, message="ASR 未初始化。请下载模型。"))
            await ws.close(code=1011)
            return

        async def chunks():
            while True:
                try:
                    data = await asyncio.wait_for(ws.receive_bytes(), timeout=60.0)
                except TimeoutError:
                    break
                except WebSocketDisconnect:
                    break
                if not data:
                    break
                yield data

        try:
            async for part in lc.asr.recognize_stream(chunks()):
                await ws.send_json(_partial_to_dict(part))
        except XiaobaiError as exc:
            await ws.send_json(dict(type="error", **exc.to_dict()))
        except Exception as exc:  # noqa: BLE001
            log.exception("ASR stream error")
            await ws.send_json(dict(type="error", code=ErrorCode.RUNTIME.value, message=str(exc)))
        finally:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    @r.post("/asr/full")
    async def asr_full(file: UploadFile = File(...), fmt: str = Query("wav")):
        lc = get_lifecycle()
        if lc.asr is None:
            raise HTTPException(503, "ASR 未初始化。请下载模型。")
        data = await file.read()
        try:
            res: ASRFullResult = await lc.asr.recognize_full(data, fmt=fmt)
        except XiaobaiError as exc:
            raise HTTPException(status_code=500, detail=exc.to_dict()) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, str(exc)) from exc
        lc.stats["asr_req"] = lc.stats.get("asr_req", 0) + 1
        return JSONResponse(_dc2dict(res))

    @r.get("/tts/stream")
    async def tts_stream(
        text: str = Query(..., min_length=1),
        voice: str = Query("xiaobai"),
        emotion: str = Query("neutral"),
        speed: float = Query(1.0, ge=0.5, le=2.0),
    ):
        return await _tts_stream_impl(text=text, voice=voice, emotion=emotion, speed=speed)

    @r.post("/tts/stream", response_model=None)
    async def tts_stream_post(
        request: Request,
        text: str | None = Query(None),
        voice: str = Query("xiaobai"),
        emotion: str = Query("neutral"),
        speed: float = Query(1.0, ge=0.5, le=2.0),
    ):
        # 双保险：同时兼容 query string 与 form-body / JSON-body（旧版前端 POST form 提交不会 404/405）
        text_q: str | None = text
        voice_q: str = voice
        emotion_q: str = emotion
        speed_q: float = speed
        try:
            ctype = (request.headers.get("content-type") or "").lower()
            if "application/json" in ctype:
                try:
                    payload = await request.json() or {}
                except Exception:  # noqa: BLE001
                    payload = {}
                if isinstance(payload, dict):
                    text_q = str(payload.get("text") or text_q or "")
                    voice_q = str(payload.get("voice") or voice_q)
                    emotion_q = str(payload.get("emotion") or emotion_q)
                    try:
                        speed_q = float(payload.get("speed") or speed_q)
                    except Exception:  # noqa: BLE001
                        pass
            else:
                # form-urlencoded / multipart
                try:
                    formdata = await request.form()
                except Exception:  # noqa: BLE001
                    formdata = None
                if formdata is not None and hasattr(formdata, "get"):
                    text_q = str(formdata.get("text") or text_q or "")
                    voice_q = str(formdata.get("voice") or voice_q)
                    emotion_q = str(formdata.get("emotion") or emotion_q)
                    try:
                        speed_q = float(formdata.get("speed") or speed_q)
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass
        return await _tts_stream_impl(
            text=text_q or "",
            voice=voice_q,
            emotion=emotion_q,
            speed=speed_q,
        )

    async def _tts_stream_impl(*, text: str, voice: str, emotion: str, speed: float):
        lc = get_lifecycle()
        if lc.tts is None:
            raise HTTPException(503, "TTS 后端未初始化。请下载 CosyVoice2 或切换 license_tier 允许 Fish。")
        text_ok = str(text or "").strip()
        if not text_ok:
            raise HTTPException(400, "text 为空")
        emotion_ok = emotion if emotion in {"neutral", "happy", "sad", "serious"} else "neutral"
        try:
            speed_ok = float(speed or 1.0)
        except Exception:  # noqa: BLE001
            speed_ok = 1.0
        speed_ok = max(0.5, min(2.0, speed_ok))
        sr = int(((lc.loader.data.get("voice") or {}).get("tts") or {}).get("sample_rate") or 22050)
        clone_ref = (((lc.loader.data.get("voice") or {}).get("tts") or {}).get("clone_reference"))
        # 尊重配置的默认 speed（default_config.yaml 里 voice.tts.speed = 1.03）
        default_speed = ((lc.loader.data.get("voice") or {}).get("tts") or {}).get("speed")
        try:
            if default_speed is not None and abs(float(default_speed) - 1.0) > 1e-3 and abs(speed_ok - 1.0) < 1e-3:
                # 前端/调用方未显式改 speed 时，使用配置默认值
                speed_ok = max(0.5, min(2.0, float(default_speed)))
        except Exception:  # noqa: BLE001
            pass
        opts = TTSOptions(
            text=text_ok,
            voice=str(voice or "xiaobai"),
            emotion=emotion_ok,
            speed=speed_ok,
            sample_rate=sr,
            clone_reference=clone_ref,
        )
        headers: dict[str, str] = {"X-TTS-Engine": lc.tts.name}
        media_type = "audio/wav"
        if lc.tts.name == "browser":
            headers["X-TTS-Fallback"] = "browser"

        try:
            # 先做一次"同步探测"取出 DSP impl。注意：TTSBackend 提供的是生成器 API，
            # synthesize_full 会完整跑完一次拿到 bytes；这里用 full 既简单又可靠，
            # 可直接拿到整段 WAV bytes 并把 X-TTS-DSP-Impl 设到 headers。
            lc.stats["tts_req"] = lc.stats.get("tts_req", 0) + 1
            wav_bytes = await asyncio.to_thread(lc.tts.synthesize_full, opts)
            dsp_impl = str(getattr(lc.tts, "_last_dsp_impl", "") or "")
            if dsp_impl:
                headers["X-TTS-DSP-Impl"] = dsp_impl
            # RFC 7233: Content-Length 便于前端进度条与播放器缓冲估算
            headers["Content-Length"] = str(len(wav_bytes))

            async def _static():
                yield wav_bytes

            return StreamingResponse(_static(), media_type=media_type, headers=headers)
        except XiaobaiError as exc:
            raise HTTPException(500, detail=exc.to_dict()) from exc
        except Exception as exc:  # noqa: BLE001
            log.exception("TTS 合成异常")
            raise HTTPException(500, str(exc)) from exc

    @r.post("/tts/clone")
    async def tts_clone(file: UploadFile = File(...)):
        lc = get_lifecycle()
        data = await file.read()
        if len(data) < 2048:
            raise HTTPException(400, "参考音频太短。请上传 3~5 秒 wav。")
        sha = hashlib.sha1(data).hexdigest()
        clip_dir = Path(os.path.expanduser("~/.mox/models/voice/voice_clips"))
        clip_dir.mkdir(parents=True, exist_ok=True)
        dst = clip_dir / f"{sha}.wav"
        if not dst.is_file():
            with open(dst, "wb") as f:
                f.write(data)
        lc.loader.save_patch({"voice": {"tts": {"clone_reference": sha}}})
        lc.build_all(prewarm=False)
        return JSONResponse(dict(voice_id=sha, sha1=sha, path=str(dst)))

    @r.get("/hotwords")
    async def get_hotwords():
        hw = ((get_lifecycle().loader.data.get("voice") or {}).get("asr") or {}).get("hotwords") or []
        return JSONResponse(dict(hotwords=hw))

    @r.post("/hotwords")
    async def set_hotwords(req: Request):
        body = await req.json()
        words = body.get("words") or []
        lc = get_lifecycle()
        lc.loader.save_patch({"voice": {"asr": {"hotwords": list(words)}}})
        if lc.asr:
            try:
                lc.asr.set_hotwords(words)
            except Exception:  # noqa: BLE001
                pass
        return JSONResponse(dict(status="ok", hotwords=list(words)))

    @r.post("/license_tier")
    async def set_license_tier(req: Request):
        body = await req.json()
        tier = str(body.get("tier") or "auto").lower().strip()
        if tier not in {"auto", "research", "apache2"}:
            raise HTTPException(400, "tier 只能是 auto/research/apache2")
        lc = get_lifecycle()
        lc.loader.save_patch({"voice": {"license_tier": tier}})
        lc.build_all(prewarm=False)
        return JSONResponse(dict(status="ok", license_tier=tier, tts_engine=lc.tts_engine_name(), asr_engine=lc.asr_engine_name()))

    @r.get("/metrics")
    async def metrics():
        lc = get_lifecycle()
        lines = [
            f"voice_asr_requests_total{{engine=\"{lc.asr_engine_name()}\"}} {lc.stats.get('asr_req', 0)}",
            f"voice_tts_requests_total{{engine=\"{lc.tts_engine_name()}\"}} {lc.stats.get('tts_req', 0)}",
            f"voice_service_uptime_seconds {lc.uptime_s():.2f}",
        ]
        return HTMLResponse("\n".join(lines) + "\n", media_type="text/plain; charset=utf-8")

    app.include_router(r)


def run_server(host: str = "127.0.0.1", port: int = 30010, log_level: str = "info") -> None:
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port, log_level=log_level, ws_ping_interval=30, ws_ping_timeout=60)

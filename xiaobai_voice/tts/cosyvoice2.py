"""CosyVoice2 封装（Apache2，信创回退默认）· 豆包级动听质感。

相对 v1 的关键音质升级：
1. 指令前缀不再啰嗦：用「温柔日常 / 柔和 / 甜美 / 专业」风格引导，避免"请用...语气朗读"机械感。
2. write 内置 speaker id 探测：循环 preferred_spk_ids，兼容官方 0.5B / 自定义 SFT 权重。
3. 重采样：线性插值（替代 nearest 索引硬切，显著减少混叠和毛刺）；可选 librosa kaiser_best。
4. 语速（speed）：轻量 SOLA-like 时域缩放（帧长 20ms / 重叠 10ms），不变调，避免"唐老鸭"。
5. 响度/限幅：目标 -18 dBFS LUFS-like 归一化 + 软限幅（tanh ≥ 0.995 阈值），防止 peak 削波。
6. 输出原生 22050 Hz WAV（CosyVoice2 原生采样率，减少一次重采样损失）。
"""
from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

from .base import TTSBackend, TTSOptions
from ..errors import ErrorCode, XiaobaiError


# ============================================================= style / emotion
# 情绪 × 风格 → 豆包级指令前缀（短而自然，避免指令-正文割裂）
_INSTRUCTION_STYLE = {
    "warm_daily": "用温柔、自然、贴近日常聊天的中文语气，清晰明亮地说：",
    "gentle_soft": "用柔和、细腻、温暖治愈的中文语气，轻声说：",
    "anchor_premium": "用专业、沉稳、圆润悦耳的播音级中文语气，娓娓道来：",
    "professional_calm": "用专业、清晰、从容不迫的中文语气说：",
    "cute_lively": "用甜美、灵动、元气满满的中文语气说：",
}

_EMOTION_STYLE_FALLBACK = {
    "neutral": "warm_daily",
    "happy": "cute_lively",
    "sad": "gentle_soft",
    "serious": "professional_calm",
}

DEFAULT_PREFERRED_SPK = ["中文女", "女", "voice_0", "Default", "中文男", "xiaobai_default"]


# ====================================================================== utils
def _resample_linear(audio, sr_out, sr_in):
    """线性插值重采样：O(n)，纯 numpy，无额外依赖，比 nearest 顺滑很多。"""
    import numpy as np

    if int(sr_out) == int(sr_in):
        return audio.astype(np.float32, copy=False)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = int(round(len(audio) * float(sr_out) / float(sr_in)))
    if n <= 0:
        return audio
    if len(audio) == 1:
        return np.full(n, audio[0], dtype=np.float32)
    # 采样点坐标：旧采样对应 [0, len-1]
    idx_f = (np.arange(n, dtype=np.float64) * (len(audio) - 1)) / max(1, (n - 1))
    idx0 = np.floor(idx_f).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, len(audio) - 1)
    frac = (idx_f - idx0).astype(np.float32)
    return audio[idx0] * (1.0 - frac) + audio[idx1] * frac


def _resample(audio, sr_out, sr_in, quality: str):
    """根据配置选择重采样实现。quality ∈ {linear, kaiser_best}。"""
    quality = (quality or "linear").strip().lower()
    if quality in {"kaiser_best", "kaiserbest", "librosa", "best"}:
        try:
            import numpy as np  # noqa: F401
            import librosa  # type: ignore

            return librosa.resample(
                y=audio, orig_sr=int(sr_in), target_sr=int(sr_out), res_type="kaiser_best"
            )
        except Exception:  # noqa: BLE001
            pass
    return _resample_linear(audio, sr_out, sr_in)


def _time_stretch_sola(audio, target_len, frame_ms=20.0, overlap_ms=10.0, sr=22050):
    """
    Light-weight WSOLA/SOLA-like time-scale modification。
    适用于 ±30% 的小幅语速调整（speed 0.8~1.3 是 90% 场景，精度足够）。
    不变调。frame/overlap 取 20/10ms：480Hz / 1000Hz 基频都能稳定对齐。
    """
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    src_len = audio.size
    if src_len <= 1 or target_len <= 1:
        return audio
    if abs(src_len - target_len) <= 1:
        return audio
    sr_int = int(sr) if sr else 22050
    frame = max(32, int(sr_int * frame_ms / 1000.0))
    overlap = max(8, min(frame // 2, int(sr_int * overlap_ms / 1000.0)))
    hop_synthesis = max(1, frame - overlap)
    # ratio < 1 → 压缩（变慢？不：我们的 stretch ratio = target / src）
    ratio = float(target_len) / float(src_len)
    hop_analysis = max(1, int(round(hop_synthesis / ratio)))
    out = np.zeros(int(target_len) + frame, dtype=np.float32)
    win = np.hanning(frame).astype(np.float32)
    n_frames = (src_len - frame) // hop_analysis + 1
    write_pos = 0
    for i in range(n_frames):
        s = i * hop_analysis
        chunk = audio[s : s + frame].astype(np.float32)
        if chunk.size < frame:
            # 补零，保证窗口长度一致
            chunk = np.concatenate([chunk, np.zeros(frame - chunk.size, dtype=np.float32)])
        # overlap-add with window (crossfade)
        if write_pos == 0:
            out[0:frame] += chunk * win
            write_pos += hop_synthesis
            continue
        # 查找 overlap 区内最大互相关的偏移（± half_overlap）
        search = min(overlap, write_pos, out.size - frame) // 2
        if search > 0 and overlap >= 4 and write_pos + frame <= out.size:
            # 当前 chunk 开头 overlap 样本，与已写末尾 overlap 样本做互相关
            tail = out[write_pos : write_pos + overlap]
            head_need = chunk[0:overlap]
            # 暴力搜索 ±search 偏移（短窗 O(search*overlap)，search~数十 可接受）
            best_off = 0
            best_corr = -1e18
            for off in range(-search, search + 1):
                t_start = write_pos + off
                t_end = t_start + overlap
                if t_start < 0 or t_end > out.size:
                    continue
                buf = out[t_start:t_end]
                c = float(np.dot(buf, head_need))
                if c > best_corr:
                    best_corr = c
                    best_off = off
            write_pos += best_off
        if write_pos < 0:
            write_pos = 0
        if write_pos + frame > out.size:
            # 扩容一点点
            grow = np.zeros(write_pos + frame - out.size + 16, dtype=np.float32)
            out = np.concatenate([out, grow])
        out[write_pos : write_pos + frame] += chunk * win
        write_pos += hop_synthesis
    # 裁剪到目标长度
    if write_pos < target_len:
        # 没填满，补零
        if out.size < target_len:
            out = np.concatenate([out, np.zeros(target_len - out.size, dtype=np.float32)])
        return out[:target_len]
    return out[:target_len]


def _apply_limiter_and_loudness(audio, target_dbfs: float = -18.0, enable: bool = True):
    """响度归一化 + 软限幅（tanh）。
    - target_dbfs: -18 ~ -14 是中文普通话对话常用响度目标（接近豆包/Apple News）。
    - enable=False 时只做软限幅防止削波。
    """
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    # RMS -> dBFS
    rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
    db = 20.0 * np.log10(rms + 1e-12)
    if enable and np.isfinite(db) and db < -6.0:
        # 只在音量明显不足时抬升（防止静音段爆炸）
        gain_db = float(target_dbfs) - float(db)
        # 最多 +22 dB，防止异常冲激
        gain_db = min(22.0, max(0.0, gain_db))
        audio = audio * float(10.0 ** (gain_db / 20.0))
    # 软限幅：|x| >= 0.95 进入 tanh knee；|x| >= 1.0 严格压到 < 0.995
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.95:
        k = 1.0 / 0.95
        mask_high = np.abs(audio) >= 0.95
        signed = np.sign(audio[mask_high])
        scaled = (np.abs(audio[mask_high]) - 0.95) * k
        audio[mask_high] = signed * (0.95 + 0.045 * np.tanh(scaled))
    return audio


# ================================================================= Rust DSP 接入
# 优先 Rust xiaobai_core.dsp（统一核心库：重采样+SOLA+响度归一+软限幅+WAV编码）。
# 其次尝试旧版 xiaobai_dsp_native（兼容已有 .pyd）。
# 任何原因加载失败 → 自动回退纯 Python 实现。
# 环境变量 `XIAOBAI_VOICE_FORCE_PY_DSP=1` 可强制走 Python（便于 AB 对照）。
_RUST_DSP = None
_RUST_DSP_KIND: str | None = None  # "xiaobai_core" | "xiaobai_dsp_native" | None
_RUST_DSP_ERROR: str | None = None
try:
    import os as _os

    if not _os.environ.get("XIAOBAI_VOICE_FORCE_PY_DSP"):
        # 1) 优先 xiaobai_core（新统一 Rust 核心）
        try:
            from xiaobai_voice.core import dsp as _core_dsp

            if _core_dsp and getattr(_core_dsp, "_rust_available", True):
                _RUST_DSP = _core_dsp
                _RUST_DSP_KIND = "xiaobai_core"
        except Exception:
            pass
        # 2) 备选旧版 xiaobai_dsp_native
        if _RUST_DSP is None:
            try:
                import xiaobai_dsp_native  # type: ignore

                _RUST_DSP = xiaobai_dsp_native
                _RUST_DSP_KIND = "xiaobai_dsp_native"
            except Exception:
                pass
except Exception as _exc:  # noqa: BLE001
    _RUST_DSP_ERROR = f"{type(_exc).__name__}: {_exc}"


# --------------------------------------------------------------- CosyVoice src
# FunAudioLLM/CosyVoice 官方仓库不带 setup.py/pyproject，通常直接 `git clone` 后
# 把父目录加到 sys.path，`import cosyvoice` 即命中子包。
# 支持三种配置：
#   1) 环境变量 XIAOBAI_VOICE_COSYVOICE_SRC = <绝对路径到 CosyVoice 仓库根>
#   2) <repo_root>/third_party/CosyVoice （项目自带 clone）
#   3) pip install cosyvoice 已就绪（已在 sys.path 默认路径）
def _ensure_cosyvoice_src_on_path() -> None:
    import sys as _sys

    candidates: list[str] = []
    env_src = os.environ.get("XIAOBAI_VOICE_COSYVOICE_SRC") or ""
    if env_src.strip():
        candidates.append(os.path.abspath(os.path.expanduser(env_src.strip())))
    # 项目内 third_party/CosyVoice
    _here = os.path.dirname(__file__)
    candidates.append(
        os.path.abspath(
            os.path.join(_here, "..", "..", "..", "..", "third_party", "CosyVoice")
        )
    )
    for p in candidates:
        if not p or not os.path.isdir(p):
            continue
        if p not in _sys.path:
            _sys.path.insert(0, p)


_ensure_cosyvoice_src_on_path()


def _patch_cosyvoice_top_level_exports(mod_cv: Any) -> None:
    """FunAudioLLM/CosyVoice 顶层 __init__ 未 re-export CosyVoice/CosyVoice2 类。
    从 cli.cosyvoice 子模块别名到 mod_cv 顶层，保证 `cosyvoice.CosyVoice2(dir)` 可用。
    同时 monkey-patch cosyvoice.utils.file_utils.load_wav，绕开 torchaudio 强制 torchcodec 的问题
    （Windows 新版 torchaudio 即使 backend=soundfile 也会走 torchcodec 桥，DLL 极易缺失）。
    """
    # ---------- 1. load_wav monkey-patch（先做，保证后续 add_zero_shot 成功） ----------
    try:
        import numpy as _np
        import soundfile as _sf
        import torch as _torch

        def _patched_load_wav(wav, target_sr: int, min_sr: int = 16000):
            """兼容原 load_wav：**只返回 speech tensor**，shape=[1, N]，float32 torch.Tensor。
            （注意：torchaudio.load 本身返回 tuple，但 cosyvoice.utils.file_utils.load_wav 的
             原始实现是内部解包消费 sample_rate，最终只返回 speech。）
            wav 支持：str/Path（文件路径）、numpy array、torch.Tensor。
            """
            import os
            from pathlib import Path

            target_sr = int(target_sr)
            # 路径 → soundfile 读取
            if isinstance(wav, (str, Path, os.PathLike)):
                data, sr = _sf.read(str(wav), always_2d=False, dtype="float32")
                data = _np.asarray(data, dtype=_np.float32)
                if data.ndim >= 2:
                    data = data.mean(axis=1)  # 多声道 → 单声道 mix
            else:
                # numpy / torch tensor → 直接用
                if hasattr(wav, "detach") and callable(getattr(wav, "detach", None)):
                    wav = wav.detach().cpu().numpy()
                elif hasattr(wav, "numpy") and callable(getattr(wav, "numpy", None)):
                    wav = wav.numpy()
                data = _np.asarray(wav, dtype=_np.float32).reshape(-1)
                sr = int(target_sr)  # 非路径输入假设已是 target_sr
            if sr != target_sr:
                # 最小可用 sr 断言
                if sr < int(min_sr):
                    raise AssertionError(f"wav sample rate {sr} must be greater than {min_sr}")
                # 线性插值重采样（仅 prompt 用，性能不重要，精度对 campplus 足够）
                n_old = data.shape[0]
                n_new = int(round(n_old * float(target_sr) / float(sr)))
                if n_new <= 0:
                    data = _np.zeros(0, dtype=_np.float32)
                else:
                    idx_src = _np.linspace(0, n_old - 1, n_new, dtype=_np.float64)
                    idx0 = _np.floor(idx_src).astype(_np.int64)
                    idx1 = _np.minimum(idx0 + 1, n_old - 1)
                    frac = (idx_src - idx0).astype(_np.float32)
                    x0 = data[idx0]
                    x1 = data[idx1]
                    data = (x0 * (1.0 - frac) + x1 * frac).astype(_np.float32)
            # 统一 shape: [1, N]，float32 torch.Tensor（与原 load_wav 完全一致）
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim >= 2:
                data = data.mean(axis=0, keepdims=True)
            # 单返回值：仅 speech tensor
            return _torch.from_numpy(data.astype(_np.float32))

        try:
            from cosyvoice.utils import file_utils as _cv_fu

            if not getattr(_cv_fu.load_wav, "__xiaobai_patched__", False):
                _orig = _cv_fu.load_wav
                _patched_load_wav.__xiaobai_patched__ = True  # type: ignore[attr-defined]
                _patched_load_wav.__orig__ = _orig  # type: ignore[attr-defined]
                _cv_fu.load_wav = _patched_load_wav
                # frontend.py 等模块可能已 from file_utils import load_wav 到本地命名空间
                # → 同步补 patch frontend 模块引用
                try:
                    from cosyvoice.cli import frontend as _cv_fe

                    if hasattr(_cv_fe, "load_wav") and not getattr(_cv_fe.load_wav, "__xiaobai_patched__", False):
                        _cv_fe.load_wav = _patched_load_wav
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass

    # ---------- 2. 顶层类 re-export ----------
    needs: list[tuple[str, str]] = [
        ("CosyVoice2", "CosyVoice2"),
        ("CosyVoice", "CosyVoice"),
        ("CosyVoice3", "CosyVoice3"),
    ]
    for attr_name, export_name in needs:
        if hasattr(mod_cv, export_name):
            continue
        try:
            from cosyvoice.cli.cosyvoice import (  # type: ignore
                CosyVoice2 as _CV2,
                CosyVoice as _CV1,
                CosyVoice3 as _CV3,
            )

            mapping = {"CosyVoice2": _CV2, "CosyVoice": _CV1, "CosyVoice3": _CV3}
            cls = mapping.get(attr_name)
            if cls is not None:
                setattr(mod_cv, export_name, cls)
        except Exception:  # noqa: BLE001
            pass


def rust_dsp_available() -> bool:
    """返回 Rust DSP 是否成功加载（测试/监控用）。"""
    return _RUST_DSP is not None


def rust_dsp_error() -> str | None:
    """返回 Rust DSP 加载失败原因（若有）。"""
    return _RUST_DSP_ERROR

class CosyVoice2Backend(TTSBackend):
    name = "cosyvoice2"

    def __init__(self, cfg: dict, models_registry: Any | None = None) -> None:
        super().__init__(cfg, models_registry)
        cosy_cfg = ((cfg.get("engines") or {}).get("cosyvoice2") or {}) if isinstance(cfg, dict) else {}
        self._preferred_spk = list(cosy_cfg.get("preferred_spk_ids") or DEFAULT_PREFERRED_SPK)
        self._style = str(cosy_cfg.get("instruction_style") or "warm_daily").strip().lower()
        self._resample_quality = str(cosy_cfg.get("resample_quality") or "linear")
        self._limiter = bool(cosy_cfg.get("limiter", True))
        self._loudness_target_dbfs = float(cosy_cfg.get("loudness_target_dbfs") or -18.0)
        self._resolved_spk_id: str | None = None
        self._model = None
        self._model_sr = 22050
        try:
            self._ckpt_dir = self._resolve_model_dir(models_registry)
            self._load_engine()
        except XiaobaiError:
            raise
        except FileNotFoundError as exc:
            missing = getattr(exc, "filename", None)
            missing_text = f"（缺失路径：{missing}）" if missing else "（所有候选路径均未命中）"
            raise XiaobaiError(
                code=ErrorCode.MISSING_MODEL,
                message=(
                    "CosyVoice2 权重目录缺失。请下载 tts-cosyvoice2-0.5b 放入 "
                    f"`~/.mox/models/voice/` 或 `projects/xiaobai_voice/models/`。{missing_text}"
                ),
                cause=exc,
            ) from exc
        except ImportError as exc:
            raise XiaobaiError(
                code=ErrorCode.MISSING_DEP,
                message=(
                    "CosyVoice2 未安装。请执行：pip install cosyvoice>=0.2.0 （Apache2）。"
                    "或把 license_tier=apache2 切回 auto，允许浏览器 TTS 兜底。"
                ),
                cause=exc,
            ) from exc
        except OSError as exc:
            raise XiaobaiError(
                code=ErrorCode.DLL_LOAD_FAIL,
                message="CosyVoice2 加载 DLL/torch/onnxruntime 失败。",
                cause=exc,
            ) from exc

    def _resolve_model_dir(self, registry: Any | None) -> str:
        if registry is not None and hasattr(registry, "resolve"):
            r = registry.resolve("tts-cosyvoice2-0.5b")
            if r:
                return r["root"]
        candidates: list[str] = []
        import sys

        # 显式环境变量：最高优先级（用户把权重下载到自定义位置时用）
        for env_name in (
            "XIAOBAI_VOICE_COSYVOICE_CKPT_DIR",
            "COSYVOICE_CKPT_DIR",
            "MODEL_DIR_TTS_COSYVOICE2",
        ):
            raw = os.environ.get(env_name) or ""
            raw = raw.strip()
            if not raw:
                continue
            p = os.path.abspath(os.path.expanduser(raw))
            if os.path.isdir(p):
                candidates.append(p)
            # 允许 env 指向父目录（voice root），子目录固定 tts-cosyvoice2-0.5b
            candidates.append(os.path.join(p, "tts-cosyvoice2-0.5b"))
        if getattr(sys, "frozen", False):
            candidates.append(os.path.join(os.path.dirname(sys.executable), "models", "tts-cosyvoice2-0.5b"))
        candidates.append(os.path.join(os.path.expanduser("~"), ".mox", "models", "voice", "tts-cosyvoice2-0.5b"))
        candidates.append(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "tts-cosyvoice2-0.5b"))
        )
        # 兼容 ModelScope/HuggingFace 默认缓存名 CosyVoice2-0.5B
        for parent in (
            os.path.expanduser(r"~/.cache/modelscope/hub/speech_tts"),
            os.path.expanduser(r"~/.cache/modelscope/hub/AI-ModelScope"),
            os.path.expanduser(r"~/.cache/huggingface/hub"),
        ):
            if not os.path.isdir(parent):
                continue
            try:
                for name in os.listdir(parent):
                    low = name.lower()
                    if low.startswith("cosyvoice2") or low.startswith("cosyvoice") and "0.5" in low:
                        candidates.append(os.path.join(parent, name))
            except OSError:
                pass
        for c in candidates:
            if not c:
                continue
            if os.path.isfile(os.path.join(c, "configuration.json")) or os.path.isdir(c):
                # entry 为空 → 目录存在即通过；额外宽松：只要有 .pt 文件也视为权重目录
                if os.path.isdir(c):
                    try:
                        any_pt = any(
                            n.lower().endswith((".pt", ".safetensors", ".ckpt", ".onnx"))
                            for n in os.listdir(c)
                        )
                        if any_pt or os.path.isfile(os.path.join(c, "configuration.json")):
                            return c
                        continue
                    except OSError:
                        pass
                    return c
        raise FileNotFoundError(candidates[-1] if candidates else "tts-cosyvoice2-0.5b")

    def _load_engine(self) -> None:
        import cosyvoice  # type: ignore  # noqa: F401

        _patch_cosyvoice_top_level_exports(cosyvoice)
        try:
            self._model = cosyvoice.CosyVoice2(self._ckpt_dir)
        except Exception:
            try:
                self._model = cosyvoice.CosyVoice(self._ckpt_dir)  # type: ignore[attr-defined]
            except Exception as e:  # noqa: BLE001
                raise XiaobaiError(
                    code=ErrorCode.MISSING_MODEL,
                    message=f"CosyVoice2 模型构造失败，请确认权重目录为 CosyVoice2-0.5B：{e}",
                    cause=e,
                ) from e
        # CosyVoice2-0.5B 官方权重默认不提供 spk2info.pt → spk2info={}。
        # 这里用 zero-shot 生成一段「温暖女声提示音频」并注册为 xiaobai_default。
        self._ensure_default_spk_registered()
        # 探测可用 speaker id 列表
        self._resolved_spk_id = self._detect_spk()

    def _ensure_default_spk_registered(self) -> None:
        """当模型 frontend.spk2info 为空时，用 zero-shot + 伪正弦提示音注册默认音色，
        并持久化到 model_dir/spk2info.pt，下次启动直接加载。
        """
        model = self._model
        frontend = getattr(model, "frontend", None)
        if frontend is None:
            return
        spk2info = getattr(frontend, "spk2info", None)
        if not isinstance(spk2info, dict):
            return
        # 已注册任何音色 → 跳过
        if len(spk2info) > 0:
            return
        default_spk_id = "xiaobai_default"
        try:
            self._bootstrap_zero_shot_default(default_spk_id)
        except Exception as exc:  # noqa: BLE001
            log = __import__("logging").getLogger("xiaobai.tts.cosyvoice2")
            log.warning("默认音色 zero-shot 初始化失败，改走 cross_lingual / 空合成探测：%s", exc)

    def _bootstrap_zero_shot_default(self, spk_id: str) -> None:
        """用 24kHz × 3 秒合成正弦淡入淡出提示音，做一次 zero-shot 克隆，得到默认 spk embedding。
        好处：无外部文件、音色柔和（正弦波基频 + 轻微颤音），后续 inference_sft 稳定工作。
        注意：prompt_wav 必须是真实 WAV 文件路径（新版 torchaudio 需要 torchcodec + soundfile），不再直接传 tensor。
        关键：CosyVoice2 frontend_sft 依赖 spk2info[spk_id]['embedding']，而 add_zero_shot_spk 只写
        llm_embedding / flow_embedding；因此无论走 add_zero_shot_spk 还是 fallback 路径，注册后都要显式
        补齐 'embedding' 字段（取 llm_embedding）。
        """
        import os
        import tempfile

        import numpy as np

        model = self._model
        frontend = model.frontend
        sr = int(getattr(model, "sample_rate", 24000) or 24000)
        self._model_sr = sr
        dur_sec = 3.0
        n = int(sr * dur_sec)
        t = np.arange(n, dtype=np.float64) / sr
        # 基频 220Hz（温柔女声 A3），+ 轻微 5Hz 颤音 + 淡入淡出包络
        f0 = 220.0 + 2.0 * np.sin(2 * np.pi * 5.0 * t)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        audio = 0.18 * np.sin(phase).astype(np.float32)
        fade_in = min(int(0.3 * sr), n // 2)
        fade_out = min(int(0.6 * sr), n // 2)
        env = np.ones(n, dtype=np.float32)
        if fade_in > 0:
            env[:fade_in] = np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
        if fade_out > 0:
            env[-fade_out:] = np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
        audio = (audio * env).astype(np.float32)
        # 持久化 WAV 到 model_dir（或 temp 兜底），供 add_zero_shot_spk / zero-shot / instruct2 共用
        model_dir = str(getattr(model, "model_dir", self._ckpt_dir))
        try:
            prompt_wav_path = os.path.join(model_dir, "prompt_default_f.wav")
            import soundfile as sf

            sf.write(prompt_wav_path, audio, sr)
            if not os.path.isfile(prompt_wav_path):
                raise OSError("write failed")
        except Exception:  # noqa: BLE001
            with tempfile.NamedTemporaryFile(prefix="cosy2_prompt_", suffix=".wav", delete=False) as fp:
                prompt_wav_path = fp.name
            try:
                import soundfile as sf

                sf.write(prompt_wav_path, audio, sr)
            except Exception as exc:  # noqa: BLE001
                raise XiaobaiError(ErrorCode.MISSING_DEP, f"soundfile 写入默认 prompt WAV 失败：{exc}") from exc
        self._prompt_wav_path = prompt_wav_path
        prompt_text = "你好，我是小白。很高兴认识你，今天我们一起温柔地聊天吧。"
        ok = False
        try:
            ok = bool(model.add_zero_shot_spk(prompt_text, prompt_wav_path, spk_id))
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            # 兜底：用 frontend_zero_shot 直接构造 entry 注册到 spk2info（不传 text 字段）
            try:
                entry = frontend.frontend_zero_shot("", prompt_text, prompt_wav_path, sr, "")
                entry.pop("text", None)
                entry.pop("text_len", None)
                frontend.spk2info[spk_id] = entry
                ok = True
            except Exception:  # noqa: BLE001
                ok = False
        # ---- 关键：无论哪种路径写入 spk2info[spk_id]，都要保证存在 'embedding' 键 ----
        if ok and spk_id in frontend.spk2info:
            entry = frontend.spk2info[spk_id]
            if isinstance(entry, dict) and "embedding" not in entry:
                if "llm_embedding" in entry:
                    entry["embedding"] = entry["llm_embedding"]
                else:
                    try:
                        entry["embedding"] = frontend._extract_spk_embedding(prompt_wav_path)  # noqa: SLF001
                    except Exception as exc:  # noqa: BLE001
                        log_extra = __import__("logging").getLogger("xiaobai.tts.cosyvoice2")
                        log_extra.warning("默认音色注册后仍缺 embedding 字段，后续 SFT 路径禁用：%s", exc)
                        # 回退：把 spk2info 清空，让 _do_infer_raw 走 zero-shot 直通
                        frontend.spk2info.pop(spk_id, None)
                        ok = False
        if ok:
            import torch

            try:
                spk2info_path = os.path.join(model_dir, "spk2info.pt")
                torch.save(frontend.spk2info, spk2info_path)
            except Exception:  # noqa: BLE001
                pass
            return
        # 两次都失败 → 至少保证 prompt_wav_path 可用，_do_infer_raw 会改走 inference_zero_shot 直出
        log = __import__("logging").getLogger("xiaobai.tts.cosyvoice2")
        log.warning("spk2info 注册未完成，后续走 zero-shot 直通模式。prompt=%s", self._prompt_wav_path)

    def _detect_spk(self) -> str:
        """从真实注册的 frontend.spk2info.keys() 中选择；若空，返回标记位 '' 表示走 zero-shot 直通。"""
        model = self._model
        frontend = getattr(model, "frontend", None)
        registered: list[str] = []
        if frontend is not None:
            spk2info = getattr(frontend, "spk2info", None)
            if isinstance(spk2info, dict):
                registered = [k for k in spk2info.keys() if isinstance(k, str)]
        if registered:
            for cand in self._preferred_spk:
                if cand in registered:
                    return cand
            for s in registered:
                if "女" in s or "female" in s.lower() or "voice_0" in s.lower() or "default" == s.lower():
                    return s
            return registered[0]
        # 没有任何 spk 注册 → 标记 '', _do_infer_raw 将走 zero-shot / instruct2
        return ""

    def _probe_spk_by_inference(self) -> str:
        """已在 _detect_spk 中处理真实注册信息。保留此函数以兼容旧版本调用。"""
        return self._detect_spk()

    def _do_infer_raw(self, text: str, spk: str, *, emotion: str = "neutral", speed: float = 1.0):
        """CosyVoice2/CosyVoice 推理调度器。

        优先级：
        1. CosyVoice2 且有 instruct2 + prompt_wav → inference_instruct2（指令驱动，豆包级自然）
        2. spk 已注册在 spk2info → inference_sft
        3. 有 prompt_wav → inference_zero_shot 直通
        4. inference（通用兜底）
        """
        import os

        model = self._model
        frontend = getattr(model, "frontend", None)
        registered_keys = set()
        if frontend is not None and isinstance(getattr(frontend, "spk2info", None), dict):
            registered_keys = set(k for k in frontend.spk2info.keys() if isinstance(k, str))
        has_prompt_wav = bool(getattr(self, "_prompt_wav_path", None)) and os.path.isfile(str(self._prompt_wav_path))
        use_instruct2 = (
            hasattr(model, "inference_instruct2")
            and callable(model.inference_instruct2)
            and "CosyVoice2" in type(model).__name__
            and has_prompt_wav
        )
        # CosyVoice2 官方权重（无预设 spk）强烈推荐 instruct2：支持情绪/语速/音量指令，拟人度最好
        if use_instruct2:
            style_key = _EMOTION_STYLE_FALLBACK.get(emotion, "warm_daily")
            if self._style and self._style in _INSTRUCTION_STYLE:
                style_key = self._style
            prefix = _INSTRUCTION_STYLE.get(style_key, _INSTRUCTION_STYLE["warm_daily"])
            # 指令：语气引导 + 明确语速（speed 映射到自然语言指令，保持 speed 参数可被用户感知）
            speed_hint = ""
            try:
                s = float(speed)
                if s >= 1.25:
                    speed_hint = "语速稍快。"
                elif s <= 0.8:
                    speed_hint = "语速较慢。"
                elif s >= 1.1:
                    speed_hint = "语速略快。"
                elif s <= 0.95:
                    speed_hint = "语速舒缓。"
            except Exception:  # noqa: BLE001
                pass
            instruct = f"{prefix}{speed_hint}".strip()
            try:
                yield from model.inference_instruct2(text, instruct, str(self._prompt_wav_path), "", stream=False, speed=1.0)
                return
            except XiaobaiError:
                raise
            except Exception as exc:  # noqa: BLE001
                log = __import__("logging").getLogger("xiaobai.tts.cosyvoice2")
                log.warning("instruct2 模式失败：%s；回退 zero-shot/sft。", exc)
        # spk 已注册 → inference_sft；前置条件：entry 必须含 'embedding' 键（frontend_sft 会读取它）
        sft_ready = bool(spk) and spk in registered_keys
        if sft_ready and frontend is not None and isinstance(getattr(frontend, "spk2info", None), dict):
            entry = frontend.spk2info.get(spk)
            if not (isinstance(entry, dict) and "embedding" in entry):
                sft_ready = False
        if sft_ready and hasattr(model, "inference_sft") and callable(model.inference_sft):
            yield from model.inference_sft(text, spk)
            return
        # zero-shot 直通（无预设音色时的标准路径，官方 CosyVoice2 最稳定）
        if hasattr(model, "inference_zero_shot") and callable(model.inference_zero_shot) and has_prompt_wav:
            prompt_text = "你好，我是小白。今天我们温柔自然地聊天吧。"
            yield from model.inference_zero_shot(text, prompt_text, str(self._prompt_wav_path), "", stream=False)
            return
        if hasattr(model, "inference_sft") and callable(model.inference_sft) and spk and spk in registered_keys:
            yield from model.inference_sft(text, spk)
            return
        if hasattr(model, "inference") and callable(model.inference):
            yield from model.inference(text, spk or "")
            return
        raise XiaobaiError(
            code=ErrorCode.RUNTIME,
            message="当前 CosyVoice2 模型对象无可用推理入口（inference_sft / zero_shot / instruct2 / inference）。",
        )

    # -------------------------------------------------------------- synthesize
    def synthesize(self, opts: TTSOptions) -> Generator[bytes, None, None]:
        import numpy as np

        model_sr = int(self._model_sr or 22050)
        sr = int(opts.sample_rate or self.sample_rate or model_sr)
        if opts.emotion not in _EMOTION_STYLE_FALLBACK:
            opts.emotion = "neutral"
        speed = float(getattr(opts, "speed", 1.0) or 1.0)
        if speed <= 0:
            speed = 1.0
        # Note：前缀式 instruction 由 _do_infer_raw 的 instruct2 模式负责（豆包级拟人度最好）。
        # 非 instruct2 模式（inference_sft / zero-shot）仍然传原文 + 语气偏好，让模型自身发挥。
        text = (opts.text or "").strip()
        if not text:
            raise XiaobaiError(ErrorCode.RUNTIME, "合成文本为空。")

        spk = self._resolved_spk_id or ""
        try:
            synth_iter = self._do_infer_raw(text, spk, emotion=opts.emotion, speed=speed)
        except XiaobaiError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise XiaobaiError(
                code=ErrorCode.RUNTIME,
                message=f"CosyVoice2 合成失败：{exc}",
                cause=exc,
            ) from exc

        audio_chunks: list[np.ndarray] = []
        sample_rate_out = model_sr
        try:
            for item in synth_iter:
                if isinstance(item, tuple) and len(item) == 2:
                    sample_rate_out, arr = item
                    sample_rate_out = int(sample_rate_out or model_sr)
                elif isinstance(item, dict) and "tts_speech" in item:
                    tts_speech = item["tts_speech"]
                    # CosyVoice 两种格式：tuple(sr, tensor) 或 直接 tensor(shape [1,N])
                    if isinstance(tts_speech, tuple) and len(tts_speech) == 2:
                        sample_rate_out = int(tts_speech[0] or model_sr)
                        arr = tts_speech[1]
                    else:
                        arr = tts_speech
                        sample_rate_out = int(getattr(item, "sample_rate", sample_rate_out) or sample_rate_out)
                else:
                    continue
                # numpy 化（兼容 torch.Tensor）
                try:
                    if hasattr(arr, "detach") and callable(getattr(arr, "detach", None)):
                        arr = arr.detach().cpu().numpy()
                    elif hasattr(arr, "numpy") and callable(getattr(arr, "numpy", None)):
                        arr = arr.numpy()
                except Exception:  # noqa: BLE001
                    pass
                audio_chunks.append(np.asarray(arr, dtype=np.float32).reshape(-1))
        except Exception as exc:  # noqa: BLE001
            raise XiaobaiError(ErrorCode.RUNTIME, f"CosyVoice2 合成中断: {exc}", cause=exc) from exc

        if not audio_chunks:
            raise XiaobaiError(ErrorCode.RUNTIME, "CosyVoice2 合成结果为空。可能是空文本或权重不完整。")

        audio = np.concatenate(audio_chunks, axis=0) if len(audio_chunks) > 1 else audio_chunks[0]

        # ---------------------------------------------------------------- DSP
        # 优先 Rust：一次性重采样+SOLA+响度归一+软限幅+WAV PCM16 编码。
        # 支持 xiaobai_core.dsp（新统一核心）和 xiaobai_dsp_native（旧版）两种后端。
        # Rust 模块不可用/抛错 → 自动 fallback 原 Python 流水线。
        dsp_impl = "Rust"
        wav_bytes: bytes | None = None
        if _RUST_DSP is not None:
            try:
                import numpy as _np  # local

                _audio_arr = _np.asarray(audio, dtype=np.float32).reshape(-1)
                _sig = _audio_arr.tolist()

                if _RUST_DSP_KIND == "xiaobai_core":
                    # 新统一核心：process_tts_audio → wav_encode
                    processed = _RUST_DSP.process_tts_audio(
                        _sig,
                        input_sr=int(sample_rate_out),
                        target_sr=int(sr),
                        speed=float(speed),
                        loudness_target_dbfs=float(self._loudness_target_dbfs),
                        limiter=bool(self._limiter),
                    )
                    wav_bytes = bytes(_RUST_DSP.wav_encode(processed, int(sr)))
                else:
                    # 旧版 xiaobai_dsp_native：apply_dsp_pipeline（直接输出 WAV bytes）
                    res = _RUST_DSP.apply_dsp_pipeline(
                        _sig,
                        {
                            "orig_sr": int(sample_rate_out),
                            "target_sr": int(sr),
                            "speed": float(speed),
                            "target_dbfs": float(self._loudness_target_dbfs),
                            "enable_loudness": bool(self._limiter),
                            "encode_wav": True,
                            "channels": 1,
                        },
                    )
                    if isinstance(res, (bytes, bytearray)):
                        wav_bytes = bytes(res)
            except Exception:  # noqa: BLE001
                wav_bytes = None
                dsp_impl = "Python(FallbackFromRustError)"

        if wav_bytes is not None:
            # Rust 已输出完整 WAV（header + PCM16 body）。按流式 chunk 切片。
            header_len = 44
            if len(wav_bytes) >= header_len:
                yield wav_bytes[:header_len]
                body = wav_bytes[header_len:]
            else:
                yield wav_bytes[:]
                body = b""
            chunk_bytes = max(1024, int(sr * 2 * (opts.stream_chunk_ms / 1000.0)))
            for i in range(0, len(body), chunk_bytes):
                yield body[i : i + chunk_bytes]
            # 记录实现来源（供健康检查/报告查询）
            self._last_dsp_impl = dsp_impl  # type: ignore[attr-defined]
            return

        # -------------------------------- fallback：原 Python 流水线
        dsp_impl = "Python"
        # 1) 重采样到目标 sr（linear/kaiser）
        audio = _resample(audio, sr, sample_rate_out, self._resample_quality)

        # 2) 语速缩放（speed != 1.0 时 SOLA）
        if abs(speed - 1.0) > 1e-3 and 0.5 <= speed <= 2.0:
            target_len = int(round(audio.size / speed))
            audio = _time_stretch_sola(audio, target_len, frame_ms=20.0, overlap_ms=10.0, sr=sr)

        # 3) 响度归一 + 软限幅（防止 peak 削波）
        audio = _apply_limiter_and_loudness(
            audio,
            target_dbfs=self._loudness_target_dbfs,
            enable=self._limiter,
        )

        # 4) float → int16 输出（按 xiaobai-dsp wav.rs 精确钳位：负 ×32768 / 正 ×32767）
        import numpy as _np2

        audio = _np2.asarray(audio, dtype=np.float32).reshape(-1)
        neg = audio < 0
        pos = ~neg
        scaled = _np2.zeros_like(audio, dtype=np.float32)
        scaled[neg] = audio[neg] * 32768.0
        scaled[pos] = audio[pos] * 32767.0
        int16 = scaled.clip(-32768, 32767).astype("<i2")
        raw = int16.tobytes()

        from .browser_fallback import _make_wav_header

        yield _make_wav_header(sr=sr, channels=1, bits=16, data_len=len(raw))
        chunk_bytes = max(1024, int(sr * 2 * (opts.stream_chunk_ms / 1000.0)))
        for i in range(0, len(raw), chunk_bytes):
            yield raw[i : i + chunk_bytes]
        self._last_dsp_impl = dsp_impl  # type: ignore[attr-defined]

    def close(self) -> None:
        if self._model is not None:
            self._model = None

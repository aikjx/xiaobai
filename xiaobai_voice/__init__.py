"""Xiaobai (xiaobai_voice) — 离线语音 + 桌面小白 AI 助手。

核心性能模块（DSP/配置/意图/算子/模型注册表）由 Rust 扩展 xiaobai_core 提供，
纯 Python 实现作为自动回退。详见 xiaobai_voice.core。

Modules
-------
asr : Paraformer-zh + sherpa-onnx（默认）/ SenseVoice 可选 ASR 后端
tts : Fish-S2-Pro (Research, 可选) / CosyVoice2 (Apache2, 默认回退) / BrowserFallback
service : FastAPI + WebSocket 语音服务，默认端口 30010
desktop : PySide6 悬浮球 + 主窗口 + 全局快捷键
models  : 模型存储与断点续传下载器
config  : 跨平台配置与模型元数据加载器
core    : Rust 核心集成层（DSP / 配置 / 意图路由 / RBAC 算子 / 模型注册表）
"""

__version__ = "0.1.0"
__all__ = ["__version__"]

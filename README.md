# xiaobai_voice — 璇玑系统离线语音 & 桌面小白（xiaobai）AI 助手

> ASR：**Paraformer-zh + sherpa-onnx**（Apache2 / 离线 CPU 最优）
> TTS：**Fish-Speech-S2-Pro**（Research，默认启用 iff 本地权重完整且 license_tier != apache2）
>        ↘ **CosyVoice2**（Apache2，信创/政务默认回退）
>        ↘ **浏览器 SpeechSynthesis**（MessageBubble 旧实现，保留兜底）
> 端口：语音服务默认 **30010**；前端 Vite dev server 走 `/voice` proxy

## 一、快速启动（开发态）

```powershell
# 1. 创建 venv
cd projects\xiaobai_voice
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装：asr + service 必装；tts(=CosyVoice2)、desktop、dev 建议装全
pip install -e ".[asr,service,desktop,dev]"
# （可选）如果要启用 Fish-S2-Pro，手动装：
# pip install fish-speech[s2pro]

# 3. 下载默认模型（ASR Paraformer INT8 + CosyVoice2-0.5B）
python -m xiaobai_voice download --defaults

# 4. 启动语音服务（端口 30010）
python -m xiaobai_voice serve

# 5. 另一个终端：启动桌面小白（浮窗 + 快捷键 + 内嵌 /#/ai WebView）
python -m xiaobai_voice desktop

# 6. 冒烟自检测试
python -m xiaobai_voice selftest
```

## 二、Rust 核心扩展（xiaobai_core）

核心性能模块由 Rust 实现（PyO3 + abi3 稳定 ABI，兼容 Python 3.9+），纯 Python 实现作为自动回退。

### 2.1 构建

```powershell
cd projects\xiaobai_voice\xiaobai_core
cargo build --release
# 产物：target\release\xiaobai_core.dll
# 复制到 Python 包目录并重命名为 .pyd：
copy target\release\xiaobai_core.dll ..\xiaobai_voice\xiaobai_core.pyd
```

或使用一键构建脚本（自动复制到包目录）：

```powershell
cd projects\xiaobai_voice
.\build_rust_core.ps1
```

### 2.2 覆盖模块

| 模块 | Rust 实现 | Python 回退 | 性能提升 |
|------|-----------|-------------|---------|
| `dsp` | 重采样 / SOLA 语速 / 响度归一 / 软限幅 / WAV 编解码 | numpy 实现 | ~5-10× |
| `config` | YAML deep-merge / 跨平台路径 / 原子写入 | PyYAML | ~2× |
| `intent` | 正则规则路由 / RBAC 置信度衰减 | Python regex | ~3× |
| `operators` | 音量 / 应用 / 文件 / 输入算子 + 4 级 RBAC 引擎 | Python 系统调用 | ~1.5× |
| `models` | 模型注册表 / SHA256 校验 / 本地路径解析 | Python | ~2× |

### 2.3 验证

```python
from xiaobai_voice.core import dsp, RUST_AVAILABLE, RUST_VERSION
print(f"Rust available: {RUST_AVAILABLE}, version: {RUST_VERSION}")

# DSP 流水线测试
samples = [0.0] * 22050  # 1 秒静音
processed = dsp.process_tts_audio(samples, 22050, 16000, speed=1.0,
                                    loudness_target_dbfs=-18.0, limiter=True)
print(f"Output length: {len(processed)}")
```

## 三、打包发布（windowed，零控制台闪退）

```powershell
.\build_exe.ps1 -UseVenv ".\.venv"
# 产物：dist\Xiaobai\Xiaobai.exe
# 以 Start-Process 双击方式启动验证（避免掩盖 stderr=None 问题）
Start-Process dist\Xiaobai\Xiaobai.exe
Start-Process dist\Xiaobai\Xiaobai.exe -ArgumentList "--selftest-full"
```

## 三、配置位置

Windows：`%APPDATA%\mox\xiaobai\config.yaml`
macOS：`~/Library/Application Support/mox/xiaobai/config.yaml`
Linux：`$XDG_CONFIG_HOME/mox/xiaobai/config.yaml`

启动后可用合规 φ Chip 对话框直接切换 `license_tier=apache2`（默认 auto）。

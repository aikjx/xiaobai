"""voice_proxy 包：voice_service ⇄ mox 双向桥（WebSocket 主通道 + HTTP fallback）。

使用示例（最小本地模式）：

    from xiaobai_voice.operator import OperatorEngine
    from xiaobai_voice.proxy import VoiceProxyClient

    engine = OperatorEngine(strategy="local_first")
    proxy  = VoiceProxyClient(engine, cfg={"strategy": "local_first"})
    # 仅本地模式可以不 await proxy.start()
"""
from .voice_proxy import (
    BridgeStatus,
    IntentDispatchCallback,
    MsgType,
    ProxyEnvelope,
    VoiceProxyClient,
)

__all__ = [
    "BridgeStatus",
    "IntentDispatchCallback",
    "MsgType",
    "ProxyEnvelope",
    "VoiceProxyClient",
]

"""voice_proxy 桥：桌面小白助手 ⇄ mox-expert/mox-system 双向网关。

协议（AIS-FR13 V1.0）
--------------------
传输：WebSocket（持久双向，主通道）+ HTTP（一次性命令，fallback）
端口：
    voice_service（本桥暴露）默认 HTTP/WS = 30010（与现有 FastAPI 语音服务端口一致，同端点多协议）
    mox-system 服务端默认 HTTP = 13130（对齐 mox 生态已有约定）
策略（与 voice.strategy 对齐）：
    - local_first    ：本地 OperatorEngine 优先执行；成功/失败同时上报 mox 审计（不阻塞），
                       若 mox 断连则静默降级为纯本地。
    - cloud_fallback ：本地 OPERATOR_UNSUPPORTED → 转发到 mox 远程算子；其它情况同 local_first。
    - cloud_only     ：强制所有 INTENT/EXEC 都先交 mox-expert 裁决，通过后再由本地或 mox 远程执行，
                       断连一律返回 BRIDGE_DISCONNECTED（企业合规模式）。

消息结构（JSON，所有字段严格 snake_case）：
    # C→S （桌面 → mox）
    { "type": "intent/rpc/audit/hello" , "id": "...", "payload": {...} }

    # S→C （mox → 桌面）
    { "type": "intent/rpc/audit/ack/exec" , "id": "...", "reply_to": "...", "payload": {...}, "code":"OK" }

本模块实现：
    - VoiceProxyClient ：跑在 voice_service 内部，单例对外暴露 `dispatch_intent(text, ctx)`，
                        内部决定"本地直干 / 转发 mox"，最终返回统一 OperatorResult
    - VoiceProxyServer ：可选的轻量 HTTP/WS 服务端（供 mox-system 反调桌面端拿截图/远程协助等场景；
                        默认不启动，仅 local_first 不需要）
"""
from __future__ import annotations

import abc
import asyncio
import atexit
import enum
import json
import logging
import platform
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..errors import ErrorCode, XiaobaiError
from ..asr import AsrResult
from .base import Identity, OperatorEngine, OperatorResult

log = logging.getLogger("xiaobai.voice_proxy")


# ---------------------------------------------------------------------------
# Enums / Messages
# ---------------------------------------------------------------------------

class MsgType(str, enum.Enum):
    HELLO = "hello"             # 握手，上报 voice_service 身份
    INTENT = "intent"           # C→S：语音文本 + 上下文，请求专家联盟裁决路由
    EXEC = "exec"               # S→C：下发算子执行命令 / C→S 远程算子执行请求
    AUDIT = "audit"             # C→S：算子执行结果审计上报
    ACK = "ack"                 # S→C：回执确认
    PING = "ping"
    PONG = "pong"


class BridgeStatus(str, enum.Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CLOSED = "closed"


@dataclass
class ProxyEnvelope:
    msg_type: MsgType
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    reply_to: str = ""
    code: str = ErrorCode.OK.value
    message: str = ""

    def to_json(self) -> str:
        return json.dumps({
            "type": self.msg_type.value,
            "id": self.id,
            "reply_to": self.reply_to,
            "code": self.code,
            "message": self.message,
            "payload": self.payload,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "ProxyEnvelope":
        d = json.loads(s)
        return cls(
            msg_type=MsgType(d.get("type", MsgType.ACK.value)),
            payload=dict(d.get("payload") or {}),
            id=str(d.get("id") or uuid.uuid4().hex),
            reply_to=str(d.get("reply_to") or ""),
            code=str(d.get("code") or ErrorCode.OK.value),
            message=str(d.get("message") or ""),
        )


# ---------------------------------------------------------------------------
# Transport 抽象（WS / HTTP 两种）
# ---------------------------------------------------------------------------

class Transport(abc.ABC):
    status: BridgeStatus = BridgeStatus.DISCONNECTED

    @abc.abstractmethod
    async def connect(self) -> None: ...
    @abc.abstractmethod
    async def close(self) -> None: ...
    @abc.abstractmethod
    async def send(self, env: ProxyEnvelope, timeout: float) -> None: ...
    @abc.abstractmethod
    async def recv(self, timeout: float) -> ProxyEnvelope | None: ...


# ---------------------------------------------------------------------------
# Client：voice_service ↔ mox-system/mox-expert
# ---------------------------------------------------------------------------

IntentDispatchCallback = Callable[
    [str, dict[str, Any]], Awaitable[OperatorResult | dict] | OperatorResult | dict | None,
]
# 签名: callback(text, ctx) → result。当收到 mox 的 "exec" 反调通知桌面侧时触发。


class VoiceProxyClient:
    """桌面 voice_service 与 mox 双向桥的主入口。

    典型生命周期：
        engine = OperatorEngine(cfg)
        proxy = VoiceProxyClient(engine, cfg={"strategy":"local_first", "mox_ws":"ws://127.0.0.1:13130/ws"})
        asyncio.run(proxy.start())
        result = asyncio.run(proxy.dispatch_intent("打开记事本", {..}))
    """

    def __init__(
        self,
        engine: OperatorEngine,
        cfg: dict | None = None,
        *,
        on_exec_from_mox: IntentDispatchCallback | None = None,
    ) -> None:
        self.cfg = dict(cfg or {})
        self.engine = engine
        self.strategy = str(self.cfg.get("strategy") or "local_first").lower()
        if self.strategy not in ("local_first", "cloud_fallback", "cloud_only"):
            raise XiaobaiError(ErrorCode.CONFIG_INVALID, f"未知 strategy: {self.strategy}")

        self.mox_http_base: str = str(self.cfg.get("mox_http") or "http://127.0.0.1:13130").rstrip("/")
        self.mox_ws_url: str = str(self.cfg.get("mox_ws") or self.mox_http_base.replace("http://", "ws://") + "/ws")

        self.identity = Identity(
            user_id=str(self.cfg.get("user_id") or f"xiaobai-{platform.node().lower()}"),
            role=str(self.cfg.get("role") or "Member"),
            tenant_id=str(self.cfg.get("tenant_id") or "default"),
        )

        self._transport: Transport | None = None
        self._status = BridgeStatus.DISCONNECTED
        self._on_exec_from_mox = on_exec_from_mox
        self._pending = dict[str, asyncio.Future[ProxyEnvelope]]()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._recv_task: asyncio.Task | None = None
        self._last_hello_at = 0.0

        atexit.register(self._sync_shutdown)

    # -------------------------------------------------------------- 生命周期
    @property
    def status(self) -> BridgeStatus:
        return self._status

    async def start(self) -> None:
        """启动桥接：连接 mox （cloud_only 下不允许失败）；启动 recv loop。"""
        self._loop = asyncio.get_running_loop()
        if self.strategy != "local_first" or self.cfg.get("always_connect"):
            self._status = BridgeStatus.CONNECTING
            try:
                await self._connect_ws()
                self._status = BridgeStatus.CONNECTED
                asyncio.create_task(self._heartbeat_loop())
            except Exception as exc:  # noqa: BLE001
                log.warning("连接 mox 失败：%s（strategy=%s）", exc, self.strategy)
                if self.strategy == "cloud_only":
                    self._status = BridgeStatus.DISCONNECTED
                    raise XiaobaiError(ErrorCode.BRIDGE_DISCONNECTED,
                                       f"cloud_only 模式必须能连上 mox：{exc}", cause=exc) from exc
                self._status = BridgeStatus.DISCONNECTED

        # 默认注册 Engine.audit_cb 用于同步上报审计
        if self.engine.audit_cb is None:
            self.engine.audit_cb = self._engine_audit_cb_sync

    async def stop(self) -> None:
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
        if self._transport is not None:
            try:
                await self._transport.close()
            except Exception:  # noqa: BLE001
                pass
        self._status = BridgeStatus.CLOSED

    def _sync_shutdown(self) -> None:
        try:
            loop = self._loop or asyncio.new_event_loop()
            loop.run_until_complete(self.stop())
        except Exception:  # noqa: BLE001
            pass

    # -------------------------------------------------------------- 核心 API：dispatch_intent
    async def dispatch_intent(
        self,
        text: str,
        asr_result: AsrResult | None = None,
        ctx: dict[str, Any] | None = None,
    ) -> OperatorResult:
        """语音转文字之后的入口：
        1) 如 strategy==cloud_only / (cloud_fallback 且本地路由不确定) → 先问联盟；
        2) 本地算子执行（优先）或远程算子执行；
        3) 同步审计上报。
        """
        t0 = time.perf_counter()
        text = (text or "").strip()
        if not text:
            return OperatorResult(op="", act="", ok=False,
                                  code=ErrorCode.INTENT_UNKNOWN.value, message="空文本，无法路由意图")
        ctx = dict(ctx or {})
        if asr_result is not None:
            ctx.setdefault("asr", {
                "text": asr_result.text,
                "confidence": asr_result.confidence,
                "backend": asr_result.backend,
                "hotwords_hint": bool(asr_result.hotwords or None),
            })

        # 1) 先做 PPR 本地路由（见 intent.router），得到候选路由
        from ..intent.router import IntentRouter
        router = IntentRouter(self.cfg.get("intent"))
        route = router.route(text, self.identity)

        # 2) strategy 分支
        if self.strategy == "cloud_only":
            # cloud_only 强制走专家联盟裁决（INTENT 消息 → 等 EXEC/ACK）
            return await self._remote_intent_flow(text, route, ctx)

        # local_first / cloud_fallback：本地先试
        if route.op_name and route.act and route.confidence >= 0.55:
            # 有明确命中：本地直接 dispatch
            r = self.engine.dispatch(
                route.op_name, route.act, route.params,
                identity=self.identity,
            )
            # cloud_fallback：本地 unsupported → 尝试远程
            if not r.ok and r.code == ErrorCode.OPERATOR_UNSUPPORTED.value and self.strategy == "cloud_fallback":
                remote = await self._remote_intent_flow(text, route, ctx)
                # 如果远程成功，用远程结果覆盖
                if remote.ok:
                    return remote
            r.duration_ms = (time.perf_counter() - t0) * 1000.0
            return r

        # 本地路由置信度低 / 未命中 → cloud_fallback 转联盟裁决；local_first 直接 INTENT_UNKNOWN
        if self.strategy == "cloud_fallback" and self._status == BridgeStatus.CONNECTED:
            return await self._remote_intent_flow(text, route, ctx)
        # local_first 低置信度：如果路由给了建议但<0.55，就直接返回 UNKNOWN + 建议候选
        msg = "本地意图未命中"
        if route.candidates:
            msg += f"；可能候选：{route.candidates[:3]}"
        r = OperatorResult(
            op=route.op_name or "", act=route.act or "", ok=False,
            code=ErrorCode.INTENT_UNKNOWN.value, message=msg,
            data={"route": route.as_dict(), "text": text},
        )
        r.duration_ms = (time.perf_counter() - t0) * 1000.0
        return r

    # -------------------------------------------------------------- 远程联盟流程
    async def _remote_intent_flow(
        self,
        text: str,
        route,
        ctx: dict[str, Any],
    ) -> OperatorResult:
        if self._status != BridgeStatus.CONNECTED:
            # local_first 仅当未连接时直接回本地；到这里说明是 cloud_only 或 fallback
            if self.strategy == "cloud_only":
                return OperatorResult(
                    op="", act="", ok=False,
                    code=ErrorCode.BRIDGE_DISCONNECTED.value,
                    message="cloud_only 模式下 mox 桥未连接，无法调用专家联盟",
                )
            # cloud_fallback: 断线走本地
            return self.engine.dispatch(
                route.op_name or "app", route.act or "open_app",
                route.params, identity=self.identity,
            )

        env = ProxyEnvelope(
            msg_type=MsgType.INTENT,
            payload={
                "text": text,
                "identity": {
                    "user_id": self.identity.user_id,
                    "role": self.identity.role,
                    "tenant_id": self.identity.tenant_id,
                },
                "local_route": route.as_dict(),
                "ctx": ctx,
            },
        )
        # wait_for_reply：cloud_only 必等；cloud_fallback 给 800ms 裁决窗口
        wait_ms = 3000 if self.strategy == "cloud_only" else 800
        try:
            reply = await self._request(env, timeout_ms=wait_ms)
        except TimeoutError as exc:
            if self.strategy == "cloud_only":
                return OperatorResult(
                    op="", act="", ok=False,
                    code=ErrorCode.BRIDGE_DISCONNECTED.value,
                    message=f"专家联盟裁决超时（{wait_ms}ms）：{exc}",
                )
            # cloud_fallback 超时 → 本地直接走
            log.info("联盟裁决超时，cloud_fallback 本地执行：%s", text)
            return self.engine.dispatch(
                route.op_name or "app", route.act or "open_app",
                route.params, identity=self.identity,
            )
        except XiaobaiError as exc:
            if self.strategy == "cloud_only":
                return OperatorResult(
                    op="", act="", ok=False, code=exc.code.value, message=exc.message,
                )
            return self.engine.dispatch(
                route.op_name or "app", route.act or "open_app",
                route.params, identity=self.identity,
            )

        # 解析专家联盟返回：payload.op + payload.act + payload.params
        pl = reply.payload or {}
        if reply.code != ErrorCode.OK.value:
            return OperatorResult(
                op=pl.get("op", ""), act=pl.get("act", ""), ok=False,
                code=reply.code, message=reply.message or "专家联盟否决/无匹配算子",
                data=pl,
            )
        op_name = str(pl.get("op") or "")
        act = str(pl.get("act") or "")
        params = dict(pl.get("params") or {})
        exec_mode = str(pl.get("mode") or "local")  # local | remote
        if not op_name or not act:
            return OperatorResult(
                op="", act="", ok=False,
                code=ErrorCode.INTENT_UNKNOWN.value,
                message="专家联盟返回缺少 op/act",
                data=pl,
            )
        if exec_mode == "remote":
            # 远程算子：发送 EXEC→mox，并等待返回
            exec_env = ProxyEnvelope(msg_type=MsgType.EXEC, reply_to=reply.id,
                                     payload={"op": op_name, "act": act, "params": params})
            try:
                rr = await self._request(exec_env, timeout_ms=int(pl.get("timeout_ms", 10000)))
            except TimeoutError as exc:
                return OperatorResult(
                    op=op_name, act=act, ok=False,
                    code=ErrorCode.BRIDGE_DISCONNECTED.value,
                    message=f"远程算子执行超时：{exc}",
                )
            return OperatorResult(
                op=op_name, act=act, ok=(rr.code == ErrorCode.OK.value),
                code=rr.code, message=rr.message,
                data=rr.payload or {},
            )
        # 本地执行
        return self.engine.dispatch(op_name, act, params, identity=self.identity)

    # -------------------------------------------------------------- Transport: WebSocket (httpx + ws io)
    async def _connect_ws(self) -> None:
        # 优先用 httpx 的 WebSocket；缺失时退 websockets
        t: Transport | None = None
        try:
            import httpx  # type: ignore[import-not-found]
            t = HttpxWsTransport(self.mox_ws_url, httpx.AsyncClient(timeout=10.0))
        except Exception:  # noqa: BLE001
            try:
                import websockets  # type: ignore[import-not-found]
                t = PyWsTransport(self.mox_ws_url)
            except Exception as exc2:  # noqa: BLE001
                raise XiaobaiError(ErrorCode.MISSING_DEP,
                                   "voice_proxy 需要 httpx 或 websockets 之一：pip install httpx",
                                   cause=exc2) from exc2
        await t.connect()
        self._transport = t
        self._recv_task = asyncio.create_task(self._recv_loop())
        # 发送 hello
        await self._send(ProxyEnvelope(msg_type=MsgType.HELLO, payload={
            "version": "AIS-FR13/1.0",
            "node": platform.node(),
            "strategy": self.strategy,
            "supported_ops": [op.name for op in self.engine.list_operators()],
            "identity": {"user_id": self.identity.user_id, "role": self.identity.role,
                         "tenant_id": self.identity.tenant_id},
        }), timeout=5.0)
        self._last_hello_at = time.time()

    async def _recv_loop(self) -> None:
        assert self._transport is not None
        try:
            while True:
                env = await self._transport.recv(timeout=30.0)
                if env is None:
                    continue
                if env.msg_type == MsgType.PING:
                    await self._send(
                        ProxyEnvelope(msg_type=MsgType.PONG, reply_to=env.id, payload={}),
                        timeout=3.0,
                    )
                    continue
                # 解 pending
                if env.reply_to:
                    fut = self._pending.get(env.reply_to)
                    if fut is not None and not fut.done():
                        fut.set_result(env)
                        continue
                # mox → 桌面：主动 exec 命令（如远程协助 / 紧急关机）
                if env.msg_type == MsgType.EXEC:
                    asyncio.create_task(self._handle_exec_from_mox(env))
                    continue
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("recv loop 异常：%s", exc)
            self._status = BridgeStatus.DISCONNECTED

    async def _handle_exec_from_mox(self, env: ProxyEnvelope) -> None:
        pl = env.payload or {}
        op_name = str(pl.get("op") or "")
        act = str(pl.get("act") or "")
        params = dict(pl.get("params") or {})
        # L3 管理员才能下发反调，且需要强制鉴权身份（MoxAdmin）
        admin = Identity(role=str(pl.get("as_role") or "MoxAdmin"),
                         user_id=str(pl.get("by") or "mox-admin"),
                         tenant_id=self.identity.tenant_id)
        r = self.engine.dispatch(op_name, act, params, identity=admin)
        # 如有回调，触发（给 BallWidget 更新 executing 状态用）
        if self._on_exec_from_mox is not None:
            try:
                cb_r = self._on_exec_from_mox(f"{op_name}.{act}", dict(params))
                if asyncio.iscoroutine(cb_r):
                    await cb_r
            except Exception:  # noqa: BLE001
                log.exception("on_exec_from_mox 回调异常")
        # 回执 mox
        await self._send(ProxyEnvelope(
            msg_type=MsgType.ACK, reply_to=env.id,
            payload=r.to_dict(), code=r.code,
        ), timeout=3.0)

    async def _heartbeat_loop(self) -> None:
        while self._status in (BridgeStatus.CONNECTED, BridgeStatus.CONNECTING):
            await asyncio.sleep(15.0)
            if self._transport is None:
                break
            try:
                await self._send(ProxyEnvelope(msg_type=MsgType.PING, payload={}), timeout=3.0)
            except Exception:  # noqa: BLE001
                self._status = BridgeStatus.DISCONNECTED
                break

    async def _request(self, env: ProxyEnvelope, timeout_ms: int = 3000) -> ProxyEnvelope:
        assert self._loop is not None
        fut = self._loop.create_future()
        self._pending[env.id] = fut
        try:
            await self._send(env, timeout=2.0)
            return await asyncio.wait_for(fut, timeout=max(50, timeout_ms) / 1000.0)
        finally:
            self._pending.pop(env.id, None)

    async def _send(self, env: ProxyEnvelope, timeout: float) -> None:
        if self._transport is None:
            raise XiaobaiError(ErrorCode.BRIDGE_DISCONNECTED, "transport 未初始化")
        await self._transport.send(env, timeout=timeout)

    # -------------------------------------------------------------- Engine audit 回调 → AUDIT to mox
    def _engine_audit_cb_sync(self, op, act, params, identity, result):
        """同步（线程安全）触发审计上报；桥未连则直接丢弃（local_first 不阻塞）。"""
        if self._status != BridgeStatus.CONNECTED or self._loop is None or self._loop.is_closed():
            return
        payload = {
            "op": getattr(op, "name", None) or (result.op if result else ""),
            "act": act,
            "params": dict(params or {}),
            "identity": {"user_id": identity.user_id, "role": identity.role,
                         "tenant_id": identity.tenant_id},
            "result": result.to_dict() if result else None,
            "ts_ms": int(time.time() * 1000),
        }
        env = ProxyEnvelope(msg_type=MsgType.AUDIT, payload=payload)
        try:
            # fire-and-forget（非关键路径）
            fut = asyncio.run_coroutine_threadsafe(self._send(env, timeout=1.5), self._loop)
            # 最多阻塞 50ms，避免审计占用主线程
            fut.result(timeout=0.05)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Transport 实现：httpx 优先；websockets 回退
# ---------------------------------------------------------------------------

class HttpxWsTransport(Transport):
    def __init__(self, url: str, client) -> None:
        self.url = url
        self.client = client
        self.ws = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self.status = BridgeStatus.CONNECTING
        try:
            self.ws = await self.client.__aenter__() if False else None  # type: ignore[truthy-bool]
            # httpx >= 0.26 的 WebSocket API
            try:
                self.ws = await self.client.websocket(self.url).__aenter__()
            except Exception:
                # 旧版 httpx：httpx.ws_connect
                try:
                    self.ws = await self.client.ws_connect(self.url).__aenter__()
                except Exception:
                    # 直接用 request streaming（兜底）
                    resp = await self.client.request("GET", self.url, headers={"Upgrade": "websocket", "Connection": "Upgrade"})
                    raise XiaobaiError(ErrorCode.BRIDGE_DISCONNECTED,
                                       f"httpx 无法建 ws（{resp.status_code}），请升级 httpx")
            self.status = BridgeStatus.CONNECTED
        except XiaobaiError:
            self.status = BridgeStatus.DISCONNECTED
            raise
        except Exception as exc:  # noqa: BLE001
            self.status = BridgeStatus.DISCONNECTED
            raise XiaobaiError(ErrorCode.BRIDGE_DISCONNECTED,
                               f"httpx ws 握手失败：{exc}", cause=exc) from exc

    async def close(self) -> None:
        self.status = BridgeStatus.CLOSED
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
            self.ws = None

    async def send(self, env: ProxyEnvelope, timeout: float) -> None:
        if self.ws is None:
            raise XiaobaiError(ErrorCode.BRIDGE_DISCONNECTED, "httpx ws 未连接")
        async with self._lock:
            await asyncio.wait_for(self.ws.send_text(env.to_json()), timeout=timeout)

    async def recv(self, timeout: float) -> ProxyEnvelope | None:
        if self.ws is None:
            return None
        try:
            data = await asyncio.wait_for(self.ws.receive_text(), timeout=timeout)
            return ProxyEnvelope.from_json(data)
        except TimeoutError:
            return None
        except Exception as exc:  # noqa: BLE001
            raise XiaobaiError(ErrorCode.BRIDGE_DISCONNECTED, f"httpx ws recv 异常：{exc}", cause=exc) from exc


class PyWsTransport(Transport):
    def __init__(self, url: str) -> None:
        self.url = url
        self.ws = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        import websockets  # type: ignore[import-not-found]
        self.status = BridgeStatus.CONNECTING
        try:
            self.ws = await websockets.connect(self.url, ping_interval=30, ping_timeout=10)
            self.status = BridgeStatus.CONNECTED
        except Exception as exc:  # noqa: BLE001
            self.status = BridgeStatus.DISCONNECTED
            raise XiaobaiError(ErrorCode.BRIDGE_DISCONNECTED,
                               f"websockets 握手失败：{exc}", cause=exc) from exc

    async def close(self) -> None:
        self.status = BridgeStatus.CLOSED
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
            self.ws = None

    async def send(self, env: ProxyEnvelope, timeout: float) -> None:
        if self.ws is None:
            raise XiaobaiError(ErrorCode.BRIDGE_DISCONNECTED, "websocket 未连接")
        async with self._lock:
            await asyncio.wait_for(self.ws.send(env.to_json()), timeout=timeout)

    async def recv(self, timeout: float) -> ProxyEnvelope | None:
        if self.ws is None:
            return None
        try:
            data = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            return ProxyEnvelope.from_json(data)
        except TimeoutError:
            return None
        except Exception as exc:  # noqa: BLE001
            raise XiaobaiError(ErrorCode.BRIDGE_DISCONNECTED, f"ws recv 异常：{exc}", cause=exc) from exc

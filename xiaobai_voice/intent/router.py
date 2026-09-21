"""意图路由（PPR 激活扩散的工程化近似 — 词典+正则+TF 的弱监督实现）。

AIS 规格定义：FR-12/FR-13 要求 T13 图谱漂移率 = 0，理想做法是对 <schema：顶点=算子动作，
边=语义近邻/共现/上位-下位> 做 Personalized PageRank。工程化分三阶段：
    S1（最小交付，当前）：关键词 → 算子动作的 1:1/N 映射，命中后按权重 confidence ∈ [0,1]。
    S2：引入 FAISS/hnswlib 语义向量化（BGE-M3），做 Top-K 候选 → 再交由联盟裁决。
    S3：真实 PPR 激活扩散，基于图谱 Schema 拓扑与 query 种子，算各动作稳态概率。

对 INTENT_AMBIGUOUS（多候选且 conf 差 < 0.1）的处理：
    - local_first 仍选 conf 最高者执行，同时打 WARN 留痕给 audit_cb；
    - cloud_only / cloud_fallback 直接返回 route.ambiguous=True 交给专家联盟裁决。
"""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any

from .base import Identity, OperatorEngine, AccessLevel


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    op_name: str = ""
    act: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    ambiguous: bool = False
    candidates: list[dict] = field(default_factory=list)  # [{op,act,conf,params}]
    matched_rule: str = ""

    def as_dict(self) -> dict:
        return {
            "op": self.op_name,
            "act": self.act,
            "params": self.params,
            "confidence": round(self.confidence, 4),
            "ambiguous": self.ambiguous,
            "candidates": self.candidates,
            "matched_rule": self.matched_rule,
        }


# ---------------------------------------------------------------------------
# 规则字典（每条规则声明一个正则 + 对应 op/act + 参数抽取命名组）
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    name: str
    op: str
    act: str
    pattern: re.Pattern
    # 从 pattern 的 named group 映射到 params：可选转换函数
    transforms: dict[str, callable] = field(default_factory=dict)  # type: ignore[valid-type]
    base_confidence: float = 0.92
    required_level: AccessLevel = AccessLevel.L1_USER


_RULES: list[Rule] = []


def _r(name: str, op: str, act: str, pat: str,
       required_level: AccessLevel = AccessLevel.L1_USER,
       base_confidence: float = 0.92,
       transforms: dict[str, Any] | None = None) -> Rule:
    return Rule(
        name=name, op=op, act=act,
        pattern=re.compile(pat, flags=re.IGNORECASE | re.UNICODE),
        required_level=required_level,
        base_confidence=base_confidence,
        transforms=dict(transforms or {}),
    )


def _build_default_rules() -> list[Rule]:
    return [
        # --------- volume 算子（L1） -----------------------------------------
        _r("vol.get", "volume", "get_volume",
           r"(当前|现在|系统)音量|音量.*多大|多大声", base_confidence=0.95),
        _r("vol.set_pct", "volume", "set_volume",
           r"把?音量(调(到|成)?|开(到|成)?|设(到|成)?)?\s*(?P<value>[0-9]{1,3})\s*(%|个|格)?|声音(?P<value2>[0-9]{1,3})",
           transforms={"value": lambda m: m.group("value2") if m.group("value2") else m.group("value")}),
        _r("vol.relative", "volume", "set_volume",
           r"音量(加|大|提高|升|往上|调高)(?P<plus>[0-9]{1,2})|音量(减|小|降|调低|往下)(?P<minus>[0-9]{1,2})",
           transforms={"value": lambda m: f"+{m.group('plus')}" if m.group("plus") else f"-{m.group('minus')}"}),
        _r("vol.mute", "volume", "mute", r"静音(开启|打开|一下)?|别出声|禁声|闭嘴"),
        _r("vol.unmute", "volume", "unmute", r"(取消|解除|去掉)静音|(开|恢复)声音|出声"),
        _r("vol.toggle", "volume", "toggle_mute", r"切换静音|切静音"),

        # --------- app 算子（L1 / L3） ---------------------------------------
        _r("app.open", "app", "open_app",
           r"(打开|启动|运行|开一下|点开)\s*(?P<target>[\u4e00-\u9fa5A-Za-z0-9_.\-·：:（）()/\\]+)",
           transforms={"target": _resolve_alias_app}, base_confidence=0.88),
        _r("app.close", "app", "close_app",
           r"(关闭|关掉|结束|停止|杀)\s*(进程|应用)?\s*(?P<name>[\u4e00-\u9fa5A-Za-z0-9_.\-·（）()]+)",
           required_level=AccessLevel.L3_ADMIN, base_confidence=0.86,
           transforms={"name": _app_exe_normalize}),
        _r("app.list", "app", "list_running", r"(列|查看|看一下|看看|列出).*进程|任务(列表|管理器)|开了什么",
           base_confidence=0.9, required_level=AccessLevel.L0_PUBLIC),
        _r("app.open_file", "app", "open_file_with_app",
           r"(打开|浏览|查看).*(文件|目录|文件夹|C:|D:|E:|/Users|/home|/tmp|桌面|文档|下载)",
           transforms={"path": _extract_path_from_text}, base_confidence=0.85),

        # --------- file 算子（L0/L1/L2/L3） ----------------------------------
        _r("file.copy_txt", "file", "copy_to_clipboard",
           r"(把|将|帮我)\s*(?P<text1>.*?)\s*(复制|拷|粘)到(剪贴板|剪切板)|复制(?P<text2>.*)",
           required_level=AccessLevel.L2_POWER, base_confidence=0.82,
           transforms={"text": lambda m: (m.group("text2") or m.group("text1") or "").strip() or None}),
        _r("file.copy_file", "file", "copy_to_clipboard",
           r"复制(内容)?文件(?P<source_file>\S+)",
           required_level=AccessLevel.L2_POWER, base_confidence=0.88),
        _r("file.delete", "file", "move_to_trash",
           r"(删除|删掉|清掉|移除)\s*(文件|目录)?\s*(?P<path>\S+)",
           required_level=AccessLevel.L3_ADMIN, base_confidence=0.90),
        _r("file.read", "file", "read_text_head",
           r"(读|看|查看|预览).*(文件)?\s*(?P<path>\S+)",
           required_level=AccessLevel.L0_PUBLIC, base_confidence=0.78,
           transforms={"lines": lambda m: 30}),
        _r("file.exists", "file", "file_exists",
           r"路径?\s*(?P<path>\S+)\s*存在吗|有没有文件(?P<path2>\S+)",
           required_level=AccessLevel.L0_PUBLIC, base_confidence=0.92,
           transforms={"path": lambda m: m.group("path2") or m.group("path")}),

        # --------- input 算子（L2 / L3） -------------------------------------
        _r("inp.type", "input", "type_text",
           r"(输入|键入|打字|写|敲入)\s*(：|:)?\s*(?P<text>.*)",
           required_level=AccessLevel.L2_POWER, base_confidence=0.80),
        _r("inp.click", "input", "mouse_click",
           r"(单击|点击|点一下|点)\s*(?P<button>左键|右键|中键)?",
           required_level=AccessLevel.L2_POWER, base_confidence=0.88,
           transforms={"button": _cn2btn, "clicks": lambda _: 1}),
        _r("inp.dblclick", "input", "mouse_click",
           r"双击|点两下", required_level=AccessLevel.L2_POWER, base_confidence=0.92,
           transforms={"button": lambda _: "left", "clicks": lambda _: 2}),
        _r("inp.move", "input", "mouse_move",
           r"鼠标(移动|移到|挪到|去)\s*\(?(?P<x>\d{1,5})\s*[,， ]\s*(?P<y>\d{1,5})\)?",
           required_level=AccessLevel.L2_POWER, base_confidence=0.95,
           transforms={"x": lambda m: int(m.group("x")), "y": lambda m: int(m.group("y"))}),
        _r("inp.pos", "input", "mouse_position", r"鼠标在哪|鼠标坐标|鼠标位置",
           required_level=AccessLevel.L0_PUBLIC, base_confidence=0.95),
        _r("inp.key", "input", "press_key",
           r"按(一下|下)?键?\s*(?P<key>[A-Za-z0-9]+|Enter|回车|Space|空格|Esc|Escape|Tab|Backspace|Delete|Insert|Home|End|PgUp|PgDn|Up|Down|Left|Right)",
           required_level=AccessLevel.L2_POWER, base_confidence=0.92,
           transforms={"key": lambda m: _key_cn(m.group("key"))}),
        _r("inp.hotkey", "input", "hotkey",
           r"(?P<keys>(ctrl|alt|shift|cmd|win|command)\s*[+＋]\s*([A-Za-z0-9]+|Enter|回车|Space|空格|Esc|Tab|Backspace|Delete|Insert|Home|End|PgUp|PgDn|Up|Down|Left|Right)(\s*[+＋]\s*([A-Za-z0-9]+|Enter|回车|Space|空格|Esc|Tab|Backspace|Delete|Insert|Home|End|PgUp|PgDn|Up|Down|Left|Right))*)",
           required_level=AccessLevel.L2_POWER, base_confidence=0.95,
           transforms={"keys": _keys_cn}),
        _r("inp.drag", "input", "mouse_drag",
           r"从?\(?(?P<x1>\d{1,5})\s*[,，]\s*(?P<y1>\d{1,5})\)?\s*(拖|拉|拖拽)到?\(?(?P<x2>\d{1,5})\s*[,，]\s*(?P<y2>\d{1,5})\)?",
           required_level=AccessLevel.L3_ADMIN, base_confidence=0.93,
           transforms={
               "x1": lambda m: int(m.group("x1")),
               "y1": lambda m: int(m.group("y1")),
               "x2": lambda m: int(m.group("x2")),
               "y2": lambda m: int(m.group("y2")),
           }),
        _r("inp.screenshot", "input", "screenshot", r"截屏|截图|抓屏|屏幕快照",
           required_level=AccessLevel.L3_ADMIN, base_confidence=0.96),
    ]


# ---------------------------------------------------------------------------
# 工具：别名/正则抽取/归一化
# ---------------------------------------------------------------------------

_APP_ALIAS = {
    "记事本": "notepad",
    "笔记本": "notepad",
    "画图": "mspaint",
    "计算器": "calc",
    "计算器程序": "calc",
    "命令行": "cmd",
    "cmd": "cmd",
    "终端": "wt",
    "资源管理器": "explorer",
    "文件管理器": "explorer",
    "此电脑": "explorer",
    "浏览器": "chrome",
    "谷歌浏览器": "chrome",
    "谷歌": "chrome",
    "chrome": "chrome",
    "edge": "msedge",
    "微软浏览器": "msedge",
    "设置": "ms-settings:",
    "控制面板": "control",
    "任务管理器": "taskmgr",
    "vscode": "code",
    "vs code": "code",
    "代码编辑器": "code",
    "微信": "wechat",
    "微信电脑版": "wechat",
    "企业微信": "wemeetapp",
    "钉钉": "dingtalklauncher",
    "飞书": "lark",
    "obsidian": "obsidian",
    "typora": "typora",
    "excel": "excel",
    "word": "winword",
    "powerpoint": "powerpnt",
    "ppt": "powerpnt",
    "outlook": "outlook",
    "wps": "wps",
    "火狐": "firefox",
    "firefox": "firefox",
    "音乐": "potplayer",
    "播放器": "potplayer",
}


def _resolve_alias_app(match: re.Match) -> str | None:
    target = (match.group("target") or "").strip("。!！?？,，")
    if not target:
        return None
    v = _APP_ALIAS.get(target.lower()) or _APP_ALIAS.get(target)
    return v or target


def _app_exe_normalize(match: re.Match) -> str:
    name = (match.group("name") or "").strip()
    if not name:
        return ""
    v = _APP_ALIAS.get(name.lower()) or _APP_ALIAS.get(name)
    base = v or name
    if sys.platform.startswith("win"):  # type: ignore[name-defined]
        if not base.lower().endswith(".exe") and "/" not in base and "\\" not in base:
            return f"{base}.exe"
    return base


def _extract_path_from_text(match: re.Match) -> str:
    # 简单抓出明显的路径片段；更精准请直接使用 open_file_with_app+path 参数
    text = match.group(0)
    m = re.search(r"(?P<p>([A-Za-z]:[\\/]|~/|/|[.]{1,2}/)[^\s'\"，。,；;]+)", text)
    return (m.group("p") if m else "").strip()


def _cn2btn(match: re.Match) -> str:
    s = (match.group("button") or "left").strip()
    return {"左键": "left", "右键": "right", "中键": "middle"}.get(s, s) or "left"


_KEY_CN = {
    "回车": "enter", "空格": "space", "退格": "backspace",
    "删除": "delete", "插入": "insert", "制表": "tab",
    "上": "up", "下": "down", "左": "left", "右": "right",
    "换码": "esc", "逃逸": "esc",
}


def _key_cn(s: str) -> str:
    return _KEY_CN.get(s.strip()) or s.strip().lower()


def _keys_cn(match: re.Match) -> list[str]:
    s = (match.group("keys") or "").replace("＋", "+")
    return [_key_cn(p.strip()) for p in s.split("+") if p.strip()]


# 避免 import sys 在每个 transform 时重复
import sys  # noqa: E402  (放在使用后，但被函数体内捕获；此 import 为模块级，确保 _app_exe_normalize 可用)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class IntentRouter:
    """基于规则的轻量意图路由器（S1 最小交付实现）。

    用法：
        r = IntentRouter().route("打开记事本", identity)
        print(r.op_name, r.act, r.params)  # app, open_app, {"target": "notepad"}
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = dict(cfg or {})
        if not _RULES:
            _RULES.extend(_build_default_rules())
        # 可选：额外的自定义规则（动态热更新）
        self.extra_rules: list[Rule] = []
        self.ambiguous_threshold: float = float(self.cfg.get("ambiguous_threshold") or 0.1)

    def add_rule(self, rule: Rule) -> None:
        self.extra_rules.append(rule)

    # --------------------------------------------------------------
    def route(self, text: str, identity: Identity | None = None) -> RouteResult:
        text = (text or "").strip()
        if not text:
            return RouteResult()
        ident = identity or Identity()
        candidates: list[dict] = []
        for rule in [*_RULES, *self.extra_rules]:
            m = rule.pattern.search(text)
            if not m:
                continue
            params: dict[str, Any] = {}
            # named groups 先拉
            for k, v in m.groupdict().items():
                if v is None:
                    continue
                fn = rule.transforms.get(k)
                try:
                    params[k] = fn(m) if callable(fn) else v
                except Exception:  # noqa: BLE001
                    params[k] = v
            # 未命名的额外 transform（如 clicks=1 / lines=30 / keys=list）
            for k, fn in rule.transforms.items():
                if k in params or not callable(fn):
                    continue
                try:
                    params[k] = fn(m)
                except Exception:  # noqa: BLE001
                    pass

            conf = float(rule.base_confidence)
            # 如果身份等级 < 动作所需等级：降低 confidence（最终 dispatch 再由 Engine 做真实鉴权）
            if ident.level < rule.required_level:
                conf *= 0.6  # 仍保留为候选，让联盟裁决决定是否提权

            # 非中文数字等 token 匹配长度越长越好
            span_len = max(1, m.end() - m.start())
            conf = min(1.0, conf + min(0.06, 0.005 * span_len))

            candidates.append({
                "op": rule.op, "act": rule.act, "params": params,
                "confidence": round(conf, 4), "rule": rule.name,
                "required_level": rule.required_level.value,
            })

        if not candidates:
            return RouteResult()

        # 按 conf 降序
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        top = candidates[0]
        # ambiguity：若前 2 个 conf 差 < threshold
        ambiguous = False
        if len(candidates) >= 2:
            if candidates[0]["confidence"] - candidates[1]["confidence"] < self.ambiguous_threshold:
                ambiguous = True

        return RouteResult(
            op_name=top["op"], act=top["act"], params=top["params"],
            confidence=top["confidence"], ambiguous=ambiguous,
            candidates=candidates[:5], matched_rule=top["rule"],
        )

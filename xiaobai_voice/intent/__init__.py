"""意图模块：PPR 激活扩散路由（S1 规则工程化实现）。"""
from .router import IntentRouter, RouteResult, Rule, _build_default_rules  # noqa: F401

__all__ = ["IntentRouter", "RouteResult", "Rule"]

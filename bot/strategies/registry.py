"""Strategy registry: maps config names to classes; instantiates from YAML."""

from __future__ import annotations

import importlib
from typing import Type

from loguru import logger

from bot.strategies.base import Strategy


def get_strategy_class(name: str) -> Type[Strategy]:
    """Resolve a strategy name ('ema_rsi_trend', 'ml_classifier', or any
    '<module>.<Class>' dotted path inside the bot.strategies package)."""
    dotted = {
        "ema_rsi_trend": "bot.strategies.trend:EMARsiTrend",
        "bollinger_reversion": "bot.strategies.mean_reversion:BollingerReversion",
        "atr_breakout": "bot.strategies.breakout:ATRBreakout",
        "ml_classifier": "bot.strategies.ml_strategy:MLClassifierStrategy",
    }
    target = dotted.get(name, name if "." in name else None)
    if not target:
        raise ValueError(f"unknown strategy {name!r}")
    module_name, class_name = target.split(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, Strategy)):
        raise ValueError(f"{target} is not a Strategy subclass")
    return cls


def build_strategies_from_config(strategies_cfg: dict) -> list[Strategy]:
    """Instantiate every enabled strategy from the config section."""
    out: list[Strategy] = []
    for name, spec in strategies_cfg.items():
        spec = spec or {}
        if not spec.get("enabled", False):
            logger.info("strategy {} disabled in config — skipping", name)
            continue
        params = dict(spec.get("params") or {})
        try:
            cls = get_strategy_class(name)
            strategy = cls(name=name, params=params)
            out.append(strategy)
            logger.info("loaded strategy {} (weight {}) with params {}",
                        name, spec.get("weight", 1.0), params)
        except Exception as exc:
            logger.error("failed to load strategy {}: {} — skipping", name, exc)
    return out

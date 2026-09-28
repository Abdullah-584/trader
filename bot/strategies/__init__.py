"""Strategies package: plugin interface, registry, and built-ins."""

from bot.strategies.base import Strategy, empty_signal_frame
from bot.strategies.registry import build_strategies_from_config, get_strategy_class

__all__ = ["Strategy", "empty_signal_frame", "get_strategy_class",
           "build_strategies_from_config"]

"""Backtesting engine: event-driven simulation, metrics, validation guards."""

from bot.engine.backtester import BacktestResult, EventDrivenBacktester
from bot.engine.metrics import compute_metrics, max_drawdown, too_good_warnings

__all__ = [
    "BacktestResult",
    "EventDrivenBacktester",
    "compute_metrics",
    "max_drawdown",
    "too_good_warnings",
]

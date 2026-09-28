"""Monitor package — analysis-only signal monitor for manual binary-option
style decisions. Places no orders and automates no broker."""

from bot.monitor.quotex import QuotexSignalMonitor, Suggestion, breakeven_hit_rate

__all__ = ["QuotexSignalMonitor", "Suggestion", "breakeven_hit_rate"]

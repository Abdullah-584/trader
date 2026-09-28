"""Performance metrics for backtests and paper trading.

All annualized figures assume zero risk-free rate and derive the periods
per year from the actual bar spacing in the equity index. Metrics are
estimates — they say nothing about future performance, and results that
look spectacular are far more likely to signal a bug or overfitting than
an edge. ``too_good_warnings`` exists to make that explicit.
"""

from __future__ import annotations

import math

import pandas as pd

from bot.core.models import Trade

MIN_TRADES_FOR_SIGNIFICANCE = 30


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    seconds = (equity.index[-1] - equity.index[0]).total_seconds()
    years = seconds / (365.25 * 24 * 3600)
    if years <= 0:
        return 0.0
    final = float(equity.iloc[-1])
    if final <= 0:
        return -1.0
    return float((final / equity.iloc[0]) ** (1.0 / years) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    """Return the max drawdown as a negative fraction (e.g. -0.23)."""
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Annualization factor from the median bar spacing."""
    if len(index) < 3:
        return 252.0
    median = index.to_series().diff().dropna().median()
    seconds = median.total_seconds()
    if seconds <= 0:
        return 252.0
    return (365.25 * 24 * 3600.0) / seconds


def sharpe_ratio(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    std = float(rets.std(ddof=1))
    if std == 0 or math.isnan(std):
        return 0.0
    return float(rets.mean() / std * math.sqrt(periods_per_year(equity.index)))


def sortino_ratio(equity: pd.Series) -> float:
    rets = equity.pct_change().dropna()
    downside = rets[rets < 0]
    if len(rets) < 2 or downside.empty:
        return 0.0
    dstd = float(downside.std(ddof=1))
    if dstd == 0 or math.isnan(dstd):
        return 0.0
    return float(rets.mean() / dstd * math.sqrt(periods_per_year(equity.index)))


def trade_stats(trades: list[Trade]) -> dict:
    """Win rate, profit factor, expectancy and friends from round trips."""
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "payoff_ratio": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
        }
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    win_rate = len(wins) / n
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0)
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = -gross_loss / len(losses) if losses else 0.0  # negative
    payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf") if avg_win > 0 else 0.0
    return {
        "n_trades": n,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": sum(pnls) / n,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def compute_metrics(equity: pd.Series, trades: list[Trade]) -> dict:
    """Full metric dict for an equity curve and its round-trip trades."""
    eq = equity.dropna()
    stats = trade_stats(trades)
    out = {
        "start_equity": float(eq.iloc[0]) if len(eq) else 0.0,
        "end_equity": float(eq.iloc[-1]) if len(eq) else 0.0,
        "total_return_pct": total_return(eq) * 100.0,
        "cagr_pct": cagr(eq) * 100.0,
        "sharpe": sharpe_ratio(eq),
        "sortino": sortino_ratio(eq),
        "max_drawdown_pct": max_drawdown(eq) * 100.0,
        "periods_per_year": periods_per_year(eq.index) if len(eq.index) >= 3 else 0.0,
        **stats,
    }
    return out


def too_good_warnings(metrics: dict) -> list[str]:
    """Human-readable warnings when results look statistically suspicious.

    Backtests overstate live performance. Results far better than what
    good professionals achieve usually mean look-ahead bias, survivorship
    bias, overfitting, or an unrealistic fill model — not a money machine.
    """
    warnings: list[str] = []
    n = metrics.get("n_trades", 0)

    if n < MIN_TRADES_FOR_SIGNIFICANCE:
        warnings.append(
            f"Only {n} trades — far too few to distinguish skill from luck "
            f"(need >= {MIN_TRADES_FOR_SIGNIFICANCE}).")
    if metrics.get("sharpe", 0) > 3.0:
        warnings.append(
            f"Sharpe {metrics['sharpe']:.2f} > 3.0 — world-class funds rarely "
            "sustain this. Suspect lookahead bias or unrealistic fills.")
    if metrics.get("profit_factor", 0) > 4.0 and n >= 5:
        warnings.append(
            f"Profit factor {metrics['profit_factor']:.2f} > 4.0 — unusually "
            "high; verify fills, fees and slippage assumptions.")
    if metrics.get("win_rate", 0) > 0.80 and n >= MIN_TRADES_FOR_SIGNIFICANCE:
        warnings.append(
            f"Win rate {metrics['win_rate']:.0%} — extremely high; check that "
            "stop-outs are simulated intrabar, not just at close.")
    if metrics.get("total_return_pct", 0) > 500 and metrics.get("max_drawdown_pct", -100) > -5:
        warnings.append(
            "Return > 500% with max drawdown < 5% — this combination is "
            "almost always a simulation artifact.")
    return warnings

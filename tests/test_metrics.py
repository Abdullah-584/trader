"""Tests for backtest metrics math and the too-good-to-be-true warnings."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from bot.core.models import OrderReason, Side, Trade
from bot.engine.metrics import (
    cagr,
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    too_good_warnings,
    total_return,
    trade_stats,
)


def _equity(values, freq="1h", start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def _trade(pnl: float) -> Trade:
    return Trade(
        symbol="X", side=Side.BUY, quantity=1.0, entry_price=100.0,
        exit_price=100.0 + pnl, entry_time=pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(),
        exit_time=pd.Timestamp("2024-01-02", tz="UTC").to_pydatetime(),
        fees_paid=0.1, pnl=pnl, pnl_pct=pnl / 100.0, reason=OrderReason.STOP,
    )


# --------------------------------------------------------------------- returns
def test_total_return():
    assert total_return(_equity([100.0, 110.0, 121.0])) == pytest.approx(0.21)


def test_max_drawdown():
    # 100 -> 120 -> 90 -> 95 : drawdown = 90/120 - 1 = -25%
    eq = _equity([100.0, 120.0, 90.0, 95.0])
    assert max_drawdown(eq) == pytest.approx(-0.25)


def test_cagr_doubling_over_two_years():
    eq = _equity([100.0] + [100.0] * 730, freq="1D")  # flat for 2 years
    eq.iloc[-1] = 200.0
    growth = cagr(eq)
    assert growth == pytest.approx(math.sqrt(2) - 1, rel=0.01)


def test_sharpe_zero_when_no_variance():
    # perfectly flat equity -> all returns exactly 0 -> zero std -> sharpe 0
    assert sharpe_ratio(_equity([100.0] * 10)) == 0.0


def test_sortino_ignores_upside():
    eq = _equity([100.0, 104.0, 100.0, 106.0, 102.0, 108.0])
    s = sortino_ratio(eq)
    assert math.isfinite(s) and s > 0


# ----------------------------------------------------------------- trade stats
def test_trade_stats_basic():
    stats = trade_stats([_trade(50), _trade(30), _trade(-20)])
    assert stats["n_trades"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["profit_factor"] == pytest.approx(4.0)
    assert stats["expectancy"] == pytest.approx(20.0)
    assert stats["avg_win"] == pytest.approx(40.0)
    assert stats["avg_loss"] == pytest.approx(-20.0)
    assert stats["payoff_ratio"] == pytest.approx(2.0)


def test_trade_stats_empty():
    stats = trade_stats([])
    assert stats["n_trades"] == 0
    assert stats["win_rate"] == 0.0


def test_compute_metrics_shapes():
    eq = _equity([1000.0, 1010.0, 1005.0, 1020.0])
    m = compute_metrics(eq, [_trade(10), _trade(-5)])
    assert m["start_equity"] == 1000.0
    assert m["end_equity"] == 1020.0
    assert m["total_return_pct"] == pytest.approx(2.0)
    assert m["n_trades"] == 2
    assert "sharpe" in m and "sortino" in m and "max_drawdown_pct" in m


# ------------------------------------------------------------- honesty warnings
def test_too_good_warnings_fire():
    m = {"n_trades": 5, "sharpe": 5.0, "profit_factor": 6.0, "win_rate": 0.9,
         "total_return_pct": 900, "max_drawdown_pct": -2}
    warnings = too_good_warnings(m)
    assert len(warnings) >= 4  # few trades, sharpe, PF, return/DD combo


def test_too_good_warnings_quiet_on_reasonable_results():
    m = {"n_trades": 60, "sharpe": 1.1, "profit_factor": 1.6, "win_rate": 0.52,
         "total_return_pct": 18, "max_drawdown_pct": -12}
    assert too_good_warnings(m) == []

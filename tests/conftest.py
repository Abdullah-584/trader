"""Shared fixtures: deterministic synthetic candles and config helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_candles(n: int = 600, seed: int = 42, start_price: float = 100.0,
                 drift: float = 0.0002, vol: float = 0.01) -> pd.DataFrame:
    """Deterministic geometric random walk OHLCV frame (1h bars)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start_price * np.cumprod(1 + rets)
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0, vol / 2, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(900, 1100, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def candles() -> pd.DataFrame:
    return make_candles()


@pytest.fixture
def risk_cfg():
    from bot.core.config import RiskConfig
    return RiskConfig(
        risk_per_trade=0.01, max_positions=3, max_daily_loss_pct=0.03,
        max_drawdown_pct=0.15, stop_atr_multiple=2.0, take_profit_r_multiple=2.0,
        trailing_stop=True, trailing_atr_multiple=3.0, min_stop_distance_pct=0.002,
        max_position_pct=0.25, fees_pct=0.001, slippage_pct=0.0005,
    )

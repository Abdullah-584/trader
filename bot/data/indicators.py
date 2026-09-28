"""Vectorized indicators (pure pandas, no external TA dependency).

Conventions:
* Functions are pure: they never mutate the input frame.
* Wilder smoothing (RSI/ATR) uses ewm(alpha=1/n, adjust=False) — the
  standard TA implementation.
* STRATEGIES MUST SHIFT INDICATOR OUTPUTS by one bar before acting on
  them (decide on bar close, execute on next bar open). The backtester
  enforces this too, but correctness starts here.
"""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI in [0, 100].

    avg_loss == 0 (no down moves) naturally yields RSI = 100 via inf
    division, matching the standard convention. NaN only at the head.
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss  # inf when avg_loss == 0 -> RSI 100
    out = 100 - 100 / (1 + rs)
    return out.astype(float)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR from OHLC columns."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def bollinger(close: pd.Series, period: int = 20, std: float = 2.0
              ) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    return mid - std * sd, mid, mid + std * sd


def donchian(df: pd.DataFrame, period: int = 55) -> tuple[pd.Series, pd.Series]:
    """Channel high/low of the PREVIOUS ``period`` bars (excludes current bar
    so a breakout test is not trivially true)."""
    upper = df["high"].rolling(period).max().shift(1)
    lower = df["low"].rolling(period).min().shift(1)
    return upper, lower


def roc(series: pd.Series, period: int) -> pd.Series:
    """Rate of change in percent."""
    return series.pct_change(period) * 100.0

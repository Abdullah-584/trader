"""Bollinger Band mean-reversion strategy (long-only, spot-safe)."""

from __future__ import annotations

import pandas as pd

from bot.data.indicators import bollinger, rsi
from bot.strategies.base import Strategy


class BollingerReversion(Strategy):
    """Buy stretched-to-the-downside moves back toward the mean.

    Long when close closes back above the lower band after being below it
    (RSI-confirmed oversold by default); exit at the middle band or when the
    upper band pings (overbought).
    """

    def __init__(self, name: str | None = None, params: dict | None = None):
        super().__init__(name, params)
        self.period = int(self.params.get("period", 20))
        self.std = float(self.params.get("std", 2.0))
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.rsi_confirm = bool(self.params.get("rsi_confirm", True))
        self.atr_period = int(self.params.get("atr_period", 14))
        self.atr_multiplier = float(self.params.get("atr_multiplier", 2.0))

    def warmup_bars(self) -> int:
        return max(self.period, self.rsi_period, self.atr_period) + 5

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        lower, mid, upper = bollinger(df["close"], self.period, self.std)
        r = rsi(df["close"], self.rsi_period)
        from bot.data.indicators import atr as _atr  # local import avoids cycles
        a = _atr(df, self.atr_period)

        below = df["close"] < lower
        cross_up = (df["close"] > lower) & below.shift(1, fill_value=False)
        oversold = (r < 35) if self.rsi_confirm else pd.Series(True, index=df.index)
        entry = cross_up & oversold.fillna(False)

        exit_mid = (df["close"] > mid) & (df["close"].shift(1) <= mid.shift(1))
        exit_upper = df["close"] > upper
        exit_sig = (exit_mid | exit_upper).shift(1, fill_value=False) & (df["close"] > lower)

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[entry] = 1
        sig[exit_sig] = -1  # explicit EXIT vote (short-bias) for the aggregator
        conf = pd.Series(0.0, index=df.index)
        depth = ((lower - df["close"]).abs() / (df["close"] * self.std)).clip(0, 1)
        conf[entry] = (0.55 + 0.35 * depth[entry]).clip(0, 1)
        conf[exit_sig] = 0.5

        out = pd.DataFrame({
            "signal": sig,
            "confidence": conf,
            "stop_distance": (a * self.atr_multiplier).clip(lower=0.0),
        })
        return self._validate_output(out, df)

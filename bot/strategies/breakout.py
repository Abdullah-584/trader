"""ATR breakout strategy (Donchian channel + ATR filter)."""

from __future__ import annotations

import pandas as pd

from bot.data.indicators import atr, donchian
from bot.strategies.base import Strategy


class ATRBreakout(Strategy):
    """Trade momentum continuations through a Donchian channel.

    Long when the close breaks ABOVE the previous N-bar high by more than
    ``atr_multiplier * ATR`` (a real expansion, not noise), with a simple
    breakout-age exit. Stop distance scales with ATR.
    """

    def __init__(self, name: str | None = None, params: dict | None = None):
        super().__init__(name, params)
        self.lookback = int(self.params.get("lookback", 55))
        self.atr_period = int(self.params.get("atr_period", 14))
        self.atr_multiplier = float(self.params.get("atr_multiplier", 2.5))
        self.exit_bars = int(self.params.get("exit_bars", 20))

    def warmup_bars(self) -> int:
        return max(self.lookback, self.atr_period) + 5

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        upper, _lower = donchian(df, self.lookback)
        a = atr(df, self.atr_period)

        breakout = (df["close"] > upper) & \
                   ((df["close"] - upper) > self.atr_multiplier * a)

        # stateful position tracking so we emit an entry once, exit once
        sig = pd.Series(0, index=df.index, dtype=int)
        conf = pd.Series(0.0, index=df.index)
        in_pos = False
        bars_in = 0
        for i in range(len(df)):
            if not in_pos:
                if bool(breakout.iloc[i]):
                    in_pos = True
                    bars_in = 0
                    sig.iloc[i] = 1
                    # conviction from breakout strength relative to ATR
                    strength = (df["close"].iloc[i] - upper.iloc[i]) / max(a.iloc[i], 1e-9)
                    conf.iloc[i] = float(min(0.9, 0.6 + 0.05 * strength))
            else:
                bars_in += 1
                if bars_in >= self.exit_bars or bool((df["close"] < _lower).iloc[i]):
                    in_pos = False
                    sig.iloc[i] = -1  # EXIT vote
                    conf.iloc[i] = 0.4

        out = pd.DataFrame({
            "signal": sig,
            "confidence": conf,
            "stop_distance": (a * self.atr_multiplier).clip(lower=0.0),
        })
        return self._validate_output(out, df)

"""EMA crossover + RSI filter trend strategy (long-only, spot-safe)."""

from __future__ import annotations

import pandas as pd

from bot.data.indicators import atr, ema, rsi
from bot.strategies.base import Strategy


class EMARsiTrend(Strategy):
    """Classic trend following: fast EMA above slow EMA and not overbought.

    * Enter when the fast EMA crosses above the slow EMA while RSI is below
      ``rsi_buy_max`` (avoid chasing blow-off tops).
    * Exit on a death cross or when RSI falls below ``rsi_sell_min``.
    * Stop distance is ATR-scaled.
    """

    def __init__(self, name: str | None = None, params: dict | None = None):
        super().__init__(name, params)
        self.fast = int(self.params.get("fast", 21))
        self.slow = int(self.params.get("slow", 55))
        self.rsi_period = int(self.params.get("rsi_period", 14))
        self.rsi_buy_max = float(self.params.get("rsi_buy_max", 70))
        self.rsi_sell_min = float(self.params.get("rsi_sell_min", 30))
        self.atr_period = int(self.params.get("atr_period", 14))
        self.atr_multiplier = float(self.params.get("atr_multiplier", 2.0))

    def warmup_bars(self) -> int:
        return max(self.fast, self.slow, self.rsi_period, self.atr_period) + 5

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = ema(df["close"], self.fast)
        ema_slow = ema(df["close"], self.slow)
        r = rsi(df["close"], self.rsi_period)
        a = atr(df, self.atr_period)

        above = ema_fast > ema_slow
        # shift(1): the cross is only known at the bar's close — the engine
        # will execute it at the NEXT bar's open.
        golden_cross = above & ~above.shift(1, fill_value=False)
        death_cross = (~above) & above.shift(1, fill_value=False)

        entry = golden_cross & (r < self.rsi_buy_max)
        exit_sig = death_cross | (r < self.rsi_sell_min)

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[entry] = 1
        sig[exit_sig] = -1  # explicit EXIT vote (short-bias) for the aggregator

        conf = pd.Series(0.0, index=df.index)
        # conviction: EMA separation normalized by price
        sep = ((ema_fast - ema_slow) / df["close"]).clip(0, 0.05) / 0.05
        conf[entry] = (0.5 + 0.4 * sep[entry]).clip(0, 1)
        conf[exit_sig] = 0.5

        out = pd.DataFrame({
            "signal": sig,
            "confidence": conf,
            "stop_distance": (a * self.atr_multiplier).clip(lower=0.0),
        })
        return self._validate_output(out, df)

"""Strategy plugin interface.

Contract
--------
``generate_signals(df)`` receives the canonical candle frame and returns a
DataFrame indexed identically with columns:

* ``signal``       : int in {-1, 0, 1}  (short bias currently maps to exit/flat)
* ``confidence``   : float in [0, 1]    (strategy-local conviction)
* ``stop_distance``: float >= 0         (suggested stop distance in price
                                      units, e.g. ATR * multiple; the risk
                                      manager enforces its own floor)

LOOKAHEAD RULES (enforced culturally here and structurally by the engine)
-------------------------------------------------------------------------
A row at timestamp T may use ONLY information available at T's close.
The engine executes any signal from row T at bar T+1's open. Crossing
indicators (crossover detection, breakout tests against the current bar's
high/low, etc.) must be ``shift(1)``-ed by the strategy itself. The
built-in strategies demonstrate the pattern.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

EMPTY_SIGNAL_FRAME_COLUMNS = ["signal", "confidence", "stop_distance"]


def empty_signal_frame(index: pd.Index) -> pd.DataFrame:
    """An all-flat signal frame aligned to ``index``."""
    return pd.DataFrame(
        {
            "signal": pd.Series(0, index=index, dtype=int),
            "confidence": pd.Series(0.0, index=index, dtype=float),
            "stop_distance": pd.Series(0.0, index=index, dtype=float),
        },
        index=index,
    )


class Strategy(ABC):
    """Base class for all strategies. Subclasses must set ``name``."""

    name: str = "base"

    def __init__(self, name: str | None = None, params: dict | None = None):
        if name:
            self.name = name
        self.params: dict = params or {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the signal frame for the given candles."""

    # ---------------------------------------------------------------- helpers
    def _validate_output(self, out: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce the strategy output to the canonical schema."""
        if out is None or len(out) == 0:
            return empty_signal_frame(df.index)
        for col in EMPTY_SIGNAL_FRAME_COLUMNS:
            if col not in out.columns:
                out[col] = 0 if col == "signal" else 0.0
        out = out[list(EMPTY_SIGNAL_FRAME_COLUMNS)].copy()
        out["signal"] = pd.to_numeric(out["signal"], errors="coerce").fillna(0).astype(int)
        out["signal"] = out["signal"].clip(-1, 1)
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)
        out["confidence"] = out["confidence"].clip(0.0, 1.0)
        out["stop_distance"] = pd.to_numeric(out["stop_distance"], errors="coerce").fillna(0.0)
        out["stop_distance"] = out["stop_distance"].clip(lower=0.0)
        return out

    def warmup_bars(self) -> int:
        """Minimum bars this strategy needs before producing valid signals.
        Subclasses may override; used by the engine to skip early rows."""
        return 50

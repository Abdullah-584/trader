"""ML classifier strategy — trained locally, walk-forward, no lookahead.

The model is trained by ``bot.ml.train`` (walk-forward over history) and
consumed here read-only. It maps feature rows to a long probability; the
strategy votes long only above ``threshold``. Features themselves must be
computed from the candle frame only — nothing that would not have been
known at each bar close.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from loguru import logger

from bot.data.indicators import atr, bollinger, ema, rsi, roc
from bot.strategies.base import Strategy


class MLClassifierStrategy(Strategy):
    """Scikit-learn HistGradientBoosting long-probability classifier."""

    def __init__(self, name: str | None = None, params: dict | None = None):
        super().__init__(name, params)
        self.horizon = int(self.params.get("horizon", 8))
        self.threshold = float(self.params.get("threshold", 0.55))
        self.atr_period = int(self.params.get("atr_period", 14))
        self.model_dir = Path(str(self.params.get("model_dir", "data/models")))
        self.model = None  # lazy-loaded joblib artifact

    # ------------------------------------------------------------------ model io
    def _ensure_model(self):
        if self.model is not None:
            return self.model
        path = self.model_dir / "ml_classifier.joblib"
        if not path.exists():
            logger.warning(
                "ML strategy enabled but no trained model at {} — "
                "run `python main.py train` first; voting flat meanwhile.", path)
            return None
        import joblib
        bundle = joblib.load(path)
        self.model = bundle
        return bundle

    # ---------------------------------------------------------------- features
    @staticmethod
    def build_features(df: pd.DataFrame) -> pd.DataFrame:
        """Bar-close features only (no shifting needed: each row uses its own
        bar's completed values; the ENGINE shifts execution to next open)."""
        out = pd.DataFrame(index=df.index)
        c = df["close"]
        out["ret_1"] = c.pct_change(1)
        out["ret_5"] = c.pct_change(5)
        out["ret_20"] = c.pct_change(20)
        out["rsi_14"] = rsi(c, 14) / 100.0
        out["ema_ratio_fast"] = c / ema(c, 21)
        out["ema_ratio_slow"] = c / ema(c, 55)
        out["atr_pct"] = atr(df, 14) / c
        bb_low, bb_mid, bb_high = bollinger(c, 20, 2.0)
        width = bb_high - bb_low
        rng = width.where(width != 0)  # NaN where flat; keeps float dtype
        out["bb_pos"] = ((c - bb_low) / rng).astype(float).clip(-0.5, 1.5)
        out["roc_10"] = roc(c, 10) / 100.0
        vol_std = df["volume"].rolling(50).std(ddof=0)
        out["vol_z"] = (df["volume"] - df["volume"].rolling(50).mean()) / \
                       vol_std.where(vol_std != 0)
        out["hour_sin"] = pd.Series(df.index.hour, index=df.index) * (2 * math.pi / 24)
        out["hour_sin"] = out["hour_sin"].apply(math.sin)
        out["hour_cos"] = pd.Series(df.index.hour, index=df.index) * (2 * math.pi / 24)
        out["hour_cos"] = out["hour_cos"].apply(math.cos)
        return out

    def warmup_bars(self) -> int:
        return 60

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {"signal": 0, "confidence": 0.0,
             "stop_distance": atr(df, self.atr_period).fillna(0.0) * 2.0},
            index=df.index,
        )
        bundle = self._ensure_model()
        if bundle is None:
            return self._validate_output(out, df)

        feats = self.build_features(df)
        X = feats[bundle["feature_names"]]
        mask = X.notna().all(axis=1) & (df["volume"] > 0)
        if not mask.any():
            return self._validate_output(out, df)

        proba = bundle["model"].predict_proba(X[mask])[:, 1]
        sig = pd.Series(0, index=df.index, dtype=int)
        conf = pd.Series(0.0, index=df.index)
        sig[mask] = (proba >= self.threshold).astype(int)
        conf[mask] = pd.Series((proba - self.threshold) / max(1e-9, (1 - self.threshold)),
                               index=X[mask].index).clip(0, 0.95)

        out["signal"] = sig
        out["confidence"] = conf
        return self._validate_output(out, df)

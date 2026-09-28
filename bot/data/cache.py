"""Parquet candle cache.

Layout: data/cache/<provider>_<symbol>_<timeframe>.parquet

* ``read``  returns whatever is cached (empty frame when absent).
* ``write`` atomically replaces the file (tmp + os.replace).
* ``update`` merges a fresh download into the cache: dedupes by timestamp
  (newest wins), keeps the frame sorted, and reports coverage.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from loguru import logger

from bot.data.provider import CandleSchema, parse_timeframe


class CandleCache:
    def __init__(self, cache_dir: str | Path):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- paths ----------------------------------------------------------------
    def _path_for(self, provider: str, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "-").replace(":", "-").replace(" ", "_")
        return self._dir / f"{provider}_{safe_symbol}_{timeframe}.parquet"

    # -- io --------------------------------------------------------------------
    def read(self, provider: str, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self._path_for(provider, symbol, timeframe)
        if not path.exists():
            return pd.DataFrame(columns=CandleSchema)
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # corrupt cache should never kill the bot
            logger.warning("cache read failed for {}: {} — treating as empty", path, exc)
            return pd.DataFrame(columns=CandleSchema)
        if df.empty:
            return df
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"
        return df[list(CandleSchema)]

    def write(self, provider: str, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        path = self._path_for(provider, symbol, timeframe)
        tmp = path.with_suffix(".parquet.tmp")
        df[list(CandleSchema)].to_parquet(tmp)
        os.replace(tmp, path)  # atomic on Windows and POSIX

    def update(self, provider: str, symbol: str, timeframe: str,
               fresh: pd.DataFrame) -> pd.DataFrame:
        """Merge ``fresh`` into the cache; returns the merged frame."""
        if fresh.empty:
            return self.read(provider, symbol, timeframe)
        cached = self.read(provider, symbol, timeframe)
        if cached.empty:
            merged = fresh
        else:
            merged = pd.concat([cached, fresh]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
        self.write(provider, symbol, timeframe, merged)
        return merged

    # -- coverage ----------------------------------------------------------------
    def coverage(self, provider: str, symbol: str, timeframe: str,
                 df: pd.DataFrame | None = None) -> dict:
        """Describe cached coverage: first, last, count, and missing-bar stats."""
        if df is None:
            df = self.read(provider, symbol, timeframe)
        if df.empty:
            return {"bars": 0, "first": None, "last": None, "gaps": None}
        step = parse_timeframe(timeframe)
        expected = pd.date_range(df.index[0], df.index[-1], freq=step, tz="UTC")
        missing = expected.difference(df.index)
        return {
            "bars": int(len(df)),
            "first": df.index[0].isoformat(),
            "last": df.index[-1].isoformat(),
            "gaps": int(len(missing)),
        }

    def find_gaps(self, provider: str, symbol: str, timeframe: str,
                  df: pd.DataFrame) -> pd.DatetimeIndex:
        """Timestamps expected by the timeframe grid but missing from ``df``."""
        if df.empty:
            return pd.DatetimeIndex([])
        step = parse_timeframe(timeframe)
        expected = pd.date_range(df.index[0], df.index[-1], freq=step, tz="UTC")
        return expected.difference(df.index)

"""Data pipeline: load from cache, download what's missing, merge, return.

Used by the backtester, the ML trainer, and the CLI ``download`` command.
Rate limits and retries are handled inside the providers.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from bot.data.cache import CandleCache
from bot.data.provider import DataProviderError, drop_forming_candle, get_provider


def load_history(cfg, symbol: str, timeframe: str, *,
                 provider_name: str | None = None,
                 bars: int | None = None) -> pd.DataFrame:
    """Return closed candles for symbol/timeframe, cache-first.

    * Reads the parquet cache.
    * Downloads when the cache is short of ``bars`` (or has no data), then
      merges. Raises DataProviderError only when both cache and network fail.
    """
    bars = int(bars or cfg.data.get("history_bars", 1500))
    provider_name = provider_name or str(cfg.data.get("provider", "ccxt"))
    cache = CandleCache(cfg.root / str(cfg.data.get("cache_dir", "data/cache")))

    cached = cache.read(provider_name, symbol, timeframe)
    if len(cached) >= bars:
        return drop_forming_candle(cached.tail(bars * 2), timeframe)

    provider = get_provider(provider_name, cfg)
    logger.info("cache has {} bars for {} {} — downloading {}",
                len(cached), symbol, timeframe, bars)
    fresh = provider.fetch_history(symbol, timeframe, bars=bars,
                                   retries=int(cfg.data.get("max_retries", 5)),
                                   backoff=float(cfg.data.get("retry_backoff", 1.8)),
                                   sleep_between=float(cfg.data.get("rate_limit_sleep", 0.25)))
    merged = cache.update(provider_name, symbol, timeframe, fresh)
    return drop_forming_candle(merged.tail(bars * 2), timeframe)

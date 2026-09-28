"""Market data layer: provider abstraction, candle cache, indicators.

Providers (ccxt for crypto, yfinance for stocks/ETFs/forex, CSV for fully
offline use) all return a normalized candle DataFrame:

* tz-aware UTC DatetimeIndex named ``timestamp``
* float columns: ``open, high, low, close, volume`` (ascending, deduped)
"""

from bot.data.cache import CandleCache
from bot.data.indicators import atr, bollinger, donchian, ema, rsi, roc
from bot.data.provider import (
    CandleSchema,
    CSVDataProvider,
    CCXTDataProvider,
    DataProvider,
    DataProviderError,
    YFinanceDataProvider,
    get_provider,
)

__all__ = [
    "CandleCache",
    "CandleSchema",
    "DataProvider",
    "DataProviderError",
    "CCXTDataProvider",
    "YFinanceDataProvider",
    "CSVDataProvider",
    "get_provider",
    "atr",
    "bollinger",
    "donchian",
    "ema",
    "rsi",
    "roc",
]

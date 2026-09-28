"""Data provider abstraction.

Every provider returns the same normalized candle frame so strategies,
backtester, and runner never care where data came from. Public endpoints
only: ccxt needs no keys for OHLCV; yfinance needs none either.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import pandas as pd
from loguru import logger

CandleSchema = ["open", "high", "low", "close", "volume"]


class DataProviderError(RuntimeError):
    """Raised when a provider cannot deliver usable candles."""


# --------------------------------------------------------------------------
# Normalization: every raw provider frame goes through this before use.
# --------------------------------------------------------------------------
def normalize_candles(df: pd.DataFrame, source: str = "?") -> pd.DataFrame:
    """Validate and normalize a candle frame to the canonical schema."""
    if df is None or len(df) == 0:
        raise DataProviderError(f"{source}: received no candle data")

    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    # tolerate yfinance-style MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]

    missing = [c for c in CandleSchema if c not in df.columns]
    if missing:
        raise DataProviderError(f"{source}: candle frame missing columns {missing}; "
                                f"got {list(df.columns)}")

    df = df[list(CandleSchema)].astype(float)

    # index -> tz-aware UTC DatetimeIndex named 'timestamp'
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(how="any")

    if (df["close"] <= 0).any():
        n = int((df["close"] <= 0).sum())
        logger.warning("{}: dropping {} non-positive close rows", source, n)
        df = df[df["close"] > 0]

    if df.empty:
        raise DataProviderError(f"{source}: no valid candles after normalization")
    return df


def retry_call(fn: Callable, *, what: str, retries: int = 5, backoff: float = 1.8,
               sleep_between: float = 0.0):
    """Run fn() with retries + exponential backoff; raise after the last try."""
    delay = 1.0
    for attempt in range(1, retries + 1):
        try:
            if sleep_between:
                time.sleep(sleep_between)
            return fn()
        except Exception as exc:  # noqa: BLE001 — providers raise many types
            if attempt == retries:
                raise DataProviderError(f"{what}: failed after {retries} attempts: {exc}") from exc
            wait = delay * backoff ** (attempt - 1)
            logger.warning("{}: attempt {}/{} failed ({}) — retrying in {:.1f}s",
                           what, attempt, retries, exc, wait)
            time.sleep(wait)
    raise DataProviderError(f"{what}: unreachable")  # pragma: no cover


def parse_timeframe(tf: str) -> pd.Timedelta:
    """'15m' -> Timedelta('15min'), '1h' -> 1h, '1d' -> 1 day."""
    table = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "1d": "1D"}
    if tf not in table:
        raise DataProviderError(f"unsupported timeframe {tf!r}")
    return pd.Timedelta(table[tf])


def drop_forming_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Remove the last row when its bar has not closed yet.

    Exchanges include the in-progress candle in OHLCV responses. Acting on
    it would trade on incomplete information, so the runner must never see
    it. Idempotent for fully-closed history.
    """
    if df.empty:
        return df
    step = parse_timeframe(timeframe)
    last_close = df.index[-1] + step
    if pd.Timestamp.now(tz="UTC") < last_close:
        return df.iloc[:-1]
    return df


class DataProvider(ABC):
    """Interface every market-data source must implement."""

    name: str = "abstract"

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    limit: int = 500, since: pd.Timestamp | None = None) -> pd.DataFrame:
        """Return up to ``limit`` candles ending at the most recent close."""

    def fetch_history(self, symbol: str, timeframe: str, *, bars: int,
                      retries: int = 5, backoff: float = 1.8,
                      sleep_between: float = 0.0) -> pd.DataFrame:
        """Chunked download of ``bars`` candles, oldest first.

        The base implementation pages backwards using fetch_ohlcv with a
        moving ``since`` cursor; subclasses with bulk download (yfinance)
        override this with one call.
        """
        step = parse_timeframe(timeframe)
        target = bars
        end_anchor: pd.Timestamp | None = None
        frames: list[pd.DataFrame] = []
        remaining = target

        while remaining > 0:
            chunk_limit = min(1000, remaining)
            df = retry_call(
                lambda: self.fetch_ohlcv(symbol, timeframe, limit=chunk_limit, since=None)
                if end_anchor is None
                else self._fetch_window(symbol, timeframe, end_anchor, chunk_limit),
                what=f"{self.name}:{symbol}:{timeframe} chunk",
                retries=retries, backoff=backoff, sleep_between=sleep_between,
            )
            if end_anchor is None:
                # first chunk gives us the newest anchor
                end_anchor = df.index[-1]
            else:
                # merge and check progress
                combined = pd.concat([df] + frames).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                if len(combined) <= sum(len(f) for f in frames):
                    logger.warning("{}: no further history available for {}",
                                   self.name, symbol)
                    break
            frames.append(df)
            remaining = target - sum(len(f) for f in frames)
            if remaining <= 0:
                break
            end_anchor = min(f.index[0] for f in frames)

        if not frames:
            raise DataProviderError(f"{self.name}: no history for {symbol}")
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")].tail(bars)
        return normalize_candles(out, source=f"{self.name}:{symbol}")

    # -- helpers -------------------------------------------------------------
    def _fetch_window(self, symbol: str, timeframe: str,
                      before: pd.Timestamp, limit: int) -> pd.DataFrame:
        """Fetch up to ``limit`` candles strictly BEFORE ``before``."""
        step = parse_timeframe(timeframe)
        since = before - step * limit
        df = self.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
        return df[df.index < before]


class CCXTDataProvider(DataProvider):
    """Public ccxt OHLCV — no API keys required."""

    name = "ccxt"

    def __init__(self, exchange_id: str = "binance", *, rate_limit_sleep: float = 0.25,
                 retries: int = 5, backoff: float = 1.8):
        try:
            import ccxt  # imported lazily so the rest of the bot runs offline
        except ImportError as exc:  # pragma: no cover
            raise DataProviderError("ccxt is not installed") from exc
        self._exchange_id = exchange_id
        self._sleep = rate_limit_sleep
        self._retries = retries
        self._backoff = backoff
        klass = getattr(ccxt, exchange_id, None)
        if klass is None:
            raise DataProviderError(f"unknown ccxt exchange id {exchange_id!r}")
        # Public market data only: no keys are passed, ever.
        self._ex = klass({"enableRateLimit": True})

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    limit: int = 500, since: pd.Timestamp | None = None) -> pd.DataFrame:
        try:
            markets = self._ex.load_markets()
            if symbol not in markets:
                raise DataProviderError(
                    f"{self._exchange_id}: symbol {symbol!r} not listed "
                    f"(example format: 'BTC/USDT')")
        except DataProviderError:
            raise
        except Exception as exc:  # market listing may fail offline; retry later
            raise DataProviderError(f"{self._exchange_id}: load_markets failed: {exc}") from exc

        params = {"since": int(since.timestamp() * 1000) if since is not None else None}
        raw = retry_call(
            lambda: self._ex.fetch_ohlcv(symbol, timeframe=timeframe,
                                         limit=limit,
                                         since=params["since"]),
            what=f"ccxt:{self._exchange_id}:{symbol}:{timeframe}",
            retries=self._retries, backoff=self._backoff, sleep_between=self._sleep,
        )
        if not raw:
            raise DataProviderError(f"ccxt returned no candles for {symbol} {timeframe}")
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        out = drop_forming_candle(normalize_candles(df, source=f"ccxt:{symbol}"), timeframe)
        return out


class YFinanceDataProvider(DataProvider):
    """yfinance for stocks, ETFs, and forex — free, no keys."""

    name = "yfinance"

    _INTERVAL = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d"}
    # yfinance intraday data is limited to recent windows per request size
    _BULK_LIMIT = {"1m": 7, "5m": 45, "15m": 45, "1h": 700, "1d": 100000}

    def __init__(self, *, rate_limit_sleep: float = 0.5):
        self._sleep = rate_limit_sleep

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    limit: int = 500, since: pd.Timestamp | None = None) -> pd.DataFrame:
        interval = self._INTERVAL.get(timeframe)
        if interval is None:
            raise DataProviderError(f"yfinance: unsupported timeframe {timeframe!r}")

        import yfinance as yf  # lazy import

        period = self._bulk_period_for(limit, timeframe)
        kwargs: dict = {"interval": interval, "period": period, "auto_adjust": False}
        if since is not None:
            kwargs.pop("period")
            kwargs["start"] = since.tz_convert("UTC").tz_localize(None)

        def _dl():
            time.sleep(self._sleep)
            return yf.download(symbol, progress=False, threads=False, **kwargs)

        df = retry_call(_dl, what=f"yfinance:{symbol}:{interval}", retries=3, backoff=2.0)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        if "close" not in df.columns and "adj close" in df.columns:
            df["close"] = df["adj close"]
        out = drop_forming_candle(normalize_candles(df, source=f"yfinance:{symbol}"), timeframe)
        return out.tail(limit)

    def fetch_history(self, symbol: str, timeframe: str, *, bars: int,
                      retries: int = 3, backoff: float = 2.0,
                      sleep_between: float = 0.0) -> pd.DataFrame:
        # one bulk call covers everything yfinance gives per period
        return self.fetch_ohlcv(symbol, timeframe, limit=bars)

    def _bulk_period_for(self, limit: int, timeframe: str) -> str:
        step = parse_timeframe(timeframe)
        need = step * limit
        table = [
            (pd.Timedelta(days=7), "7d"), (pd.Timedelta(days=30), "1mo"),
            (pd.Timedelta(days=90), "3mo"), (pd.Timedelta(days=180), "6mo"),
            (pd.Timedelta(days=365), "1y"), (pd.Timedelta(days=730), "2y"),
            (pd.Timedelta(days=1825), "5y"), (pd.Timedelta(days=36500), "max"),
        ]
        for span, label in table:
            if need <= span:
                return label
        return "max"


class CSVDataProvider(DataProvider):
    """Read candles from data/csv/<symbol>.csv — fully offline, great for tests.

    Expected header: timestamp,open,high,low,close,volume  (timestamp is
    ISO-8601; naive timestamps are treated as UTC).
    """

    name = "csv"

    def __init__(self, csv_dir: str | Path = "data/csv"):
        self._dir = Path(csv_dir)

    def _path_for(self, symbol: str) -> Path:
        safe = symbol.replace("/", "-").replace(":", "-")
        return self._dir / f"{safe}.csv"

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    limit: int = 500, since: pd.Timestamp | None = None) -> pd.DataFrame:
        path = self._path_for(symbol)
        if not path.exists():
            raise DataProviderError(f"csv: file not found for {symbol}: {path}")
        df = pd.read_csv(path)
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        if since is not None:
            df = df[df.index >= since]
        return normalize_candles(df, source=f"csv:{symbol}").tail(limit)

    def fetch_history(self, symbol: str, timeframe: str, *, bars: int,
                      retries: int = 1, backoff: float = 1.0,
                      sleep_between: float = 0.0) -> pd.DataFrame:
        # local file: read everything at once, no paging needed
        return self.fetch_ohlcv(symbol, timeframe, limit=bars)


def get_provider(name: str, cfg) -> DataProvider:
    """Factory resolving the configured provider name into an instance."""
    name = str(name).lower()
    if name == "ccxt":
        return CCXTDataProvider(
            exchange_id=str(cfg.data.get("exchange", "binance")),
            rate_limit_sleep=float(cfg.data.get("rate_limit_sleep", 0.25)),
            retries=int(cfg.data.get("max_retries", 5)),
            backoff=float(cfg.data.get("retry_backoff", 1.8)),
        )
    if name == "yfinance":
        return YFinanceDataProvider(
            rate_limit_sleep=float(cfg.data.get("rate_limit_sleep", 0.25)),
        )
    if name == "csv":
        return CSVDataProvider(cfg.root / "data/csv")
    raise DataProviderError(f"unknown provider {name!r} (expected ccxt|yfinance|csv)")


def resolve_provider_for_symbol(entry: dict, default_provider: str) -> str:
    """Watchlist entries may override the provider; 'auto' picks by symbol shape.

    Symbols containing '/' (like BTC/USDT) go to ccxt; bare tickers (SPY)
    go to yfinance.
    """
    explicit = str(entry.get("provider", "auto")).lower()
    if explicit != "auto":
        return explicit
    symbol = str(entry.get("symbol", ""))
    if default_provider in ("ccxt", "yfinance"):
        shape = "ccxt" if "/" in symbol else "yfinance"
        return shape
    return default_provider

"""Quotex data provider — optional, unofficial, wrapped defensively.

Implements the same DataProvider interface as ccxt/yfinance, backed by
ChipaDevTeam/QuotexAPI (an async, reverse-engineered library where some
endpoints may still be mock/WIP upstream). Every call is retry-wrapped,
error-logged, and degrades to clear DataProviderError failures instead of
silent garbage. Credentials come exclusively from the environment.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from bot.data.provider import DataProvider, DataProviderError
from bot.quotex.bridge import run_async

_SSID_FULL_RE = re.compile(r"42\[.*?\"session\":\"([^\"]+)\".*?\]", re.DOTALL)


def extract_ssid_token(raw: str) -> str:
    """QuotexAPI examples accept the full cookie value; extract the token.

    Accepts either '42["authorization",{...session":"TOKEN"...}]' or a bare
    token. Raises DataProviderError when nothing usable is found.
    """
    raw = (raw or "").strip()
    if not raw:
        raise DataProviderError("QUOTEX_SSID is empty — set it in .env")
    m = _SSID_FULL_RE.search(raw)
    return m.group(1) if m else raw


class QuotexDataProvider(DataProvider):
    """Quotex candles via the unofficial ChipaDevTeam/QuotexAPI library.

    Quotex is the SOURCE OF TRUTH for its own (especially OTC) instruments;
    no other free feed approximates them. The library is unofficial and may
    break at any time — hence: lazy import, retries with backoff, explicit
    health checks, and clear failures.
    """

    name = "quotex"

    def __init__(self, *, ssid: str | None = None, email: str | None = None,
                 password: str | None = None, reconnect_enabled: bool = True,
                 max_reconnect_attempts: int = 5, reconnect_delay: int = 5,
                 rate_limit_sleep: float = 0.4, max_retries: int = 4,
                 retry_backoff: float = 1.8):
        self._ssid = extract_ssid_token(ssid or os.getenv("QUOTEX_SSID", ""))
        self._email = email or os.getenv("QUOTEX_EMAIL", "")
        self._password = password or os.getenv("QUOTEX_PASSWORD", "")
        if not self._ssid and not (self._email and self._password):
            raise DataProviderError(
                "Quotex credentials missing: set QUOTEX_SSID (preferred) or "
                "QUOTEX_EMAIL + QUOTEX_PASSWORD in .env")
        self._sleep = rate_limit_sleep
        self._retries = max_retries
        self._backoff = retry_backoff
        self._reconnect = {
            "reconnect_enabled": reconnect_enabled,
            "max_reconnect_attempts": max_reconnect_attempts,
            "reconnect_delay": reconnect_delay,
        }
        self._api = None  # lazy client

    # ------------------------------------------------------------------ client
    def _client(self):
        if self._api is not None:
            return self._api
        try:
            from QuotexAPI import QuotexAPI  # lazy import
        except ImportError as exc:
            raise DataProviderError(
                "QuotexAPI is not installed. Install it with:\n"
                "  pip install git+https://github.com/ChipaDevTeam/QuotexAPI.git"
            ) from exc

        try:
            # QuotexConfig kwargs may evolve upstream; build defensively.
            from QuotexAPI import QuotexConfig
            kwargs = {"email": self._email, "password": self._password, **self._reconnect}
            try:
                cfg = QuotexConfig(**kwargs)
                api = QuotexAPI(config=cfg)
            except TypeError:
                # older/newer signature: fall back to constructor kwargs
                api = QuotexAPI(email=self._email, password=self._password, **self._reconnect)
        except Exception as exc:
            raise DataProviderError(f"QuotexAPI client init failed: {exc}") from exc

        self._api = api
        return api

    def _call(self, coro_fn, what: str):
        """Retry-wrapped sync call into the async client with backoff."""
        delay = 1.0
        for attempt in range(1, self._retries + 1):
            try:
                time.sleep(self._sleep)
                return run_async(coro_fn(), timeout=60.0)
            except Exception as exc:  # upstream raises many shapes
                if attempt == self._retries:
                    raise DataProviderError(f"quotex {what}: failed after "
                                            f"{self._retries} attempts: {exc}") from exc
                wait = delay * self._backoff ** (attempt - 1)
                logger.warning("quotex {}: attempt {}/{} failed ({}) — retrying in {:.1f}s",
                               what, attempt, self._retries, exc, wait)
                time.sleep(wait)

    # ------------------------------------------------------------------ session
    def _ensure_connected(self):
        api = self._client()
        try:
            connected = bool(getattr(api, "is_connected", False))
        except Exception:
            connected = False
        if not connected:
            # Prefer SSID auth; fall back to email/password when absent.
            if self._ssid:
                self._call(lambda: api.connect_by_ssid(self._ssid), "connect_by_ssid")
            else:
                self._call(lambda: api.connect(), "connect")
        return api

    # ------------------------------------------------------------------ DataProvider
    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    limit: int = 500, since: pd.Timestamp | None = None) -> pd.DataFrame:
        api = self._ensure_connected()
        period = self._period_seconds(timeframe)
        end_ts = int(datetime.now(timezone.utc).timestamp())
        start_ts = end_ts - period * max(limit, 2)

        def _fetch():
            # Upstream get_candle_data(asset, end, period, start) style; adapt
            # defensively to signature drift with a kwargs fallback.
            try:
                return api.get_candle_data(symbol, end_from_time=end_ts,
                                           offset=period, start_from_time=start_ts)
            except TypeError:
                return api.get_candle_data(symbol, end_ts, period, start_ts)

        raw = self._call(_fetch, f"candles {symbol} {timeframe}")
        df = self._normalize_raw(raw, symbol)
        if since is not None:
            df = df[df.index >= since]
        return df.tail(limit)

    def get_assets(self) -> dict:
        """Raw asset list with payouts for the health check / dashboard."""
        api = self._ensure_connected()
        return self._call(lambda: api.get_assets(), "get_assets") or {}

    def get_payout(self, symbol: str) -> float | None:
        api = self._ensure_connected()
        try:
            value = self._call(lambda: api.get_payout(symbol), f"payout {symbol}")
        except DataProviderError:
            return None
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _period_seconds(timeframe: str) -> int:
        """Accepts '60' (Quotex seconds) or pandas-style '1m'/'5m'/'1h'."""
        tf = str(timeframe).strip().lower()
        if tf.isdigit():
            return int(tf)
        table = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}
        if tf not in table:
            raise DataProviderError(f"quotex: unsupported timeframe {timeframe!r}")
        return table[tf]

    @staticmethod
    def _normalize_raw(raw, symbol: str) -> pd.DataFrame:
        """Best-effort normalization of upstream candle payloads (dict of
        dicts, list of dicts, or list of lists — the library is WIP)."""
        rows: list[dict] = []
        if isinstance(raw, dict):
            iterable = raw.get("data", raw.get("candles", raw))
            if isinstance(iterable, dict):
                iterable = list(iterable.values())
            items = iterable if isinstance(iterable, list) else []
            for item in items:
                if isinstance(item, dict):
                    rows.append({
                        "timestamp": item.get("time") or item.get("timestamp"),
                        "open": item.get("open"), "high": item.get("max") or item.get("high"),
                        "low": item.get("min") or item.get("low"),
                        "close": item.get("close"),
                        "volume": item.get("volume", 0.0),
                    })
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    rows.append({
                        "timestamp": item.get("time") or item.get("timestamp"),
                        "open": item.get("open"), "high": item.get("max") or item.get("high"),
                        "low": item.get("min") or item.get("low"),
                        "close": item.get("close"),
                        "volume": item.get("volume", 0.0),
                    })
                elif isinstance(item, (list, tuple)) and len(item) >= 5:
                    rows.append({"timestamp": item[0], "open": item[1], "high": item[2],
                                 "low": item[3], "close": item[4], "volume": item[5] if len(item) > 5 else 0.0})
        if not rows:
            raise DataProviderError(f"quotex: unparseable candle payload for {symbol} "
                                    f"(library may be returning mock/WIP data)")
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True,
                                         errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        from bot.data.provider import normalize_candles
        return normalize_candles(df, source=f"quotex:{symbol}")

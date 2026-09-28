"""Quotex signal monitor — ANALYSIS ONLY.

IMPORTANT SCOPE
---------------
This module produces *suggestions* only. It never places an order, clicks
anything, or automates any broker. A human reads the dashboard and decides.
Binary options are negative-expectation for most participants (break-even
hit rate = 1 / (1 + payout)); this tool exists to make that arithmetic
visible, not to pretend it away.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from bot.core.config import Config
from bot.data.provider import DataProviderError, get_provider
from bot.signals.aggregator import SignalAggregator
from bot.storage.db import Database
from bot.strategies.registry import build_strategies_from_config


@dataclass(frozen=True)
class Suggestion:
    quotex_name: str
    direction: str            # "UP" | "DOWN" | "SKIP"
    confidence: float         # 0..1
    price: float | None
    expiry_seconds: int
    generated_at: datetime
    contributors: tuple[str, ...] = ()
    note: str = ""


def breakeven_hit_rate(payout_pct: float) -> float:
    """Fraction of wins needed to break even at a given payout.

    At payout p (e.g. 0.85) and full stake loss on a loss, the EV-zero
    condition is  w*(1+p) - (1-w)*1 = 0  ->  w = 1 / (2 + p - 1) = 1/(1+p).
    At 85% payout you need 54.05% sustained winners just to break even.
    """
    if payout_pct <= 0:
        return 1.0
    return 1.0 / (1.0 + float(payout_pct))


def outcome_for(direction: int, entry_price: float, exit_price: float) -> str:
    """Resolve a binary-style call against feed prices: WIN / LOSS / TIE."""
    eps = 1e-12
    if abs(exit_price - entry_price) <= eps * max(1.0, abs(entry_price)):
        return "TIE"
    went_up = exit_price > entry_price
    if direction > 0:
        return "WIN" if went_up else "LOSS"
    return "WIN" if not went_up else "LOSS"


class QuotexSignalMonitor:
    """Continuous analysis loop: feeds -> strategies -> suggestion."""

    def __init__(self, cfg: Config, db: Database):
        self.cfg = cfg
        self.db = db
        mcfg = dict(cfg.raw.get("quotex_monitor") or {})
        if not mcfg:
            raise ValueError("quotex_monitor section missing from config.yaml")
        self.mcfg = mcfg
        self.expiry = int(mcfg.get("expiry_seconds", 60))
        self.min_confidence = float(mcfg.get("min_confidence", 0.60))
        self.poll_seconds = int(mcfg.get("poll_seconds", 30))
        self.cooldown = int(mcfg.get("cooldown_seconds", 180))
        self.timeframe = str(mcfg.get("timeframe", "1m"))
        self.instruments = list(mcfg.get("instruments") or [])

        self.strategies = build_strategies_from_config(cfg.strategies)
        if not self.strategies:
            raise RuntimeError("no enabled strategies — monitor has nothing to analyze with")
        weights = {k: float((v or {}).get("weight", 1.0)) for k, v in cfg.strategies.items()}
        self.aggregator = SignalAggregator(
            mode=str(cfg.signals.get("mode", "weighted_vote")),
            min_confidence=float(cfg.signals.get("min_confidence", 0.45)),
            weights=weights,
        )
        self._providers: dict[str, object] = {}
        self._last_suggestion_at: dict[str, datetime] = {}
        self._last_bar_ts: dict[str, pd.Timestamp] = {}

        self.db.ensure_monitor_schema()

    # ------------------------------------------------------------------ feeds
    def _provider_for(self, feed: str):
        if feed not in self._providers:
            if feed == "crypto":
                self._providers[feed] = get_provider("ccxt", self.cfg)
            elif feed == "fx":
                self._providers[feed] = get_provider("yfinance", self.cfg)
            elif feed == "quotex":
                # unofficial feed — see README "Quotex integration"; requires
                # QUOTEX_SSID in .env and the optional QuotexAPI library.
                # Degrades gracefully when unavailable (instrument is skipped).
                try:
                    from bot.quotex.provider import QuotexDataProvider
                    qcfg = dict(self.cfg.raw.get("quotex") or {})
                    conn = dict(qcfg.get("connection") or {})
                    self._providers[feed] = QuotexDataProvider(
                        reconnect_enabled=bool(conn.get("reconnect_enabled", True)),
                        max_reconnect_attempts=int(conn.get("max_reconnect_attempts", 5)),
                        reconnect_delay=int(conn.get("reconnect_delay", 5)),
                        rate_limit_sleep=float(conn.get("rate_limit_sleep", 0.4)),
                        max_retries=int(conn.get("max_retries", 4)),
                        retry_backoff=float(conn.get("retry_backoff", 1.8)))
                except Exception as exc:
                    logger.warning("monitor: quotex feed unavailable ({}) — "
                                   "quotex/OTC instruments will be skipped", exc)
                    self._providers[feed] = None
            else:
                return None
        return self._providers[feed]

    def fetch_frame(self, feed: str, symbol: str) -> pd.DataFrame | None:
        provider = self._provider_for(feed)
        if provider is None:
            return None
        try:
            df = provider.fetch_ohlcv(symbol, self.timeframe, limit=600)
            if df is None or len(df) < 200:
                logger.debug("monitor: {} {} too few bars ({})", feed, symbol,
                             0 if df is None else len(df))
                return None
            return df
        except DataProviderError as exc:
            logger.debug("monitor: feed unavailable for {} {}: {}", feed, symbol, exc)
            return None

    # ------------------------------------------------------------------ cycle
    def cycle(self) -> list[Suggestion]:
        """One analysis pass over every instrument. Returns suggestions."""
        now = datetime.now(timezone.utc)
        out: list[Suggestion] = []
        for inst in self.instruments:
            name = str(inst.get("quotex_name", "?"))
            feed = str(inst.get("feed", "otc"))
            symbol = str(inst.get("symbol", "?"))

            if feed == "otc":
                logger.info("monitor: {} — OTC pair, no free feed; skipping", name)
                continue

            last = self._last_suggestion_at.get(name)
            if last and (now - last).total_seconds() < self.cooldown:
                continue

            df = self.fetch_frame(feed, symbol)
            if df is None:
                continue
            latest_ts = df.index[-1]
            if self._last_bar_ts.get(name) == latest_ts:
                continue  # no new bar since last analysis
            self._last_bar_ts[name] = latest_ts

            frames = {s.name: s.generate_signals(df) for s in self.strategies}
            i = len(df) - 1
            verdict = self.aggregator.aggregate_one(name, latest_ts, frames, i)
            price = float(df["close"].iloc[-1])

            self.db.save_monitor_signal(
                ts=latest_ts.to_pydatetime(), quotex_name=name, feed=feed,
                symbol=symbol, direction=verdict.direction,
                confidence=verdict.confidence, price=price,
                expiry_seconds=self.expiry,
                contributors=",".join(verdict.contributors),
                votes_long=verdict.votes_long, votes_short=verdict.votes_short,
                votes_flat=verdict.votes_flat,
            )

            if verdict.direction == 0 or verdict.confidence < self.min_confidence:
                continue

            direction = "UP" if verdict.direction > 0 else "DOWN"
            self._last_suggestion_at[name] = now
            sug = Suggestion(
                quotex_name=name, direction=direction,
                confidence=verdict.confidence, price=price,
                expiry_seconds=self.expiry, generated_at=now,
                contributors=tuple(verdict.contributors),
            )
            out.append(sug)
            logger.info("monitor SUGGESTION: {} {} conf={:.2f} @ {:.5f} ({}s expiry)",
                        direction, name, sug.confidence, price, self.expiry)
        return out

    # ------------------------------------------------------------- resolution
    def resolve_pending(self) -> int:
        """Score expired suggestions against the feed: WIN / LOSS / TIE.

        Honesty note: resolution measures the suggested direction from the
        suggestion-time feed price to the feed price after expiry. It is a
        proxy for, not a guarantee of, what a broker would have settled —
        brokers grade against their own (for OTC pairs, proprietary) feed.
        """
        now = datetime.now(timezone.utc)
        resolved = 0
        for row in self.db.pending_monitor_signals():
            try:
                suggested = datetime.fromisoformat(row["suggested_at"])
            except (TypeError, ValueError):
                continue
            elapsed = (now - suggested).total_seconds()
            if elapsed < row["expiry_seconds"] + 2:
                continue  # still live
            provider = self._provider_for(str(row["feed"] or ""))
            if provider is None:
                continue
            try:
                df = provider.fetch_ohlcv(str(row["symbol"]), self.timeframe, limit=5)
            except DataProviderError:
                continue
            if df is None or df.empty:
                continue
            exit_price = float(df["close"].iloc[-1])
            outcome = outcome_for(int(row["direction"]), float(row["price"]), exit_price)
            self.db.resolve_monitor_signal(int(row["id"]), exit_price, outcome)
            resolved += 1
            logger.info("monitor RESOLVED {} {}: {} (entry {:.5f} -> exit {:.5f})",
                        row["quotex_name"],
                        "UP" if int(row["direction"]) > 0 else "DOWN",
                        outcome, float(row["price"]), exit_price)
        return resolved

    # ------------------------------------------------------------------ loop
    def run_forever(self) -> None:
        logger.info("quotex signal monitor started (ANALYSIS ONLY — places no orders; "
                    "poll {}s, min confidence {:.2f})", self.poll_seconds, self.min_confidence)
        try:
            while True:
                started = time.monotonic()
                cmd = self.db.get_state("monitor_command")
                if cmd and cmd.get("action") == "stop":
                    logger.info("stop command received — monitor exiting")
                    self.db.set_state("monitor_command", {"action": "none"})
                    break
                try:
                    self.resolve_pending()
                    self.cycle()
                except Exception as exc:
                    logger.exception("monitor cycle failed: {}", exc)
                sleep_for = max(1.0, self.poll_seconds - (time.monotonic() - started))
                time.sleep(sleep_for)
        except KeyboardInterrupt:
            logger.info("monitor stopped by user")

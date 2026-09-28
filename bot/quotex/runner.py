"""Quotex runner — continuous analysis, and (only when explicitly enabled)
binary trade placement through the guarded broker.

Startup contract:
  * health check MUST pass before any trading is armed
  * DEMO by default; REAL requires config opt-in + typed confirmation
  * every candidate trade passes BinaryRiskManager.approve() first
  * data-only mode (trading.enabled: false) logs suggestions and settles
    nothing — it is the monitor, feeding the same SQLite store
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from bot.core.config import Config
from bot.quotex.broker import QuotexBroker, QuotexBrokerError
from bot.quotex.health import run_health_check
from bot.quotex.provider import QuotexDataProvider
from bot.risk.binary import BinaryRiskConfig, BinaryRiskManager
from bot.signals.aggregator import SignalAggregator
from bot.storage.db import Database
from bot.strategies.registry import build_strategies_from_config


class QuotexRunner:
    def __init__(self, cfg: Config, db: Database, *, confirmed_real: bool = False,
                 skip_health: bool = False):
        self.cfg = cfg
        self.db = db
        self.db.ensure_quotex_schema()

        qcfg = dict(cfg.raw.get("quotex") or {})
        if not qcfg:
            raise ValueError("quotex section missing from config.yaml")
        self.qcfg = qcfg
        conn = dict(qcfg.get("connection") or {})
        tcfg = dict(qcfg.get("trading") or {})
        dcfg = dict(qcfg.get("data") or {})

        self.assets = [str(a) for a in (qcfg.get("assets") or [])]
        if not self.assets:
            raise ValueError("quotex.assets must list at least one asset id")
        self.timeframe = str(dcfg.get("timeframe", "60"))
        self.poll_seconds = int(tcfg.get("poll_seconds", 30))
        self.expiry = int(tcfg.get("expiry_seconds", 60))
        self.min_confidence = float(cfg.signals.get("min_confidence", 0.45))

        self.trading_enabled = bool(tcfg.get("enabled", False))
        self.account = str(tcfg.get("account", "demo")).lower()
        if self.account not in ("demo", "real"):
            raise ValueError("quotex.trading.account must be demo|real")
        if self.trading_enabled and self.account == "real" and not confirmed_real:
            raise PermissionError(
                "quotex trading on REAL requires the typed confirmation. "
                "Refusing to start.")

        self.provider = QuotexDataProvider(
            reconnect_enabled=bool(conn.get("reconnect_enabled", True)),
            max_reconnect_attempts=int(conn.get("max_reconnect_attempts", 5)),
            reconnect_delay=int(conn.get("reconnect_delay", 5)),
            rate_limit_sleep=float(conn.get("rate_limit_sleep", 0.4)),
            max_retries=int(conn.get("max_retries", 4)),
            retry_backoff=float(conn.get("retry_backoff", 1.8)),
        )

        # health check is mandatory before trading, advisory otherwise
        ok, checks = run_health_check(cfg, interactive=confirmed_real)
        if not ok and self.trading_enabled:
            raise PermissionError(
                "quotex health check failed — fix the issues above before "
                "trading. Trading stays disabled.")
        if not ok:
            logger.warning("health check failed; continuing in DATA-ONLY mode "
                           "(trading disabled in config)")

        self.strategies = build_strategies_from_config(cfg.strategies)
        weights = {k: float((v or {}).get("weight", 1.0)) for k, v in cfg.strategies.items()}
        self.aggregator = SignalAggregator(
            mode=str(cfg.signals.get("mode", "weighted_vote")),
            min_confidence=self.min_confidence, weights=weights)

        balance = None
        self.broker: QuotexBroker | None = None
        if self.trading_enabled:
            self.broker = QuotexBroker(self.provider, account=self.account,
                                       confirmed_real=confirmed_real)
            balance = self.broker.balance()
            if balance is None or balance <= 0:
                raise PermissionError(
                    "could not read a positive account balance — refusing to "
                    "trade blind. Check the health check output.")
        starting = balance if balance else 10_000.0  # demo default notional
        self.risk = BinaryRiskManager(
            BinaryRiskConfig.from_mapping(qcfg.get("risk")), starting_balance=starting)
        self._last_bar_ts: dict[str, pd.Timestamp] = {}
        self._last_trade_at: dict[str, datetime] = {}
        logger.info("quotex runner ready: trading={} account={} assets={}",
                    self.trading_enabled, self.account, self.assets)

    # ------------------------------------------------------------------ loop
    def run_forever(self) -> None:
        logger.info("quotex runner loop started (poll {}s)", self.poll_seconds)
        try:
            while True:
                started = time.monotonic()
                cmd = self.db.get_state("quotex_command")
                if cmd and cmd.get("action") == "stop":
                    logger.info("stop command received — quotex runner exiting")
                    self.db.set_state("quotex_command", {"action": "none"})
                    break
                try:
                    if self.trading_enabled:
                        self._resolve_open_trades()
                    self._tick()
                except Exception as exc:
                    logger.exception("quotex tick failed: {}", exc)
                time.sleep(max(1.0, self.poll_seconds - (time.monotonic() - started)))
        except KeyboardInterrupt:
            logger.info("quotex runner stopped by user")

    # ------------------------------------------------------------------ tick
    def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        for asset in self.assets:
            try:
                df = self.provider.fetch_ohlcv(asset, self.timeframe,
                                               limit=int((self.qcfg.get("data") or {})
                                                         .get("history_bars", 600)))
            except Exception as exc:
                logger.warning("quotex data unavailable for {}: {}", asset, exc)
                continue
            if df is None or len(df) < 200:
                continue
            latest_ts = df.index[-1]
            if self._last_bar_ts.get(asset) == latest_ts:
                continue
            self._last_bar_ts[asset] = latest_ts

            frames = {s.name: s.generate_signals(df) for s in self.strategies}
            verdict = self.aggregator.aggregate_one(asset, latest_ts, frames,
                                                    len(df) - 1)
            price = float(df["close"].iloc[-1])

            if verdict.direction == 0 or verdict.confidence < self.min_confidence:
                logger.debug("quotex {}: SKIP (dir={}, conf={:.2f})",
                             asset, verdict.direction, verdict.confidence)
                continue

            payout = self.provider.get_payout(asset)
            direction = "CALL" if verdict.direction > 0 else "PUT"

            if not self.trading_enabled or self.broker is None:
                logger.info("quotex SUGGESTION: {} {} conf={:.2f} @ {:.5f} "
                            "payout={} (trading disabled — analysis only)",
                            direction, asset, verdict.confidence, price, payout)
                continue

            open_count = len(self.db.open_quotex_trades())
            decision = self.risk.approve(
                asset=asset, payout_pct=payout, open_trades=open_count,
                now=now, balance=self.broker.balance())
            if not decision.approved:
                logger.info("quotex trade skipped for {}: {}", asset, decision.reason)
                continue

            try:
                trade = self.broker.buy(asset, decision.stake, direction,
                                        self.expiry, payout=payout)
            except QuotexBrokerError as exc:
                logger.error("quotex buy failed for {}: {}", asset, exc)
                continue
            self.db.save_quotex_trade(
                ts=now, asset=asset, direction=direction,
                amount=trade.amount, expiry_seconds=self.expiry,
                account=self.account, payout_at_placement=payout,
                order_id=trade.order_id)

    # ------------------------------------------------------------- settlements
    def _resolve_open_trades(self) -> None:
        now = datetime.now(timezone.utc)
        for row in self.db.open_quotex_trades():
            placed = datetime.fromisoformat(row["ts"])
            if (now - placed).total_seconds() < row["expiry_seconds"] + 5:
                continue  # still live
            if not row.get("order_id"):
                self.db.settle_quotex_trade(int(row["id"]), "UNKNOWN", 0.0)
                continue
            result = self.broker.wait_for_result(row["order_id"], timeout=30.0)
            self.db.settle_quotex_trade(int(row["id"]), result.result, result.profit)
            new_balance = self.risk.balance + (result.profit if result.result == "WIN"
                                               else -row["amount"] if result.result == "LOSS"
                                               else 0.0)
            self.risk.on_settled(now, result.profit if result.result == "WIN"
                                 else -row["amount"] if result.result == "LOSS" else 0.0,
                                 new_balance)
            logger.info("quotex trade {} {} {} -> {} ({} {:.2f}); balance ~{:.2f}",
                        row["id"], row["asset"], row["direction"], result.result,
                        "profit" if result.profit >= 0 else "loss",
                        abs(result.profit), self.risk.balance)

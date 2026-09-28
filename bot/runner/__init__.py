"""Paper/live runner: fetch candles -> strategies -> risk -> execute -> log.

Safety model
------------
* Default and only pre-armed mode is PAPER. Live requires config opt-ins
  plus the typed confirmation (bot.core.safety) before a CcxtBroker exists.
* All state (cash, balances, open positions, risk counters) is persisted
  to SQLite after every action, so a crash or restart resumes cleanly.
* Kill switch / circuit breaker close everything via market orders and
  then refuse new entries for the rest of the day / until equity recovers
  above the drawdown threshold.
* One poll = one attempt per bar per symbol; signals act only on CLOSED
  candles (the providers drop the forming candle).
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from bot.alerts.notifier import Notifier
from bot.brokers.base import Broker, BrokerError
from bot.brokers.paper import PaperBroker
from bot.core.config import Config
from bot.core.models import OrderReason, Position, Side, utc_now
from bot.data.cache import CandleCache
from bot.data.provider import (
    DataProviderError,
    get_provider,
    resolve_provider_for_symbol,
)
from bot.risk.manager import RiskManager
from bot.signals.aggregator import SignalAggregator
from bot.storage.db import Database
from bot.strategies.registry import build_strategies_from_config


class Runner:
    def __init__(self, cfg: Config, *, broker: Broker | None = None,
                 confirmed_live: bool = False):
        self.cfg = cfg
        self.run_id = uuid.uuid4().hex[:12]
        self.symbol = str(cfg.data.get("primary_symbol", "BTC/USDT"))
        self.timeframe = str(cfg.data.get("primary_timeframe", "1h"))
        self.poll_seconds = int(cfg.runner.get("poll_seconds", 60))

        self.db = Database(cfg.root / str(cfg.storage.get("db_path", "data/trader.db")))
        self.cache = CandleCache(cfg.root / str(cfg.data.get("cache_dir", "data/cache")))
        self.notifier = Notifier(cfg.alerts)
        self.strategies = build_strategies_from_config(cfg.strategies)
        if not self.strategies:
            raise RuntimeError("no enabled strategies configured")

        self.aggregator = SignalAggregator(
            mode=str(cfg.signals.get("mode", "weighted_vote")),
            min_confidence=float(cfg.signals.get("min_confidence", 0.45)),
            weights={k: float((v or {}).get("weight", 1.0))
                     for k, v in cfg.strategies.items()},
        )
        self.risk = RiskManager(cfg.risk, starting_equity=cfg.paper_starting_cash)

        if broker is not None:
            self.broker = broker
        elif cfg.live_armable and confirmed_live:
            from bot.brokers.ccxt_live import CcxtBroker
            key, secret = cfg.exchange_keys(cfg.live.exchange)
            self.broker = CcxtBroker(cfg.live.exchange, key, secret)
            logger.warning("LIVE broker active — real money at risk")
        else:
            self.broker = PaperBroker(
                starting_cash=cfg.paper_starting_cash,
                fees_pct=cfg.risk.fees_pct,
                slippage_pct=cfg.risk.slippage_pct,
            )
            logger.info("PAPER broker active (venue never touches real money)")

        self.positions: dict[str, Position] = {}
        self._last_bar_ts: dict[str, pd.Timestamp] = {}
        self._stop = False
        self._restore_or_init_state()

    # ------------------------------------------------------------------ state
    def _restore_or_init_state(self) -> None:
        """Resume from SQLite after a crash/restart, or start fresh."""
        state = self.db.get_state("runner")
        starting_equity = self.cfg.paper_starting_cash
        if state:
            logger.info("restoring previous runner state (run {})", state.get("run_id"))
            cash = float(state.get("cash", starting_equity))
            balances = {k: float(v) for k, v in (state.get("balances") or {}).items()}
            if hasattr(self.broker, "restore_state"):
                self.broker.restore_state(cash, balances)
            else:
                logger.warning("live broker: cannot restore cash/balances from state; "
                               "exchange is the source of truth")
            for p in state.get("positions") or []:
                pos = Position(
                    symbol=p["symbol"], side=Side.BUY, quantity=float(p["quantity"]),
                    entry_price=float(p["entry_price"]),
                    entry_time=datetime.fromisoformat(p["entry_time"]),
                    stop_price=float(p["stop_price"]),
                    take_profit_price=p.get("take_profit_price"),
                    entry_fee=float(p.get("entry_fee", 0.0)),
                    trail_price=p.get("trail_price"),
                    highest_price_since_entry=float(p.get("highest_price_since_entry",
                                                           p["entry_price"])),
                )
                self.positions[pos.symbol] = pos
            risk_state = state.get("risk") or {}
            self.risk.equity = float(risk_state.get("equity", cash))
            self.risk.peak_equity = float(risk_state.get("peak_equity", cash))
            self.risk.day_start_equity = float(risk_state.get("day_start_equity", cash))
            logger.info("restored {} position(s), equity={:.2f}",
                        len(self.positions), self.risk.equity)
        else:
            logger.info("no prior state — starting fresh paper session "
                        "(run {}, equity {:.2f})", self.run_id, starting_equity)
        self._persist_state()

    def _persist_state(self) -> None:
        self.db.set_state("runner", {
            "run_id": self.run_id,
            "mode": "live" if self.cfg.live_armable else "paper",
            "cash": self.broker.cash() if hasattr(self.broker, "cash") else 0.0,
            "balances": self.broker.positions(),
            "positions": [
                {
                    "symbol": p.symbol, "quantity": p.quantity,
                    "entry_price": p.entry_price, "entry_time": p.entry_time.isoformat(),
                    "stop_price": p.stop_price,
                    "take_profit_price": p.take_profit_price,
                    "entry_fee": p.entry_fee, "trail_price": p.trail_price,
                    "highest_price_since_entry": p.highest_price_since_entry,
                } for p in self.positions.values()
            ],
            "risk": {
                "equity": self.risk.equity,
                "peak_equity": self.risk.peak_equity,
                "day_start_equity": self.risk.day_start_equity,
            },
        })

    # ------------------------------------------------------------------ data
    def _watchlist(self) -> list[dict]:
        return list(self.cfg.data.get("watchlist") or [])

    def _fetch_candles(self, entry: dict) -> pd.DataFrame | None:
        symbol = str(entry.get("symbol"))
        provider_name = resolve_provider_for_symbol(entry, self.cfg.data.get("provider", "ccxt"))
        try:
            provider = get_provider(provider_name, self.cfg)
            df = provider.fetch_ohlcv(symbol, self.timeframe,
                                      limit=int(self.cfg.data.get("history_bars", 1500)))
            merged = self.cache.update(provider_name, symbol, self.timeframe, df)
            return merged
        except DataProviderError as exc:
            logger.warning("data unavailable for {}: {}", symbol, exc)
            return None

    def _price_lookup(self):
        cache = {}

        def lookup(symbol: str) -> float:
            if symbol in cache:
                return cache[symbol]
            entry = {"symbol": symbol}
            df = self._fetch_candles(entry)
            if df is None or df.empty:
                raise BrokerError(f"no price data for {symbol}")
            px = float(df["close"].iloc[-1])
            cache[symbol] = px
            return px

        return lookup

    # ------------------------------------------------------------------ loop
    def run_forever(self) -> None:
        logger.info("runner loop started (run {}, poll every {}s)",
                    self.run_id, self.poll_seconds)
        self.notifier.notify("trader started",
                             f"mode={self.cfg.mode} run={self.run_id}")
        try:
            while not self._stop:
                started = time.monotonic()
                try:
                    self.tick()
                except Exception as exc:
                    logger.exception("tick failed ({}), continuing", exc)
                sleep_for = max(1.0, self.poll_seconds - (time.monotonic() - started))
                for _ in range(int(sleep_for)):
                    if self._stop:
                        break
                    time.sleep(1)
        finally:
            self._persist_state()
            self.db.close()
            logger.info("runner stopped cleanly; state persisted")

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------ tick
    def tick(self) -> None:
        cmd = self.db.get_state("command")
        if cmd and cmd.get("action") == "stop":
            logger.info("stop command received from dashboard — shutting down")
            self.db.set_state("command", {"action": "none"})
            self.stop()
            return
        marks: dict[str, float] = {}
        new_bars: dict[str, pd.Timestamp] = {}
        for entry in self._watchlist():
            symbol = str(entry.get("symbol"))
            df = self._fetch_candles(entry)
            if df is None or df.empty:
                continue
            new_bars[symbol] = df.index[-1]
            marks[symbol] = float(df["close"].iloc[-1])
            self._manage_position(symbol, marks[symbol], df)
        if not marks:
            logger.warning("tick had no data; nothing to do")
            return

        marked_equity = (self.broker.cash()
                         + sum(qty * marks.get(sym, 0.0)
                               for sym, qty in self.broker.positions().items()))
        now = datetime.now(timezone.utc)
        self.risk.on_bar(now, marked_equity, len(self.positions))

        if self.risk.trading_halted and self.risk.kill_switch_tripped:
            self._close_all(marks, OrderReason.KILL_SWITCH)
        elif self.risk.trading_halted:
            logger.warning("trading halted: {} — managing exits only",
                           self.risk.halt_reason())

        for entry in self._watchlist():
            symbol = str(entry.get("symbol"))
            if symbol not in new_bars:
                continue
            self._evaluate_symbol(entry, new_bars[symbol], marks)

        self._log_equity(now, marked_equity, marks)
        self._persist_state()

    # ------------------------------------------------------------------ per symbol
    def _evaluate_symbol(self, entry: dict, latest_ts: pd.Timestamp,
                         marks: dict[str, float]) -> None:
        symbol = str(entry.get("symbol"))
        if self._last_bar_ts.get(symbol) == latest_ts:
            return  # same candle as last tick — nothing new
        self._last_bar_ts[symbol] = latest_ts
        if symbol in self.positions:
            return  # one position per symbol; exits handled in _manage_position
        if self.risk.trading_halted:
            return

        df = self.cache.read(self._provider_name(entry), symbol, self.timeframe)
        if df is None or len(df) < self._warmup():
            logger.debug("not enough cached history for {} ({} bars)", symbol,
                         0 if df is None else len(df))
            return
        df = df.tail(600)  # strategies only need recent context

        frames = {s.name: s.generate_signals(df) for s in self.strategies}
        i = len(df) - 1
        verdict = self.aggregator.aggregate_one(symbol, df.index[-1], frames, i)
        self.db.save_signal(df.index[-1].to_pydatetime(), symbol, verdict.direction,
                            verdict.confidence, verdict.votes_long,
                            verdict.votes_short, verdict.votes_flat,
                            ",".join(verdict.contributors), self.run_id)
        self.db.commit_signals()

        if verdict.direction > 0:
            last = df.iloc[-1]
            stop_distance = self._median_stop_distance(frames, i)
            price = marks.get(symbol) or float(last["close"])
            decision = self.risk.plan_entry(
                symbol=symbol, price=price, stop_distance=stop_distance,
                open_positions=list(self.positions.values()),
            )
            if not decision.approved:
                logger.info("entry skipped for {}: {}", symbol, decision.reason)
                return
            try:
                order = self.broker.buy_market(symbol, decision.quantity, reason="entry")
            except BrokerError as exc:
                logger.error("buy failed for {}: {}", symbol, exc)
                return
            fee = order.fee_paid
            pos = Position(
                symbol=symbol, side=Side.BUY, quantity=order.filled_qty or decision.quantity,
                entry_price=order.filled_price or price, entry_time=utc_now(),
                stop_price=decision.stop_price,
                take_profit_price=decision.take_profit_price,
                entry_fee=fee, highest_price_since_entry=order.filled_price or price,
            )
            self.positions[symbol] = pos
            self.db.save_order(order, venue=self.broker.name)
            self.notifier.notify("entry opened",
                                 f"{symbol} qty={pos.quantity:.6f} @ {pos.entry_price:.2f}")
        elif verdict.direction < 0:
            logger.info("short-bias verdict for {} ignored (long-only system)", symbol)

    def _manage_position(self, symbol: str, mark: float, df: pd.DataFrame) -> None:
        pos = self.positions.get(symbol)
        if pos is None:
            return
        pos.highest_price_since_entry = max(pos.highest_price_since_entry, mark)

        # trailing stop uses the ATR hint from strategy frames when available
        trail = None
        if self.risk.cfg.trailing_stop:
            frames = {s.name: s.generate_signals(df.tail(300)) for s in self.strategies}
            atr_hint = self._median_stop_distance(frames, len(df.tail(300)) - 1)
            if atr_hint and atr_hint > 0:
                trail = self.risk.trailing_stop_price(pos, atr_hint / 2.0)
        if trail is not None:
            pos.trail_price = max(pos.trail_price or 0.0, trail)

        stop_level = max(pos.stop_price,
                         pos.trail_price if pos.trail_price else pos.stop_price)
        reason: OrderReason | None = None
        if mark <= stop_level:
            reason = (OrderReason.TRAILING_STOP
                      if pos.trail_price and pos.trail_price > pos.stop_price
                      else OrderReason.STOP)
        elif pos.take_profit_price and mark >= pos.take_profit_price:
            reason = OrderReason.TAKE_PROFIT
        if reason is None:
            return

        try:
            order = self.broker.sell_market(symbol, pos.quantity, reason=reason.value)
        except BrokerError as exc:
            logger.error("exit failed for {}: {} — will retry next tick", symbol, exc)
            return
        proceeds = (order.filled_price or mark) * pos.quantity
        fee = order.fee_paid
        pnl = proceeds - fee - (pos.entry_price * pos.quantity + pos.entry_fee)
        from bot.core.models import Trade
        trade = Trade(
            symbol=symbol, side=Side.BUY, quantity=pos.quantity,
            entry_price=pos.entry_price, exit_price=order.filled_price or mark,
            entry_time=pos.entry_time, exit_time=utc_now(),
            fees_paid=pos.entry_fee + fee, pnl=pnl,
            pnl_pct=pnl / max(1e-9, pos.entry_price * pos.quantity),
            reason=reason,
        )
        self.db.save_trade(trade)
        self.db.save_order(order, venue=self.broker.name)
        self.risk.record_realized(pnl)
        del self.positions[symbol]
        self.notifier.notify("position closed",
                             f"{symbol} pnl={pnl:.2f} ({trade.pnl_pct:.2%}) reason={reason.value}")

    def _close_all(self, marks: dict[str, float], reason: OrderReason) -> None:
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            try:
                order = self.broker.sell_market(symbol, pos.quantity, reason=reason.value)
            except BrokerError as exc:
                logger.error("kill-switch exit failed for {}: {} — retry next tick", symbol, exc)
                continue
            proceeds = (order.filled_price or marks.get(symbol, pos.entry_price)) * pos.quantity
            fee = order.fee_paid
            pnl = proceeds - fee - (pos.entry_price * pos.quantity + pos.entry_fee)
            from bot.core.models import Trade
            trade = Trade(
                symbol=symbol, side=Side.BUY, quantity=pos.quantity,
                entry_price=pos.entry_price, exit_price=order.filled_price
                or marks.get(symbol, pos.entry_price),
                entry_time=pos.entry_time, exit_time=utc_now(),
                fees_paid=pos.entry_fee + fee, pnl=pnl,
                pnl_pct=pnl / max(1e-9, pos.entry_price * pos.quantity),
                reason=reason,
            )
            self.db.save_trade(trade)
            self.db.save_order(order, venue=self.broker.name)
            self.risk.record_realized(pnl)
            del self.positions[symbol]
        self._persist_state()

    # ------------------------------------------------------------------ misc
    def _provider_name(self, entry: dict) -> str:
        return resolve_provider_for_symbol(entry, self.cfg.data.get("provider", "ccxt"))

    def _warmup(self) -> int:
        return max((s.warmup_bars() for s in self.strategies), default=100)

    def _median_stop_distance(self, frames: dict[str, pd.DataFrame], row: int) -> float:
        import numpy as np
        vals = []
        for frame in frames.values():
            if row < len(frame):
                v = float(frame["stop_distance"].iloc[row])
                if v > 0:
                    vals.append(v)
        return float(np.median(vals)) if vals else 0.0

    def _log_equity(self, now: datetime, equity: float, marks: dict[str, float]) -> None:
        cash = self.broker.cash()
        positions_value = equity - cash
        self.db.save_equity_point(now, equity, cash, positions_value,
                                  len(self.positions), self.run_id)
        self.db.conn.commit()

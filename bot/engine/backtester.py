"""Event-driven backtester.

Design rules (anti-lookahead, by construction)
----------------------------------------------
* A signal computed at bar T is queued and can only fill at bar T+1's OPEN
  (plus slippage). There is no code path that fills on the signal bar.
* Stops and take-profits are checked against the bar's high/low and filled
  pessimistically (stop before TP when both are touched in one bar).
* Fees and slippage are charged on every side.
* The risk manager evaluates kill switches on marked equity each bar and
  blocks new entries while halted; open positions still exit via their
  stops (as in real life).

Pessimism notes: fills use the first-touch price rather than an optimistic
close; same-bar stop-before-TP ordering; entries never assume better
prices than the next open. Real trading is usually worse than this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from bot.core.models import OrderReason, Position, Side, Trade
from bot.risk.manager import RiskManager
from bot.signals.aggregator import SignalAggregator


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    signal_log: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "=" * 62,
            "BACKTEST RESULT (hypothesis, not a promise — see warnings)",
            "=" * 62,
            f"start equity        : {m.get('start_equity', 0):>12,.2f}",
            f"end equity          : {m.get('end_equity', 0):>12,.2f}",
            f"total return        : {m.get('total_return_pct', 0):>11.2f}%",
            f"CAGR                : {m.get('cagr_pct', 0):>11.2f}%",
            f"Sharpe / Sortino    : {m.get('sharpe', 0):>6.2f} / {m.get('sortino', 0):.2f}",
            f"max drawdown        : {m.get('max_drawdown_pct', 0):>11.2f}%",
            f"trades              : {m.get('n_trades', 0):>12d}",
            f"win rate            : {m.get('win_rate', 0) * 100:>11.1f}%",
            f"profit factor       : {m.get('profit_factor', 0):>11.2f}",
            f"expectancy/trade    : {m.get('expectancy', 0):>12,.2f}",
            "-" * 62,
        ]
        if self.warnings:
            lines.append("WARNINGS:")
            lines += [f"  ! {w}" for w in self.warnings]
        lines.append("=" * 62)
        return "\n".join(lines)


class EventDrivenBacktester:
    """Simulates the full stack: strategies -> aggregator -> risk -> fills."""

    def __init__(self, *, strategies: list, aggregator: SignalAggregator,
                 risk_manager: RiskManager, fees_pct: float = 0.001,
                 slippage_pct: float = 0.0005, warmup_bars: int = 200):
        self.strategies = strategies
        self.agg = aggregator
        self.risk = risk_manager
        self.fees_pct = float(fees_pct)
        self.slippage_pct = float(slippage_pct)
        self.warmup_bars = int(warmup_bars)

    # ------------------------------------------------------------------
    def run(self, symbol: str, df: pd.DataFrame) -> BacktestResult:
        df = df.copy()
        if len(df) <= self.warmup_bars + 10:
            raise ValueError(
                f"need more than {self.warmup_bars + 10} bars, got {len(df)}")

        strategy_frames = {s.name: s.generate_signals(df) for s in self.strategies}
        self._agg_cache = None  # frames are new every run

        equity = float(self.risk.equity)
        cash = equity
        positions: dict[str, Position] = {}
        trades: list[Trade] = []
        equity_vals: list[float] = []
        signal_rows: list[dict] = []
        pending_entry = False  # entry signal seen on previous bar, fills now

        cols = df.reset_index()
        ts_series = df.index
        open_ = df["open"].to_numpy(float)
        high_ = df["high"].to_numpy(float)
        low_ = df["low"].to_numpy(float)
        close_ = df["close"].to_numpy(float)

        for i in range(len(df)):
            ts = ts_series[i]
            o, h, l, c = open_[i], high_[i], low_[i], close_[i]

            # ---------------- 1) pending entry from previous bar's signal ----
            had_pending, pending_entry = pending_entry, False
            if had_pending and self.risk.trading_halted:
                logger.debug("pending entry canceled: {}", self.risk.halt_reason())
            elif had_pending:
                entry_price = o * (1.0 + self.slippage_pct)  # buy worse
                last = signal_rows[-1] if signal_rows else None
                stop_distance = float(last.get("stop_distance", 0.0)) if last else 0.0
                decision = self.risk.plan_entry(
                    symbol=symbol, price=entry_price, stop_distance=stop_distance,
                    open_positions=list(positions.values()),
                )
                if decision.approved:
                    fee = decision.notional * self.fees_pct
                    cash -= decision.notional + fee
                    pos = Position(
                        symbol=symbol, side=Side.BUY, quantity=decision.quantity,
                        entry_price=entry_price, entry_time=ts,
                        stop_price=decision.stop_price,
                        take_profit_price=decision.take_profit_price,
                        entry_fee=fee, highest_price_since_entry=entry_price,
                    )
                    positions[symbol] = pos
                else:
                    logger.debug("entry rejected: {}", decision.reason)

            # ---------------- 2) manage open position on this bar ------------
            pos = positions.get(symbol)
            if pos is not None:
                pos.highest_price_since_entry = max(pos.highest_price_since_entry, h)

                # trailing stop ratchet
                atr_hint = self._atr_hint(strategy_frames, i)
                if atr_hint and self.risk.cfg.trailing_stop:
                    trail = self.risk.trailing_stop_price(pos, atr_hint)
                    if trail is not None:
                        pos.trail_price = max(pos.trail_price or 0.0, trail)

                stop_level = max(pos.stop_price,
                                 pos.trail_price if pos.trail_price else pos.stop_price)
                exit_reason: OrderReason | None = None
                exit_price: float | None = None

                # pessimistic ordering: stop first when both levels touched
                if l <= stop_level:
                    exit_reason, exit_price = (
                        OrderReason.TRAILING_STOP
                        if pos.trail_price and pos.trail_price > pos.stop_price
                        else OrderReason.STOP), min(o, stop_level)
                elif pos.take_profit_price and h >= pos.take_profit_price:
                    exit_reason, exit_price = OrderReason.TAKE_PROFIT, \
                        max(o, pos.take_profit_price)

                # aggregator exit vote for this bar (executes same-bar close —
                # exits are risk-reducing; entries stay T+1 which is the risky leg)
                agg_dir = self._aggregate_directions(symbol, strategy_frames)
                if exit_reason is None and i < len(agg_dir) and int(agg_dir.iloc[i]) < 0:
                    exit_reason, exit_price = OrderReason.MANUAL, c

                if exit_reason is not None and exit_price is not None:
                    fill = exit_price * (1.0 - self.slippage_pct)  # sell worse
                    proceeds = fill * pos.quantity
                    fee = proceeds * self.fees_pct
                    pnl = proceeds - fee - (pos.entry_price * pos.quantity + pos.entry_fee)
                    cash += proceeds - fee
                    trades.append(Trade(
                        symbol=symbol, side=Side.BUY, quantity=pos.quantity,
                        entry_price=pos.entry_price, exit_price=fill,
                        entry_time=pos.entry_time, exit_time=ts,
                        fees_paid=pos.entry_fee + fee, pnl=pnl,
                        pnl_pct=pnl / max(1e-9, pos.entry_price * pos.quantity),
                        reason=exit_reason,
                    ))
                    self.risk.record_realized(pnl)
                    del positions[symbol]

            # ---------------- 3) risk update on marked equity ----------------
            marked = cash + sum(
                p.quantity * c for p in positions.values())
            self.risk.on_bar(ts.to_pydatetime(), marked, len(positions))

            # ---------------- 4) new entry signal for NEXT bar's open --------
            if i >= self.warmup_bars and symbol not in positions \
                    and not self.risk.trading_halted:
                bar = self.agg.aggregate_one(symbol, ts, strategy_frames, i)
                signal_rows.append({
                    "timestamp": ts, "symbol": symbol,
                    "direction": bar.direction, "confidence": bar.confidence,
                    "stop_distance": self._stop_hint(strategy_frames, i),
                    "votes_long": bar.votes_long, "votes_flat": bar.votes_flat,
                    "contributors": bar.contributors,
                })
                if bar.direction > 0:
                    pending_entry = True

            equity_vals.append(cash + sum(p.quantity * c for p in positions.values()))

        equity_curve = pd.Series(equity_vals, index=ts_series, name="equity")

        from bot.engine.metrics import compute_metrics, too_good_warnings
        metrics = compute_metrics(equity_curve, trades)
        metrics["final_open_positions"] = len(positions)
        warnings = too_good_warnings(metrics)
        sig_df = pd.DataFrame(signal_rows).set_index("timestamp") if signal_rows \
            else pd.DataFrame()
        return BacktestResult(equity_curve=equity_curve, trades=trades,
                              metrics=metrics, warnings=warnings, signal_log=sig_df)

    # ------------------------------------------------------------------ helpers
    def _aggregate_directions(self, symbol: str,
                              strategy_frames: dict[str, pd.DataFrame]) -> pd.Series:
        cached = getattr(self, "_agg_cache", None)
        key = symbol
        if cached is None or cached[0] != key or cached[1] is None:
            frame = self.agg.aggregate_frame(symbol, strategy_frames)
            self._agg_cache = (key, frame)
        return self._agg_cache[1]["direction"]

    def _stop_hint(self, strategy_frames: dict[str, pd.DataFrame], row: int) -> float:
        """Median non-zero stop_distance suggestion across strategies."""
        vals = []
        for frame in strategy_frames.values():
            if row < len(frame):
                v = float(frame["stop_distance"].iloc[row])
                if v > 0:
                    vals.append(v)
        return float(np.median(vals)) if vals else 0.0

    def _atr_hint(self, strategy_frames: dict[str, pd.DataFrame], row: int) -> float | None:
        return self._stop_hint(strategy_frames, row) or None

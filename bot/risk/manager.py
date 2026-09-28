"""Risk manager — the mandatory, non-bypassable risk layer.

Responsibilities
----------------
* Position sizing from risk-per-trade and stop distance.
* Stop-loss / take-profit / trailing-stop level computation.
* Max concurrent positions, max position notional.
* Daily-loss kill switch and max-drawdown circuit breaker.

Hard caps live in ``bot.core.config`` and are enforced at config load; the
numbers here operate on the already-clamped values. No caller can disable
this layer — strategies and brokers receive it as a constructor dependency.

Long-only by design (shorting adds borrow/locate complexity and infinite
risk in crypto; see README).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime

from loguru import logger

from bot.core.config import RiskConfig
from bot.core.models import Position

MIN_NOTIONAL = 10.0     # refuse dust orders
QTY_STEP = 10 ** -8     # round quantity down to 8 decimals (exchange-like)


@dataclass(frozen=True)
class EntryDecision:
    """Result of asking the risk manager to plan an entry."""

    approved: bool
    reason: str
    quantity: float = 0.0
    stop_price: float = 0.0
    take_profit_price: float | None = None
    notional: float = 0.0


class RiskManager:
    """Stateful risk layer: sizing, stops, and circuit breakers."""

    def __init__(self, risk: RiskConfig, starting_equity: float):
        if starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        self.cfg = risk
        self.equity = float(starting_equity)
        self.peak_equity = float(starting_equity)
        self._day: date | None = None
        self.day_start_equity = float(starting_equity)
        self.realized_today = 0.0
        self.kill_switch_tripped = False
        self.kill_switch_reason = ""
        self.drawdown_halted = False

    # ------------------------------------------------------------------ state
    def on_bar(self, timestamp: datetime, marked_equity: float,
               open_positions: int) -> None:
        """Update equity/peak, roll the day, and evaluate breakers.

        Call once per bar (backtest) or per loop tick (live) BEFORE planning
        entries. ``marked_equity`` includes unrealized P&L.
        """
        if not math.isfinite(marked_equity) or marked_equity <= 0:
            logger.warning("on_bar received non-positive equity {} — ignoring", marked_equity)
            return

        self.equity = float(marked_equity)
        self.peak_equity = max(self.peak_equity, self.equity)

        # day rollover resets the kill switch (a new trading day)
        day = timestamp.date() if isinstance(timestamp, datetime) else timestamp
        if self._day is None:
            # first observation: keep the day-start equity we were constructed
            # with (or restored from state) — the day began before this bar
            self._day = day
        elif day != self._day:
            self._day = day
            self.day_start_equity = self.equity
            self.realized_today = 0.0
            if self.kill_switch_tripped:
                logger.info("New trading day {} — daily kill switch reset", day)
            self.kill_switch_tripped = False
            self.kill_switch_reason = ""

        # daily loss kill switch (realized + unrealized vs day start)
        day_loss_pct = 0.0
        if self.day_start_equity > 0:
            day_loss_pct = (self.day_start_equity - self.equity) / self.day_start_equity
        if day_loss_pct >= self.cfg.max_daily_loss_pct and not self.kill_switch_tripped:
            self.kill_switch_tripped = True
            self.kill_switch_reason = (
                f"daily loss {day_loss_pct:.2%} >= limit "
                f"{self.cfg.max_daily_loss_pct:.2%}")
            logger.warning("KILL SWITCH: {}", self.kill_switch_reason)

        # drawdown circuit breaker (peak-to-current equity)
        if not self.drawdown_halted and self.peak_equity > 0:
            dd = 1.0 - self.equity / self.peak_equity
            if dd >= self.cfg.max_drawdown_pct:
                self.drawdown_halted = True
                logger.warning(
                    "CIRCUIT BREAKER: drawdown {:.2%} >= limit {:.2%} — trading halted",
                    dd, self.cfg.max_drawdown_pct)

    # ------------------------------------------------------------- kill states
    @property
    def trading_halted(self) -> bool:
        """True when NO new entries may be opened."""
        return self.kill_switch_tripped or self.drawdown_halted

    def halt_reason(self) -> str:
        if self.drawdown_halted:
            return "max drawdown circuit breaker"
        if self.kill_switch_tripped:
            return f"daily kill switch: {self.kill_switch_reason}"
        return ""

    def record_realized(self, pnl: float) -> None:
        """Add realized P&L (positive or negative) for today's accounting."""
        self.realized_today += float(pnl)

    # ------------------------------------------------------------------ sizing
    def stop_price_for(self, price: float, stop_distance: float) -> float:
        """Initial stop for a long: price - stop_distance, with a minimum
        distance floor to avoid absurdly tight stops."""
        min_distance = price * self.cfg.min_stop_distance_pct
        distance = max(float(stop_distance), min_distance)
        if not math.isfinite(distance) or distance <= 0:
            distance = min_distance
        return price - distance

    def take_profit_for(self, entry_price: float, stop_price: float) -> float | None:
        """TP = (entry - stop) * R-multiple above entry; None when disabled."""
        r = self.cfg.take_profit_r_multiple
        if r <= 0:
            return None
        stop_distance = entry_price - stop_price
        return entry_price + stop_distance * r

    def plan_entry(self, *, symbol: str, price: float, stop_distance: float,
                   open_positions: list[Position], equity: float | None = None
                   ) -> EntryDecision:
        """Size and validate a candidate long entry.

        Returns an EntryDecision; approved=False carries the reason. The
        position size is derived from risk_per_trade and the stop distance,
        then capped by max_position_pct of equity and rounded down.
        """
        equity = self.equity if equity is None else float(equity)

        if self.trading_halted:
            return EntryDecision(False, f"risk halted: {self.halt_reason()}")
        if any(p.symbol == symbol for p in open_positions):
            return EntryDecision(False, f"already in position: {symbol}")
        if len(open_positions) >= self.cfg.max_positions:
            return EntryDecision(
                False, f"max concurrent positions reached ({self.cfg.max_positions})")
        if not math.isfinite(price) or price <= 0:
            return EntryDecision(False, f"invalid price {price}")
        if not math.isfinite(stop_distance) or stop_distance <= 0:
            return EntryDecision(False, f"invalid stop distance {stop_distance}")

        stop_price = self.stop_price_for(price, stop_distance)
        risk_per_unit = price - stop_price
        if risk_per_unit <= 0:
            return EntryDecision(False, "non-positive stop distance")

        # 1) risk-based size: lose at most risk_per_trade * equity if stopped
        risk_budget = equity * self.cfg.risk_per_trade
        qty = risk_budget / risk_per_unit

        # 2) notional cap: position value <= max_position_pct * equity
        max_notional = equity * self.cfg.max_position_pct
        qty = min(qty, max_notional / price)

        # 3) round down to lot step and re-validate
        qty = math.floor(qty / QTY_STEP) * QTY_STEP
        notional = qty * price

        if notional < MIN_NOTIONAL:
            return EntryDecision(
                False,
                f"notional {notional:.2f} below minimum {MIN_NOTIONAL:.2f} "
                "(risk too small for this price/stop)")

        tp = self.take_profit_for(price, stop_price)
        return EntryDecision(
            True, "ok",
            quantity=qty, stop_price=stop_price,
            take_profit_price=tp, notional=notional,
        )

    # ---------------------------------------------------------------- trailing
    def trailing_stop_price(self, position: Position, atr_value: float) -> float | None:
        """Trailing stop for a long: highest_price_since_entry minus
        trailing_atr_multiple * ATR. None when trailing is disabled."""
        if not self.cfg.trailing_stop:
            return None
        if not math.isfinite(atr_value) or atr_value <= 0:
            return None
        candidate = position.highest_price_since_entry - atr_value * self.cfg.trailing_atr_multiple
        # never ratchet the stop DOWN; also never below the initial stop
        floor = position.stop_price
        candidate = max(candidate, floor)
        if position.trail_price is not None:
            candidate = max(candidate, position.trail_price)
        return candidate

    # ------------------------------------------------------------------ reports
    def snapshot(self) -> dict:
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "day_start_equity": self.day_start_equity,
            "realized_today": self.realized_today,
            "kill_switch": self.kill_switch_tripped,
            "drawdown_halted": self.drawdown_halted,
            "risk_per_trade": self.cfg.risk_per_trade,
            "max_positions": self.cfg.max_positions,
        }

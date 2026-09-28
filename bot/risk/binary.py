"""Binary-options risk manager — replaces the ATR-stop model.

Binary trades have no stop-loss/take-profit: stake is fully lost on a
loss, so risk = stake. The controls that matter are:

* max stake per trade as a fraction of balance (hard cap 5%)
* max concurrent open trades (hard cap 5)
* daily loss kill switch on settled trades (hard cap 20%)
* minimum payout percentage before a signal may be traded (floor 50%)
* per-asset cooldown between trades

Hard caps are enforced in ``BINARY_HARD_CAPS`` and cannot be bypassed
from config.yaml — the same pattern as the spot risk manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from loguru import logger

BINARY_HARD_CAPS = {
    "max_trade_pct": 0.05,       # never stake more than 5% per trade
    "max_open_trades": 5,
    "max_daily_loss_pct": 0.20,  # never lose more than 20% in a day
    "min_payout_pct": 0.50,      # never accept payouts below 50%
}

MIN_STAKE_USD = 1.0            # broker minimum


@dataclass(frozen=True)
class BinaryRiskConfig:
    max_trade_pct: float
    max_open_trades: int
    max_daily_loss_pct: float
    min_payout_pct: float
    cooldown_seconds: int

    @classmethod
    def from_mapping(cls, m: dict) -> "BinaryRiskConfig":
        m = m or {}
        return cls(
            max_trade_pct=_clamp(m.get("max_trade_pct", 0.01), 0.001,
                                 BINARY_HARD_CAPS["max_trade_pct"]),
            max_open_trades=int(min(max(int(m.get("max_open_trades", 2)), 1),
                                    BINARY_HARD_CAPS["max_open_trades"])),
            max_daily_loss_pct=_clamp(m.get("max_daily_loss_pct", 0.05), 0.01,
                                      BINARY_HARD_CAPS["max_daily_loss_pct"]),
            min_payout_pct=_clamp(m.get("min_payout_pct", 0.80),
                                  BINARY_HARD_CAPS["min_payout_pct"], 1.0),
            cooldown_seconds=max(0, int(m.get("cooldown_seconds", 120))),
        )


def _clamp(value: float, lo: float, hi: float) -> float:
    v = float(value)
    c = min(max(v, lo), hi)
    if c != v:
        logger.warning("binary risk value {} out of range [{}, {}] — clamped to {}",
                       v, lo, hi, c)
    return c


@dataclass(frozen=True)
class BinaryTradeDecision:
    approved: bool
    reason: str
    stake: float = 0.0


class BinaryRiskManager:
    """Stateful gatekeeper for binary trades. No order reaches the broker
    without passing through ``approve()``."""

    def __init__(self, cfg: BinaryRiskConfig, starting_balance: float):
        if starting_balance <= 0:
            raise ValueError("starting_balance must be positive")
        self.cfg = cfg
        self.balance = float(starting_balance)
        self.peak_balance = float(starting_balance)
        self._day: date | None = None
        self.day_start_balance = float(starting_balance)
        self.kill_switch_tripped = False
        self._last_trade_at: dict[str, datetime] = {}

    # ------------------------------------------------------------------ state
    def on_settled(self, ts: datetime, pnl: float, new_balance: float) -> None:
        """Record a settled trade result (pnl: +profit or -stake)."""
        self._roll_day(ts)
        self.balance = float(new_balance)
        self.peak_balance = max(self.peak_balance, self.balance)
        day_loss = (self.day_start_balance - self.balance) / self.day_start_balance
        if day_loss >= self.cfg.max_daily_loss_pct and not self.kill_switch_tripped:
            self.kill_switch_tripped = True
            logger.warning("BINARY KILL SWITCH: daily loss {:.2%} >= limit {:.2%} "
                           "— no more trades today", day_loss, self.cfg.max_daily_loss_pct)

    def _roll_day(self, ts: datetime) -> None:
        day = ts.date()
        if self._day != day:
            self._day = day
            self.day_start_balance = self.balance
            if self.kill_switch_tripped:
                logger.info("new day {} — binary kill switch reset", day)
            self.kill_switch_tripped = False

    # ------------------------------------------------------------------ gate
    def approve(self, *, asset: str, payout_pct: float | None,
                open_trades: int, now: datetime,
                balance: float | None = None) -> BinaryTradeDecision:
        """Decide whether a candidate trade may be placed, and at what stake."""
        if balance is not None and balance > 0:
            self.balance = float(balance)
        self._roll_day(now)

        if self.kill_switch_tripped:
            return BinaryTradeDecision(False, "daily loss kill switch is active")
        if open_trades >= self.cfg.max_open_trades:
            return BinaryTradeDecision(
                False, f"max open trades reached ({self.cfg.max_open_trades})")
        if payout_pct is not None and payout_pct < self.cfg.min_payout_pct:
            return BinaryTradeDecision(
                False, f"payout {payout_pct:.0%} below required "
                       f"{self.cfg.min_payout_pct:.0%}")

        last = self._last_trade_at.get(asset)
        if last is not None and (now - last).total_seconds() < self.cfg.cooldown_seconds:
            return BinaryTradeDecision(False, f"cooldown active for {asset}")

        stake = self.balance * self.cfg.max_trade_pct
        # brokers have a minimum stake; if the risk-based stake can't cover
        # it, refuse rather than over-risking (no silent upsizing)
        if stake < MIN_STAKE_USD:
            return BinaryTradeDecision(
                False, f"risk-based stake {stake:.2f} below broker minimum "
                       f"{MIN_STAKE_USD:.2f} — refusing to upsize")
        self._last_trade_at[asset] = now
        return BinaryTradeDecision(True, "ok", stake=round(stake, 2))

    # ------------------------------------------------------------------ report
    def snapshot(self) -> dict:
        return {
            "balance": self.balance,
            "peak_balance": self.peak_balance,
            "day_start_balance": self.day_start_balance,
            "kill_switch": self.kill_switch_tripped,
            "max_trade_pct": self.cfg.max_trade_pct,
            "max_open_trades": self.cfg.max_open_trades,
            "min_payout_pct": self.cfg.min_payout_pct,
        }

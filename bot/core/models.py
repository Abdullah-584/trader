"""Domain models shared by every layer: orders, positions, trades, signals.

These are plain, typed dataclasses — deliberately decoupled from pandas and
from any broker SDK so the same code path works for paper and live trading.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Short readable unique id, e.g. 'ord-3f9c2a1b'."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderKind(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELED = "canceled"


class OrderReason(str, Enum):
    ENTRY = "entry"
    STOP = "stop"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    KILL_SWITCH = "kill_switch"
    MANUAL = "manual"
    END_OF_DATA = "end_of_data"  # backtest close-out


@dataclass
class Order:
    symbol: str
    side: Side
    quantity: float
    kind: OrderKind = OrderKind.MARKET
    limit_price: float | None = None
    reason: OrderReason = OrderReason.ENTRY
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float | None = None
    filled_qty: float = 0.0
    fee_paid: float = 0.0
    created_at: datetime = field(default_factory=utc_now)
    filled_at: datetime | None = None
    broker_id: str | None = None
    client_id: str = field(default_factory=lambda: new_id("ord"))

    @property
    def notional(self) -> float:
        if self.filled_price:
            return self.filled_qty * self.filled_price
        ref = self.limit_price if self.limit_price else 0.0
        return self.quantity * ref


@dataclass
class Position:
    """An open position. quantity > 0 means long (shorting is out of scope)."""

    symbol: str
    side: Side
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_price: float
    take_profit_price: float | None
    entry_fee: float = 0.0
    trail_price: float | None = None  # current trailing-stop level
    highest_price_since_entry: float = 0.0
    strategy_tag: str = "aggregate"
    position_id: str = field(default_factory=lambda: new_id("pos"))

    def unrealized_pnl(self, mark_price: float) -> float:
        return (mark_price - self.entry_price) * self.quantity

    def unrealized_pnl_pct(self, mark_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (mark_price - self.entry_price) / self.entry_price

    def risk_per_unit(self) -> float:
        """Initial risk per unit = entry minus initial stop (per unit)."""
        return max(self.entry_price - self.stop_price, 0.0)


@dataclass
class Trade:
    """A completed round-trip trade (entry + exit), used for stats/history."""

    symbol: str
    side: Side
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    fees_paid: float
    pnl: float
    pnl_pct: float
    reason: OrderReason
    strategy_tag: str = "aggregate"
    trade_id: str = field(default_factory=lambda: new_id("trd"))


@dataclass(frozen=True)
class Signal:
    """One strategy's opinion about one symbol at one bar close."""

    symbol: str
    timestamp: datetime
    strategy: str
    direction: int          # +1 long, -1 short/exit-bias, 0 flat
    confidence: float       # 0..1 (strategy-local, pre-aggregation)
    price: float
    note: str = ""

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"Signal.direction must be -1/0/1, got {self.direction}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Signal.confidence must be within [0, 1], got {self.confidence}")
        if math.isnan(self.price) or self.price < 0:
            raise ValueError(f"Signal.price must be a non-negative number, got {self.price}")


@dataclass(frozen=True)
class AggregatedSignal:
    """The aggregator's combined verdict for one symbol at one bar close."""

    symbol: str
    timestamp: datetime
    direction: int
    confidence: float       # combined 0..1
    votes_long: int
    votes_short: int
    votes_flat: int
    contributors: tuple[str, ...] = ()

    @property
    def is_actionable(self) -> bool:
        return self.direction != 0 and self.confidence > 0.0


@dataclass
class EquityPoint:
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    open_positions: int

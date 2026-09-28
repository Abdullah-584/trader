"""PaperBroker — default execution venue. Simulates fills locally.

Fill model (deliberately pessimistic):
* Market buys fill at last_price * (1 + slippage), sells at (1 - slippage).
* Fees charged on every fill's notional.
* No partial fills, no rejections for size — but the risk manager already
  caps sizes before orders reach here.

State is held in memory; the SQLite layer (bot.storage) persists it across
restarts. Paper trading is the DEFAULT and requires no keys.
"""

from __future__ import annotations

from loguru import logger

from bot.brokers.base import Broker, BrokerError
from bot.core.models import Order, OrderStatus, Side, utc_now


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, *, starting_cash: float, fees_pct: float = 0.001,
                 slippage_pct: float = 0.0005, price_lookup=None):
        """``price_lookup`` is a callable symbol -> float provided by the
        runner (live candle close or data provider's last price)."""
        if starting_cash <= 0:
            raise BrokerError("starting_cash must be positive")
        self._cash = float(starting_cash)
        self.starting_cash = float(starting_cash)
        self.fees_pct = float(fees_pct)
        self.slippage_pct = float(slippage_pct)
        self._price_lookup = price_lookup
        self._balances: dict[str, float] = {}  # base asset quantity per symbol
        self.orders: list[Order] = []

    # ------------------------------------------------------------------ wiring
    def set_price_lookup(self, fn) -> None:
        self._price_lookup = fn

    # ------------------------------------------------------------------ prices
    def last_price(self, symbol: str) -> float:
        if self._price_lookup is None:
            raise BrokerError("PaperBroker has no price lookup configured")
        px = float(self._price_lookup(symbol))
        if px <= 0:
            raise BrokerError(f"no valid price for {symbol}")
        return px

    # ------------------------------------------------------------------ orders
    def buy_market(self, symbol: str, quantity: float, reason: str = "entry") -> Order:
        px = self.last_price(symbol) * (1.0 + self.slippage_pct)
        notional = px * quantity
        fee = notional * self.fees_pct
        if notional + fee > self._cash + 1e-9:
            raise BrokerError(
                f"insufficient paper cash: need {notional + fee:.2f}, have {self._cash:.2f}")
        self._cash -= notional + fee
        self._balances[symbol] = self._balances.get(symbol, 0.0) + quantity
        order = Order(
            symbol=symbol, side=Side.BUY, quantity=quantity, filled_price=px,
            filled_qty=quantity, fee_paid=fee, status=OrderStatus.FILLED,
            filled_at=utc_now(), broker_id=f"paper-{len(self.orders) + 1}",
        )
        self.orders.append(order)
        logger.info("PAPER BUY  {} qty={:.8f} @ {:.6f} fee={:.4f} ({} )",
                    symbol, quantity, px, fee, reason)
        return order

    def sell_market(self, symbol: str, quantity: float, reason: str = "exit") -> Order:
        held = self._balances.get(symbol, 0.0)
        if quantity > held + 1e-12:
            raise BrokerError(
                f"cannot sell {quantity:.8f} {symbol}; paper balance is {held:.8f}")
        px = self.last_price(symbol) * (1.0 - self.slippage_pct)
        proceeds = px * quantity
        fee = proceeds * self.fees_pct
        self._cash += proceeds - fee
        self._balances[symbol] = held - quantity
        order = Order(
            symbol=symbol, side=Side.SELL, quantity=quantity, filled_price=px,
            filled_qty=quantity, fee_paid=fee, status=OrderStatus.FILLED,
            filled_at=utc_now(), broker_id=f"paper-{len(self.orders) + 1}",
        )
        self.orders.append(order)
        logger.info("PAPER SELL {} qty={:.8f} @ {:.6f} fee={:.4f} ({} )",
                    symbol, quantity, px, fee, reason)
        return order

    # ------------------------------------------------------------------ state
    def cash(self) -> float:
        return self._cash

    def positions(self) -> dict[str, float]:
        return {s: q for s, q in self._balances.items() if q > 1e-12}

    def equity(self, marks: dict[str, float] | None = None) -> float:
        marks = marks or {}
        positions_value = 0.0
        for symbol, qty in self.positions().items():
            price = marks.get(symbol) or self.last_price(symbol)
            positions_value += qty * price
        return self._cash + positions_value

    def restore_state(self, cash: float, balances: dict[str, float]) -> None:
        """Reload state after a restart (from SQLite)."""
        self._cash = float(cash)
        self._balances = dict(balances)
        logger.info("PaperBroker state restored: cash={:.2f} balances={}",
                    self._cash, {k: round(v, 8) for k, v in self._balances.items()})

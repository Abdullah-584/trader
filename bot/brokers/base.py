"""Broker interface.

Both PaperBroker and CcxtBroker implement this, so the runner code path is
identical regardless of execution venue. Only spot, long-only market
orders are supported — by design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from bot.core.models import Order


class BrokerError(RuntimeError):
    """Raised when an order cannot be placed or a broker call fails."""


class Broker(ABC):
    name: str = "abstract"

    @abstractmethod
    def buy_market(self, symbol: str, quantity: float,
                   reason: str = "entry") -> Order:
        """Buy ``quantity`` units at market."""

    @abstractmethod
    def sell_market(self, symbol: str, quantity: float,
                    reason: str = "exit") -> Order:
        """Sell ``quantity`` units at market."""

    @abstractmethod
    def last_price(self, symbol: str) -> float:
        """Latest tradable price for ``symbol``."""

    @abstractmethod
    def cash(self) -> float:
        """Free quote-currency balance."""

    @abstractmethod
    def positions(self) -> dict[str, float]:
        """Base-asset quantities currently held, keyed by symbol."""

"""Broker abstraction: one interface for paper and live execution."""

from bot.brokers.base import Broker, BrokerError

__all__ = ["Broker", "BrokerError"]

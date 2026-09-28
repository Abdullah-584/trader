"""Quotex integration: unofficial provider/broker, DEMO-first, guarded."""

from bot.quotex.provider import QuotexDataProvider, extract_ssid_token
from bot.quotex.broker import QuotexBroker, QuotexBrokerError

__all__ = ["QuotexDataProvider", "extract_ssid_token", "QuotexBroker",
           "QuotexBrokerError"]

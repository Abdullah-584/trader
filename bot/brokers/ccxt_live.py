"""CcxtBroker — OPTIONAL live execution. Never used unless the operator
arms live mode in config.yaml AND types the confirmation phrase.

Safety properties:
* Spot market orders only, long-only, quantity capped by the risk manager
  before the order is created.
* Keys come exclusively from the environment (.env). They are never logged.
* Every order id, fee, and fill is returned to the caller for persistence
  and reconciliation.
"""

from __future__ import annotations

from loguru import logger

from bot.brokers.base import Broker, BrokerError
from bot.core.models import Order, OrderStatus, Side, utc_now


class CcxtBroker(Broker):
    name = "ccxt_live"

    def __init__(self, exchange_id: str, api_key: str, secret: str,
                 *, dry_price_lookup=None):
        if not api_key or not secret:
            raise BrokerError(
                "live broker requires API keys in the environment "
                "(.env, git-ignored). Never commit keys.")
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover
            raise BrokerError("ccxt is not installed") from exc
        klass = getattr(ccxt, exchange_id, None)
        if klass is None:
            raise BrokerError(f"unknown exchange id {exchange_id!r}")
        self._ex = klass({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self._price_lookup = dry_price_lookup

    # ------------------------------------------------------------------ prices
    def last_price(self, symbol: str) -> float:
        ticker = self._ex.fetch_ticker(symbol)
        px = ticker.get("last") or ticker.get("close")
        if not px:
            raise BrokerError(f"no last price for {symbol}")
        return float(px)

    # ------------------------------------------------------------------ orders
    def buy_market(self, symbol: str, quantity: float, reason: str = "entry") -> Order:
        order = self._market(symbol, "buy", quantity, reason)
        return order

    def sell_market(self, symbol: str, quantity: float, reason: str = "exit") -> Order:
        return self._market(symbol, "sell", quantity, reason)

    def _market(self, symbol: str, side: str, quantity: float, reason: str) -> Order:
        logger.warning("LIVE ORDER: {} {} {} ({})", side.upper(), quantity, symbol, reason)
        try:
            res = self._ex.create_order(symbol, "market", side, quantity)
        except Exception as exc:
            logger.error("live order failed: {}", exc)
            raise BrokerError(f"live {side} order failed: {exc}") from exc
        filled_qty = float(res.get("filled") or 0.0)
        avg = res.get("average") or res.get("price") or 0.0
        fee_cost = 0.0
        fees = res.get("fees") or []
        for f in fees:
            fee_cost += float(f.get("cost") or 0.0)
        order = Order(
            symbol=symbol,
            side=Side.BUY if side == "buy" else Side.SELL,
            quantity=quantity,
            filled_price=float(avg) if avg else None,
            filled_qty=filled_qty,
            fee_paid=fee_cost,
            status=OrderStatus.FILLED if filled_qty > 0 else OrderStatus.PENDING,
            filled_at=utc_now(),
            broker_id=str(res.get("id") or ""),
        )
        logger.info("LIVE FILL {}: id={} qty={:.8f} avg={}", side, order.broker_id,
                    filled_qty, avg)
        return order

    # ------------------------------------------------------------------ state
    def cash(self) -> float:
        balance = self._ex.fetch_balance()
        return float(balance.get("USDT", {}).get("free", 0.0) if "USDT"
                     in balance else balance.get("free", {}).get("USDT", 0.0))

    def positions(self) -> dict[str, float]:
        balance = self._ex.fetch_balance()
        out: dict[str, float] = {}
        for asset, qty in (balance.get("total") or {}).items():
            if asset == "USDT" or float(qty or 0.0) <= 1e-12:
                continue
            # convert asset balances to symbol keys used by the watchlist
            out[f"{asset}/USDT"] = float(qty)
        return out

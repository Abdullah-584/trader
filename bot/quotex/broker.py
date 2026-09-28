"""Quotex binary-options broker — OPTIONAL, unofficial, high risk.

This broker does NOT fit the spot Broker interface because binary options
have no stops, no position management, and a binary settlement: a trade is
a fixed-stake CALL/PUT with a fixed expiry that resolves to WIN (stake *
payout), LOSS (stake), or DRAW (stake returned). It exposes the
binary-native operation instead:

    order = broker.buy(asset, amount, direction, expiry_seconds)

SAFETY MODEL
------------
* AccountType.DEMO is the default and is re-asserted on every connect.
* REAL requires quotex.trading.account: real in config AND a typed
  confirmation phrase (bot.quotex.safety) — the same two-key pattern as
  spot live trading.
* Every trade is placed through BinaryRiskManager; nothing here decides size.
* Credentials come only from the environment (.env).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger

from bot.quotex.bridge import run_async
from bot.quotex.provider import QuotexDataProvider, extract_ssid_token


class QuotexBrokerError(RuntimeError):
    """Raised when a binary trade cannot be placed or the client fails."""


@dataclass(frozen=True)
class BinaryTrade:
    asset: str
    direction: str              # "CALL" | "PUT"
    amount: float
    expiry_seconds: int
    account: str                # "demo" | "real"
    order_id: str = ""
    placed_at: datetime = datetime.now(timezone.utc)
    payout_at_placement: float | None = None


@dataclass(frozen=True)
class BinaryTradeResult:
    order_id: str
    result: str                 # "WIN" | "LOSS" | "DRAW" | "UNKNOWN"
    profit: float
    raw: dict | None = None


class QuotexBroker:
    name = "quotex"

    def __init__(self, provider: QuotexDataProvider, *, account: str = "demo",
                 confirmed_real: bool = False):
        if account not in ("demo", "real"):
            raise QuotexBrokerError(f"unknown account {account!r}")
        if account == "real" and not confirmed_real:
            raise QuotexBrokerError(
                "REAL account requested without typed confirmation — refusing. "
                "Run the confirmation flow (bot.quotex.safety) first.")
        self.provider = provider
        self.account = account
        self._api = None

    # ------------------------------------------------------------------ client
    def _client(self):
        if self._api is not None:
            return self._api
        # Reuse the provider's connected client (same session, same SSID).
        self._api = self.provider._ensure_connected()
        self._assert_account()
        return self._api

    def _assert_account(self) -> None:
        """Switch to the configured account and verify it took effect."""
        api = self._client0()
        want = "demo" if self.account == "demo" else "real"
        try:
            from QuotexAPI import AccountType
            from bot.quotex.bridge import run_async as _ra
            _ra(api.switch_account(
                AccountType.DEMO if want == "demo" else AccountType.REAL))
        except Exception as exc:
            if self.account == "real":
                raise QuotexBrokerError(f"account switch to REAL failed: {exc}") from exc
            logger.warning("quotex: account switch to DEMO failed ({}); "
                           "will verify balance instead", exc)
        if self.account == "real":
            logger.warning("QUOTEX REAL ACCOUNT ACTIVE — real money at risk")
        else:
            logger.info("quotex account: DEMO (paper stakes only)")

    def _client0(self):
        return self._api if self._api is not None else self.provider._ensure_connected()

    # ------------------------------------------------------------------ trading
    def buy(self, asset: str, amount: float, direction: str,
            expiry_seconds: int, *, payout: float | None = None) -> BinaryTrade:
        """Place a binary CALL/PUT. direction: 'call'|'put' (case-insensitive)."""
        if direction.lower() not in ("call", "put"):
            raise QuotexBrokerError(f"binary direction must be call/put, got {direction!r}")
        if amount <= 0:
            raise QuotexBrokerError(f"stake must be positive, got {amount}")
        api = self._client()

        def _buy():
            from QuotexAPI import TradeDirection
            td = TradeDirection.CALL if direction.lower() == "call" else TradeDirection.PUT
            return run_async(api.buy(asset=asset, amount=amount, direction=td,
                                     expiry=expiry_seconds))

        try:
            res = _buy()
        except Exception as exc:
            raise QuotexBrokerError(f"quotex buy {direction} {asset} failed: {exc}") from exc

        order_id = ""
        if isinstance(res, dict):
            order_id = str(res.get("id") or res.get("order_id") or "")
        elif res is not None and hasattr(res, "order_id"):
            order_id = str(getattr(res, "order_id", ""))
        trade = BinaryTrade(
            asset=asset, direction=direction.upper(), amount=amount,
            expiry_seconds=expiry_seconds, account=self.account,
            order_id=order_id, payout_at_placement=payout,
        )
        logger.warning("QUOTEX {} TRADE: {} {} ${:.2f} expiry={}s account={} id={}",
                       trade.account.upper(), trade.direction, trade.asset,
                       trade.amount, trade.expiry_seconds, trade.account,
                       trade.order_id or "n/a")
        return trade

    # ------------------------------------------------------------------ tracking
    def wait_for_result(self, order_id: str, timeout: float = 300.0) -> BinaryTradeResult:
        api = self._client()

        def _wait():
            return run_async(api.wait_for_result(order_id, timeout=timeout),
                             timeout=timeout + 30.0)

        try:
            res = _wait()
        except Exception as exc:
            logger.error("quotex wait_for_result({}) failed: {}", order_id, exc)
            return BinaryTradeResult(order_id=order_id, result="UNKNOWN", profit=0.0)

        result = "UNKNOWN"
        profit = 0.0
        raw = None
        if isinstance(res, dict):
            raw = res
            result = str(res.get("result") or "UNKNOWN").upper()
            profit = float(res.get("profit") or 0.0)
        else:
            result = str(getattr(res, "result", "UNKNOWN")).upper().split(".")[-1]
            profit = float(getattr(res, "profit", 0.0) or 0.0)
            raw = {"result": result, "profit": profit}
        logger.info("quotex trade {} -> {} (profit {:.2f})", order_id, result, profit)
        return BinaryTradeResult(order_id=order_id, result=result,
                                 profit=profit, raw=raw)

    # ------------------------------------------------------------------ account
    def balance(self) -> float | None:
        try:
            api = self._client()

            def _bal():
                return run_async(api.get_balance())

            res = _bal()
            for attr in ("amount", "balance", "value"):
                if isinstance(res, dict) and res.get(attr) is not None:
                    return float(res[attr])
                if res is not None and hasattr(res, attr):
                    return float(getattr(res, attr))
            return None
        except Exception as exc:
            logger.warning("quotex balance fetch failed: {}", exc)
            return None

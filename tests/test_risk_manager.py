"""Tests for the risk manager: sizing, caps, kill switches, trailing stops."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot.core.models import Position, Side
from bot.risk.manager import RiskManager

D1 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
D2 = datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc)


def _pos(symbol: str = "BTC/USDT", qty: float = 0.1, entry: float = 100.0) -> Position:
    return Position(
        symbol=symbol, side=Side.BUY, quantity=qty, entry_price=entry,
        entry_time=D1, stop_price=entry * 0.98, take_profit_price=entry * 1.04,
    )


# --------------------------------------------------------------------- sizing
def test_sizing_from_risk_and_stop_distance(risk_cfg):
    rm = RiskManager(risk_cfg, starting_equity=10_000)
    # risk budget = 1% * 10k = 100; stop distance = 10 -> qty 10
    d = rm.plan_entry(symbol="BTC/USDT", price=100.0, stop_distance=10.0,
                      open_positions=[])
    assert d.approved
    assert d.quantity == pytest.approx(10.0)
    assert d.stop_price == pytest.approx(90.0)
    assert d.notional == pytest.approx(1000.0)


def test_sizing_capped_by_max_position_pct(risk_cfg):
    rm = RiskManager(risk_cfg, starting_equity=10_000)
    # tiny stop distance would imply a huge position; must be capped at 25% equity
    d = rm.plan_entry(symbol="BTC/USDT", price=100.0, stop_distance=0.1,
                      open_positions=[])
    assert d.approved
    assert d.notional <= 10_000 * 0.25 + 1e-9
    assert d.quantity == pytest.approx(25.0)  # 2500 / 100


def test_sizing_respects_risk_budget_after_cap(risk_cfg):
    rm = RiskManager(risk_cfg, starting_equity=10_000)
    d = rm.plan_entry(symbol="BTC/USDT", price=50_000.0, stop_distance=1_000.0,
                      open_positions=[])
    # risk qty = 100 / 1000 = 0.1 units -> notional 5000 > 2500 cap -> capped
    assert d.quantity * d.stop_price is not None
    assert d.notional <= 2_500 + 1e-6


def test_take_profit_is_r_multiple_of_stop(risk_cfg):
    rm = RiskManager(risk_cfg, starting_equity=10_000)
    d = rm.plan_entry(symbol="X", price=100.0, stop_distance=10.0, open_positions=[])
    assert d.take_profit_price == pytest.approx(120.0)  # 2R


def test_min_stop_distance_floor(risk_cfg):
    rm = RiskManager(risk_cfg, starting_equity=10_000)
    d = rm.plan_entry(symbol="X", price=100.0, stop_distance=0.0001, open_positions=[])
    # floor = 0.2% of price -> stop at 99.8 despite requested 0.0001 distance
    assert d.stop_price == pytest.approx(100.0 * (1 - 0.002))


def test_rejects_duplicate_symbol_and_max_positions(risk_cfg):
    rm = RiskManager(risk_cfg, starting_equity=10_000)
    held = [_pos("BTC/USDT")]
    d = rm.plan_entry(symbol="BTC/USDT", price=100.0, stop_distance=5.0,
                      open_positions=held)
    assert not d.approved and "already in position" in d.reason

    three = [_pos(f"S{i}") for i in range(3)]
    d = rm.plan_entry(symbol="NEW", price=100.0, stop_distance=5.0,
                      open_positions=three)
    assert not d.approved and "max concurrent" in d.reason


def test_rejects_when_dust_notional(risk_cfg):
    rm = RiskManager(risk_cfg, starting_equity=30)  # 25% cap -> 7.50 < 10 min
    d = rm.plan_entry(symbol="X", price=50_000.0, stop_distance=500.0,
                      open_positions=[])
    assert not d.approved and "below minimum" in d.reason


# ---------------------------------------------------------------- kill switches
def test_daily_loss_kill_switch_trips_and_resets_next_day(risk_cfg):
    rm = RiskManager(risk_cfg, starting_equity=10_000)
    rm.on_bar(D1, marked_equity=9_650, open_positions=0)  # -3.5% on the day
    assert rm.kill_switch_tripped
    assert rm.trading_halted

    rm.on_bar(D2, marked_equity=9_650, open_positions=0)  # new day
    assert not rm.kill_switch_tripped
    assert not rm.trading_halted


def test_drawdown_circuit_breaker_latches(risk_cfg):
    rm = RiskManager(risk_cfg, starting_equity=10_000)
    rm.on_bar(D1, marked_equity=12_000, open_positions=0)
    rm.on_bar(D1.replace(hour=13), marked_equity=10_100, open_positions=0)  # -15.8% dd
    assert rm.drawdown_halted
    # stays halted even on a new day (unlike the daily kill switch)
    rm.on_bar(D2, marked_equity=10_500, open_positions=0)
    assert rm.drawdown_halted
    assert rm.trading_halted


def test_no_entries_while_halted(risk_cfg):
    rm = RiskManager(risk_cfg, starting_equity=10_000)
    rm.on_bar(D1, marked_equity=9_000, open_positions=0)
    d = rm.plan_entry(symbol="X", price=100.0, stop_distance=5.0, open_positions=[])
    assert not d.approved and "halted" in d.reason


# -------------------------------------------------------------------- trailing
def test_trailing_stop_ratchets_up_only(risk_cfg):
    rm = RiskManager(risk_cfg, starting_equity=10_000)
    pos = _pos(entry=100.0)
    pos.stop_price = 90.0
    pos.highest_price_since_entry = 120.0
    trail = rm.trailing_stop_price(pos, atr_value=2.0)  # 120 - 6 = 114
    assert trail == pytest.approx(114.0)
    pos.trail_price = trail
    pos.highest_price_since_entry = 118.0  # price falls
    trail2 = rm.trailing_stop_price(pos, atr_value=2.0)  # 118-6=112 < 114
    assert trail2 == pytest.approx(114.0)  # never ratchets down
    # never below the initial stop either
    pos.highest_price_since_entry = 92.0
    assert rm.trailing_stop_price(pos, 2.0) >= pos.stop_price


def test_trailing_disabled_returns_none(risk_cfg):
    from dataclasses import replace
    cfg = replace(risk_cfg, trailing_stop=False)
    rm = RiskManager(cfg, starting_equity=10_000)
    assert rm.trailing_stop_price(_pos(), 2.0) is None

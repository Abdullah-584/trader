"""Tests for the Quotex integration: binary risk gating (hard caps),
SSID parsing, timeframe parsing, and candle payload normalization.
All offline — no network, no QuotexAPI import needed."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from bot.risk.binary import (
    BINARY_HARD_CAPS,
    BinaryRiskConfig,
    BinaryRiskManager,
)
from bot.quotex.provider import QuotexDataProvider, extract_ssid_token

NOW = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)


def _mgr(**overrides) -> BinaryRiskManager:
    cfg_map = {"max_trade_pct": 0.01, "max_open_trades": 2,
               "max_daily_loss_pct": 0.05, "min_payout_pct": 0.80,
               "cooldown_seconds": 0, **overrides}
    return BinaryRiskManager(BinaryRiskConfig.from_mapping(cfg_map),
                             starting_balance=1_000.0)


# ------------------------------------------------------------ hard caps
def test_config_clamps_to_hard_caps():
    cfg = BinaryRiskConfig.from_mapping({
        "max_trade_pct": 0.50, "max_open_trades": 50,
        "max_daily_loss_pct": 0.90, "min_payout_pct": 0.10,
    })
    assert cfg.max_trade_pct == BINARY_HARD_CAPS["max_trade_pct"]
    assert cfg.max_open_trades == BINARY_HARD_CAPS["max_open_trades"]
    assert cfg.max_daily_loss_pct == BINARY_HARD_CAPS["max_daily_loss_pct"]
    assert cfg.min_payout_pct == BINARY_HARD_CAPS["min_payout_pct"]


def test_stake_is_pct_of_balance():
    mgr = _mgr(max_trade_pct=0.02)
    d = mgr.approve(asset="EURUSD", payout_pct=0.85, open_trades=0, now=NOW)
    assert d.approved
    assert d.stake == pytest.approx(20.0)  # 2% of 1000


# ---------------------------------------------------------------- gates
def test_payout_gate_blocks_low_payouts():
    mgr = _mgr(min_payout_pct=0.80)
    d = mgr.approve(asset="X", payout_pct=0.70, open_trades=0, now=NOW)
    assert not d.approved and "payout" in d.reason
    d2 = mgr.approve(asset="X", payout_pct=None, open_trades=0, now=NOW)
    # unknown payout: allowed (health check surfaces data problems separately)
    assert d2.approved


def test_max_open_trades_gate():
    mgr = _mgr(max_open_trades=2)
    d = mgr.approve(asset="X", payout_pct=0.9, open_trades=2, now=NOW)
    assert not d.approved and "max open trades" in d.reason


def test_cooldown_gate():
    mgr = _mgr(cooldown_seconds=120)
    d1 = mgr.approve(asset="EURUSD", payout_pct=0.9, open_trades=0, now=NOW)
    assert d1.approved
    d2 = mgr.approve(asset="EURUSD", payout_pct=0.9,
                     open_trades=0, now=datetime(2025, 6, 1, 12, 1, tzinfo=timezone.utc))
    assert not d2.approved and "cooldown" in d2.reason
    # different asset unaffected
    d3 = mgr.approve(asset="GBPUSD", payout_pct=0.9,
                     open_trades=0, now=datetime(2025, 6, 1, 12, 1, tzinfo=timezone.utc))
    assert d3.approved


def test_refuses_to_upsize_to_broker_minimum():
    # 1% of 50 = 0.50 < $1 minimum: must refuse, never silently upsize
    mgr = BinaryRiskManager(BinaryRiskConfig.from_mapping({"max_trade_pct": 0.01}),
                            starting_balance=50.0)
    d = mgr.approve(asset="X", payout_pct=0.9, open_trades=0, now=NOW)
    assert not d.approved and "below broker minimum" in d.reason


def test_daily_kill_switch_and_reset():
    mgr = _mgr(max_daily_loss_pct=0.05)
    mgr.on_settled(NOW, pnl=-30.0, new_balance=970.0)  # -3%
    assert not mgr.kill_switch_tripped
    mgr.on_settled(NOW, pnl=-25.0, new_balance=945.0)  # -5.5% on the day
    assert mgr.kill_switch_tripped
    d = mgr.approve(asset="X", payout_pct=0.95, open_trades=0, now=NOW)
    assert not d.approved and "kill switch" in d.reason
    # next day resets
    nxt = datetime(2025, 6, 2, 9, 0, tzinfo=timezone.utc)
    mgr._roll_day(nxt)
    d = mgr.approve(asset="X", payout_pct=0.95, open_trades=0, now=nxt)
    assert d.approved


# ----------------------------------------------------------- ssid parsing
def test_extract_ssid_from_full_cookie_format():
    raw = '42["authorization",{"session":"dJzhzzKSR6N4Lr5OvTFuvcGLCfyDjtdb","isDemo":1}]'
    assert extract_ssid_token(raw) == "dJzhzzKSR6N4Lr5OvTFuvcGLCfyDjtdb"


def test_extract_ssid_bare_token_passthrough():
    assert extract_ssid_token("  abc123  ") == "abc123"


def test_extract_ssid_empty_raises():
    from bot.data.provider import DataProviderError
    with pytest.raises(DataProviderError):
        extract_ssid_token("   ")


def test_period_seconds_parsing():
    assert QuotexDataProvider._period_seconds("60") == 60
    assert QuotexDataProvider._period_seconds("1m") == 60
    assert QuotexDataProvider._period_seconds("5m") == 300
    assert QuotexDataProvider._period_seconds("1h") == 3600
    from bot.data.provider import DataProviderError
    with pytest.raises(DataProviderError):
        QuotexDataProvider._period_seconds("7m")


# ------------------------------------------------- payload normalization
def test_normalize_dict_payload():
    raw = {"data": [
        {"time": 1700000000, "open": 1.1, "max": 1.2, "min": 1.0, "close": 1.15,
         "volume": 10},
        {"time": 1700000060, "open": 1.15, "max": 1.25, "min": 1.05, "close": 1.2,
         "volume": 12},
    ]}
    df = QuotexDataProvider._normalize_raw(raw, "EURUSD")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df["close"].iloc[-1] == pytest.approx(1.2)


def test_normalize_list_payload_and_garbage():
    raw = [[1700000000, 1.1, 1.2, 1.0, 1.15, 10]]
    df = QuotexDataProvider._normalize_raw(raw, "EURUSD")
    assert len(df) == 1
    from bot.data.provider import DataProviderError
    with pytest.raises(DataProviderError):
        QuotexDataProvider._normalize_raw({"unexpected": "shape"}, "X")


def test_broker_refuses_real_without_confirmation():
    from bot.quotex.broker import QuotexBroker, QuotexBrokerError
    with pytest.raises(QuotexBrokerError):
        QuotexBroker(provider=None, account="real", confirmed_real=False)

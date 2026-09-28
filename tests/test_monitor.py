"""Tests for the Quotex signal monitor: break-even math, outcome
resolution, and the suggestion gating logic."""

from __future__ import annotations

import pytest

from bot.monitor.quotex import breakeven_hit_rate, outcome_for


# ---------------------------------------------------------------- break-even
def test_breakeven_at_85_payout():
    # classic binary math: 1 / 1.85 = 0.5405...
    assert breakeven_hit_rate(0.85) == pytest.approx(0.5405, abs=1e-3)


def test_breakeven_at_92_payout():
    assert breakeven_hit_rate(0.92) == pytest.approx(0.521, abs=1e-3)


def test_breakeven_monotonic_in_payout():
    # better payout -> lower required hit rate, always
    assert breakeven_hit_rate(0.95) < breakeven_hit_rate(0.80) < breakeven_hit_rate(0.50)


def test_breakeven_never_below_coin_flip():
    # no payout can make break-even easier than 50%
    assert breakeven_hit_rate(1.0) == pytest.approx(0.5)


# ----------------------------------------------------------------- outcomes
def test_outcome_up_win_and_loss():
    assert outcome_for(1, 100.0, 101.0) == "WIN"
    assert outcome_for(1, 100.0, 99.0) == "LOSS"


def test_outcome_down_win_and_loss():
    assert outcome_for(-1, 100.0, 99.0) == "WIN"
    assert outcome_for(-1, 100.0, 101.0) == "LOSS"


def test_outcome_tie_on_flat():
    assert outcome_for(1, 100.0, 100.0) == "TIE"
    assert outcome_for(-1, 100.0, 100.0) == "TIE"


def test_outcome_tie_tolerance_on_tiny_moves():
    # relative move under ~1e-12 counts as flat, not a 1-tick win
    assert outcome_for(1, 100.0, 100.0 + 1e-13) == "TIE"

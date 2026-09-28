"""Signal aggregator: combine strategy votes into a single verdict.

Modes
-----
* ``weighted_vote`` — sum(weight * confidence) per direction; the winning
  direction must also hold a minimum share of total positive weight.
* ``majority``      — simple majority of non-flat voters.
* ``unanimous``     — every enabled strategy must agree.

The output confidence is normalized to [0, 1] and is NOT a probability of
profit — it is agreement among strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class AggregatedBar:
    symbol: str
    timestamp: pd.Timestamp
    direction: int
    confidence: float
    votes_long: int
    votes_short: int
    votes_flat: int
    contributors: tuple[str, ...]


class SignalAggregator:
    """Combine strategy votes.

    ``weighted_vote`` semantics (default): only strategies with an actual
    opinion (long or short) form the denominator; flat strategies are not
    counted AGAINST a direction — a mostly-flat ensemble would otherwise
    veto every entry. ``min_confidence`` is therefore the share of
    directional weight that must agree, and ``min_votes`` (default 1)
    requires at least that many strategies to vote the same direction.
    """

    def __init__(self, mode: str = "weighted_vote", min_confidence: float = 0.45,
                 weights: dict[str, float] | None = None,
                 min_votes: int = 1):
        if mode not in {"weighted_vote", "majority", "unanimous"}:
            raise ValueError(f"unknown aggregation mode {mode!r}")
        self.mode = mode
        self.min_confidence = float(min_confidence)
        self.weights = dict(weights or {})
        self.min_votes = max(1, int(min_votes))

    # ------------------------------------------------------------------
    def aggregate_one(self, symbol: str, timestamp: pd.Timestamp,
                      strategy_frames: dict[str, pd.DataFrame], row: int) -> AggregatedBar:
        """Aggregate one bar across strategies for one symbol."""
        long_w = short_w = flat_w = 0.0
        n_long = n_short = n_flat = 0
        contributors: list[str] = []

        for name, frame in strategy_frames.items():
            if row >= len(frame):
                continue
            w = float(self.weights.get(name, 1.0))
            if w <= 0:
                continue
            s = int(frame["signal"].iloc[row])
            c = float(frame["confidence"].iloc[row])
            if s > 0:
                long_w += w * c
                n_long += 1
                contributors.append(name)
            elif s < 0:
                short_w += w * abs(c)
                n_short += 1
            else:
                flat_w += w
                n_flat += 1

        direction, confidence = self._decide(long_w, short_w, flat_w,
                                             n_long, n_short, n_flat)
        return AggregatedBar(symbol, timestamp, direction, confidence,
                             n_long, n_short, n_flat, tuple(contributors))

    # ------------------------------------------------------------------
    def _decide(self, long_w, short_w, flat_w, n_long, n_short, n_flat
                ) -> tuple[int, float]:
        if self.mode == "majority":
            total_voters = n_long + n_short + n_flat
            if total_voters == 0:
                return 0, 0.0
            if n_long > total_voters / 2:
                return 1, min(1.0, n_long / max(1, n_long + n_short))
            return 0, 0.0

        if self.mode == "unanimous":
            enabled = n_long + n_short + n_flat
            if enabled == 0:
                return 0, 0.0
            if n_long == enabled:
                return 1, min(1.0, long_w / max(1e-9, enabled))
            return 0, 0.0

        # weighted_vote (default): directional weight only; flat abstains
        directional = long_w + short_w
        if directional <= 0:
            return 0, 0.0
        if long_w > short_w and long_w / directional >= self.min_confidence \
                and n_long >= self.min_votes:
            return 1, min(1.0, long_w / directional)
        # short-bias maps to exit in a long-only system; report it, the
        # executor treats an aggregate -1 as an exit vote
        if short_w > long_w and short_w / directional >= self.min_confidence \
                and n_short >= self.min_votes:
            return -1, min(1.0, short_w / directional)
        return 0, 0.0

    # ------------------------------------------------------------------
    def aggregate_frame(self, symbol: str, strategy_frames: dict[str, pd.DataFrame]
                        ) -> pd.DataFrame:
        """Vectorized-ish aggregation over whole frames (used by backtests)."""
        if not strategy_frames:
            return pd.DataFrame(columns=["direction", "confidence"])
        index = next(iter(strategy_frames.values())).index
        rows = [
            self.aggregate_one(symbol, ts, strategy_frames, i)
            for i, ts in enumerate(index)
        ]
        return pd.DataFrame({
            "direction": [r.direction for r in rows],
            "confidence": [r.confidence for r in rows],
            "votes_long": [r.votes_long for r in rows],
            "votes_short": [r.votes_short for r in rows],
            "votes_flat": [r.votes_flat for r in rows],
            "contributors": [",".join(r.contributors) for r in rows],
        }, index=index)

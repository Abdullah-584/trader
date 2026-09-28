"""SQLite persistence.

Tables
------
* orders       — every order attempt (paper and live) with fills and fees
* trades       — completed round trips
* signals      — aggregated verdicts per bar (audit trail)
* equity       — equity curve points (backtests are written with a run_id)
* runner_state — key/value blob for crash recovery (cash, balances, etc.)

SQLite with WAL mode is chosen deliberately: local, zero-config, free,
and crash-tolerant enough for a single-process trading bot.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from loguru import logger

from bot.core.models import Order, OrderStatus, Side, Trade, utc_now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    client_id TEXT PRIMARY KEY,
    broker_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    filled_qty REAL NOT NULL DEFAULT 0,
    filled_price REAL,
    fee REAL NOT NULL DEFAULT 0,
    reason TEXT,
    status TEXT NOT NULL,
    venue TEXT NOT NULL,
    created_at TEXT NOT NULL,
    filled_at TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    exit_time TEXT NOT NULL,
    fees_paid REAL NOT NULL,
    pnl REAL NOT NULL,
    pnl_pct REAL NOT NULL,
    reason TEXT,
    strategy_tag TEXT,
    run_id TEXT
);
CREATE TABLE IF NOT EXISTS signals (
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction INTEGER NOT NULL,
    confidence REAL NOT NULL,
    votes_long INTEGER, votes_short INTEGER, votes_flat INTEGER,
    contributors TEXT,
    run_id TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    ts TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    open_positions INTEGER NOT NULL,
    run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_equity_run ON equity(run_id, ts);
CREATE TABLE IF NOT EXISTS runner_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt.isoformat()


class Database:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.commit()
        logger.info("sqlite ready: {}", self.path)

    def close(self) -> None:
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------ orders
    def save_order(self, order: Order, venue: str) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO orders
               (client_id, broker_id, symbol, side, quantity, filled_qty,
                filled_price, fee, reason, status, venue, created_at, filled_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (order.client_id, order.broker_id, order.symbol, order.side.value,
             order.quantity, order.filled_qty, order.filled_price, order.fee_paid,
             order.reason.value if isinstance(order.reason, object) else str(order.reason),
             order.status.value, venue, _iso(order.created_at), _iso(order.filled_at)),
        )
        self.conn.commit()

    # ------------------------------------------------------------------ trades
    def save_trade(self, trade: Trade, run_id: str | None = None) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO trades
               (trade_id, symbol, side, quantity, entry_price, exit_price,
                entry_time, exit_time, fees_paid, pnl, pnl_pct, reason,
                strategy_tag, run_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (trade.trade_id, trade.symbol, trade.side.value, trade.quantity,
             trade.entry_price, trade.exit_price, _iso(trade.entry_time),
             _iso(trade.exit_time), trade.fees_paid, trade.pnl, trade.pnl_pct,
             trade.reason.value, trade.strategy_tag, run_id),
        )
        self.conn.commit()

    def recent_trades(self, limit: int = 200, run_id: str | None = None) -> list[dict]:
        if run_id:
            cur = self.conn.execute(
                "SELECT * FROM trades WHERE run_id=? ORDER BY exit_time DESC LIMIT ?",
                (run_id, limit))
        else:
            cur = self.conn.execute(
                "SELECT * FROM trades ORDER BY exit_time DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ signals
    def save_signal(self, ts: datetime, symbol: str, direction: int,
                    confidence: float, votes_long: int, votes_short: int,
                    votes_flat: int, contributors: str, run_id: str) -> None:
        self.conn.execute(
            """INSERT INTO signals
               (ts, symbol, direction, confidence, votes_long, votes_short,
                votes_flat, contributors, run_id)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (_iso(ts), symbol, direction, confidence, votes_long, votes_short,
             votes_flat, contributors, run_id),
        )

    def commit_signals(self) -> None:
        self.conn.commit()

    # ------------------------------------------------------------------ equity
    def save_equity_point(self, ts: datetime, equity: float, cash: float,
                          positions_value: float, open_positions: int,
                          run_id: str) -> None:
        self.conn.execute(
            """INSERT INTO equity (ts, equity, cash, positions_value,
                                   open_positions, run_id)
               VALUES (?,?,?,?,?,?)""",
            (_iso(ts), equity, cash, positions_value, open_positions, run_id),
        )

    def equity_curve(self, run_id: str | None = None) -> list[dict]:
        if run_id:
            cur = self.conn.execute(
                "SELECT * FROM equity WHERE run_id=? ORDER BY ts", (run_id,))
        else:
            cur = self.conn.execute("SELECT * FROM equity ORDER BY ts")
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------ state kv
    def set_state(self, key: str, value: dict) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO runner_state (key, value, updated_at)
               VALUES (?,?,?)""",
            (key, json.dumps(value), _iso(utc_now())),
        )
        self.conn.commit()

    def get_state(self, key: str) -> dict | None:
        cur = self.conn.execute(
            "SELECT value FROM runner_state WHERE key=?", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            logger.warning("corrupt state blob for key {} — ignored", key)
            return None

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def order_status_from(model_status: OrderStatus) -> str:
        return model_status.value

    @staticmethod
    def side_value(side: Side) -> str:
        return side.value

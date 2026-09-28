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
CREATE TABLE IF NOT EXISTS monitor_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    quotex_name TEXT NOT NULL,
    feed TEXT,
    symbol TEXT,
    direction INTEGER NOT NULL,
    confidence REAL NOT NULL,
    price REAL NOT NULL,
    expiry_seconds INTEGER NOT NULL,
    contributors TEXT,
    votes_long INTEGER, votes_short INTEGER, votes_flat INTEGER,
    suggested_at TEXT NOT NULL,
    resolved_at TEXT,
    exit_price REAL,
    outcome TEXT NOT NULL DEFAULT 'PENDING'
);
CREATE INDEX IF NOT EXISTS idx_monitor_name ON monitor_signals(quotex_name, ts);
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

    # ------------------------------------------------------------------ monitor
    def ensure_monitor_schema(self) -> None:
        self.conn.executescript(
            "CREATE TABLE IF NOT EXISTS monitor_signals ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts TEXT NOT NULL, quotex_name TEXT NOT NULL,"
            " feed TEXT, symbol TEXT,"
            " direction INTEGER NOT NULL, confidence REAL NOT NULL,"
            " price REAL NOT NULL, expiry_seconds INTEGER NOT NULL,"
            " contributors TEXT,"
            " votes_long INTEGER, votes_short INTEGER, votes_flat INTEGER,"
            " suggested_at TEXT NOT NULL, resolved_at TEXT,"
            " exit_price REAL, outcome TEXT NOT NULL DEFAULT 'PENDING')"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_monitor_name "
            "ON monitor_signals(quotex_name, ts)")
        self.conn.commit()

    def save_monitor_signal(self, *, ts: datetime, quotex_name: str, feed: str,
                            symbol: str, direction: int, confidence: float,
                            price: float, expiry_seconds: int,
                            contributors: str, votes_long: int,
                            votes_short: int, votes_flat: int) -> None:
        self.conn.execute(
            """INSERT INTO monitor_signals
               (ts, quotex_name, feed, symbol, direction, confidence, price,
                expiry_seconds, contributors, votes_long, votes_short,
                votes_flat, suggested_at, outcome)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, 'PENDING')""",
            (_iso(ts), quotex_name, feed, symbol, direction, confidence,
             price, expiry_seconds, contributors, votes_long, votes_short,
             votes_flat, _iso(utc_now())),
        )
        self.conn.commit()

    def pending_monitor_signals(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM monitor_signals WHERE outcome='PENDING' ORDER BY ts"
        ).fetchall()
        return [dict(r) for r in rows]

    def resolve_monitor_signal(self, signal_id: int, exit_price: float,
                               outcome: str) -> None:
        self.conn.execute(
            "UPDATE monitor_signals SET exit_price=?, outcome=?, resolved_at=? "
            "WHERE id=?",
            (exit_price, outcome, _iso(utc_now()), signal_id),
        )
        self.conn.commit()

    def monitor_signals_recent(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM monitor_signals ORDER BY suggested_at DESC LIMIT ?",
            (limit,))
        return [dict(r) for r in rows.fetchall()]

    # ------------------------------------------------------------------ quotex trades
    def ensure_quotex_schema(self) -> None:
        self.conn.executescript(
            "CREATE TABLE IF NOT EXISTS quotex_trades ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts TEXT NOT NULL, asset TEXT NOT NULL,"
            " direction TEXT NOT NULL, amount REAL NOT NULL,"
            " expiry_seconds INTEGER NOT NULL, account TEXT NOT NULL,"
            " payout_at_placement REAL, order_id TEXT,"
            " status TEXT NOT NULL DEFAULT 'OPEN',"
            " result TEXT, profit REAL, settled_at TEXT)"
        )
        self.conn.commit()

    def save_quotex_trade(self, *, ts: datetime, asset: str, direction: str,
                          amount: float, expiry_seconds: int, account: str,
                          payout_at_placement: float | None,
                          order_id: str) -> int:
        cur = self.conn.execute(
            """INSERT INTO quotex_trades
               (ts, asset, direction, amount, expiry_seconds, account,
                payout_at_placement, order_id, status)
               VALUES (?,?,?,?,?,?,?,?, 'OPEN')""",
            (_iso(ts), asset, direction, amount, expiry_seconds, account,
             payout_at_placement, order_id),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def open_quotex_trades(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM quotex_trades WHERE status='OPEN' ORDER BY ts")
        return [dict(r) for r in rows.fetchall()]

    def settle_quotex_trade(self, trade_id: int, result: str,
                            profit: float) -> None:
        self.conn.execute(
            "UPDATE quotex_trades SET status='SETTLED', result=?, profit=?, "
            "settled_at=? WHERE id=?",
            (result, profit, _iso(utc_now()), trade_id),
        )
        self.conn.commit()

    def quotex_trades_recent(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM quotex_trades ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows.fetchall()]

    def quotex_stats(self) -> dict:
        row = dict(self.conn.execute(
            "SELECT COUNT(*) AS n,"
            " SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) AS wins,"
            " SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) AS losses,"
            " SUM(CASE WHEN result='DRAW' THEN 1 ELSE 0 END) AS draws,"
            " SUM(profit) AS profit_total"
            " FROM quotex_trades WHERE status='SETTLED'").fetchone())
        resolved = (row.get("wins") or 0) + (row.get("losses") or 0) \
            + (row.get("draws") or 0)
        row["resolved"] = resolved
        row["hit_rate"] = (row.get("wins") or 0) / resolved if resolved else None
        return row

    def monitor_stats(self) -> dict:
        """Hit-rate stats over RESOLVED suggestions, overall and per instrument."""
        overall = dict(self.conn.execute(
            "SELECT COUNT(*) AS n,"
            " SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) AS wins,"
            " SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) AS losses,"
            " SUM(CASE WHEN outcome='TIE' THEN 1 ELSE 0 END) AS ties,"
            " SUM(CASE WHEN outcome='PENDING' THEN 1 ELSE 0 END) AS pending"
            " FROM monitor_signals").fetchone())
        by_inst_rows = self.conn.execute(
            "SELECT quotex_name, COUNT(*) AS n,"
            " SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) AS wins,"
            " SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) AS losses,"
            " SUM(CASE WHEN outcome='TIE' THEN 1 ELSE 0 END) AS ties"
            " FROM monitor_signals GROUP BY quotex_name ORDER BY n DESC"
        ).fetchall()
        resolved = (overall.get("wins") or 0) + (overall.get("losses") or 0) \
            + (overall.get("ties") or 0)
        overall["resolved"] = resolved
        overall["hit_rate"] = (overall.get("wins") or 0) / resolved if resolved else None
        return {
            "overall": overall,
            "by_instrument": [dict(r) for r in by_inst_rows],
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def order_status_from(model_status: OrderStatus) -> str:
        return model_status.value

    @staticmethod
    def side_value(side: Side) -> str:
        return side.value

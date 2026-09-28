"""Local Streamlit dashboard: `python main.py dashboard` -> http://localhost:8501

Read-only views over the SQLite store written by the paper/live runner, plus
start/stop controls that communicate with the runner through the same DB
(no extra ports, no cloud).

Reminder shown in-app: nothing here predicts the future; equity is
hypothetical in paper mode.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@st.cache_resource(ttl=30)
def _load_cfg(path: str):
    from bot.core.config import load_config
    return load_config(path)


@st.cache_resource(ttl=30)
def _load_db(path: str):
    from bot.storage.db import Database
    return Database(path)


cfg = _load_cfg("config.yaml")
db = _load_db(cfg.root / str(cfg.storage.get("db_path", "data/trader.db")))

REFRESH = int(cfg.dashboard.get("refresh_seconds", 30))

st.set_page_config(page_title="trader — local paper trading",
                   page_icon="📈", layout="wide")

# --------------------------------------------------------------------------- header
mode = cfg.mode
st.title("📈 trader — local paper trading dashboard")
st.caption(
    f"mode: **{mode}** · paper trading is simulated — equity figures are "
    "hypothetical. No strategy guarantees profit and live trading can lose "
    "all capital."
)

# --------------------------------------------------------------------------- controls
c1, c2, c3 = st.columns([1, 1, 4])
if c1.button("▶ Start paper runner", use_container_width=True):
    import subprocess
    subprocess.Popen(
        [sys.executable, "main.py", "paper"],
        cwd=str(cfg.root),
        stdout=open(cfg.root / "data/logs/runner_stdout.log", "ab"),
        stderr=subprocess.STDOUT,
    )
    c1.success("runner starting in background — see logs for status")

if c2.button("■ Stop runner", use_container_width=True):
    db.set_state("command", {"action": "stop"})
    c2.warning("stop command queued — the runner exits at the next tick")

# --------------------------------------------------------------------------- state
state = db.get_state("runner") or {}
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
col_s1.metric("Runner run id", state.get("run_id", "—"))
col_s2.metric("Mode", state.get("mode", "paper"))
col_s3.metric("Cash", f"{state.get('cash', 0):,.2f}")
_positions_count = len(state.get("positions") or [])
col_s4.metric("Open positions", _positions_count)

st.divider()

# --------------------------------------------------------------------------- open positions
st.subheader("Open positions")
positions = state.get("positions") or []
if positions:
    st.dataframe(pd.DataFrame(positions), use_container_width=True)
else:
    st.info("No open positions.")

# --------------------------------------------------------------------------- equity curve
st.subheader("Equity curve")
eq_rows = db.equity_curve()
if eq_rows:
    eq = pd.DataFrame(eq_rows)
    eq["ts"] = pd.to_datetime(eq["ts"], utc=True)
    latest_run = eq.iloc[-1]["run_id"]
    eq = eq[eq["run_id"] == latest_run]
    st.line_chart(eq.set_index("ts")["equity"])
else:
    st.info("No equity points yet — start the paper runner or run a backtest.")

# --------------------------------------------------------------------------- trades
st.subheader("Trade history")
trades = db.recent_trades(limit=200)
if trades:
    tdf = pd.DataFrame(trades)
    tdf["exit_time"] = pd.to_datetime(tdf["exit_time"], utc=True)
    st.dataframe(
        tdf[["exit_time", "symbol", "quantity", "entry_price", "exit_price",
             "pnl", "pnl_pct", "reason", "strategy_tag"]],
        use_container_width=True,
    )
    st.caption(f"Total realized P&L across all runs: "
               f"**{tdf['pnl'].sum():,.2f} {cfg.base_currency}** (simulated)")
else:
    st.info("No trades yet.")

# --------------------------------------------------------------------------- signals
st.subheader("Signal log (latest run)")
sig_rows = db.conn.execute(
    "SELECT ts, symbol, direction, confidence, votes_long, votes_short, "
    "votes_flat, contributors FROM signals ORDER BY ts DESC LIMIT 100"
).fetchall()
if sig_rows:
    sdf = pd.DataFrame([dict(r) for r in sig_rows])
    st.dataframe(sdf, use_container_width=True)
else:
    st.info("No signals logged yet.")

# --------------------------------------------------------------------------- settings editor
st.subheader("Settings (config.yaml)")
st.caption("Editing here writes to config.yaml — the runner must be restarted "
           "to pick up changes. Risk values above the hard caps are clamped "
           "on load; they cannot be bypassed.")
current = cfg.raw
risk_cfg = current.get("risk") or {}
with st.expander("Risk settings", expanded=False):
    new_risk = {}
    numeric_fields = {
        "risk_per_trade": (0.0, 0.02, 0.001),
        "max_positions": (1, 10, 1),
        "max_daily_loss_pct": (0.0, 0.10, 0.005),
        "max_drawdown_pct": (0.0, 0.50, 0.01),
        "stop_atr_multiple": (0.5, 10.0, 0.1),
        "take_profit_r_multiple": (0.5, 10.0, 0.1),
        "fees_pct": (0.0, 0.01, 0.0001),
        "slippage_pct": (0.0, 0.01, 0.0001),
    }
    for key, (lo, hi, step) in numeric_fields.items():
        default = float(risk_cfg.get(key, 0))
        new_risk[key] = st.number_input(key, min_value=lo, max_value=hi,
                                        value=default, step=step)
    if st.button("Save risk settings"):
        import yaml as _yaml
        current["risk"] = {**risk_cfg, **new_risk}
        (cfg.root / "config.yaml").write_text(
            _yaml.safe_dump(current, sort_keys=False), encoding="utf-8")
        st.success("config.yaml updated — restart the runner to apply")

st.divider()
st.caption(
    "This dashboard is informational only and is not investment advice. "
    "Equity and P&L in paper mode are simulated. No strategy guarantees profit."
)

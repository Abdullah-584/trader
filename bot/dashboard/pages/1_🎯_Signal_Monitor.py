"""Signal Monitor page (Streamlit multipage) — analysis-only Quotex helper.

Shows UP/DOWN/SKIP suggestions from free feeds, scores every suggestion
after expiry, and compares the tracked hit rate against the binary-options
break-even math. Places no orders, automates nothing. A human decides.
"""

from __future__ import annotations

import sys
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
mcfg = dict(cfg.raw.get("quotex_monitor") or {})

st.set_page_config(page_title="Signal Monitor", page_icon="🎯", layout="wide")
st.title("🎯 Quotex signal monitor — analysis only")
st.error(
    "This page SUGGESTS up/down/skip and scores its own hit rate. It never "
    "places an order and never touches the broker UI — you decide every trade. "
    "Binary options are negative-expectation for most people: the break-even "
    "line below is the number to beat, sustained, before any of this means "
    "anything.", icon="⚠️")

payout = float(mcfg.get("payout_pct", 0.85))
from bot.monitor.quotex import breakeven_hit_rate  # noqa: E402
be = breakeven_hit_rate(payout)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Payout assumed", f"{payout:.0%}")
c2.metric("Break-even win rate", f"{be:.1%}")
c3.metric("Min confidence", f"{float(mcfg.get('min_confidence', 0.60)):.0%}")
c4.metric("Expiry", f"{int(mcfg.get('expiry_seconds', 60))}s")

tab_mon, tab_qx = st.tabs(["Signal monitor", "Quotex broker (unofficial)"])

# ---------------------------------------------------------------- monitor tab
with tab_mon:
    stats = db.monitor_stats()
    overall = stats.get("overall") or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Suggestions logged", overall.get("n", 0) or 0)
    m2.metric("Resolved", overall.get("resolved", 0) or 0)
    hit = overall.get("hit_rate")
    m3.metric("Tracked hit rate",
              f"{hit:.1%}" if hit is not None else "—",
              delta=(f"{(hit - be) * 100:+.1f} pts vs break-even" if hit is not None
                     else "needs resolved suggestions"),
              delta_color="normal" if hit is not None else "off")
    m4.metric("Pending resolution", overall.get("pending", 0) or 0)

    if hit is not None and overall.get("resolved", 0) < 30:
        st.warning("Fewer than 30 resolved suggestions — the hit rate is not "
                   "yet meaningful. Let the monitor run.", icon="⏳")
    if hit is not None and hit < be:
        st.error("Tracked hit rate is BELOW break-even: following these "
                 "suggestions with real stakes would be losing money. That "
                 "is a real, useful result — believe it.", icon="📉")

    st.subheader("Recent suggestions")
    rows = db.monitor_signals_recent(limit=100)
    if rows:
        df = pd.DataFrame(rows)
        df["direction"] = df["direction"].map({1: "UP", -1: "DOWN", 0: "SKIP"})
        cols = ["suggested_at", "quotex_name", "direction", "confidence",
                "price", "outcome", "exit_price", "contributors"]
        st.dataframe(df[[c for c in cols if c in df.columns]], use_container_width=True)
    else:
        st.info("No suggestions yet. Start the monitor: `python main.py monitor`")

    st.subheader("Per instrument")
    by_inst = stats.get("by_instrument") or []
    if by_inst:
        st.dataframe(pd.DataFrame(by_inst), use_container_width=True)
    else:
        st.caption("No data yet.")

    mc1, mc2 = st.columns([1, 4])
    if mc1.button("▶ Start monitor", use_container_width=True):
        import subprocess
        subprocess.Popen(
            [sys.executable, "main.py", "monitor"],
            cwd=str(cfg.root),
            stdout=open(cfg.root / "data/logs/monitor_stdout.log", "ab"),
            stderr=subprocess.STDOUT)
        mc1.success("monitor starting in background — suggestions appear here")

# ------------------------------------------------------------- quotex tab
with tab_qx:
    qcfg = dict(cfg.raw.get("quotex") or {})
    tcfg = dict(qcfg.get("trading") or {})
    st.caption(
        f"trading: **{'ENABLED' if tcfg.get('enabled') else 'disabled (analysis only)'}** · "
        f"account: **{tcfg.get('account', 'demo')}** · expiry: "
        f"{int(tcfg.get('expiry_seconds', 60))}s · stake capped by the binary "
        "risk manager (hard caps in code)")
    st.caption(
        "Unofficial reverse-engineered API (ChipaDevTeam/QuotexAPI) — may "
        "break without notice. Run `python main.py quotex-health` before "
        "enabling anything. Most retail binary-options traders lose money "
        "over time.")

    try:
        qx_stats = db.quotex_stats()
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Trades settled", qx_stats.get("resolved", 0) or 0)
        qhr = qx_stats.get("hit_rate")
        q2.metric("Hit rate", f"{qhr:.1%}" if qhr is not None else "—",
                  delta=(f"{(qhr - be) * 100:+.1f} pts vs break-even"
                         if qhr is not None else None),
                  delta_color="normal" if qhr is not None else "off")
        q3.metric("Total P&L", f"{qx_stats.get('profit_total') or 0:,.2f}")
        q4.metric("Kill switch", "armed" if "n/a" else "n/a", "see logs")
    except Exception:
        st.info("No quotex trades recorded yet (trading is disabled by default).")

    try:
        qrows = db.quotex_trades_recent(limit=50)
        if qrows:
            st.dataframe(pd.DataFrame(qrows), use_container_width=True)
    except Exception:
        pass

st.divider()
st.caption("Nothing here predicts the future. In paper mode all equity and "
           "P&L figures are simulated; in Quotex demo mode they are broker "
           "play-money. Neither guarantees anything about real markets.")

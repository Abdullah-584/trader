"""trader CLI — everything starts here.

Commands
--------
  python main.py download   # fetch & cache history for the whole watchlist
  python main.py backtest   # run the event-driven backtest, print metrics
  python main.py paper      # start the paper-trading loop (DEFAULT mode)
  python main.py live       # arm live trading (needs config opt-ins + typing
                            # the confirmation phrase; otherwise refuses)
  python main.py dashboard  # launch the local Streamlit UI
  python main.py train      # walk-forward train the ML classifier locally
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger


def _load(path: str):
    from bot.core.config import load_config
    cfg = load_config(path)
    from bot.core.logging_setup import setup_logging
    setup_logging(cfg)
    return cfg


# --------------------------------------------------------------------- download
def cmd_download(args) -> int:
    cfg = _load(args.config)
    from bot.data.cache import CandleCache
    from bot.data.provider import get_provider, resolve_provider_for_symbol

    cache = CandleCache(cfg.root / str(cfg.data.get("cache_dir", "data/cache")))
    timeframes = [str(t) for t in cfg.data.get("timeframes", ["1h"])]
    bars = int(cfg.data.get("history_bars", 1500))
    failures = 0

    for entry in cfg.data.get("watchlist", []):
        symbol = str(entry.get("symbol"))
        provider_name = resolve_provider_for_symbol(
            entry, cfg.data.get("provider", "ccxt"))
        try:
            provider = get_provider(provider_name, cfg)
        except Exception as exc:
            logger.error("provider {} unavailable: {}", provider_name, exc)
            failures += 1
            continue
        for tf in timeframes:
            try:
                df = provider.fetch_history(symbol, tf, bars=bars,
                                            retries=int(cfg.data.get("max_retries", 5)),
                                            backoff=float(cfg.data.get("retry_backoff", 1.8)),
                                            sleep_between=float(cfg.data.get("rate_limit_sleep", 0.25)))
                merged = cache.update(provider_name, symbol, tf, df)
                cov = cache.coverage(provider_name, symbol, tf, merged)
                logger.info("{:>10} {:>3} [{}]: {} bars, {} gaps, {} .. {}",
                            symbol, tf, provider_name, cov["bars"], cov["gaps"],
                            cov["first"], cov["last"])
            except Exception as exc:
                logger.error("download failed for {} {}: {}", symbol, tf, exc)
                failures += 1
    if failures:
        logger.warning("finished with {} failure(s)", failures)
        return 1
    logger.success("download complete")
    return 0


# --------------------------------------------------------------------- backtest
def cmd_backtest(args) -> int:
    cfg = _load(args.config)
    from bot.data.pipeline import load_history
    from bot.engine.backtester import EventDrivenBacktester
    from bot.risk.manager import RiskManager
    from bot.signals.aggregator import SignalAggregator
    from bot.storage.db import Database
    from bot.strategies.registry import build_strategies_from_config

    symbol = args.symbol or str(cfg.data.get("primary_symbol", "BTC/USDT"))
    timeframe = args.timeframe or str(cfg.data.get("primary_timeframe", "1h"))
    provider_name = args.provider or str(cfg.data.get("provider", "ccxt"))
    if "/" not in symbol and provider_name == "ccxt":
        provider_name = "yfinance"

    df = load_history(cfg, symbol, timeframe, provider_name=provider_name)
    logger.info("backtesting {} on {} {}: {} bars ({} .. {})",
                symbol, timeframe, provider_name, len(df), df.index[0], df.index[-1])

    strategies = build_strategies_from_config(cfg.strategies)
    if not strategies:
        logger.error("no enabled strategies in config.yaml")
        return 2
    weights = {k: float((v or {}).get("weight", 1.0)) for k, v in cfg.strategies.items()}
    aggregator = SignalAggregator(
        mode=str(cfg.signals.get("mode", "weighted_vote")),
        min_confidence=float(cfg.signals.get("min_confidence", 0.45)),
        weights=weights,
    )
    risk = RiskManager(cfg.risk, starting_equity=cfg.paper_starting_cash)
    engine = EventDrivenBacktester(
        strategies=strategies, aggregator=aggregator, risk_manager=risk,
        fees_pct=cfg.risk.fees_pct, slippage_pct=cfg.risk.slippage_pct,
        warmup_bars=int(cfg.backtest.get("warmup_bars", 200)),
    )
    result = engine.run(symbol, df)
    print(result.summary())

    # persist for the dashboard / later inspection
    import uuid
    run_id = f"bt-{uuid.uuid4().hex[:8]}"
    db = Database(cfg.root / str(cfg.storage.get("db_path", "data/trader.db")))
    for ts, row in result.equity_curve.items():
        db.save_equity_point(ts.to_pydatetime(), float(row), float(row), 0.0, 0, run_id)
    db.conn.commit()
    for t in result.trades:
        db.save_trade(t, run_id=run_id)
    out_csv = cfg.root / "data" / f"backtest_{run_id}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(out_csv, header=True)
    logger.info("run {} persisted ({} trades); equity curve -> {}",
                run_id, len(result.trades), out_csv.name)
    return 0


# --------------------------------------------------------------------- paper/live
def cmd_paper(args) -> int:
    cfg = _load(args.config)
    if cfg.mode == "live":
        logger.warning("config mode is 'live' but you invoked `paper` — "
                       "running PAPER trading anyway")
    from bot.core.safety import confirm_live_mode
    confirm_live_mode(cfg, interactive=False)  # logs that live is not armed
    from bot.runner import Runner
    runner = Runner(cfg, confirmed_live=False)
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        logger.info("interrupted by user — shutting down cleanly")
    return 0


def cmd_live(args) -> int:
    cfg = _load(args.config)
    from bot.core.safety import confirm_live_mode, is_live_mode

    if not is_live_mode(cfg):
        logger.error(
            "live mode is NOT armed: set both  mode: live  AND  "
            "live.enabled: true  in config.yaml first. Starting nothing.")
        return 2
    if not confirm_live_mode(cfg, interactive=True):
        logger.error("live confirmation failed — refusing to start. "
                     "Nothing was traded.")
        return 3
    from bot.runner import Runner
    runner = Runner(cfg, confirmed_live=True)
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        logger.info("interrupted by user — shutting down cleanly")
    return 0


# --------------------------------------------------------------------- dashboard
def cmd_dashboard(args) -> int:
    cfg = _load(args.config)
    import subprocess
    app_path = Path(__file__).parent / "bot" / "dashboard" / "app.py"
    port = str(cfg.dashboard.get("port", 8501))
    logger.info("starting Streamlit dashboard on http://localhost:{}", port)
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path),
             "--server.port", port, "--server.headless", "true",
             "--browser.gatherUsageStats", "false"],
            check=True,
        )
    except KeyboardInterrupt:
        logger.info("dashboard stopped")
    return 0


# --------------------------------------------------------------------- train
def cmd_train(args) -> int:
    cfg = _load(args.config)
    from bot.data.pipeline import load_history
    from bot.ml.train import train_classifier

    symbol = args.symbol or str(cfg.data.get("primary_symbol", "BTC/USDT"))
    timeframe = args.timeframe or str(cfg.data.get("primary_timeframe", "1h"))
    provider_name = args.provider or str(cfg.data.get("provider", "ccxt"))
    if "/" not in symbol and provider_name == "ccxt":
        provider_name = "yfinance"

    df = load_history(cfg, symbol, timeframe, provider_name=provider_name)
    logger.info("training ML classifier on {} {} ({} bars)", symbol, timeframe, len(df))
    result = train_classifier(
        df,
        horizon=int(cfg.ml.get("horizon", 8)) if cfg.ml.get("horizon") else 8,
        folds=int(cfg.ml.get("walk_forward_folds", 5)),
        embargo_bars=int(cfg.ml.get("embargo_bars", 24)),
        model_dir=str(cfg.root / str(cfg.ml.get("model_dir", "data/models"))),
        model_kind="hist_gb",
    )
    print("\n=== WALK-FORWARD TRAINING RESULT ===")
    print(f"labeled rows : {result.n_rows}")
    print(f"features     : {result.n_features}")
    print(f"oos accuracy : {result.oos_accuracy:.3f}")
    print(f"oos precision: {result.oos_precision:.3f}")
    print(f"model saved  : {result.model_path}")
    print("NOTE: out-of-sample accuracy barely above the base rate means NO edge.")
    return 0


# --------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trader",
        description="Local, paper-first algorithmic trading bot. "
                    "No strategy guarantees profit; live trading can lose all capital.",
    )
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    for name, help_ in [
        ("download", "download & cache history for the watchlist"),
        ("backtest", "run the event-driven backtest"),
        ("paper", "start the paper-trading loop (default, safe)"),
        ("live", "arm live trading (requires config + typed confirmation)"),
        ("dashboard", "launch the local Streamlit dashboard"),
        ("train", "walk-forward train the ML classifier locally"),
    ]:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--symbol", default=None, help="override symbol")
        sp.add_argument("--timeframe", default=None,
                        help="override timeframe (1m|5m|15m|1h|1d)")
        sp.add_argument("--provider", default=None,
                        help="override provider (ccxt|yfinance|csv)")
        sp.set_defaults(func={"download": cmd_download, "backtest": cmd_backtest,
                              "paper": cmd_paper, "live": cmd_live,
                              "dashboard": cmd_dashboard,
                              "train": cmd_train}[name])
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        logger.exception("fatal: {}", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

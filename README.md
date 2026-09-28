# trader — local, paper-first algorithmic trading bot

A 100% local and free algorithmic trading bot in the style of "Digimun":

- **Market data**: crypto via `ccxt` public endpoints (no keys needed), stocks/ETFs/forex via `yfinance`
- **Strategies**: plugin system — EMA/RSI trend, Bollinger mean reversion, ATR breakout, optional locally-trained ML classifier
- **Risk first**: per-trade risk caps, ATR stops, trailing stops, daily-loss kill switch, drawdown circuit breaker — none of which can be bypassed
- **Execution**: paper broker by default; live exchange trading behind two explicit opt-ins
- **UI**: local Streamlit dashboard at `localhost`
- **Storage**: SQLite + Parquet cache, everything on your machine

## ⚠️ Honest disclaimer — read this first

- **No strategy guarantees profit.** Most retail trading strategies lose money after fees and slippage.
- **Backtests overstate live performance.** They cannot fully model slippage, latency, partial fills, outages, or regime changes.
- **Live trading can lose all of your capital.** This bot defaults to **paper trading**. Live mode stays off until you explicitly enable it in `config.yaml` *and* re-type a confirmation sentence at startup.
- Nothing in this repository is financial advice. Paper trade for weeks, start small, and never risk money you cannot afford to lose.

## Build stages

| Stage | Contents | Status |
|------|----------|--------|
| 1 | Architecture, config system, domain models, safety gating | ✅ in progress |
| 2 | Data pipeline (ccxt/yfinance + cache) and event-driven backtester | ⏳ |
| 3 | Strategies + signal aggregator + risk manager | ⏳ |
| 4 | Paper broker + live runner + SQLite persistence/recovery | ⏳ |
| 5 | Streamlit dashboard | ⏳ |
| 6 | Tests, full docs, sample backtest output | ⏳ |

## Target project tree

```
trader/
├── main.py                  # CLI: backtest | paper | live | dashboard | download
├── config.yaml              # all settings (paper-first defaults)
├── .env.example             # template for optional secrets (never committed)
├── requirements.txt
├── README.md
├── bot/
│   ├── core/                # config, domain models, safety gates, logging
│   ├── data/                # provider abstraction, ccxt/yfinance, parquet cache
│   ├── strategies/          # strategy plugins + registry
│   ├── signals/             # weighted aggregator -> confidence score
│   ├── risk/                # position sizing, stops, kill switches
│   ├── brokers/             # Broker interface, PaperBroker, optional CcxtBroker
│   ├── engine/              # event-driven backtester, metrics, walk-forward
│   ├── runner/              # live/paper main loop with crash recovery
│   ├── storage/             # SQLite persistence (orders, trades, equity)
│   ├── alerts/              # optional desktop + Telegram notifications
│   ├── ml/                  # local model training (walk-forward, no lookahead)
│   └── dashboard/           # Streamlit app
├── tests/                   # pytest: risk manager, sizing, metrics
└── data/                    # git-ignored runtime data (cache, sqlite, logs)
```

Full step-by-step setup instructions for beginners arrive in Stage 6.

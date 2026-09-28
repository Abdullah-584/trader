# trader — local, paper-first algorithmic trading bot

A 100% local and free algorithmic trading bot in the style of "Digimun":

- **Market data**: crypto via `ccxt` public endpoints (no keys needed), stocks/ETFs/forex via `yfinance`
- **Strategies**: plugin system — EMA/RSI trend, Bollinger mean reversion, ATR breakout, optional locally-trained ML classifier
- **Risk first**: per-trade risk caps, ATR stops, trailing stops, daily-loss kill switch, drawdown circuit breaker — none of which can be bypassed
- **Execution**: paper broker by default; live exchange trading behind two explicit opt-ins plus a typed confirmation
- **UI**: local Streamlit dashboard at `localhost`
- **Storage**: SQLite + Parquet cache, everything on your machine

## ⚠️ Honest disclaimer — read this first

- **No strategy guarantees profit.** Most retail trading strategies lose money after fees and slippage.
- **Backtests overstate live performance.** They cannot fully model slippage, latency, partial fills, outages, or regime changes.
- **Live trading can lose all of your capital.** This bot defaults to **paper trading**. Live mode stays off until you explicitly enable it in `config.yaml` *and* re-type a confirmation sentence at startup.
- Nothing in this repository is financial advice. Paper trade for weeks, start small, and never risk money you cannot afford to lose.

## Quick start (60 seconds)

```bash
# 1. venv + deps (Windows shown; Linux/macOS: source .venv/bin/activate)
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. offline sample backtest — verifies the whole pipeline, no download needed
python main.py backtest --provider csv --symbol TEST --timeframe 1h

# 3. start paper trading (safe default) — in a second terminal run:
#    python main.py dashboard   ->  http://localhost:8501
python main.py paper
```

The repo ships `data/csv/TEST.csv` (offline synthetic 1h candles) so you can
reproduce the sample result below before downloading any market data.

## Verified status

- **32/32 pytest tests passing** — position sizing, hard risk caps, kill
  switch / circuit breaker, trailing-stop ratchet, metric math, aggregator
  voting, and anti-lookahead fill rules (next-bar-open entries, fees and
  slippage always charged, warmup respected).
- **Crash recovery verified** — positions, cash, and risk counters are
  restored from SQLite after a simulated restart.
- **Sample backtest reproduced** (offline synthetic data — plumbing proof,
  not an edge): +2.61% over 16 trades, PF 1.85, max DD −2.22%, with the
  "too few trades" warning firing as designed.
- **ML trainer honesty check** — on random-walk data it reports
  out-of-sample accuracy *below* the base rate and warns the model likely
  has no edge. Exactly what it should do.

## Features

1. **Data pipeline** — download, cache (Parquet), and update OHLCV for a watchlist and timeframes (1m, 5m, 15m, 1h, 1d); rate-limit sleeps, exponential-backoff retries, gap detection, forming-candle dropping.
2. **Strategy engine** — plugin interface (`Strategy.generate_signals(df) -> signal frame`) with EMA/RSI trend, Bollinger reversion, ATR breakout, and a walk-forward ML classifier.
3. **Signal aggregator** — weighted voting / majority / unanimous with a normalized confidence score.
4. **Risk management (mandatory)** — 1% max risk per trade, ATR stops, R-multiple take profit, trailing stop, max concurrent positions, daily-loss kill switch, max-drawdown circuit breaker, risk-based position sizing, simulated fees and slippage. Hard caps are clamped in code and cannot be bypassed from the YAML.
5. **Execution layer** — one `Broker` interface; `PaperBroker` (default) and optional `CcxtBroker` for live use share the same runner code path.
6. **Backtesting** — event-driven with next-bar-open fills, pessimistic intrabar stop/TP ordering, fees + slippage; metrics: total return, CAGR, Sharpe, Sortino, max drawdown, win rate, profit factor, expectancy; "too good to be true" warnings; no lookahead by construction.
7. **Streamlit dashboard** — prices snapshot, open positions, trade history, equity curve, signal log, start/stop controls, risk settings editor (clamped to the hard caps).
8. **Runner** — main loop that fetches candles, runs strategies, applies risk, executes, logs to SQLite, and recovers safely after crashes/restarts.
9. **CLI** — `python main.py backtest|paper|live|dashboard|download|train`.

## Project tree

```
trader/
├── main.py                  # CLI: backtest | paper | live | dashboard | download | train
├── config.yaml              # all settings (paper-first defaults)
├── .env.example             # template for optional secrets (never committed)
├── requirements.txt
├── README.md
├── bot/
│   ├── core/                # config (+ hard risk caps), models, safety gate, logging
│   ├── data/                # provider abstraction (ccxt/yfinance/csv), parquet cache, indicators
│   ├── strategies/          # strategy plugins + registry
│   ├── signals/             # weighted aggregator -> confidence score
│   ├── risk/                # position sizing, stops, kill switches
│   ├── brokers/             # Broker interface, PaperBroker, optional CcxtBroker
│   ├── engine/              # event-driven backtester, metrics
│   ├── runner/              # live/paper main loop with crash recovery
│   ├── storage/             # SQLite persistence (orders, trades, signals, equity, state)
│   ├── alerts/              # optional desktop (plyer) + Telegram notifications
│   ├── ml/                  # local walk-forward training (scikit-learn)
│   └── dashboard/           # Streamlit app
├── tests/                   # pytest: risk manager, sizing, metrics, backtester
└── data/                    # git-ignored runtime data (cache, db, logs, models, csv)
```

## Step-by-step setup (beginners)

These steps assume Windows; Linux/macOS differences are noted inline.

### 1. Install Python 3.11+

Check with `python --version`. On Windows, install from python.org and tick
"Add Python to PATH". macOS: `brew install python`. Ubuntu/Debian:
`sudo apt install python3.11 python3.11-venv`.

### 2. Get the code and create a virtual environment

```bash
cd path\to\trader
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies (all free)

```bash
pip install -r requirements.txt
```

### 4. (Optional) Configure secrets — not needed for paper mode

```bash
copy .env.example .env      # Windows
cp .env.example .env        # Linux/macOS
```

Leave everything empty to just paper trade. `.env` is git-ignored; never
commit it. To receive Telegram alerts later, create a bot with @BotFather,
put the token and your chat id in `.env`, and set `alerts.telegram: true`.

### 5. Download history for the default watchlist

```bash
python main.py download
```

This fetches BTC/USDT and ETH/USDT (Binance public endpoints) and SPY
(yfinance) at the configured timeframes into the Parquet cache under
`data/cache/`.

### 6. Run a backtest

```bash
python main.py backtest                                  # BTC/USDT, 1h, from cache
python main.py backtest --symbol ETH/USDT --timeframe 1d # anything else
```

Expected output format (numbers will differ; this run is on offline sample data):

```
==============================================================
BACKTEST RESULT (hypothesis, not a promise — see warnings)
==============================================================
start equity        :    10,000.00
end equity          :    10,261.10
total return        :        2.61%
CAGR                :       16.27%
Sharpe / Sortino    :   2.74 / 0.79
max drawdown        :       -2.22%
trades              :           16
win rate            :        68.8%
profit factor       :        1.85
expectancy/trade    :        16.32
--------------------------------------------------------------
WARNINGS:
  ! Only 16 trades — far too few to distinguish skill from luck (need >= 30).
==============================================================
```

Take the warnings seriously. Results this good on random-walk sample data
mostly demonstrate that the plumbing works — not that a real edge exists.

### 7. Start paper trading (the default and recommended mode)

```bash
python main.py paper
```

The loop fetches new candles each poll, acts once per closed bar, applies
risk rules, simulates fills with fees + slippage, and persists everything
to SQLite. Kill it any time with Ctrl+C — state is saved and the next start
resumes where it left off.

### 8. Open the dashboard

In a second terminal (same venv):

```bash
python main.py dashboard
```

Then open http://localhost:8501 — live equity curve, positions, trades,
signals, and start/stop controls.

### 9. (Optional) Train the local ML classifier

```bash
python main.py train
python main.py backtest     # then enable ml_classifier in config.yaml
```

The trainer uses walk-forward splits with an embargo gap (no leakage) and
warns explicitly when out-of-sample accuracy is barely better than chance.
It runs entirely on your machine with scikit-learn.

### 10. Live trading — deliberately hard to enable

1. Edit `config.yaml`: set `mode: live` AND `live.enabled: true`.
2. Put your exchange key/secret in `.env` (trading permission ONLY,
   withdrawals disabled, IP-whitelisted).
3. Run `python main.py live` and **type the confirmation phrase** exactly.
4. Anything less (one missing flag, wrong phrase, non-interactive shell)
   and the bot refuses or falls back to paper.

## Running the tests

```bash
pytest -q
```

Covers position sizing, risk caps, kill switch/circuit breaker, trailing
stops, metric math, aggregator voting, and anti-lookahead fill rules
(next-bar-open entries, fees/slippage charged, warmup respected).

## Quotex integration (unofficial — read before enabling)

The bot can connect to **Quotex** (market-qx.trade / qxbroker.com) through
[ChipaDevTeam/QuotexAPI](https://github.com/ChipaDevTeam/QuotexAPI) as an
optional data source and — only if you explicitly arm it — a binary-options
execution path.

### Read this first — honest risk statement

- **Binary options carry very high risk.** The format is structurally closer
  to gambling than investing: fixed expiry, all-or-nothing settlement. At an
  85% payout you must sustain a **54.1% win rate just to break even** (52.1%
  at 92%). **Most retail binary-options traders lose money over time.**
- **Quotex is an offshore, unregulated counterparty.** Settlement, withdrawal,
  and account-freezing risk sit entirely with you.
- **The API is unofficial and reverse-engineered.** It is not affiliated with
  Quotex, violates the platform's terms of service if used for trading, and
  **may break without notice** — upstream itself documents that some endpoints
  still return mock/WIP data. Every call in this integration is retry-wrapped
  and error-logged, but that cannot make an unofficial API reliable.
- **No strategy predicts 1-minute direction reliably.** The monitor's job is
  to make its own hit rate visible — if it says SKIP most of the time, that
  is the system working.

### What was added

| Component | File | Purpose |
|-----------|------|---------|
| Data provider | `bot/quotex/provider.py` | Same `DataProvider` interface as ccxt/yfinance; SSID auth, retry/backoff, payload normalization for WIP upstream formats |
| Binary broker | `bot/quotex/broker.py` | `buy(asset, amount, direction, expiry)` CALL/PUT trades — no stops exist in binaries, so it does not pretend to fit the spot `Broker` interface |
| Binary risk | `bot/risk/binary.py` | Max % of balance per trade, max concurrent trades, daily-loss kill switch, **minimum payout gate** — all hard-capped in code, non-bypassable |
| Safety gate | `bot/quotex/safety.py` | REAL account requires a typed confirmation phrase |
| Health check | `bot/quotex/health.py` | Verifies credentials → library → session → assets → payouts → **real candle flow** before anything trades |
| Runner | `bot/quotex/runner.py` | Continuous analysis loop; trades only when explicitly enabled |

### Setup

```bash
# 1. install the unofficial library (deliberately not in requirements.txt)
pip install git+https://github.com/ChipaDevTeam/QuotexAPI.git

# 2. credentials in .env (git-ignored) — SSID preferred
#    Quotex web -> F12 -> Application -> Cookies -> copy the `ssid` value
QUOTEX_SSID=<paste the cookie value; full 42["authorization",{...}] form also works>
QUOTEX_EMAIL=          # fallback login; SSID is more reliable
QUOTEX_PASSWORD=

# 3. ALWAYS verify the session + feed first
python main.py quotex-health

# 4a. analysis-only monitoring (default — places no orders)
python main.py quotex

# 4b. DEMO-account trading: set quotex.trading.enabled: true in config.yaml
# 4c. REAL-account trading: ALSO set account: real — then type the
#     confirmation phrase at startup. Two config keys + typed phrase.
```

### Safety defaults

* `AccountType.DEMO` is the default and is re-asserted on every connect.
* `trading.enabled: false` (default) = suggestions and health checks only;
  no order path is ever constructed.
* REAL mode refuses to start if the health check fails, if the balance can't
  be read, or if the typed confirmation doesn't match.
* SSIDs expire after hours — refresh from your browser; the health check
  detects it.

## Quotex signal monitor (analysis only, no orders)

`python main.py monitor` runs a continuous monitor that maps Quotex-style
instrument names to price feeds — crypto via Binance, FX via yfinance, and
(optionally, with the SSID installed) Quotex's own feed for OTC pairs, which
exist nowhere else — runs the strategy engine on each closed 1-minute bar,
and logs **UP / DOWN / SKIP** suggestions with confidence to SQLite. Every
suggestion is scored after expiry (**WIN / LOSS / TIE**) and the dashboard
shows the real hit rate next to the break-even math for your payout. It
never places an order and never touches the Quotex UI — a human decides
everything.

## Configuration quick reference

Everything lives in `config.yaml`: watchlist + timeframes, strategy
params/weights, aggregation mode + `min_confidence`, risk rules, paper
starting cash, alert toggles, dashboard port. Risk values above the hard
caps are clamped with a log warning on every load — for example
`risk_per_trade` can never exceed 2%.

## Next steps — before you risk a single real dollar

1. **Paper trade for at least 4 weeks, ideally 8.** Run `python main.py paper` continuously and review the dashboard daily. Compare realized results against the backtest for the same period: if paper P&L is much worse (it usually is), trust the paper number.
2. **Check the honest metrics, not the dream.** You want >= 30 trades in the backtest, a profit factor you'd still accept halved, a max drawdown you can emotionally and financially survive at your chosen size, and Sharpe below ~2 (higher usually means a bug or overfitting).
3. **Stress the assumptions.** Raise fees and slippage in `config.yaml` (e.g. double them) and re-run the backtest. If the edge disappears with realistic costs, it was never an edge.
4. **Walk-forward, not curve-fit.** If you tune strategy parameters, tune them on one slice of history and verify on a later slice the model never saw. Changing parameters until the backtest looks great is overfitting, and the out-of-sample result will disappoint you.
5. **Start live at minimum size.** If — and only if — weeks of paper trading stayed consistent with backtests, enable live mode with the smallest position the exchange allows and treat the first month as tuition. Most people should stop here and stay on paper.
6. **Keep the kill switches armed.** `max_daily_loss_pct` and `max_drawdown_pct` exist to save you from yourself. Do not raise them after a bad day.

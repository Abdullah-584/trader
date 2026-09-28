"""Configuration loading and validation.

Design rules
------------
* Paper-first: the config defaults to ``mode: paper`` and live trading needs
  two independent opt-ins (``mode: live`` AND ``live.enabled: true``) plus an
  interactive confirmation typed at startup (see ``bot/core/safety.py``).
* Risk settings have HARD CAPS enforced here. A user can lower the risk in
  ``config.yaml`` but can never raise it above the caps in ``HARD_CAPS``.
  Risk management cannot be bypassed by editing the YAML.
* Secrets are never read from this file. Optional keys come from the
  environment (a git-ignored ``.env`` file, loaded once in ``load_config``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

# --------------------------------------------------------------------------
# Hard risk caps — enforced at load time, immune to config edits.
# These are deliberately conservative: the bot refuses to start (or clamps)
# when a user tries to configure more risk than this.
# --------------------------------------------------------------------------
HARD_CAPS = {
    "risk_per_trade": 0.02,        # max 2% of equity per trade
    "max_daily_loss_pct": 0.10,    # max 10% daily loss kill switch
    "max_drawdown_pct": 0.50,      # max 50% drawdown circuit breaker
    "max_positions": 10,
    "max_position_pct": 1.00,
    "stop_atr_multiple_min": 0.5,   # stop must be >= 0.5 * ATR (never zero)
    "stop_atr_multiple_max": 10.0,  # ...but no wider than 10 * ATR
}

VALID_MODES = {"paper", "live"}
VALID_TIMEFRAMES = {"1m", "5m", "15m", "1h", "1d"}
VALID_PROVIDERS = {"ccxt", "yfinance", "auto", "csv"}


class ConfigError(ValueError):
    """Raised when config.yaml is missing, malformed, or unsafe."""


# --------------------------------------------------------------------------
# Typed accessors — light-weight: we keep the raw dict (easy to extend) but
# expose well-typed sub-views for the hot paths.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float
    max_positions: int
    max_daily_loss_pct: float
    max_drawdown_pct: float
    stop_atr_multiple: float
    take_profit_r_multiple: float
    trailing_stop: bool
    trailing_atr_multiple: float
    min_stop_distance_pct: float
    max_position_pct: float
    fees_pct: float
    slippage_pct: float


@dataclass(frozen=True)
class LiveConfig:
    enabled: bool
    exchange: str
    confirm_phrase: str
    default_order_type: str


@dataclass(frozen=True)
class Config:
    """Immutable view over config.yaml plus resolved paths and env flags."""

    raw: dict[str, Any]
    root: Path

    # -- mode / live gating -------------------------------------------------
    @property
    def mode(self) -> str:
        mode = str(self.raw.get("mode", "paper")).lower()
        if mode not in VALID_MODES:
            raise ConfigError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
        return mode

    @property
    def live(self) -> LiveConfig:
        raw_live = self.raw.get("live") or {}
        return LiveConfig(
            enabled=bool(raw_live.get("enabled", False)),
            exchange=str(raw_live.get("exchange", "binance")),
            confirm_phrase=str(raw_live.get("confirm_phrase", "")),
            default_order_type=str(raw_live.get("default_order_type", "market")),
        )

    @property
    def live_armable(self) -> bool:
        """True only when BOTH independent opt-ins are set."""
        return self.mode == "live" and self.live.enabled

    # -- sections ------------------------------------------------------------
    @property
    def risk(self) -> RiskConfig:
        r = dict(self.raw.get("risk") or {})
        cfg = RiskConfig(
            risk_per_trade=_clamp_float(r, "risk_per_trade", 0.01, HARD_CAPS["risk_per_trade"]),
            max_positions=_clamp_int(r, "max_positions", 3, HARD_CAPS["max_positions"]),
            max_daily_loss_pct=_clamp_float(r, "max_daily_loss_pct", 0.03, HARD_CAPS["max_daily_loss_pct"]),
            max_drawdown_pct=_clamp_float(r, "max_drawdown_pct", 0.15, HARD_CAPS["max_drawdown_pct"]),
            stop_atr_multiple=_clamp_float(
                r, "stop_atr_multiple", 2.0,
                HARD_CAPS["stop_atr_multiple_max"],
                minimum=HARD_CAPS["stop_atr_multiple_min"]),
            take_profit_r_multiple=max(0.5, float(r.get("take_profit_r_multiple", 2.0))),
            trailing_stop=bool(r.get("trailing_stop", True)),
            trailing_atr_multiple=max(0.5, float(r.get("trailing_atr_multiple", 3.0))),
            min_stop_distance_pct=max(0.0, float(r.get("min_stop_distance_pct", 0.002))),
            max_position_pct=_clamp_float(r, "max_position_pct", 0.25, HARD_CAPS["max_position_pct"]),
            fees_pct=max(0.0, float(r.get("fees_pct", 0.001))),
            slippage_pct=max(0.0, float(r.get("slippage_pct", 0.0005))),
        )
        return cfg

    @property
    def paper_starting_cash(self) -> float:
        return float((self.raw.get("paper") or {}).get("starting_cash", 10_000.0))

    @property
    def base_currency(self) -> str:
        return str((self.raw.get("paper") or {}).get("base_currency", "USDT"))

    @property
    def data(self) -> dict[str, Any]:
        return dict(self.raw.get("data") or {})

    @property
    def strategies(self) -> dict[str, Any]:
        return dict(self.raw.get("strategies") or {})

    @property
    def signals(self) -> dict[str, Any]:
        return dict(self.raw.get("signals") or {})

    @property
    def runner(self) -> dict[str, Any]:
        return dict(self.raw.get("runner") or {})

    @property
    def backtest(self) -> dict[str, Any]:
        return dict(self.raw.get("backtest") or {})

    @property
    def ml(self) -> dict[str, Any]:
        return dict(self.raw.get("ml") or {})

    @property
    def alerts(self) -> dict[str, Any]:
        return dict(self.raw.get("alerts") or {})

    @property
    def logging(self) -> dict[str, Any]:
        return dict(self.raw.get("logging") or {})

    @property
    def dashboard(self) -> dict[str, Any]:
        return dict(self.raw.get("dashboard") or {})

    @property
    def storage(self) -> dict[str, Any]:
        return dict(self.raw.get("storage") or {})

    # -- paths (created lazily, git-ignored under data/) ---------------------
    def ensure_dirs(self) -> None:
        for key in ("cache_dir",):
            d = self.data.get(key)
            if d:
                (self.root / d).mkdir(parents=True, exist_ok=True)
        db = self.storage.get("db_path")
        if db:
            Path(db).parent.mkdir(parents=True, exist_ok=True)
        log_dir = self.logging.get("dir")
        if log_dir:
            (self.root / log_dir).mkdir(parents=True, exist_ok=True)
        model_dir = self.ml.get("model_dir")
        if model_dir:
            (self.root / model_dir).mkdir(parents=True, exist_ok=True)

    # -- secrets (environment only, never the YAML) ---------------------------
    def telegram_credentials(self) -> tuple[str, str]:
        return os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", "")

    def exchange_keys(self, exchange: str) -> tuple[str, str]:
        """Optional live-trading keys from the environment. Never for data."""
        prefix = exchange.upper().replace("-", "_")
        return os.getenv(f"{prefix}_API_KEY", ""), os.getenv(f"{prefix}_SECRET", "")


# --------------------------------------------------------------------------
# Loading + validation
# --------------------------------------------------------------------------
def load_config(path: str | Path = "config.yaml") -> Config:
    """Load, validate, and clamp config.yaml. Raises ConfigError when unsafe.

    A local ``.env`` file (git-ignored) is loaded for optional secrets; it is
    never required — paper mode runs with no secrets at all.
    """
    load_dotenv()  # no-op when .env absent

    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p.resolve()}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # malformed yaml
        raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

    cfg = Config(raw=raw, root=Path.cwd())

    _validate(cfg)
    cfg.ensure_dirs()
    return cfg


def _validate(cfg: Config) -> None:
    # mode + live gating (both must be explicit for live)
    if cfg.mode == "live" and not cfg.live.enabled:
        raise ConfigError(
            "mode is 'live' but live.enabled is false. "
            "Live trading requires BOTH 'mode: live' AND 'live.enabled: true'."
        )
    if cfg.mode == "live" and not cfg.live.confirm_phrase:
        raise ConfigError("live.confirm_phrase must be set for live mode.")
    if cfg.mode == "live":
        logger.warning("LIVE TRADING MODE ARMED in config — startup confirmation still required.")

    # data section
    data = cfg.data
    provider = str(data.get("provider", "ccxt"))
    if provider not in VALID_PROVIDERS:
        raise ConfigError(f"data.provider must be one of {sorted(VALID_PROVIDERS)}, got {provider!r}")

    timeframes = data.get("timeframes") or []
    if not timeframes:
        raise ConfigError("data.timeframes must list at least one timeframe.")
    for tf in timeframes:
        if str(tf) not in VALID_TIMEFRAMES:
            raise ConfigError(f"timeframe {tf!r} not supported. Allowed: {sorted(VALID_TIMEFRAMES)}")

    watchlist = data.get("watchlist") or []
    if not watchlist:
        raise ConfigError("data.watchlist must contain at least one symbol.")
    for entry in watchlist:
        if not isinstance(entry, dict) or not entry.get("symbol"):
            raise ConfigError(f"watchlist entries must be mappings with a 'symbol': {entry!r}")

    history = int(data.get("history_bars", 1500))
    if history < 300:
        raise ConfigError("data.history_bars must be >= 300 so indicators have warmup data.")

    # strategies section must map name -> mapping
    for name, spec in cfg.strategies.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"strategies.{name} must be a mapping.")
        weight = float(spec.get("weight", 1.0))
        if weight < 0:
            raise ConfigError(f"strategies.{name}.weight must be >= 0.")

    # paper cash
    if cfg.paper_starting_cash <= 0:
        raise ConfigError("paper.starting_cash must be positive.")

    # backtest window sanity
    bt = cfg.backtest
    if bt.get("start") and bt.get("end") and str(bt["start"]) >= str(bt["end"]):
        raise ConfigError("backtest.start must be earlier than backtest.end.")


def _clamp_float(section: dict[str, Any], key: str, default: float, cap: float, minimum: float = 0.0) -> float:
    """Read a float, clamp it into [minimum, cap]. Warn when clamped."""
    try:
        val = float(section.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"risk.{key} must be a number.") from exc
    clamped = min(max(val, minimum), cap)
    if clamped != val:
        logger.warning("risk.{} = {} exceeds allowed range; clamped to {}", key, val, clamped)
    return clamped


def _clamp_int(section: dict[str, Any], key: str, default: int, cap: int) -> int:
    try:
        val = int(section.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"risk.{key} must be an integer.") from exc
    clamped = min(max(val, 0), cap)
    if clamped != val:
        logger.warning("risk.{} = {} exceeds allowed range; clamped to {}", key, val, clamped)
    return clamped

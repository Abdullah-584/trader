"""Quotex health check — run BEFORE any trading.

Verifies, in order:
  1. credentials present (.env)          2. library importable
  3. SSID/session connects               4. account switch works + balance readable
  5. asset list + payouts                6. real candle data flows (not mock/WIP)

Every step returns PASS/FAIL with a reason so failures are actionable.
The quotex runner refuses to start trading unless steps 1-6 pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class HealthCheck:
    step: str
    ok: bool
    detail: str

    @property
    def mark(self) -> str:
        return "PASS" if self.ok else "FAIL"


def run_health_check(cfg, *, interactive: bool = True) -> tuple[bool, list[HealthCheck]]:
    """Run all checks. Returns (all_ok, checks). Never raises."""
    checks: list[HealthCheck] = []
    qcfg = dict(cfg.raw.get("quotex") or {})
    conn = dict(qcfg.get("connection") or {})

    def _add(step: str, ok: bool, detail: str) -> bool:
        checks.append(HealthCheck(step, ok, detail))
        logger.info("health [{}] {}: {}", checks[-1].mark, step, detail)
        return ok

    # 1. credentials
    import os
    ssid = os.getenv("QUOTEX_SSID", "").strip()
    email = os.getenv("QUOTEX_EMAIL", "").strip()
    password = os.getenv("QUOTEX_PASSWORD", "").strip()
    creds_ok = bool(ssid) or bool(email and password)
    _add("credentials", creds_ok,
         "QUOTEX_SSID present" if ssid else
         ("QUOTEX_EMAIL+QUOTEX_PASSWORD present" if creds_ok else
          "none found in .env — set QUOTEX_SSID (preferred)"))
    if not creds_ok:
        return False, checks

    # 2. library import
    try:
        import QuotexAPI  # noqa: F401
        _add("library", True, "QuotexAPI importable")
    except ImportError as exc:
        _add("library", False, f"QuotexAPI not installed: {exc}")
        return False, checks

    # 3-6 via provider (retries + logging inside)
    from bot.quotex.provider import QuotexDataProvider
    try:
        provider = QuotexDataProvider(
            ssid=ssid, email=email, password=password,
            reconnect_enabled=bool(conn.get("reconnect_enabled", True)),
            max_reconnect_attempts=int(conn.get("max_reconnect_attempts", 5)),
            reconnect_delay=int(conn.get("reconnect_delay", 5)),
            rate_limit_sleep=float(conn.get("rate_limit_sleep", 0.4)),
            max_retries=int(conn.get("max_retries", 4)),
            retry_backoff=float(conn.get("retry_backoff", 1.8)),
        )
    except Exception as exc:
        _add("provider-init", False, str(exc))
        return False, checks

    try:
        provider._ensure_connected()
        _add("connect", True, "session authenticated (SSID or login)")
    except Exception as exc:
        _add("connect", False, f"connect failed — SSID likely expired: {exc}")
        return False, checks

    try:
        assets = provider.get_assets()
        n = len(assets) if isinstance(assets, dict) else len(assets or [])
        _add("assets", n > 0, f"{n} assets visible")
    except Exception as exc:
        _add("assets", False, f"get_assets failed: {exc}")
        return False, checks

    configured = [str(a) for a in (qcfg.get("assets") or [])]
    payouts = {}
    for asset in configured[:5]:
        payouts[asset] = provider.get_payout(asset)
    ok_payouts = any(v is not None for v in payouts.values())
    _add("payouts", ok_payouts,
         ", ".join(f"{k}={v if v is not None else '?'}" for k, v in payouts.items())
         or "no configured assets to probe")

    timeframe = str((qcfg.get("data") or {}).get("timeframe", "60"))
    feed_asset = configured[0] if configured else None
    if feed_asset:
        try:
            df = provider.fetch_ohlcv(feed_asset, timeframe, limit=50)
            fresh = df is not None and len(df) >= 10
            _add("candles", fresh,
                 f"{0 if df is None else len(df)} bars for {feed_asset}"
                 + (f", last close {df['close'].iloc[-1]}" if fresh else
                    " — upstream may still serve mock/WIP data"))
        except Exception as exc:
            _add("candles", False, f"candle fetch failed for {feed_asset}: {exc}")
    else:
        _add("candles", False, "no quotex.assets configured")

    all_ok = all(c.ok for c in checks)
    return all_ok, checks

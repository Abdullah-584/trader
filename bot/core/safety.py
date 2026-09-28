"""Live-trading safety gate.

Live mode requires BOTH independent opt-ins in config.yaml (``mode: live``
and ``live.enabled: true``) AND an interactive confirmation typed at
startup. Nothing here can be satisfied by a flag alone, and paper mode is
always available as the default.

The confirmation refuses to proceed on empty input, EOF, or interrupt.
"""

from __future__ import annotations

import sys

from loguru import logger

from bot.core.config import Config

_LIVE_BANNER = r"""
======================================================================
                        !!  LIVE TRADING MODE  !!

  * No strategy guarantees profit.
  * Backtests overstate live results; slippage, latency, outages and
    regime changes are NOT fully modeled.
  * Real money is at risk. You can lose ALL of your capital.
  * Use an exchange key with withdrawals DISABLED and IP restrictions.

  Paper mode remains available by setting  mode: paper  in config.yaml.
======================================================================
"""


def is_live_mode(cfg: Config) -> bool:
    """True only when both independent config opt-ins are set."""
    return cfg.live_armable


def confirm_live_mode(cfg: Config, interactive: bool = True) -> bool:
    """Require the operator to type the exact confirmation phrase.

    Returns True only when the phrase matches. Non-interactive contexts
    (tests, CI, headless runs) must pass interactive=False and will never
    be able to arm live mode.
    """
    if not is_live_mode(cfg):
        logger.info("Live trading is NOT armed (mode: {} / live.enabled: {}).",
                    cfg.mode, cfg.live.enabled)
        return False

    print(_LIVE_BANNER, file=sys.stderr)
    if not interactive:
        logger.error("Refusing to start live mode in a non-interactive session.")
        return False

    try:
        answer = input(f'Type exactly "{cfg.live.confirm_phrase}" to continue: ')
    except (EOFError, KeyboardInterrupt):
        print("\nLive mode confirmation aborted.", file=sys.stderr)
        return False

    if answer.strip() == cfg.live.confirm_phrase:
        logger.warning("Live mode CONFIRMED by operator. Trade responsibly.")
        return True
    print("Confirmation phrase did not match — falling back to PAPER mode.", file=sys.stderr)
    return False


def assert_paper_or_confirmed(cfg: Config, confirmed: bool) -> None:
    """Final gate used by the runner before touching a live broker."""
    if cfg.mode == "live" and not confirmed:
        raise PermissionError(
            "Live mode is configured but was not confirmed. "
            "Set mode: paper or complete the startup confirmation."
        )

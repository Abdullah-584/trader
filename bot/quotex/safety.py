"""Quotex REAL-account gate — same two-key + typed-confirmation pattern as
spot live trading. DEMO is always the default; this module is the only
place a REAL session may be armed."""

from __future__ import annotations

import sys

from loguru import logger

_QUOTEX_BANNER = r"""
======================================================================
                  !!  QUOTEX REAL ACCOUNT MODE  !!

  * Binary options are a negative-expectation format for most people:
    at 85% payout you must sustain ~54% winners JUST to break even.
  * The format is structurally closer to gambling than investing.
  * This is an UNOFFICIAL, reverse-engineered API. It may break without
    notice; the upstream library itself documents mock/WIP endpoints.
  * Offshore unregulated counterparty: withdrawal and settlement risk
    sits with you.
  * DEMO remains available at any time (account: demo in config.yaml).
======================================================================
"""


def confirm_real_account(confirm_phrase: str, interactive: bool = True) -> bool:
    """Return True only when the operator types the exact phrase."""
    print(_QUOTEX_BANNER, file=sys.stderr)
    if not interactive:
        logger.error("Refusing to enable REAL account in a non-interactive session.")
        return False
    try:
        answer = input(f'Type exactly "{confirm_phrase}" to continue: ')
    except (EOFError, KeyboardInterrupt):
        print("\nREAL-account confirmation aborted.", file=sys.stderr)
        return False
    if answer.strip() == confirm_phrase:
        logger.warning("Quotex REAL account CONFIRMED by operator.")
        return True
    print("Confirmation did not match — staying on DEMO.", file=sys.stderr)
    return False

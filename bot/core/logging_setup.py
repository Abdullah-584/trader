"""Logging setup built on loguru.

* Console: level from config.yaml (default INFO).
* File:    always DEBUG, rotated daily with retention, under data/logs.
* Noisy third-party loggers are silenced to WARNING.
"""

from __future__ import annotations

import sys

from loguru import logger

from bot.core.config import Config

_QUIET_LIBS = ("urllib3", "ccxt", "asyncio", "peewee", "streamlit", "yfinance")


def setup_logging(cfg: Config) -> None:
    """Configure loguru sinks. Safe to call once at process start."""
    logger.remove()  # drop default handler

    console_level = str(cfg.logging.get("level", "INFO")).upper()
    logger.add(
        sys.stderr,
        level=console_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <7}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> - {message}",
        backtrace=False,
        diagnose=False,
    )

    log_dir = cfg.root / str(cfg.logging.get("dir", "data/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "trader_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="14 days",
        encoding="utf-8",
        backtrace=True,
        diagnose=False,
    )

    import logging

    for name in _QUIET_LIBS:
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.debug("Logging configured: console={} file={}", console_level, log_dir)

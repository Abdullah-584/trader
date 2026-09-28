"""Optional notifications: local desktop (plyer) and Telegram (stdlib HTTP).

Both channels are OFF by default in config.yaml. Telegram needs
TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in the git-ignored .env. Notification
failures are logged and swallowed — an alert outage must never take down
the trading loop. No third-party Telegram SDK: a plain stdlib POST is used.
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request

from loguru import logger


class Notifier:
    def __init__(self, alerts_cfg: dict, root=None):
        self.desktop_enabled = bool(alerts_cfg.get("desktop", False))
        self.telegram_enabled = bool(alerts_cfg.get("telegram", False))
        self._root = root

    # ------------------------------------------------------------------ api
    def notify(self, title: str, message: str) -> None:
        """Send via every enabled channel; never raises."""
        if self.desktop_enabled:
            self._desktop(title, message)
        if self.telegram_enabled:
            self._telegram(f"{title}\n{message}")

    # ------------------------------------------------------------------ desktop
    def _desktop(self, title: str, message: str) -> None:
        try:
            from plyer import notification
            notification.notify(
                title=title[:64], message=message[:240], timeout=8,
                app_name="trader")
        except Exception as exc:  # plyer is optional and platform-quirky
            logger.debug("desktop notification failed: {}", exc)

    # ------------------------------------------------------------------ telegram
    def _telegram(self, text: str) -> None:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            logger.warning(
                "alerts.telegram=true but TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
                "missing in .env — Telegram alert skipped")
            return
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = urllib.parse.urlencode(
                {"chat_id": chat_id, "text": text[:4000]}).encode()
            req = urllib.request.Request(url, data=payload, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        except Exception as exc:
            logger.debug("telegram notification failed: {}", exc)

"""
Telegram alerter — non-blocking, queue-based alert delivery.

Alerts are queued and sent by a background daemon thread, so the
critical trading path is never blocked by network I/O.

Priority levels:
  'critical'  — order fills, crash recovery, loss limiter triggers (retry once)
  'normal'    — signal fires, regime changes, daily reports
  'low'       — shadow tracking, FII patterns (dropped if queue full)

Setup: Create bot via @BotFather, get token and chat_id.
"""

import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_FALLBACK_LOG = Path(__file__).parent / "telegram_fallback.log"


class TelegramAlerter:

    def __init__(self, token: str = None, chat_id: str = None,
                 max_queue: int = 200):
        import os
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        token = self.token
        self.base_url = f"https://api.telegram.org/bot{token}" if token else ""
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._consecutive_failures = 0
        self._last_failure_time = 0.0

        # Start background sender
        self._worker = threading.Thread(
            target=self._send_loop, name="telegram-sender", daemon=True
        )
        self._worker.start()

    # ── Public API (non-blocking) ────────────────────────────────

    def send(self, level: str, message: str,
             signal_id: str = None, priority: str = "normal"):
        """
        Queue an alert for background delivery. Returns immediately.

        Args:
            level: INFO, WARNING, CRITICAL, EMERGENCY
            message: alert text
            signal_id: optional signal reference
            priority: 'critical', 'normal', or 'low'
        """
        emoji = {
            'INFO': '\u2139\ufe0f',
            'WARNING': '\u26a0\ufe0f',
            'CRITICAL': '\U0001f534',
            'EMERGENCY': '\U0001f6a8',
        }.get(level, '\u2753')

        text = f"{emoji} *{level}*\n{message}"
        if signal_id:
            text += f"\nSignal: `{signal_id}`"
        text += f"\n_{datetime.now().strftime('%H:%M:%S IST')}_"

        try:
            self._queue.put_nowait((text, priority))
        except queue.Full:
            if priority == "critical":
                # For critical: evict oldest low-priority message
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait((text, priority))
                except queue.Empty:
                    pass
            else:
                logger.warning("Telegram queue full — dropping %s message", level)

    # ── Background worker ────────────────────────────────────────

    def _send_loop(self):
        """Drain queue and deliver messages. Runs forever as daemon."""
        while True:
            try:
                text, priority = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            success = self._do_send(text, timeout=5)

            if not success:
                self._consecutive_failures += 1
                self._last_failure_time = time.time()

                # Retry once for critical
                if priority == "critical":
                    time.sleep(1)
                    success = self._do_send(text, timeout=10)

                # Fallback to file if still failing
                if not success:
                    self._log_to_file(text)

                # If down for > 5 min, batch remaining and log
                if self._consecutive_failures > 60:
                    self._drain_to_file()
            else:
                if self._consecutive_failures > 0:
                    # Recovered — send summary of missed alerts
                    missed = self._consecutive_failures
                    self._consecutive_failures = 0
                    if missed > 5:
                        self._do_send(
                            f"\u26a0\ufe0f Telegram recovered. "
                            f"Missed ~{missed} alerts (see fallback log).",
                            timeout=5,
                        )
                self._consecutive_failures = 0

    def _do_send(self, text: str, timeout: int = 5) -> bool:
        """Attempt to send one message. Returns True on success."""
        if not self.base_url:
            return True  # No token configured — silent success

        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                },
                timeout=timeout,
            )
            return resp.ok
        except Exception as e:
            logger.debug("Telegram send error: %s", e)
            return False

    def _log_to_file(self, text: str):
        """Fallback: append to local file when Telegram is unreachable."""
        try:
            with open(_FALLBACK_LOG, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] {text}\n")
        except Exception:
            pass

    def _drain_to_file(self):
        """Drain entire queue to file (Telegram extended outage)."""
        drained = 0
        while not self._queue.empty():
            try:
                text, _ = self._queue.get_nowait()
                self._log_to_file(text)
                drained += 1
            except queue.Empty:
                break
        if drained:
            logger.warning("Drained %d alerts to fallback log", drained)

"""
Telegram alerter — sends alerts to your personal Telegram chat.
Uses python-telegram-bot library.
Setup: Create bot via @BotFather, get token and chat_id.
"""

from datetime import datetime
import requests


class TelegramAlerter:

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    def send(self, level: str, message: str,
             signal_id: str = None):
        """Send alert. Blocks until delivered."""
        emoji = {
            'INFO':      '\u2139\ufe0f',
            'WARNING':   '\u26a0\ufe0f',
            'CRITICAL':  '\U0001f534',
            'EMERGENCY': '\U0001f6a8'
        }.get(level, '\u2753')

        text = f"{emoji} *{level}*\n{message}"
        if signal_id:
            text += f"\nSignal: `{signal_id}`"
        text += f"\n_{datetime.now().strftime('%H:%M:%S IST')}_"

        requests.post(
            f"{self.base_url}/sendMessage",
            json={
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'Markdown'
            },
            timeout=10
        )

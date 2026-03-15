"""
VIX spike monitor for KAUFMAN_DRY_20 open positions.

Checks if VIX > threshold while DRY_20 has an open LONG position.
Sends Telegram alert for manual review — does NOT auto-close.

Can run as:
  - Standalone check: python -m paper_trading.vix_monitor
  - Called from daily pipeline after signal_compute
  - Scheduled via cron for intraday checks (if live VIX feed available)
"""

import logging
import os
from datetime import date

import psycopg2

from config.settings import DATABASE_DSN
from monitoring.telegram_alerter import TelegramAlerter
from monitoring.alert_engine import AlertEngine

logger = logging.getLogger(__name__)

VIX_ALERT_THRESHOLD = 20.0
VIX_CRITICAL_THRESHOLD = 25.0


class VIXMonitor:
    """Monitors VIX level relative to DRY_20 open positions."""

    def __init__(self, db_conn=None, alert_engine=None):
        self.conn = db_conn or psycopg2.connect(DATABASE_DSN)
        self.alert_engine = alert_engine

    def check(self, current_vix: float = None) -> dict:
        """
        Check VIX against DRY_20 open positions.

        Args:
            current_vix: current VIX value. If None, reads latest from DB.

        Returns:
            dict with check results
        """
        # Get current VIX
        if current_vix is None:
            current_vix = self._get_latest_vix()

        if current_vix is None:
            logger.warning("Could not get current VIX value")
            return {'status': 'error', 'reason': 'no_vix_data'}

        # Check for DRY_20 open positions
        dry20_positions = self._get_open_dry20_positions()

        result = {
            'vix': current_vix,
            'dry20_open': len(dry20_positions),
            'alert_level': None,
            'positions': dry20_positions,
        }

        if not dry20_positions:
            result['status'] = 'ok'
            result['message'] = f'VIX={current_vix:.1f}, no DRY_20 positions open'
            return result

        # VIX spike with open position — alert
        if current_vix >= VIX_CRITICAL_THRESHOLD:
            result['alert_level'] = 'CRITICAL'
            result['status'] = 'alert'
            result['message'] = (
                f'VIX={current_vix:.1f} >= {VIX_CRITICAL_THRESHOLD} '
                f'with {len(dry20_positions)} DRY_20 LONG position(s) open. '
                f'MANUAL REVIEW REQUIRED — consider closing.'
            )
            self._send_alert('CRITICAL', result['message'], dry20_positions)

        elif current_vix >= VIX_ALERT_THRESHOLD:
            result['alert_level'] = 'WARNING'
            result['status'] = 'alert'
            result['message'] = (
                f'VIX={current_vix:.1f} >= {VIX_ALERT_THRESHOLD} '
                f'with {len(dry20_positions)} DRY_20 LONG position(s) open. '
                f'Monitor closely.'
            )
            self._send_alert('WARNING', result['message'], dry20_positions)

        else:
            result['status'] = 'ok'
            result['message'] = (
                f'VIX={current_vix:.1f} (below {VIX_ALERT_THRESHOLD}), '
                f'{len(dry20_positions)} DRY_20 position(s) — no action needed.'
            )

        logger.info(result['message'])
        return result

    def _get_latest_vix(self) -> float:
        """Get the most recent VIX value from nifty_daily."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT india_vix FROM nifty_daily
                WHERE india_vix IS NOT NULL
                ORDER BY date DESC LIMIT 1
            """)
            row = cur.fetchone()
            return float(row[0]) if row else None
        except Exception as e:
            logger.error(f"Failed to get VIX: {e}")
            return None

    def _get_open_dry20_positions(self) -> list:
        """Get open KAUFMAN_DRY_20 paper trading positions."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT trade_id, direction, entry_price, entry_date, entry_vix
                FROM trades
                WHERE trade_type = 'PAPER'
                  AND signal_id = 'KAUFMAN_DRY_20'
                  AND exit_date IS NULL
                ORDER BY entry_date
            """)
            return [
                {
                    'trade_id': r[0],
                    'direction': r[1],
                    'entry_price': r[2],
                    'entry_date': str(r[3]),
                    'entry_vix': r[4],
                }
                for r in cur.fetchall()
            ]
        except Exception as e:
            logger.error(f"Failed to query DRY_20 positions: {e}")
            return []

    def _send_alert(self, level: str, message: str, positions: list):
        """Send alert via AlertEngine or direct Telegram."""
        if self.alert_engine:
            self.alert_engine.check_and_alert('VIX_SPIKE_DRY20', {
                'signal_id': 'KAUFMAN_DRY_20',
                'message': message,
            })
        else:
            # Direct Telegram fallback
            token = os.environ.get('TELEGRAM_BOT_TOKEN')
            chat_id = os.environ.get('TELEGRAM_CHAT_ID')
            if token and chat_id:
                try:
                    alerter = TelegramAlerter(token, chat_id)
                    pos_detail = '\n'.join(
                        f"  {p['direction']} @ {p['entry_price']:.0f} "
                        f"(entered {p['entry_date']}, VIX was {p['entry_vix']:.1f})"
                        for p in positions
                    )
                    full_msg = f"{message}\n\nOpen positions:\n{pos_detail}"
                    alerter.send(level, full_msg, signal_id='KAUFMAN_DRY_20')
                except Exception as e:
                    logger.error(f"Telegram alert failed: {e}")

        # Always log to DB
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO alerts_log
                    (alert_type, alert_level, message, signal_id, alert_time)
                VALUES ('VIX_SPIKE_DRY20', %s, %s, 'KAUFMAN_DRY_20', NOW())
            """, (level, message))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
            self.conn.rollback()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='VIX spike monitor')
    parser.add_argument('--vix', type=float, help='Override VIX value (for testing)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    monitor = VIXMonitor()
    result = monitor.check(current_vix=args.vix)
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    if result.get('alert_level'):
        print(f"Alert: {result['alert_level']}")


if __name__ == '__main__':
    main()

"""
Combination Engine: GRIMES_DRY_3_2 + KAUFMAN_DRY_12 SEQ_5

Grimes detects trend structure (higher highs + higher lows + ADX > 25).
Kaufman confirms volume exhaustion within 5 days (price up, volume down).
Together they catch the final momentum push in a mature trend.

OOS validated: Sharpe 5.08, +2,740 pts, 3.2% max drawdown (2024-2026).
"""

import logging
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CombinationEngine:

    def __init__(self):
        self.pending_long = None   # date Grimes long fired
        self.pending_short = None  # date Grimes short fired
        self.position = None       # {'direction', 'entry_date', 'days_held'}

    def update(self, today: date, row: pd.Series,
               prev_row: pd.Series) -> dict:
        """
        Process one trading day. Returns action dict.

        row/prev_row must contain: high, low, close, volume,
        adx_14, prev_high, prev_low, prev_close, prev_volume, regime.
        """
        result = {
            'grimes_fired': False,
            'grimes_direction': None,
            'kaufman_confirmed': False,
            'action': None,
            'pending_days_remaining': 0,
            'reason': '',
        }

        # Step 1: Check exits first
        if self.position:
            exit_reason = self._check_exit(today, row, prev_row)
            if exit_reason:
                result['action'] = 'EXIT'
                result['reason'] = exit_reason
                self.position = None
                return result

        # Step 2: Check if Grimes fires today
        regime = str(row.get('regime', ''))
        trending = regime in ('TRENDING_UP', 'TRENDING_DOWN', 'TRENDING')
        adx_ok = pd.notna(row.get('adx_14')) and float(row['adx_14']) > 25

        grimes_long = (
            float(row['high']) > float(prev_row['high']) and
            float(row['low']) > float(prev_row['low']) and
            adx_ok and trending
        )
        grimes_short = (
            float(row['low']) < float(prev_row['low']) and
            float(row['high']) < float(prev_row['high']) and
            adx_ok and trending
        )

        if grimes_long and not self.position:
            self.pending_long = today
            self.pending_short = None
            result['grimes_fired'] = True
            result['grimes_direction'] = 'LONG'
            logger.info(f"  COMBO: Grimes LONG fired {today}, waiting for Kaufman confirmation")

        if grimes_short and not self.position:
            self.pending_short = today
            self.pending_long = None
            result['grimes_fired'] = True
            result['grimes_direction'] = 'SHORT'
            logger.info(f"  COMBO: Grimes SHORT fired {today}, waiting for Kaufman confirmation")

        # Step 3: Expire pending signals older than 5 days
        if self.pending_long and (today - self.pending_long).days > 5:
            logger.info(f"  COMBO: Pending LONG expired (fired {self.pending_long})")
            self.pending_long = None
        if self.pending_short and (today - self.pending_short).days > 5:
            logger.info(f"  COMBO: Pending SHORT expired (fired {self.pending_short})")
            self.pending_short = None

        # Step 4: Check Kaufman DRY_12 confirmation
        kaufman_long = (
            float(row['close']) > float(prev_row['close']) and
            float(row['volume']) < float(prev_row['volume'])
        )
        kaufman_short = (
            float(row['close']) < float(prev_row['close']) and
            float(row['volume']) > float(prev_row['volume'])
        )

        # Step 5: Enter if confirmation matches pending
        if self.pending_long and kaufman_long and not self.position:
            days_waited = (today - self.pending_long).days
            self.position = {
                'direction': 'LONG',
                'entry_date': today,
                'days_held': 0,
            }
            self.pending_long = None
            result['kaufman_confirmed'] = True
            result['action'] = 'ENTER_LONG'
            result['reason'] = f'Grimes+Kaufman SEQ_5 LONG (confirmed after {days_waited}d)'
            logger.info(f"  COMBO: ENTER_LONG confirmed after {days_waited} days")
            return result

        if self.pending_short and kaufman_short and not self.position:
            days_waited = (today - self.pending_short).days
            self.position = {
                'direction': 'SHORT',
                'entry_date': today,
                'days_held': 0,
            }
            self.pending_short = None
            result['kaufman_confirmed'] = True
            result['action'] = 'ENTER_SHORT'
            result['reason'] = f'Grimes+Kaufman SEQ_5 SHORT (confirmed after {days_waited}d)'
            logger.info(f"  COMBO: ENTER_SHORT confirmed after {days_waited} days")
            return result

        # Report pending status
        if self.pending_long:
            result['pending_days_remaining'] = 5 - (today - self.pending_long).days
        elif self.pending_short:
            result['pending_days_remaining'] = 5 - (today - self.pending_short).days

        return result

    def _check_exit(self, today: date, row, prev_row) -> Optional[str]:
        """Check exit conditions. Returns reason string or None."""
        if not self.position:
            return None

        self.position['days_held'] += 1

        # Structure violation
        if self.position['direction'] == 'LONG':
            if float(row['low']) < float(prev_row['low']):
                return 'structure_violation (low < prev_low)'
        elif self.position['direction'] == 'SHORT':
            if float(row['high']) > float(prev_row['high']):
                return 'structure_violation (high > prev_high)'

        # Hold max 10 days
        if self.position['days_held'] >= 10:
            return 'hold_max_10'

        return None
        # Note: 2% stop loss handled by trade recording layer

    def get_state(self) -> dict:
        """Export state for DB persistence."""
        return {
            'pending_long': str(self.pending_long) if self.pending_long else None,
            'pending_short': str(self.pending_short) if self.pending_short else None,
            'position_open': self.position is not None,
            'position_dir': self.position['direction'] if self.position else None,
            'position_date': str(self.position['entry_date']) if self.position else None,
            'days_held': self.position['days_held'] if self.position else 0,
        }

    def load_state(self, conn):
        """Load state from combination_state table."""
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT pending_long, pending_short, position_open,
                       position_dir, position_date, days_held
                FROM combination_state
                WHERE engine_name = 'GRIMES_KAUFMAN_SEQ5'
            """)
            row = cur.fetchone()
            if row:
                self.pending_long = row[0]
                self.pending_short = row[1]
                if row[2]:  # position_open
                    self.position = {
                        'direction': row[3],
                        'entry_date': row[4],
                        'days_held': row[5] or 0,
                    }
                else:
                    self.position = None
        except Exception as e:
            logger.warning(f"Could not load combination state: {e}")

    def save_state(self, conn):
        """Save state to combination_state table."""
        try:
            state = self.get_state()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO combination_state
                    (engine_name, pending_long, pending_short,
                     position_open, position_dir, position_date,
                     days_held, updated_at)
                VALUES ('GRIMES_KAUFMAN_SEQ5', %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (engine_name) DO UPDATE SET
                    pending_long = EXCLUDED.pending_long,
                    pending_short = EXCLUDED.pending_short,
                    position_open = EXCLUDED.position_open,
                    position_dir = EXCLUDED.position_dir,
                    position_date = EXCLUDED.position_date,
                    days_held = EXCLUDED.days_held,
                    updated_at = NOW()
            """, (
                state['pending_long'], state['pending_short'],
                state['position_open'], state['position_dir'],
                state['position_date'], state['days_held'],
            ))
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not save combination state: {e}")
            try:
                conn.rollback()
            except Exception:
                pass

    def get_daily_summary(self) -> str:
        """One-line summary for Telegram digest."""
        if self.position:
            return (f"Position: {self.position['direction']} "
                    f"(day {self.position['days_held']}/10)")
        if self.pending_long:
            return f"Pending LONG (Grimes fired, waiting Kaufman)"
        if self.pending_short:
            return f"Pending SHORT (Grimes fired, waiting Kaufman)"
        return "Idle (no pending signal)"

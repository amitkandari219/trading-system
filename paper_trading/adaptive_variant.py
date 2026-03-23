"""
Adaptive signal variant switching based on rolling performance.

Monitors live signal performance and switches between variants
when rolling metrics cross defined thresholds.

DRY_12 variants:
  PRIMARY:      No extra filter, relies on volatility floor (range >= 0.8%)
  REDUCED:      Halved position size (0.5x), activated when rolling Sharpe degrades

Switch logic:
  PRIMARY → REDUCED when rolling_60d_sharpe < 0.5
  REDUCED → PRIMARY when rolling_60d_sharpe > 1.0
  Hysteresis prevents rapid switching.

Note: The original ADX_FILTERED variant (ADX < 25) was found to make DRY_12
WORSE in low-ADX environments (the very regime it was meant to help).
Replaced with position-size reduction which preserves optionality.
"""
import logging
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveVariantManager:
    """Manages signal variant switching based on rolling performance."""

    def __init__(self, db_conn=None):
        self.conn = db_conn
        # Track active variants per signal
        # REDUCED = 0.5x position size (still trades, but smaller)
        self.active_variants: Dict[str, str] = {
            'KAUFMAN_DRY_12': 'PRIMARY',
        }
        # Size multiplier per variant
        self.variant_size_mult: Dict[str, float] = {
            'PRIMARY': 1.0,
            'REDUCED': 0.5,
        }
        # Cooldown: minimum days between switches
        self.switch_cooldown_days = 30
        self.last_switch_dates: Dict[str, date] = {}

    def get_active_variant(self, signal_id: str) -> str:
        """Return currently active variant for a signal."""
        return self.active_variants.get(signal_id, 'PRIMARY')

    def should_apply_adx_filter(self, signal_id: str) -> bool:
        """Deprecated — ADX filtering removed. Use get_size_multiplier instead."""
        return False

    def get_size_multiplier(self, signal_id: str) -> float:
        """Get position size multiplier for the active variant."""
        variant = self.active_variants.get(signal_id, 'PRIMARY')
        return self.variant_size_mult.get(variant, 1.0)

    def evaluate_switch(self, signal_id: str, as_of: date = None) -> Dict:
        """
        Evaluate whether to switch variants based on rolling performance.

        Args:
            signal_id: Signal to evaluate
            as_of: Date to evaluate as of

        Returns:
            {
                'signal_id': str,
                'current_variant': str,
                'new_variant': str or None (None = no switch),
                'rolling_sharpe': float,
                'rolling_win_rate': float,
                'rolling_trades': int,
                'reason': str,
            }
        """
        as_of = as_of or date.today()
        current = self.active_variants.get(signal_id, 'PRIMARY')

        # Check cooldown
        last_switch = self.last_switch_dates.get(signal_id)
        if last_switch and (as_of - last_switch).days < self.switch_cooldown_days:
            return {
                'signal_id': signal_id,
                'current_variant': current,
                'new_variant': None,
                'reason': f'cooldown ({self.switch_cooldown_days - (as_of - last_switch).days}d remaining)',
            }

        # Get rolling 60-day performance
        metrics = self._compute_rolling_metrics(signal_id, as_of, window_days=60)
        rolling_sharpe = metrics.get('sharpe', 0.0)
        rolling_wr = metrics.get('win_rate', 0.0)
        n_trades = metrics.get('trade_count', 0)

        result = {
            'signal_id': signal_id,
            'current_variant': current,
            'new_variant': None,
            'rolling_sharpe': rolling_sharpe,
            'rolling_win_rate': rolling_wr,
            'rolling_trades': n_trades,
        }

        # Need minimum 5 trades to evaluate
        if n_trades < 5:
            result['reason'] = f'insufficient trades ({n_trades} < 5)'
            return result

        # Switch logic with hysteresis
        if current == 'PRIMARY' and rolling_sharpe < 0.5:
            result['new_variant'] = 'REDUCED'
            result['reason'] = f'rolling Sharpe {rolling_sharpe:.2f} < 0.5 threshold'
            self._execute_switch(signal_id, 'REDUCED', as_of)
            logger.warning(
                f"VARIANT SWITCH: {signal_id} PRIMARY → REDUCED (0.5x size) "
                f"(rolling Sharpe={rolling_sharpe:.2f}, {n_trades} trades)"
            )
        elif current == 'REDUCED' and rolling_sharpe > 1.0:
            result['new_variant'] = 'PRIMARY'
            result['reason'] = f'rolling Sharpe {rolling_sharpe:.2f} > 1.0 recovery'
            self._execute_switch(signal_id, 'PRIMARY', as_of)
            logger.info(
                f"VARIANT SWITCH: {signal_id} REDUCED → PRIMARY "
                f"(rolling Sharpe={rolling_sharpe:.2f}, recovered)"
            )
        else:
            result['reason'] = f'no switch needed (Sharpe={rolling_sharpe:.2f})'

        return result

    def _execute_switch(self, signal_id: str, new_variant: str, as_of: date):
        """Record the variant switch."""
        self.active_variants[signal_id] = new_variant
        self.last_switch_dates[signal_id] = as_of

        # Persist to DB if available
        if self.conn:
            try:
                cur = self.conn.cursor()
                cur.execute("""
                    INSERT INTO variant_switches (signal_id, old_variant, new_variant,
                                                   switch_date, reason)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (signal_id,
                      'PRIMARY' if new_variant == 'ADX_FILTERED' else 'ADX_FILTERED',
                      new_variant, as_of, 'rolling_sharpe_threshold'))
                self.conn.commit()
            except Exception as e:
                logger.debug(f"Variant switch DB log failed: {e}")

    def _compute_rolling_metrics(self, signal_id: str, as_of: date,
                                  window_days: int = 60) -> Dict:
        """Compute rolling Sharpe and win rate from recent trades."""
        if not self.conn:
            return {'sharpe': 0.0, 'win_rate': 0.0, 'trade_count': 0}

        try:
            cutoff = as_of - timedelta(days=window_days)
            cur = self.conn.cursor()
            cur.execute("""
                SELECT return_pct FROM trades
                WHERE signal_id = %s
                  AND exit_date IS NOT NULL
                  AND exit_date >= %s
                  AND exit_date <= %s
                ORDER BY exit_date
            """, (signal_id, cutoff, as_of))
            rows = cur.fetchall()

            if not rows:
                return {'sharpe': 0.0, 'win_rate': 0.0, 'trade_count': 0}

            returns = [float(r[0]) for r in rows if r[0] is not None]
            if len(returns) < 2:
                return {'sharpe': 0.0, 'win_rate': 0.0, 'trade_count': len(returns)}

            arr = np.array(returns)
            mean_r = arr.mean()
            std_r = arr.std()
            sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
            win_rate = (arr > 0).sum() / len(arr)

            return {
                'sharpe': round(float(sharpe), 3),
                'win_rate': round(float(win_rate), 3),
                'trade_count': len(returns),
                'mean_return': round(float(mean_r), 5),
                'std_return': round(float(std_r), 5),
            }
        except Exception as e:
            logger.warning(f"Rolling metrics computation failed for {signal_id}: {e}")
            return {'sharpe': 0.0, 'win_rate': 0.0, 'trade_count': 0}

    def load_state(self, conn):
        """Load persisted variant states from DB."""
        self.conn = conn
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT ON (signal_id) signal_id, new_variant, switch_date
                FROM variant_switches
                ORDER BY signal_id, switch_date DESC
            """)
            for row in cur.fetchall():
                self.active_variants[row[0]] = row[1]
                self.last_switch_dates[row[0]] = row[2]
                logger.info(f"Loaded variant state: {row[0]} = {row[1]} (since {row[2]})")
        except Exception as e:
            logger.debug(f"No variant switch history: {e}")

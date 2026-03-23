"""
Max Pain Gravitational Pull Signal.

Exploits the empirical tendency of Nifty to close near the max-pain strike
on expiry day.  Max pain is the strike at which option writers (sellers)
collectively face the minimum payout.  Because option sellers are
overwhelmingly institutional, the market "gravitates" toward max pain as
hedging flows push the index.

Signal logic:
    deviation = (price - max_pain) / max_pain * 100

    deviation > +1%  ->  SHORT  (gravity pulls price down to max pain)
    deviation < -1%  ->  LONG   (gravity pulls price up to max pain)
    |deviation| < 1% ->  NO TRADE  (already near max pain)

    Strength scales with:
        - Distance from max pain (larger deviation = stronger pull)
        - Proximity to expiry (Monday > Friday; Tuesday expiry-day strongest)

    Target  : max_pain +/- 0.3%
    SL      : entry + 1.2 * |deviation| in adverse direction

Nifty weekly expiry moved to Tuesday (post-Sept 2025).  Signal is strongest
on Monday close and Tuesday intraday.

Walk-forward parameters are exposed as class constants for easy tuning.

Columns required in df:
    close, max_pain_strike

Usage:
    from signals.options.max_pain_gravity import MaxPainGravitySignal
    sig = MaxPainGravitySignal()
    result = sig.evaluate(df, date)
"""

import logging
import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS  (walk-forward tunable)
# ================================================================

SIGNAL_ID = 'MAX_PAIN_GRAVITY'

# Deviation thresholds (percentage of max pain)
DEVIATION_MIN_PCT = 1.0          # Minimum % away from max pain to fire
DEVIATION_MAX_PCT = 5.0          # Sanity cap — beyond this, max pain data suspect

# Target / stop
TGT_BAND_PCT = 0.3               # Exit when price within 0.3% of max pain
SL_MULTIPLIER = 1.2               # SL at 1.2x the deviation in adverse direction

# Strength scaling
BASE_STRENGTH = 0.40
EXPIRY_DAY_BOOST = 0.25           # Tuesday (expiry day)
PRE_EXPIRY_BOOST = 0.15           # Monday (day before expiry)
DEVIATION_STRENGTH_SCALE = 0.08   # Per 1% deviation beyond threshold
MAX_STRENGTH = 0.95
MIN_STRENGTH = 0.10

# Expiry day = Tuesday (post-Sept 2025 for Nifty weeklies)
EXPIRY_WEEKDAY = 1                # Monday=0, Tuesday=1, ...


# ================================================================
# HELPERS
# ================================================================

def _safe_float(val: Any, default: float = float('nan')) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _days_to_expiry(trade_date: date) -> int:
    """
    Compute calendar days until the next weekly expiry (Tuesday).
    Returns 0 on expiry day itself.
    """
    weekday = trade_date.weekday()  # Mon=0 .. Sun=6
    if weekday <= EXPIRY_WEEKDAY:
        return EXPIRY_WEEKDAY - weekday
    return 7 - weekday + EXPIRY_WEEKDAY


# ================================================================
# SIGNAL CLASS
# ================================================================

class MaxPainGravitySignal:
    """
    Max pain gravitational pull signal for Nifty options.

    Fires when Nifty deviates > 1% from the max-pain strike, betting
    on mean-reversion toward max pain.  Strongest near weekly expiry
    (Tuesday).
    """

    SIGNAL_ID = SIGNAL_ID

    # -- Walk-forward params (class-level for WF engine) --
    WF_DEVIATION_MIN_PCT = DEVIATION_MIN_PCT
    WF_DEVIATION_MAX_PCT = DEVIATION_MAX_PCT
    WF_TGT_BAND_PCT = TGT_BAND_PCT
    WF_SL_MULTIPLIER = SL_MULTIPLIER
    WF_BASE_STRENGTH = BASE_STRENGTH
    WF_EXPIRY_DAY_BOOST = EXPIRY_DAY_BOOST
    WF_PRE_EXPIRY_BOOST = PRE_EXPIRY_BOOST

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('MaxPainGravitySignal initialised')

    # ----------------------------------------------------------
    # evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate max-pain gravity signal.

        Parameters
        ----------
        df         : DataFrame with columns [close, max_pain_strike].
                     Rows up to and including trade_date.
        trade_date : The date to evaluate.

        Returns
        -------
        dict with keys: signal_id, direction, strength, price, reason,
        metadata — or None if no trade.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('MaxPainGravitySignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        # ── Guard: one fire per day ─────────────────────────────
        if self._last_fire_date == trade_date:
            return None

        # ── Extract latest row ──────────────────────────────────
        if df is None or df.empty:
            logger.debug('Empty DataFrame')
            return None

        row = df.iloc[-1]
        price = _safe_float(row.get('close'))
        max_pain = _safe_float(row.get('max_pain_strike'))

        if math.isnan(price) or price <= 0:
            logger.debug('Invalid price: %s', price)
            return None
        if math.isnan(max_pain) or max_pain <= 0:
            logger.debug('Missing max_pain_strike — options data unavailable')
            return None

        # ── Deviation ───────────────────────────────────────────
        deviation_pct = ((price - max_pain) / max_pain) * 100.0

        if abs(deviation_pct) < self.WF_DEVIATION_MIN_PCT:
            logger.debug('Deviation %.2f%% below threshold', deviation_pct)
            return None
        if abs(deviation_pct) > self.WF_DEVIATION_MAX_PCT:
            logger.debug('Deviation %.2f%% exceeds sanity cap', deviation_pct)
            return None

        # ── Direction ───────────────────────────────────────────
        if deviation_pct > 0:
            direction = 'SHORT'
        else:
            direction = 'LONG'

        # ── Strength ────────────────────────────────────────────
        dte = _days_to_expiry(trade_date)
        strength = self.WF_BASE_STRENGTH

        # Expiry proximity boost
        if dte == 0:
            strength += self.WF_EXPIRY_DAY_BOOST
        elif dte == 1:
            strength += self.WF_PRE_EXPIRY_BOOST

        # Deviation magnitude boost
        extra_dev = abs(deviation_pct) - self.WF_DEVIATION_MIN_PCT
        strength += extra_dev * DEVIATION_STRENGTH_SCALE

        strength = min(MAX_STRENGTH, max(MIN_STRENGTH, strength))

        # ── Target / SL ─────────────────────────────────────────
        target = max_pain * (1.0 + self.WF_TGT_BAND_PCT / 100.0) if direction == 'LONG' \
            else max_pain * (1.0 - self.WF_TGT_BAND_PCT / 100.0)

        deviation_pts = abs(price - max_pain)
        if direction == 'LONG':
            stop_loss = price - deviation_pts * self.WF_SL_MULTIPLIER
        else:
            stop_loss = price + deviation_pts * self.WF_SL_MULTIPLIER

        # ── Reason ──────────────────────────────────────────────
        reason_parts = [
            'MAX_PAIN_GRAVITY',
            f'Price={price:.2f}',
            f'MaxPain={max_pain:.2f}',
            f'Deviation={deviation_pct:+.2f}%',
            f'DTE={dte}d',
            f'Strength={strength:.2f}',
        ]

        self._last_fire_date = trade_date

        logger.info(
            '%s signal: %s %s dev=%.2f%% dte=%d strength=%.3f',
            self.SIGNAL_ID, direction, trade_date, deviation_pct, dte, strength,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2),
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'max_pain_strike': round(max_pain, 2),
                'deviation_pct': round(deviation_pct, 4),
                'days_to_expiry': dte,
                'target': round(target, 2),
                'stop_loss': round(stop_loss, 2),
                'expiry_weekday': EXPIRY_WEEKDAY,
            },
        }

    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"MaxPainGravitySignal(signal_id='{self.SIGNAL_ID}')"

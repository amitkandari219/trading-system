"""
IV Skew Momentum Signal.

Tracks the put-call implied-volatility skew and its 5-day rate of change.
In equity index options, puts normally trade at a higher IV than calls
(negative skew / "fear premium").  Rapid changes in this skew reveal
shifts in institutional hedging demand.

Signal logic:
    skew = atm_put_iv - atm_call_iv   (normally positive for indices)
    skew_momentum = skew - skew_5d_ago

    Skew flattening (skew_momentum < -threshold):
        -> Fear receding, hedgers unwinding puts -> LONG

    Skew steepening (skew_momentum > +threshold):
        -> Fear building, put demand surging -> SHORT

    Skew reversal (skew < 0, puts cheaper than calls):
        -> Rare extreme bullishness -> contrarian SHORT

    Strength scales with magnitude of momentum and absolute skew level.

Columns required in df:
    close, atm_put_iv, atm_call_iv  (or iv_skew precomputed)

Walk-forward parameters exposed as class constants.

Usage:
    from signals.options.iv_skew_momentum import IVSkewMomentumSignal
    sig = IVSkewMomentumSignal()
    result = sig.evaluate(df, date)
"""

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS  (walk-forward tunable)
# ================================================================

SIGNAL_ID = 'IV_SKEW_MOMENTUM'

# Skew momentum thresholds (IV percentage points)
SKEW_MOM_THRESHOLD = 2.0          # Min momentum magnitude to fire
SKEW_MOM_MAX = 15.0               # Sanity cap — data likely stale/bad
SKEW_REVERSAL_THRESHOLD = -0.5    # Skew < this = puts cheaper than calls (rare)

# Lookback
SKEW_MOM_LOOKBACK = 5             # 5-day momentum window

# Strength
BASE_STRENGTH = 0.40
MOM_STRENGTH_PER_POINT = 0.06     # Per IV point of momentum beyond threshold
SKEW_REVERSAL_STRENGTH = 0.70     # Contrarian signal on skew reversal
HIGH_ABS_SKEW_BOOST = 0.08        # When absolute skew > 5 pts
MAX_STRENGTH = 0.90
MIN_STRENGTH = 0.10


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


def _compute_skew(row: pd.Series) -> float:
    """
    Compute put-call IV skew from a row.
    Uses precomputed iv_skew if available, else atm_put_iv - atm_call_iv.
    """
    # Try precomputed column first
    precomputed = _safe_float(row.get('iv_skew'))
    if not math.isnan(precomputed):
        return precomputed

    put_iv = _safe_float(row.get('atm_put_iv'))
    call_iv = _safe_float(row.get('atm_call_iv'))

    if math.isnan(put_iv) or math.isnan(call_iv):
        return float('nan')
    if put_iv <= 0 or call_iv <= 0:
        return float('nan')

    return put_iv - call_iv


# ================================================================
# SIGNAL CLASS
# ================================================================

class IVSkewMomentumSignal:
    """
    IV skew momentum signal for Nifty options.

    Tracks the rate of change in put-call IV skew over a 5-day window.
    Rapid flattening (fear receding) triggers LONG; rapid steepening
    (fear building) triggers SHORT.  A rare skew reversal (puts cheaper
    than calls) fires a contrarian SHORT.
    """

    SIGNAL_ID = SIGNAL_ID

    # -- Walk-forward params --
    WF_SKEW_MOM_THRESHOLD = SKEW_MOM_THRESHOLD
    WF_SKEW_MOM_MAX = SKEW_MOM_MAX
    WF_SKEW_REVERSAL_THRESHOLD = SKEW_REVERSAL_THRESHOLD
    WF_SKEW_MOM_LOOKBACK = SKEW_MOM_LOOKBACK
    WF_BASE_STRENGTH = BASE_STRENGTH
    WF_MOM_STRENGTH_PER_POINT = MOM_STRENGTH_PER_POINT
    WF_SKEW_REVERSAL_STRENGTH = SKEW_REVERSAL_STRENGTH

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('IVSkewMomentumSignal initialised')

    # ----------------------------------------------------------
    # evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate IV skew momentum signal.

        Parameters
        ----------
        df         : DataFrame with columns [close, atm_put_iv, atm_call_iv]
                     or [close, iv_skew].  Need >= 6 rows for 5-day momentum.
        trade_date : The date to evaluate.

        Returns
        -------
        dict with keys: signal_id, direction, strength, price, reason,
        metadata — or None if no trade.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('IVSkewMomentumSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if self._last_fire_date == trade_date:
            return None

        min_rows = self.WF_SKEW_MOM_LOOKBACK + 1
        if df is None or len(df) < min_rows:
            logger.debug('Insufficient data (%s rows, need %d)', 0 if df is None else len(df), min_rows)
            return None

        # ── Extract price ───────────────────────────────────────
        today = df.iloc[-1]
        price = _safe_float(today.get('close'))
        if math.isnan(price) or price <= 0:
            logger.debug('Invalid price')
            return None

        # ── Compute current and lagged skew ─────────────────────
        skew_today = _compute_skew(df.iloc[-1])
        skew_lagged = _compute_skew(df.iloc[-min_rows])

        if math.isnan(skew_today) or math.isnan(skew_lagged):
            logger.debug('Missing IV skew data — options data unavailable')
            return None

        skew_momentum = skew_today - skew_lagged

        # ── Check for skew reversal (contrarian) ────────────────
        is_reversal = skew_today < self.WF_SKEW_REVERSAL_THRESHOLD

        if is_reversal:
            direction = 'SHORT'
            strength = self.WF_SKEW_REVERSAL_STRENGTH
            trigger = 'SKEW_REVERSAL'
        elif abs(skew_momentum) < self.WF_SKEW_MOM_THRESHOLD:
            logger.debug('Skew momentum %.2f below threshold', skew_momentum)
            return None
        elif abs(skew_momentum) > self.WF_SKEW_MOM_MAX:
            logger.debug('Skew momentum %.2f exceeds sanity cap', skew_momentum)
            return None
        elif skew_momentum < 0:
            # Flattening — fear receding
            direction = 'LONG'
            strength = self.WF_BASE_STRENGTH
            trigger = 'SKEW_FLATTENING'
        else:
            # Steepening — fear building
            direction = 'SHORT'
            strength = self.WF_BASE_STRENGTH
            trigger = 'SKEW_STEEPENING'

        # ── Strength adjustment ─────────────────────────────────
        if not is_reversal:
            extra_mom = abs(skew_momentum) - self.WF_SKEW_MOM_THRESHOLD
            strength += extra_mom * self.WF_MOM_STRENGTH_PER_POINT

            if abs(skew_today) > 5.0:
                strength += HIGH_ABS_SKEW_BOOST

        strength = min(MAX_STRENGTH, max(MIN_STRENGTH, strength))

        # ── Reason ──────────────────────────────────────────────
        reason_parts = [
            f'IV_SKEW_MOMENTUM ({trigger})',
            f'Price={price:.2f}',
            f'Skew={skew_today:.2f}',
            f'SkewMom5d={skew_momentum:+.2f}',
            f'Strength={strength:.2f}',
        ]

        self._last_fire_date = trade_date

        logger.info(
            '%s signal: %s %s skew=%.2f mom=%.2f trigger=%s strength=%.3f',
            self.SIGNAL_ID, direction, trade_date, skew_today,
            skew_momentum, trigger, strength,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2),
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'skew_today': round(skew_today, 4),
                'skew_lagged': round(skew_lagged, 4),
                'skew_momentum': round(skew_momentum, 4),
                'lookback_days': self.WF_SKEW_MOM_LOOKBACK,
                'trigger': trigger,
                'atm_put_iv': round(_safe_float(today.get('atm_put_iv')), 4)
                    if not math.isnan(_safe_float(today.get('atm_put_iv'))) else None,
                'atm_call_iv': round(_safe_float(today.get('atm_call_iv')), 4)
                    if not math.isnan(_safe_float(today.get('atm_call_iv'))) else None,
            },
        }

    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"IVSkewMomentumSignal(signal_id='{self.SIGNAL_ID}')"

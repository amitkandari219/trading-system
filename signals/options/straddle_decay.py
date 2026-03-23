"""
Straddle Theta Decay Curve Signal.

Tracks ATM straddle premium (put + call) relative to its theoretical
Black-Scholes fair value for the remaining time to expiry.  On expiry day
(Tuesday for Nifty weeklies post-Sept 2025), the last 2 hours see
accelerated theta decay.  Mispricing of the straddle relative to
theoretical value creates opportunities.

Signal logic:
    ratio = atm_straddle_premium / bs_theoretical_straddle

    ratio > 1.5  ->  SELL straddle  (overpriced, theta will crush it)
    ratio < 0.7  ->  BUY  straddle  (underpriced, expecting vol expansion)
    otherwise    ->  NO TRADE

    This is primarily an expiry-day intraday signal.  On non-expiry days
    the signal still fires but with reduced strength.

    BS theoretical straddle is approximated as:
        straddle_theo = spot * sigma * sqrt(T)  * (2 / sqrt(2*pi))
    where T = time_to_expiry_hours / (252 * 6.25)  [trading hours in a year]

Columns required in df:
    close, atm_straddle_premium, time_to_expiry_hours

Walk-forward parameters exposed as class constants.

Usage:
    from signals.options.straddle_decay import StraddleDecaySignal
    sig = StraddleDecaySignal()
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

SIGNAL_ID = 'STRADDLE_DECAY'

# Pricing ratio thresholds
OVERPRICED_RATIO = 1.50           # Premium > 1.5x theoretical -> sell
UNDERPRICED_RATIO = 0.70          # Premium < 0.7x theoretical -> buy
EXTREME_OVERPRICED = 2.50         # Sanity cap
EXTREME_UNDERPRICED = 0.20        # Sanity cap

# Default IV for BS approximation when not available
DEFAULT_IV = 15.0                 # Annualised IV % for Nifty (typical)
TRADING_HOURS_PER_YEAR = 252 * 6.25  # NSE: ~6.25 hours per session

# Strength
BASE_STRENGTH_SELL = 0.50
BASE_STRENGTH_BUY = 0.45
EXPIRY_DAY_BOOST = 0.20
LAST_2HR_BOOST = 0.15            # time_to_expiry < 2 hours
RATIO_STRENGTH_SCALE = 0.10      # Per 0.1 ratio deviation beyond threshold
MAX_STRENGTH = 0.90
MIN_STRENGTH = 0.10

# Expiry
EXPIRY_WEEKDAY = 1               # Tuesday
LAST_HOURS_THRESHOLD = 2.0       # Last 2 hours of expiry day


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


def _bs_straddle_approx(spot: float, iv_pct: float, tte_hours: float) -> float:
    """
    Approximate ATM straddle value using simplified Black-Scholes.

    For ATM options, straddle ~ spot * sigma * sqrt(T) * 2/sqrt(2*pi)
    where T is in years.

    Parameters
    ----------
    spot      : Underlying spot price.
    iv_pct    : Implied volatility as percentage (e.g. 15 for 15%).
    tte_hours : Time to expiry in trading hours.

    Returns
    -------
    Theoretical straddle premium (Nifty points).
    """
    if tte_hours <= 0 or spot <= 0 or iv_pct <= 0:
        return float('nan')

    sigma = iv_pct / 100.0
    t_years = tte_hours / TRADING_HOURS_PER_YEAR
    sqrt_t = math.sqrt(t_years)
    coeff = 2.0 / math.sqrt(2.0 * math.pi)

    return spot * sigma * sqrt_t * coeff


# ================================================================
# SIGNAL CLASS
# ================================================================

class StraddleDecaySignal:
    """
    Straddle theta decay curve signal.

    Compares actual ATM straddle premium to Black-Scholes theoretical
    value.  Overpriced straddles are sold (theta harvesting); underpriced
    ones are bought (vol expansion bet).  Strongest on expiry day during
    the last 2 hours.
    """

    SIGNAL_ID = SIGNAL_ID

    # -- Walk-forward params --
    WF_OVERPRICED_RATIO = OVERPRICED_RATIO
    WF_UNDERPRICED_RATIO = UNDERPRICED_RATIO
    WF_BASE_STRENGTH_SELL = BASE_STRENGTH_SELL
    WF_BASE_STRENGTH_BUY = BASE_STRENGTH_BUY
    WF_EXPIRY_DAY_BOOST = EXPIRY_DAY_BOOST
    WF_LAST_2HR_BOOST = LAST_2HR_BOOST
    WF_DEFAULT_IV = DEFAULT_IV

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        logger.info('StraddleDecaySignal initialised')

    # ----------------------------------------------------------
    # evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate straddle decay signal.

        Parameters
        ----------
        df         : DataFrame with columns [close, atm_straddle_premium,
                     time_to_expiry_hours].  Optionally [atm_put_iv,
                     atm_call_iv] for more accurate BS calc.
        trade_date : The date to evaluate.

        Returns
        -------
        dict with keys: signal_id, direction, strength, price, reason,
        metadata — or None if no trade.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('StraddleDecaySignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if self._last_fire_date == trade_date:
            return None

        if df is None or df.empty:
            logger.debug('Empty DataFrame')
            return None

        row = df.iloc[-1]
        price = _safe_float(row.get('close'))
        straddle_premium = _safe_float(row.get('atm_straddle_premium'))
        tte_hours = _safe_float(row.get('time_to_expiry_hours'))

        # ── Validate required data ──────────────────────────────
        if math.isnan(price) or price <= 0:
            logger.debug('Invalid price')
            return None
        if math.isnan(straddle_premium) or straddle_premium <= 0:
            logger.debug('Missing atm_straddle_premium — options data unavailable')
            return None
        if math.isnan(tte_hours) or tte_hours < 0:
            logger.debug('Missing time_to_expiry_hours')
            return None

        # ── Determine IV for BS calculation ─────────────────────
        put_iv = _safe_float(row.get('atm_put_iv'))
        call_iv = _safe_float(row.get('atm_call_iv'))

        if not math.isnan(put_iv) and not math.isnan(call_iv) and put_iv > 0 and call_iv > 0:
            avg_iv = (put_iv + call_iv) / 2.0
        else:
            avg_iv = self.WF_DEFAULT_IV

        # ── Compute theoretical straddle ────────────────────────
        # Use a small floor for tte to avoid division issues at exact expiry
        tte_floor = max(tte_hours, 0.05)
        theo = _bs_straddle_approx(price, avg_iv, tte_floor)

        if math.isnan(theo) or theo <= 0:
            logger.debug('BS theoretical calc returned invalid: %.4f', theo)
            return None

        # ── Pricing ratio ───────────────────────────────────────
        ratio = straddle_premium / theo

        if ratio > EXTREME_OVERPRICED or ratio < EXTREME_UNDERPRICED:
            logger.debug('Ratio %.2f outside sanity bounds', ratio)
            return None

        # ── Signal direction ────────────────────────────────────
        if ratio > self.WF_OVERPRICED_RATIO:
            direction = 'SHORT'      # Sell straddle
            base_strength = self.WF_BASE_STRENGTH_SELL
            trigger = 'OVERPRICED_STRADDLE'
            ratio_excess = ratio - self.WF_OVERPRICED_RATIO
        elif ratio < self.WF_UNDERPRICED_RATIO:
            direction = 'LONG'       # Buy straddle (vol expansion)
            base_strength = self.WF_BASE_STRENGTH_BUY
            trigger = 'UNDERPRICED_STRADDLE'
            ratio_excess = self.WF_UNDERPRICED_RATIO - ratio
        else:
            logger.debug('Ratio %.2f in no-trade zone', ratio)
            return None

        # ── Strength ────────────────────────────────────────────
        strength = base_strength

        is_expiry_day = trade_date.weekday() == EXPIRY_WEEKDAY
        if is_expiry_day:
            strength += self.WF_EXPIRY_DAY_BOOST

        if tte_hours <= LAST_HOURS_THRESHOLD and is_expiry_day:
            strength += self.WF_LAST_2HR_BOOST

        strength += ratio_excess * RATIO_STRENGTH_SCALE

        strength = min(MAX_STRENGTH, max(MIN_STRENGTH, strength))

        # ── Reason ──────────────────────────────────────────────
        reason_parts = [
            f'STRADDLE_DECAY ({trigger})',
            f'Price={price:.2f}',
            f'Straddle={straddle_premium:.2f}',
            f'Theo={theo:.2f}',
            f'Ratio={ratio:.2f}',
            f'TTE={tte_hours:.1f}h',
            f'ExpiryDay={"Y" if is_expiry_day else "N"}',
            f'Strength={strength:.2f}',
        ]

        self._last_fire_date = trade_date

        logger.info(
            '%s signal: %s %s ratio=%.2f tte=%.1fh trigger=%s strength=%.3f',
            self.SIGNAL_ID, direction, trade_date, ratio, tte_hours, trigger, strength,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2),
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'atm_straddle_premium': round(straddle_premium, 2),
                'bs_theoretical': round(theo, 2),
                'premium_ratio': round(ratio, 4),
                'time_to_expiry_hours': round(tte_hours, 2),
                'avg_iv_used': round(avg_iv, 2),
                'is_expiry_day': is_expiry_day,
                'trigger': trigger,
            },
        }

    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None

    def __repr__(self) -> str:
        return f"StraddleDecaySignal(signal_id='{self.SIGNAL_ID}')"

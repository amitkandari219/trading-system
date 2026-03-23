"""
Realized vs Implied Volatility Divergence Signal.

Compares 20-day realized volatility (from daily returns) with India VIX
(implied volatility) and the VIX's 252-day percentile rank (IV_rank) to
detect vol premium selling or vol expansion opportunities.

Signal logic:
    RV_20  = std(daily_returns, 20 days) × sqrt(252) × 100
    IV     = India VIX
    IV_rank = percentile of current VIX over 252-day lookback

    IV_rank > 80 AND RV declining (RV < RV_10d_ago):
        → SHORT vol / premium selling opportunity
        Vol regime: OVERPRICED_IV

    IV_rank < 20 AND RV rising (RV > RV_10d_ago):
        → LONG vol / reduce exposure
        Vol regime: UNDERPRICED_IV

Strength scales with the divergence magnitude between IV and RV.
"""

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS / WF PARAMETERS
# ================================================================

SIGNAL_ID = 'RV_IV_DIVERGENCE'

# Realized vol computation
RV_WINDOW = 20               # 20-day realized vol
RV_LOOKBACK_COMPARE = 10     # Compare current RV vs 10 days ago
ANNUALIZE_FACTOR = math.sqrt(252)

# IV rank
IV_RANK_WINDOW = 252          # 1-year lookback for percentile

# Thresholds
IV_RANK_HIGH = 80             # IV_rank > 80 → overpriced implied
IV_RANK_LOW = 20              # IV_rank < 20 → underpriced implied

# Strength scaling
STRENGTH_FLOOR = 0.30
STRENGTH_CEIL = 1.00

# Columns
VIX_COLUMN = 'india_vix'
CLOSE_COLUMN = 'close'


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


def _compute_realized_vol(close_series: pd.Series, window: int) -> float:
    """
    Compute annualised realized volatility from close prices.

    Returns RV as a percentage (comparable to VIX).
    """
    if len(close_series) < window + 1:
        return float('nan')

    returns = close_series.pct_change().dropna()
    if len(returns) < window:
        return float('nan')

    rv = float(returns.iloc[-window:].std(ddof=1)) * ANNUALIZE_FACTOR * 100.0
    return rv


def _compute_iv_rank(vix_series: pd.Series, window: int) -> float:
    """
    Compute percentile rank of the current VIX over the lookback window.

    Returns a value in [0, 100].
    """
    if len(vix_series) < 2:
        return float('nan')

    lookback = vix_series.iloc[-window:] if len(vix_series) >= window else vix_series
    lookback_clean = lookback.dropna()

    if len(lookback_clean) < 20:
        return float('nan')

    current = float(lookback_clean.iloc[-1])
    rank = float((lookback_clean < current).sum()) / len(lookback_clean) * 100.0
    return rank


# ================================================================
# SIGNAL CLASS
# ================================================================

class RvIvDivergenceSignal:
    """
    Realized vs Implied Volatility divergence signal.

    Detects when implied vol (VIX) is significantly over- or under-priced
    relative to realized vol, using IV_rank and RV trend as confirmation.
    """

    SIGNAL_ID = SIGNAL_ID

    # Walk-forward parameters
    WF_RV_WINDOW = RV_WINDOW
    WF_RV_LOOKBACK_COMPARE = RV_LOOKBACK_COMPARE
    WF_IV_RANK_WINDOW = IV_RANK_WINDOW
    WF_IV_RANK_HIGH = IV_RANK_HIGH
    WF_IV_RANK_LOW = IV_RANK_LOW

    def __init__(self) -> None:
        logger.info('RvIvDivergenceSignal initialised')

    def evaluate(self, df: pd.DataFrame, date_val: date) -> Optional[Dict]:
        """
        Evaluate RV vs IV divergence signal.

        Parameters
        ----------
        df       : DataFrame with at least RV_WINDOW+1 rows.
                   Must contain columns: 'india_vix', 'close'.
        date_val : The evaluation date.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata
        (including vol_regime tag).  Or None if no signal.
        """
        try:
            return self._evaluate_inner(df, date_val)
        except Exception as e:
            logger.error(
                'RvIvDivergenceSignal.evaluate error: %s', e, exc_info=True
            )
            return None

    def _evaluate_inner(
        self, df: pd.DataFrame, date_val: date
    ) -> Optional[Dict]:
        # ── Validate DataFrame ──────────────────────────────────────
        if df is None or df.empty:
            logger.debug('Empty DataFrame')
            return None

        if VIX_COLUMN not in df.columns:
            logger.debug('Missing column: %s', VIX_COLUMN)
            return None

        if CLOSE_COLUMN not in df.columns:
            logger.debug('Missing column: %s', CLOSE_COLUMN)
            return None

        min_rows = RV_WINDOW + RV_LOOKBACK_COMPARE + 1
        if len(df) < min_rows:
            logger.debug(
                'Insufficient data: %d rows < %d required',
                len(df), min_rows,
            )
            return None

        # ── Current IV (VIX) ───────────────────────────────────────
        vix_series = df[VIX_COLUMN].astype(float)
        iv_current = _safe_float(vix_series.iloc[-1])

        if math.isnan(iv_current) or iv_current <= 0:
            logger.debug('Invalid current VIX: %s', iv_current)
            return None

        # ── Compute IV rank ────────────────────────────────────────
        iv_rank = _compute_iv_rank(vix_series, IV_RANK_WINDOW)
        if math.isnan(iv_rank):
            logger.debug('Could not compute IV rank')
            return None

        # ── Compute current RV ─────────────────────────────────────
        close_series = df[CLOSE_COLUMN].astype(float)
        rv_current = _compute_realized_vol(close_series, RV_WINDOW)

        if math.isnan(rv_current):
            logger.debug('Could not compute current RV')
            return None

        # ── Compute RV 10 days ago ─────────────────────────────────
        # Shift the window back by RV_LOOKBACK_COMPARE days
        offset = RV_LOOKBACK_COMPARE
        if len(close_series) < RV_WINDOW + offset + 1:
            logger.debug('Insufficient data for RV comparison')
            return None

        rv_past = _compute_realized_vol(
            close_series.iloc[:-(offset)], RV_WINDOW
        )
        if math.isnan(rv_past):
            logger.debug('Could not compute past RV')
            return None

        rv_declining = rv_current < rv_past
        rv_rising = rv_current > rv_past

        # ── Signal logic ───────────────────────────────────────────
        if iv_rank > IV_RANK_HIGH and rv_declining:
            direction = 'SHORT'
            vol_regime = 'OVERPRICED_IV'
        elif iv_rank < IV_RANK_LOW and rv_rising:
            direction = 'LONG'
            vol_regime = 'UNDERPRICED_IV'
        else:
            logger.debug(
                'No signal: IV_rank=%.1f RV_declining=%s RV_rising=%s',
                iv_rank, rv_declining, rv_rising,
            )
            return None

        # ── Strength ───────────────────────────────────────────────
        # Scale with divergence between IV and RV
        iv_rv_spread = abs(iv_current - rv_current)
        # Normalise: spread of 10+ vol points → strength 1.0
        raw_strength = iv_rv_spread / 10.0
        strength = min(STRENGTH_CEIL, max(STRENGTH_FLOOR, raw_strength))

        # ── Price ──────────────────────────────────────────────────
        price = _safe_float(close_series.iloc[-1])
        if math.isnan(price) or price <= 0:
            logger.debug('Invalid close price')
            return None

        # ── Reason ─────────────────────────────────────────────────
        reason = (
            f"RV_IV_DIVERGENCE | Regime={vol_regime} | "
            f"IV={iv_current:.1f} | RV_20d={rv_current:.1f} | "
            f"IV_rank={iv_rank:.0f} | "
            f"RV_trend={'declining' if rv_declining else 'rising'} | "
            f"Spread={iv_rv_spread:.1f}"
        )

        logger.info(
            '%s signal: %s regime=%s iv=%.1f rv=%.1f iv_rank=%.0f on %s',
            self.SIGNAL_ID, direction, vol_regime, iv_current, rv_current,
            iv_rank, date_val,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2),
            'reason': reason,
            'metadata': {
                'vol_regime': vol_regime,
                'iv_current': round(iv_current, 2),
                'rv_20d': round(rv_current, 2),
                'rv_10d_ago': round(rv_past, 2),
                'iv_rank': round(iv_rank, 2),
                'iv_rv_spread': round(iv_rv_spread, 2),
                'rv_declining': rv_declining,
                'rv_rising': rv_rising,
                'date': str(date_val),
            },
        }

    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        pass

    def __repr__(self) -> str:
        return f"RvIvDivergenceSignal(signal_id='{self.SIGNAL_ID}')"

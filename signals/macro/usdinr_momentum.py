"""
USD/INR Currency Momentum Signal.

Tracks USD/INR exchange rate momentum as a proxy for FII flow direction.
Rupee weakening (USDINR rising) signals foreign capital outflows and is
bearish for Nifty; rupee strengthening (USDINR falling) signals inflows
and is bullish.

Signal logic:
    roc_5d  = (usdinr_today / usdinr_5d_ago - 1) * 100
    roc_20d = (usdinr_today / usdinr_20d_ago - 1) * 100

    roc_5d > 1.0% AND roc_20d > 2.0%  -> SHORT Nifty (FII outflows)
    roc_5d < -0.5%                     -> LONG  Nifty (FII inflows)

    Lag: FX moves lead equity impact by 1-3 sessions.
    Strength scales with magnitude of ROC.

Data requirements:
    - Column 'usdinr_close' in df (or fetched from fx_daily table)
    - Column 'close' for Nifty price

Walk-forward parameters:
    ROC_5D_SHORT_THRESH, ROC_20D_SHORT_THRESH, ROC_5D_LONG_THRESH,
    LOOKBACK_5D, LOOKBACK_20D, STRENGTH_SCALE_FACTOR
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

SIGNAL_ID = 'USDINR_MOMENTUM'

# Lookback windows
LOOKBACK_5D = 5
LOOKBACK_20D = 20

# SHORT thresholds: rupee weakening
ROC_5D_SHORT_THRESH = 1.0       # 5-day ROC > 1% → bearish
ROC_20D_SHORT_THRESH = 2.0      # 20-day ROC > 2% → sustained weakness

# LONG threshold: rupee strengthening
ROC_5D_LONG_THRESH = -0.5       # 5-day ROC < -0.5% → bullish

# Strength scaling
STRENGTH_SCALE_SHORT = 2.0      # Divide 5d ROC by this for SHORT strength
STRENGTH_SCALE_LONG = 1.5       # Divide abs(5d ROC) by this for LONG strength
MIN_STRENGTH = 0.3
MAX_STRENGTH = 1.0

# Minimum data points required
MIN_ROWS = LOOKBACK_20D + 1

# Column names
COL_USDINR = 'usdinr_close'
COL_NIFTY_CLOSE = 'close'


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


def _rate_of_change(series: pd.Series, period: int) -> Optional[float]:
    """Compute percentage rate of change over `period` rows."""
    if len(series) < period + 1:
        return None
    current = _safe_float(series.iloc[-1])
    past = _safe_float(series.iloc[-(period + 1)])
    if math.isnan(current) or math.isnan(past) or past <= 0:
        return None
    return ((current / past) - 1.0) * 100.0


# ================================================================
# SIGNAL CLASS
# ================================================================

class UsdInrMomentumSignal:
    """
    USD/INR currency momentum signal for Nifty.

    Rupee weakening → SHORT bias (FII outflows).
    Rupee strengthening → LONG bias (FII inflows).
    FX moves typically lead equity by 1-3 sessions.
    """

    SIGNAL_ID = SIGNAL_ID

    # Walk-forward parameters
    WF_LOOKBACK_5D = LOOKBACK_5D
    WF_LOOKBACK_20D = LOOKBACK_20D
    WF_ROC_5D_SHORT_THRESH = ROC_5D_SHORT_THRESH
    WF_ROC_20D_SHORT_THRESH = ROC_20D_SHORT_THRESH
    WF_ROC_5D_LONG_THRESH = ROC_5D_LONG_THRESH

    def __init__(self) -> None:
        logger.info('UsdInrMomentumSignal initialised')

    # ----------------------------------------------------------
    # evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, date: date) -> Optional[Dict]:
        """
        Evaluate USD/INR momentum signal.

        Parameters
        ----------
        df   : DataFrame with columns 'usdinr_close' and 'close',
               indexed or containing dates up to `date`.
        date : Evaluation date.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata
        or None if no signal / missing data.
        """
        try:
            return self._evaluate_inner(df, date)
        except Exception as e:
            logger.error('%s.evaluate error: %s', self.SIGNAL_ID, e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, eval_date: date) -> Optional[Dict]:
        # ── Check required column exists ─────────────────────────
        if COL_USDINR not in df.columns:
            logger.debug('%s: column %s not found in df', self.SIGNAL_ID, COL_USDINR)
            return None

        if COL_NIFTY_CLOSE not in df.columns:
            logger.debug('%s: column %s not found in df', self.SIGNAL_ID, COL_NIFTY_CLOSE)
            return None

        # ── Slice data up to eval_date ───────────────────────────
        if hasattr(df.index, 'date'):
            mask = df.index.date <= eval_date
        elif 'date' in df.columns:
            mask = pd.to_datetime(df['date']).dt.date <= eval_date
        else:
            mask = pd.Series([True] * len(df), index=df.index)

        subset = df.loc[mask].copy()

        if len(subset) < MIN_ROWS:
            logger.debug('%s: insufficient data (%d < %d)', self.SIGNAL_ID,
                         len(subset), MIN_ROWS)
            return None

        # ── Drop NaN in USDINR column ────────────────────────────
        usdinr = subset[COL_USDINR].dropna()
        if len(usdinr) < MIN_ROWS:
            logger.debug('%s: insufficient non-null USDINR data', self.SIGNAL_ID)
            return None

        # ── Compute rates of change ──────────────────────────────
        roc_5d = _rate_of_change(usdinr, LOOKBACK_5D)
        roc_20d = _rate_of_change(usdinr, LOOKBACK_20D)

        if roc_5d is None:
            logger.debug('%s: could not compute 5d ROC', self.SIGNAL_ID)
            return None

        # ── Current prices ───────────────────────────────────────
        nifty_price = _safe_float(subset[COL_NIFTY_CLOSE].iloc[-1])
        usdinr_current = _safe_float(usdinr.iloc[-1])

        if math.isnan(nifty_price) or nifty_price <= 0:
            return None
        if math.isnan(usdinr_current) or usdinr_current <= 0:
            return None

        # ── Signal logic ─────────────────────────────────────────
        direction = None
        strength = 0.0
        reason_parts = [self.SIGNAL_ID]

        # SHORT: rupee weakening — both 5d and 20d confirm
        if roc_5d > ROC_5D_SHORT_THRESH and roc_20d is not None and roc_20d > ROC_20D_SHORT_THRESH:
            direction = 'SHORT'
            raw_strength = roc_5d / STRENGTH_SCALE_SHORT
            strength = min(MAX_STRENGTH, max(MIN_STRENGTH, raw_strength))
            reason_parts.extend([
                f"USDINR_5d_ROC={roc_5d:+.2f}%",
                f"USDINR_20d_ROC={roc_20d:+.2f}%",
                "Rupee weakening -> FII outflows -> bearish",
            ])

        # LONG: rupee strengthening
        elif roc_5d < ROC_5D_LONG_THRESH:
            direction = 'LONG'
            raw_strength = abs(roc_5d) / STRENGTH_SCALE_LONG
            strength = min(MAX_STRENGTH, max(MIN_STRENGTH, raw_strength))
            reason_parts.extend([
                f"USDINR_5d_ROC={roc_5d:+.2f}%",
                f"USDINR_20d_ROC={roc_20d:+.2f}%" if roc_20d is not None else "20d_ROC=N/A",
                "Rupee strengthening -> FII inflows -> bullish",
            ])

        if direction is None:
            logger.debug('%s: no signal — ROC_5d=%.3f within thresholds',
                         self.SIGNAL_ID, roc_5d)
            return None

        # ── Build metadata ───────────────────────────────────────
        metadata = {
            'usdinr_current': round(usdinr_current, 4),
            'roc_5d': round(roc_5d, 4),
            'roc_20d': round(roc_20d, 4) if roc_20d is not None else None,
            'lookback_5d': LOOKBACK_5D,
            'lookback_20d': LOOKBACK_20D,
            'lag_note': 'FX moves lead equity by 1-3 sessions',
        }

        logger.info(
            '%s signal: %s on %s | strength=%.3f | USDINR=%.4f | ROC_5d=%.3f%%',
            self.SIGNAL_ID, direction, eval_date, strength, usdinr_current, roc_5d,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(nifty_price, 2),
            'reason': ' | '.join(reason_parts),
            'metadata': metadata,
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        pass

    def __repr__(self) -> str:
        return f"UsdInrMomentumSignal(signal_id='{self.SIGNAL_ID}')"

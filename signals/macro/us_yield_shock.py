"""
US 10-Year Yield Shock Signal.

Tracks sudden moves in the US 10-year Treasury yield as a leading
indicator for emerging market (EM) equity risk appetite.  Sharp yield
spikes trigger EM capital flight (bearish Nifty); yield drops signal
risk-on rotation into EM (bullish Nifty).

Signal logic:
    daily_chg_bp = (us10y_today - us10y_yesterday) * 100  (basis points)
    cum_5d_bp    = (us10y_today - us10y_5d_ago) * 100

    daily_chg_bp > +10 bp   -> SHORT bias (EM selloff risk)
    daily_chg_bp < -10 bp   -> LONG  bias (risk-on)
    cum_5d_bp    > +25 bp   -> sustained tightening overlay (amplifies SHORT)

    Strength scales with magnitude of daily move.

Data requirements:
    - Column 'us10y_yield' in df (yield in percent, e.g. 4.25)
    - Column 'close' for Nifty price

Walk-forward parameters:
    DAILY_CHG_SHORT_BP, DAILY_CHG_LONG_BP, CUM_5D_TIGHTENING_BP,
    LOOKBACK_1D, LOOKBACK_5D, STRENGTH_SCALE
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

SIGNAL_ID = 'US_YIELD_SHOCK'

# Lookback windows
LOOKBACK_1D = 1
LOOKBACK_5D = 5

# Thresholds in basis points
DAILY_CHG_SHORT_BP = 10.0       # Daily rise > 10bp → SHORT
DAILY_CHG_LONG_BP = -10.0       # Daily fall > 10bp → LONG
CUM_5D_TIGHTENING_BP = 25.0    # 5-day cumulative rise > 25bp = sustained tightening

# Strength scaling
STRENGTH_SCALE = 20.0           # Divide bp move by this for strength
SUSTAINED_TIGHTENING_BOOST = 0.15
MIN_STRENGTH = 0.3
MAX_STRENGTH = 1.0

# Minimum data points
MIN_ROWS = LOOKBACK_5D + 1

# Column names
COL_US10Y = 'us10y_yield'
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


# ================================================================
# SIGNAL CLASS
# ================================================================

class UsYieldShockSignal:
    """
    US 10-Year yield shock signal for Nifty.

    Sudden yield spikes → EM selloff risk → SHORT Nifty.
    Sudden yield drops  → risk-on         → LONG  Nifty.
    Sustained tightening (5-day cumulative) amplifies SHORT strength.
    """

    SIGNAL_ID = SIGNAL_ID

    # Walk-forward parameters
    WF_DAILY_CHG_SHORT_BP = DAILY_CHG_SHORT_BP
    WF_DAILY_CHG_LONG_BP = DAILY_CHG_LONG_BP
    WF_CUM_5D_TIGHTENING_BP = CUM_5D_TIGHTENING_BP
    WF_LOOKBACK_5D = LOOKBACK_5D

    def __init__(self) -> None:
        logger.info('UsYieldShockSignal initialised')

    # ----------------------------------------------------------
    # evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, date: date) -> Optional[Dict]:
        """
        Evaluate US 10Y yield shock signal.

        Parameters
        ----------
        df   : DataFrame with column 'us10y_yield' and 'close'.
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
        # ── Check required columns ───────────────────────────────
        if COL_US10Y not in df.columns:
            logger.debug('%s: column %s not found', self.SIGNAL_ID, COL_US10Y)
            return None

        if COL_NIFTY_CLOSE not in df.columns:
            logger.debug('%s: column %s not found', self.SIGNAL_ID, COL_NIFTY_CLOSE)
            return None

        # ── Slice data up to eval_date ───────────────────────────
        if hasattr(df.index, 'date'):
            mask = df.index.date <= eval_date
        elif 'date' in df.columns:
            mask = pd.to_datetime(df['date']).dt.date <= eval_date
        else:
            mask = pd.Series([True] * len(df), index=df.index)

        subset = df.loc[mask].copy()

        if len(subset) < 2:
            logger.debug('%s: insufficient data (%d rows)', self.SIGNAL_ID, len(subset))
            return None

        # ── Extract yield series ─────────────────────────────────
        yields = subset[COL_US10Y].dropna()
        if len(yields) < 2:
            return None

        # ── Compute daily change (basis points) ──────────────────
        y_today = _safe_float(yields.iloc[-1])
        y_yesterday = _safe_float(yields.iloc[-2])

        if math.isnan(y_today) or math.isnan(y_yesterday):
            return None

        daily_chg_bp = (y_today - y_yesterday) * 100.0

        # ── Compute 5-day cumulative change ──────────────────────
        cum_5d_bp = None
        if len(yields) >= MIN_ROWS:
            y_5d_ago = _safe_float(yields.iloc[-(LOOKBACK_5D + 1)])
            if not math.isnan(y_5d_ago):
                cum_5d_bp = (y_today - y_5d_ago) * 100.0

        # ── Nifty price ──────────────────────────────────────────
        nifty_price = _safe_float(subset[COL_NIFTY_CLOSE].iloc[-1])
        if math.isnan(nifty_price) or nifty_price <= 0:
            return None

        # ── Signal logic ─────────────────────────────────────────
        direction = None
        strength = 0.0
        reason_parts = [self.SIGNAL_ID]

        # SHORT: yield spike
        if daily_chg_bp > DAILY_CHG_SHORT_BP:
            direction = 'SHORT'
            raw_strength = abs(daily_chg_bp) / STRENGTH_SCALE
            # Boost if sustained tightening
            if cum_5d_bp is not None and cum_5d_bp > CUM_5D_TIGHTENING_BP:
                raw_strength += SUSTAINED_TIGHTENING_BOOST
                reason_parts.append(f"Sustained_tightening_5d={cum_5d_bp:+.1f}bp")
            strength = min(MAX_STRENGTH, max(MIN_STRENGTH, raw_strength))
            reason_parts.extend([
                f"US10Y_daily_chg={daily_chg_bp:+.1f}bp",
                f"US10Y={y_today:.3f}%",
                "Yield spike -> EM selloff risk -> bearish",
            ])

        # LONG: yield drop
        elif daily_chg_bp < DAILY_CHG_LONG_BP:
            direction = 'LONG'
            raw_strength = abs(daily_chg_bp) / STRENGTH_SCALE
            strength = min(MAX_STRENGTH, max(MIN_STRENGTH, raw_strength))
            reason_parts.extend([
                f"US10Y_daily_chg={daily_chg_bp:+.1f}bp",
                f"US10Y={y_today:.3f}%",
                "Yield drop -> risk-on -> bullish",
            ])

        if direction is None:
            logger.debug('%s: no signal — daily_chg=%.1fbp within thresholds',
                         self.SIGNAL_ID, daily_chg_bp)
            return None

        # ── Build metadata ───────────────────────────────────────
        metadata = {
            'us10y_yield': round(y_today, 4),
            'us10y_prev': round(y_yesterday, 4),
            'daily_chg_bp': round(daily_chg_bp, 2),
            'cum_5d_bp': round(cum_5d_bp, 2) if cum_5d_bp is not None else None,
            'sustained_tightening': cum_5d_bp is not None and cum_5d_bp > CUM_5D_TIGHTENING_BP,
        }

        logger.info(
            '%s signal: %s on %s | strength=%.3f | US10Y=%.3f%% | daily_chg=%+.1fbp',
            self.SIGNAL_ID, direction, eval_date, strength, y_today, daily_chg_bp,
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
        return f"UsYieldShockSignal(signal_id='{self.SIGNAL_ID}')"

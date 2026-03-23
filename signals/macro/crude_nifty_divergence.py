"""
Crude Oil / Nifty Divergence Signal.

Tracks divergence between Brent crude oil and Nifty returns.  India
imports ~85% of its crude oil, so oil price surges directly impact
the fiscal deficit, OMC (oil marketing companies) margins, and
corporate input costs.

Signal logic:
    crude_10d_ret = 10-day return of Brent crude
    nifty_10d_ret = 10-day return of Nifty

    Divergence:
        crude_10d_ret > +5% AND nifty flat/up  -> SHORT
            (OMC drag / margin compression incoming)
    Convergence:
        crude_10d_ret < -5% AND nifty flat     -> LONG
            (tailwind for Indian corporates)

    Structural headwind overlay:
        crude_close > 90 (USD/bbl) -> persistent bearish bias

    Strength scales with magnitude of crude move and divergence.

Data requirements:
    - Column 'crude_close' in df (Brent crude, USD/bbl)
    - Column 'close' for Nifty price

Walk-forward parameters:
    RETURN_WINDOW, CRUDE_UP_THRESH, CRUDE_DOWN_THRESH, NIFTY_FLAT_BAND,
    STRUCTURAL_HEADWIND_LEVEL
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

SIGNAL_ID = 'CRUDE_NIFTY_DIVERGENCE'

# Return window
RETURN_WINDOW = 10              # 10-day returns for comparison

# Crude thresholds
CRUDE_UP_THRESH = 5.0           # Crude up > 5% in 10d → stress
CRUDE_DOWN_THRESH = -5.0        # Crude down > 5% in 10d → tailwind

# Nifty "flat" band — treat as flat if within this range
NIFTY_FLAT_BAND = 2.0           # |nifty_10d_ret| < 2% = "flat"

# Structural headwind
STRUCTURAL_HEADWIND_LEVEL = 90.0  # Crude > $90/bbl → structural drag

# Strength scaling
STRENGTH_BASE = 0.45
STRENGTH_PER_PCT_CRUDE = 0.03   # Per 1% of crude move beyond threshold
HEADWIND_OVERLAY_BOOST = 0.10
MIN_STRENGTH = 0.3
MAX_STRENGTH = 1.0

# Minimum data points
MIN_ROWS = RETURN_WINDOW + 2

# Column names
COL_CRUDE = 'crude_close'
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


def _pct_return(series: pd.Series, period: int) -> Optional[float]:
    """Compute percentage return over `period` rows."""
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

class CrudeNiftyDivergenceSignal:
    """
    Crude oil / Nifty divergence signal.

    Crude rising sharply while Nifty holds → OMC drag incoming → SHORT.
    Crude falling sharply while Nifty flat  → tailwind coming   → LONG.
    Crude > $90/bbl → structural headwind overlay for India.
    India imports 85% of crude — direct fiscal and corporate margin impact.
    """

    SIGNAL_ID = SIGNAL_ID

    # Walk-forward parameters
    WF_RETURN_WINDOW = RETURN_WINDOW
    WF_CRUDE_UP_THRESH = CRUDE_UP_THRESH
    WF_CRUDE_DOWN_THRESH = CRUDE_DOWN_THRESH
    WF_NIFTY_FLAT_BAND = NIFTY_FLAT_BAND
    WF_STRUCTURAL_HEADWIND_LEVEL = STRUCTURAL_HEADWIND_LEVEL

    def __init__(self) -> None:
        logger.info('CrudeNiftyDivergenceSignal initialised')

    # ----------------------------------------------------------
    # evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, date: date) -> Optional[Dict]:
        """
        Evaluate Crude/Nifty divergence signal.

        Parameters
        ----------
        df   : DataFrame with columns 'crude_close' and 'close'.
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
        if COL_CRUDE not in df.columns:
            logger.debug('%s: column %s not found', self.SIGNAL_ID, COL_CRUDE)
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

        if len(subset) < MIN_ROWS:
            logger.debug('%s: insufficient data (%d < %d)', self.SIGNAL_ID,
                         len(subset), MIN_ROWS)
            return None

        # ── Extract series ───────────────────────────────────────
        crude = subset[COL_CRUDE].dropna()
        nifty = subset[COL_NIFTY_CLOSE].dropna()

        if len(crude) < MIN_ROWS or len(nifty) < MIN_ROWS:
            return None

        # ── Compute 10-day returns ───────────────────────────────
        crude_ret = _pct_return(crude, RETURN_WINDOW)
        nifty_ret = _pct_return(nifty, RETURN_WINDOW)

        if crude_ret is None or nifty_ret is None:
            return None

        # ── Current prices ───────────────────────────────────────
        nifty_price = _safe_float(nifty.iloc[-1])
        crude_price = _safe_float(crude.iloc[-1])

        if math.isnan(nifty_price) or nifty_price <= 0:
            return None
        if math.isnan(crude_price) or crude_price <= 0:
            return None

        # ── Structural headwind check ────────────────────────────
        structural_headwind = crude_price > STRUCTURAL_HEADWIND_LEVEL

        # ── Signal logic ─────────────────────────────────────────
        direction = None
        strength = 0.0
        reason_parts = [self.SIGNAL_ID]
        nifty_is_flat = abs(nifty_ret) < NIFTY_FLAT_BAND

        # Divergence: crude surging, Nifty flat or up → OMC drag incoming
        if crude_ret > CRUDE_UP_THRESH and (nifty_is_flat or nifty_ret > 0):
            direction = 'SHORT'
            excess_crude = crude_ret - CRUDE_UP_THRESH
            raw_strength = STRENGTH_BASE + excess_crude * STRENGTH_PER_PCT_CRUDE
            if structural_headwind:
                raw_strength += HEADWIND_OVERLAY_BOOST
                reason_parts.append(f"STRUCTURAL: crude=${crude_price:.1f} > ${STRUCTURAL_HEADWIND_LEVEL}")
            strength = min(MAX_STRENGTH, max(MIN_STRENGTH, raw_strength))
            reason_parts.extend([
                f"Crude_10d={crude_ret:+.2f}%",
                f"Nifty_10d={nifty_ret:+.2f}%",
                "Divergence: crude surging, Nifty hasn't repriced -> OMC drag incoming",
                "India imports 85% crude -> fiscal deficit + margin compression",
            ])

        # Convergence: crude dropping, Nifty flat → tailwind coming
        elif crude_ret < CRUDE_DOWN_THRESH and nifty_is_flat:
            direction = 'LONG'
            excess_drop = abs(crude_ret) - abs(CRUDE_DOWN_THRESH)
            raw_strength = STRENGTH_BASE + excess_drop * STRENGTH_PER_PCT_CRUDE
            strength = min(MAX_STRENGTH, max(MIN_STRENGTH, raw_strength))
            reason_parts.extend([
                f"Crude_10d={crude_ret:+.2f}%",
                f"Nifty_10d={nifty_ret:+.2f}%",
                "Convergence: crude dropping, Nifty flat -> tailwind incoming",
                "Lower crude -> fiscal relief + improved corporate margins",
            ])

        # Structural headwind only (no divergence, but crude very high)
        elif structural_headwind and direction is None:
            # Only fire if crude is significantly high — this is an overlay
            if crude_price > STRUCTURAL_HEADWIND_LEVEL * 1.05:
                direction = 'SHORT'
                strength = MIN_STRENGTH  # Weak overlay signal
                reason_parts.extend([
                    f"Crude=${crude_price:.1f}/bbl",
                    f"STRUCTURAL headwind: > ${STRUCTURAL_HEADWIND_LEVEL} threshold",
                    "Persistent fiscal deficit drag at elevated crude levels",
                ])

        if direction is None:
            logger.debug(
                '%s: no signal — crude_10d=%.2f%% nifty_10d=%.2f%% crude=$%.1f',
                self.SIGNAL_ID, crude_ret, nifty_ret, crude_price,
            )
            return None

        # ── Build metadata ───────────────────────────────────────
        metadata = {
            'crude_price': round(crude_price, 2),
            'crude_10d_return': round(crude_ret, 4),
            'nifty_10d_return': round(nifty_ret, 4),
            'structural_headwind': structural_headwind,
            'return_window': RETURN_WINDOW,
            'divergence_type': 'crude_up_nifty_flat' if crude_ret > 0 else 'crude_down_nifty_flat',
        }

        logger.info(
            '%s signal: %s on %s | strength=%.3f | crude=$%.1f (%+.1f%%) | nifty %+.1f%%',
            self.SIGNAL_ID, direction, eval_date, strength,
            crude_price, crude_ret, nifty_ret,
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
        return f"CrudeNiftyDivergenceSignal(signal_id='{self.SIGNAL_ID}')"

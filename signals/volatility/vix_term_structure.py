"""
VIX Term Structure Inversion Signal (Overlay).

Compares near-month India VIX with its term structure (next-month VIX
futures or, if unavailable, the 20-day SMA of VIX as a proxy for the
"normal" level).

Signal logic:
    Backwardation (fear):  VIX > 1.2 × SMA20_VIX  → LONG overlay (1.3x size)
    Contango (complacency): VIX < 0.8 × SMA20_VIX → reduce size (0.5x)
    Normal range:           otherwise               → neutral (1.0x)

This is an OVERLAY signal — it modifies position size rather than
generating standalone entry/exit signals.  The strength field encodes
the size multiplier in the range [0.5, 1.3].

If both india_vix and india_vix_fut columns are present, the signal
uses the actual term structure.  Otherwise, it falls back to the
SMA-based proxy.
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

SIGNAL_ID = 'VIX_TERM_STRUCTURE'

# Rolling window for SMA proxy
SMA_WINDOW = 20

# Term structure thresholds (ratio of near to far/SMA)
BACKWARDATION_RATIO = 1.20   # near > 1.2 × far → backwardation (fear)
CONTANGO_RATIO = 0.80        # near < 0.8 × far → contango (complacency)

# Size modifiers (overlay output)
SIZE_BACKWARDATION = 1.30    # Scale up on fear
SIZE_CONTANGO = 0.50         # Scale down on complacency
SIZE_NEUTRAL = 1.00

# Columns
VIX_COLUMN = 'india_vix'
VIX_FUT_COLUMN = 'india_vix_fut'
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


# ================================================================
# SIGNAL CLASS
# ================================================================

class VixTermStructureSignal:
    """
    VIX term structure overlay signal.

    Outputs a size modifier (0.5x to 1.3x) based on whether the VIX
    term structure is in backwardation (fear) or contango (complacency).
    """

    SIGNAL_ID = SIGNAL_ID

    # Walk-forward parameters
    WF_SMA_WINDOW = SMA_WINDOW
    WF_BACKWARDATION_RATIO = BACKWARDATION_RATIO
    WF_CONTANGO_RATIO = CONTANGO_RATIO
    WF_SIZE_BACKWARDATION = SIZE_BACKWARDATION
    WF_SIZE_CONTANGO = SIZE_CONTANGO

    def __init__(self) -> None:
        logger.info('VixTermStructureSignal initialised')

    def evaluate(self, df: pd.DataFrame, date_val: date) -> Optional[Dict]:
        """
        Evaluate VIX term structure overlay.

        Parameters
        ----------
        df       : DataFrame with VIX history.  Must contain 'india_vix'
                   and 'close'.  Optionally 'india_vix_fut' for actual
                   term structure.
        date_val : The evaluation date.

        Returns
        -------
        dict with signal_id, direction, strength (=size_modifier),
        price, reason, metadata.  Or None if insufficient data.
        """
        try:
            return self._evaluate_inner(df, date_val)
        except Exception as e:
            logger.error(
                'VixTermStructureSignal.evaluate error: %s', e, exc_info=True
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

        if len(df) < SMA_WINDOW:
            logger.debug(
                'Insufficient data: %d rows < %d required',
                len(df), SMA_WINDOW,
            )
            return None

        # ── Current VIX (near-month proxy) ──────────────────────────
        vix_series = df[VIX_COLUMN].astype(float)
        vix_near = _safe_float(vix_series.iloc[-1])

        if math.isnan(vix_near) or vix_near <= 0:
            logger.debug('Invalid current VIX: %s', vix_near)
            return None

        # ── Determine far-month or SMA proxy ────────────────────────
        use_futures = (
            VIX_FUT_COLUMN in df.columns
            and not pd.isna(df[VIX_FUT_COLUMN].iloc[-1])
        )

        if use_futures:
            vix_far = _safe_float(df[VIX_FUT_COLUMN].iloc[-1])
            if math.isnan(vix_far) or vix_far <= 0:
                use_futures = False

        if use_futures:
            reference = vix_far
            ref_label = f"VIX_Fut={vix_far:.1f}"
        else:
            # Fallback: use 20-day SMA of VIX as "normal" level
            vix_window = vix_series.iloc[-SMA_WINDOW:]
            valid_count = vix_window.dropna().shape[0]
            if valid_count < SMA_WINDOW * 0.8:
                logger.debug('Too many NaN in VIX window')
                return None
            reference = float(vix_window.mean())
            ref_label = f"SMA20_VIX={reference:.1f}"

        if reference <= 0:
            logger.debug('Invalid reference VIX level')
            return None

        # ── Compute ratio ──────────────────────────────────────────
        ratio = vix_near / reference

        # ── Determine regime and size modifier ─────────────────────
        if ratio > BACKWARDATION_RATIO:
            regime = 'BACKWARDATION'
            direction = 'LONG'
            size_modifier = SIZE_BACKWARDATION
        elif ratio < CONTANGO_RATIO:
            regime = 'CONTANGO'
            direction = 'SHORT'
            size_modifier = SIZE_CONTANGO
        else:
            regime = 'NEUTRAL'
            direction = 'NEUTRAL'
            size_modifier = SIZE_NEUTRAL

        # ── Strength = size modifier normalised to 0-1 ────────────
        # Map [0.5, 1.3] → [0, 1] for consistency
        strength = min(1.0, max(0.0, (size_modifier - 0.5) / 0.8))

        # ── Price ──────────────────────────────────────────────────
        price = _safe_float(df[CLOSE_COLUMN].iloc[-1])
        if math.isnan(price) or price <= 0:
            logger.debug('Invalid close price')
            return None

        # ── Reason ─────────────────────────────────────────────────
        reason = (
            f"VIX_TERM_STRUCTURE | Regime={regime} | "
            f"VIX={vix_near:.1f} | {ref_label} | "
            f"Ratio={ratio:.3f} | SizeMod={size_modifier:.2f}x"
        )

        logger.info(
            '%s signal: %s regime=%s ratio=%.3f size_mod=%.2f on %s',
            self.SIGNAL_ID, direction, regime, ratio, size_modifier, date_val,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2),
            'reason': reason,
            'metadata': {
                'regime': regime,
                'vix_near': round(vix_near, 2),
                'reference_level': round(reference, 2),
                'ratio': round(ratio, 4),
                'size_modifier': round(size_modifier, 2),
                'use_futures': use_futures,
                'date': str(date_val),
            },
        }

    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        pass

    def __repr__(self) -> str:
        return f"VixTermStructureSignal(signal_id='{self.SIGNAL_ID}')"

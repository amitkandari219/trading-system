"""
Gold/Nifty Ratio Extremes Signal.

Computes the ratio of Gold price (INR) to Nifty close and monitors its
z-score over a 60-day rolling window.  A high ratio (gold outperforming
equities) signals risk-off sentiment; a low ratio (equities outperforming
gold) signals risk-on momentum.

Signal logic:
    ratio   = gold_inr / nifty_close
    z_score = (ratio - mean_60d) / std_60d

    z > +1.5  -> SHORT (risk-off, gold outperforming)
    z < -1.5  -> LONG  (risk-on, equities outperforming)
    |z| > 2.5 -> contrarian layer (extreme ratios mean-revert)

    Strength scales with |z-score|.

Data requirements:
    - Column 'gold_inr' or 'mcx_gold' in df
    - Column 'close' for Nifty price

Walk-forward parameters:
    ZSCORE_WINDOW, Z_SHORT_THRESH, Z_LONG_THRESH, Z_CONTRARIAN_THRESH
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

SIGNAL_ID = 'GOLD_NIFTY_RATIO'

# Z-score parameters
ZSCORE_WINDOW = 60              # Rolling window for z-score
Z_SHORT_THRESH = 1.5            # z > 1.5 → risk-off → SHORT
Z_LONG_THRESH = -1.5            # z < -1.5 → risk-on → LONG
Z_CONTRARIAN_THRESH = 2.5       # |z| > 2.5 → extreme, expect mean reversion

# Strength scaling
STRENGTH_BASE = 0.4
STRENGTH_PER_Z = 0.15           # Additional strength per unit of z
CONTRARIAN_PENALTY = -0.10      # Reduce strength at extremes (contrarian caution)
MIN_STRENGTH = 0.3
MAX_STRENGTH = 1.0

# Minimum data points
MIN_ROWS = ZSCORE_WINDOW + 5

# Column names (try gold_inr first, then mcx_gold)
COL_GOLD_PRIMARY = 'gold_inr'
COL_GOLD_FALLBACK = 'mcx_gold'
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


def _resolve_gold_column(df: pd.DataFrame) -> Optional[str]:
    """Find the gold price column, trying primary then fallback."""
    if COL_GOLD_PRIMARY in df.columns:
        return COL_GOLD_PRIMARY
    if COL_GOLD_FALLBACK in df.columns:
        return COL_GOLD_FALLBACK
    return None


# ================================================================
# SIGNAL CLASS
# ================================================================

class GoldNiftyRatioSignal:
    """
    Gold/Nifty ratio extremes signal.

    High gold/equity ratio (z > 1.5) → risk-off → SHORT Nifty.
    Low gold/equity ratio (z < -1.5) → risk-on  → LONG  Nifty.
    Extreme ratios (|z| > 2.5) carry a contrarian mean-reversion bias.
    """

    SIGNAL_ID = SIGNAL_ID

    # Walk-forward parameters
    WF_ZSCORE_WINDOW = ZSCORE_WINDOW
    WF_Z_SHORT_THRESH = Z_SHORT_THRESH
    WF_Z_LONG_THRESH = Z_LONG_THRESH
    WF_Z_CONTRARIAN_THRESH = Z_CONTRARIAN_THRESH

    def __init__(self) -> None:
        logger.info('GoldNiftyRatioSignal initialised')

    # ----------------------------------------------------------
    # evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, date: date) -> Optional[Dict]:
        """
        Evaluate Gold/Nifty ratio signal.

        Parameters
        ----------
        df   : DataFrame with gold price column and 'close'.
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
        # ── Resolve gold column ──────────────────────────────────
        gold_col = _resolve_gold_column(df)
        if gold_col is None:
            logger.debug('%s: no gold column found (%s / %s)',
                         self.SIGNAL_ID, COL_GOLD_PRIMARY, COL_GOLD_FALLBACK)
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

        # ── Compute ratio ────────────────────────────────────────
        gold = subset[gold_col].astype(float)
        nifty = subset[COL_NIFTY_CLOSE].astype(float)

        # Drop rows where either is NaN or zero
        valid = (gold > 0) & (nifty > 0) & gold.notna() & nifty.notna()
        gold = gold[valid]
        nifty = nifty[valid]

        if len(gold) < MIN_ROWS:
            logger.debug('%s: insufficient valid ratio data', self.SIGNAL_ID)
            return None

        ratio = gold / nifty

        # ── Z-score over rolling window ──────────────────────────
        rolling_mean = ratio.rolling(window=ZSCORE_WINDOW).mean()
        rolling_std = ratio.rolling(window=ZSCORE_WINDOW).std()

        current_ratio = _safe_float(ratio.iloc[-1])
        current_mean = _safe_float(rolling_mean.iloc[-1])
        current_std = _safe_float(rolling_std.iloc[-1])

        if math.isnan(current_ratio) or math.isnan(current_mean) or math.isnan(current_std):
            return None

        if current_std <= 0:
            logger.debug('%s: zero std — cannot compute z-score', self.SIGNAL_ID)
            return None

        z_score = (current_ratio - current_mean) / current_std

        # ── Nifty price ──────────────────────────────────────────
        nifty_price = _safe_float(nifty.iloc[-1])
        if math.isnan(nifty_price) or nifty_price <= 0:
            return None

        gold_price = _safe_float(gold.iloc[-1])

        # ── Signal logic ─────────────────────────────────────────
        direction = None
        strength = 0.0
        is_contrarian = abs(z_score) > Z_CONTRARIAN_THRESH
        reason_parts = [self.SIGNAL_ID]

        if z_score > Z_SHORT_THRESH:
            # Risk-off: gold outperforming → SHORT Nifty
            direction = 'SHORT'
            raw_strength = STRENGTH_BASE + abs(z_score) * STRENGTH_PER_Z
            if is_contrarian:
                # Extreme → expect mean reversion, reduce conviction
                raw_strength += CONTRARIAN_PENALTY
                reason_parts.append("CONTRARIAN: extreme ratio likely to revert")
            strength = min(MAX_STRENGTH, max(MIN_STRENGTH, raw_strength))
            reason_parts.extend([
                f"Gold/Nifty_ratio={current_ratio:.6f}",
                f"Z-score={z_score:+.2f}",
                "Risk-off regime: gold outperforming equities -> bearish",
            ])

        elif z_score < Z_LONG_THRESH:
            # Risk-on: equities outperforming gold → LONG Nifty
            direction = 'LONG'
            raw_strength = STRENGTH_BASE + abs(z_score) * STRENGTH_PER_Z
            if is_contrarian:
                raw_strength += CONTRARIAN_PENALTY
                reason_parts.append("CONTRARIAN: extreme ratio likely to revert")
            strength = min(MAX_STRENGTH, max(MIN_STRENGTH, raw_strength))
            reason_parts.extend([
                f"Gold/Nifty_ratio={current_ratio:.6f}",
                f"Z-score={z_score:+.2f}",
                "Risk-on regime: equities outperforming gold -> bullish",
            ])

        if direction is None:
            logger.debug('%s: no signal — z=%.3f within thresholds [%.1f, %.1f]',
                         self.SIGNAL_ID, z_score, Z_LONG_THRESH, Z_SHORT_THRESH)
            return None

        # ── Build metadata ───────────────────────────────────────
        metadata = {
            'gold_price': round(gold_price, 2),
            'gold_column_used': gold_col,
            'ratio': round(current_ratio, 6),
            'z_score': round(z_score, 4),
            'rolling_mean': round(current_mean, 6),
            'rolling_std': round(current_std, 6),
            'zscore_window': ZSCORE_WINDOW,
            'is_contrarian_extreme': is_contrarian,
        }

        logger.info(
            '%s signal: %s on %s | strength=%.3f | z=%.3f | ratio=%.6f',
            self.SIGNAL_ID, direction, eval_date, strength, z_score, current_ratio,
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
        return f"GoldNiftyRatioSignal(signal_id='{self.SIGNAL_ID}')"

"""
Bid-Ask Spread Regime Signal.

Tracks average bid-ask spread as a percentage of price to classify
the current liquidity regime.  This is not a directional signal — it
modifies position sizing and stop placement for other signals.

Signal logic:
    avg_spread_pct < 0.02%  -> HIGH liquidity   (full size, tight stops ok)
    avg_spread_pct > 0.05%  -> LOW liquidity    (50% size, widen stops, avoid entries)
    else                    -> NORMAL liquidity  (standard sizing)

Column required: avg_spread_pct (float, in percent e.g. 0.03 = 0.03%)
                 OR bid/ask columns to compute from.
"""

import logging
import math
from datetime import date
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

SIGNAL_ID = 'BID_ASK_REGIME'

# Spread thresholds (in percent of price)
TIGHT_SPREAD_THRESHOLD = 0.02    # < 0.02% = high liquidity
WIDE_SPREAD_THRESHOLD = 0.05     # > 0.05% = low liquidity

# Size modifiers
HIGH_LIQ_SIZE = 1.0
NORMAL_LIQ_SIZE = 0.85
LOW_LIQ_SIZE = 0.50

# Stop width multipliers
HIGH_LIQ_STOP = 1.0
NORMAL_LIQ_STOP = 1.2
LOW_LIQ_STOP = 1.5

SPREAD_COL = 'avg_spread_pct'


# ================================================================
# HELPERS
# ================================================================

def _safe_float(val: Any, default: float = float('nan')) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _compute_spread_pct(row: pd.Series) -> float:
    """Compute spread % from bid/ask if available."""
    bid = _safe_float(row.get('bid') or row.get('best_bid'))
    ask = _safe_float(row.get('ask') or row.get('best_ask'))
    if math.isnan(bid) or math.isnan(ask) or bid <= 0 or ask <= 0:
        return float('nan')
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return float('nan')
    return ((ask - bid) / mid) * 100.0


# ================================================================
# SIGNAL CLASS
# ================================================================

class BidAskRegimeSignal:
    """
    Classifies the current liquidity regime from bid-ask spread data.
    Outputs a size modifier and stop-width adjustment, not a direction.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('BidAskRegimeSignal initialised')

    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate bid-ask spread regime.

        Parameters
        ----------
        df         : DataFrame with `avg_spread_pct` column (or bid/ask).
        trade_date : The date to evaluate.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata
        or None.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('BidAskRegimeSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if df is None or df.empty:
            return None

        # ── Align to trade_date ──────────────────────────────────
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            else:
                return None

        df = df.sort_index()
        mask = df.index.date <= trade_date
        df_hist = df.loc[mask]

        if df_hist.empty:
            return None

        row = df_hist.iloc[-1]

        # ── Get spread pct ───────────────────────────────────────
        spread_pct = float('nan')

        if SPREAD_COL in df_hist.columns:
            spread_pct = _safe_float(row.get(SPREAD_COL))

        # Fallback: compute from bid/ask
        if math.isnan(spread_pct):
            spread_pct = _compute_spread_pct(row)

        if math.isnan(spread_pct) or spread_pct < 0:
            logger.debug('Invalid spread_pct: %s', spread_pct)
            return None

        # ── Get price ────────────────────────────────────────────
        price = float('nan')
        for col in ('close', 'Close', 'price', 'nifty_close'):
            if col in df_hist.columns:
                price = _safe_float(df_hist[col].iloc[-1])
                break

        # ── Classify regime ──────────────────────────────────────
        if spread_pct < TIGHT_SPREAD_THRESHOLD:
            regime = 'HIGH'
            size_modifier = HIGH_LIQ_SIZE
            stop_multiplier = HIGH_LIQ_STOP
        elif spread_pct > WIDE_SPREAD_THRESHOLD:
            regime = 'LOW'
            size_modifier = LOW_LIQ_SIZE
            stop_multiplier = LOW_LIQ_STOP
        else:
            regime = 'NORMAL'
            size_modifier = NORMAL_LIQ_SIZE
            stop_multiplier = NORMAL_LIQ_STOP

        # Direction is NEUTRAL — this is a regime/filter signal
        direction = 'NEUTRAL'
        # Strength = 0 for NORMAL, higher for extremes
        if regime == 'LOW':
            strength = min(0.80, 0.3 + (spread_pct - WIDE_SPREAD_THRESHOLD) * 5.0)
        elif regime == 'HIGH':
            strength = min(0.60, 0.2 + (TIGHT_SPREAD_THRESHOLD - spread_pct) * 10.0)
        else:
            strength = 0.0

        strength = max(0.0, strength)

        reason_parts = [
            SIGNAL_ID,
            f"spread={spread_pct:.4f}%",
            f"regime={regime}_LIQUIDITY",
            f"size_mod={size_modifier}x",
            f"stop_width={stop_multiplier}x",
        ]

        logger.info('%s signal: regime=%s on %s spread=%.4f%%',
                     self.SIGNAL_ID, regime, trade_date, spread_pct)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2) if not math.isnan(price) else None,
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'avg_spread_pct': round(spread_pct, 6),
                'liquidity_regime': regime,
                'size_modifier': size_modifier,
                'stop_width_multiplier': stop_multiplier,
                'trade_date': trade_date.isoformat(),
            },
        }

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"BidAskRegimeSignal(signal_id='{self.SIGNAL_ID}')"

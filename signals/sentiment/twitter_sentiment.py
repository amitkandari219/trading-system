"""
Twitter/X FinTwit India Sentiment Signal.

Aggregates sentiment from Indian financial Twitter.  This is a contrarian
indicator — the crowd is usually wrong at extremes.

Signal logic:
    score = twitter_sentiment_score (-1 to +1)

    score < -0.5  -> LONG   (extreme negative = panic = contrarian buy)
    score >  0.5  -> SHORT  (extreme positive = euphoria = contrarian sell)
    -0.2 to 0.2   -> no signal (neutral zone)
    otherwise      -> no signal (mild sentiment, not actionable)

    Strength scales with distance from neutral zone.

Column required: twitter_sentiment_score (float, -1 to +1)
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

SIGNAL_ID = 'TWITTER_SENTIMENT'

# Sentiment thresholds
EXTREME_NEGATIVE = -0.5
EXTREME_POSITIVE = 0.5
NEUTRAL_LOW = -0.2
NEUTRAL_HIGH = 0.2

# Strength scaling
MAX_STRENGTH = 0.90
MIN_STRENGTH = 0.15

# Size modifiers
EXTREME_SIZE_BOOST = 1.15
BASE_SIZE = 1.0

COLUMN = 'twitter_sentiment_score'


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


# ================================================================
# SIGNAL CLASS
# ================================================================

class TwitterSentimentSignal:
    """
    Contrarian signal from Indian financial Twitter sentiment.
    Extreme negative = retail panic = LONG; extreme positive = euphoria = SHORT.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('TwitterSentimentSignal initialised')

    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate Twitter sentiment signal.

        Parameters
        ----------
        df         : DataFrame with `twitter_sentiment_score` column.
        trade_date : The date to evaluate.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata
        or None.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('TwitterSentimentSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if df is None or df.empty:
            return None

        if COLUMN not in df.columns:
            logger.debug('Column %s not found', COLUMN)
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

        score = _safe_float(df_hist[COLUMN].iloc[-1])
        if math.isnan(score):
            return None

        # Clamp to [-1, 1]
        score = max(-1.0, min(1.0, score))

        # ── Get price ────────────────────────────────────────────
        price = float('nan')
        for col in ('close', 'Close', 'price', 'nifty_close'):
            if col in df_hist.columns:
                price = _safe_float(df_hist[col].iloc[-1])
                break

        # ── Signal logic ─────────────────────────────────────────
        direction = None
        strength = 0.0
        size_modifier = BASE_SIZE

        if score <= EXTREME_NEGATIVE:
            # Extreme panic -> contrarian LONG
            direction = 'LONG'
            # -0.5 -> 0.40 strength, -1.0 -> 0.90
            strength = min(MAX_STRENGTH, 0.40 + (abs(score) - 0.5) * 1.0)
            strength = max(MIN_STRENGTH, strength)
            size_modifier = EXTREME_SIZE_BOOST

        elif score >= EXTREME_POSITIVE:
            # Euphoria -> contrarian SHORT
            direction = 'SHORT'
            strength = min(MAX_STRENGTH, 0.40 + (score - 0.5) * 1.0)
            strength = max(MIN_STRENGTH, strength)
            size_modifier = EXTREME_SIZE_BOOST

        elif NEUTRAL_LOW <= score <= NEUTRAL_HIGH:
            # Dead zone — no signal
            return None

        else:
            # Mild sentiment (between neutral and extreme) — not actionable
            return None

        sentiment_label = 'PANIC' if direction == 'LONG' else 'EUPHORIA'
        reason_parts = [
            SIGNAL_ID,
            f"score={score:+.3f}",
            f"{sentiment_label}",
            f"contrarian_{direction}",
        ]

        logger.info('%s signal: %s on %s score=%.3f strength=%.3f',
                     self.SIGNAL_ID, direction, trade_date, score, strength)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2) if not math.isnan(price) else None,
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'sentiment_score': round(score, 4),
                'sentiment_label': sentiment_label,
                'size_modifier': size_modifier,
                'trade_date': trade_date.isoformat(),
            },
        }

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"TwitterSentimentSignal(signal_id='{self.SIGNAL_ID}')"

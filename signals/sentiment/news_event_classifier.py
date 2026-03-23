"""
News Event Regime Tagger Signal.

Classifies the current day's news environment into one of four regimes
and outputs a size modifier reflecting the expected volatility characteristics.

Regimes:
    EARNINGS_SEASON  : Higher vol, stock-specific — reduce index sizing (0.8x)
    POLICY_EVENT     : IV expansion pre-event — reduce sizing (0.7x)
    GEOPOLITICAL     : Gap risk, widen stops — reduce sizing (0.7x)
    NORMAL           : Standard conditions — full sizing (1.2x)

The signal does not produce a directional trade.  Instead it acts as a
regime overlay that modifies sizing and stop-width for other signals.

Column required: news_category (str: one of the above regimes)
                 OR derived from calendar/event data columns.
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

SIGNAL_ID = 'NEWS_EVENT_CLASSIFIER'

# Valid regimes and their characteristics
REGIME_CONFIG = {
    'EARNINGS_SEASON': {
        'size_modifier': 0.80,
        'stop_width_multiplier': 1.2,
        'description': 'Higher vol, stock-specific moves — reduce index sizing',
    },
    'POLICY_EVENT': {
        'size_modifier': 0.70,
        'stop_width_multiplier': 1.5,
        'description': 'IV expansion pre-event — reduce sizing, widen stops',
    },
    'GEOPOLITICAL': {
        'size_modifier': 0.70,
        'stop_width_multiplier': 1.5,
        'description': 'Gap risk — widen stops, reduce sizing',
    },
    'NORMAL': {
        'size_modifier': 1.20,
        'stop_width_multiplier': 1.0,
        'description': 'Standard conditions — full sizing',
    },
}

DEFAULT_REGIME = 'NORMAL'

COLUMN = 'news_category'

# Keyword-based fallback detection from headline/text columns
EARNINGS_KEYWORDS = ['earnings', 'results', 'quarterly', 'q1', 'q2', 'q3', 'q4',
                     'profit', 'revenue', 'guidance']
POLICY_KEYWORDS = ['rbi', 'policy', 'rate decision', 'monetary policy', 'budget',
                   'fomc', 'fed', 'fiscal']
GEO_KEYWORDS = ['war', 'conflict', 'sanctions', 'geopolitical', 'military',
                'attack', 'border', 'tensions', 'missile']


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


def _classify_from_text(text: str) -> str:
    """Fallback regime classification from headline text."""
    if not text or not isinstance(text, str):
        return DEFAULT_REGIME
    text_lower = text.lower()
    if any(kw in text_lower for kw in GEO_KEYWORDS):
        return 'GEOPOLITICAL'
    if any(kw in text_lower for kw in POLICY_KEYWORDS):
        return 'POLICY_EVENT'
    if any(kw in text_lower for kw in EARNINGS_KEYWORDS):
        return 'EARNINGS_SEASON'
    return DEFAULT_REGIME


# ================================================================
# SIGNAL CLASS
# ================================================================

class NewsEventClassifierSignal:
    """
    Classifies the news environment into a regime and outputs a
    size modifier and stop-width adjustment.  Not directional.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('NewsEventClassifierSignal initialised')

    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate news regime for the given date.

        Parameters
        ----------
        df         : DataFrame with `news_category` column (or text columns
                     like 'headline' for fallback classification).
        trade_date : The date to evaluate.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata
        or None if data missing entirely.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('NewsEventClassifierSignal.evaluate error: %s', e, exc_info=True)
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

        # ── Determine regime ─────────────────────────────────────
        regime = DEFAULT_REGIME

        if COLUMN in df_hist.columns:
            raw = row.get(COLUMN)
            if isinstance(raw, str) and raw.upper() in REGIME_CONFIG:
                regime = raw.upper()
            elif isinstance(raw, str):
                regime = _classify_from_text(raw)
        else:
            # Fallback: try headline/text column
            for text_col in ('headline', 'news_text', 'title', 'event'):
                if text_col in df_hist.columns:
                    regime = _classify_from_text(str(row.get(text_col, '')))
                    break

        config = REGIME_CONFIG[regime]

        # ── Get price ────────────────────────────────────────────
        price = float('nan')
        for col in ('close', 'Close', 'price', 'nifty_close'):
            if col in df_hist.columns:
                price = _safe_float(df_hist[col].iloc[-1])
                break

        # ── Build output ─────────────────────────────────────────
        # Direction is NEUTRAL — this is a regime overlay, not directional
        direction = 'NEUTRAL'
        # Strength reflects how much the regime deviates from NORMAL
        strength = 0.0 if regime == 'NORMAL' else 0.5

        reason_parts = [
            SIGNAL_ID,
            f"regime={regime}",
            f"size_mod={config['size_modifier']}x",
            f"stop_width={config['stop_width_multiplier']}x",
            config['description'],
        ]

        logger.info('%s signal: regime=%s on %s size_mod=%.2f',
                     self.SIGNAL_ID, regime, trade_date, config['size_modifier'])

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2) if not math.isnan(price) else None,
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'regime': regime,
                'size_modifier': config['size_modifier'],
                'stop_width_multiplier': config['stop_width_multiplier'],
                'regime_description': config['description'],
                'trade_date': trade_date.isoformat(),
            },
        }

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"NewsEventClassifierSignal(signal_id='{self.SIGNAL_ID}')"

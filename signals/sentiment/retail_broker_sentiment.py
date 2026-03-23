"""
Retail Broker Activity Proxy Signal.

Tracks retail participation via delivery percentage and DII flows.
Delivery percentage below 30% indicates speculative frenzy; above 50%
indicates genuine accumulation.  Combined with price relative to 50-SMA,
this identifies overheated and capitulation conditions.

Signal logic:
    delivery_pct < 25% AND close > sma_50  -> SHORT (overheated / froth)
    delivery_pct > 55% AND close < sma_50  -> LONG  (accumulation / capitulation buy)

    Intermediate zones:
    delivery_pct < 30% AND close > sma_50  -> mild SHORT
    delivery_pct > 50% AND close < sma_50  -> mild LONG

    DII net flows add confirmation when available.

Columns required: delivery_pct (float, 0-100), close (float)
Optional: dii_net (float, in crores)
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

SIGNAL_ID = 'RETAIL_BROKER_SENTIMENT'

# SMA period for trend context
SMA_PERIOD = 50

# Delivery percentage thresholds
EXTREME_LOW_DELIVERY = 25.0    # < 25% = extreme speculation
LOW_DELIVERY = 30.0            # < 30% = speculation dominant
HIGH_DELIVERY = 50.0           # > 50% = accumulation
EXTREME_HIGH_DELIVERY = 55.0   # > 55% = deep accumulation / capitulation

# Strength
MAX_STRENGTH = 0.85
MIN_STRENGTH = 0.15
BASE_STRENGTH_MILD = 0.30
BASE_STRENGTH_EXTREME = 0.60

# Size
OVERHEATED_SIZE = 0.9
ACCUMULATION_SIZE = 1.1
BASE_SIZE = 1.0

# DII confirmation boost
DII_CONFIRM_BOOST = 0.10

DELIVERY_COL = 'delivery_pct'
DII_COL = 'dii_net'


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

class RetailBrokerSentimentSignal:
    """
    Identifies overheated (retail speculation) and capitulation
    (accumulation) conditions using delivery percentage and DII flows.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('RetailBrokerSentimentSignal initialised')

    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate retail broker sentiment.

        Parameters
        ----------
        df         : DataFrame with `delivery_pct`, `close` columns.
                     Optional: `dii_net`.
        trade_date : The date to evaluate.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata
        or None.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('RetailBrokerSentimentSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if df is None or df.empty:
            return None

        if DELIVERY_COL not in df.columns:
            logger.debug('Column %s not found', DELIVERY_COL)
            return None

        # ── Need a close price column ────────────────────────────
        close_col = None
        for c in ('close', 'Close', 'nifty_close', 'price'):
            if c in df.columns:
                close_col = c
                break
        if close_col is None:
            logger.debug('No close price column found')
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

        if len(df_hist) < SMA_PERIOD + 1:
            logger.debug('Insufficient history: %d rows (need %d)',
                         len(df_hist), SMA_PERIOD + 1)
            return None

        # ── Compute SMA ──────────────────────────────────────────
        close_series = df_hist[close_col].astype(float)
        sma_50 = close_series.rolling(window=SMA_PERIOD).mean().iloc[-1]
        current_close = _safe_float(close_series.iloc[-1])
        delivery = _safe_float(df_hist[DELIVERY_COL].iloc[-1])

        if math.isnan(current_close) or math.isnan(sma_50) or math.isnan(delivery):
            return None

        # ── Optional DII data ────────────────────────────────────
        dii_net = float('nan')
        if DII_COL in df_hist.columns:
            dii_net = _safe_float(df_hist[DII_COL].iloc[-1])

        above_sma = current_close > sma_50
        below_sma = current_close < sma_50

        # ── Signal logic ─────────────────────────────────────────
        direction = None
        strength = 0.0
        size_modifier = BASE_SIZE
        label = ''

        if delivery < EXTREME_LOW_DELIVERY and above_sma:
            direction = 'SHORT'
            strength = BASE_STRENGTH_EXTREME
            size_modifier = OVERHEATED_SIZE
            label = 'EXTREME_OVERHEATED'

        elif delivery < LOW_DELIVERY and above_sma:
            direction = 'SHORT'
            strength = BASE_STRENGTH_MILD
            size_modifier = OVERHEATED_SIZE
            label = 'OVERHEATED'

        elif delivery > EXTREME_HIGH_DELIVERY and below_sma:
            direction = 'LONG'
            strength = BASE_STRENGTH_EXTREME
            size_modifier = ACCUMULATION_SIZE
            label = 'ACCUMULATION'

        elif delivery > HIGH_DELIVERY and below_sma:
            direction = 'LONG'
            strength = BASE_STRENGTH_MILD
            size_modifier = ACCUMULATION_SIZE
            label = 'MILD_ACCUMULATION'

        else:
            return None

        # ── DII confirmation ─────────────────────────────────────
        if not math.isnan(dii_net):
            if direction == 'LONG' and dii_net > 0:
                strength += DII_CONFIRM_BOOST
                label += '+DII_BUYING'
            elif direction == 'SHORT' and dii_net < 0:
                strength += DII_CONFIRM_BOOST
                label += '+DII_SELLING'

        strength = min(MAX_STRENGTH, max(MIN_STRENGTH, strength))

        reason_parts = [
            SIGNAL_ID,
            f"delivery_pct={delivery:.1f}%",
            f"close={current_close:.2f}",
            f"sma50={sma_50:.2f}",
            f"regime={label}",
        ]
        if not math.isnan(dii_net):
            reason_parts.append(f"dii_net={dii_net:+.0f}cr")

        logger.info('%s signal: %s on %s delivery=%.1f%% label=%s',
                     self.SIGNAL_ID, direction, trade_date, delivery, label)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(current_close, 2),
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'delivery_pct': round(delivery, 2),
                'close': round(current_close, 2),
                'sma_50': round(sma_50, 2),
                'above_sma': above_sma,
                'dii_net': round(dii_net, 2) if not math.isnan(dii_net) else None,
                'regime_label': label,
                'size_modifier': size_modifier,
                'trade_date': trade_date.isoformat(),
            },
        }

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"RetailBrokerSentimentSignal(signal_id='{self.SIGNAL_ID}')"

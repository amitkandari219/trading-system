"""
Trade Aggressor Imbalance Signal.

Measures the ratio of market buy orders vs market sell orders from tick
feed data.  Buy aggression indicates momentum demand; sell aggression
indicates distribution.

Signal logic:
    buy_aggressor_pct = percentage of trades initiated by buyers (0-100)

    buy_aggressor_pct > 60  -> LONG  (buy-side momentum)
    buy_aggressor_pct < 40  -> SHORT (sell-side momentum)
    45-55                   -> no signal (neutral)
    40-45 or 55-60          -> no signal (weak zone)

    5-bar momentum of aggressor ratio adds confirmation:
        if aggressor_pct is rising AND > 60 -> stronger LONG
        if aggressor_pct is falling AND < 40 -> stronger SHORT

Column required: buy_aggressor_pct (float, 0-100)
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

SIGNAL_ID = 'TRADE_AGGRESSOR'

# Aggressor thresholds
STRONG_BUY_THRESHOLD = 60.0
STRONG_SELL_THRESHOLD = 40.0
NEUTRAL_HIGH = 55.0
NEUTRAL_LOW = 45.0

# Momentum lookback
MOMENTUM_BARS = 5

# Strength scaling
MAX_STRENGTH = 0.90
MIN_STRENGTH = 0.15
BASE_STRENGTH = 0.40
MOMENTUM_BOOST = 0.15

# Size
STRONG_AGGRESSION_SIZE = 1.10
BASE_SIZE = 1.0

AGGRESSOR_COL = 'buy_aggressor_pct'


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

class TradeAggressorSignal:
    """
    Directional signal from trade aggressor imbalance — buy-side vs
    sell-side initiated trades from tick data.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('TradeAggressorSignal initialised')

    def evaluate(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        """
        Evaluate trade aggressor imbalance.

        Parameters
        ----------
        df         : DataFrame with `buy_aggressor_pct` column.
        trade_date : The date to evaluate.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata
        or None.
        """
        try:
            return self._evaluate_inner(df, trade_date)
        except Exception as e:
            logger.error('TradeAggressorSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: pd.DataFrame, trade_date: date) -> Optional[Dict]:
        if df is None or df.empty:
            return None

        if AGGRESSOR_COL not in df.columns:
            logger.debug('Column %s not found', AGGRESSOR_COL)
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

        aggressor_series = df_hist[AGGRESSOR_COL].astype(float)
        current_pct = _safe_float(aggressor_series.iloc[-1])

        if math.isnan(current_pct):
            return None

        # Clamp to [0, 100]
        current_pct = max(0.0, min(100.0, current_pct))

        # ── Compute 5-bar momentum ───────────────────────────────
        has_momentum = len(aggressor_series) >= MOMENTUM_BARS + 1
        momentum = 0.0
        if has_momentum:
            prev_pct = _safe_float(aggressor_series.iloc[-(MOMENTUM_BARS + 1)])
            if not math.isnan(prev_pct):
                momentum = current_pct - prev_pct

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

        if current_pct > STRONG_BUY_THRESHOLD:
            direction = 'LONG'
            # Base strength + scale with how far above threshold
            strength = BASE_STRENGTH + (current_pct - STRONG_BUY_THRESHOLD) * 0.012
            size_modifier = STRONG_AGGRESSION_SIZE

            # Momentum confirmation: rising aggressor strengthens signal
            if has_momentum and momentum > 0:
                strength += MOMENTUM_BOOST

        elif current_pct < STRONG_SELL_THRESHOLD:
            direction = 'SHORT'
            strength = BASE_STRENGTH + (STRONG_SELL_THRESHOLD - current_pct) * 0.012
            size_modifier = STRONG_AGGRESSION_SIZE

            # Momentum confirmation: falling aggressor strengthens signal
            if has_momentum and momentum < 0:
                strength += MOMENTUM_BOOST

        else:
            # Neutral / weak zone — no signal
            return None

        strength = min(MAX_STRENGTH, max(MIN_STRENGTH, strength))

        momentum_label = 'RISING' if momentum > 0 else ('FALLING' if momentum < 0 else 'FLAT')

        reason_parts = [
            SIGNAL_ID,
            f"aggressor_pct={current_pct:.1f}%",
            f"5bar_momentum={momentum:+.1f}",
            f"momentum_dir={momentum_label}",
            f"{'BUY' if direction == 'LONG' else 'SELL'}_AGGRESSION",
        ]

        logger.info('%s signal: %s on %s aggressor=%.1f%% momentum=%+.1f',
                     self.SIGNAL_ID, direction, trade_date, current_pct, momentum)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(price, 2) if not math.isnan(price) else None,
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'buy_aggressor_pct': round(current_pct, 2),
                'sell_aggressor_pct': round(100.0 - current_pct, 2),
                'momentum_5bar': round(momentum, 2),
                'momentum_direction': momentum_label,
                'size_modifier': size_modifier,
                'trade_date': trade_date.isoformat(),
            },
        }

    def reset(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"TradeAggressorSignal(signal_id='{self.SIGNAL_ID}')"

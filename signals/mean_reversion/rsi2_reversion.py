"""
RSI(2) Mean-Reversion Signal (Larry Connors).

Edge:
    The 2-period RSI is an extremely sensitive overbought/oversold
    oscillator.  When RSI(2) drops below 10 in an uptrend (close > SMA(200)),
    the next 1-5 day returns are strongly positive ~70-75% of the time.
    Conversely, RSI(2) > 90 in a downtrend (close < SMA(200)) precedes
    negative returns.

    This is one of the most well-documented short-term mean-reversion
    edges in equity indices (Connors & Alvarez, "Short Term Trading
    Strategies That Work").

Signal logic:
    rsi2 = RSI(close, period=2)
    sma200 = SMA(close, 200)

    LONG  when RSI(2) < 10  AND close > SMA(200)  (oversold in uptrend)
    SHORT when RSI(2) > 90  AND close < SMA(200)  (overbought in downtrend)
    EXIT  when RSI(2) crosses 50

Usage:
    from signals.mean_reversion.rsi2_reversion import RSI2ReversionSignal

    sig = RSI2ReversionSignal()
    result = sig.evaluate(df, date)
"""

import logging
import math
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    """Compute RSI using exponential (Wilder) smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


class RSI2ReversionSignal:
    """
    Larry Connors RSI(2) mean-reversion signal.

    Buys extreme oversold readings in uptrends, shorts extreme
    overbought readings in downtrends.  Historically ~70%+ win rate
    on major equity indices with 1-5 day holding periods.
    """

    SIGNAL_ID = 'RSI2_REVERSION'

    # ── Walk-forward parameters ──────────────────────────────────
    RSI_PERIOD = 2                 # Ultra-short RSI
    SMA_TREND_PERIOD = 200         # Trend filter
    RSI_OVERSOLD = 10.0            # Buy threshold
    RSI_OVERBOUGHT = 90.0          # Sell threshold
    RSI_EXIT = 50.0                # Exit when RSI crosses 50
    MIN_DATA_ROWS = 201            # Need at least SMA(200) + 1

    # ── Column names ─────────────────────────────────────────────
    CLOSE_COL = 'close'

    # ── Confidence ───────────────────────────────────────────────
    BASE_CONFIDENCE = 0.62
    EXTREME_RSI_BOOST = 0.10       # RSI < 5 or > 95
    TREND_ALIGNMENT_BOOST = 0.05   # Price well above/below SMA(200)

    def __init__(self) -> None:
        self._position_state: Optional[str] = None  # 'LONG', 'SHORT', or None
        logger.info('RSI2ReversionSignal initialised')

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame, date: date) -> Optional[Dict]:
        """
        Evaluate RSI(2) mean-reversion signal.

        Parameters
        ----------
        df   : DataFrame with at least 201 rows of daily data ending on
               or before `date`.  Must contain 'close' column.
        date : The evaluation date.

        Returns
        -------
        dict with signal details, or None if no trade.
        """
        try:
            return self._evaluate_inner(df, date)
        except Exception as e:
            logger.error(
                'RSI2ReversionSignal.evaluate error: %s', e, exc_info=True,
            )
            return None

    # ──────────────────────────────────────────────────────────────
    # INTERNALS
    # ──────────────────────────────────────────────────────────────

    def _evaluate_inner(self, df: pd.DataFrame, eval_date: date) -> Optional[Dict]:
        # ── Validate DataFrame ───────────────────────────────────
        if df is None or df.empty:
            logger.debug('Empty DataFrame — skip')
            return None

        if self.CLOSE_COL not in df.columns:
            logger.debug('Missing column %s — skip', self.CLOSE_COL)
            return None

        # ── Slice up to eval_date ────────────────────────────────
        if hasattr(df.index, 'date'):
            mask = df.index.date <= eval_date if isinstance(eval_date, date) else df.index <= eval_date
            df_slice = df.loc[mask]
        else:
            df_slice = df

        if len(df_slice) < self.MIN_DATA_ROWS:
            logger.debug(
                'Insufficient data: %d rows < %d required',
                len(df_slice), self.MIN_DATA_ROWS,
            )
            return None

        # ── Compute indicators ───────────────────────────────────
        close = df_slice[self.CLOSE_COL].astype(float)
        rsi2 = _compute_rsi(close, self.RSI_PERIOD)
        sma200 = close.rolling(window=self.SMA_TREND_PERIOD, min_periods=self.SMA_TREND_PERIOD).mean()

        current_close = float(close.iloc[-1])
        current_rsi = float(rsi2.iloc[-1])
        current_sma200 = float(sma200.iloc[-1])

        if math.isnan(current_close) or math.isnan(current_rsi) or math.isnan(current_sma200):
            logger.debug('NaN in indicators — skip')
            return None

        is_uptrend = current_close > current_sma200
        is_downtrend = current_close < current_sma200
        trend_distance_pct = ((current_close - current_sma200) / current_sma200) * 100.0

        # ── Check for EXIT condition ─────────────────────────────
        if self._position_state is not None:
            should_exit = False
            if self._position_state == 'LONG' and current_rsi >= self.RSI_EXIT:
                should_exit = True
            elif self._position_state == 'SHORT' and current_rsi <= self.RSI_EXIT:
                should_exit = True

            if should_exit:
                exit_direction = self._position_state
                self._position_state = None
                logger.info(
                    '%s EXIT: RSI(2)=%.1f crossed 50 (was %s)',
                    self.SIGNAL_ID, current_rsi, exit_direction,
                )
                return {
                    'signal_id': self.SIGNAL_ID,
                    'direction': 'EXIT',
                    'strength': 0.0,
                    'price': round(current_close, 2),
                    'reason': (
                        f"{self.SIGNAL_ID} | EXIT {exit_direction} | "
                        f"RSI(2)={current_rsi:.1f} crossed {self.RSI_EXIT}"
                    ),
                    'metadata': {
                        'rsi2': round(current_rsi, 2),
                        'sma200': round(current_sma200, 2),
                        'trend_distance_pct': round(trend_distance_pct, 2),
                        'eval_date': str(eval_date),
                        'previous_position': exit_direction,
                    },
                }

        # ── Check for ENTRY condition ────────────────────────────
        direction = None
        if current_rsi < self.RSI_OVERSOLD and is_uptrend:
            direction = 'LONG'
        elif current_rsi > self.RSI_OVERBOUGHT and is_downtrend:
            direction = 'SHORT'

        if direction is None:
            logger.debug(
                '%s: RSI(2)=%.1f trend=%s — no signal',
                self.SIGNAL_ID, current_rsi,
                'UP' if is_uptrend else 'DOWN',
            )
            return None

        # ── Confidence / strength ────────────────────────────────
        strength = self.BASE_CONFIDENCE

        # Boost for extreme RSI readings
        if current_rsi < 5 or current_rsi > 95:
            strength += self.EXTREME_RSI_BOOST

        # Boost for strong trend alignment
        if abs(trend_distance_pct) > 5.0:
            strength += self.TREND_ALIGNMENT_BOOST

        strength = min(1.0, max(0.0, strength))

        self._position_state = direction

        reason_parts = [
            self.SIGNAL_ID,
            f"RSI(2)={current_rsi:.1f}",
            f"Close={current_close:.2f}",
            f"SMA(200)={current_sma200:.2f}",
            f"Trend={'UP' if is_uptrend else 'DOWN'}",
            f"TrendDist={trend_distance_pct:+.2f}%",
        ]

        logger.info(
            '%s signal: %s RSI(2)=%.1f close=%.2f sma200=%.2f date=%s',
            self.SIGNAL_ID, direction, current_rsi, current_close,
            current_sma200, eval_date,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(current_close, 2),
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'rsi2': round(current_rsi, 2),
                'sma200': round(current_sma200, 2),
                'is_uptrend': is_uptrend,
                'trend_distance_pct': round(trend_distance_pct, 2),
                'rsi_period': self.RSI_PERIOD,
                'sma_period': self.SMA_TREND_PERIOD,
                'eval_date': str(eval_date),
            },
        }

    # ──────────────────────────────────────────────────────────────
    # UTILITY
    # ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._position_state = None

    def __repr__(self) -> str:
        return f"RSI2ReversionSignal(signal_id='{self.SIGNAL_ID}')"

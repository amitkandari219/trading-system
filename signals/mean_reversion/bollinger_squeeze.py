"""
Bollinger Band Squeeze -> Expansion Breakout Signal.

Edge:
    Volatility is mean-reverting: periods of extreme compression
    (Bollinger Bandwidth at multi-month lows) are reliably followed
    by expansion moves.  The "squeeze" — when bandwidth drops below
    the 5th percentile of its 120-day range for 3+ consecutive days —
    signals that a large directional move is imminent.

    The signal does NOT predict direction during the squeeze; it waits
    for the first close outside the bands after the squeeze to confirm
    direction.  This captures the initial momentum of the expansion
    while the squeeze provides the timing filter.

    Historically, post-squeeze moves on Nifty travel 2-4x the
    pre-squeeze bandwidth within 5-10 days.

Signal logic:
    bandwidth       = (bb_upper - bb_lower) / sma_20
    bw_percentile   = percentile_rank(bandwidth, 120 days)

    SQUEEZE detected when bw_percentile < 5th for 3+ consecutive days.

    After squeeze:
        close > bb_upper -> LONG  (upside breakout)
        close < bb_lower -> SHORT (downside breakout)

Usage:
    from signals.mean_reversion.bollinger_squeeze import BollingerSqueezeSignal

    sig = BollingerSqueezeSignal()
    result = sig.evaluate(df, date)
"""

import logging
import math
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BollingerSqueezeSignal:
    """
    Bollinger Band squeeze-to-expansion breakout signal.

    Detects low-volatility squeeze regimes and generates directional
    signals on the first breakout close outside the bands.
    """

    SIGNAL_ID = 'BOLLINGER_SQUEEZE'

    # ── Walk-forward parameters ──────────────────────────────────
    BANDWIDTH_PERCENTILE_WINDOW = 120  # Days for percentile ranking
    SQUEEZE_PERCENTILE = 5.0           # Bandwidth below this = squeeze
    SQUEEZE_MIN_DAYS = 3               # Consecutive squeeze days required
    MIN_DATA_ROWS = 125                # 120 + small buffer

    # ── Column names (use pre-computed BB columns) ───────────────
    BB_UPPER_COL = 'bb_upper'
    BB_LOWER_COL = 'bb_lower'
    SMA_20_COL = 'sma_20'
    CLOSE_COL = 'close'

    # ── Confidence ───────────────────────────────────────────────
    BASE_CONFIDENCE = 0.55
    LONG_SQUEEZE_BOOST = 0.08      # Squeeze lasted 5+ days
    VERY_LOW_BW_BOOST = 0.06       # Bandwidth < 2nd percentile
    WEAK_BREAKOUT_PENALTY = -0.04  # Close barely outside band

    def __init__(self) -> None:
        self._squeeze_days: int = 0
        self._in_squeeze: bool = False
        self._squeeze_fired: bool = False  # One signal per squeeze
        logger.info('BollingerSqueezeSignal initialised')

    # ──────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame, date: date) -> Optional[Dict]:
        """
        Evaluate the Bollinger squeeze breakout signal.

        Parameters
        ----------
        df   : DataFrame with at least `MIN_DATA_ROWS` rows ending on
               or before `date`.  Must contain columns: 'close',
               'bb_upper', 'bb_lower', 'sma_20'.
        date : The evaluation date.

        Returns
        -------
        dict with signal details, or None if no trade.
        """
        try:
            return self._evaluate_inner(df, date)
        except Exception as e:
            logger.error(
                'BollingerSqueezeSignal.evaluate error: %s', e, exc_info=True,
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

        required_cols = (
            self.CLOSE_COL, self.BB_UPPER_COL,
            self.BB_LOWER_COL, self.SMA_20_COL,
        )
        for col in required_cols:
            if col not in df.columns:
                logger.debug('Missing column %s — skip', col)
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

        # ── Extract columns ──────────────────────────────────────
        close = df_slice[self.CLOSE_COL].astype(float)
        bb_upper = df_slice[self.BB_UPPER_COL].astype(float)
        bb_lower = df_slice[self.BB_LOWER_COL].astype(float)
        sma_20 = df_slice[self.SMA_20_COL].astype(float)

        # ── Compute bandwidth ────────────────────────────────────
        bandwidth = (bb_upper - bb_lower) / sma_20.replace(0, np.nan)

        if bandwidth.iloc[-1] is None or math.isnan(float(bandwidth.iloc[-1])):
            logger.debug('NaN bandwidth — skip')
            return None

        # ── Compute bandwidth percentile over trailing window ────
        bw_window = bandwidth.iloc[-self.BANDWIDTH_PERCENTILE_WINDOW:]
        current_bw = float(bandwidth.iloc[-1])

        # percentile_rank: what % of trailing values are <= current
        bw_percentile = (
            (bw_window <= current_bw).sum() / len(bw_window)
        ) * 100.0

        # ── Update squeeze state ─────────────────────────────────
        if bw_percentile < self.SQUEEZE_PERCENTILE:
            self._squeeze_days += 1
            if self._squeeze_days >= self.SQUEEZE_MIN_DAYS:
                if not self._in_squeeze:
                    logger.debug(
                        '%s: SQUEEZE detected — %d days, BW_pctl=%.1f%%',
                        self.SIGNAL_ID, self._squeeze_days, bw_percentile,
                    )
                self._in_squeeze = True
        else:
            # No longer squeezing
            if self._in_squeeze:
                # Squeeze ended — check for breakout on THIS bar
                pass  # handled below
            else:
                # Reset if we never entered a valid squeeze
                self._squeeze_days = 0
                self._squeeze_fired = False

        # ── Check for breakout after squeeze ─────────────────────
        if not self._in_squeeze:
            logger.debug(
                '%s: No active squeeze (days=%d, pctl=%.1f%%) — skip',
                self.SIGNAL_ID, self._squeeze_days, bw_percentile,
            )
            return None

        current_close = float(close.iloc[-1])
        current_upper = float(bb_upper.iloc[-1])
        current_lower = float(bb_lower.iloc[-1])
        current_sma = float(sma_20.iloc[-1])

        if any(math.isnan(v) for v in (current_close, current_upper, current_lower, current_sma)):
            logger.debug('NaN in BB values — skip')
            return None

        # Already fired a signal for this squeeze cycle
        if self._squeeze_fired:
            # If bandwidth expands above squeeze threshold, reset for next cycle
            if bw_percentile >= self.SQUEEZE_PERCENTILE:
                self._in_squeeze = False
                self._squeeze_days = 0
                self._squeeze_fired = False
            return None

        # ── Direction from breakout ──────────────────────────────
        direction = None
        if current_close > current_upper:
            direction = 'LONG'
        elif current_close < current_lower:
            direction = 'SHORT'

        if direction is None:
            logger.debug(
                '%s: Squeeze active but no breakout — close=%.2f, '
                'upper=%.2f, lower=%.2f',
                self.SIGNAL_ID, current_close, current_upper, current_lower,
            )
            return None

        # ── Confidence / strength ────────────────────────────────
        strength = self.BASE_CONFIDENCE

        # Boost for prolonged squeeze (more compression = bigger expansion)
        if self._squeeze_days >= 5:
            strength += self.LONG_SQUEEZE_BOOST

        # Boost for extremely low bandwidth
        if bw_percentile < 2.0:
            strength += self.VERY_LOW_BW_BOOST

        # Penalty for marginal breakout (close barely outside band)
        breakout_margin = (
            (current_close - current_upper) / current_sma
            if direction == 'LONG'
            else (current_lower - current_close) / current_sma
        )
        if breakout_margin < 0.002:  # Less than 0.2% beyond band
            strength += self.WEAK_BREAKOUT_PENALTY

        strength = min(1.0, max(0.0, strength))

        # ── Mark squeeze as fired ────────────────────────────────
        self._squeeze_fired = True
        # Reset squeeze state — signal has been generated
        self._in_squeeze = False
        self._squeeze_days = 0

        reason_parts = [
            self.SIGNAL_ID,
            f"Breakout={'ABOVE upper' if direction == 'LONG' else 'BELOW lower'}",
            f"Close={current_close:.2f}",
            f"BB_Upper={current_upper:.2f}",
            f"BB_Lower={current_lower:.2f}",
            f"BW={current_bw:.4f}",
            f"BW_Pctl={bw_percentile:.1f}%",
            f"SqueezeDays={self._squeeze_days}",
        ]

        logger.info(
            '%s signal: %s close=%.2f bw_pctl=%.1f%% date=%s',
            self.SIGNAL_ID, direction, current_close, bw_percentile, eval_date,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(current_close, 2),
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'bandwidth': round(current_bw, 6),
                'bw_percentile': round(bw_percentile, 2),
                'bb_upper': round(current_upper, 2),
                'bb_lower': round(current_lower, 2),
                'sma_20': round(current_sma, 2),
                'squeeze_days': self._squeeze_days,
                'breakout_margin_pct': round(breakout_margin * 100, 4),
                'bw_percentile_window': self.BANDWIDTH_PERCENTILE_WINDOW,
                'eval_date': str(eval_date),
            },
        }

    # ──────────────────────────────────────────────────────────────
    # UTILITY
    # ──────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._squeeze_days = 0
        self._in_squeeze = False
        self._squeeze_fired = False

    def __repr__(self) -> str:
        return f"BollingerSqueezeSignal(signal_id='{self.SIGNAL_ID}')"

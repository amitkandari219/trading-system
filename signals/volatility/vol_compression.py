"""
Volatility Compression Breakout Signal (TTM Squeeze for Nifty).

Detects periods of low volatility (Bollinger Bands inside Keltner Channels)
followed by directional breakout.  This is the TTM Squeeze indicator adapted
for NSE Nifty.

Signal logic:
    Bollinger Bandwidth = (bb_upper - bb_lower) / sma_20
    Keltner Width       = 2 × ATR(20) / sma_20

    Squeeze ON:  bandwidth < keltner_width  for 6+ consecutive bars
    Momentum:    close - sma_20

    LONG  breakout: first bar where momentum turns positive after squeeze
    SHORT breakout: first bar where momentum turns negative after squeeze

Required columns: bb_upper, bb_lower, sma_20, and one of (atr_20, atr_14).
If bb columns are missing, they are computed from close with a 20-period /
2-std Bollinger.  If ATR is missing, it is computed from high/low/close.

Academic basis: TTM Squeeze (John Carter) — volatility contraction precedes
expansion, and the direction of the first momentum move after squeeze release
tends to persist for 8-12 bars.
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

SIGNAL_ID = 'VOL_COMPRESSION'

# Bollinger Bands
BB_PERIOD = 20
BB_STD_MULT = 2.0

# Keltner Channels
KC_ATR_PERIOD = 20
KC_ATR_MULT = 2.0             # Keltner width = 2 × ATR / SMA

# Squeeze detection
MIN_SQUEEZE_BARS = 6           # Minimum consecutive squeeze bars

# Strength scaling
STRENGTH_FLOOR = 0.35
STRENGTH_CEIL = 1.00
MOMENTUM_NORM = 0.02           # 2% momentum → strength 1.0

# Columns (preferred)
BB_UPPER_COL = 'bb_upper'
BB_LOWER_COL = 'bb_lower'
SMA_COL = 'sma_20'
ATR_COL_PRIMARY = 'atr_20'
ATR_COL_FALLBACK = 'atr_14'
CLOSE_COL = 'close'
HIGH_COL = 'high'
LOW_COL = 'low'


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


def _compute_sma(series: pd.Series, period: int) -> pd.Series:
    """Compute simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def _compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> pd.Series:
    """Compute Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def _compute_bollinger(
    close: pd.Series, period: int, std_mult: float
) -> tuple:
    """Compute Bollinger Bands.  Returns (upper, lower, sma)."""
    sma = _compute_sma(close, period)
    std = close.rolling(window=period, min_periods=period).std(ddof=1)
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma


# ================================================================
# SIGNAL CLASS
# ================================================================

class VolCompressionSignal:
    """
    Volatility compression breakout signal (TTM Squeeze).

    Detects BB-inside-Keltner squeeze for 6+ bars, then fires on the
    first momentum breakout bar.
    """

    SIGNAL_ID = SIGNAL_ID

    # Walk-forward parameters
    WF_BB_PERIOD = BB_PERIOD
    WF_BB_STD_MULT = BB_STD_MULT
    WF_KC_ATR_PERIOD = KC_ATR_PERIOD
    WF_KC_ATR_MULT = KC_ATR_MULT
    WF_MIN_SQUEEZE_BARS = MIN_SQUEEZE_BARS

    def __init__(self) -> None:
        logger.info('VolCompressionSignal initialised')

    def evaluate(self, df: pd.DataFrame, date_val: date) -> Optional[Dict]:
        """
        Evaluate volatility compression breakout signal.

        Parameters
        ----------
        df       : DataFrame with sufficient history.  Preferred columns:
                   bb_upper, bb_lower, sma_20, atr_20 (or atr_14), close.
                   If BB/ATR columns are missing, they are computed from
                   close (and high/low for ATR).
        date_val : The evaluation date.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata.
        Or None if no signal.
        """
        try:
            return self._evaluate_inner(df, date_val)
        except Exception as e:
            logger.error(
                'VolCompressionSignal.evaluate error: %s', e, exc_info=True
            )
            return None

    def _evaluate_inner(
        self, df: pd.DataFrame, date_val: date
    ) -> Optional[Dict]:
        # ── Validate DataFrame ──────────────────────────────────────
        if df is None or df.empty:
            logger.debug('Empty DataFrame')
            return None

        if CLOSE_COL not in df.columns:
            logger.debug('Missing column: %s', CLOSE_COL)
            return None

        min_rows = max(BB_PERIOD, KC_ATR_PERIOD) + MIN_SQUEEZE_BARS + 2
        if len(df) < min_rows:
            logger.debug(
                'Insufficient data: %d rows < %d required',
                len(df), min_rows,
            )
            return None

        close = df[CLOSE_COL].astype(float)

        # ── Resolve BB columns ─────────────────────────────────────
        has_bb = (
            BB_UPPER_COL in df.columns
            and BB_LOWER_COL in df.columns
            and SMA_COL in df.columns
        )

        if has_bb:
            bb_upper = df[BB_UPPER_COL].astype(float)
            bb_lower = df[BB_LOWER_COL].astype(float)
            sma = df[SMA_COL].astype(float)
        else:
            logger.debug('Computing Bollinger Bands from close prices')
            bb_upper, bb_lower, sma = _compute_bollinger(
                close, BB_PERIOD, BB_STD_MULT
            )

        # ── Resolve ATR column ─────────────────────────────────────
        if ATR_COL_PRIMARY in df.columns:
            atr = df[ATR_COL_PRIMARY].astype(float)
        elif ATR_COL_FALLBACK in df.columns:
            atr = df[ATR_COL_FALLBACK].astype(float)
        elif HIGH_COL in df.columns and LOW_COL in df.columns:
            logger.debug('Computing ATR from high/low/close')
            atr = _compute_atr(
                df[HIGH_COL].astype(float),
                df[LOW_COL].astype(float),
                close,
                KC_ATR_PERIOD,
            )
        else:
            logger.debug('Cannot compute ATR: missing high/low columns')
            return None

        # ── Compute bandwidth and Keltner width ────────────────────
        bandwidth = (bb_upper - bb_lower) / sma
        keltner_width = KC_ATR_MULT * atr / sma

        # ── Detect squeeze (BB inside Keltner) ─────────────────────
        squeeze = bandwidth < keltner_width

        # Count consecutive squeeze bars ending at each position
        squeeze_count = pd.Series(0, index=df.index, dtype=int)
        count = 0
        for i in range(len(squeeze)):
            if squeeze.iloc[i]:
                count += 1
            else:
                count = 0
            squeeze_count.iloc[i] = count

        # ── Check for squeeze release on current bar ───────────────
        # Current bar must NOT be in squeeze (squeeze just ended)
        # Previous bar must have been in squeeze for 6+ bars
        curr_idx = len(df) - 1
        prev_idx = curr_idx - 1

        if prev_idx < 0:
            return None

        curr_in_squeeze = squeeze.iloc[curr_idx]
        prev_squeeze_count = squeeze_count.iloc[prev_idx]

        if curr_in_squeeze:
            logger.debug(
                'Still in squeeze (count=%d) — waiting for release',
                squeeze_count.iloc[curr_idx],
            )
            return None

        if prev_squeeze_count < MIN_SQUEEZE_BARS:
            logger.debug(
                'Previous squeeze count %d < %d minimum',
                prev_squeeze_count, MIN_SQUEEZE_BARS,
            )
            return None

        # ── Momentum direction ─────────────────────────────────────
        sma_current = _safe_float(sma.iloc[curr_idx])
        close_current = _safe_float(close.iloc[curr_idx])

        if math.isnan(sma_current) or sma_current <= 0:
            return None
        if math.isnan(close_current) or close_current <= 0:
            return None

        momentum = close_current - sma_current
        momentum_pct = momentum / sma_current

        if momentum > 0:
            direction = 'LONG'
        elif momentum < 0:
            direction = 'SHORT'
        else:
            logger.debug('Zero momentum at squeeze release — skip')
            return None

        # ── Strength ───────────────────────────────────────────────
        # Scale with momentum magnitude and squeeze duration
        mom_strength = min(1.0, abs(momentum_pct) / MOMENTUM_NORM)
        duration_bonus = min(0.2, (prev_squeeze_count - MIN_SQUEEZE_BARS) * 0.04)
        strength = min(
            STRENGTH_CEIL,
            max(STRENGTH_FLOOR, mom_strength + duration_bonus),
        )

        # ── Compute current metrics for metadata ──────────────────
        bw_val = _safe_float(bandwidth.iloc[curr_idx])
        kw_val = _safe_float(keltner_width.iloc[curr_idx])
        atr_val = _safe_float(atr.iloc[curr_idx])

        # ── Reason ─────────────────────────────────────────────────
        reason = (
            f"VOL_COMPRESSION | Squeeze={prev_squeeze_count} bars | "
            f"Momentum={momentum:+.1f} ({momentum_pct:+.3%}) | "
            f"BB_BW={bw_val:.4f} | KC_W={kw_val:.4f} | "
            f"Direction={direction}"
        )

        logger.info(
            '%s signal: %s squeeze=%d bars mom=%.1f strength=%.2f on %s',
            self.SIGNAL_ID, direction, prev_squeeze_count, momentum,
            strength, date_val,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 4),
            'price': round(close_current, 2),
            'reason': reason,
            'metadata': {
                'squeeze_bars': int(prev_squeeze_count),
                'momentum': round(momentum, 2),
                'momentum_pct': round(momentum_pct, 6),
                'bollinger_bandwidth': round(bw_val, 6) if not math.isnan(bw_val) else None,
                'keltner_width': round(kw_val, 6) if not math.isnan(kw_val) else None,
                'atr': round(atr_val, 2) if not math.isnan(atr_val) else None,
                'sma_20': round(sma_current, 2),
                'close': round(close_current, 2),
                'date': str(date_val),
            },
        }

    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        pass

    def __repr__(self) -> str:
        return f"VolCompressionSignal(signal_id='{self.SIGNAL_ID}')"

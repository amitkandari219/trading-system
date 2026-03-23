"""
RSI Divergence Signal on 5-min chart.

Detects classic bullish and bearish RSI divergences by comparing recent
price swing lows/highs with corresponding RSI swing lows/highs.

Signal logic:
    RSI(14) computed on 5-min close prices.

    Bullish divergence:
      Price makes a LOWER low  (swing_low_2 < swing_low_1)
      RSI   makes a HIGHER low (rsi_low_2   > rsi_low_1)
      → LONG

    Bearish divergence:
      Price makes a HIGHER high (swing_high_2 > swing_high_1)
      RSI   makes a LOWER high  (rsi_high_2  < rsi_high_1)
      → SHORT

    Swing detection:
      A swing low  = bar whose low  is lower than the 2 bars on each side
      A swing high = bar whose high is higher than the 2 bars on each side

    Risk: SL = beyond the divergence extreme, TGT = 1.5× risk
    Requires at least 30 bars of history.
    Active window: 10:00 – 14:30 IST.

Usage:
    from signals.intraday.rsi_divergence import RSIDivergenceSignal
    sig = RSIDivergenceSignal()
    result = sig.evaluate(trade_date, current_time, bar_data, context)
"""

import logging
import math
from datetime import date, time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================
RSI_PERIOD = 14
MIN_BARS = 30
SWING_LOOKBACK = 2               # Bars on each side for swing detection
TARGET_RISK_MULT = 1.5

SIGNAL_WINDOW_START = time(10, 0)
SIGNAL_WINDOW_END = time(14, 30)

# Confidence
CONF_BULLISH_DIV = 0.55
CONF_BEARISH_DIV = 0.55

# RSI extreme bonus (divergence near extremes is more potent)
RSI_OVERSOLD = 35.0
RSI_OVERBOUGHT = 65.0
RSI_EXTREME_BOOST = 0.10


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


def _compute_rsi_series(closes: Sequence[float], period: int = RSI_PERIOD) -> List[float]:
    """
    Compute RSI for each bar in *closes* using Wilder's smoothing.

    Returns a list of RSI values aligned 1:1 with *closes*.
    The first *period* values will be NaN.
    """
    n = len(closes)
    rsi = [float('nan')] * n

    if n < period + 1:
        return rsi

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _find_swing_lows(
    lows: Sequence[float],
    rsi_values: Sequence[float],
    lookback: int = SWING_LOOKBACK,
) -> List[Tuple[int, float, float]]:
    """
    Find swing lows in the *lows* series.

    A swing low at index i: lows[i] < lows[j] for all j in
    [i-lookback, i+lookback] (j != i).

    Returns list of (index, price_low, rsi_value) sorted by index.
    """
    results: List[Tuple[int, float, float]] = []
    n = len(lows)
    for i in range(lookback, n - lookback):
        if math.isnan(lows[i]) or math.isnan(rsi_values[i]):
            continue
        is_swing = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if j < 0 or j >= n:
                continue
            if math.isnan(lows[j]):
                is_swing = False
                break
            if lows[j] <= lows[i]:
                is_swing = False
                break
        if is_swing:
            results.append((i, lows[i], rsi_values[i]))
    return results


def _find_swing_highs(
    highs: Sequence[float],
    rsi_values: Sequence[float],
    lookback: int = SWING_LOOKBACK,
) -> List[Tuple[int, float, float]]:
    """
    Find swing highs in the *highs* series.

    Returns list of (index, price_high, rsi_value) sorted by index.
    """
    results: List[Tuple[int, float, float]] = []
    n = len(highs)
    for i in range(lookback, n - lookback):
        if math.isnan(highs[i]) or math.isnan(rsi_values[i]):
            continue
        is_swing = True
        for j in range(i - lookback, i + lookback + 1):
            if j == i:
                continue
            if j < 0 or j >= n:
                continue
            if math.isnan(highs[j]):
                is_swing = False
                break
            if highs[j] >= highs[i]:
                is_swing = False
                break
        if is_swing:
            results.append((i, highs[i], rsi_values[i]))
    return results


# ================================================================
# SIGNAL CLASS
# ================================================================
class RSIDivergenceSignal:
    """
    RSI divergence detector on 5-min bars.

    Compares the last two swing lows / highs in price vs RSI to identify
    bullish or bearish divergence setups.
    """

    SIGNAL_ID = 'RSI_DIVERGENCE'

    def __init__(self) -> None:
        logger.info('RSIDivergenceSignal initialised')

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: date,
        current_time: time,
        bar_data: Dict,
        context: Dict,
    ) -> Optional[Dict]:
        """
        Evaluate RSI divergence on the current bar.

        Parameters
        ----------
        trade_date : date
        current_time : time
        bar_data : dict
            Current 5-min bar.
        context : dict
            Must contain 'session_bars' (list of all bars so far today).

        Returns
        -------
        dict or None
        """
        try:
            return self._evaluate_inner(trade_date, current_time, bar_data, context)
        except Exception as e:
            logger.error('RSIDivergenceSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(
        self,
        trade_date: date,
        current_time: time,
        bar_data: Dict,
        context: Dict,
    ) -> Optional[Dict]:
        # ── Time guard ──────────────────────────────────────────
        if current_time < SIGNAL_WINDOW_START or current_time >= SIGNAL_WINDOW_END:
            return None

        # ── Data guard ──────────────────────────────────────────
        session_bars = context.get('session_bars', [])
        if len(session_bars) < MIN_BARS:
            return None

        # ── Extract price arrays ────────────────────────────────
        closes: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        for b in session_bars:
            c = _safe_float(b.get('close'))
            h = _safe_float(b.get('high'))
            lo = _safe_float(b.get('low'))
            closes.append(c)
            highs.append(h)
            lows.append(lo)

        # ── Compute RSI series ──────────────────────────────────
        rsi_series = _compute_rsi_series(closes, RSI_PERIOD)

        latest_rsi = rsi_series[-1] if rsi_series else float('nan')
        if math.isnan(latest_rsi):
            return None

        # ── Find swings ─────────────────────────────────────────
        swing_lows = _find_swing_lows(lows, rsi_series, SWING_LOOKBACK)
        swing_highs = _find_swing_highs(highs, rsi_series, SWING_LOOKBACK)

        curr_close = _safe_float(bar_data.get('close'))
        if math.isnan(curr_close):
            return None

        # ── Check bullish divergence ────────────────────────────
        if len(swing_lows) >= 2:
            sl_prev = swing_lows[-2]   # (idx, price, rsi)
            sl_last = swing_lows[-1]

            # Price: lower low,  RSI: higher low
            if sl_last[1] < sl_prev[1] and sl_last[2] > sl_prev[2]:
                confidence = CONF_BULLISH_DIV
                # Boost if RSI is in oversold zone
                if sl_last[2] < RSI_OVERSOLD:
                    confidence += RSI_EXTREME_BOOST

                stop_loss = sl_last[1]            # Below the divergence low
                risk = abs(curr_close - stop_loss)
                if risk <= 0:
                    risk = abs(curr_close * 0.003)  # Fallback: 0.3% of price
                    stop_loss = curr_close - risk
                target = curr_close + risk * TARGET_RISK_MULT

                confidence = min(0.90, max(0.10, confidence))

                return {
                    'signal_id': self.SIGNAL_ID,
                    'direction': 'LONG',
                    'confidence': round(confidence, 3),
                    'size_modifier': round(self._size_from_rsi(latest_rsi, 'LONG'), 2),
                    'entry_price': round(curr_close, 2),
                    'stop_loss': round(stop_loss, 2),
                    'target': round(target, 2),
                    'reason': (
                        f"Bullish divergence | PriceLow: {sl_prev[1]:.2f}→{sl_last[1]:.2f} "
                        f"| RSILow: {sl_prev[2]:.1f}→{sl_last[2]:.1f} "
                        f"| RSI={latest_rsi:.1f} | R:R={TARGET_RISK_MULT:.1f}"
                    ),
                }

        # ── Check bearish divergence ────────────────────────────
        if len(swing_highs) >= 2:
            sh_prev = swing_highs[-2]
            sh_last = swing_highs[-1]

            # Price: higher high,  RSI: lower high
            if sh_last[1] > sh_prev[1] and sh_last[2] < sh_prev[2]:
                confidence = CONF_BEARISH_DIV
                # Boost if RSI is in overbought zone
                if sh_last[2] > RSI_OVERBOUGHT:
                    confidence += RSI_EXTREME_BOOST

                stop_loss = sh_last[1]            # Above the divergence high
                risk = abs(stop_loss - curr_close)
                if risk <= 0:
                    risk = abs(curr_close * 0.003)
                    stop_loss = curr_close + risk
                target = curr_close - risk * TARGET_RISK_MULT

                confidence = min(0.90, max(0.10, confidence))

                return {
                    'signal_id': self.SIGNAL_ID,
                    'direction': 'SHORT',
                    'confidence': round(confidence, 3),
                    'size_modifier': round(self._size_from_rsi(latest_rsi, 'SHORT'), 2),
                    'entry_price': round(curr_close, 2),
                    'stop_loss': round(stop_loss, 2),
                    'target': round(target, 2),
                    'reason': (
                        f"Bearish divergence | PriceHigh: {sh_prev[1]:.2f}→{sh_last[1]:.2f} "
                        f"| RSIHigh: {sh_prev[2]:.1f}→{sh_last[2]:.1f} "
                        f"| RSI={latest_rsi:.1f} | R:R={TARGET_RISK_MULT:.1f}"
                    ),
                }

        return None

    # ----------------------------------------------------------
    # Size from RSI extremity
    # ----------------------------------------------------------
    @staticmethod
    def _size_from_rsi(rsi: float, direction: str) -> float:
        """
        Scale position size based on how extreme the RSI reading is.

        For LONG: lower RSI = more oversold = larger size.
        For SHORT: higher RSI = more overbought = larger size.
        """
        if math.isnan(rsi):
            return 1.0

        if direction == 'LONG':
            if rsi < 25:
                return 1.4
            elif rsi < 35:
                return 1.2
            elif rsi < 45:
                return 1.0
            else:
                return 0.8
        else:  # SHORT
            if rsi > 75:
                return 1.4
            elif rsi > 65:
                return 1.2
            elif rsi > 55:
                return 1.0
            else:
                return 0.8

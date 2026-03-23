"""
VWAP Crossover & Mean-Reversion Signal.

Computes rolling VWAP and standard-deviation bands from intraday 5-min bars
and generates two flavours of signal:

  1. VWAP Crossover — price crosses above VWAP → LONG, below → SHORT
     (trend-following; most effective in directional sessions)

  2. VWAP Mean-Reversion — price touches -2σ band → LONG, +2σ → SHORT
     (mean-reversion; most effective in range-bound / positive-gamma sessions)

Signal logic:
    VWAP = Σ(typical_price × volume) / Σ(volume)
    σ    = sqrt(Σ(volume × (typical_price - VWAP)²) / Σ(volume))

    Bands: VWAP ± 1σ, VWAP ± 2σ

    Crossover:
      prev_close <= VWAP  and  curr_close > VWAP  → LONG
      prev_close >= VWAP  and  curr_close < VWAP  → SHORT

    Mean-reversion:
      close <= VWAP - 2σ → LONG  (oversold snap-back)
      close >= VWAP + 2σ → SHORT (overbought snap-back)

    SL: VWAP ± 1σ (opposite side)
    TGT: VWAP (crossover) or opposite 1σ band (mean-rev)

    Requires at least 10 bars before firing.
    Active window: 09:45 – 14:30 IST.

Usage:
    from signals.intraday.vwap_signal import VWAPSignal
    sig = VWAPSignal()
    result = sig.evaluate(trade_date, current_time, bar_data, context)
"""

import logging
import math
from datetime import date, time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================
MIN_BARS = 10
SIGNAL_WINDOW_START = time(9, 45)
SIGNAL_WINDOW_END = time(14, 30)

# Band multipliers
BAND_1_SIGMA = 1.0
BAND_2_SIGMA = 2.0

# Confidence
BASE_CONFIDENCE_CROSSOVER = 0.50
BASE_CONFIDENCE_MEANREV = 0.55  # mean-rev slightly higher base (empirical)

# Size
SIZE_CROSSOVER = 1.0
SIZE_MEANREV = 1.1               # mean-rev edges have higher historical win-rate


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


def _typical_price(bar: Dict) -> float:
    h = _safe_float(bar.get('high'))
    lo = _safe_float(bar.get('low'))
    c = _safe_float(bar.get('close'))
    if math.isnan(h) or math.isnan(lo) or math.isnan(c):
        return float('nan')
    return (h + lo + c) / 3.0


# ================================================================
# SIGNAL CLASS
# ================================================================
class VWAPSignal:
    """
    VWAP crossover + mean-reversion signal on 5-min bars.

    Maintains no state between days — everything is derived from
    ``context['session_bars']``.
    """

    SIGNAL_ID_CROSSOVER = 'VWAP_CROSSOVER'
    SIGNAL_ID_MEANREV = 'VWAP_MEAN_REVERSION'

    def __init__(self) -> None:
        logger.info('VWAPSignal initialised')

    # ----------------------------------------------------------
    # VWAP + bands computation
    # ----------------------------------------------------------
    @staticmethod
    def _compute_vwap_bands(
        session_bars: Sequence[Dict],
    ) -> Optional[Dict[str, float]]:
        """
        Compute VWAP, 1σ and 2σ bands from session bars.

        Returns dict with: vwap, sigma, upper_1, lower_1, upper_2, lower_2
        or None on insufficient / bad data.
        """
        cum_pv = 0.0
        cum_vol = 0.0
        cum_pv2 = 0.0   # Σ(vol × tp²) for variance

        for bar in session_bars:
            tp = _typical_price(bar)
            vol = _safe_float(bar.get('volume'), 0)
            if math.isnan(tp) or vol <= 0:
                continue
            cum_pv += tp * vol
            cum_pv2 += tp * tp * vol
            cum_vol += vol

        if cum_vol <= 0:
            return None

        vwap = cum_pv / cum_vol

        # Volume-weighted variance: E[X²] - E[X]²
        variance = (cum_pv2 / cum_vol) - (vwap * vwap)
        if variance < 0:
            variance = 0.0
        sigma = math.sqrt(variance)

        return {
            'vwap': vwap,
            'sigma': sigma,
            'upper_1': vwap + BAND_1_SIGMA * sigma,
            'lower_1': vwap - BAND_1_SIGMA * sigma,
            'upper_2': vwap + BAND_2_SIGMA * sigma,
            'lower_2': vwap - BAND_2_SIGMA * sigma,
        }

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
        Evaluate VWAP signal on the current 5-min bar.

        Parameters
        ----------
        trade_date : date
            Current trading day.
        current_time : time
            Timestamp of the current bar.
        bar_data : dict
            Current bar: open, high, low, close, volume, timestamp.
        context : dict
            Must contain 'session_bars' (list of all bars so far today
            including the current bar) and optionally 'prev_bar'.

        Returns
        -------
        dict or None
        """
        try:
            return self._evaluate_inner(trade_date, current_time, bar_data, context)
        except Exception as e:
            logger.error('VWAPSignal.evaluate error: %s', e, exc_info=True)
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

        prev_bar = context.get('prev_bar')
        if prev_bar is None and len(session_bars) >= 2:
            prev_bar = session_bars[-2]
        if prev_bar is None:
            return None

        # ── VWAP + bands ───────────────────────────────────────
        bands = self._compute_vwap_bands(session_bars)
        if bands is None:
            return None

        vwap = bands['vwap']
        sigma = bands['sigma']
        upper_1 = bands['upper_1']
        lower_1 = bands['lower_1']
        upper_2 = bands['upper_2']
        lower_2 = bands['lower_2']

        if sigma <= 0:
            return None  # Flat session, no actionable bands

        curr_close = _safe_float(bar_data.get('close'))
        prev_close = _safe_float(prev_bar.get('close'))
        if math.isnan(curr_close) or math.isnan(prev_close):
            return None

        # ── Signal 1: Mean-reversion at ±2σ (higher priority) ──
        if curr_close <= lower_2:
            return self._build_result(
                signal_id=self.SIGNAL_ID_MEANREV,
                direction='LONG',
                entry_price=curr_close,
                stop_loss=curr_close - sigma,     # SL below entry by 1σ
                target=vwap,                      # Target: VWAP
                confidence=BASE_CONFIDENCE_MEANREV,
                size_modifier=SIZE_MEANREV,
                bands=bands,
                extra_reason='Price at -2σ band → mean-reversion LONG',
            )

        if curr_close >= upper_2:
            return self._build_result(
                signal_id=self.SIGNAL_ID_MEANREV,
                direction='SHORT',
                entry_price=curr_close,
                stop_loss=curr_close + sigma,
                target=vwap,
                confidence=BASE_CONFIDENCE_MEANREV,
                size_modifier=SIZE_MEANREV,
                bands=bands,
                extra_reason='Price at +2σ band → mean-reversion SHORT',
            )

        # ── Signal 2: VWAP crossover ───────────────────────────
        if prev_close <= vwap and curr_close > vwap:
            return self._build_result(
                signal_id=self.SIGNAL_ID_CROSSOVER,
                direction='LONG',
                entry_price=curr_close,
                stop_loss=lower_1,
                target=upper_1,
                confidence=BASE_CONFIDENCE_CROSSOVER,
                size_modifier=SIZE_CROSSOVER,
                bands=bands,
                extra_reason='VWAP cross-above → LONG',
            )

        if prev_close >= vwap and curr_close < vwap:
            return self._build_result(
                signal_id=self.SIGNAL_ID_CROSSOVER,
                direction='SHORT',
                entry_price=curr_close,
                stop_loss=upper_1,
                target=lower_1,
                confidence=BASE_CONFIDENCE_CROSSOVER,
                size_modifier=SIZE_CROSSOVER,
                bands=bands,
                extra_reason='VWAP cross-below → SHORT',
            )

        return None

    # ----------------------------------------------------------
    # Result builder
    # ----------------------------------------------------------
    @staticmethod
    def _build_result(
        signal_id: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        confidence: float,
        size_modifier: float,
        bands: Dict[str, float],
        extra_reason: str,
    ) -> Dict:
        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0.0

        # Boost confidence when R:R is favourable
        if rr_ratio >= 2.0:
            confidence += 0.08
        elif rr_ratio >= 1.5:
            confidence += 0.04
        confidence = min(0.90, max(0.10, confidence))

        reason_parts = [
            extra_reason,
            f"VWAP={bands['vwap']:.2f}",
            f"σ={bands['sigma']:.2f}",
            f"Entry={entry_price:.2f}",
            f"R:R={rr_ratio:.1f}",
        ]

        return {
            'signal_id': signal_id,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 2),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'reason': ' | '.join(reason_parts),
        }

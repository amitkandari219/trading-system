"""
Momentum Candle Detection Signal.

Identifies three high-conviction candlestick momentum patterns on 5-min bars:

  1. Wide-Range Bar  — body > 2× 20-bar ATR → explosive directional move
  2. Three-Bar Momentum — 3 consecutive same-direction closes → trend continuation
  3. Engulfing       — current bar completely engulfs previous bar → reversal

Signal logic:
    Wide-Range:
      body = abs(close - open)
      atr_20 = 20-bar average true range
      body > 2 × atr_20 → signal in bar direction

    Three-Bar Momentum:
      bars[-3].close > bars[-3].open  AND
      bars[-2].close > bars[-2].open  AND
      bars[-1].close > bars[-1].open  → LONG (vice versa for SHORT)

    Engulfing:
      Bullish: curr_open < prev_close AND curr_close > prev_open (prev was bearish)
      Bearish: curr_open > prev_close AND curr_close < prev_open (prev was bullish)

    Risk: SL = low of signal bar (LONG) or high (SHORT), TGT = 2× risk
    Active window: 10:00 – 14:30 IST

Usage:
    from signals.intraday.momentum_candles import MomentumCandleSignal
    sig = MomentumCandleSignal()
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
ATR_PERIOD = 20
WIDE_RANGE_MULT = 2.0            # body > 2× ATR
THREE_BAR_COUNT = 3
TARGET_RISK_MULT = 2.0           # TGT = 2× risk

SIGNAL_WINDOW_START = time(10, 0)
SIGNAL_WINDOW_END = time(14, 30)

# Confidence bases
CONF_WIDE_RANGE = 0.60
CONF_THREE_BAR = 0.55
CONF_ENGULFING = 0.52


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


def _compute_atr(bars: Sequence[Dict], period: int = ATR_PERIOD) -> float:
    """
    Compute Average True Range over the last *period* bars.

    True Range = max(H-L, |H-prev_C|, |L-prev_C|)
    """
    if len(bars) < period + 1:
        # Fall back to simple high-low range average
        ranges = []
        for b in bars:
            h = _safe_float(b.get('high'))
            lo = _safe_float(b.get('low'))
            if not (math.isnan(h) or math.isnan(lo)):
                ranges.append(h - lo)
        return float(np.mean(ranges)) if ranges else float('nan')

    true_ranges: List[float] = []
    recent = list(bars[-(period + 1):])
    for i in range(1, len(recent)):
        h = _safe_float(recent[i].get('high'))
        lo = _safe_float(recent[i].get('low'))
        prev_c = _safe_float(recent[i - 1].get('close'))
        if math.isnan(h) or math.isnan(lo) or math.isnan(prev_c):
            continue
        tr = max(h - lo, abs(h - prev_c), abs(lo - prev_c))
        true_ranges.append(tr)

    return float(np.mean(true_ranges)) if true_ranges else float('nan')


# ================================================================
# SIGNAL CLASS
# ================================================================
class MomentumCandleSignal:
    """
    Detects wide-range bars, 3-bar momentum runs, and engulfing patterns
    on intraday 5-min bars.
    """

    SIGNAL_ID_WIDE = 'MOMENTUM_WIDE_RANGE'
    SIGNAL_ID_THREE = 'MOMENTUM_THREE_BAR'
    SIGNAL_ID_ENGULF = 'MOMENTUM_ENGULFING'

    def __init__(self) -> None:
        logger.info('MomentumCandleSignal initialised')

    # ----------------------------------------------------------
    # Pattern detectors
    # ----------------------------------------------------------
    @staticmethod
    def _check_wide_range(
        bar: Dict, atr: float
    ) -> Optional[str]:
        """Return 'LONG' / 'SHORT' if bar body exceeds 2× ATR, else None."""
        o = _safe_float(bar.get('open'))
        c = _safe_float(bar.get('close'))
        if math.isnan(o) or math.isnan(c) or math.isnan(atr) or atr <= 0:
            return None

        body = abs(c - o)
        if body > WIDE_RANGE_MULT * atr:
            return 'LONG' if c > o else 'SHORT'
        return None

    @staticmethod
    def _check_three_bar(
        bars: Sequence[Dict],
    ) -> Optional[str]:
        """Return direction if last 3 bars close in the same direction."""
        if len(bars) < THREE_BAR_COUNT:
            return None

        recent = list(bars[-THREE_BAR_COUNT:])
        bull_count = 0
        bear_count = 0
        for b in recent:
            o = _safe_float(b.get('open'))
            c = _safe_float(b.get('close'))
            if math.isnan(o) or math.isnan(c):
                return None
            if c > o:
                bull_count += 1
            elif c < o:
                bear_count += 1
            # doji (c == o) counts for neither

        if bull_count == THREE_BAR_COUNT:
            return 'LONG'
        if bear_count == THREE_BAR_COUNT:
            return 'SHORT'
        return None

    @staticmethod
    def _check_engulfing(
        bar: Dict, prev_bar: Dict
    ) -> Optional[str]:
        """Return direction if current bar engulfs previous bar."""
        o = _safe_float(bar.get('open'))
        c = _safe_float(bar.get('close'))
        po = _safe_float(prev_bar.get('open'))
        pc = _safe_float(prev_bar.get('close'))

        if any(math.isnan(v) for v in (o, c, po, pc)):
            return None

        # Bullish engulfing: prev bearish, curr body wraps prev
        if pc < po and o <= pc and c >= po:
            return 'LONG'

        # Bearish engulfing: prev bullish, curr body wraps prev
        if pc > po and o >= pc and c <= po:
            return 'SHORT'

        return None

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
        Evaluate momentum candle patterns.

        Parameters
        ----------
        trade_date : date
        current_time : time
        bar_data : dict
            Current 5-min bar.
        context : dict
            Must contain 'session_bars' and optionally 'prev_bar'.

        Returns
        -------
        dict or None
        """
        try:
            return self._evaluate_inner(trade_date, current_time, bar_data, context)
        except Exception as e:
            logger.error('MomentumCandleSignal.evaluate error: %s', e, exc_info=True)
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

        session_bars = context.get('session_bars', [])
        if len(session_bars) < ATR_PERIOD:
            return None

        prev_bar = context.get('prev_bar')
        if prev_bar is None and len(session_bars) >= 2:
            prev_bar = session_bars[-2]

        atr = _compute_atr(session_bars, ATR_PERIOD)

        # ── Priority 1: Wide-Range Bar ──────────────────────────
        wr_dir = self._check_wide_range(bar_data, atr)
        if wr_dir is not None:
            return self._build_result(
                signal_id=self.SIGNAL_ID_WIDE,
                direction=wr_dir,
                bar_data=bar_data,
                atr=atr,
                confidence=CONF_WIDE_RANGE,
                reason=f'Wide-range bar (body > {WIDE_RANGE_MULT:.0f}× ATR) → {wr_dir}',
            )

        # ── Priority 2: Three-Bar Momentum ─────────────────────
        tb_dir = self._check_three_bar(session_bars)
        if tb_dir is not None:
            return self._build_result(
                signal_id=self.SIGNAL_ID_THREE,
                direction=tb_dir,
                bar_data=bar_data,
                atr=atr,
                confidence=CONF_THREE_BAR,
                reason=f'3-bar momentum run → {tb_dir}',
            )

        # ── Priority 3: Engulfing ──────────────────────────────
        if prev_bar is not None:
            eng_dir = self._check_engulfing(bar_data, prev_bar)
            if eng_dir is not None:
                return self._build_result(
                    signal_id=self.SIGNAL_ID_ENGULF,
                    direction=eng_dir,
                    bar_data=bar_data,
                    atr=atr,
                    confidence=CONF_ENGULFING,
                    reason=f'Engulfing pattern → {eng_dir}',
                )

        return None

    # ----------------------------------------------------------
    # Result builder
    # ----------------------------------------------------------
    @staticmethod
    def _build_result(
        signal_id: str,
        direction: str,
        bar_data: Dict,
        atr: float,
        confidence: float,
        reason: str,
    ) -> Dict:
        close = _safe_float(bar_data.get('close'))
        high = _safe_float(bar_data.get('high'))
        low = _safe_float(bar_data.get('low'))

        if direction == 'LONG':
            stop_loss = low if not math.isnan(low) else close - atr
            risk = abs(close - stop_loss)
            target = close + risk * TARGET_RISK_MULT
        else:
            stop_loss = high if not math.isnan(high) else close + atr
            risk = abs(stop_loss - close)
            target = close - risk * TARGET_RISK_MULT

        # Size modifier: scale with conviction — wide-range gets more
        if signal_id.endswith('WIDE_RANGE'):
            size_modifier = 1.3
        elif signal_id.endswith('THREE_BAR'):
            size_modifier = 1.1
        else:
            size_modifier = 0.9  # engulfing is reversal — smaller initial size

        confidence = min(0.90, max(0.10, confidence))

        rr = (abs(target - close) / risk) if risk > 0 else 0.0
        reason_full = f"{reason} | ATR={atr:.2f} | Risk={risk:.2f} | R:R={rr:.1f}"

        return {
            'signal_id': signal_id,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 2),
            'entry_price': round(close, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'reason': reason_full,
        }

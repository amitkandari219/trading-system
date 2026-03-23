"""
Market Microstructure — intraday OVERLAY signal.

Approximates order-flow microstructure from OHLCV bar data to produce
a directional bias and sizing modifier.

Components:
  1. Bid-ask spread proxy: (high − low) / close normalised by ATR.
     Wide spread → liquidity withdrawal → reduce size.
  2. Order book imbalance proxy: volume × bar direction.
     Consecutive buy bars with rising volume → buy imbalance.
  3. Trade aggressor: bar close relative to range.
     Close near high = buy aggressor, close near low = sell aggressor.

Size modifier mapping:
  Balanced flow / neutral → 1.0
  Strong buy aggression   → 1.2
  Strong sell aggression  → 0.8
  Spread widening (>2× avg) → 0.7 (volatility precursor)

Data expectations (via `bar_data` and `context`):
  bar_data : dict with open, high, low, close, volume, timestamp
  context['bars_today']  : list[dict] — all bars so far today
  context['daily_atr']   : float — yesterday's ATR(14) for normalisation
  context['daily_vix']   : float (optional) — India VIX

Safety:
  - Returns neutral (size_modifier=1.0) when insufficient data.
  - All modifiers clamped to [0.5, 1.5].

Usage:
    from signals.intraday.microstructure import MarketMicrostructure
    ms = MarketMicrostructure()
    result = ms.evaluate(trade_date, current_time, bar_data, context)
"""

import logging
import math
from datetime import date, time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ════════════════════════════════════════════════════════════════════

# Spread proxy: (H-L)/C normalised by ATR — "wide" threshold
SPREAD_WIDE_MULT = 2.0          # > 2× average → wide spread
SPREAD_EXTREME_MULT = 3.5       # > 3.5× average → extreme

# Aggressor: close position within bar range
AGGRESSOR_BUY_THRESHOLD = 0.75  # close in top 25% of range
AGGRESSOR_SELL_THRESHOLD = 0.25 # close in bottom 25% of range

# Imbalance: rolling signed-volume ratio
IMBALANCE_STRONG = 0.65         # > 65% of volume is buy-initiated
IMBALANCE_WEAK = 0.35           # < 35% → sell-dominated

# Lookback for rolling calculations
ROLLING_BARS = 10

MIN_MODIFIER = 0.5
MAX_MODIFIER = 1.5
MIN_BARS_REQUIRED = 3


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _bar_range(bar: Dict) -> float:
    h = _safe_float(bar.get('high'))
    l = _safe_float(bar.get('low'))
    return max(h - l, 0.01)  # avoid zero-division


def _bar_direction(bar: Dict) -> float:
    """
    +1 if close > open, -1 if close < open, 0 if equal.
    """
    c = _safe_float(bar.get('close'))
    o = _safe_float(bar.get('open'))
    if c > o:
        return 1.0
    elif c < o:
        return -1.0
    return 0.0


def _close_position_in_range(bar: Dict) -> float:
    """
    Where the close sits within the bar range.
    Returns 0.0 (at low) to 1.0 (at high).
    """
    h = _safe_float(bar.get('high'))
    l = _safe_float(bar.get('low'))
    c = _safe_float(bar.get('close'))
    rng = h - l
    if rng < 0.01:
        return 0.5
    return _clamp((c - l) / rng, 0.0, 1.0)


# ════════════════════════════════════════════════════════════════════
# SIGNAL CLASS
# ════════════════════════════════════════════════════════════════════

class MarketMicrostructure:
    """
    Intraday overlay that approximates microstructure from OHLCV bars.
    """

    SIGNAL_ID = 'INTRADAY_MICROSTRUCTURE'

    def evaluate(
        self,
        trade_date: date,
        current_time: time,
        bar_data: Dict,
        context: Dict,
    ) -> Dict:
        """
        Evaluate microstructure and return overlay result.

        Returns:
            dict with signal_id, direction, confidence, size_modifier,
            reason, spread_z, aggressor, imbalance_ratio
        """
        neutral = self._neutral_result('Insufficient bar data')

        bars_today: List[Dict] = context.get('bars_today', [])
        daily_atr = _safe_float(context.get('daily_atr'))

        if len(bars_today) < MIN_BARS_REQUIRED:
            return neutral

        recent = bars_today[-ROLLING_BARS:]

        # ── 1. Spread proxy ─────────────────────────────────────
        spread_z = self._spread_analysis(recent, daily_atr)
        spread_modifier = self._spread_to_modifier(spread_z)

        # ── 2. Trade aggressor ──────────────────────────────────
        aggressor_score = self._aggressor_analysis(recent)
        aggressor_direction = self._aggressor_direction(aggressor_score)
        aggressor_modifier = self._aggressor_to_modifier(aggressor_score)

        # ── 3. Order book imbalance ─────────────────────────────
        imbalance = self._volume_imbalance(recent)
        imbalance_modifier = self._imbalance_to_modifier(imbalance)
        imbalance_direction = self._imbalance_direction(imbalance)

        # ── 4. Composite ────────────────────────────────────────
        # Spread is a risk modifier — always applied.
        # Aggressor and imbalance are directional — average them.
        directional_modifier = (aggressor_modifier + imbalance_modifier) / 2.0
        raw_modifier = spread_modifier * directional_modifier
        size_modifier = _clamp(raw_modifier, MIN_MODIFIER, MAX_MODIFIER)

        # ── 5. Direction consensus ──────────────────────────────
        direction = self._consensus_direction(aggressor_direction, imbalance_direction)
        confidence = self._compute_confidence(
            spread_z, aggressor_score, imbalance,
        )

        reason_parts = []
        if spread_z > SPREAD_WIDE_MULT:
            reason_parts.append(f'Wide spread ({spread_z:.1f}x)')
        if abs(aggressor_score) > 0.3:
            label = 'buy' if aggressor_score > 0 else 'sell'
            reason_parts.append(f'{label} aggressor {aggressor_score:+.2f}')
        if abs(imbalance - 0.5) > 0.1:
            reason_parts.append(f'Vol imbalance {imbalance:.0%}')
        reason = '; '.join(reason_parts) if reason_parts else 'Neutral microstructure'

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 3),
            'reason': reason,
            'spread_z': round(spread_z, 2),
            'aggressor': round(aggressor_score, 3),
            'imbalance_ratio': round(imbalance, 3),
        }

    # ----------------------------------------------------------------
    # spread analysis
    # ----------------------------------------------------------------

    @staticmethod
    def _spread_analysis(bars: List[Dict], daily_atr: float) -> float:
        """
        Compute current bar spread relative to session average.
        Returns z-score-like multiplier (1.0 = average).
        """
        if not bars:
            return 1.0

        ranges = [_bar_range(b) for b in bars]
        avg_range = float(np.mean(ranges)) if ranges else 1.0
        if avg_range < 0.01:
            avg_range = 0.01

        current_range = ranges[-1]
        z = current_range / avg_range

        # If we have daily ATR, also check absolute spread
        if daily_atr > 0:
            atr_ratio = current_range / daily_atr
            # 5-min bar should be ~5-8% of daily ATR
            # If > 15% → extreme spread
            if atr_ratio > 0.15:
                z = max(z, SPREAD_EXTREME_MULT)

        return z

    @staticmethod
    def _spread_to_modifier(spread_z: float) -> float:
        if spread_z >= SPREAD_EXTREME_MULT:
            return 0.6   # extreme spread → hard reduce
        if spread_z >= SPREAD_WIDE_MULT:
            return 0.8   # wide spread → reduce
        return 1.0

    # ----------------------------------------------------------------
    # aggressor analysis
    # ----------------------------------------------------------------

    @staticmethod
    def _aggressor_analysis(bars: List[Dict]) -> float:
        """
        Average close-position across recent bars.
        Returns -1 (all sell aggressor) to +1 (all buy aggressor).
        """
        if not bars:
            return 0.0
        positions = [_close_position_in_range(b) for b in bars]
        avg = float(np.mean(positions))
        # Map 0-1 to -1..+1
        return (avg - 0.5) * 2.0

    @staticmethod
    def _aggressor_direction(score: float) -> Optional[str]:
        if score >= 0.3:
            return 'LONG'
        if score <= -0.3:
            return 'SHORT'
        return None

    @staticmethod
    def _aggressor_to_modifier(score: float) -> float:
        """Map aggressor score to modifier."""
        if score >= 0.5:
            return 1.2
        if score >= 0.25:
            return 1.1
        if score <= -0.5:
            return 0.8
        if score <= -0.25:
            return 0.9
        return 1.0

    # ----------------------------------------------------------------
    # volume imbalance
    # ----------------------------------------------------------------

    @staticmethod
    def _volume_imbalance(bars: List[Dict]) -> float:
        """
        Fraction of total volume that is buy-initiated.
        Buy bar volume counted as buy, sell bar as sell.
        Returns 0.0 (all sell) to 1.0 (all buy).
        """
        buy_vol = 0.0
        sell_vol = 0.0

        for b in bars:
            vol = _safe_float(b.get('volume'), 0.0)
            direction = _bar_direction(b)
            if direction > 0:
                buy_vol += vol
            elif direction < 0:
                sell_vol += vol
            else:
                # Doji — split evenly
                buy_vol += vol * 0.5
                sell_vol += vol * 0.5

        total = buy_vol + sell_vol
        if total == 0:
            return 0.5
        return buy_vol / total

    @staticmethod
    def _imbalance_to_modifier(ratio: float) -> float:
        if ratio >= IMBALANCE_STRONG:
            return 1.2
        if ratio <= IMBALANCE_WEAK:
            return 0.8
        return 1.0

    @staticmethod
    def _imbalance_direction(ratio: float) -> Optional[str]:
        if ratio >= IMBALANCE_STRONG:
            return 'LONG'
        if ratio <= IMBALANCE_WEAK:
            return 'SHORT'
        return None

    # ----------------------------------------------------------------
    # consensus
    # ----------------------------------------------------------------

    @staticmethod
    def _consensus_direction(
        aggressor_dir: Optional[str],
        imbalance_dir: Optional[str],
    ) -> Optional[str]:
        if aggressor_dir == imbalance_dir:
            return aggressor_dir  # agreement (or both None)
        if aggressor_dir is not None and imbalance_dir is None:
            return aggressor_dir
        if imbalance_dir is not None and aggressor_dir is None:
            return imbalance_dir
        return None  # disagreement → no direction

    @staticmethod
    def _compute_confidence(
        spread_z: float,
        aggressor: float,
        imbalance: float,
    ) -> float:
        score = 0.0
        # Aggressor strength (0–0.4)
        score += min(abs(aggressor), 1.0) * 0.4
        # Imbalance strength (0–0.4)
        score += abs(imbalance - 0.5) * 2.0 * 0.4
        # Spread penalty (0–0.2 deduction)
        if spread_z > SPREAD_WIDE_MULT:
            score -= 0.15
        return _clamp(score, 0.0, 1.0)

    # ----------------------------------------------------------------
    # neutral fallback
    # ----------------------------------------------------------------

    def _neutral_result(self, reason: str = '') -> Dict:
        return {
            'signal_id': self.SIGNAL_ID,
            'direction': None,
            'confidence': 0.0,
            'size_modifier': 1.0,
            'reason': reason,
            'spread_z': 1.0,
            'aggressor': 0.0,
            'imbalance_ratio': 0.5,
        }

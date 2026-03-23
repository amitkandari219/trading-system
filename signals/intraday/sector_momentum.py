"""
Intraday Sector Momentum — intraday OVERLAY signal.

Tracks the relative performance of Nifty vs BankNifty intraday to detect
sector rotation / leadership and adjust position sizing accordingly.

Core logic:
  - BankNifty leads Nifty by >0.3% → Nifty likely follows → bullish LONG boost
  - BankNifty lags Nifty by >0.3%  → divergence / bearish → reduce LONG size
  - Compute intraday momentum score from returns over 15min / 30min / 60min

Size modifier mapping:
  Strong sector alignment (both up, BankNifty leading)  → 1.3
  Mild alignment                                         → 1.1
  Neutral / mixed                                        → 1.0
  Mild divergence (BankNifty lagging)                    → 0.85
  Strong divergence                                      → 0.7

Data expectations (via `context`):
  context['nifty_open']       : float — Nifty opening price (9:15)
  context['banknifty_open']   : float — BankNifty opening price (9:15)
  context['nifty_bars']       : list[dict] — Nifty intraday bars
  context['banknifty_bars']   : list[dict] — BankNifty intraday bars
  context['spot_price']       : float — current Nifty price
  context['banknifty_price']  : float — current BankNifty price

Safety:
  - Returns neutral (size_modifier=1.0) when BankNifty data is missing.
  - All modifiers clamped to [0.5, 1.5].

Usage:
    from signals.intraday.sector_momentum import SectorMomentum
    sm = SectorMomentum()
    result = sm.evaluate(trade_date, current_time, bar_data, context)
"""

import logging
from datetime import date, time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ════════════════════════════════════════════════════════════════════

# BankNifty lead thresholds (BankNifty return − Nifty return)
LEAD_STRONG = 0.005       # 0.5% — strong bullish lead
LEAD_MILD = 0.003         # 0.3% — mild bullish lead
LAG_MILD = -0.003         # -0.3% — mild bearish lag
LAG_STRONG = -0.005       # -0.5% — strong bearish lag

# Momentum score lookback windows (in bars)
MOMENTUM_SHORT = 3        # ~15 min (5-min bars)
MOMENTUM_MED = 6          # ~30 min
MOMENTUM_LONG = 12        # ~60 min

# Momentum weights
W_SHORT = 0.5
W_MED = 0.3
W_LONG = 0.2

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


def _return_from_open(current: float, open_price: float) -> float:
    """Intraday return from open."""
    if open_price <= 0:
        return 0.0
    return (current - open_price) / open_price


def _bars_return(bars: List[Dict], lookback: int) -> float:
    """Return over last `lookback` bars (close-to-close)."""
    if len(bars) < lookback + 1:
        return 0.0
    start_close = _safe_float(bars[-(lookback + 1)].get('close'))
    end_close = _safe_float(bars[-1].get('close'))
    if start_close <= 0:
        return 0.0
    return (end_close - start_close) / start_close


# ════════════════════════════════════════════════════════════════════
# SIGNAL CLASS
# ════════════════════════════════════════════════════════════════════

class SectorMomentum:
    """
    Intraday overlay that tracks Nifty vs BankNifty relative performance
    and computes a momentum-based sizing modifier.
    """

    SIGNAL_ID = 'INTRADAY_SECTOR_MOMENTUM'

    def evaluate(
        self,
        trade_date: date,
        current_time: time,
        bar_data: Dict,
        context: Dict,
    ) -> Dict:
        """
        Evaluate sector momentum and return overlay result.

        Returns:
            dict with signal_id, direction, confidence, size_modifier,
            reason, nifty_return, banknifty_return, lead_pct,
            momentum_score
        """
        neutral = self._neutral_result('Missing sector data')

        nifty_price = _safe_float(context.get('spot_price'))
        bnf_price = _safe_float(context.get('banknifty_price'))
        nifty_open = _safe_float(context.get('nifty_open'))
        bnf_open = _safe_float(context.get('banknifty_open'))

        if nifty_price <= 0 or bnf_price <= 0:
            return neutral
        if nifty_open <= 0 or bnf_open <= 0:
            return neutral

        nifty_bars: List[Dict] = context.get('nifty_bars', [])
        bnf_bars: List[Dict] = context.get('banknifty_bars', [])

        # ── 1. Intraday returns from open ───────────────────────
        nifty_ret = _return_from_open(nifty_price, nifty_open)
        bnf_ret = _return_from_open(bnf_price, bnf_open)
        lead_pct = bnf_ret - nifty_ret  # positive = BNF leading

        lead_modifier = self._lead_to_modifier(lead_pct)
        lead_direction = self._lead_to_direction(lead_pct, nifty_ret)

        # ── 2. Multi-timeframe momentum score ───────────────────
        momentum_score = self._compute_momentum_score(nifty_bars, bnf_bars)
        momentum_modifier = self._momentum_to_modifier(momentum_score)

        # ── 3. Trend alignment check ────────────────────────────
        alignment = self._check_alignment(nifty_ret, bnf_ret)
        alignment_modifier = self._alignment_to_modifier(alignment)

        # ── 4. Composite ────────────────────────────────────────
        raw = lead_modifier * momentum_modifier * alignment_modifier
        size_modifier = _clamp(raw, MIN_MODIFIER, MAX_MODIFIER)

        # ── 5. Direction and confidence ─────────────────────────
        direction = lead_direction
        confidence = self._compute_confidence(lead_pct, momentum_score, alignment)

        reason_parts = []
        if abs(lead_pct) > 0.002:
            leader = 'BNF leads' if lead_pct > 0 else 'BNF lags'
            reason_parts.append(f'{leader} by {abs(lead_pct)*100:.2f}%')
        if abs(momentum_score) > 0.3:
            reason_parts.append(f'Momentum {momentum_score:+.2f}')
        reason_parts.append(f'Alignment: {alignment}')
        reason = '; '.join(reason_parts)

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 3),
            'reason': reason,
            'nifty_return': round(nifty_ret * 100, 3),
            'banknifty_return': round(bnf_ret * 100, 3),
            'lead_pct': round(lead_pct * 100, 3),
            'momentum_score': round(momentum_score, 3),
        }

    # ----------------------------------------------------------------
    # lead analysis
    # ----------------------------------------------------------------

    @staticmethod
    def _lead_to_modifier(lead_pct: float) -> float:
        """Map BankNifty lead/lag to modifier."""
        if lead_pct >= LEAD_STRONG:
            return 1.20
        if lead_pct >= LEAD_MILD:
            return 1.10
        if lead_pct <= LAG_STRONG:
            return 0.75
        if lead_pct <= LAG_MILD:
            return 0.85
        return 1.0

    @staticmethod
    def _lead_to_direction(lead_pct: float, nifty_ret: float) -> Optional[str]:
        """
        Direction based on BankNifty leadership.
        Only issue direction when BNF leads significantly.
        """
        if lead_pct >= LEAD_MILD:
            return 'LONG'  # BNF leads up → Nifty follow-through expected
        if lead_pct <= LAG_MILD:
            return 'SHORT'  # BNF lagging → bearish for Nifty
        return None

    # ----------------------------------------------------------------
    # multi-timeframe momentum
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_momentum_score(
        nifty_bars: List[Dict],
        bnf_bars: List[Dict],
    ) -> float:
        """
        Weighted momentum score from Nifty+BankNifty returns over
        multiple lookback windows.  Returns -1 (strong bearish) to +1 (strong bullish).
        """
        if len(nifty_bars) < MOMENTUM_LONG + 1:
            # Fall back to what we have
            if len(nifty_bars) < MIN_BARS_REQUIRED:
                return 0.0

        scores = []
        weights = []

        for lookback, weight in [
            (MOMENTUM_SHORT, W_SHORT),
            (MOMENTUM_MED, W_MED),
            (MOMENTUM_LONG, W_LONG),
        ]:
            nifty_r = _bars_return(nifty_bars, lookback)
            bnf_r = _bars_return(bnf_bars, lookback) if len(bnf_bars) > lookback else 0.0

            # Average of both instruments, normalised to roughly -1..+1
            # 0.5% move in lookback → score of ~1.0
            combined = (nifty_r + bnf_r) / 2.0
            normalised = _clamp(combined / 0.005, -1.0, 1.0)
            scores.append(normalised)
            weights.append(weight)

        if not scores:
            return 0.0
        return float(np.average(scores, weights=weights))

    @staticmethod
    def _momentum_to_modifier(score: float) -> float:
        if score >= 0.6:
            return 1.15
        if score >= 0.3:
            return 1.05
        if score <= -0.6:
            return 0.85
        if score <= -0.3:
            return 0.95
        return 1.0

    # ----------------------------------------------------------------
    # trend alignment
    # ----------------------------------------------------------------

    @staticmethod
    def _check_alignment(nifty_ret: float, bnf_ret: float) -> str:
        """
        Check if both indices move in the same direction.
        Returns 'ALIGNED_UP', 'ALIGNED_DOWN', 'DIVERGENT', or 'FLAT'.
        """
        nifty_up = nifty_ret > 0.001
        nifty_dn = nifty_ret < -0.001
        bnf_up = bnf_ret > 0.001
        bnf_dn = bnf_ret < -0.001

        if nifty_up and bnf_up:
            return 'ALIGNED_UP'
        if nifty_dn and bnf_dn:
            return 'ALIGNED_DOWN'
        if (nifty_up and bnf_dn) or (nifty_dn and bnf_up):
            return 'DIVERGENT'
        return 'FLAT'

    @staticmethod
    def _alignment_to_modifier(alignment: str) -> float:
        return {
            'ALIGNED_UP': 1.10,
            'ALIGNED_DOWN': 0.90,
            'DIVERGENT': 0.85,
            'FLAT': 1.0,
        }.get(alignment, 1.0)

    # ----------------------------------------------------------------
    # confidence
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        lead_pct: float,
        momentum_score: float,
        alignment: str,
    ) -> float:
        score = 0.0

        # Lead component (0–0.4)
        lead_abs = abs(lead_pct)
        if lead_abs >= 0.005:
            score += 0.4
        elif lead_abs >= 0.003:
            score += 0.25
        elif lead_abs >= 0.001:
            score += 0.1

        # Momentum component (0–0.3)
        mom_abs = abs(momentum_score)
        if mom_abs >= 0.6:
            score += 0.3
        elif mom_abs >= 0.3:
            score += 0.15

        # Alignment component (0–0.3)
        if alignment in ('ALIGNED_UP', 'ALIGNED_DOWN'):
            score += 0.25
        elif alignment == 'DIVERGENT':
            score += 0.1   # divergence is informative too

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
            'nifty_return': 0.0,
            'banknifty_return': 0.0,
            'lead_pct': 0.0,
            'momentum_score': 0.0,
        }

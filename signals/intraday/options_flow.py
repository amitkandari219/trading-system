"""
Options Flow Scanner — intraday OVERLAY signal.

Tracks CE/PE open-interest changes, PCR drift, sudden OI buildup at
individual strikes, and IV skew shifts to produce a directional bias
and sizing modifier.

Logic:
  - PCR shift: compare current PCR to morning (9:30) PCR snapshot.
    Rising PCR  → bearish flow (put buying / call writing) → size_modifier 0.7
    Falling PCR → bullish flow (put writing / call buying) → size_modifier 1.2
  - OI buildup: sudden OI increase at a strike within 2% of spot
    flags support (PE OI) or resistance (CE OI).
  - IV skew: (ATM IV − OTM_put IV). Rising skew = demand for downside
    protection → mildly bearish.

Data expectations (via `context` dict):
  - context['option_chain']  : list[dict] with keys
        strike, ce_oi, pe_oi, ce_iv, pe_iv, ce_ltp, pe_ltp
  - context['morning_pcr']   : float — PCR at 9:30 snapshot
  - context['spot_price']    : float — current Nifty spot
  - context['india_vix']     : float — current India VIX

Safety:
  - Returns neutral (size_modifier=1.0) when option chain is missing.
  - All modifiers clamped to [0.5, 1.5].

Usage:
    from signals.intraday.options_flow import OptionsFlowScanner
    scanner = OptionsFlowScanner()
    result = scanner.evaluate(trade_date, current_time, bar_data, context)
"""

import logging
import math
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
# THRESHOLDS
# ════════════════════════════════════════════════════════════════════
PCR_BULLISH_SHIFT = -0.08      # PCR dropped by this much → bullish
PCR_BEARISH_SHIFT = 0.10       # PCR rose by this much → bearish
PCR_STRONG_BULL = -0.15        # Strong bullish flow
PCR_STRONG_BEAR = 0.20         # Strong bearish flow

OI_BUILDUP_THRESHOLD = 0.25   # 25% OI increase = significant buildup
OI_PROXIMITY_PCT = 0.02       # Strike within 2% of spot is relevant

IV_SKEW_BULL_THRESHOLD = -2.0  # OTM put IV < ATM → mild bullish
IV_SKEW_BEAR_THRESHOLD = 3.0   # OTM put IV >> ATM → bearish protection bid

NIFTY_STRIKE_INTERVAL = 50

MIN_MODIFIER = 0.5
MAX_MODIFIER = 1.5


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert to float gracefully."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _nearest_strike(spot: float, interval: int = NIFTY_STRIKE_INTERVAL) -> float:
    return round(spot / interval) * interval


# ════════════════════════════════════════════════════════════════════
# SIGNAL CLASS
# ════════════════════════════════════════════════════════════════════

class OptionsFlowScanner:
    """
    Intraday overlay that reads option-chain snapshots and computes
    a directional bias + size modifier from flow dynamics.
    """

    SIGNAL_ID = 'INTRADAY_OPTIONS_FLOW'

    def __init__(self, pcr_ema_alpha: float = 0.3):
        self.pcr_ema_alpha = pcr_ema_alpha
        self._pcr_history: List[float] = []  # intraday PCR snapshots

    # ----------------------------------------------------------------
    # public API
    # ----------------------------------------------------------------

    def evaluate(
        self,
        trade_date: date,
        current_time: time,
        bar_data: Dict,
        context: Dict,
    ) -> Dict:
        """
        Evaluate options flow and return overlay result.

        Returns:
            dict with signal_id, direction, confidence, size_modifier,
            reason, pcr_current, pcr_shift, iv_skew, oi_support, oi_resistance
        """
        neutral = self._neutral_result('No option chain data')

        option_chain: Optional[List[Dict]] = context.get('option_chain')
        spot = _safe_float(context.get('spot_price'))
        morning_pcr = _safe_float(context.get('morning_pcr'))

        if not option_chain or spot <= 0:
            return neutral

        # ── 1. Compute current PCR ──────────────────────────────
        total_pe_oi, total_ce_oi = self._aggregate_oi(option_chain)
        if total_ce_oi == 0:
            return neutral

        current_pcr = total_pe_oi / total_ce_oi
        self._pcr_history.append(current_pcr)

        # ── 2. PCR shift from morning ───────────────────────────
        pcr_shift = 0.0
        if morning_pcr > 0:
            pcr_shift = current_pcr - morning_pcr

        pcr_modifier = self._pcr_to_modifier(pcr_shift)
        pcr_direction = self._pcr_direction(pcr_shift)

        # ── 3. OI buildup detection ─────────────────────────────
        oi_support, oi_resistance = self._detect_oi_buildup(
            option_chain, spot,
        )

        oi_modifier = 1.0
        if oi_support and not oi_resistance:
            oi_modifier = 1.05   # PE OI wall below → mild bullish
        elif oi_resistance and not oi_support:
            oi_modifier = 0.95   # CE OI wall above → mild bearish
        # Both walls → neutral (range-bound)

        # ── 4. IV skew ──────────────────────────────────────────
        iv_skew = self._compute_iv_skew(option_chain, spot)
        iv_modifier = self._iv_skew_to_modifier(iv_skew)

        # ── 5. Composite modifier ───────────────────────────────
        raw_modifier = pcr_modifier * oi_modifier * iv_modifier
        size_modifier = _clamp(raw_modifier, MIN_MODIFIER, MAX_MODIFIER)

        # ── 6. Direction and confidence ─────────────────────────
        direction = pcr_direction
        confidence = self._compute_confidence(pcr_shift, iv_skew, oi_support, oi_resistance)

        reason_parts = []
        if abs(pcr_shift) > 0.03:
            reason_parts.append(f'PCR shift {pcr_shift:+.2f}')
        if oi_support:
            reason_parts.append(f'PE OI wall at {oi_support}')
        if oi_resistance:
            reason_parts.append(f'CE OI wall at {oi_resistance}')
        if abs(iv_skew) > 1.0:
            reason_parts.append(f'IV skew {iv_skew:+.1f}')
        reason = '; '.join(reason_parts) if reason_parts else 'Neutral options flow'

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 3),
            'reason': reason,
            'pcr_current': round(current_pcr, 3),
            'pcr_shift': round(pcr_shift, 3),
            'iv_skew': round(iv_skew, 2),
            'oi_support': oi_support,
            'oi_resistance': oi_resistance,
        }

    # ----------------------------------------------------------------
    # internal — PCR
    # ----------------------------------------------------------------

    @staticmethod
    def _aggregate_oi(chain: List[Dict]):
        total_pe = sum(_safe_float(s.get('pe_oi')) for s in chain)
        total_ce = sum(_safe_float(s.get('ce_oi')) for s in chain)
        return total_pe, total_ce

    @staticmethod
    def _pcr_to_modifier(pcr_shift: float) -> float:
        """Map PCR shift to size modifier."""
        if pcr_shift <= PCR_STRONG_BULL:
            return 1.25
        if pcr_shift <= PCR_BULLISH_SHIFT:
            return 1.15
        if pcr_shift >= PCR_STRONG_BEAR:
            return 0.70
        if pcr_shift >= PCR_BEARISH_SHIFT:
            return 0.80
        return 1.0

    @staticmethod
    def _pcr_direction(pcr_shift: float) -> Optional[str]:
        if pcr_shift <= PCR_BULLISH_SHIFT:
            return 'LONG'
        if pcr_shift >= PCR_BEARISH_SHIFT:
            return 'SHORT'
        return None

    # ----------------------------------------------------------------
    # internal — OI buildup
    # ----------------------------------------------------------------

    @staticmethod
    def _detect_oi_buildup(
        chain: List[Dict], spot: float,
    ) -> tuple:
        """
        Find the strike with max PE OI (support) and max CE OI
        (resistance) within OI_PROXIMITY_PCT of spot.

        Returns (support_strike, resistance_strike) — either can be None.
        """
        max_pe_oi, max_ce_oi = 0.0, 0.0
        support_strike = None
        resistance_strike = None

        for s in chain:
            strike = _safe_float(s.get('strike'))
            if strike <= 0:
                continue
            distance_pct = abs(strike - spot) / spot
            if distance_pct > OI_PROXIMITY_PCT:
                continue

            pe_oi = _safe_float(s.get('pe_oi'))
            ce_oi = _safe_float(s.get('ce_oi'))

            if pe_oi > max_pe_oi:
                max_pe_oi = pe_oi
                support_strike = int(strike)
            if ce_oi > max_ce_oi:
                max_ce_oi = ce_oi
                resistance_strike = int(strike)

        # Only flag if OI is meaningfully large (at least 1M total OI)
        if max_pe_oi < 1_000_000:
            support_strike = None
        if max_ce_oi < 1_000_000:
            resistance_strike = None

        return support_strike, resistance_strike

    # ----------------------------------------------------------------
    # internal — IV skew
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_iv_skew(chain: List[Dict], spot: float) -> float:
        """
        IV skew = OTM_put_IV − ATM_IV.
        Positive skew = demand for downside protection (bearish).
        """
        atm_strike = _nearest_strike(spot)
        atm_iv = None
        otm_put_ivs = []

        for s in chain:
            strike = _safe_float(s.get('strike'))
            if strike <= 0:
                continue
            if abs(strike - atm_strike) < 1:
                atm_iv = _safe_float(s.get('pe_iv')) or _safe_float(s.get('ce_iv'))
            # OTM puts: strikes 2-5% below spot
            distance_pct = (spot - strike) / spot
            if 0.02 <= distance_pct <= 0.05:
                piv = _safe_float(s.get('pe_iv'))
                if piv > 0:
                    otm_put_ivs.append(piv)

        if atm_iv is None or atm_iv <= 0 or not otm_put_ivs:
            return 0.0

        avg_otm_put_iv = float(np.mean(otm_put_ivs))
        return avg_otm_put_iv - atm_iv

    @staticmethod
    def _iv_skew_to_modifier(iv_skew: float) -> float:
        if iv_skew >= IV_SKEW_BEAR_THRESHOLD:
            return 0.90   # Downside protection bid → mildly reduce longs
        if iv_skew <= IV_SKEW_BULL_THRESHOLD:
            return 1.05   # No skew fear → mild boost
        return 1.0

    # ----------------------------------------------------------------
    # internal — confidence
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        pcr_shift: float,
        iv_skew: float,
        oi_support: Optional[int],
        oi_resistance: Optional[int],
    ) -> float:
        """Confidence 0-1 based on signal agreement."""
        score = 0.0

        # PCR component (0–0.4)
        pcr_abs = abs(pcr_shift)
        if pcr_abs >= 0.20:
            score += 0.4
        elif pcr_abs >= 0.10:
            score += 0.25
        elif pcr_abs >= 0.05:
            score += 0.1

        # IV skew component (0–0.3)
        skew_abs = abs(iv_skew)
        if skew_abs >= 4.0:
            score += 0.3
        elif skew_abs >= 2.0:
            score += 0.15

        # OI component (0–0.3)
        if oi_support and not oi_resistance:
            score += 0.2
        elif oi_resistance and not oi_support:
            score += 0.2
        if oi_support and oi_resistance:
            score += 0.1  # range-bound, less directional confidence

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
            'pcr_current': None,
            'pcr_shift': 0.0,
            'iv_skew': 0.0,
            'oi_support': None,
            'oi_resistance': None,
        }

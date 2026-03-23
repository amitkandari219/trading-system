"""
Skew Reversal Signal — trades normalization of extreme put/call skew.

Options skew (put IV minus call IV at the same delta) fluctuates around
a mean.  When skew reaches extremes and starts reverting, it signals a
directional move:

    - Extreme put skew (put IV > call IV by 3+ pts) reverting → BULLISH
      (fear subsiding, institutions unwinding hedges).
    - Extreme call skew (call IV > put IV by 2+ pts) reverting → BEARISH
      (euphoria fading, retail call buying drying up).

Signal logic:
    1. Compute current skew = atm_put_iv - atm_call_iv.
    2. Extreme PUT skew: skew > +3 vol points.
    3. Extreme CALL skew: skew < -2 vol points.
    4. Reversion: skew has moved >=1 vol point toward mean over 5 days.
    5. Confidence scales with reversion magnitude and distance from 20d avg.
    6. Proxy mode: when IV data unavailable, use VIX + PCR as proxy.

Data source:
    - Primary: ATM put/call IV from NSE option chain.
    - Proxy: India VIX + put/call ratio.

Usage:
    from signals.structural.skew_reversal import SkewReversalSignal

    sig = SkewReversalSignal()
    result = sig.evaluate({
        'atm_put_iv': 18.5,
        'atm_call_iv': 14.0,
        'prev_day_skew': 5.0,
        'skew_5d_ago': 6.0,
        'avg_skew_20d': 1.5,
    })

Academic basis: Bollen & Whaley (2004) — "Does net buying pressure
affect the shape of implied volatility functions?"  Extreme skew
reverts within 5-10 trading days.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ================================================================
# Constants
# ================================================================

SIGNAL_ID = 'SKEW_REVERSAL'

# Skew extremes (put IV - call IV)
EXTREME_PUT_SKEW = 3.0        # put IV > call IV by 3+ vol pts → extreme fear
EXTREME_CALL_SKEW = -2.0      # call IV > put IV by 2+ vol pts → extreme greed

# Reversion requirement
MIN_REVERSION_PTS = 1.0       # skew must have reverted >= 1 vol pt over 5 days

# Proxy mode thresholds (when IV data unavailable)
PROXY_VIX_HIGH = 18.0         # VIX > 18 suggests elevated put skew
PROXY_VIX_LOW = 12.0          # VIX < 12 suggests potential call skew
PROXY_PCR_HIGH = 1.3          # PCR > 1.3 → heavy put writing
PROXY_PCR_LOW = 0.7           # PCR < 0.7 → heavy call writing

# Confidence
BASE_CONFIDENCE = 0.52
REVERSION_CONF_SCALE = 0.03   # per vol point of reversion
DISTANCE_FROM_MEAN_SCALE = 0.02  # per vol point from 20d avg
MAX_CONFIDENCE = 0.80
PROXY_CONFIDENCE_PENALTY = 0.08  # lower confidence in proxy mode

# Risk management
MAX_HOLD_BARS = 48             # 48 × 5-min = 4 hours
STOP_LOSS_PCT = 0.006          # 0.6%
TARGET_PCT = 0.004             # 0.4%


# ================================================================
# Helpers
# ================================================================

def _safe_float(val: Any, default: float = float('nan')) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _is_valid(val: float) -> bool:
    """Check if a float is valid (not NaN and finite)."""
    return not (math.isnan(val) or math.isinf(val))


# ================================================================
# Signal Class
# ================================================================

class SkewReversalSignal:
    """
    Options skew reversal signal.

    Detects extreme put or call skew that is reverting toward the mean,
    and generates a directional signal based on the reversion direction.
    Falls back to VIX + PCR proxy when IV data is unavailable.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('SkewReversalSignal initialised')

    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------
    def evaluate(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate skew reversal signal.

        Parameters
        ----------
        market_data : dict
            Primary mode keys:
                atm_put_iv       : float — ATM put implied volatility
                atm_call_iv      : float — ATM call implied volatility
                prev_day_skew    : float — yesterday's skew (put_iv - call_iv)
                skew_5d_ago      : float — skew 5 trading days ago
                avg_skew_20d     : float — 20-day average skew
            Proxy/fallback keys:
                india_vix        : float — India VIX
                prev_india_vix   : float — previous day India VIX
                put_call_ratio   : float — put/call OI ratio
                prev_put_call_ratio : float — previous day PCR

        Returns
        -------
        dict with signal details, or None if no signal.
        """
        try:
            return self._evaluate_inner(market_data)
        except Exception as e:
            logger.error(
                'SkewReversalSignal.evaluate error: %s', e, exc_info=True
            )
            return None

    def _evaluate_inner(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # ── Try primary IV-based mode ─────────────────────────────
        put_iv = _safe_float(market_data.get('atm_put_iv'))
        call_iv = _safe_float(market_data.get('atm_call_iv'))

        if _is_valid(put_iv) and _is_valid(call_iv):
            return self._evaluate_iv_mode(market_data, put_iv, call_iv)

        # ── Fallback to proxy mode ────────────────────────────────
        vix = _safe_float(market_data.get('india_vix'))
        pcr = _safe_float(market_data.get('put_call_ratio'))

        if _is_valid(vix) and _is_valid(pcr):
            return self._evaluate_proxy_mode(market_data, vix, pcr)

        logger.debug('SKEW_REVERSAL: insufficient data for both IV and proxy modes')
        return None

    # ----------------------------------------------------------
    # IV-based evaluation
    # ----------------------------------------------------------
    def _evaluate_iv_mode(
        self,
        market_data: Dict[str, Any],
        put_iv: float,
        call_iv: float,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate using actual IV data."""
        current_skew = put_iv - call_iv
        prev_skew = _safe_float(market_data.get('prev_day_skew'))
        skew_5d = _safe_float(market_data.get('skew_5d_ago'))
        avg_skew = _safe_float(market_data.get('avg_skew_20d'), 1.0)

        if not _is_valid(avg_skew):
            avg_skew = 1.0

        # ── Determine if extreme ──────────────────────────────────
        extreme_type = None
        if current_skew >= EXTREME_PUT_SKEW:
            extreme_type = 'EXTREME_PUT_SKEW'
        elif current_skew <= EXTREME_CALL_SKEW:
            extreme_type = 'EXTREME_CALL_SKEW'

        # Also check if was extreme and reverting
        if extreme_type is None and _is_valid(skew_5d):
            if skew_5d >= EXTREME_PUT_SKEW and current_skew < skew_5d:
                extreme_type = 'EXTREME_PUT_SKEW'
            elif skew_5d <= EXTREME_CALL_SKEW and current_skew > skew_5d:
                extreme_type = 'EXTREME_CALL_SKEW'

        if extreme_type is None:
            logger.debug(
                'SKEW_REVERSAL: skew %.2f not extreme (thresholds: +%.1f / %.1f)',
                current_skew, EXTREME_PUT_SKEW, EXTREME_CALL_SKEW,
            )
            return None

        # ── Check reversion ───────────────────────────────────────
        reversion_pts = 0.0
        if _is_valid(skew_5d):
            if extreme_type == 'EXTREME_PUT_SKEW':
                reversion_pts = skew_5d - current_skew  # positive = reverting down
            else:
                reversion_pts = current_skew - skew_5d  # positive = reverting up

        if reversion_pts < MIN_REVERSION_PTS:
            logger.debug(
                'SKEW_REVERSAL: reversion %.2f pts < %.1f minimum',
                reversion_pts, MIN_REVERSION_PTS,
            )
            return None

        # ── Direction ─────────────────────────────────────────────
        if extreme_type == 'EXTREME_PUT_SKEW':
            direction = 'LONG'   # put skew reverting → fear subsiding → bullish
        else:
            direction = 'SHORT'  # call skew reverting → euphoria fading → bearish

        # ── Confidence ────────────────────────────────────────────
        confidence = BASE_CONFIDENCE
        confidence += reversion_pts * REVERSION_CONF_SCALE
        distance_from_mean = abs(current_skew - avg_skew)
        confidence += distance_from_mean * DISTANCE_FROM_MEAN_SCALE
        confidence = min(MAX_CONFIDENCE, max(0.10, confidence))

        # ── Build result ──────────────────────────────────────────
        reason_parts = [
            f"SKEW_REVERSAL ({extreme_type})",
            f"Dir={direction}",
            f"Skew={current_skew:.2f}",
            f"5dAgo={skew_5d:.2f}" if _is_valid(skew_5d) else "5dAgo=N/A",
            f"Reversion={reversion_pts:.2f}pts",
            f"AvgSkew={avg_skew:.2f}",
        ]

        logger.info(
            '%s signal: %s %s skew=%.2f reversion=%.2f conf=%.3f',
            self.SIGNAL_ID, extreme_type, direction,
            current_skew, reversion_pts, confidence,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'confidence': round(confidence, 3),
            'extreme_type': extreme_type,
            'current_skew': round(current_skew, 2),
            'skew_5d_ago': round(skew_5d, 2) if _is_valid(skew_5d) else None,
            'reversion_pts': round(reversion_pts, 2),
            'avg_skew_20d': round(avg_skew, 2),
            'atm_put_iv': round(put_iv, 2),
            'atm_call_iv': round(call_iv, 2),
            'proxy_mode': False,
            'stop_loss_pct': STOP_LOSS_PCT,
            'target_pct': TARGET_PCT,
            'max_hold_bars': MAX_HOLD_BARS,
            'reason': ' | '.join(reason_parts),
        }

    # ----------------------------------------------------------
    # Proxy-based evaluation
    # ----------------------------------------------------------
    def _evaluate_proxy_mode(
        self,
        market_data: Dict[str, Any],
        vix: float,
        pcr: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Proxy evaluation using VIX and PCR when IV data is unavailable.

        High VIX + high PCR → extreme put skew proxy.
        Low VIX + low PCR → extreme call skew proxy.
        """
        prev_vix = _safe_float(market_data.get('prev_india_vix'))
        prev_pcr = _safe_float(market_data.get('prev_put_call_ratio'))

        if not _is_valid(prev_vix) or not _is_valid(prev_pcr):
            return None

        # Determine proxy skew regime
        extreme_type = None

        if vix > PROXY_VIX_HIGH and pcr > PROXY_PCR_HIGH:
            # High fear — proxy for extreme put skew
            # Check reversion: VIX or PCR dropping
            if vix < prev_vix or pcr < prev_pcr:
                extreme_type = 'EXTREME_PUT_SKEW'

        elif vix < PROXY_VIX_LOW and pcr < PROXY_PCR_LOW:
            # Low fear — proxy for extreme call skew
            # Check reversion: VIX or PCR rising
            if vix > prev_vix or pcr > prev_pcr:
                extreme_type = 'EXTREME_CALL_SKEW'

        if extreme_type is None:
            return None

        # Direction
        if extreme_type == 'EXTREME_PUT_SKEW':
            direction = 'LONG'
        else:
            direction = 'SHORT'

        # Confidence — lower in proxy mode
        confidence = BASE_CONFIDENCE - PROXY_CONFIDENCE_PENALTY
        # Small boost from VIX magnitude
        if extreme_type == 'EXTREME_PUT_SKEW':
            confidence += min(0.06, (vix - PROXY_VIX_HIGH) * 0.01)
        else:
            confidence += min(0.06, (PROXY_VIX_LOW - vix) * 0.01)

        confidence = min(MAX_CONFIDENCE, max(0.10, confidence))

        reason_parts = [
            f"SKEW_REVERSAL_PROXY ({extreme_type})",
            f"Dir={direction}",
            f"VIX={vix:.1f}",
            f"PrevVIX={prev_vix:.1f}",
            f"PCR={pcr:.2f}",
            f"PrevPCR={prev_pcr:.2f}",
        ]

        logger.info(
            '%s proxy signal: %s %s vix=%.1f pcr=%.2f conf=%.3f',
            self.SIGNAL_ID, extreme_type, direction, vix, pcr, confidence,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'confidence': round(confidence, 3),
            'extreme_type': extreme_type,
            'current_skew': None,
            'skew_5d_ago': None,
            'reversion_pts': None,
            'avg_skew_20d': None,
            'atm_put_iv': None,
            'atm_call_iv': None,
            'india_vix': round(vix, 2),
            'prev_india_vix': round(prev_vix, 2),
            'put_call_ratio': round(pcr, 2),
            'prev_put_call_ratio': round(prev_pcr, 2),
            'proxy_mode': True,
            'stop_loss_pct': STOP_LOSS_PCT,
            'target_pct': TARGET_PCT,
            'max_hold_bars': MAX_HOLD_BARS,
            'reason': ' | '.join(reason_parts),
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return f"SkewReversalSignal(signal_id='{self.SIGNAL_ID}')"

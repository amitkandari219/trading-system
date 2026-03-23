"""
SIP Flow Signal — MF SIP deployment date detector.

Mutual Fund SIP (Systematic Investment Plan) deployments occur on fixed
dates each month: 1st, 5th, 7th, 10th (primary — ~60% of AUM), and
15th, 20th, 25th (secondary).  These create predictable buy-side flow,
especially in falling markets where DII buying offsets FII selling.

Signal logic:
    1. Always BULLISH on SIP dates (institutional buy flow).
    2. Primary dates (1/5/7/10) score higher than secondary (15/20/25).
    3. Falling markets (negative 5-day return) amplify the signal.
    4. High VIX (fear) also amplifies — DIIs buying into fear.
    5. Score components:
        - Date type: primary=3, secondary=2
        - 5d return < -2% → +2, < -1% → +1
        - VIX > 18 → +1, VIX > 22 → +2
        - Monthly SIP flow > 22000 Cr → +1
    6. Score >= 5 → entry, else skip.
    7. Hold ~3 hours (36 × 5-min bars), SL 0.5%, target 0.3-0.5%.
    8. If date + 1 is also a SIP date (spillover), confidence +0.03.

Data source:
    - Date-based calendar signal — no live data needed.
    - Optional: VIX, 5-day return, monthly SIP flow for scoring.

Usage:
    from signals.structural.sip_flow import SIPFlowSignal

    sig = SIPFlowSignal()
    result = sig.evaluate({
        'date': date(2026, 3, 5),
        'nifty_5d_return_pct': -1.5,
        'india_vix': 19.0,
    })

Academic basis: Predictable institutional flow from SIP deployments —
Amihud & Mendelson (1986) liquidity premium theory applied to periodic
mandatory buy flows.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ================================================================
# Constants
# ================================================================

SIGNAL_ID = 'SIP_FLOW'

# SIP deployment dates within a month
PRIMARY_SIP_DATES = {1, 5, 7, 10}     # ~60% of SIP AUM deployed
SECONDARY_SIP_DATES = {15, 20, 25}    # remaining deployments

ALL_SIP_DATES = PRIMARY_SIP_DATES | SECONDARY_SIP_DATES

# Scoring
DATE_SCORE_PRIMARY = 3
DATE_SCORE_SECONDARY = 2

RETURN_5D_STRONG_THRESHOLD = -2.0     # 5d return < -2% → +2
RETURN_5D_MODERATE_THRESHOLD = -1.0   # 5d return < -1% → +1

VIX_HIGH_THRESHOLD = 22.0            # VIX > 22 → +2
VIX_ELEVATED_THRESHOLD = 18.0        # VIX > 18 → +1

MONTHLY_SIP_FLOW_BOOST = 22000       # Monthly SIP flow > 22000 Cr → +1

SCORE_ENTRY_THRESHOLD = 5

# Risk management
MAX_HOLD_BARS = 36                    # 36 × 5-min = 3 hours
STOP_LOSS_PCT = 0.005                 # 0.5%
TARGET_BASE_PCT = 0.003               # 0.3% base target
TARGET_HIGH_PCT = 0.005               # 0.5% in strong setups

# Confidence
BASE_CONFIDENCE = 0.50
SPILLOVER_BOOST = 0.03                # next day also SIP date
FALLING_MARKET_CONF_BOOST = 0.04
HIGH_VIX_CONF_BOOST = 0.03


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


def _is_sip_date(d: date) -> bool:
    """Check if the given date is a SIP deployment date."""
    return d.day in ALL_SIP_DATES


def _is_primary_sip_date(d: date) -> bool:
    """Check if the given date is a primary SIP date."""
    return d.day in PRIMARY_SIP_DATES


def _next_day_is_sip(d: date) -> bool:
    """Check if the next calendar day is also a SIP date."""
    next_day = d + timedelta(days=1)
    return next_day.day in ALL_SIP_DATES


# ================================================================
# Signal Class
# ================================================================

class SIPFlowSignal:
    """
    MF SIP deployment date signal.

    Always BULLISH on SIP dates.  Stronger in falling markets where
    DII buying (via SIP deployments) offsets FII selling pressure.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('SIPFlowSignal initialised')

    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------
    def evaluate(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate SIP flow signal.

        Parameters
        ----------
        market_data : dict
            Required keys:
                date                   : date — trading date
            Optional keys:
                nifty_5d_return_pct    : float — 5-day Nifty return in % (e.g. -1.5)
                india_vix              : float — India VIX level
                monthly_sip_flow_crore : float — monthly SIP flow in crore (default 20000)

        Returns
        -------
        dict with signal details, or None if no signal.
        """
        try:
            return self._evaluate_inner(market_data)
        except Exception as e:
            logger.error(
                'SIPFlowSignal.evaluate error: %s', e, exc_info=True
            )
            return None

    def _evaluate_inner(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # ── Extract date ─────────────────────────────────────────
        trade_date = market_data.get('date')
        if trade_date is None:
            logger.debug('SIP_FLOW: no date provided')
            return None

        # ── Check if SIP date ─────────────────────────────────────
        if not _is_sip_date(trade_date):
            logger.debug('SIP_FLOW: %s is not a SIP date', trade_date)
            return None

        # ── Extract optional fields ──────────────────────────────
        ret_5d = _safe_float(market_data.get('nifty_5d_return_pct'), 0.0)
        if math.isnan(ret_5d):
            ret_5d = 0.0

        vix = _safe_float(market_data.get('india_vix'), 15.0)
        if math.isnan(vix):
            vix = 15.0

        sip_flow = _safe_float(market_data.get('monthly_sip_flow_crore'), 20000.0)
        if math.isnan(sip_flow):
            sip_flow = 20000.0

        # ── Score computation ─────────────────────────────────────
        is_primary = _is_primary_sip_date(trade_date)
        date_score = DATE_SCORE_PRIMARY if is_primary else DATE_SCORE_SECONDARY

        # 5-day return score (falling market = stronger signal)
        return_score = 0
        if ret_5d < RETURN_5D_STRONG_THRESHOLD:
            return_score = 2
        elif ret_5d < RETURN_5D_MODERATE_THRESHOLD:
            return_score = 1

        # VIX score
        vix_score = 0
        if vix > VIX_HIGH_THRESHOLD:
            vix_score = 2
        elif vix > VIX_ELEVATED_THRESHOLD:
            vix_score = 1

        # SIP flow score
        flow_score = 1 if sip_flow > MONTHLY_SIP_FLOW_BOOST else 0

        # Composite score
        composite_score = date_score + return_score + vix_score + flow_score

        # ── Entry threshold ───────────────────────────────────────
        if composite_score < SCORE_ENTRY_THRESHOLD:
            logger.debug(
                'SIP_FLOW: score %d < %d — skip',
                composite_score, SCORE_ENTRY_THRESHOLD,
            )
            return None

        # ── Direction: always BULLISH ─────────────────────────────
        direction = 'LONG'

        # ── Confidence ────────────────────────────────────────────
        confidence = BASE_CONFIDENCE

        if ret_5d < RETURN_5D_MODERATE_THRESHOLD:
            confidence += FALLING_MARKET_CONF_BOOST

        if vix > VIX_ELEVATED_THRESHOLD:
            confidence += HIGH_VIX_CONF_BOOST

        # Spillover boost: next day also SIP date
        spillover = _next_day_is_sip(trade_date)
        if spillover:
            confidence += SPILLOVER_BOOST

        confidence = min(0.85, max(0.10, confidence))

        # ── Target ────────────────────────────────────────────────
        if composite_score >= 7:
            target_pct = TARGET_HIGH_PCT
        else:
            target_pct = TARGET_BASE_PCT

        # ── Date type label ───────────────────────────────────────
        date_type = 'PRIMARY' if is_primary else 'SECONDARY'

        # ── Reason string ─────────────────────────────────────────
        reason_parts = [
            f"SIP_FLOW ({date_type})",
            f"Dir={direction}",
            f"Score={composite_score}",
            f"Day={trade_date.day}",
            f"5dRet={ret_5d:+.1f}%",
            f"VIX={vix:.1f}",
            f"SIPFlow={sip_flow:.0f}Cr",
        ]
        if spillover:
            reason_parts.append('SPILLOVER')

        logger.info(
            '%s signal: %s score=%d date_type=%s conf=%.3f',
            self.SIGNAL_ID, direction, composite_score, date_type, confidence,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'confidence': round(confidence, 3),
            'composite_score': composite_score,
            'date_type': date_type,
            'date_score': date_score,
            'return_score': return_score,
            'vix_score': vix_score,
            'flow_score': flow_score,
            'trade_date': trade_date.isoformat(),
            'nifty_5d_return_pct': round(ret_5d, 2),
            'india_vix': round(vix, 2),
            'monthly_sip_flow_crore': round(sip_flow, 0),
            'stop_loss_pct': STOP_LOSS_PCT,
            'target_pct': target_pct,
            'max_hold_bars': MAX_HOLD_BARS,
            'spillover': spillover,
            'reason': ' | '.join(reason_parts),
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return f"SIPFlowSignal(signal_id='{self.SIGNAL_ID}')"

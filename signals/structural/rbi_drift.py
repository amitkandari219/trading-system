"""
RBI Drift Signal — pre-RBI event institutional drift.

Institutional traders pre-position 30-60 minutes before RBI MPC
announcements (typically at 10:00 AM IST).  This creates a detectable
drift between 9:30-10:00 AM on MPC decision days.

Signal logic:
    1. Only fires on RBI MPC decision dates (hardcoded 2025-2026).
    2. Entry window: 9:30 AM (after initial 15-min price discovery).
    3. Exit: 10:15 AM (post-announcement, wait for dust to settle).
    4. Consensus HOLD or CUT → bullish drift (market expects benign outcome).
    5. Consensus HIKE → bearish drift (market expects tightening).
    6. Confirm drift direction matches consensus direction.
    7. Also fires on RBI minutes release days (lower confidence).

Data source:
    - Date calendar for MPC dates.
    - Price at 9:15 and 9:30 for drift direction detection.
    - Optional: RBI consensus from Bloomberg/Reuters surveys.

Usage:
    from signals.structural.rbi_drift import RBIDriftSignal

    sig = RBIDriftSignal()
    result = sig.evaluate({
        'date': date(2026, 2, 7),
        'price_at_915': 24100,
        'price_at_930': 24130,
        'prev_close': 24050,
        'india_vix': 16.5,
        'rbi_consensus': 'HOLD',
    })

Academic basis: Pre-announcement drift in monetary policy — Lucca &
Moench (2015) "The Pre-FOMC Announcement Drift" adapted to RBI MPC.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ================================================================
# Constants
# ================================================================

SIGNAL_ID = 'RBI_DRIFT'

# RBI MPC decision dates (2025-2026)
RBI_MPC_DATES_2025: List[date] = [
    date(2025, 2, 7),
    date(2025, 4, 9),
    date(2025, 6, 6),
    date(2025, 8, 8),
    date(2025, 10, 8),
    date(2025, 12, 5),
]

RBI_MPC_DATES_2026: List[date] = [
    date(2026, 2, 7),
    date(2026, 4, 9),
    date(2026, 6, 5),
    date(2026, 8, 6),
    date(2026, 10, 8),
    date(2026, 12, 5),
]

RBI_MPC_DATES: Set[date] = set(RBI_MPC_DATES_2025 + RBI_MPC_DATES_2026)

# RBI minutes release dates (approximately 2 weeks after MPC)
# These are approximate — actual dates confirmed closer to time.
RBI_MINUTES_DATES_2026: List[date] = [
    date(2026, 2, 21),
    date(2026, 4, 23),
    date(2026, 6, 19),
    date(2026, 8, 20),
    date(2026, 10, 22),
    date(2026, 12, 19),
]

RBI_MINUTES_DATES: Set[date] = set(RBI_MINUTES_DATES_2026)

# Consensus → expected drift direction
CONSENSUS_DIRECTION = {
    'HOLD': 'LONG',    # hold = benign → bullish drift
    'CUT': 'LONG',     # rate cut = stimulative → bullish drift
    'HIKE': 'SHORT',   # rate hike = restrictive → bearish drift
}

# Drift confirmation threshold
MIN_DRIFT_PCT = 0.05           # 0.05% minimum drift from 9:15 to 9:30

# Confidence
BASE_CONFIDENCE_MPC = 0.58
BASE_CONFIDENCE_MINUTES = 0.45  # minutes days are less impactful
VIX_HIGH_BOOST = 0.04          # VIX > 18 → more pre-positioning
STRONG_DRIFT_BOOST = 0.03      # drift > 0.15% → stronger signal
CONSENSUS_PENALTY_CONTRARIAN = 0.08  # drift against consensus → lower confidence

# Risk management
MAX_HOLD_BARS = 9              # 9 × 5-min = 45 min (9:30 to 10:15)
STOP_LOSS_PCT = 0.004          # 0.4%
TARGET_PCT = 0.003             # 0.3%


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


def _is_rbi_mpc_date(d: date) -> bool:
    """Check if date is an RBI MPC decision date."""
    return d in RBI_MPC_DATES


def _is_rbi_minutes_date(d: date) -> bool:
    """Check if date is an RBI minutes release date."""
    return d in RBI_MINUTES_DATES


# ================================================================
# Signal Class
# ================================================================

class RBIDriftSignal:
    """
    Pre-RBI event drift signal.

    Detects institutional pre-positioning 30-60 minutes before RBI MPC
    announcements.  Entry at 9:30, exit at 10:15 (post-announcement).
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        logger.info('RBIDriftSignal initialised')

    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------
    def evaluate(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate RBI drift signal.

        Parameters
        ----------
        market_data : dict
            Required keys:
                date              : date — trading date
                price_at_915      : float — Nifty price at 9:15 AM
                price_at_930      : float — Nifty price at 9:30 AM
                prev_close        : float — previous session close
                india_vix         : float — India VIX
            Optional keys:
                rbi_consensus     : str — 'HOLD', 'CUT', or 'HIKE' (default 'HOLD')
                is_rbi_minutes_day: bool — override for minutes day detection

        Returns
        -------
        dict with signal details, or None if no signal.
        """
        try:
            return self._evaluate_inner(market_data)
        except Exception as e:
            logger.error(
                'RBIDriftSignal.evaluate error: %s', e, exc_info=True
            )
            return None

    def _evaluate_inner(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # ── Extract date ─────────────────────────────────────────
        trade_date = market_data.get('date')
        if trade_date is None:
            return None

        # ── Check if RBI day ─────────────────────────────────────
        is_mpc = _is_rbi_mpc_date(trade_date)
        is_minutes = market_data.get('is_rbi_minutes_day')
        if is_minutes is None:
            is_minutes = _is_rbi_minutes_date(trade_date)

        if not is_mpc and not is_minutes:
            logger.debug('RBI_DRIFT: %s is not an RBI event day', trade_date)
            return None

        event_type = 'MPC_DECISION' if is_mpc else 'MPC_MINUTES'

        # ── Extract prices ───────────────────────────────────────
        price_915 = _safe_float(market_data.get('price_at_915'))
        price_930 = _safe_float(market_data.get('price_at_930'))
        prev_close = _safe_float(market_data.get('prev_close'))
        vix = _safe_float(market_data.get('india_vix'), 15.0)

        if math.isnan(vix):
            vix = 15.0

        if math.isnan(price_915) or price_915 <= 0:
            return None
        if math.isnan(price_930) or price_930 <= 0:
            return None
        if math.isnan(prev_close) or prev_close <= 0:
            return None

        # ── Compute drift ────────────────────────────────────────
        drift_pct = (price_930 - price_915) / price_915 * 100.0
        gap_pct = (price_915 - prev_close) / prev_close * 100.0

        # Direction from drift
        if drift_pct > 0:
            drift_direction = 'LONG'
        elif drift_pct < 0:
            drift_direction = 'SHORT'
        else:
            return None  # no drift detected

        # ── Minimum drift check ───────────────────────────────────
        if abs(drift_pct) < MIN_DRIFT_PCT:
            logger.debug(
                'RBI_DRIFT: drift %.4f%% < %.2f%% minimum',
                drift_pct, MIN_DRIFT_PCT,
            )
            return None

        # ── Consensus ─────────────────────────────────────────────
        consensus = market_data.get('rbi_consensus', 'HOLD')
        if consensus not in CONSENSUS_DIRECTION:
            consensus = 'HOLD'

        expected_direction = CONSENSUS_DIRECTION[consensus]
        is_contrarian = (drift_direction != expected_direction)

        # Take the drift direction (what the market is actually doing)
        direction = drift_direction

        # ── Confidence ────────────────────────────────────────────
        if is_mpc:
            confidence = BASE_CONFIDENCE_MPC
        else:
            confidence = BASE_CONFIDENCE_MINUTES

        # Contrarian penalty
        if is_contrarian:
            confidence -= CONSENSUS_PENALTY_CONTRARIAN

        # VIX boost
        if vix > 18.0:
            confidence += VIX_HIGH_BOOST

        # Strong drift boost
        if abs(drift_pct) > 0.15:
            confidence += STRONG_DRIFT_BOOST

        confidence = min(0.85, max(0.10, confidence))

        # ── Risk management ───────────────────────────────────────
        entry_price = price_930
        if direction == 'LONG':
            stop_loss = round(entry_price * (1 - STOP_LOSS_PCT), 2)
            target = round(entry_price * (1 + TARGET_PCT), 2)
        else:
            stop_loss = round(entry_price * (1 + STOP_LOSS_PCT), 2)
            target = round(entry_price * (1 - TARGET_PCT), 2)

        # ── Reason string ─────────────────────────────────────────
        reason_parts = [
            f"RBI_DRIFT ({event_type})",
            f"Dir={direction}",
            f"Drift={drift_pct:+.3f}%",
            f"Consensus={consensus}",
            f"VIX={vix:.1f}",
            f"Gap={gap_pct:+.2f}%",
        ]
        if is_contrarian:
            reason_parts.append('CONTRARIAN')

        logger.info(
            '%s signal: %s %s drift=%.3f%% consensus=%s conf=%.3f',
            self.SIGNAL_ID, event_type, direction, drift_pct,
            consensus, confidence,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'confidence': round(confidence, 3),
            'event_type': event_type,
            'trade_date': trade_date.isoformat(),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'drift_pct': round(drift_pct, 4),
            'gap_pct': round(gap_pct, 4),
            'price_at_915': round(price_915, 2),
            'price_at_930': round(price_930, 2),
            'prev_close': round(prev_close, 2),
            'india_vix': round(vix, 2),
            'rbi_consensus': consensus,
            'is_contrarian': is_contrarian,
            'max_hold_bars': MAX_HOLD_BARS,
            'reason': ' | '.join(reason_parts),
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def __repr__(self) -> str:
        return f"RBIDriftSignal(signal_id='{self.SIGNAL_ID}')"

"""
IV Rank & Percentile Filter for L8 Options Strategies.

Uses India VIX (from nifty_daily table) as proxy for Nifty implied
volatility.  Computes IV rank, IV percentile over a 252-day rolling
window, and maps each strategy type to an IV-regime action
(BLOCK / ALLOW / PREFER) with an optional size boost.

Usage:
    from signals.iv_rank_filter import IVRankFilter
    filt = IVRankFilter()
    filt.load_vix_history(conn, as_of=date.today())
    result = filt.check_strategy('SHORT_STRANGLE', current_vix=18.5)
"""

import logging
from collections import deque
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import psycopg2

from config.settings import DATABASE_DSN

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

ROLLING_WINDOW = 252  # trading days ≈ 1 calendar year

# Regime boundary percentiles (of the 252-day VIX distribution)
REGIME_LOW_UPPER = 25
REGIME_MID_LOW_UPPER = 50
REGIME_MID_HIGH_UPPER = 75

# Size-boost multipliers per action
SIZE_MULTIPLIER = {
    'BLOCK': 0.0,
    'ALLOW': 1.0,
    'PREFER': 1.2,
}

# ================================================================
# STRATEGY – IV REGIME MATRIX
# ================================================================
# Each strategy maps to an action for four IV regime buckets.
# 'low'      = IV percentile <  25th
# 'mid_low'  = IV percentile 25-50th
# 'mid_high' = IV percentile 50-75th
# 'high'     = IV percentile >  75th

IV_REGIME_MATRIX: Dict[str, Dict[str, str]] = {
    'SHORT_STRADDLE':   {'low': 'BLOCK',  'mid_low': 'BLOCK',  'mid_high': 'ALLOW',  'high': 'PREFER'},
    'SHORT_STRANGLE':   {'low': 'BLOCK',  'mid_low': 'BLOCK',  'mid_high': 'ALLOW',  'high': 'PREFER'},
    'IRON_CONDOR':      {'low': 'BLOCK',  'mid_low': 'ALLOW',  'mid_high': 'PREFER', 'high': 'PREFER'},
    'BULL_PUT_SPREAD':  {'low': 'ALLOW',  'mid_low': 'ALLOW',  'mid_high': 'PREFER', 'high': 'ALLOW'},
    'BEAR_CALL_SPREAD': {'low': 'ALLOW',  'mid_low': 'ALLOW',  'mid_high': 'PREFER', 'high': 'ALLOW'},
    'CALENDAR_SPREAD':  {'low': 'PREFER', 'mid_low': 'ALLOW',  'mid_high': 'BLOCK',  'high': 'BLOCK'},
    'PROTECTIVE_PUT':   {'low': 'ALLOW',  'mid_low': 'ALLOW',  'mid_high': 'PREFER', 'high': 'PREFER'},
    'COVERED_CALL':     {'low': 'ALLOW',  'mid_low': 'PREFER', 'mid_high': 'ALLOW',  'high': 'BLOCK'},
    'RATIO_SPREAD':     {'low': 'BLOCK',  'mid_low': 'ALLOW',  'mid_high': 'PREFER', 'high': 'PREFER'},
}


# ================================================================
# IVRankFilter
# ================================================================

class IVRankFilter:
    """Rolling IV rank / percentile calculator and strategy gatekeeper."""

    def __init__(self, db_conn=None):
        self._vix_history: deque = deque(maxlen=ROLLING_WINDOW)
        self._db_conn = db_conn

    # ── VIX history management ─────────────────────────────────

    def load_vix_history(self, db_conn=None, as_of: Optional[date] = None) -> int:
        """Load last 252 trading days of india_vix from nifty_daily.

        Returns the number of rows loaded.
        """
        conn = db_conn or self._db_conn
        if conn is None:
            raise ValueError("No database connection provided")

        as_of = as_of or date.today()

        query = """
            SELECT india_vix
            FROM   nifty_daily
            WHERE  india_vix IS NOT NULL
              AND  date <= %s
            ORDER  BY date DESC
            LIMIT  %s
        """
        with conn.cursor() as cur:
            cur.execute(query, (as_of, ROLLING_WINDOW))
            rows = cur.fetchall()

        # Rows arrive newest-first; reverse so oldest is at the front.
        self._vix_history.clear()
        for (vix_val,) in reversed(rows):
            self._vix_history.append(float(vix_val))

        logger.info("Loaded %d VIX observations (window=%d)", len(self._vix_history), ROLLING_WINDOW)
        return len(self._vix_history)

    def update_vix(self, vix_value: float) -> None:
        """Append today's VIX to the rolling history.

        The deque automatically trims to 252 entries.
        """
        if vix_value <= 0:
            logger.warning("Ignoring non-positive VIX value: %s", vix_value)
            return
        self._vix_history.append(float(vix_value))

    # ── Core calculations ──────────────────────────────────────

    def compute_iv_rank(self, current_vix: float) -> float:
        """IV Rank = (current - 52w_low) / (52w_high - 52w_low) × 100.

        Returns 0–100.  Returns 50.0 if history is empty or range is zero.
        """
        if len(self._vix_history) < 2:
            logger.warning("Insufficient VIX history (%d pts); returning 50.0", len(self._vix_history))
            return 50.0

        low_52w = min(self._vix_history)
        high_52w = max(self._vix_history)
        spread = high_52w - low_52w

        if spread == 0:
            return 50.0

        rank = (current_vix - low_52w) / spread * 100.0
        return max(0.0, min(100.0, rank))

    def compute_iv_percentile(self, current_vix: float) -> float:
        """What % of days in the last 252 had a *lower* VIX than current.

        Returns 0–100.
        """
        if len(self._vix_history) == 0:
            logger.warning("Empty VIX history; returning 50.0")
            return 50.0

        count_below = sum(1 for v in self._vix_history if v < current_vix)
        return count_below / len(self._vix_history) * 100.0

    # ── Regime classification ──────────────────────────────────

    def _classify_regime(self, iv_percentile: float) -> str:
        """Map an IV percentile to a regime bucket name."""
        if iv_percentile < REGIME_LOW_UPPER:
            return 'low'
        elif iv_percentile < REGIME_MID_LOW_UPPER:
            return 'mid_low'
        elif iv_percentile < REGIME_MID_HIGH_UPPER:
            return 'mid_high'
        else:
            return 'high'

    # ── Public API ─────────────────────────────────────────────

    def check_strategy(self, strategy_type: str, current_vix: float) -> Dict[str, Any]:
        """Evaluate whether *strategy_type* should fire at *current_vix*.

        Returns
        -------
        dict with keys:
            action        : 'BLOCK' | 'ALLOW' | 'PREFER'
            iv_rank       : float 0-100
            iv_percentile : float 0-100
            regime        : str   ('low', 'mid_low', 'mid_high', 'high')
            size_boost    : float (0.0 for BLOCK, 1.0 for ALLOW, 1.2 for PREFER)
        """
        iv_rank = self.compute_iv_rank(current_vix)
        iv_pctl = self.compute_iv_percentile(current_vix)
        regime = self._classify_regime(iv_pctl)

        strategy_key = strategy_type.upper()
        regime_map = IV_REGIME_MATRIX.get(strategy_key)

        if regime_map is None:
            logger.warning("Unknown strategy '%s'; defaulting to ALLOW", strategy_type)
            action = 'ALLOW'
        else:
            action = regime_map.get(regime, 'ALLOW')

        size_boost = SIZE_MULTIPLIER[action]

        return {
            'action': action,
            'iv_rank': round(iv_rank, 2),
            'iv_percentile': round(iv_pctl, 2),
            'regime': regime,
            'size_boost': size_boost,
        }

    def get_status(self) -> Dict[str, Any]:
        """Snapshot of current IV state (needs a current VIX to be meaningful).

        Uses the most recent VIX in history as "current".
        """
        if len(self._vix_history) == 0:
            return {
                'current_vix': None,
                'iv_rank': None,
                'iv_percentile': None,
                'regime': None,
                'history_len': 0,
            }

        current = self._vix_history[-1]
        iv_rank = self.compute_iv_rank(current)
        iv_pctl = self.compute_iv_percentile(current)

        return {
            'current_vix': round(current, 2),
            'iv_rank': round(iv_rank, 2),
            'iv_percentile': round(iv_pctl, 2),
            'regime': self._classify_regime(iv_pctl),
            'history_len': len(self._vix_history),
            'vix_52w_high': round(max(self._vix_history), 2),
            'vix_52w_low': round(min(self._vix_history), 2),
        }


# ================================================================
# CLI
# ================================================================

if __name__ == '__main__':
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

    try:
        conn = psycopg2.connect(DATABASE_DSN)
    except Exception as exc:
        logger.error("Cannot connect to DB (%s): %s", DATABASE_DSN, exc)
        sys.exit(1)

    filt = IVRankFilter(db_conn=conn)

    as_of = date.today()
    if len(sys.argv) > 1:
        as_of = date.fromisoformat(sys.argv[1])

    n = filt.load_vix_history(conn, as_of=as_of)
    if n == 0:
        logger.error("No VIX data found up to %s", as_of)
        sys.exit(1)

    status = filt.get_status()
    print("\n── IV Rank Status ──")
    print(json.dumps(status, indent=2))

    current_vix = status['current_vix']
    print("\n── Strategy Regime Matrix ──")
    for strategy in sorted(IV_REGIME_MATRIX):
        result = filt.check_strategy(strategy, current_vix)
        flag = '✗' if result['action'] == 'BLOCK' else ('★' if result['action'] == 'PREFER' else '·')
        print(f"  {flag} {strategy:<22s}  {result['action']:<7s}  boost={result['size_boost']:.1f}x")

    conn.close()

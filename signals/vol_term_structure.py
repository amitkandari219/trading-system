"""
Volatility Term Structure Signal.

India VIX futures term structure (front vs back month) reveals market
complacency vs fear, with transitions being early warning for regime shifts.

Signal logic:
  Term structure = (Back_Month_VIX - Front_Month_VIX) / Front_Month_VIX

  States:
    CONTANGO (back > front, ratio > 0.05):
      Normal state — market expects higher future vol — COMPLACENT
      → Continuation of current trend, fade vol spikes

    FLAT (ratio -0.05 to 0.05):
      Transition state — market uncertain about vol direction
      → Cautious, reduce sizing

    BACKWARDATION (front > back, ratio < -0.05):
      Abnormal state — immediate fear > future fear — STRESSED
      → Potential crash or already in crisis, momentum strategies work

  Transitions (most powerful signals):
    CONTANGO → BACKWARDATION: Early warning of regime shift → REDUCE ALL
    BACKWARDATION → CONTANGO: Crisis ending → INCREASE sizing

  Additional: VIX term slope (linear regression across tenors)
    Steep contango → very complacent, mean-reversion fade trades
    Steep backwardation → max fear, potential capitulation buy

Data source:
  - India VIX: from india_vix table (daily)
  - VIX futures: NSE VIX futures (if available) or synthetic from options IV
  - Approximation: ATM IV of near-month vs next-month options

Usage:
    from signals.vol_term_structure import VolTermStructureSignal
    sig = VolTermStructureSignal(db_conn=conn)
    result = sig.evaluate(trade_date=date.today())
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# THRESHOLDS
# ================================================================
CONTANGO_THRESHOLD = 0.05     # >5% = contango
BACKWARDATION_THRESHOLD = -0.05  # <-5% = backwardation
STEEP_CONTANGO = 0.15         # >15% = steep contango
STEEP_BACKWARDATION = -0.15   # <-15% = steep backwardation

# Transition detection
TRANSITION_LOOKBACK = 5  # days to detect state change

# Size modifiers
SIZE_MAP = {
    'STEEP_CONTANGO': 1.15,       # Complacent → fade vol, continue trend
    'CONTANGO': 1.05,
    'FLAT': 0.90,                 # Uncertain → reduce
    'BACKWARDATION': 0.75,        # Stressed → reduce significantly
    'STEEP_BACKWARDATION': 0.60,  # Crisis — risk-off
}

# Transition modifiers (applied on top of state)
TRANSITION_CONTANGO_TO_BACK = 0.70    # Regime shift warning → reduce 30%
TRANSITION_BACK_TO_CONTANGO = 1.20    # Crisis ending → increase 20%


@dataclass
class VolTermContext:
    """Evaluation result from volatility term structure signal."""
    front_month_iv: float         # Near-month ATM IV
    back_month_iv: float          # Next-month ATM IV
    term_spread: float            # (back - front) / front
    structure_state: str          # STEEP_CONTANGO, CONTANGO, FLAT, BACKWARDATION, STEEP_BACKWARDATION
    prev_state: str               # Previous state for transition detection
    transition: str               # NONE, CONTANGO_TO_BACK, BACK_TO_CONTANGO, etc.
    vix_level: float
    regime_warning: bool          # True if transition detected
    direction: str
    confidence: float
    size_modifier: float
    reason: str

    def to_dict(self) -> Dict:
        return {
            'signal_id': 'VOL_TERM_STRUCTURE',
            'front_month_iv': round(self.front_month_iv, 2),
            'back_month_iv': round(self.back_month_iv, 2),
            'term_spread': round(self.term_spread, 4),
            'structure_state': self.structure_state,
            'structure_zone': self.structure_state,  # Alias for meta-learner
            'prev_state': self.prev_state,
            'transition': self.transition,
            'vix_level': round(self.vix_level, 2),
            'regime_warning': self.regime_warning,
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'size_modifier': round(self.size_modifier, 2),
            'reason': self.reason,
        }

    def to_telegram(self) -> str:
        emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}.get(
            self.direction, '⚪')
        warn = ' ⚠️REGIME_SHIFT' if self.regime_warning else ''
        return (
            f"{emoji} Vol Term Structure{warn}\n"
            f"  Front IV: {self.front_month_iv:.1f}% | Back IV: {self.back_month_iv:.1f}%\n"
            f"  Spread: {self.term_spread*100:+.1f}% ({self.structure_state})\n"
            f"  Transition: {self.transition}\n"
            f"  VIX: {self.vix_level:.1f}\n"
            f"  Dir: {self.direction} | Size: {self.size_modifier:.2f}x"
        )


class VolTermStructureSignal:
    """
    Volatility term structure signal.

    Detects contango/backwardation in VIX term structure and
    transitions between states as early regime warnings.
    """

    SIGNAL_ID = 'VOL_TERM_STRUCTURE'

    def __init__(self, db_conn=None):
        self.conn = db_conn

    def _get_conn(self):
        if self.conn:
            try:
                if not self.conn.closed:
                    return self.conn
            except Exception:
                pass
        try:
            import psycopg2
            from config.settings import DATABASE_DSN
            self.conn = psycopg2.connect(DATABASE_DSN)
            return self.conn
        except Exception as e:
            logger.error("DB connection failed: %s", e)
            return None

    # ----------------------------------------------------------
    # Data retrieval
    # ----------------------------------------------------------
    def _get_atm_iv_by_expiry(
        self, trade_date: date
    ) -> Optional[Dict]:
        """
        Get ATM implied volatility for near-month and next-month options.

        Returns dict with front_iv, back_iv.
        """
        conn = self._get_conn()
        if not conn:
            return None

        try:
            # Get the two nearest expiries with ATM IV data
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT expiry,
                           AVG(implied_volatility) as avg_iv
                    FROM nifty_options
                    WHERE date = %s
                      AND implied_volatility > 0
                      AND ABS(delta) BETWEEN 0.35 AND 0.65
                    GROUP BY expiry
                    ORDER BY expiry
                    LIMIT 2
                    """,
                    (trade_date,)
                )
                rows = cur.fetchall()

            if len(rows) >= 2:
                return {
                    'front_iv': float(rows[0][1]),
                    'back_iv': float(rows[1][1]),
                    'front_expiry': rows[0][0],
                    'back_expiry': rows[1][0],
                }
            elif len(rows) == 1:
                # Only one expiry available
                return {
                    'front_iv': float(rows[0][1]),
                    'back_iv': float(rows[0][1]) * 1.02,  # Assume mild contango
                    'front_expiry': rows[0][0],
                    'back_expiry': None,
                }
        except Exception as e:
            logger.error("Failed to fetch ATM IV: %s", e)

        return None

    def _get_vix_history(
        self, trade_date: date, lookback: int = 10
    ) -> Optional[pd.DataFrame]:
        """Fetch India VIX history."""
        conn = self._get_conn()
        if not conn:
            return None

        start_date = trade_date - timedelta(days=lookback * 2)

        try:
            df = pd.read_sql(
                """
                SELECT date, close as vix FROM india_vix
                WHERE date BETWEEN %s AND %s
                ORDER BY date
                """,
                conn, params=(start_date, trade_date)
            )
            return df if len(df) > 0 else None
        except Exception:
            return None

    def _get_previous_state(self, trade_date: date) -> str:
        """Get the term structure state from TRANSITION_LOOKBACK days ago."""
        conn = self._get_conn()
        if not conn:
            return 'UNKNOWN'

        try:
            prev_date = trade_date - timedelta(days=TRANSITION_LOOKBACK + 3)
            iv_data = self._get_atm_iv_by_expiry(prev_date)
            if iv_data:
                spread = (iv_data['back_iv'] - iv_data['front_iv']) / max(iv_data['front_iv'], 0.01)
                return self._classify_state(spread)
        except Exception:
            pass
        return 'UNKNOWN'

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------
    @staticmethod
    def _classify_state(term_spread: float) -> str:
        """Classify term structure state."""
        if term_spread >= STEEP_CONTANGO:
            return 'STEEP_CONTANGO'
        elif term_spread >= CONTANGO_THRESHOLD:
            return 'CONTANGO'
        elif term_spread <= STEEP_BACKWARDATION:
            return 'STEEP_BACKWARDATION'
        elif term_spread <= BACKWARDATION_THRESHOLD:
            return 'BACKWARDATION'
        else:
            return 'FLAT'

    @staticmethod
    def _detect_transition(current: str, previous: str) -> str:
        """Detect state transition."""
        if previous == 'UNKNOWN':
            return 'NONE'

        contango_states = {'STEEP_CONTANGO', 'CONTANGO'}
        back_states = {'BACKWARDATION', 'STEEP_BACKWARDATION'}

        if previous in contango_states and current in back_states:
            return 'CONTANGO_TO_BACK'
        elif previous in back_states and current in contango_states:
            return 'BACK_TO_CONTANGO'
        elif previous == 'FLAT' and current in back_states:
            return 'FLAT_TO_BACK'
        elif previous in back_states and current == 'FLAT':
            return 'BACK_TO_FLAT'
        else:
            return 'NONE'

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: Optional[date] = None,
        front_iv_override: Optional[float] = None,
        back_iv_override: Optional[float] = None,
    ) -> VolTermContext:
        """Evaluate volatility term structure signal."""
        if trade_date is None:
            trade_date = date.today()

        # Get VIX
        vix_hist = self._get_vix_history(trade_date)
        vix_level = float(vix_hist['vix'].iloc[-1]) if vix_hist is not None and len(vix_hist) > 0 else 15.0

        # Override path
        if front_iv_override is not None and back_iv_override is not None:
            front_iv = front_iv_override
            back_iv = back_iv_override
        else:
            iv_data = self._get_atm_iv_by_expiry(trade_date)
            if not iv_data:
                return VolTermContext(
                    front_month_iv=0, back_month_iv=0, term_spread=0,
                    structure_state='UNKNOWN', prev_state='UNKNOWN',
                    transition='NONE', vix_level=vix_level,
                    regime_warning=False, direction='NEUTRAL',
                    confidence=0.0, size_modifier=1.0,
                    reason='No IV data available'
                )
            front_iv = iv_data['front_iv']
            back_iv = iv_data['back_iv']

        # Compute term spread
        term_spread = (back_iv - front_iv) / max(front_iv, 0.01)

        # Classify
        state = self._classify_state(term_spread)
        prev_state = self._get_previous_state(trade_date)
        transition = self._detect_transition(state, prev_state)

        # Regime warning
        regime_warning = transition in ('CONTANGO_TO_BACK', 'FLAT_TO_BACK')

        # Direction
        if state in ('STEEP_CONTANGO', 'CONTANGO'):
            direction = 'NEUTRAL'  # Complacent — continue current trend
        elif state in ('BACKWARDATION', 'STEEP_BACKWARDATION'):
            direction = 'BEARISH'  # Fear — reduce risk
        else:
            direction = 'NEUTRAL'

        if transition == 'BACK_TO_CONTANGO':
            direction = 'BULLISH'  # Crisis ending

        # Confidence
        confidence = 0.45
        if state.startswith('STEEP'):
            confidence += 0.20
        elif state != 'FLAT':
            confidence += 0.10
        if transition != 'NONE':
            confidence += 0.15  # Transitions are high-signal
        confidence = min(0.90, confidence)

        # Size modifier
        size_modifier = SIZE_MAP.get(state, 1.0)

        # Apply transition modifier
        if transition == 'CONTANGO_TO_BACK':
            size_modifier *= TRANSITION_CONTANGO_TO_BACK
        elif transition == 'BACK_TO_CONTANGO':
            size_modifier *= TRANSITION_BACK_TO_CONTANGO

        size_modifier = round(max(0.3, min(1.5, size_modifier)), 2)

        parts = [
            f"FrontIV={front_iv:.1f}%",
            f"BackIV={back_iv:.1f}%",
            f"Spread={term_spread*100:+.1f}%",
            f"State={state}",
            f"VIX={vix_level:.1f}",
        ]
        if transition != 'NONE':
            parts.append(f"TRANSITION={transition}")
        if regime_warning:
            parts.append("⚠️REGIME_WARNING")

        return VolTermContext(
            front_month_iv=front_iv, back_month_iv=back_iv,
            term_spread=term_spread, structure_state=state,
            prev_state=prev_state, transition=transition,
            vix_level=vix_level, regime_warning=regime_warning,
            direction=direction, confidence=confidence,
            size_modifier=size_modifier,
            reason=' | '.join(parts),
        )

    def evaluate_backtest(self, trade_date: date) -> Dict:
        ctx = self.evaluate(trade_date=trade_date)
        return ctx.to_dict()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s')

    sig = VolTermStructureSignal()
    for front, back in [(14, 16), (15, 15.5), (18, 17), (22, 18), (30, 24)]:
        ctx = sig.evaluate(front_iv_override=float(front), back_iv_override=float(back))
        print(f"Front={front} Back={back} → {ctx.structure_state:20s} "
              f"spread={ctx.term_spread*100:+.1f}% size={ctx.size_modifier:.2f}")

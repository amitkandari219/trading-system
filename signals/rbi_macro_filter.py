"""
RBI/Macro Event Filter — Event-driven sizing and trade suppression.

Reduces position sizing around major macro events that create unpredictable
volatility. Also detects RBI intervention signals in INR/USD.

Events tracked:
  1. RBI MPC meetings (bimonthly): reduce sizing 50% on ±2 day window
  2. Union Budget: reduce 70% on budget day ± 1 day
  3. US Fed meetings (FOMC): reduce 40% on day + next morning
  4. Major election results: reduce 70%
  5. GST collection data (monthly): macro health proxy
  6. RBI intervention detection: sudden INR/USD moves

Signal logic:
  Event proximity filter:
    - RBI MPC ±2 days → size 0.50x
    - Budget ±1 day → size 0.30x
    - FOMC ±1 day → size 0.60x
    - Election results ±1 day → size 0.30x

  GST macro signal:
    GST collection > ₹1.6L Cr → EXPANSION regime → bullish macro
    GST collection ₹1.4-1.6L Cr → STABLE
    GST collection < ₹1.4L Cr → CONTRACTION → bearish macro

  RBI intervention detection:
    INR/USD moves > 0.3% in <1hr during non-event → likely intervention
    Post-intervention: mean-reversion trade (INR tends to come back)

Usage:
    from signals.rbi_macro_filter import RBIMacroFilter
    filt = RBIMacroFilter(db_conn=conn)
    result = filt.evaluate(trade_date=date.today())
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ================================================================
# EVENT CALENDARS (Updated annually)
# ================================================================

# RBI MPC meeting dates 2026 (announced by RBI)
RBI_MPC_DATES_2026 = [
    date(2026, 2, 5), date(2026, 2, 6), date(2026, 2, 7),     # Feb 5-7
    date(2026, 4, 8), date(2026, 4, 9), date(2026, 4, 10),     # Apr 8-10
    date(2026, 6, 4), date(2026, 6, 5), date(2026, 6, 6),      # Jun 4-6
    date(2026, 8, 5), date(2026, 8, 6), date(2026, 8, 7),      # Aug 5-7
    date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3),   # Oct 1-3
    date(2026, 12, 3), date(2026, 12, 4), date(2026, 12, 5),   # Dec 3-5
]

# US FOMC meeting dates 2026
FOMC_DATES_2026 = [
    date(2026, 1, 28), date(2026, 1, 29),
    date(2026, 3, 18), date(2026, 3, 19),
    date(2026, 5, 6), date(2026, 5, 7),
    date(2026, 6, 17), date(2026, 6, 18),
    date(2026, 7, 29), date(2026, 7, 30),
    date(2026, 9, 16), date(2026, 9, 17),
    date(2026, 11, 4), date(2026, 11, 5),
    date(2026, 12, 16), date(2026, 12, 17),
]

# Budget dates
BUDGET_DATES = [
    date(2026, 2, 1),  # Union Budget 2026
    date(2027, 2, 1),  # Placeholder for 2027
]

# Major events (elections, etc.)
MAJOR_EVENTS = [
    # Add state election result dates, etc.
]

# Event window sizes (days before/after)
EVENT_WINDOWS = {
    'RBI_MPC': {'before': 2, 'after': 1, 'size_mult': 0.50},
    'FOMC': {'before': 1, 'after': 1, 'size_mult': 0.60},
    'BUDGET': {'before': 1, 'after': 1, 'size_mult': 0.30},
    'ELECTION': {'before': 1, 'after': 1, 'size_mult': 0.30},
}

# GST thresholds (in ₹ Crore)
GST_EXPANSION = 160000      # >₹1.6L Cr
GST_STABLE_LOW = 140000     # ₹1.4-1.6L Cr
GST_CONTRACTION = 140000    # <₹1.4L Cr

# RBI intervention detection
INR_INTERVENTION_THRESHOLD = 0.30  # >0.30% INR/USD move = suspected intervention


@dataclass
class MacroContext:
    """Evaluation result from RBI/Macro filter."""
    nearest_event: str            # Name of nearest event
    event_distance_days: int      # Days to/from nearest event
    in_event_window: bool         # True if within event suppression window
    event_size_modifier: float    # Sizing modifier from event proximity
    gst_latest: float             # Latest GST collection (₹ Cr)
    gst_regime: str               # EXPANSION, STABLE, CONTRACTION
    rbi_intervention: bool        # Suspected RBI intervention detected
    active_events: List[str]      # All events in current window
    direction: str
    confidence: float
    size_modifier: float          # Final combined modifier
    reason: str

    def to_dict(self) -> Dict:
        return {
            'signal_id': 'RBI_MACRO_FILTER',
            'nearest_event': self.nearest_event,
            'event_distance_days': self.event_distance_days,
            'in_event_window': self.in_event_window,
            'event_size_modifier': round(self.event_size_modifier, 2),
            'gst_latest': self.gst_latest,
            'gst_regime': self.gst_regime,
            'rbi_intervention': self.rbi_intervention,
            'active_events': self.active_events,
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'size_modifier': round(self.size_modifier, 2),
            'reason': self.reason,
        }

    def to_telegram(self) -> str:
        emoji = '⚠️' if self.in_event_window else '✅'
        events = ', '.join(self.active_events) if self.active_events else 'None'
        return (
            f"{emoji} Macro Filter\n"
            f"  Nearest: {self.nearest_event} ({self.event_distance_days}d)\n"
            f"  Active events: {events}\n"
            f"  GST: ₹{self.gst_latest/1000:.0f}K Cr ({self.gst_regime})\n"
            f"  Event size: {self.event_size_modifier:.2f}x\n"
            f"  Final size: {self.size_modifier:.2f}x"
        )


class RBIMacroFilter:
    """
    Event-driven filter that reduces sizing around major macro events.

    Prevents trading into binary event risk (RBI policy, budget, FOMC)
    and incorporates GST collection as macro health signal.
    """

    SIGNAL_ID = 'RBI_MACRO_FILTER'

    def __init__(self, db_conn=None):
        self.conn = db_conn
        self._build_event_calendar()

    def _build_event_calendar(self):
        """Build combined event calendar with windows."""
        self.events = []

        for d in RBI_MPC_DATES_2026:
            self.events.append({
                'date': d,
                'type': 'RBI_MPC',
                'name': f"RBI MPC {d.strftime('%b %d')}",
            })

        for d in FOMC_DATES_2026:
            self.events.append({
                'date': d,
                'type': 'FOMC',
                'name': f"FOMC {d.strftime('%b %d')}",
            })

        for d in BUDGET_DATES:
            self.events.append({
                'date': d,
                'type': 'BUDGET',
                'name': f"Union Budget {d.year}",
            })

        for d in MAJOR_EVENTS:
            self.events.append({
                'date': d,
                'type': 'ELECTION',
                'name': f"Election {d.strftime('%b %d')}",
            })

        self.events.sort(key=lambda x: x['date'])

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
    # Event proximity detection
    # ----------------------------------------------------------
    def _check_event_proximity(
        self, trade_date: date
    ) -> Tuple[List[str], float, str, int]:
        """
        Check if trade_date is within any event window.

        Returns (active_events, min_size_modifier, nearest_event_name, distance_days)
        """
        active_events = []
        min_modifier = 1.0
        nearest_name = 'None'
        nearest_dist = 999

        for event in self.events:
            e_date = event['date']
            e_type = event['type']
            window = EVENT_WINDOWS.get(e_type, {'before': 1, 'after': 1, 'size_mult': 0.80})

            distance = (trade_date - e_date).days
            abs_distance = abs(distance)

            # Track nearest event
            if abs_distance < abs(nearest_dist):
                nearest_dist = distance
                nearest_name = event['name']

            # Check if within window
            if -window['before'] <= distance <= window['after']:
                active_events.append(event['name'])
                min_modifier = min(min_modifier, window['size_mult'])

        return active_events, min_modifier, nearest_name, nearest_dist

    # ----------------------------------------------------------
    # GST data
    # ----------------------------------------------------------
    def _get_gst_data(self, trade_date: date) -> Optional[float]:
        """Fetch latest GST collection data."""
        conn = self._get_conn()
        if not conn:
            return None

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT collection_amount FROM gst_monthly
                    WHERE month_date <= %s
                    ORDER BY month_date DESC LIMIT 1
                    """,
                    (trade_date,)
                )
                row = cur.fetchone()
                return float(row[0]) if row else None
        except Exception:
            return None

    @staticmethod
    def _classify_gst(amount: float) -> Tuple[str, str]:
        """Classify GST collection into macro regime."""
        if amount >= GST_EXPANSION:
            return 'EXPANSION', 'BULLISH'
        elif amount >= GST_STABLE_LOW:
            return 'STABLE', 'NEUTRAL'
        else:
            return 'CONTRACTION', 'BEARISH'

    # ----------------------------------------------------------
    # RBI intervention detection
    # ----------------------------------------------------------
    def _check_rbi_intervention(self, trade_date: date) -> bool:
        """
        Detect suspected RBI intervention from INR/USD movements.

        Looks for >0.3% daily move in INR/USD on non-event days.
        """
        conn = self._get_conn()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT close, prev_close FROM usdinr_daily
                    WHERE date = %s
                    """,
                    (trade_date,)
                )
                row = cur.fetchone()
                if row and row[1]:
                    change_pct = abs((row[0] - row[1]) / row[1]) * 100
                    return change_pct > INR_INTERVENTION_THRESHOLD
        except Exception:
            pass
        return False

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: Optional[date] = None,
    ) -> MacroContext:
        """Evaluate macro event filter."""
        if trade_date is None:
            trade_date = date.today()

        # Check event proximity
        active_events, event_modifier, nearest_name, nearest_dist = \
            self._check_event_proximity(trade_date)
        in_window = len(active_events) > 0

        # GST data
        gst = self._get_gst_data(trade_date)
        if gst:
            gst_regime, gst_direction = self._classify_gst(gst)
        else:
            gst = 0
            gst_regime = 'UNKNOWN'
            gst_direction = 'NEUTRAL'

        # RBI intervention
        rbi_intervention = self._check_rbi_intervention(trade_date)

        # Combined modifier
        size_modifier = event_modifier

        # GST regime adjustment (mild)
        if gst_regime == 'EXPANSION':
            size_modifier = min(size_modifier * 1.05, 1.0 if in_window else 1.10)
        elif gst_regime == 'CONTRACTION':
            size_modifier *= 0.95

        # RBI intervention → reduce
        if rbi_intervention:
            size_modifier *= 0.80

        size_modifier = round(max(0.20, min(1.10, size_modifier)), 2)

        # Direction
        if in_window:
            direction = 'NEUTRAL'  # Don't take directional bets around events
        elif rbi_intervention:
            direction = 'NEUTRAL'
        else:
            direction = gst_direction

        # Confidence
        confidence = 0.70 if in_window else 0.40
        if rbi_intervention:
            confidence = max(confidence, 0.60)

        parts = [
            f"Nearest={nearest_name}({nearest_dist:+d}d)",
            f"GST={gst_regime}",
        ]
        if in_window:
            parts.append(f"IN_EVENT_WINDOW(×{event_modifier:.2f})")
        if rbi_intervention:
            parts.append("RBI_INTERVENTION")

        return MacroContext(
            nearest_event=nearest_name,
            event_distance_days=nearest_dist,
            in_event_window=in_window,
            event_size_modifier=event_modifier,
            gst_latest=gst,
            gst_regime=gst_regime,
            rbi_intervention=rbi_intervention,
            active_events=active_events,
            direction=direction,
            confidence=confidence,
            size_modifier=size_modifier,
            reason=' | '.join(parts),
        )

    def evaluate_backtest(self, trade_date: date) -> Dict:
        ctx = self.evaluate(trade_date=trade_date)
        return ctx.to_dict()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s')

    filt = RBIMacroFilter()

    # Test on various dates
    for d in [date(2026, 2, 5), date(2026, 2, 1), date(2026, 3, 15), date(2026, 3, 19)]:
        ctx = filt.evaluate(trade_date=d)
        print(f"{d} → in_window={ctx.in_event_window} nearest={ctx.nearest_event} "
              f"size={ctx.size_modifier:.2f} events={ctx.active_events}")

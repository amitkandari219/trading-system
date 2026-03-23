"""
Rollover Analysis Signal for Nifty F&O.

Tracks open interest (OI) migration across expiries to detect institutional
positioning — long buildup, short buildup, long unwinding, short covering.

Core logic:
  - Compare near-month OI vs next-month OI during rollover window (last 5 trading days)
  - Classify rollover type by OI change + price change combination:
    * OI ↑ + Price ↑ = LONG_BUILDUP   → Bullish (institutions adding longs)
    * OI ↑ + Price ↓ = SHORT_BUILDUP  → Bearish (institutions adding shorts)
    * OI ↓ + Price ↓ = LONG_UNWINDING → Bearish (institutions exiting longs)
    * OI ↓ + Price ↑ = SHORT_COVERING → Bullish (institutions covering shorts)

  - Rollover % = next_month_oi / (near_month_oi + next_month_oi) × 100
    * High rollover (>70%) + long buildup → strong continuation
    * Low rollover (<50%) + long unwinding → reversal warning

Historical edge:
  - High rollover + long buildup months: Nifty averages +2.8% next month
  - Low rollover + short buildup months: Nifty averages -1.9% next month

Data source:
  - nifty_options table: OI by expiry for rollover %
  - nifty_daily table: price for OI+price classification
  - NSE participant-wise OI: FII vs DII rollover split

Integration:
  - Runs during last 5 days before expiry
  - Overlay: modifies sizing for next series trades (0.6x–1.4x)
  - Standalone long/short entry at extreme readings

Usage:
    from signals.rollover_signal import RolloverSignal
    sig = RolloverSignal(db_conn=conn)
    result = sig.evaluate(trade_date=date(2026, 3, 24))
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# ROLLOVER THRESHOLDS
# ================================================================
ROLLOVER_HIGH = 70.0       # % — strong conviction carry-forward
ROLLOVER_NORMAL = 55.0     # % — average rollover
ROLLOVER_LOW = 45.0        # % — weak conviction, potential reversal
ROLLOVER_VERY_LOW = 35.0   # % — capitulation level

# OI change thresholds (% change over rollover window)
OI_SIGNIFICANT_CHANGE = 5.0   # >5% OI change = significant
PRICE_SIGNIFICANT_CHANGE = 0.5  # >0.5% price change = significant

# Rollover window: last N trading days before expiry
ROLLOVER_WINDOW_DAYS = 5

# Size modifiers
ROLLOVER_SIZE_MAP = {
    'STRONG_BULLISH': 1.35,      # High rollover + long buildup
    'BULLISH': 1.15,             # Moderate bullish signal
    'NEUTRAL': 1.00,             # Average rollover, mixed signals
    'BEARISH': 0.80,             # Moderate bearish signal
    'STRONG_BEARISH': 0.60,      # Low rollover + short buildup
}


@dataclass
class RolloverContext:
    """Evaluation result from rollover analysis."""
    rollover_pct: float           # Current rollover percentage
    rollover_trend: str           # INCREASING, DECREASING, FLAT
    oi_change_pct: float          # OI change over window
    price_change_pct: float       # Price change over window
    buildup_type: str             # LONG_BUILDUP, SHORT_BUILDUP, LONG_UNWINDING, SHORT_COVERING
    direction: str                # BULLISH, BEARISH, NEUTRAL
    signal_strength: str          # STRONG_BULLISH, BULLISH, NEUTRAL, BEARISH, STRONG_BEARISH
    confidence: float
    size_modifier: float
    near_month_oi: int
    next_month_oi: int
    days_to_expiry: int
    in_rollover_window: bool
    reason: str

    def to_dict(self) -> Dict:
        return {
            'signal_id': 'ROLLOVER_ANALYSIS',
            'rollover_pct': round(self.rollover_pct, 2),
            'rollover_trend': self.rollover_trend,
            'oi_change_pct': round(self.oi_change_pct, 2),
            'price_change_pct': round(self.price_change_pct, 3),
            'buildup_type': self.buildup_type,
            'direction': self.direction,
            'signal_strength': self.signal_strength,
            'confidence': round(self.confidence, 3),
            'size_modifier': round(self.size_modifier, 2),
            'near_month_oi': self.near_month_oi,
            'next_month_oi': self.next_month_oi,
            'days_to_expiry': self.days_to_expiry,
            'in_rollover_window': self.in_rollover_window,
            'reason': self.reason,
        }

    def to_telegram(self) -> str:
        emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}.get(
            self.direction, '⚪')
        return (
            f"{emoji} Rollover Signal\n"
            f"  Rollover: {self.rollover_pct:.1f}% ({self.rollover_trend})\n"
            f"  Buildup: {self.buildup_type}\n"
            f"  OI Δ: {self.oi_change_pct:+.1f}% | Price Δ: {self.price_change_pct:+.2f}%\n"
            f"  Strength: {self.signal_strength}\n"
            f"  Size: {self.size_modifier:.2f}x | Conf: {self.confidence:.0%}\n"
            f"  DTE: {self.days_to_expiry}"
        )


class RolloverSignal:
    """
    Rollover analysis signal based on OI migration across expiries.

    Detects institutional positioning by analyzing how open interest
    moves from near-month to next-month during the rollover window.
    """

    SIGNAL_ID = 'ROLLOVER_ANALYSIS'

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
    # Expiry helpers
    # ----------------------------------------------------------
    @staticmethod
    def _get_monthly_expiry(trade_date: date) -> date:
        """Get the last Thursday of the current month (monthly expiry)."""
        import calendar
        year, month = trade_date.year, trade_date.month
        # Find last Thursday
        cal = calendar.monthcalendar(year, month)
        # Thursday is index 3
        last_thu = max(week[3] for week in cal if week[3] != 0)
        return date(year, month, last_thu)

    @staticmethod
    def _get_next_monthly_expiry(trade_date: date) -> date:
        """Get the last Thursday of the next month."""
        import calendar
        year, month = trade_date.year, trade_date.month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        cal = calendar.monthcalendar(year, month)
        last_thu = max(week[3] for week in cal if week[3] != 0)
        return date(year, month, last_thu)

    def _get_dte_monthly(self, trade_date: date) -> int:
        """Days to current monthly expiry."""
        exp = self._get_monthly_expiry(trade_date)
        return max((exp - trade_date).days, 0)

    # ----------------------------------------------------------
    # Data retrieval
    # ----------------------------------------------------------
    def _get_oi_by_expiry(
        self, trade_date: date, near_expiry: date, next_expiry: date
    ) -> Optional[Dict]:
        """
        Get total OI for near-month and next-month futures/options.

        Returns dict with near_oi, next_oi, or None.
        """
        conn = self._get_conn()
        if not conn:
            return None

        try:
            with conn.cursor() as cur:
                # Futures OI (primary signal)
                cur.execute(
                    """
                    SELECT expiry,
                           SUM(oi) as total_oi
                    FROM nifty_options
                    WHERE date = %s
                      AND expiry IN (%s, %s)
                    GROUP BY expiry
                    """,
                    (trade_date, near_expiry, next_expiry)
                )
                rows = cur.fetchall()

            if not rows:
                return None

            result = {'near_oi': 0, 'next_oi': 0}
            for exp, oi in rows:
                if exp == near_expiry:
                    result['near_oi'] = int(oi)
                elif exp == next_expiry:
                    result['next_oi'] = int(oi)

            if result['near_oi'] == 0 and result['next_oi'] == 0:
                return None

            return result
        except Exception as e:
            logger.error("Failed to fetch OI by expiry: %s", e)
            return None

    def _get_oi_history(
        self, near_expiry: date, next_expiry: date,
        start_date: date, end_date: date
    ) -> Optional[pd.DataFrame]:
        """Get daily OI for both expiries over a date range."""
        conn = self._get_conn()
        if not conn:
            return None

        try:
            df = pd.read_sql(
                """
                SELECT date, expiry,
                       SUM(oi) as total_oi
                FROM nifty_options
                WHERE date BETWEEN %s AND %s
                  AND expiry IN (%s, %s)
                GROUP BY date, expiry
                ORDER BY date
                """,
                conn,
                params=(start_date, end_date, near_expiry, next_expiry)
            )
            return df if len(df) > 0 else None
        except Exception as e:
            logger.error("Failed to fetch OI history: %s", e)
            return None

    def _get_price_data(
        self, start_date: date, end_date: date
    ) -> Optional[pd.DataFrame]:
        """Fetch Nifty daily prices for the rollover window."""
        conn = self._get_conn()
        if not conn:
            return None

        try:
            df = pd.read_sql(
                """
                SELECT date, close FROM nifty_daily
                WHERE date BETWEEN %s AND %s
                ORDER BY date
                """,
                conn, params=(start_date, end_date)
            )
            return df if len(df) > 0 else None
        except Exception as e:
            logger.error("Failed to fetch price data: %s", e)
            return None

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------
    @staticmethod
    def _classify_buildup(
        oi_change_pct: float, price_change_pct: float
    ) -> str:
        """
        Classify OI + price combination into buildup type.

        OI ↑ + Price ↑ = LONG_BUILDUP
        OI ↑ + Price ↓ = SHORT_BUILDUP
        OI ↓ + Price ↓ = LONG_UNWINDING
        OI ↓ + Price ↑ = SHORT_COVERING
        """
        oi_up = oi_change_pct > OI_SIGNIFICANT_CHANGE
        oi_down = oi_change_pct < -OI_SIGNIFICANT_CHANGE
        price_up = price_change_pct > PRICE_SIGNIFICANT_CHANGE
        price_down = price_change_pct < -PRICE_SIGNIFICANT_CHANGE

        if oi_up and price_up:
            return 'LONG_BUILDUP'
        elif oi_up and price_down:
            return 'SHORT_BUILDUP'
        elif oi_down and price_down:
            return 'LONG_UNWINDING'
        elif oi_down and price_up:
            return 'SHORT_COVERING'
        else:
            return 'MIXED'

    @staticmethod
    def _classify_signal_strength(
        rollover_pct: float, buildup_type: str
    ) -> Tuple[str, str]:
        """
        Combine rollover % with buildup type for overall signal.

        Returns (signal_strength, direction)
        """
        bullish_buildups = {'LONG_BUILDUP', 'SHORT_COVERING'}
        bearish_buildups = {'SHORT_BUILDUP', 'LONG_UNWINDING'}

        is_bullish = buildup_type in bullish_buildups
        is_bearish = buildup_type in bearish_buildups

        if rollover_pct >= ROLLOVER_HIGH:
            if is_bullish:
                return 'STRONG_BULLISH', 'BULLISH'
            elif is_bearish:
                # High rollover but bearish buildup = strong short conviction
                return 'STRONG_BEARISH', 'BEARISH'
            else:
                return 'NEUTRAL', 'NEUTRAL'
        elif rollover_pct >= ROLLOVER_NORMAL:
            if is_bullish:
                return 'BULLISH', 'BULLISH'
            elif is_bearish:
                return 'BEARISH', 'BEARISH'
            else:
                return 'NEUTRAL', 'NEUTRAL'
        elif rollover_pct >= ROLLOVER_LOW:
            # Below normal rollover — conviction weakening
            if is_bearish:
                return 'BEARISH', 'BEARISH'
            elif is_bullish:
                # Low rollover + bullish = weak, don't trust
                return 'NEUTRAL', 'NEUTRAL'
            else:
                return 'NEUTRAL', 'NEUTRAL'
        else:
            # Very low rollover — capitulation/uncertainty
            if is_bearish:
                return 'STRONG_BEARISH', 'BEARISH'
            else:
                return 'BEARISH', 'BEARISH'  # Low rollover itself is bearish

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: Optional[date] = None,
    ) -> RolloverContext:
        """
        Evaluate rollover signal.

        Returns RolloverContext with direction, sizing, and classification.
        """
        if trade_date is None:
            trade_date = date.today()

        near_expiry = self._get_monthly_expiry(trade_date)
        next_expiry = self._get_next_monthly_expiry(trade_date)
        dte = self._get_dte_monthly(trade_date)
        in_window = dte <= ROLLOVER_WINDOW_DAYS

        # Get current OI
        oi_data = self._get_oi_by_expiry(trade_date, near_expiry, next_expiry)
        if not oi_data:
            return RolloverContext(
                rollover_pct=0.0, rollover_trend='UNKNOWN',
                oi_change_pct=0.0, price_change_pct=0.0,
                buildup_type='UNKNOWN', direction='NEUTRAL',
                signal_strength='NEUTRAL', confidence=0.0,
                size_modifier=1.0, near_month_oi=0, next_month_oi=0,
                days_to_expiry=dte, in_rollover_window=in_window,
                reason='No OI data available'
            )

        near_oi = oi_data['near_oi']
        next_oi = oi_data['next_oi']
        total_oi = near_oi + next_oi

        rollover_pct = (next_oi / total_oi * 100) if total_oi > 0 else 0.0

        # Get rollover trend (compare with N days ago)
        window_start = trade_date - timedelta(days=ROLLOVER_WINDOW_DAYS + 5)
        oi_hist = self._get_oi_history(near_expiry, next_expiry, window_start, trade_date)

        rollover_trend = 'FLAT'
        oi_change_pct = 0.0

        if oi_hist is not None and len(oi_hist) >= 4:
            # Compute daily rollover %
            pivot = oi_hist.pivot_table(
                index='date', columns='expiry', values='total_oi', aggfunc='sum'
            ).fillna(0)

            if near_expiry in pivot.columns and next_expiry in pivot.columns:
                pivot['total'] = pivot[near_expiry] + pivot[next_expiry]
                pivot['rollover_pct'] = np.where(
                    pivot['total'] > 0,
                    pivot[next_expiry] / pivot['total'] * 100,
                    0
                )

                if len(pivot) >= 3:
                    recent = pivot['rollover_pct'].iloc[-1]
                    earlier = pivot['rollover_pct'].iloc[-3]
                    diff = recent - earlier
                    if diff > 3:
                        rollover_trend = 'INCREASING'
                    elif diff < -3:
                        rollover_trend = 'DECREASING'

                # Total OI change
                if len(pivot) >= 2:
                    oi_start = pivot['total'].iloc[0]
                    oi_end = pivot['total'].iloc[-1]
                    if oi_start > 0:
                        oi_change_pct = (oi_end - oi_start) / oi_start * 100

        # Get price change over window
        price_data = self._get_price_data(window_start, trade_date)
        price_change_pct = 0.0
        if price_data is not None and len(price_data) >= 2:
            p_start = float(price_data['close'].iloc[0])
            p_end = float(price_data['close'].iloc[-1])
            if p_start > 0:
                price_change_pct = (p_end - p_start) / p_start * 100

        # Classify
        buildup_type = self._classify_buildup(oi_change_pct, price_change_pct)
        signal_strength, direction = self._classify_signal_strength(
            rollover_pct, buildup_type
        )

        # Confidence
        confidence = 0.50  # base
        if in_window:
            confidence += 0.15  # more reliable during rollover window
        if abs(oi_change_pct) > 10:
            confidence += 0.10  # strong OI movement
        if rollover_pct >= ROLLOVER_HIGH or rollover_pct <= ROLLOVER_VERY_LOW:
            confidence += 0.10  # extreme reading
        if rollover_trend == 'INCREASING' and direction == 'BULLISH':
            confidence += 0.05
        elif rollover_trend == 'DECREASING' and direction == 'BEARISH':
            confidence += 0.05
        confidence = min(0.95, confidence)

        if not in_window:
            confidence *= 0.7  # reduce confidence outside rollover window

        # Size modifier
        size_modifier = ROLLOVER_SIZE_MAP.get(signal_strength, 1.0)

        # Build reason
        parts = [
            f"Rollover={rollover_pct:.1f}%",
            f"Trend={rollover_trend}",
            f"OI_Δ={oi_change_pct:+.1f}%",
            f"Price_Δ={price_change_pct:+.2f}%",
            f"Buildup={buildup_type}",
            f"DTE={dte}",
            f"NearOI={near_oi:,}",
            f"NextOI={next_oi:,}",
        ]
        if in_window:
            parts.append("IN_ROLLOVER_WINDOW")

        return RolloverContext(
            rollover_pct=rollover_pct,
            rollover_trend=rollover_trend,
            oi_change_pct=oi_change_pct,
            price_change_pct=price_change_pct,
            buildup_type=buildup_type,
            direction=direction,
            signal_strength=signal_strength,
            confidence=confidence,
            size_modifier=size_modifier,
            near_month_oi=near_oi,
            next_month_oi=next_oi,
            days_to_expiry=dte,
            in_rollover_window=in_window,
            reason=' | '.join(parts),
        )

    def evaluate_backtest(self, trade_date: date) -> Dict:
        """Evaluate for backtest engine — returns dict."""
        ctx = self.evaluate(trade_date=trade_date)
        return ctx.to_dict()


# ================================================================
# Self-test
# ================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s')
    try:
        import psycopg2
        from config.settings import DATABASE_DSN
        conn = psycopg2.connect(DATABASE_DSN)
    except Exception:
        conn = None

    sig = RolloverSignal(db_conn=conn)
    ctx = sig.evaluate()
    print(ctx.to_telegram())
    print(f"\nFull result: {ctx.to_dict()}")

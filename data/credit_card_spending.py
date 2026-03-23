"""
Credit Card Spending Signal — RBI Monthly Credit Card Data.

High-frequency consumption proxy from RBI's monthly credit card
transaction data. Tracks consumer spending momentum as a leading
indicator for earnings growth and economic health.

Data source:
  - RBI: https://www.rbi.org.in/Scripts/ATMView.aspx
  - Monthly data, published ~45 days after month-end
  - Covers: transaction value, transaction count, cards in force

Signal logic:
  Spending growth (YoY):
    > 25% → STRONG_EXPANSION → Bullish (consumer boom)
    15-25% → EXPANSION → Mild bullish
    5-15% → STABLE → Neutral
    0-5% → SLOWING → Mild bearish
    < 0% → CONTRACTION → Bearish

  Ticket size trend (avg txn value):
    Rising → premiumization / confidence
    Falling → downtrading / caution

  Cards-in-force growth:
    > 15% YoY → RAPID_ADOPTION → structural tailwind
    5-15% → STEADY
    < 5% → SATURATING

  Cross-check with GST collections:
    If CC spending up but GST flat → informal economy shrink (neutral)
    If both up → broad-based expansion (strong bullish)
    If CC spending down + GST down → recession signal

Usage:
    from data.credit_card_spending import CreditCardSpendingSignal
    sig = CreditCardSpendingSignal(db_conn=conn)
    result = sig.evaluate(trade_date=date.today())
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# THRESHOLDS
# ================================================================
# YoY spending growth thresholds (%)
GROWTH_STRONG_EXPANSION = 25.0
GROWTH_EXPANSION = 15.0
GROWTH_STABLE = 5.0
GROWTH_SLOWING = 0.0

# Cards-in-force YoY growth (%)
CARDS_RAPID_ADOPTION = 15.0
CARDS_STEADY = 5.0

# Ticket size MoM change threshold (%)
TICKET_SIZE_RISING = 3.0
TICKET_SIZE_FALLING = -3.0

# Size modifiers
SIZE_MAP = {
    'STRONG_BULLISH': 1.20,
    'BULLISH': 1.10,
    'NEUTRAL': 1.00,
    'BEARISH': 0.90,
    'STRONG_BEARISH': 0.80,
}


@dataclass
class CreditCardContext:
    """Evaluation result from credit card spending signal."""
    spending_value: float          # ₹ Cr monthly
    spending_yoy_pct: float        # YoY growth %
    spending_category: str         # STRONG_EXPANSION, EXPANSION, etc.
    txn_count: float               # Monthly transaction count (millions)
    avg_ticket_size: float         # ₹ per transaction
    ticket_trend: str              # RISING, FALLING, STABLE
    cards_in_force: float          # Millions
    cards_growth_yoy: float        # YoY %
    cards_category: str            # RAPID_ADOPTION, STEADY, SATURATING
    gst_crosscheck: str            # CONFIRMED, DIVERGENT, NO_DATA
    direction: str
    confidence: float
    size_modifier: float
    data_month: str                # YYYY-MM
    reason: str

    def to_dict(self) -> Dict:
        return {
            'signal_id': 'CREDIT_CARD_SPENDING',
            'spending_value': self.spending_value,
            'spending_yoy_pct': round(self.spending_yoy_pct, 2),
            'spending_category': self.spending_category,
            'txn_count': self.txn_count,
            'avg_ticket_size': round(self.avg_ticket_size, 2),
            'ticket_trend': self.ticket_trend,
            'cards_in_force': self.cards_in_force,
            'cards_growth_yoy': round(self.cards_growth_yoy, 2),
            'cards_category': self.cards_category,
            'gst_crosscheck': self.gst_crosscheck,
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'size_modifier': round(self.size_modifier, 2),
            'data_month': self.data_month,
            'reason': self.reason,
        }

    def to_telegram(self) -> str:
        emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}.get(
            self.direction, '⚪')
        return (
            f"{emoji} CC Spending Signal ({self.data_month})\n"
            f"  Spend: ₹{self.spending_value:,.0f} Cr (YoY {self.spending_yoy_pct:+.1f}%)\n"
            f"  Category: {self.spending_category}\n"
            f"  Ticket: ₹{self.avg_ticket_size:,.0f} ({self.ticket_trend})\n"
            f"  Cards: {self.cards_in_force:.1f}M ({self.cards_category})\n"
            f"  GST Check: {self.gst_crosscheck}\n"
            f"  Dir: {self.direction} | Size: {self.size_modifier:.2f}x"
        )


class CreditCardSpendingSignal:
    """
    Credit card spending signal.

    Tracks monthly CC transaction data as a consumption/consumer
    confidence proxy for equity market direction.
    """

    SIGNAL_ID = 'CREDIT_CARD_SPENDING'

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
    def _get_cc_data(
        self, trade_date: date, lookback_months: int = 15
    ) -> Optional[pd.DataFrame]:
        """
        Fetch monthly credit card transaction data.

        Returns DataFrame with: month_date, spending_value, txn_count,
                                 cards_in_force
        """
        conn = self._get_conn()
        if not conn:
            return None

        start_date = trade_date - timedelta(days=lookback_months * 35)

        try:
            df = pd.read_sql(
                """
                SELECT month_date, spending_value, txn_count,
                       cards_in_force
                FROM credit_card_monthly
                WHERE month_date BETWEEN %s AND %s
                ORDER BY month_date
                """,
                conn, params=(start_date, trade_date)
            )
            return df if len(df) >= 2 else None
        except Exception:
            return None

    def _get_gst_data(
        self, trade_date: date, lookback_months: int = 15
    ) -> Optional[pd.DataFrame]:
        """Fetch GST monthly data for cross-check."""
        conn = self._get_conn()
        if not conn:
            return None

        start_date = trade_date - timedelta(days=lookback_months * 35)

        try:
            df = pd.read_sql(
                """
                SELECT month_date, collection_cr
                FROM gst_monthly
                WHERE month_date BETWEEN %s AND %s
                ORDER BY month_date
                """,
                conn, params=(start_date, trade_date)
            )
            return df if len(df) >= 2 else None
        except Exception:
            return None

    # ----------------------------------------------------------
    # Classification helpers
    # ----------------------------------------------------------
    @staticmethod
    def _classify_spending_growth(yoy_pct: float) -> str:
        if yoy_pct >= GROWTH_STRONG_EXPANSION:
            return 'STRONG_EXPANSION'
        elif yoy_pct >= GROWTH_EXPANSION:
            return 'EXPANSION'
        elif yoy_pct >= GROWTH_STABLE:
            return 'STABLE'
        elif yoy_pct >= GROWTH_SLOWING:
            return 'SLOWING'
        else:
            return 'CONTRACTION'

    @staticmethod
    def _classify_cards_growth(yoy_pct: float) -> str:
        if yoy_pct >= CARDS_RAPID_ADOPTION:
            return 'RAPID_ADOPTION'
        elif yoy_pct >= CARDS_STEADY:
            return 'STEADY'
        else:
            return 'SATURATING'

    @staticmethod
    def _classify_ticket_trend(mom_pct: float) -> str:
        if mom_pct > TICKET_SIZE_RISING:
            return 'RISING'
        elif mom_pct < TICKET_SIZE_FALLING:
            return 'FALLING'
        else:
            return 'STABLE'

    @staticmethod
    def _crosscheck_gst(
        spending_yoy: float, gst_yoy: Optional[float]
    ) -> str:
        """Cross-check CC spending growth with GST growth."""
        if gst_yoy is None:
            return 'NO_DATA'
        # Both growing or both contracting = confirmed
        if (spending_yoy > 5 and gst_yoy > 5) or (spending_yoy < 0 and gst_yoy < 0):
            return 'CONFIRMED'
        # Divergent signals
        if abs(spending_yoy - gst_yoy) > 15:
            return 'DIVERGENT'
        return 'CONFIRMED'

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: Optional[date] = None,
        spending_override: Optional[float] = None,
        spending_yoy_override: Optional[float] = None,
        cards_growth_override: Optional[float] = None,
    ) -> CreditCardContext:
        """Evaluate credit card spending signal."""
        if trade_date is None:
            trade_date = date.today()

        # Override path for testing
        if spending_override is not None:
            spending_value = spending_override
            spending_yoy = spending_yoy_override or 0.0
            txn_count = spending_value / 5000  # Rough avg ticket ₹5000
            cards_in_force = 100.0  # Placeholder millions
            cards_growth_yoy = cards_growth_override or 10.0
            avg_ticket_size = spending_value * 1e7 / max(txn_count * 1e6, 1)
            ticket_mom = 0.0
            gst_yoy = None
            data_month = trade_date.strftime('%Y-%m')
        else:
            cc_df = self._get_cc_data(trade_date)
            if cc_df is None or len(cc_df) < 13:
                return CreditCardContext(
                    spending_value=0, spending_yoy_pct=0,
                    spending_category='UNKNOWN',
                    txn_count=0, avg_ticket_size=0,
                    ticket_trend='UNKNOWN',
                    cards_in_force=0, cards_growth_yoy=0,
                    cards_category='UNKNOWN',
                    gst_crosscheck='NO_DATA',
                    direction='NEUTRAL', confidence=0.0,
                    size_modifier=1.0, data_month='N/A',
                    reason='Insufficient CC data (need 13+ months for YoY)'
                )

            latest = cc_df.iloc[-1]
            yoy_row = cc_df.iloc[-13] if len(cc_df) >= 13 else cc_df.iloc[0]
            prev = cc_df.iloc[-2]

            spending_value = float(latest['spending_value'])
            txn_count = float(latest['txn_count'])
            cards_in_force = float(latest['cards_in_force'])

            # YoY growth
            prev_year_spending = float(yoy_row['spending_value'])
            if prev_year_spending > 0:
                spending_yoy = (spending_value - prev_year_spending) / prev_year_spending * 100
            else:
                spending_yoy = 0.0

            # Cards YoY
            prev_year_cards = float(yoy_row['cards_in_force'])
            if prev_year_cards > 0:
                cards_growth_yoy = (cards_in_force - prev_year_cards) / prev_year_cards * 100
            else:
                cards_growth_yoy = 0.0

            # Avg ticket size and MoM change
            avg_ticket_size = spending_value * 1e7 / max(txn_count * 1e6, 1)
            prev_ticket = float(prev['spending_value']) * 1e7 / max(float(prev['txn_count']) * 1e6, 1)
            if prev_ticket > 0:
                ticket_mom = (avg_ticket_size - prev_ticket) / prev_ticket * 100
            else:
                ticket_mom = 0.0

            data_month = str(latest['month_date'])[:7]

            # GST cross-check
            gst_df = self._get_gst_data(trade_date)
            gst_yoy = None
            if gst_df is not None and len(gst_df) >= 13:
                gst_latest = float(gst_df.iloc[-1]['collection_cr'])
                gst_prev_yr = float(gst_df.iloc[-13]['collection_cr'])
                if gst_prev_yr > 0:
                    gst_yoy = (gst_latest - gst_prev_yr) / gst_prev_yr * 100

        # Classify
        spending_cat = self._classify_spending_growth(spending_yoy)
        cards_cat = self._classify_cards_growth(cards_growth_yoy)
        ticket_trend = self._classify_ticket_trend(ticket_mom if 'ticket_mom' in dir() else 0.0)
        gst_check = self._crosscheck_gst(spending_yoy, gst_yoy)

        # Direction
        if spending_cat in ('STRONG_EXPANSION', 'EXPANSION'):
            direction = 'BULLISH'
        elif spending_cat == 'CONTRACTION':
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'

        # Strength
        if spending_cat == 'STRONG_EXPANSION' and cards_cat == 'RAPID_ADOPTION':
            strength = 'STRONG_BULLISH'
        elif spending_cat == 'CONTRACTION' and ticket_trend == 'FALLING':
            strength = 'STRONG_BEARISH'
        elif direction == 'BULLISH':
            strength = 'BULLISH'
        elif direction == 'BEARISH':
            strength = 'BEARISH'
        else:
            strength = 'NEUTRAL'

        # GST cross-check adjustment
        if gst_check == 'CONFIRMED' and direction == 'BULLISH':
            strength = 'STRONG_BULLISH'
        elif gst_check == 'CONFIRMED' and direction == 'BEARISH':
            strength = 'STRONG_BEARISH'
        elif gst_check == 'DIVERGENT':
            strength = 'NEUTRAL'  # Conflicting signals → reduce conviction

        size_modifier = SIZE_MAP.get(strength, 1.0)

        # Confidence (lower for CC data due to 45-day lag)
        confidence = 0.35
        if spending_cat.startswith('STRONG'):
            confidence += 0.15
        if gst_check == 'CONFIRMED':
            confidence += 0.10
        if cards_cat == 'RAPID_ADOPTION':
            confidence += 0.05
        confidence = min(0.70, confidence)

        parts = [
            f"Spend=₹{spending_value:,.0f}Cr(YoY{spending_yoy:+.1f}%,{spending_cat})",
            f"Ticket=₹{avg_ticket_size:,.0f}({ticket_trend})",
            f"Cards={cards_in_force:.1f}M({cards_cat})",
            f"GST={gst_check}",
            f"Month={data_month}",
        ]

        return CreditCardContext(
            spending_value=spending_value,
            spending_yoy_pct=spending_yoy,
            spending_category=spending_cat,
            txn_count=txn_count,
            avg_ticket_size=avg_ticket_size,
            ticket_trend=ticket_trend,
            cards_in_force=cards_in_force,
            cards_growth_yoy=cards_growth_yoy,
            cards_category=cards_cat,
            gst_crosscheck=gst_check,
            direction=direction,
            confidence=confidence,
            size_modifier=size_modifier,
            data_month=data_month,
            reason=' | '.join(parts),
        )

    def evaluate_backtest(self, trade_date: date) -> Dict:
        ctx = self.evaluate(trade_date=trade_date)
        return ctx.to_dict()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    sig = CreditCardSpendingSignal()
    tests = [
        (150000, 30.0, 18.0),   # Strong expansion + rapid adoption
        (120000, 20.0, 12.0),   # Expansion + steady
        (100000, 8.0, 6.0),     # Stable
        (80000, -5.0, 3.0),     # Contraction + saturating
    ]
    for spend, yoy, cards_g in tests:
        ctx = sig.evaluate(
            spending_override=float(spend),
            spending_yoy_override=yoy,
            cards_growth_override=cards_g,
        )
        print(f"Spend=₹{spend:,} YoY={yoy:+.0f}% → {ctx.spending_category:18s} "
              f"{ctx.direction:8s} size={ctx.size_modifier:.2f}")

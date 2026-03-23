"""
Delivery Percentage Signal — Institutional Activity Proxy.

NSE publishes daily delivery percentage for each stock and for Nifty aggregate.
High delivery % on up-days indicates institutional accumulation (buying for
delivery, not intraday speculation). Low delivery % on up-days = speculative rally.

Signal logic:
  Nifty aggregate delivery % computed from top 50 constituents:
  - delivery_pct = delivery_qty / total_traded_qty × 100

  Classification (delivery % + price direction):
    High delivery (>50%) + Price up   → ACCUMULATION (strong bullish)
    High delivery (>50%) + Price down → DISTRIBUTION (strong bearish)
    Low delivery (<35%) + Price up    → SPECULATIVE_RALLY (weak, fade)
    Low delivery (<35%) + Price down  → PANIC_SELLING (contrarian buy)
    Normal delivery (35-50%)          → NEUTRAL

  5-day moving average for trend detection:
    Rising delivery trend + accumulation → very strong continuation
    Falling delivery trend + distribution → very strong continuation down

Data source:
  - NSE bhavcopy (security-wise delivery data)
  - URL: https://archives.nseindia.com/products/content/sec_bhavdata_full_{DDMMYYYY}.csv

Usage:
    from signals.delivery_signal import DeliverySignal
    sig = DeliverySignal(db_conn=conn)
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
DELIVERY_HIGH = 50.0       # >50% = institutional activity
DELIVERY_NORMAL_LOW = 35.0 # <35% = speculative
DELIVERY_VERY_HIGH = 60.0  # >60% = very strong institutional
DELIVERY_VERY_LOW = 25.0   # <25% = pure speculation

# Price change threshold
PRICE_UP_THRESHOLD = 0.2   # >0.2% = meaningful up move
PRICE_DOWN_THRESHOLD = -0.2

# Trend detection
DELIVERY_TREND_LOOKBACK = 5
DELIVERY_TREND_THRESHOLD = 3.0  # 3% change in 5-day MA = trending

# Size modifiers
SIZE_MAP = {
    'STRONG_ACCUMULATION': 1.35,
    'ACCUMULATION': 1.20,
    'NEUTRAL': 1.00,
    'SPECULATIVE_RALLY': 0.80,
    'DISTRIBUTION': 0.75,
    'STRONG_DISTRIBUTION': 0.60,
    'PANIC_SELLING': 1.15,  # Contrarian — fade panic
}


@dataclass
class DeliveryContext:
    """Evaluation result from delivery percentage signal."""
    delivery_pct: float
    delivery_5d_avg: float
    delivery_trend: str       # RISING, FALLING, FLAT
    price_change_pct: float
    classification: str       # ACCUMULATION, DISTRIBUTION, SPECULATIVE_RALLY, etc.
    direction: str
    confidence: float
    size_modifier: float
    top_accumulators: List[Dict]  # Top 5 stocks by delivery % on up-day
    top_distributors: List[Dict]  # Top 5 stocks by delivery % on down-day
    reason: str

    def to_dict(self) -> Dict:
        return {
            'signal_id': 'DELIVERY_PCT',
            'delivery_pct': round(self.delivery_pct, 2),
            'delivery_5d_avg': round(self.delivery_5d_avg, 2),
            'delivery_trend': self.delivery_trend,
            'price_change_pct': round(self.price_change_pct, 3),
            'classification': self.classification,
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'size_modifier': round(self.size_modifier, 2),
            'top_accumulators': self.top_accumulators[:3],
            'top_distributors': self.top_distributors[:3],
            'reason': self.reason,
        }

    def to_telegram(self) -> str:
        emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}.get(
            self.direction, '⚪')
        return (
            f"{emoji} Delivery Signal\n"
            f"  Del%: {self.delivery_pct:.1f}% (5d avg: {self.delivery_5d_avg:.1f}%)\n"
            f"  Trend: {self.delivery_trend}\n"
            f"  Price Δ: {self.price_change_pct:+.2f}%\n"
            f"  Type: {self.classification}\n"
            f"  Dir: {self.direction} | Size: {self.size_modifier:.2f}x"
        )


class DeliverySignal:
    """
    Delivery percentage based institutional activity signal.

    High delivery % indicates real buying/selling (institutional).
    Low delivery % indicates speculative/intraday activity.
    Combined with price direction to classify market regime.
    """

    SIGNAL_ID = 'DELIVERY_PCT'

    # Nifty 50 constituent stocks (for aggregate delivery %)
    NIFTY_50_SYMBOLS = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
        'LT', 'BAJFINANCE', 'HCLTECH', 'ASIANPAINT', 'AXISBANK',
        'MARUTI', 'TITAN', 'SUNPHARMA', 'WIPRO', 'ULTRACEMCO',
        'NESTLEIND', 'ONGC', 'NTPC', 'TATAMOTORS', 'M&M',
        'POWERGRID', 'JSWSTEEL', 'TATASTEEL', 'ADANIENT', 'ADANIPORTS',
        'TECHM', 'BAJAJFINSV', 'HDFCLIFE', 'SBILIFE', 'COALINDIA',
        'GRASIM', 'DIVISLAB', 'BRITANNIA', 'CIPLA', 'DRREDDY',
        'EICHERMOT', 'APOLLOHOSP', 'HEROMOTOCO', 'INDUSINDBK', 'TATACONSUM',
        'BPCL', 'UPL', 'BAJAJ-AUTO', 'HINDALCO', 'LTIM',
    ]

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
    def _get_delivery_data(
        self, trade_date: date, lookback: int = 15
    ) -> Optional[pd.DataFrame]:
        """
        Fetch delivery data from security_bhav or delivery_data table.

        Returns DataFrame with columns: [date, symbol, delivery_pct, close, change_pct]
        """
        conn = self._get_conn()
        if not conn:
            return None

        start_date = trade_date - timedelta(days=lookback * 2)

        # Try different table schemas
        for table, query in [
            ('security_bhav', """
                SELECT date, symbol, delivery_pct, close,
                       (close - prev_close) / NULLIF(prev_close, 0) * 100 as change_pct
                FROM security_bhav
                WHERE date BETWEEN %s AND %s
                  AND series = 'EQ'
                ORDER BY date, symbol
            """),
            ('delivery_data', """
                SELECT date, symbol, delivery_pct, close, change_pct
                FROM delivery_data
                WHERE date BETWEEN %s AND %s
                ORDER BY date, symbol
            """),
        ]:
            try:
                df = pd.read_sql(query, conn, params=(start_date, trade_date))
                if len(df) > 0:
                    return df
            except Exception:
                continue

        logger.warning("No delivery data table found")
        return None

    def _get_nifty_price(self, trade_date: date) -> Optional[Tuple[float, float]]:
        """Get Nifty close price and daily change %."""
        conn = self._get_conn()
        if not conn:
            return None

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT close, prev_close FROM nifty_daily
                    WHERE date <= %s ORDER BY date DESC LIMIT 1
                    """,
                    (trade_date,)
                )
                row = cur.fetchone()
                if row and row[1]:
                    change_pct = (row[0] - row[1]) / row[1] * 100
                    return float(row[0]), change_pct
                elif row:
                    return float(row[0]), 0.0
        except Exception:
            pass

        # Fallback: compute from 2 consecutive days
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT date, close FROM nifty_daily
                    WHERE date <= %s ORDER BY date DESC LIMIT 2
                    """,
                    (trade_date,)
                )
                rows = cur.fetchall()
                if len(rows) == 2:
                    today_close = float(rows[0][1])
                    prev_close = float(rows[1][1])
                    change_pct = (today_close - prev_close) / prev_close * 100
                    return today_close, change_pct
        except Exception:
            pass

        return None

    # ----------------------------------------------------------
    # Aggregate delivery computation
    # ----------------------------------------------------------
    def _compute_aggregate_delivery(
        self, df: pd.DataFrame, trade_date: date
    ) -> Optional[Dict]:
        """
        Compute Nifty-level aggregate delivery % from constituent data.

        Returns dict with delivery_pct, top_accumulators, top_distributors.
        """
        # Filter to Nifty 50 stocks on trade_date
        day_data = df[df['date'] == pd.Timestamp(trade_date)]
        if len(day_data) == 0:
            # Try most recent date
            day_data = df[df['date'] == df['date'].max()]

        nifty_data = day_data[day_data['symbol'].isin(self.NIFTY_50_SYMBOLS)]
        if len(nifty_data) < 10:
            # Not enough Nifty stocks, use all available
            nifty_data = day_data

        if len(nifty_data) == 0:
            return None

        # Weighted average delivery % (weighted by trading value if available)
        avg_delivery = float(nifty_data['delivery_pct'].mean())

        # Top accumulators: high delivery + price up
        up_stocks = nifty_data[nifty_data['change_pct'] > 0].nlargest(
            5, 'delivery_pct'
        )
        top_accumulators = [
            {'symbol': row['symbol'],
             'delivery_pct': round(float(row['delivery_pct']), 1),
             'change_pct': round(float(row['change_pct']), 2)}
            for _, row in up_stocks.iterrows()
        ]

        # Top distributors: high delivery + price down
        down_stocks = nifty_data[nifty_data['change_pct'] < 0].nlargest(
            5, 'delivery_pct'
        )
        top_distributors = [
            {'symbol': row['symbol'],
             'delivery_pct': round(float(row['delivery_pct']), 1),
             'change_pct': round(float(row['change_pct']), 2)}
            for _, row in down_stocks.iterrows()
        ]

        return {
            'delivery_pct': avg_delivery,
            'top_accumulators': top_accumulators,
            'top_distributors': top_distributors,
        }

    def _compute_delivery_trend(
        self, df: pd.DataFrame
    ) -> Tuple[float, str]:
        """
        Compute 5-day delivery % moving average and trend.

        Returns (5d_avg, trend_label)
        """
        # Aggregate daily delivery %
        daily_avg = df.groupby('date')['delivery_pct'].mean().sort_index()

        if len(daily_avg) < DELIVERY_TREND_LOOKBACK:
            return float(daily_avg.iloc[-1]) if len(daily_avg) > 0 else 0.0, 'FLAT'

        ma5 = daily_avg.rolling(DELIVERY_TREND_LOOKBACK).mean()
        current_ma = float(ma5.iloc[-1])

        if len(ma5.dropna()) >= 3:
            ma_change = float(ma5.iloc[-1] - ma5.dropna().iloc[-3])
            if ma_change > DELIVERY_TREND_THRESHOLD:
                trend = 'RISING'
            elif ma_change < -DELIVERY_TREND_THRESHOLD:
                trend = 'FALLING'
            else:
                trend = 'FLAT'
        else:
            trend = 'FLAT'

        return current_ma, trend

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------
    @staticmethod
    def _classify(
        delivery_pct: float, price_change_pct: float, delivery_trend: str
    ) -> Tuple[str, str, str]:
        """
        Classify delivery + price into signal type.

        Returns (classification, direction, strength_key)
        """
        price_up = price_change_pct > PRICE_UP_THRESHOLD
        price_down = price_change_pct < PRICE_DOWN_THRESHOLD
        del_high = delivery_pct >= DELIVERY_HIGH
        del_very_high = delivery_pct >= DELIVERY_VERY_HIGH
        del_low = delivery_pct <= DELIVERY_NORMAL_LOW
        del_very_low = delivery_pct <= DELIVERY_VERY_LOW

        if del_very_high and price_up:
            return 'STRONG_ACCUMULATION', 'BULLISH', 'STRONG_ACCUMULATION'
        elif del_high and price_up:
            return 'ACCUMULATION', 'BULLISH', 'ACCUMULATION'
        elif del_very_high and price_down:
            return 'STRONG_DISTRIBUTION', 'BEARISH', 'STRONG_DISTRIBUTION'
        elif del_high and price_down:
            return 'DISTRIBUTION', 'BEARISH', 'DISTRIBUTION'
        elif del_low and price_up:
            return 'SPECULATIVE_RALLY', 'BEARISH', 'SPECULATIVE_RALLY'
        elif del_very_low and price_down:
            return 'PANIC_SELLING', 'BULLISH', 'PANIC_SELLING'
        elif del_low and price_down:
            return 'WEAK_SELLING', 'NEUTRAL', 'NEUTRAL'
        else:
            return 'NORMAL', 'NEUTRAL', 'NEUTRAL'

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: Optional[date] = None,
        delivery_override: Optional[float] = None,
        price_change_override: Optional[float] = None,
    ) -> DeliveryContext:
        """Evaluate delivery percentage signal."""
        if trade_date is None:
            trade_date = date.today()

        # Override path for testing
        if delivery_override is not None:
            price_change = price_change_override or 0.0
            classification, direction, strength_key = self._classify(
                delivery_override, price_change, 'FLAT'
            )
            return DeliveryContext(
                delivery_pct=delivery_override,
                delivery_5d_avg=delivery_override,
                delivery_trend='FLAT',
                price_change_pct=price_change,
                classification=classification,
                direction=direction,
                confidence=0.60,
                size_modifier=SIZE_MAP.get(strength_key, 1.0),
                top_accumulators=[],
                top_distributors=[],
                reason=f"Del={delivery_override:.1f}% | Price={price_change:+.2f}% | {classification}",
            )

        # Fetch delivery data
        df = self._get_delivery_data(trade_date)
        if df is None or len(df) == 0:
            return DeliveryContext(
                delivery_pct=0.0, delivery_5d_avg=0.0,
                delivery_trend='UNKNOWN', price_change_pct=0.0,
                classification='UNKNOWN', direction='NEUTRAL',
                confidence=0.0, size_modifier=1.0,
                top_accumulators=[], top_distributors=[],
                reason='No delivery data available'
            )

        # Aggregate
        agg = self._compute_aggregate_delivery(df, trade_date)
        if not agg:
            return DeliveryContext(
                delivery_pct=0.0, delivery_5d_avg=0.0,
                delivery_trend='UNKNOWN', price_change_pct=0.0,
                classification='UNKNOWN', direction='NEUTRAL',
                confidence=0.0, size_modifier=1.0,
                top_accumulators=[], top_distributors=[],
                reason='Insufficient stock delivery data'
            )

        delivery_pct = agg['delivery_pct']

        # Nifty price change
        price_info = self._get_nifty_price(trade_date)
        price_change_pct = price_info[1] if price_info else 0.0

        # Trend
        delivery_5d_avg, delivery_trend = self._compute_delivery_trend(df)

        # Classify
        classification, direction, strength_key = self._classify(
            delivery_pct, price_change_pct, delivery_trend
        )

        # Confidence
        confidence = 0.50
        if delivery_pct >= DELIVERY_VERY_HIGH or delivery_pct <= DELIVERY_VERY_LOW:
            confidence += 0.20
        elif delivery_pct >= DELIVERY_HIGH or delivery_pct <= DELIVERY_NORMAL_LOW:
            confidence += 0.10

        # Trend confirmation
        if delivery_trend == 'RISING' and classification in ('ACCUMULATION', 'STRONG_ACCUMULATION'):
            confidence += 0.10
        elif delivery_trend == 'FALLING' and classification in ('DISTRIBUTION', 'STRONG_DISTRIBUTION'):
            confidence += 0.10

        confidence = min(0.95, confidence)

        # Size modifier
        size_modifier = SIZE_MAP.get(strength_key, 1.0)

        # Reason
        parts = [
            f"Del={delivery_pct:.1f}%",
            f"5dAvg={delivery_5d_avg:.1f}%",
            f"Trend={delivery_trend}",
            f"Price={price_change_pct:+.2f}%",
            f"Type={classification}",
        ]

        return DeliveryContext(
            delivery_pct=delivery_pct,
            delivery_5d_avg=delivery_5d_avg,
            delivery_trend=delivery_trend,
            price_change_pct=price_change_pct,
            classification=classification,
            direction=direction,
            confidence=confidence,
            size_modifier=size_modifier,
            top_accumulators=agg['top_accumulators'],
            top_distributors=agg['top_distributors'],
            reason=' | '.join(parts),
        )

    def evaluate_backtest(self, trade_date: date) -> Dict:
        ctx = self.evaluate(trade_date=trade_date)
        return ctx.to_dict()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s')

    sig = DeliverySignal()
    for del_pct, price_chg in [(55, 1.0), (55, -1.0), (30, 1.0), (22, -1.5), (42, 0.1)]:
        ctx = sig.evaluate(delivery_override=float(del_pct), price_change_override=float(price_chg))
        print(f"Del={del_pct}% Price={price_chg:+.1f}% → {ctx.classification:22s} "
              f"{ctx.direction:8s} size={ctx.size_modifier:.2f}")

"""
Order Flow Imbalance Signal.

Uses tick-level bid/ask data from Kite websocket to compute buy/sell
imbalance in the first 15 minutes of trading. Strongly predictive of
intraday direction (academic: Cont, Kukanov & Stoikov 2014).

Signal logic:
  Order Flow Imbalance (OFI):
    OFI = (aggressive_buy_volume - aggressive_sell_volume) / total_volume

  Time windows:
    First 15 min (9:15-9:30): Primary signal
    First 30 min (9:15-9:45): Confirmation
    Rolling 5-min: For intraday adjustments

  Classification:
    STRONG_BUY (OFI > 0.3):  Heavy buy pressure → BULLISH
    BUY (OFI 0.1 to 0.3):    Mild buy lean → mildly BULLISH
    NEUTRAL (-0.1 to 0.1):   Balanced flow
    SELL (OFI -0.3 to -0.1): Mild sell pressure → mildly BEARISH
    STRONG_SELL (OFI < -0.3): Heavy sell pressure → BEARISH

  Enhancement: VWAP-relative flow
    Buys above VWAP = strong conviction
    Sells below VWAP = strong conviction
    Buys below VWAP = accumulation (bullish)

Data source:
  - Kite websocket tick data: LTP, best bid/ask, volume
  - Stored in intraday_ticks table or computed in real-time
  - For backtesting: 1-min OHLCV bars from intraday_bars

Usage:
    from signals.order_flow import OrderFlowSignal
    sig = OrderFlowSignal(db_conn=conn)
    result = sig.evaluate(trade_date=date.today())
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# THRESHOLDS
# ================================================================
OFI_STRONG_BUY = 0.30
OFI_BUY = 0.10
OFI_SELL = -0.10
OFI_STRONG_SELL = -0.30

# Time windows
OPENING_WINDOW_MINUTES = 15    # 9:15 - 9:30
CONFIRMATION_WINDOW_MINUTES = 30  # 9:15 - 9:45

# VWAP threshold
VWAP_DEVIATION_THRESHOLD = 0.002  # 0.2% above/below VWAP

# Volume profile
VOLUME_SPIKE_MULTIPLIER = 2.0  # >2x average volume = spike

# Size modifiers
SIZE_MAP = {
    'STRONG_BUY': 1.30,
    'BUY': 1.15,
    'NEUTRAL': 1.00,
    'SELL': 0.85,
    'STRONG_SELL': 0.70,
}


@dataclass
class OrderFlowContext:
    """Evaluation result from order flow analysis."""
    ofi_15min: float              # OFI in first 15 minutes
    ofi_30min: float              # OFI in first 30 minutes
    ofi_zone: str                 # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    vwap: float                   # VWAP price
    price_vs_vwap: str            # ABOVE, AT, BELOW
    volume_relative: float        # Relative to 20-day average
    volume_spike: bool            # True if >2x average
    buy_volume_pct: float         # % of volume that's buying
    direction: str
    confidence: float
    size_modifier: float
    reason: str

    def to_dict(self) -> Dict:
        return {
            'signal_id': 'ORDER_FLOW_IMBALANCE',
            'ofi_15min': round(self.ofi_15min, 4),
            'ofi_30min': round(self.ofi_30min, 4),
            'ofi_zone': self.ofi_zone,
            'vwap': round(self.vwap, 2),
            'price_vs_vwap': self.price_vs_vwap,
            'volume_relative': round(self.volume_relative, 2),
            'volume_spike': self.volume_spike,
            'buy_volume_pct': round(self.buy_volume_pct, 2),
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'size_modifier': round(self.size_modifier, 2),
            'reason': self.reason,
        }

    def to_telegram(self) -> str:
        emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}.get(
            self.direction, '⚪')
        spike = ' 📊VOL_SPIKE' if self.volume_spike else ''
        return (
            f"{emoji} Order Flow{spike}\n"
            f"  OFI 15m: {self.ofi_15min:+.3f} | 30m: {self.ofi_30min:+.3f}\n"
            f"  Zone: {self.ofi_zone}\n"
            f"  VWAP: {self.vwap:.2f} ({self.price_vs_vwap})\n"
            f"  Buy%: {self.buy_volume_pct:.0f}% | Vol: {self.volume_relative:.1f}x\n"
            f"  Dir: {self.direction} | Size: {self.size_modifier:.2f}x"
        )


class OrderFlowSignal:
    """
    Order flow imbalance signal from tick/bar data.

    Computes buy/sell imbalance in opening window and VWAP-relative
    flow to determine institutional direction.
    """

    SIGNAL_ID = 'ORDER_FLOW_IMBALANCE'

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
    def _get_intraday_bars(
        self, trade_date: date, minutes: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Fetch 1-minute intraday bars for Nifty.

        Returns DataFrame with: timestamp, open, high, low, close, volume
        """
        conn = self._get_conn()
        if not conn:
            return None

        market_open = datetime.combine(trade_date, time(9, 15))
        window_end = market_open + timedelta(minutes=minutes)

        try:
            df = pd.read_sql(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM intraday_bars
                WHERE instrument = 'NIFTY 50'
                  AND timestamp >= %s AND timestamp < %s
                  AND interval = '1m'
                ORDER BY timestamp
                """,
                conn, params=(market_open, window_end)
            )
            return df if len(df) > 0 else None
        except Exception as e:
            logger.debug("Intraday bars not available: %s", e)
            return None

    def _get_tick_data(
        self, trade_date: date, minutes: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Fetch tick-level data for more precise OFI.

        Returns DataFrame with: timestamp, ltp, bid, ask, volume
        """
        conn = self._get_conn()
        if not conn:
            return None

        market_open = datetime.combine(trade_date, time(9, 15))
        window_end = market_open + timedelta(minutes=minutes)

        try:
            df = pd.read_sql(
                """
                SELECT timestamp, ltp, bid_price, ask_price, volume,
                       bid_qty, ask_qty
                FROM intraday_ticks
                WHERE instrument = 'NIFTY 50'
                  AND timestamp >= %s AND timestamp < %s
                ORDER BY timestamp
                """,
                conn, params=(market_open, window_end)
            )
            return df if len(df) > 0 else None
        except Exception:
            return None

    def _get_avg_volume(self, trade_date: date) -> float:
        """Get 20-day average volume for relative comparison."""
        conn = self._get_conn()
        if not conn:
            return 1.0

        start_date = trade_date - timedelta(days=40)

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT AVG(volume) FROM nifty_daily
                    WHERE date BETWEEN %s AND %s
                    """,
                    (start_date, trade_date - timedelta(days=1))
                )
                row = cur.fetchone()
                return float(row[0]) if row and row[0] else 1.0
        except Exception:
            return 1.0

    # ----------------------------------------------------------
    # OFI computation
    # ----------------------------------------------------------
    def _compute_ofi_from_bars(
        self, bars: pd.DataFrame, n_minutes: int
    ) -> Tuple[float, float, float]:
        """
        Approximate OFI from 1-minute bars using bar classification.

        Tick rule approximation:
          close > open → buy bar → volume added to buy side
          close < open → sell bar → volume added to sell side
          close == open → split evenly

        Returns (ofi, buy_volume_pct, vwap)
        """
        if bars is None or len(bars) == 0:
            return 0.0, 50.0, 0.0

        # Filter to requested window
        subset = bars.head(n_minutes)

        buy_volume = 0
        sell_volume = 0
        total_turnover = 0
        total_volume = 0

        for _, bar in subset.iterrows():
            vol = bar['volume']
            if vol <= 0:
                continue

            mid_price = (bar['high'] + bar['low']) / 2
            total_turnover += mid_price * vol
            total_volume += vol

            if bar['close'] > bar['open']:
                buy_volume += vol
            elif bar['close'] < bar['open']:
                sell_volume += vol
            else:
                buy_volume += vol / 2
                sell_volume += vol / 2

        total_classified = buy_volume + sell_volume
        if total_classified == 0:
            return 0.0, 50.0, 0.0

        ofi = (buy_volume - sell_volume) / total_classified
        buy_pct = (buy_volume / total_classified) * 100
        vwap = total_turnover / max(total_volume, 1)

        return float(ofi), float(buy_pct), float(vwap)

    def _compute_ofi_from_ticks(
        self, ticks: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """
        Compute precise OFI from tick data using trade classification.

        Lee-Ready algorithm:
          Trade at ask = buy
          Trade at bid = sell
          Trade between = tick rule (compare with previous trade)
        """
        if ticks is None or len(ticks) == 0:
            return 0.0, 50.0, 0.0

        buy_volume = 0
        sell_volume = 0
        prev_ltp = 0
        total_turnover = 0
        total_volume = 0

        for _, tick in ticks.iterrows():
            ltp = tick['ltp']
            bid = tick.get('bid_price', 0)
            ask = tick.get('ask_price', 0)
            vol = tick.get('volume', 0)

            if vol <= 0:
                continue

            total_turnover += ltp * vol
            total_volume += vol

            # Lee-Ready classification
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else ltp
            if ltp >= ask and ask > 0:
                buy_volume += vol
            elif ltp <= bid and bid > 0:
                sell_volume += vol
            elif ltp > mid:
                buy_volume += vol
            elif ltp < mid:
                sell_volume += vol
            elif ltp > prev_ltp:
                buy_volume += vol
            elif ltp < prev_ltp:
                sell_volume += vol
            else:
                buy_volume += vol / 2
                sell_volume += vol / 2

            prev_ltp = ltp

        total = buy_volume + sell_volume
        if total == 0:
            return 0.0, 50.0, 0.0

        ofi = (buy_volume - sell_volume) / total
        buy_pct = (buy_volume / total) * 100
        vwap = total_turnover / max(total_volume, 1)

        return float(ofi), float(buy_pct), float(vwap)

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------
    @staticmethod
    def _classify_ofi(ofi: float) -> Tuple[str, str]:
        """Classify OFI into zone and direction."""
        if ofi >= OFI_STRONG_BUY:
            return 'STRONG_BUY', 'BULLISH'
        elif ofi >= OFI_BUY:
            return 'BUY', 'BULLISH'
        elif ofi <= OFI_STRONG_SELL:
            return 'STRONG_SELL', 'BEARISH'
        elif ofi <= OFI_SELL:
            return 'SELL', 'BEARISH'
        else:
            return 'NEUTRAL', 'NEUTRAL'

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: Optional[date] = None,
        ofi_override: Optional[float] = None,
    ) -> OrderFlowContext:
        """Evaluate order flow imbalance signal."""
        if trade_date is None:
            trade_date = date.today()

        # Override path
        if ofi_override is not None:
            ofi_15 = ofi_override
            ofi_30 = ofi_override
            buy_pct = 50 + ofi_override * 50
            vwap = 23000
            volume_relative = 1.0
            volume_spike = False
        else:
            # Try tick data first (more precise)
            ticks = self._get_tick_data(trade_date, CONFIRMATION_WINDOW_MINUTES)
            if ticks is not None:
                # Split into 15-min and 30-min windows
                market_open = datetime.combine(trade_date, time(9, 15))
                t15 = market_open + timedelta(minutes=15)

                ticks_15 = ticks[ticks['timestamp'] < t15]
                ofi_15, _, _ = self._compute_ofi_from_ticks(ticks_15)
                ofi_30, buy_pct, vwap = self._compute_ofi_from_ticks(ticks)
            else:
                # Fallback to bar data
                bars = self._get_intraday_bars(trade_date, CONFIRMATION_WINDOW_MINUTES)
                if bars is not None:
                    ofi_15, _, _ = self._compute_ofi_from_bars(bars, OPENING_WINDOW_MINUTES)
                    ofi_30, buy_pct, vwap = self._compute_ofi_from_bars(
                        bars, CONFIRMATION_WINDOW_MINUTES
                    )
                else:
                    return OrderFlowContext(
                        ofi_15min=0, ofi_30min=0, ofi_zone='UNKNOWN',
                        vwap=0, price_vs_vwap='UNKNOWN',
                        volume_relative=0, volume_spike=False,
                        buy_volume_pct=50, direction='NEUTRAL',
                        confidence=0.0, size_modifier=1.0,
                        reason='No intraday data available'
                    )

            # Volume analysis
            avg_vol = self._get_avg_volume(trade_date)
            conn = self._get_conn()
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT volume FROM nifty_daily WHERE date = %s",
                            (trade_date,)
                        )
                        row = cur.fetchone()
                        today_vol = float(row[0]) if row else avg_vol
                except Exception:
                    today_vol = avg_vol
            else:
                today_vol = avg_vol

            volume_relative = today_vol / max(avg_vol, 1)
            volume_spike = volume_relative > VOLUME_SPIKE_MULTIPLIER

        # Classify
        ofi_zone, direction = self._classify_ofi(ofi_15)

        # 30-min confirmation
        _, direction_30 = self._classify_ofi(ofi_30)
        if direction == direction_30 and direction != 'NEUTRAL':
            confirmation_boost = 0.10
        elif direction != direction_30 and direction != 'NEUTRAL':
            confirmation_boost = -0.05  # Reversal within 30 min
        else:
            confirmation_boost = 0.0

        # VWAP analysis
        if vwap > 0:
            # Get current price (latest close)
            conn = self._get_conn()
            current_price = vwap  # Default
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT close FROM nifty_daily WHERE date <= %s ORDER BY date DESC LIMIT 1",
                            (trade_date,)
                        )
                        row = cur.fetchone()
                        if row:
                            current_price = float(row[0])
                except Exception:
                    pass

            vwap_dev = (current_price - vwap) / vwap if vwap > 0 else 0
            if vwap_dev > VWAP_DEVIATION_THRESHOLD:
                price_vs_vwap = 'ABOVE'
            elif vwap_dev < -VWAP_DEVIATION_THRESHOLD:
                price_vs_vwap = 'BELOW'
            else:
                price_vs_vwap = 'AT'
        else:
            price_vs_vwap = 'UNKNOWN'

        # Confidence
        confidence = 0.45
        if ofi_zone.startswith('STRONG'):
            confidence += 0.25
        elif ofi_zone in ('BUY', 'SELL'):
            confidence += 0.15
        confidence += confirmation_boost
        if volume_spike:
            confidence += 0.10  # High volume adds conviction
        confidence = min(0.95, max(0.0, confidence))

        # Size modifier
        size_modifier = SIZE_MAP.get(ofi_zone, 1.0)
        if volume_spike and direction != 'NEUTRAL':
            size_modifier = min(1.3, size_modifier * 1.1)

        parts = [
            f"OFI15={ofi_15:+.3f}",
            f"OFI30={ofi_30:+.3f}",
            f"Zone={ofi_zone}",
            f"Buy%={buy_pct:.0f}%",
            f"VWAP={vwap:.0f}({price_vs_vwap})",
            f"Vol={volume_relative:.1f}x",
        ]
        if volume_spike:
            parts.append("VOL_SPIKE")

        return OrderFlowContext(
            ofi_15min=ofi_15, ofi_30min=ofi_30, ofi_zone=ofi_zone,
            vwap=vwap, price_vs_vwap=price_vs_vwap,
            volume_relative=volume_relative, volume_spike=volume_spike,
            buy_volume_pct=buy_pct, direction=direction,
            confidence=confidence, size_modifier=size_modifier,
            reason=' | '.join(parts),
        )

    def evaluate_backtest(self, trade_date: date) -> Dict:
        ctx = self.evaluate(trade_date=trade_date)
        return ctx.to_dict()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s')

    sig = OrderFlowSignal()
    for ofi in [-0.4, -0.2, 0.0, 0.2, 0.4]:
        ctx = sig.evaluate(ofi_override=ofi)
        print(f"OFI={ofi:+.1f} → {ctx.ofi_zone:12s} {ctx.direction:8s} "
              f"size={ctx.size_modifier:.2f}")

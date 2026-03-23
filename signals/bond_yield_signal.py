"""
Bond Yield Spread Signal — India-US yield differential for FII flow prediction.

The India-US 10Y bond yield spread is a key driver of FII equity flows.
When the spread widens (India yield rises relative to US), carry trade
favors India → FII inflows → bullish for Nifty.
When spread narrows → FII outflow pressure → bearish.

Signal logic:
  Spread = India_10Y - US_10Y

  Spread zones:
    WIDE (>4.5%):        Strong FII inflow pressure → BULLISH
    NORMAL (3.0-4.5%):   Neutral
    NARROW (<3.0%):      FII outflow risk → BEARISH
    INVERTED (<2.0%):    Severe outflow risk → STRONG BEARISH

  Spread momentum (20-day change):
    Widening >30bps:     BULLISH acceleration
    Narrowing >30bps:    BEARISH acceleration

  DXY correlation:
    Strong DXY (>104) + narrow spread → double negative for India
    Weak DXY (<100) + wide spread → double positive

Data source:
  - US 10Y: ^TNX via yfinance (already in global_market_fetcher)
  - India 10Y: IN10Y via yfinance or RBI FBIL data
  - DXY: DX-Y.NYB via yfinance (already fetched)

Usage:
    from signals.bond_yield_signal import BondYieldSignal
    sig = BondYieldSignal(db_conn=conn)
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
SPREAD_WIDE = 4.50          # % — strong carry trade
SPREAD_NORMAL_HIGH = 4.50
SPREAD_NORMAL_LOW = 3.00
SPREAD_NARROW = 3.00
SPREAD_CRITICAL = 2.00      # % — severe outflow risk

# Momentum
SPREAD_MOMENTUM_LOOKBACK = 20  # trading days
SPREAD_MOMENTUM_THRESHOLD = 0.30  # 30 bps

# DXY thresholds
DXY_STRONG = 104.0
DXY_WEAK = 100.0

# Size modifiers
SIZE_MAP = {
    'STRONG_BULLISH': 1.25,
    'BULLISH': 1.15,
    'NEUTRAL': 1.00,
    'BEARISH': 0.85,
    'STRONG_BEARISH': 0.70,
}


@dataclass
class BondYieldContext:
    """Evaluation result from bond yield spread signal."""
    india_10y: float
    us_10y: float
    spread: float
    spread_zone: str          # WIDE, NORMAL, NARROW, CRITICAL
    spread_20d_change: float  # bps
    spread_momentum: str      # WIDENING, NARROWING, FLAT
    dxy_level: float
    dxy_impact: str           # POSITIVE, NEUTRAL, NEGATIVE
    direction: str
    confidence: float
    size_modifier: float
    reason: str

    def to_dict(self) -> Dict:
        return {
            'signal_id': 'BOND_YIELD_SPREAD',
            'india_10y': round(self.india_10y, 3),
            'us_10y': round(self.us_10y, 3),
            'spread': round(self.spread, 3),
            'spread_zone': self.spread_zone,
            'spread_20d_change': round(self.spread_20d_change, 1),
            'spread_momentum': self.spread_momentum,
            'dxy_level': round(self.dxy_level, 2),
            'dxy_impact': self.dxy_impact,
            'direction': self.direction,
            'confidence': round(self.confidence, 3),
            'size_modifier': round(self.size_modifier, 2),
            'reason': self.reason,
        }

    def to_telegram(self) -> str:
        emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}.get(
            self.direction, '⚪')
        return (
            f"{emoji} Bond Yield Spread\n"
            f"  IN10Y: {self.india_10y:.2f}% | US10Y: {self.us_10y:.2f}%\n"
            f"  Spread: {self.spread:.2f}% ({self.spread_zone})\n"
            f"  20d Δ: {self.spread_20d_change:+.0f}bps ({self.spread_momentum})\n"
            f"  DXY: {self.dxy_level:.1f} ({self.dxy_impact})\n"
            f"  Dir: {self.direction} | Size: {self.size_modifier:.2f}x"
        )


class BondYieldSignal:
    """
    India-US bond yield spread signal for FII flow prediction.

    Wider spread → carry trade favors India → bullish
    Narrower spread → outflow risk → bearish
    """

    SIGNAL_ID = 'BOND_YIELD_SPREAD'

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
    def _get_yield_data(
        self, trade_date: date, lookback: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        Fetch India 10Y and US 10Y yield data.

        Returns DataFrame with columns: [date, india_10y, us_10y, spread]
        """
        conn = self._get_conn()
        if not conn:
            return None

        start_date = trade_date - timedelta(days=lookback * 2)

        # Try bond_yields table
        try:
            df = pd.read_sql(
                """
                SELECT date, india_10y, us_10y,
                       (india_10y - us_10y) as spread
                FROM bond_yields
                WHERE date BETWEEN %s AND %s
                ORDER BY date
                """,
                conn, params=(start_date, trade_date)
            )
            if len(df) >= 5:
                return df
        except Exception:
            pass

        # Try global_market_daily (US 10Y might be there as ^TNX)
        try:
            us_df = pd.read_sql(
                """
                SELECT trade_date as date, close as us_10y
                FROM global_market_daily
                WHERE instrument = '^TNX'
                  AND trade_date BETWEEN %s AND %s
                ORDER BY trade_date
                """,
                conn, params=(start_date, trade_date)
            )
            if len(us_df) >= 5:
                # India 10Y: try to fetch separately or use fixed estimate
                india_10y = 7.10  # Approximate current India 10Y
                us_df['india_10y'] = india_10y
                us_df['spread'] = us_df['india_10y'] - us_df['us_10y']
                return us_df[['date', 'india_10y', 'us_10y', 'spread']]
        except Exception:
            pass

        # Fallback: try yfinance
        try:
            import yfinance as yf
            us = yf.download('^TNX', start=start_date, end=trade_date + timedelta(days=1),
                            progress=False)
            if len(us) >= 5:
                df = pd.DataFrame({
                    'date': us.index,
                    'us_10y': us['Close'].values.flatten(),
                })
                df['india_10y'] = 7.10  # Will be updated when India data available
                df['spread'] = df['india_10y'] - df['us_10y']
                df['date'] = pd.to_datetime(df['date']).dt.date
                return df.reset_index(drop=True)
        except Exception as e:
            logger.debug("yfinance yield fetch failed: %s", e)

        return None

    def _get_dxy(self, trade_date: date) -> float:
        """Fetch DXY level."""
        conn = self._get_conn()
        if not conn:
            return 103.0

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT close FROM global_market_daily
                    WHERE instrument = 'DX-Y.NYB'
                      AND trade_date <= %s
                    ORDER BY trade_date DESC LIMIT 1
                    """,
                    (trade_date,)
                )
                row = cur.fetchone()
                return float(row[0]) if row else 103.0
        except Exception:
            return 103.0

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------
    @staticmethod
    def _classify_spread(spread: float) -> Tuple[str, str]:
        """Classify spread into zone and direction."""
        if spread >= SPREAD_WIDE:
            return 'WIDE', 'BULLISH'
        elif spread >= SPREAD_NORMAL_LOW:
            return 'NORMAL', 'NEUTRAL'
        elif spread >= SPREAD_CRITICAL:
            return 'NARROW', 'BEARISH'
        else:
            return 'CRITICAL', 'BEARISH'

    @staticmethod
    def _classify_dxy(dxy: float) -> str:
        """Classify DXY impact on India flows."""
        if dxy >= DXY_STRONG:
            return 'NEGATIVE'  # Strong dollar → outflows
        elif dxy <= DXY_WEAK:
            return 'POSITIVE'  # Weak dollar → inflows
        else:
            return 'NEUTRAL'

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: Optional[date] = None,
        spread_override: Optional[float] = None,
        dxy_override: Optional[float] = None,
    ) -> BondYieldContext:
        """Evaluate bond yield spread signal."""
        if trade_date is None:
            trade_date = date.today()

        # Override path
        if spread_override is not None:
            india_10y = 7.10
            us_10y = india_10y - spread_override
            spread = spread_override
            spread_20d_change = 0.0
            spread_momentum = 'FLAT'
        else:
            yield_data = self._get_yield_data(trade_date)
            if yield_data is None or len(yield_data) < 5:
                return BondYieldContext(
                    india_10y=0.0, us_10y=0.0, spread=0.0,
                    spread_zone='UNKNOWN', spread_20d_change=0.0,
                    spread_momentum='FLAT', dxy_level=103.0,
                    dxy_impact='NEUTRAL', direction='NEUTRAL',
                    confidence=0.0, size_modifier=1.0,
                    reason='No bond yield data available'
                )

            india_10y = float(yield_data['india_10y'].iloc[-1])
            us_10y = float(yield_data['us_10y'].iloc[-1])
            spread = float(yield_data['spread'].iloc[-1])

            # 20-day momentum
            if len(yield_data) >= SPREAD_MOMENTUM_LOOKBACK + 1:
                spread_20d_change = (
                    float(yield_data['spread'].iloc[-1]) -
                    float(yield_data['spread'].iloc[-SPREAD_MOMENTUM_LOOKBACK - 1])
                ) * 100  # Convert to bps
            else:
                spread_20d_change = 0.0

            if spread_20d_change > SPREAD_MOMENTUM_THRESHOLD * 100:
                spread_momentum = 'WIDENING'
            elif spread_20d_change < -SPREAD_MOMENTUM_THRESHOLD * 100:
                spread_momentum = 'NARROWING'
            else:
                spread_momentum = 'FLAT'

        # DXY
        dxy = dxy_override if dxy_override is not None else self._get_dxy(trade_date)
        dxy_impact = self._classify_dxy(dxy)

        # Classify spread
        spread_zone, direction = self._classify_spread(spread)

        # DXY can amplify or dampen signal
        if dxy_impact == 'NEGATIVE' and direction == 'BEARISH':
            # Double negative → strengthen bearish
            direction = 'BEARISH'
            confidence_boost = 0.10
        elif dxy_impact == 'POSITIVE' and direction == 'BULLISH':
            # Double positive → strengthen bullish
            confidence_boost = 0.10
        elif dxy_impact == 'NEGATIVE' and direction == 'BULLISH':
            # Contradiction → reduce conviction
            confidence_boost = -0.10
        elif dxy_impact == 'POSITIVE' and direction == 'BEARISH':
            confidence_boost = -0.10
        else:
            confidence_boost = 0.0

        # Confidence
        confidence = 0.45
        if spread_zone in ('WIDE', 'CRITICAL'):
            confidence += 0.20
        elif spread_zone == 'NARROW':
            confidence += 0.10
        if abs(spread_20d_change) > 50:
            confidence += 0.10
        confidence += confidence_boost
        confidence = min(0.95, max(0.0, confidence))

        # Signal strength
        if spread_zone == 'WIDE' and spread_momentum == 'WIDENING':
            strength = 'STRONG_BULLISH'
        elif spread_zone == 'WIDE':
            strength = 'BULLISH'
        elif spread_zone == 'CRITICAL':
            strength = 'STRONG_BEARISH'
        elif spread_zone == 'NARROW' and spread_momentum == 'NARROWING':
            strength = 'STRONG_BEARISH'
        elif spread_zone == 'NARROW':
            strength = 'BEARISH'
        else:
            strength = 'NEUTRAL'

        size_modifier = SIZE_MAP.get(strength, 1.0)

        parts = [
            f"IN10Y={india_10y:.2f}%",
            f"US10Y={us_10y:.2f}%",
            f"Spread={spread:.2f}%({spread_zone})",
            f"20dΔ={spread_20d_change:+.0f}bps",
            f"DXY={dxy:.1f}({dxy_impact})",
        ]

        return BondYieldContext(
            india_10y=india_10y, us_10y=us_10y, spread=spread,
            spread_zone=spread_zone, spread_20d_change=spread_20d_change,
            spread_momentum=spread_momentum, dxy_level=dxy,
            dxy_impact=dxy_impact, direction=direction,
            confidence=confidence, size_modifier=size_modifier,
            reason=' | '.join(parts),
        )

    def evaluate_backtest(self, trade_date: date) -> Dict:
        ctx = self.evaluate(trade_date=trade_date)
        return ctx.to_dict()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s')

    sig = BondYieldSignal()
    for sp in [5.0, 4.0, 3.0, 2.5, 1.5]:
        ctx = sig.evaluate(spread_override=sp, dxy_override=103.0)
        print(f"Spread={sp:.1f}% → {ctx.spread_zone:8s} {ctx.direction:8s} "
              f"size={ctx.size_modifier:.2f}")

"""
Market Sentiment Signal — Tickertape Market Mood Index + Google Trends.

Combines two free sentiment data sources for contrarian/confirmation signals:

1. Tickertape Market Mood Index (MMI):
   - Composite fear/greed gauge for Indian markets
   - Range: 0 (extreme fear) to 100 (extreme greed)
   - Contrarian at extremes: MMI < 25 → bullish, MMI > 75 → bearish
   - Data: https://www.tickertape.in/market-mood-index (scrape or API)

2. Google Trends:
   - Search interest for panic/fear terms: "nifty crash", "stock market fall"
   - 1-week mean reversion: spike in fear searches → bullish 5 days later
   - Confirmation: sustained high interest → continued move
   - Data: pytrends library (Google Trends API)

Signal logic:
  MMI zones:
    EXTREME_FEAR (0-20):   Strong contrarian BULLISH (historically +2.1x returns)
    FEAR (20-35):          Mild contrarian BULLISH
    NEUTRAL (35-65):       No signal
    GREED (65-80):         Mild contrarian BEARISH
    EXTREME_GREED (80-100): Strong contrarian BEARISH

  Google Trends confirmation:
    Fear search spike (>2x normal) + EXTREME_FEAR → boost confidence
    Fear search spike + GREED → contradiction → reduce confidence

Usage:
    from signals.sentiment_signal import SentimentSignal
    sig = SentimentSignal(db_conn=conn)
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
# MMI THRESHOLDS
# ================================================================
MMI_EXTREME_FEAR = 20.0
MMI_FEAR = 35.0
MMI_NEUTRAL_LOW = 35.0
MMI_NEUTRAL_HIGH = 65.0
MMI_GREED = 65.0
MMI_EXTREME_GREED = 80.0

# ================================================================
# GOOGLE TRENDS THRESHOLDS
# ================================================================
GTRENDS_SPIKE_MULTIPLIER = 2.0    # >2x normal = spike
GTRENDS_LOOKBACK_WEEKS = 12       # Baseline period
GTRENDS_KEYWORDS = [
    'nifty crash', 'stock market fall', 'market crash india',
    'nifty down', 'sensex crash'
]

# ================================================================
# SIZE MODIFIERS
# ================================================================
SIZE_MAP = {
    'EXTREME_FEAR': 1.30,      # Contrarian buy
    'FEAR': 1.15,
    'NEUTRAL': 1.00,
    'GREED': 0.85,
    'EXTREME_GREED': 0.70,     # Contrarian sell / reduce
}


@dataclass
class SentimentContext:
    """Evaluation result from sentiment signal."""
    mmi_value: float              # 0-100
    mmi_zone: str                 # EXTREME_FEAR, FEAR, NEUTRAL, GREED, EXTREME_GREED
    mmi_5d_change: float          # 5-day change
    gtrends_score: float          # Relative search interest (1.0 = normal)
    gtrends_spike: bool           # True if fear search spike detected
    combined_direction: str       # BULLISH, BEARISH, NEUTRAL
    confidence: float
    size_modifier: float
    is_contrarian: bool           # True at extreme readings
    reason: str

    def to_dict(self) -> Dict:
        return {
            'signal_id': 'SENTIMENT_COMPOSITE',
            'mmi_value': round(self.mmi_value, 1),
            'mmi_zone': self.mmi_zone,
            'mmi_5d_change': round(self.mmi_5d_change, 1),
            'gtrends_score': round(self.gtrends_score, 2),
            'gtrends_spike': self.gtrends_spike,
            'direction': self.combined_direction,
            'combined_direction': self.combined_direction,
            'confidence': round(self.confidence, 3),
            'size_modifier': round(self.size_modifier, 2),
            'is_contrarian': self.is_contrarian,
            'reason': self.reason,
        }

    def to_telegram(self) -> str:
        emoji = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}.get(
            self.combined_direction, '⚪')
        ctr = ' ⚡CONTRARIAN' if self.is_contrarian else ''
        spike = ' 📈FEAR_SPIKE' if self.gtrends_spike else ''
        return (
            f"{emoji} Sentiment Signal{ctr}{spike}\n"
            f"  MMI: {self.mmi_value:.0f} ({self.mmi_zone})\n"
            f"  MMI 5d Δ: {self.mmi_5d_change:+.0f}\n"
            f"  GTrends: {self.gtrends_score:.1f}x normal\n"
            f"  Dir: {self.combined_direction} | Size: {self.size_modifier:.2f}x"
        )


class SentimentSignal:
    """
    Composite sentiment signal combining MMI and Google Trends.

    Both signals are contrarian at extremes and confirmation in the middle.
    """

    SIGNAL_ID = 'SENTIMENT_COMPOSITE'

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
    # MMI data retrieval
    # ----------------------------------------------------------
    def _get_mmi_data(
        self, trade_date: date, lookback: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Fetch Market Mood Index from database.

        Returns DataFrame with columns: [date, mmi_value]
        """
        conn = self._get_conn()
        if not conn:
            return None

        start_date = trade_date - timedelta(days=lookback * 2)

        for table, query in [
            ('market_mood_index', """
                SELECT date, mmi_value FROM market_mood_index
                WHERE date BETWEEN %s AND %s ORDER BY date
            """),
            ('sentiment_data', """
                SELECT date, mmi_value FROM sentiment_data
                WHERE date BETWEEN %s AND %s ORDER BY date
            """),
        ]:
            try:
                df = pd.read_sql(query, conn, params=(start_date, trade_date))
                if len(df) >= 3:
                    return df
            except Exception:
                continue

        return None

    def _fetch_mmi_live(self) -> Optional[float]:
        """
        Fetch live MMI from Tickertape.

        Returns MMI value or None if unavailable.
        """
        try:
            import requests
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/120.0.0.0 Safari/537.36',
            }
            resp = requests.get(
                'https://www.tickertape.in/market-mood-index',
                headers=headers, timeout=10
            )
            if resp.status_code == 200:
                # Parse MMI value from page (JSON embedded)
                import re
                match = re.search(r'"mmiValue":\s*(\d+\.?\d*)', resp.text)
                if match:
                    return float(match.group(1))
        except Exception as e:
            logger.debug("MMI live fetch failed: %s", e)
        return None

    # ----------------------------------------------------------
    # Google Trends data
    # ----------------------------------------------------------
    def _get_gtrends_data(
        self, trade_date: date
    ) -> Optional[Dict]:
        """
        Fetch Google Trends data for fear keywords.

        Returns dict with score (relative to baseline) and spike flag.
        """
        conn = self._get_conn()
        if conn:
            # Try stored data first
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT score, is_spike FROM google_trends_sentiment
                        WHERE date = %s
                        """,
                        (trade_date,)
                    )
                    row = cur.fetchone()
                    if row:
                        return {'score': float(row[0]), 'spike': bool(row[1])}
            except Exception:
                pass

        # Try live fetch via pytrends
        try:
            from pytrends.request import TrendReq
            pytrends = TrendReq(hl='en-IN', tz=330)  # IST
            kw_list = GTRENDS_KEYWORDS[:3]  # Max 5 keywords per request
            pytrends.build_payload(
                kw_list,
                cat=0,
                timeframe=f'{(trade_date - timedelta(days=90)).isoformat()} {trade_date.isoformat()}',
                geo='IN',
            )
            df = pytrends.interest_over_time()
            if df is not None and len(df) > 0:
                # Average across keywords
                avg_interest = df[kw_list].mean(axis=1)
                baseline = avg_interest.iloc[:-7].mean() if len(avg_interest) > 7 else 50
                current = avg_interest.iloc[-7:].mean()
                score = current / max(baseline, 1)
                spike = score > GTRENDS_SPIKE_MULTIPLIER
                return {'score': float(score), 'spike': spike}
        except Exception as e:
            logger.debug("Google Trends fetch failed: %s", e)

        return None

    # ----------------------------------------------------------
    # Classification
    # ----------------------------------------------------------
    @staticmethod
    def _classify_mmi(mmi: float) -> Tuple[str, str]:
        """Classify MMI into zone and contrarian direction."""
        if mmi <= MMI_EXTREME_FEAR:
            return 'EXTREME_FEAR', 'BULLISH'
        elif mmi <= MMI_FEAR:
            return 'FEAR', 'BULLISH'
        elif mmi >= MMI_EXTREME_GREED:
            return 'EXTREME_GREED', 'BEARISH'
        elif mmi >= MMI_GREED:
            return 'GREED', 'BEARISH'
        else:
            return 'NEUTRAL', 'NEUTRAL'

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: Optional[date] = None,
        mmi_override: Optional[float] = None,
        gtrends_override: Optional[Dict] = None,
    ) -> SentimentContext:
        """Evaluate composite sentiment signal."""
        if trade_date is None:
            trade_date = date.today()

        # Get MMI
        if mmi_override is not None:
            mmi_value = mmi_override
            mmi_5d_change = 0.0
        else:
            mmi_data = self._get_mmi_data(trade_date)
            if mmi_data is not None and len(mmi_data) >= 2:
                mmi_value = float(mmi_data['mmi_value'].iloc[-1])
                if len(mmi_data) >= 6:
                    mmi_5d_change = float(
                        mmi_data['mmi_value'].iloc[-1] - mmi_data['mmi_value'].iloc[-6]
                    )
                else:
                    mmi_5d_change = 0.0
            else:
                # Try live
                live_mmi = self._fetch_mmi_live()
                if live_mmi is not None:
                    mmi_value = live_mmi
                    mmi_5d_change = 0.0
                else:
                    return SentimentContext(
                        mmi_value=50.0, mmi_zone='UNKNOWN',
                        mmi_5d_change=0.0, gtrends_score=1.0,
                        gtrends_spike=False, combined_direction='NEUTRAL',
                        confidence=0.0, size_modifier=1.0,
                        is_contrarian=False,
                        reason='No sentiment data available'
                    )

        # Classify MMI
        mmi_zone, mmi_direction = self._classify_mmi(mmi_value)

        # Get Google Trends
        if gtrends_override is not None:
            gtrends = gtrends_override
        else:
            gtrends = self._get_gtrends_data(trade_date)

        gtrends_score = gtrends['score'] if gtrends else 1.0
        gtrends_spike = gtrends['spike'] if gtrends else False

        # Combine signals
        combined_direction = mmi_direction

        # Google Trends confirmation/contradiction
        confidence = 0.50
        if mmi_zone.startswith('EXTREME'):
            confidence += 0.20
            if gtrends_spike and mmi_zone == 'EXTREME_FEAR':
                # Fear spike + extreme fear = very strong contrarian buy
                confidence += 0.15
            elif gtrends_spike and mmi_zone == 'EXTREME_GREED':
                # Fear spike contradicts greed → uncertainty
                confidence -= 0.10
        elif mmi_zone in ('FEAR', 'GREED'):
            confidence += 0.10

        # MMI momentum
        if abs(mmi_5d_change) > 15:
            confidence += 0.05  # Rapid sentiment shift

        confidence = min(0.95, max(0.0, confidence))

        # Is contrarian?
        is_contrarian = mmi_zone in ('EXTREME_FEAR', 'EXTREME_GREED')

        # Size modifier
        size_modifier = SIZE_MAP.get(mmi_zone, 1.0)

        # Build reason
        parts = [
            f"MMI={mmi_value:.0f}",
            f"Zone={mmi_zone}",
            f"5dΔ={mmi_5d_change:+.0f}",
            f"GTrends={gtrends_score:.1f}x",
        ]
        if gtrends_spike:
            parts.append("FEAR_SPIKE")
        if is_contrarian:
            parts.append("CONTRARIAN")

        return SentimentContext(
            mmi_value=mmi_value,
            mmi_zone=mmi_zone,
            mmi_5d_change=mmi_5d_change,
            gtrends_score=gtrends_score,
            gtrends_spike=gtrends_spike,
            combined_direction=combined_direction,
            confidence=confidence,
            size_modifier=size_modifier,
            is_contrarian=is_contrarian,
            reason=' | '.join(parts),
        )

    def evaluate_backtest(self, trade_date: date) -> Dict:
        ctx = self.evaluate(trade_date=trade_date)
        return ctx.to_dict()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(name)s — %(message)s')

    sig = SentimentSignal()
    for mmi in [15, 30, 50, 70, 85]:
        ctx = sig.evaluate(mmi_override=float(mmi))
        print(f"MMI={mmi:3d} → {ctx.mmi_zone:16s} {ctx.combined_direction:8s} "
              f"size={ctx.size_modifier:.2f} ctr={ctx.is_contrarian}")

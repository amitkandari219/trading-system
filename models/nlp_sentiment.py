"""
NLP Sentiment Analysis — Earnings Calls + RBI Speech Analysis.

Two NLP signal sources:
1. Earnings call sentiment for Nifty 50 constituents
   - Aggregate sentiment of top 10 weightage companies' latest calls
   - Positive surprise → bullish, negative surprise → bearish
   - Uses FinBERT or LLM-based classification

2. RBI Governor speech sentiment
   - Hawkish vs dovish tone from MPC statements
   - Forward guidance extraction
   - Rate decision prediction

Architecture:
  - Text → Tokenizer → Transformer/FinBERT → Sentiment score [-1, 1]
  - Aggregate across companies with market-cap weighting
  - Combine with consensus surprise (actual vs estimated)

Data sources:
  - Earnings transcripts: BSE/NSE filings, MoneyControl, Screener.in
  - RBI speeches: rbi.org.in press releases
  - Fallback: news headlines from RSS feeds

Usage:
    from models.nlp_sentiment import NLPSentiment
    nlp = NLPSentiment(db_conn=conn)
    result = nlp.evaluate(trade_date=date.today())
"""

import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# CONFIGURATION
# ================================================================
MODEL_DIR = Path(__file__).parent / 'nlp'

# Top weightage Nifty 50 stocks for earnings analysis
TOP_WEIGHT_STOCKS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
    'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK',
]

# RBI hawkish/dovish keyword lists
HAWKISH_KEYWORDS = [
    'inflation concerns', 'price stability', 'tightening', 'rate hike',
    'upside risks to inflation', 'vigilant', 'restrictive', 'calibrated',
    'withdrawal of accommodation', 'elevated inflation', 'persistent',
    'above target', 'supply-side pressures',
]

DOVISH_KEYWORDS = [
    'growth supportive', 'accommodative', 'rate cut', 'easing',
    'downside risks to growth', 'benign inflation', 'below target',
    'supportive of recovery', 'liquidity surplus', 'growth impulse',
    'transmission of rate cuts', 'soft landing',
]

# Earnings sentiment keywords
POSITIVE_EARNINGS_KW = [
    'beat estimates', 'record revenue', 'margin expansion', 'strong growth',
    'guidance raised', 'market share gain', 'robust demand', 'outperformed',
    'exceeded expectations', 'positive surprise', 'upgrade',
]

NEGATIVE_EARNINGS_KW = [
    'missed estimates', 'revenue decline', 'margin compression', 'weak demand',
    'guidance lowered', 'market share loss', 'underperformed', 'downgrade',
    'below expectations', 'negative surprise', 'challenging environment',
]

# Size modifiers
SENTIMENT_SIZE_MAP = {
    'VERY_POSITIVE': 1.20,
    'POSITIVE': 1.10,
    'NEUTRAL': 1.00,
    'NEGATIVE': 0.90,
    'VERY_NEGATIVE': 0.80,
}


class KeywordSentimentAnalyzer:
    """
    Simple keyword-based sentiment analyzer.

    For production: replace with FinBERT or GPT-based analysis.
    """

    def __init__(self):
        self.positive_patterns = [re.compile(kw, re.IGNORECASE)
                                   for kw in POSITIVE_EARNINGS_KW]
        self.negative_patterns = [re.compile(kw, re.IGNORECASE)
                                   for kw in NEGATIVE_EARNINGS_KW]
        self.hawkish_patterns = [re.compile(kw, re.IGNORECASE)
                                  for kw in HAWKISH_KEYWORDS]
        self.dovish_patterns = [re.compile(kw, re.IGNORECASE)
                                 for kw in DOVISH_KEYWORDS]

    def analyze_earnings(self, text: str) -> Dict:
        """Analyze earnings call/report text."""
        pos_count = sum(1 for p in self.positive_patterns if p.search(text))
        neg_count = sum(1 for p in self.negative_patterns if p.search(text))
        total = pos_count + neg_count

        if total == 0:
            return {'score': 0.0, 'sentiment': 'NEUTRAL', 'pos': 0, 'neg': 0}

        score = (pos_count - neg_count) / total
        if score > 0.3:
            sentiment = 'POSITIVE'
        elif score < -0.3:
            sentiment = 'NEGATIVE'
        else:
            sentiment = 'NEUTRAL'

        return {'score': float(score), 'sentiment': sentiment,
                'pos': pos_count, 'neg': neg_count}

    def analyze_rbi(self, text: str) -> Dict:
        """Analyze RBI policy statement."""
        hawk_count = sum(1 for p in self.hawkish_patterns if p.search(text))
        dove_count = sum(1 for p in self.dovish_patterns if p.search(text))
        total = hawk_count + dove_count

        if total == 0:
            return {'score': 0.0, 'stance': 'NEUTRAL', 'hawkish': 0, 'dovish': 0}

        score = (dove_count - hawk_count) / total  # Positive = dovish = bullish for equity
        if score > 0.3:
            stance = 'DOVISH'
        elif score < -0.3:
            stance = 'HAWKISH'
        else:
            stance = 'NEUTRAL'

        return {'score': float(score), 'stance': stance,
                'hawkish': hawk_count, 'dovish': dove_count}


class NLPSentiment:
    """
    NLP-based sentiment signal combining earnings and RBI analysis.
    """

    SIGNAL_ID = 'NLP_SENTIMENT'

    def __init__(self, db_conn=None):
        self.conn = db_conn
        self.analyzer = KeywordSentimentAnalyzer()

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
    def _get_earnings_data(
        self, trade_date: date, lookback_days: int = 90
    ) -> List[Dict]:
        """Fetch recent earnings transcripts/headlines."""
        conn = self._get_conn()
        if not conn:
            return []

        start_date = trade_date - timedelta(days=lookback_days)

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT symbol, headline, content, date
                    FROM earnings_data
                    WHERE date BETWEEN %s AND %s
                      AND symbol IN %s
                    ORDER BY date DESC
                    """,
                    (start_date, trade_date, tuple(TOP_WEIGHT_STOCKS))
                )
                rows = cur.fetchall()
                return [
                    {'symbol': r[0], 'headline': r[1], 'content': r[2] or '', 'date': r[3]}
                    for r in rows
                ]
        except Exception:
            return []

    def _get_rbi_statements(
        self, trade_date: date, lookback_days: int = 90
    ) -> List[Dict]:
        """Fetch recent RBI policy statements."""
        conn = self._get_conn()
        if not conn:
            return []

        start_date = trade_date - timedelta(days=lookback_days)

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT title, content, date FROM rbi_statements
                    WHERE date BETWEEN %s AND %s
                    ORDER BY date DESC
                    """,
                    (start_date, trade_date)
                )
                rows = cur.fetchall()
                return [
                    {'title': r[0], 'content': r[1] or '', 'date': r[2]}
                    for r in rows
                ]
        except Exception:
            return []

    def _get_news_headlines(
        self, trade_date: date, lookback_days: int = 7
    ) -> List[Dict]:
        """Fetch recent market news headlines as fallback."""
        conn = self._get_conn()
        if not conn:
            return []

        start_date = trade_date - timedelta(days=lookback_days)

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT headline, source, date FROM market_news
                    WHERE date BETWEEN %s AND %s
                    ORDER BY date DESC LIMIT 50
                    """,
                    (start_date, trade_date)
                )
                rows = cur.fetchall()
                return [
                    {'headline': r[0], 'source': r[1], 'date': r[2]}
                    for r in rows
                ]
        except Exception:
            return []

    # ----------------------------------------------------------
    # Analysis
    # ----------------------------------------------------------
    def _analyze_earnings_aggregate(
        self, earnings: List[Dict]
    ) -> Dict:
        """Aggregate earnings sentiment across companies."""
        if not earnings:
            return {'score': 0.0, 'sentiment': 'NEUTRAL', 'n_analyzed': 0, 'by_stock': {}}

        scores_by_stock = {}
        for item in earnings:
            text = f"{item.get('headline', '')} {item.get('content', '')}"
            result = self.analyzer.analyze_earnings(text)
            stock = item['symbol']
            if stock not in scores_by_stock:
                scores_by_stock[stock] = []
            scores_by_stock[stock].append(result['score'])

        # Weighted average (equal weight for now; use market cap later)
        all_scores = []
        stock_summaries = {}
        for stock, scores in scores_by_stock.items():
            avg = np.mean(scores)
            stock_summaries[stock] = round(float(avg), 3)
            all_scores.extend(scores)

        overall = float(np.mean(all_scores)) if all_scores else 0.0

        if overall > 0.3:
            sentiment = 'POSITIVE'
        elif overall > 0.1:
            sentiment = 'MILDLY_POSITIVE'
        elif overall < -0.3:
            sentiment = 'NEGATIVE'
        elif overall < -0.1:
            sentiment = 'MILDLY_NEGATIVE'
        else:
            sentiment = 'NEUTRAL'

        return {
            'score': round(overall, 3),
            'sentiment': sentiment,
            'n_analyzed': len(earnings),
            'by_stock': stock_summaries,
        }

    def _analyze_rbi_sentiment(
        self, statements: List[Dict]
    ) -> Dict:
        """Analyze RBI policy stance."""
        if not statements:
            return {'score': 0.0, 'stance': 'NEUTRAL', 'n_analyzed': 0}

        scores = []
        for stmt in statements:
            text = f"{stmt.get('title', '')} {stmt.get('content', '')}"
            result = self.analyzer.analyze_rbi(text)
            scores.append(result['score'])

        overall = float(np.mean(scores))
        if overall > 0.2:
            stance = 'DOVISH'
        elif overall < -0.2:
            stance = 'HAWKISH'
        else:
            stance = 'NEUTRAL'

        return {
            'score': round(overall, 3),
            'stance': stance,
            'n_analyzed': len(statements),
        }

    # ----------------------------------------------------------
    # Main evaluation
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: Optional[date] = None,
        earnings_override: Optional[float] = None,
        rbi_override: Optional[float] = None,
    ) -> Dict:
        """Evaluate NLP sentiment signal."""
        if trade_date is None:
            trade_date = date.today()

        # Override path for testing
        if earnings_override is not None or rbi_override is not None:
            earnings_score = earnings_override or 0.0
            rbi_score = rbi_override or 0.0
            combined = earnings_score * 0.6 + rbi_score * 0.4
        else:
            # Fetch and analyze
            earnings_data = self._get_earnings_data(trade_date)
            earnings_result = self._analyze_earnings_aggregate(earnings_data)

            rbi_data = self._get_rbi_statements(trade_date)
            rbi_result = self._analyze_rbi_sentiment(rbi_data)

            earnings_score = earnings_result['score']
            rbi_score = rbi_result['score']

            # Weighted combination (earnings 60%, RBI 40%)
            combined = earnings_score * 0.6 + rbi_score * 0.4

        # Classify
        if combined > 0.3:
            category = 'VERY_POSITIVE'
            direction = 'BULLISH'
        elif combined > 0.1:
            category = 'POSITIVE'
            direction = 'BULLISH'
        elif combined < -0.3:
            category = 'VERY_NEGATIVE'
            direction = 'BEARISH'
        elif combined < -0.1:
            category = 'NEGATIVE'
            direction = 'BEARISH'
        else:
            category = 'NEUTRAL'
            direction = 'NEUTRAL'

        size_modifier = SENTIMENT_SIZE_MAP.get(category, 1.0)

        confidence = min(0.80, 0.30 + abs(combined))

        return {
            'signal_id': self.SIGNAL_ID,
            'earnings_score': round(float(earnings_score), 3),
            'rbi_score': round(float(rbi_score), 3),
            'combined_score': round(float(combined), 3),
            'category': category,
            'direction': direction,
            'confidence': round(float(confidence), 3),
            'size_modifier': round(float(size_modifier), 2),
            'method': 'keyword_based',
            'reason': f"Earnings={earnings_score:+.3f} | RBI={rbi_score:+.3f} | "
                      f"Combined={combined:+.3f} | {category}",
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    nlp = NLPSentiment()
    for earn, rbi in [(0.5, 0.3), (0.0, 0.0), (-0.4, -0.2), (0.3, -0.5)]:
        result = nlp.evaluate(earnings_override=earn, rbi_override=rbi)
        print(f"Earnings={earn:+.1f} RBI={rbi:+.1f} → {result['category']:16s} "
              f"{result['direction']:8s} size={result['size_modifier']:.2f}")

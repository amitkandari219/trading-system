"""
NLP News Sentiment Signal.

Processes financial news headlines for sentiment scoring to generate
trading signals.  When no trained model is available, falls back to
keyword-based sentiment analysis.

Fallback (no model):
    Positive keywords: "rally", "surge", "record high", "FII buying",
                       "rate cut", "bullish", "breakout", "upgrade",
                       "outperform", "strong results"
    Negative keywords: "crash", "selloff", "FII selling", "rate hike",
                       "recession", "bearish", "breakdown", "downgrade",
                       "underperform", "weak results"
    Score = (positive_count - negative_count) / total_headlines
    LONG when score > 0.3, SHORT when score < -0.3

Column expectations:
    - news_headlines: JSON list of headline strings
    - OR news_sentiment_score: precomputed float score

Model path: models/nlp_sentiment.pkl

Usage:
    from signals.ml.nlp_sentiment import NLPSentimentSignal

    nlp = NLPSentimentSignal()
    result = nlp.evaluate(df, date)
"""

import json
import logging
import os
import pickle
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

SIGNAL_ID = 'NLP_SENTIMENT'

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

# Keyword lists for fallback sentiment
POSITIVE_KEYWORDS = [
    'rally', 'surge', 'record high', 'fii buying', 'rate cut',
    'bullish', 'breakout', 'upgrade', 'outperform', 'strong results',
    'all-time high', 'buying spree', 'green', 'gains', 'uptick',
    'recovery', 'optimism', 'stimulus', 'growth', 'positive',
    'inflows', 'accumulation', 'beat estimates', 'above expectations',
]

NEGATIVE_KEYWORDS = [
    'crash', 'selloff', 'sell-off', 'fii selling', 'rate hike',
    'recession', 'bearish', 'breakdown', 'downgrade', 'underperform',
    'weak results', 'record low', 'selling spree', 'red', 'losses',
    'correction', 'pessimism', 'crisis', 'slowdown', 'negative',
    'outflows', 'distribution', 'miss estimates', 'below expectations',
]

# Signal thresholds
LONG_THRESHOLD = 0.3
SHORT_THRESHOLD = -0.3

# Confidence
BASE_CONFIDENCE = 0.45
SENTIMENT_CONFIDENCE_SCALE = 0.50  # Extra confidence per unit |score|
MIN_HEADLINES = 3  # Minimum headlines needed for a signal


# ================================================================
# HELPERS
# ================================================================

def _safe_float(val: Any, default: float = float('nan')) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        v = float(val)
        return v
    except (TypeError, ValueError):
        return default


def _extract_headlines(raw: Any) -> List[str]:
    """Extract list of headline strings from various formats."""
    if isinstance(raw, list):
        return [str(h) for h in raw if h]
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith('['):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(h) for h in parsed if h]
            except (json.JSONDecodeError, TypeError):
                pass
        # Treat as single headline or semicolon/newline separated
        if ';' in raw:
            return [h.strip() for h in raw.split(';') if h.strip()]
        if '\n' in raw:
            return [h.strip() for h in raw.split('\n') if h.strip()]
        return [raw] if raw else []
    return []


def _keyword_sentiment(headline: str, positive: List[str], negative: List[str]) -> int:
    """Score a single headline: +1 if positive, -1 if negative, 0 if neutral."""
    h_lower = headline.lower()
    pos_match = any(kw in h_lower for kw in positive)
    neg_match = any(kw in h_lower for kw in negative)

    if pos_match and not neg_match:
        return 1
    elif neg_match and not pos_match:
        return -1
    elif pos_match and neg_match:
        # Ambiguous — count keyword matches
        pos_count = sum(1 for kw in positive if kw in h_lower)
        neg_count = sum(1 for kw in negative if kw in h_lower)
        return 1 if pos_count > neg_count else -1 if neg_count > pos_count else 0
    return 0


# ================================================================
# SIGNAL CLASS
# ================================================================

class NLPSentimentSignal:
    """
    NLP-based news sentiment signal.  Falls back to keyword-based
    sentiment when no trained model is available.
    """

    SIGNAL_ID = SIGNAL_ID
    MODEL_PATH = os.path.join(MODEL_DIR, 'nlp_sentiment.pkl')

    def __init__(self) -> None:
        self._model = self._load_model()
        mode = 'ML' if self._model is not None else 'FALLBACK'
        logger.info('NLPSentimentSignal initialised (mode=%s)', mode)

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_model(self) -> Any:
        """Try to load a trained NLP sentiment model from disk."""
        try:
            if os.path.exists(self.MODEL_PATH):
                with open(self.MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                logger.info('Loaded NLP sentiment model from %s', self.MODEL_PATH)
                return model
        except ImportError:
            logger.warning('transformers/torch not installed — using fallback')
        except Exception as e:
            logger.warning('Failed to load NLP sentiment model: %s', e)
        return None

    # ----------------------------------------------------------
    # ML prediction
    # ----------------------------------------------------------
    def _predict_ml(self, features: Any) -> Dict:
        """Predict sentiment using the trained NLP model."""
        try:
            if isinstance(features, list):
                # Features are headlines — model expects text input
                if hasattr(self._model, 'predict'):
                    scores = self._model.predict(features)
                    if hasattr(scores, '__len__'):
                        avg_score = float(np.mean(scores))
                    else:
                        avg_score = float(scores)
                elif hasattr(self._model, '__call__'):
                    # Pipeline-style model
                    results = self._model(features)
                    scores = []
                    for r in results:
                        if isinstance(r, dict):
                            label = r.get('label', '').lower()
                            score = r.get('score', 0.5)
                            if 'positive' in label:
                                scores.append(score)
                            elif 'negative' in label:
                                scores.append(-score)
                            else:
                                scores.append(0)
                        else:
                            scores.append(float(r))
                    avg_score = np.mean(scores) if scores else 0.0
                else:
                    return {}

                return {
                    'sentiment_score': avg_score,
                    'num_headlines': len(features),
                    'mode': 'ML',
                }
            else:
                # Numeric features
                pred = self._model.predict(np.array(features).reshape(1, -1))
                return {
                    'sentiment_score': float(pred[0]),
                    'mode': 'ML',
                }
        except Exception as e:
            logger.warning('NLP ML prediction failed: %s', e)
            return {}

    # ----------------------------------------------------------
    # Fallback prediction
    # ----------------------------------------------------------
    def _predict_fallback(self, df: Any, dt: date) -> Optional[Dict]:
        """
        Keyword-based sentiment analysis.

        Looks for news_headlines or news_sentiment_score columns
        in the dataframe.
        """
        try:
            if not hasattr(df, 'loc') or not hasattr(df, 'columns'):
                return None

            # Get row for date
            if hasattr(df.index, 'date'):
                row = df.loc[df.index.date == dt]
            else:
                row = df.loc[df.index == str(dt)]

            if hasattr(row, 'empty') and row.empty:
                return None

            if len(row) > 1:
                row = row.iloc[-1:]
            row_data = row.squeeze()

            # Check for precomputed sentiment score
            for col in ['news_sentiment_score', 'sentiment_score', 'sentiment']:
                if col in (row_data.index if hasattr(row_data, 'index') else []):
                    score = _safe_float(row_data[col])
                    if not np.isnan(score):
                        return {
                            'sentiment_score': score,
                            'num_headlines': 0,
                            'headline_summary': 'Precomputed score',
                            'positive_count': 0,
                            'negative_count': 0,
                            'neutral_count': 0,
                            'mode': 'FALLBACK_PRECOMPUTED',
                        }

            # Check for headlines column
            headlines: List[str] = []
            for col in ['news_headlines', 'headlines', 'news']:
                if col in (row_data.index if hasattr(row_data, 'index') else []):
                    raw = row_data[col]
                    headlines = _extract_headlines(raw)
                    break

            if not headlines:
                return None

            if len(headlines) < MIN_HEADLINES:
                logger.debug('Only %d headlines (min=%d) — skipping', len(headlines), MIN_HEADLINES)
                return None

            # Score each headline
            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for h in headlines:
                score = _keyword_sentiment(h, POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS)
                if score > 0:
                    positive_count += 1
                elif score < 0:
                    negative_count += 1
                else:
                    neutral_count += 1

            total = len(headlines)
            sentiment_score = (positive_count - negative_count) / total if total > 0 else 0.0

            # Summarise top headlines
            summary_parts = []
            if positive_count > 0:
                summary_parts.append(f"{positive_count} positive")
            if negative_count > 0:
                summary_parts.append(f"{negative_count} negative")
            if neutral_count > 0:
                summary_parts.append(f"{neutral_count} neutral")
            headline_summary = f"Of {total} headlines: {', '.join(summary_parts)}"

            return {
                'sentiment_score': round(sentiment_score, 4),
                'num_headlines': total,
                'headline_summary': headline_summary,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'mode': 'FALLBACK',
            }
        except Exception as e:
            logger.debug('NLP fallback prediction failed: %s', e)
            return None

    # ----------------------------------------------------------
    # Main evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: Any, dt: date) -> Optional[Dict]:
        """
        Evaluate the NLP sentiment signal.

        Parameters
        ----------
        df : DataFrame with news_headlines or news_sentiment_score column.
        dt : Trade date.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata.
        None if no signal.
        """
        try:
            return self._evaluate_inner(df, dt)
        except Exception as e:
            logger.error('NLPSentimentSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: Any, dt: date) -> Optional[Dict]:
        result = None

        # Try ML mode first
        if self._model is not None:
            # Extract headlines for ML model
            headlines = self._extract_headlines_for_ml(df, dt)
            if headlines:
                result = self._predict_ml(headlines)
                if not result:
                    result = None

        # Fallback
        if result is None:
            result = self._predict_fallback(df, dt)

        if result is None:
            return None

        sentiment_score = result.get('sentiment_score', 0.0)

        # Direction
        if sentiment_score > LONG_THRESHOLD:
            direction = 'LONG'
        elif sentiment_score < SHORT_THRESHOLD:
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'

        # Confidence
        confidence = BASE_CONFIDENCE + abs(sentiment_score) * SENTIMENT_CONFIDENCE_SCALE
        if direction == 'NEUTRAL':
            confidence = max(0.15, confidence * 0.5)
        confidence = max(0.10, min(0.85, confidence))

        # Extract current price
        price = 0.0
        try:
            if hasattr(df, 'loc'):
                close_col = None
                for col in ['close', 'Close', 'last_price']:
                    if col in df.columns:
                        close_col = col
                        break
                if close_col:
                    if hasattr(df.index, 'date'):
                        row = df.loc[df.index.date == dt]
                    else:
                        row = df.loc[df.index == str(dt)]
                    if not row.empty:
                        price = round(float(row[close_col].iloc[-1]), 2)
        except Exception:
            pass

        reason_parts = [
            'NLP_SENTIMENT',
            f"Mode={result.get('mode', 'UNKNOWN')}",
            f"Score={sentiment_score:+.3f}",
            f"Direction={direction}",
            f"Headlines={result.get('num_headlines', 0)}",
        ]

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(confidence, 3),
            'price': price,
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'mode': result.get('mode', 'UNKNOWN'),
                'sentiment_score': sentiment_score,
                'headline_summary': result.get('headline_summary', ''),
                'num_headlines': result.get('num_headlines', 0),
                'positive_count': result.get('positive_count', 0),
                'negative_count': result.get('negative_count', 0),
                'neutral_count': result.get('neutral_count', 0),
                'confidence': round(confidence, 3),
            },
        }

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _extract_headlines_for_ml(self, df: Any, dt: date) -> List[str]:
        """Extract headlines from the dataframe for ML model input."""
        try:
            if not hasattr(df, 'loc'):
                return []

            if hasattr(df.index, 'date'):
                row = df.loc[df.index.date == dt]
            else:
                row = df.loc[df.index == str(dt)]

            if hasattr(row, 'empty') and row.empty:
                return []

            row_data = row.iloc[-1] if len(row) > 1 else row.squeeze()

            for col in ['news_headlines', 'headlines', 'news']:
                if col in (row_data.index if hasattr(row_data, 'index') else []):
                    return _extract_headlines(row_data[col])
        except Exception:
            pass
        return []

    def reset(self) -> None:
        """Reset internal state."""
        pass

    def __repr__(self) -> str:
        mode = 'ML' if self._model is not None else 'FALLBACK'
        return f"NLPSentimentSignal(signal_id='{self.SIGNAL_ID}', mode='{mode}')"

"""
Sentiment and alternative data signals for NSE Nifty trading.

Contrarian indicators derived from retail/social media sentiment,
news regime classification, and broker activity proxies.
"""

from signals.sentiment.google_trends import GoogleTrendsFearSignal
from signals.sentiment.twitter_sentiment import TwitterSentimentSignal
from signals.sentiment.news_event_classifier import NewsEventClassifierSignal
from signals.sentiment.retail_broker_sentiment import RetailBrokerSentimentSignal

__all__ = [
    'GoogleTrendsFearSignal',
    'TwitterSentimentSignal',
    'NewsEventClassifierSignal',
    'RetailBrokerSentimentSignal',
]

"""
Intraday signals for Nifty F&O.

SCORING signals — fire actual trades on 5-min bar data:
    ORBSignal            — Opening Range Breakout (15-min OR)
    VWAPSignal           — VWAP Crossover & Mean-Reversion bands
    MomentumCandleSignal — Wide-range bars, 3-bar momentum, engulfing
    GiftGapSignal        — Pre-market gap fill / gap-follow
    RSIDivergenceSignal  — RSI divergence on 5-min chart

OVERLAY signals — modify sizing and provide real-time context:
    OptionsFlowScanner   — PCR shift, OI buildup, IV skew
    MarketMicrostructure — bid-ask spread, order imbalance, aggressor
    SectorMomentum       — Nifty vs BankNifty relative performance
    TimeSeasonality      — time-of-day sizing rules
    ExpiryScalper        — Thursday expiry gamma/pin/theta effects
"""

from signals.intraday.orb_signal import ORBSignal
from signals.intraday.vwap_signal import VWAPSignal
from signals.intraday.momentum_candles import MomentumCandleSignal
from signals.intraday.gift_gap_signal import GiftGapSignal
from signals.intraday.rsi_divergence import RSIDivergenceSignal
from signals.intraday.options_flow import OptionsFlowScanner
from signals.intraday.microstructure import MarketMicrostructure
from signals.intraday.sector_momentum import SectorMomentum
from signals.intraday.time_seasonality import TimeSeasonality
from signals.intraday.expiry_scalper import ExpiryScalper

__all__ = [
    # Scoring
    'ORBSignal',
    'VWAPSignal',
    'MomentumCandleSignal',
    'GiftGapSignal',
    'RSIDivergenceSignal',
    # Overlay
    'OptionsFlowScanner',
    'MarketMicrostructure',
    'SectorMomentum',
    'TimeSeasonality',
    'ExpiryScalper',
]

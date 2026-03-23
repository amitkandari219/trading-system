"""
Regime detectors for the trading system.

Contains two detectors:
1. RegimeDetector — original 5-regime VIX-based detector (existing, unchanged).
2. UnifiedRegimeDetector — 8-regime detector using VIX + ADX + SMA 50/200.

UnifiedRegimeDetector replaces the simpler GujralRegimeDetector
(paper_trading/regime_detector.py) which only uses shadow trade win rate.
Both can run in parallel — UnifiedRegimeDetector provides structural
classification, Gujral provides signal-quality overlay.

Usage:
    from core.regime_detector import UnifiedRegimeDetector

    detector = UnifiedRegimeDetector()
    result = detector.detect(market_context)
    # result = {
    #     'regime': 'BULL',
    #     'trend': 'UP',
    #     'volatility': 'NORMAL',
    #     'size_modifier': 1.0,
    #     'confidence': 0.75,
    # }
"""

import logging
from enum import Enum
from typing import Dict, Optional

from config.unified_config import (
    ADX_TREND_THRESHOLD,
    VIX_EXTREME,
    VIX_HIGH,
    VIX_LOW,
    VIX_NORMAL,
)

logger = logging.getLogger(__name__)


# ================================================================
# ORIGINAL RegimeDetector (preserved for backward compatibility)
# ================================================================

# VIX-based regime thresholds (from signals.dynamic_regime)
VIX_REGIMES = [
    ('CRISIS',   25.0),
    ('HIGH_VOL', 18.0),
    ('ELEVATED', 14.0),
    ('NORMAL',   10.0),
    ('CALM',      0.0),
]

# Size modifier per regime
REGIME_SIZE_MODIFIER = {
    'CALM':     1.00,
    'NORMAL':   1.00,
    'ELEVATED': 0.60,
    'HIGH_VOL': 0.30,
    'CRISIS':   0.00,
}

REGIME_SEVERITY = {
    'CALM': 0,
    'NORMAL': 1,
    'ELEVATED': 2,
    'HIGH_VOL': 3,
    'CRISIS': 4,
}


class RegimeDetector:
    """
    Stateless regime detector based on VIX + trend.

    Returns a dict with:
        regime: str — regime label
        size_modifier: float — position sizing multiplier (0.0 - 1.0)
        severity: int — 0 (calm) to 4 (crisis)
        trend: str — UPTREND / DOWNTREND / NEUTRAL
    """

    def __init__(self):
        self._last_regime: Optional[str] = None

    def detect(
        self,
        vix: float = 15.0,
        adx: float = 20.0,
        close: float = 0.0,
        sma_50: float = 0.0,
    ) -> Dict:
        """Classify current market regime."""
        regime = self._classify_vix(vix)
        trend = self._classify_trend(close, sma_50, adx)

        # Bull override: low VIX + strong uptrend = promote to CALM even from NORMAL
        if vix < 12 and trend == 'UPTREND' and adx > 25:
            regime = 'CALM'

        result = {
            'regime': regime,
            'size_modifier': REGIME_SIZE_MODIFIER[regime],
            'severity': REGIME_SEVERITY[regime],
            'trend': trend,
            'vix': vix,
            'adx': adx,
        }
        self._last_regime = regime
        return result

    @property
    def last_regime(self) -> Optional[str]:
        return self._last_regime

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_vix(vix: float) -> str:
        for regime, threshold in VIX_REGIMES:
            if vix >= threshold:
                return regime
        return 'CALM'

    @staticmethod
    def _classify_trend(close: float, sma_50: float, adx: float) -> str:
        if sma_50 <= 0 or close <= 0:
            return 'NEUTRAL'
        pct_above = (close - sma_50) / sma_50
        if pct_above > 0.02 and adx > 20:
            return 'UPTREND'
        elif pct_above < -0.02 and adx > 20:
            return 'DOWNTREND'
        return 'NEUTRAL'


# ================================================================
# UNIFIED REGIME DETECTOR — 8 regimes
# ================================================================

class UnifiedRegime(str, Enum):
    """Market regime classifications."""
    STRONG_BULL = 'STRONG_BULL'
    BULL = 'BULL'
    NEUTRAL = 'NEUTRAL'
    BEAR = 'BEAR'
    STRONG_BEAR = 'STRONG_BEAR'
    HIGH_VOL_BULL = 'HIGH_VOL_BULL'
    HIGH_VOL_BEAR = 'HIGH_VOL_BEAR'
    CRISIS = 'CRISIS'


class UnifiedTrend(str, Enum):
    """Trend direction based on SMA alignment."""
    STRONG_UP = 'STRONG_UP'
    UP = 'UP'
    FLAT = 'FLAT'
    DOWN = 'DOWN'
    STRONG_DOWN = 'STRONG_DOWN'


class UnifiedVolatility(str, Enum):
    """Volatility regime from VIX level."""
    LOW = 'LOW'
    NORMAL = 'NORMAL'
    HIGH = 'HIGH'
    EXTREME = 'EXTREME'


# Regime -> position size modifier
# Conservative: reduce in high vol, increase in clear trends
UNIFIED_REGIME_SIZE_MODIFIERS = {
    UnifiedRegime.STRONG_BULL:   1.20,
    UnifiedRegime.BULL:          1.00,
    UnifiedRegime.NEUTRAL:       0.70,
    UnifiedRegime.BEAR:          0.80,
    UnifiedRegime.STRONG_BEAR:   0.60,
    UnifiedRegime.HIGH_VOL_BULL: 0.80,
    UnifiedRegime.HIGH_VOL_BEAR: 0.50,
    UnifiedRegime.CRISIS:        0.30,
}


class UnifiedRegimeDetector:
    """
    Classifies the current market into one of 8 regimes using:
    - VIX level (volatility regime)
    - ADX value (trend strength)
    - SMA 50 vs SMA 200 position (trend direction)
    - Close vs SMA 50/200 (price position)

    The detector is stateless — each call to detect() is independent.
    """

    def detect(self, market_context: dict) -> Dict:
        """
        Classify current market regime from a market context dict.

        Parameters
        ----------
        market_context : dict
            Output from UnifiedDataProvider.get_market_context().
            Required keys: close, sma_50, sma_200, adx_14, india_vix.

        Returns
        -------
        dict with keys:
            regime : str
                One of the 8 UnifiedRegime values.
            trend : str
                Trend direction (STRONG_UP, UP, FLAT, DOWN, STRONG_DOWN).
            volatility : str
                Volatility classification (LOW, NORMAL, HIGH, EXTREME).
            size_modifier : float
                Position size multiplier for this regime (0.30 to 1.20).
            confidence : float
                How clearly the regime is identified (0.0 to 1.0).
                Higher when multiple indicators agree.
        """
        close = market_context.get('close', 0.0)
        sma_50 = market_context.get('sma_50', 0.0)
        sma_200 = market_context.get('sma_200', 0.0)
        adx = market_context.get('adx_14', 0.0)
        vix = market_context.get('india_vix', 0.0)

        # Step 1: Classify volatility from VIX
        vol_regime = self._classify_volatility(vix)

        # Step 2: Classify trend from SMA alignment + ADX
        trend = self._classify_trend(close, sma_50, sma_200, adx)

        # Step 3: Combine into regime
        regime = self._combine_regime(trend, vol_regime, adx, vix)

        # Step 4: Compute confidence (how strongly indicators agree)
        confidence = self._compute_confidence(
            close, sma_50, sma_200, adx, vix, regime
        )

        size_modifier = UNIFIED_REGIME_SIZE_MODIFIERS.get(regime, 0.70)

        result = {
            'regime': regime.value,
            'trend': trend.value,
            'volatility': vol_regime.value,
            'size_modifier': size_modifier,
            'confidence': round(confidence, 3),
        }

        logger.info(
            f"Regime: {regime.value} | trend={trend.value} "
            f"vol={vol_regime.value} | size_mod={size_modifier:.2f} "
            f"conf={confidence:.2f} | VIX={vix:.1f} ADX={adx:.1f}"
        )

        return result

    # ----------------------------------------------------------------
    # INTERNAL CLASSIFIERS
    # ----------------------------------------------------------------

    @staticmethod
    def _classify_volatility(vix: float) -> UnifiedVolatility:
        """Classify volatility regime from India VIX level."""
        if vix >= VIX_EXTREME:
            return UnifiedVolatility.EXTREME
        elif vix >= VIX_HIGH:
            return UnifiedVolatility.HIGH
        elif vix >= VIX_NORMAL:
            return UnifiedVolatility.NORMAL
        else:
            return UnifiedVolatility.LOW

    @staticmethod
    def _classify_trend(close: float, sma_50: float,
                        sma_200: float, adx: float) -> UnifiedTrend:
        """
        Classify trend direction from SMA alignment and ADX strength.

        SMA alignment:
            close > sma_50 > sma_200  ->  UP / STRONG_UP
            close < sma_50 < sma_200  ->  DOWN / STRONG_DOWN
            otherwise                 ->  FLAT

        ADX amplifies:
            ADX >= threshold  ->  STRONG_UP or STRONG_DOWN
            ADX < threshold   ->  UP, DOWN, or FLAT
        """
        if sma_50 <= 0 or sma_200 <= 0:
            return UnifiedTrend.FLAT

        is_trending = adx >= ADX_TREND_THRESHOLD

        if close > sma_50 and sma_50 > sma_200:
            return UnifiedTrend.STRONG_UP if is_trending else UnifiedTrend.UP
        elif close < sma_50 and sma_50 < sma_200:
            return UnifiedTrend.STRONG_DOWN if is_trending else UnifiedTrend.DOWN
        elif close > sma_50:
            # Above 50 but 50 below 200 — early recovery or choppy
            return UnifiedTrend.UP if is_trending else UnifiedTrend.FLAT
        elif close < sma_50:
            # Below 50 but 50 above 200 — early decline or choppy
            return UnifiedTrend.DOWN if is_trending else UnifiedTrend.FLAT
        else:
            return UnifiedTrend.FLAT

    @staticmethod
    def _combine_regime(trend: UnifiedTrend, vol: UnifiedVolatility,
                        adx: float, vix: float) -> UnifiedRegime:
        """
        Combine trend and volatility into a single regime classification.

        Priority order:
        1. CRISIS overrides everything (VIX extreme + bearish trend)
        2. HIGH_VOL variants when VIX is high
        3. STRONG variants when ADX confirms strong trend
        4. Base regimes otherwise
        """
        # CRISIS: extreme VIX + downtrend
        if vol == UnifiedVolatility.EXTREME:
            if trend in (UnifiedTrend.DOWN, UnifiedTrend.STRONG_DOWN):
                return UnifiedRegime.CRISIS
            # Extreme VIX but bullish — rare but possible (post-crash bounce)
            return UnifiedRegime.HIGH_VOL_BULL

        # HIGH VOLATILITY regimes
        if vol == UnifiedVolatility.HIGH:
            if trend in (UnifiedTrend.UP, UnifiedTrend.STRONG_UP):
                return UnifiedRegime.HIGH_VOL_BULL
            elif trend in (UnifiedTrend.DOWN, UnifiedTrend.STRONG_DOWN):
                return UnifiedRegime.HIGH_VOL_BEAR
            else:
                # High vol + flat = treat as bearish (uncertainty)
                return UnifiedRegime.HIGH_VOL_BEAR

        # NORMAL / LOW volatility — trend-driven classification
        if trend == UnifiedTrend.STRONG_UP:
            return UnifiedRegime.STRONG_BULL
        elif trend == UnifiedTrend.UP:
            return UnifiedRegime.BULL
        elif trend == UnifiedTrend.STRONG_DOWN:
            return UnifiedRegime.STRONG_BEAR
        elif trend == UnifiedTrend.DOWN:
            return UnifiedRegime.BEAR
        else:
            return UnifiedRegime.NEUTRAL

    @staticmethod
    def _compute_confidence(close: float, sma_50: float,
                            sma_200: float, adx: float,
                            vix: float, regime: UnifiedRegime) -> float:
        """
        Compute confidence score (0.0 to 1.0) based on indicator agreement.

        Factors:
        - ADX strength: higher ADX = more confident in trend regimes
        - SMA separation: wider gap between 50 and 200 = clearer regime
        - VIX clarity: extreme values (very low or very high) = clearer
        - Price distance from SMA: further from mean = clearer
        """
        scores = []

        # ADX contribution (0-1): strong trend = high confidence
        if adx > 0:
            adx_score = min(1.0, adx / 40.0)  # ADX 40+ = max confidence
            scores.append(adx_score)

        # SMA separation (0-1): how far apart are the SMAs?
        if sma_200 > 0:
            sma_sep = abs(sma_50 - sma_200) / sma_200
            sma_score = min(1.0, sma_sep / 0.05)  # 5% separation = max
            scores.append(sma_score)

        # Price distance from SMA50 (0-1)
        if sma_50 > 0:
            price_dist = abs(close - sma_50) / sma_50
            dist_score = min(1.0, price_dist / 0.03)  # 3% = max
            scores.append(dist_score)

        # VIX clarity: extreme or very low = more confident
        if vix > 0:
            if vix >= VIX_EXTREME or vix <= VIX_LOW:
                scores.append(0.9)
            elif vix >= VIX_HIGH or vix <= 14:
                scores.append(0.6)
            else:
                scores.append(0.4)

        if not scores:
            return 0.5

        # Average of all contributing factor scores
        return sum(scores) / len(scores)

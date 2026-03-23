"""
Temporal Fusion Transformer Forecast Signal.

Multi-horizon forecast producing expected 1-day, 3-day, and 5-day returns.
When no trained model is available, falls back to a simple linear
extrapolation blending 5-day momentum with mean-reversion.

Features (trained mode):
    - OHLCV
    - Technical indicators (SMA, RSI, MACD, Bollinger)
    - Regime label
    - India VIX
    - Global signals (US futures, DXY)

Fallback (no model):
    - forecast_1d = 0.3 * momentum_5d + 0.7 * mean_reversion_signal
    - LONG if forecast > 0.3%, SHORT if forecast < -0.3%

Model path: models/tft_forecast.pkl

Usage:
    from signals.ml.tft_forecast import TFTForecastSignal

    tft = TFTForecastSignal()
    result = tft.evaluate(df, date)
"""

import logging
import os
import pickle
from datetime import date
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

SIGNAL_ID = 'TFT_FORECAST'

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

# Fallback parameters
MOMENTUM_WEIGHT = 0.3
MEAN_REVERSION_WEIGHT = 0.7
FORECAST_LONG_THRESHOLD = 0.003    # 0.3%
FORECAST_SHORT_THRESHOLD = -0.003  # -0.3%

# Mean reversion lookback
MR_LOOKBACK = 20  # 20-day mean for mean reversion
MR_SCALE = 0.5    # Scale factor for mean reversion signal

# Confidence scaling
BASE_CONFIDENCE = 0.50
FORECAST_CONFIDENCE_SCALE = 50.0   # |forecast| * scale -> confidence boost


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


# ================================================================
# SIGNAL CLASS
# ================================================================

class TFTForecastSignal:
    """
    Temporal Fusion Transformer multi-horizon forecast signal.
    Falls back to momentum + mean-reversion blend when no trained
    model is available.
    """

    SIGNAL_ID = SIGNAL_ID
    MODEL_PATH = os.path.join(MODEL_DIR, 'tft_forecast.pkl')

    def __init__(self) -> None:
        self._model = self._load_model()
        mode = 'ML' if self._model is not None else 'FALLBACK'
        logger.info('TFTForecastSignal initialised (mode=%s)', mode)

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_model(self) -> Any:
        """Try to load a trained TFT model from disk."""
        try:
            if os.path.exists(self.MODEL_PATH):
                with open(self.MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                logger.info('Loaded TFT forecast model from %s', self.MODEL_PATH)
                return model
        except ImportError:
            logger.warning('pytorch-forecasting/torch not installed — using fallback')
        except Exception as e:
            logger.warning('Failed to load TFT forecast model: %s', e)
        return None

    # ----------------------------------------------------------
    # ML prediction
    # ----------------------------------------------------------
    def _predict_ml(self, features: np.ndarray) -> Dict:
        """Predict using the trained TFT model."""
        try:
            pred = self._model.predict(features.reshape(1, -1))
            if hasattr(pred, 'shape') and pred.shape[-1] >= 3:
                forecast_1d = float(pred[0, 0])
                forecast_3d = float(pred[0, 1])
                forecast_5d = float(pred[0, 2])
            elif hasattr(pred, '__len__') and len(pred) >= 3:
                forecast_1d = float(pred[0])
                forecast_3d = float(pred[1])
                forecast_5d = float(pred[2])
            else:
                forecast_1d = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
                forecast_3d = forecast_1d * 1.5
                forecast_5d = forecast_1d * 2.0

            return {
                'forecast_1d': forecast_1d,
                'forecast_3d': forecast_3d,
                'forecast_5d': forecast_5d,
                'mode': 'ML',
            }
        except Exception as e:
            logger.warning('TFT ML prediction failed: %s', e)
            return {}

    # ----------------------------------------------------------
    # Fallback prediction
    # ----------------------------------------------------------
    def _predict_fallback(self, df: Any, dt: date) -> Optional[Dict]:
        """
        Simple linear extrapolation: blend of 5-day momentum and
        20-day mean-reversion signal.
        """
        try:
            if not hasattr(df, 'loc') or not hasattr(df, 'columns'):
                return None

            # Get data up to date
            if hasattr(df.index, 'date'):
                mask = df.index.date <= dt
            else:
                mask = df.index <= str(dt)
            sub = df.loc[mask]

            if sub.empty or len(sub) < MR_LOOKBACK + 1:
                return None

            # Close price series
            close_col = None
            for col in ['close', 'Close', 'last_price']:
                if col in sub.columns:
                    close_col = col
                    break
            if close_col is None:
                return None

            closes = sub[close_col].astype(float).values
            current = closes[-1]

            if current <= 0 or len(closes) < MR_LOOKBACK + 1:
                return None

            # 5-day momentum (as return fraction)
            prev_5d = closes[-6] if len(closes) >= 6 else closes[0]
            momentum_5d = (current / prev_5d - 1.0) if prev_5d > 0 else 0.0

            # Mean reversion signal: deviation from 20-day SMA
            sma_20 = np.mean(closes[-MR_LOOKBACK:])
            deviation = (current - sma_20) / sma_20 if sma_20 > 0 else 0.0
            mean_reversion = -deviation * MR_SCALE  # Negative = fade the deviation

            # Blended 1-day forecast
            forecast_1d = MOMENTUM_WEIGHT * momentum_5d + MEAN_REVERSION_WEIGHT * mean_reversion

            # Extrapolate multi-horizon (simple scaling)
            forecast_3d = forecast_1d * 2.0
            forecast_5d = forecast_1d * 3.0

            return {
                'forecast_1d': round(forecast_1d, 6),
                'forecast_3d': round(forecast_3d, 6),
                'forecast_5d': round(forecast_5d, 6),
                'momentum_5d': round(momentum_5d, 6),
                'mean_reversion': round(mean_reversion, 6),
                'sma_20': round(sma_20, 2),
                'mode': 'FALLBACK',
            }
        except Exception as e:
            logger.debug('TFT fallback prediction failed: %s', e)
            return None

    # ----------------------------------------------------------
    # Feature construction (for ML mode)
    # ----------------------------------------------------------
    def _build_features(self, df: Any, dt: date) -> Optional[np.ndarray]:
        """Build feature vector for ML model."""
        try:
            if not hasattr(df, 'loc'):
                return None

            if hasattr(df.index, 'date'):
                mask = df.index.date <= dt
            else:
                mask = df.index <= str(dt)
            sub = df.loc[mask]

            if sub.empty or len(sub) < 21:
                return None

            close_col = 'close' if 'close' in sub.columns else 'Close'
            closes = sub[close_col].astype(float).values

            # Returns
            ret_1d = closes[-1] / closes[-2] - 1 if len(closes) >= 2 else 0
            ret_5d = closes[-1] / closes[-6] - 1 if len(closes) >= 6 else 0
            ret_20d = closes[-1] / closes[-21] - 1 if len(closes) >= 21 else 0

            # SMA ratio
            sma_20 = np.mean(closes[-20:])
            sma_ratio = closes[-1] / sma_20 if sma_20 > 0 else 1.0

            # Volatility (20-day)
            daily_rets = np.diff(closes[-21:]) / closes[-21:-1]
            vol_20d = float(np.std(daily_rets)) if len(daily_rets) > 0 else 0.0

            # Volume
            vol_col = 'volume' if 'volume' in sub.columns else 'Volume' if 'Volume' in sub.columns else None
            vol_ratio = 1.0
            if vol_col and len(sub) >= 20:
                vol = sub[vol_col].astype(float).values
                avg_vol = np.mean(vol[-20:])
                vol_ratio = vol[-1] / avg_vol if avg_vol > 0 else 1.0

            # VIX
            vix = 18.0
            for col in ['india_vix', 'vix', 'VIX']:
                if col in sub.columns:
                    vix = _safe_float(sub[col].iloc[-1], 18.0)
                    break

            # RSI (14-period)
            rsi = 50.0
            for col in ['rsi', 'RSI', 'rsi_14']:
                if col in sub.columns:
                    rsi = _safe_float(sub[col].iloc[-1], 50.0)
                    break

            features = np.array([
                ret_1d, ret_5d, ret_20d,
                sma_ratio, vol_20d, vol_ratio,
                vix, rsi,
            ], dtype=np.float64)
            return features
        except Exception as e:
            logger.debug('TFT feature build failed: %s', e)
            return None

    # ----------------------------------------------------------
    # Main evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: Any, dt: date) -> Optional[Dict]:
        """
        Evaluate the TFT forecast signal.

        Parameters
        ----------
        df : DataFrame with OHLCV + indicator data.
        dt : Trade date.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata.
        None if no signal.
        """
        try:
            return self._evaluate_inner(df, dt)
        except Exception as e:
            logger.error('TFTForecastSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(self, df: Any, dt: date) -> Optional[Dict]:
        result = None

        # Try ML mode first
        if self._model is not None:
            features = self._build_features(df, dt)
            if features is not None:
                result = self._predict_ml(features)
                if not result:
                    result = None

        # Fallback
        if result is None:
            result = self._predict_fallback(df, dt)

        if result is None:
            return None

        forecast_1d = result.get('forecast_1d', 0.0)
        forecast_3d = result.get('forecast_3d', 0.0)
        forecast_5d = result.get('forecast_5d', 0.0)

        # Direction from 1d forecast
        if forecast_1d > FORECAST_LONG_THRESHOLD:
            direction = 'LONG'
        elif forecast_1d < FORECAST_SHORT_THRESHOLD:
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'

        # Confidence
        confidence = BASE_CONFIDENCE + min(abs(forecast_1d) * FORECAST_CONFIDENCE_SCALE, 0.40)
        confidence = max(0.10, min(0.90, confidence))

        # Extract current price
        price = 0.0
        try:
            if hasattr(df, 'loc'):
                close_col = 'close' if 'close' in df.columns else 'Close' if 'Close' in df.columns else None
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
            'TFT_FORECAST',
            f"Mode={result.get('mode', 'UNKNOWN')}",
            f"F1d={forecast_1d:+.4f}",
            f"F3d={forecast_3d:+.4f}",
            f"F5d={forecast_5d:+.4f}",
            f"Direction={direction}",
        ]

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(confidence, 3),
            'price': price,
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'mode': result.get('mode', 'UNKNOWN'),
                'forecast_returns': {
                    '1d': round(forecast_1d, 6),
                    '3d': round(forecast_3d, 6),
                    '5d': round(forecast_5d, 6),
                },
                'confidence': round(confidence, 3),
                'momentum_5d': result.get('momentum_5d'),
                'mean_reversion': result.get('mean_reversion'),
                'sma_20': result.get('sma_20'),
            },
        }

    def reset(self) -> None:
        """Reset internal state."""
        pass

    def __repr__(self) -> str:
        mode = 'ML' if self._model is not None else 'FALLBACK'
        return f"TFTForecastSignal(signal_id='{self.SIGNAL_ID}', mode='{mode}')"

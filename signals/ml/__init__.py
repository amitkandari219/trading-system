"""
ML Overlay Signals for NSE Nifty Trading System.

Six ML-based overlay signals that operate in two modes:
    - Trained mode: loads a saved model from models/ and predicts
    - Fallback mode: uses heuristic rules when no trained model exists

Signals:
    XGBoostMetaLearner    — Combines all signal outputs into a meta-prediction
    MambaRegimeDetector   — State-space model for regime detection
    TFTForecastSignal     — Multi-horizon return forecast (1d/3d/5d)
    RLPositionSizer       — RL-based position sizing overlay
    GNNSectorRotation     — Graph-based sector rotation detector
    NLPSentimentSignal    — News headline sentiment scoring
"""

from signals.ml.xgboost_meta import XGBoostMetaLearner
from signals.ml.mamba_regime import MambaRegimeDetector
from signals.ml.tft_forecast import TFTForecastSignal
from signals.ml.rl_position_sizer import RLPositionSizer
from signals.ml.gnn_sector_rotation import GNNSectorRotation
from signals.ml.nlp_sentiment import NLPSentimentSignal

__all__ = [
    'XGBoostMetaLearner',
    'MambaRegimeDetector',
    'TFTForecastSignal',
    'RLPositionSizer',
    'GNNSectorRotation',
    'NLPSentimentSignal',
]

"""
Reinforcement Learning Position Sizer Signal.

Learns optimal position sizing from reward = risk-adjusted returns.
When no trained model is available, falls back to rule-based adaptive
sizing using drawdown, VIX, and win-rate thresholds.

State (trained mode):
    - Current drawdown %
    - VIX regime
    - Recent win rate (last 20 trades)
    - Portfolio heat (total open risk)

Fallback (no model):
    - DD > 10%           -> 0.5x size
    - DD > 5%            -> 0.75x size
    - Win streak > 5     -> 0.85x size (overconfidence guard)
    - VIX > 25           -> 0.6x size
    - Default            -> 1.0x

Model path: models/rl_sizer.pkl

Usage:
    from signals.ml.rl_position_sizer import RLPositionSizer

    sizer = RLPositionSizer()
    result = sizer.evaluate(df, date)
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

SIGNAL_ID = 'RL_POSITION_SIZER'

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

# Size bounds
MIN_SIZE_MODIFIER = 0.3
MAX_SIZE_MODIFIER = 1.5
DEFAULT_SIZE_MODIFIER = 1.0

# Fallback thresholds
DD_SEVERE_PCT = 10.0        # Severe drawdown
DD_MODERATE_PCT = 5.0       # Moderate drawdown
DD_SEVERE_SIZE = 0.50       # Size at severe DD
DD_MODERATE_SIZE = 0.75     # Size at moderate DD
WIN_STREAK_LIMIT = 5        # Overconfidence guard trigger
WIN_STREAK_SIZE = 0.85      # Size when win streak too long
VIX_HIGH_THRESHOLD = 25.0   # High VIX threshold
VIX_HIGH_SIZE = 0.60        # Size at high VIX
PORTFOLIO_HEAT_MAX = 6.0    # Max portfolio heat % before reducing
PORTFOLIO_HEAT_SIZE = 0.70  # Size when portfolio too hot

# Win rate lookback
WIN_RATE_LOOKBACK = 20


# ================================================================
# HELPERS
# ================================================================

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


# ================================================================
# SIGNAL CLASS
# ================================================================

class RLPositionSizer:
    """
    RL-based position sizer.  Falls back to rule-based adaptive
    sizing when no trained model is available.
    """

    SIGNAL_ID = SIGNAL_ID
    MODEL_PATH = os.path.join(MODEL_DIR, 'rl_sizer.pkl')

    def __init__(self) -> None:
        self._model = self._load_model()
        mode = 'ML' if self._model is not None else 'FALLBACK'
        logger.info('RLPositionSizer initialised (mode=%s)', mode)

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_model(self) -> Any:
        """Try to load a trained RL sizer model from disk."""
        try:
            if os.path.exists(self.MODEL_PATH):
                with open(self.MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                logger.info('Loaded RL sizer model from %s', self.MODEL_PATH)
                return model
        except ImportError:
            logger.warning('stable-baselines3/torch not installed — using fallback')
        except Exception as e:
            logger.warning('Failed to load RL sizer model: %s', e)
        return None

    # ----------------------------------------------------------
    # ML prediction
    # ----------------------------------------------------------
    def _predict_ml(self, features: np.ndarray) -> Dict:
        """Predict position size using the trained RL model."""
        try:
            # RL models typically have a predict method returning action
            if hasattr(self._model, 'predict'):
                action, _ = self._model.predict(features, deterministic=True)
                # Action is the raw size modifier output
                size_modifier = float(action) if np.isscalar(action) else float(action[0])
            else:
                # Fallback: treat as regression model
                size_modifier = float(self._model.predict(features.reshape(1, -1))[0])

            # Clamp
            size_modifier = max(MIN_SIZE_MODIFIER, min(MAX_SIZE_MODIFIER, size_modifier))
            return {
                'size_modifier': size_modifier,
                'mode': 'ML',
            }
        except Exception as e:
            logger.warning('RL ML prediction failed: %s', e)
            return {}

    # ----------------------------------------------------------
    # Fallback prediction
    # ----------------------------------------------------------
    def _predict_fallback(self, df: Any, dt: date) -> Optional[Dict]:
        """
        Rule-based adaptive position sizing.

        Examines drawdown, VIX, win streak, and portfolio heat
        to determine position size modifier.
        """
        drawdown_pct = 0.0
        vix = 18.0
        win_streak = 0
        win_rate = 0.5
        portfolio_heat = 0.0

        try:
            if hasattr(df, 'loc') and hasattr(df, 'columns'):
                # Get data up to date
                if hasattr(df.index, 'date'):
                    mask = df.index.date <= dt
                else:
                    mask = df.index <= str(dt)
                sub = df.loc[mask]

                if sub.empty:
                    return None

                # Drawdown from equity curve or close
                close_col = None
                for col in ['equity', 'portfolio_value', 'close', 'Close']:
                    if col in sub.columns:
                        close_col = col
                        break

                if close_col:
                    values = sub[close_col].astype(float).values
                    if len(values) >= 2:
                        peak = np.maximum.accumulate(values)
                        current = values[-1]
                        drawdown_pct = ((peak[-1] - current) / peak[-1]) * 100 if peak[-1] > 0 else 0.0

                # VIX
                for col in ['india_vix', 'vix', 'VIX']:
                    if col in sub.columns:
                        vix = _safe_float(sub[col].iloc[-1], 18.0)
                        break

                # Win streak and win rate from trade_pnl or pnl column
                pnl_col = None
                for col in ['trade_pnl', 'pnl', 'daily_pnl', 'returns']:
                    if col in sub.columns:
                        pnl_col = col
                        break

                if pnl_col:
                    pnls = sub[pnl_col].astype(float).dropna().values
                    if len(pnls) > 0:
                        # Recent win rate
                        recent = pnls[-WIN_RATE_LOOKBACK:]
                        wins = np.sum(recent > 0)
                        win_rate = wins / len(recent) if len(recent) > 0 else 0.5

                        # Win streak (consecutive wins from the end)
                        win_streak = 0
                        for p in reversed(pnls):
                            if p > 0:
                                win_streak += 1
                            else:
                                break

                # Portfolio heat
                for col in ['portfolio_heat', 'total_risk', 'open_risk']:
                    if col in sub.columns:
                        portfolio_heat = _safe_float(sub[col].iloc[-1], 0.0)
                        break
        except Exception as e:
            logger.debug('Fallback data extraction failed: %s', e)

        # Apply rules (most restrictive wins)
        size_modifier = DEFAULT_SIZE_MODIFIER
        applied_rules = []

        if drawdown_pct > DD_SEVERE_PCT:
            size_modifier = min(size_modifier, DD_SEVERE_SIZE)
            applied_rules.append(f"DD={drawdown_pct:.1f}%>10%->0.5x")
        elif drawdown_pct > DD_MODERATE_PCT:
            size_modifier = min(size_modifier, DD_MODERATE_SIZE)
            applied_rules.append(f"DD={drawdown_pct:.1f}%>5%->0.75x")

        if vix > VIX_HIGH_THRESHOLD:
            size_modifier = min(size_modifier, VIX_HIGH_SIZE)
            applied_rules.append(f"VIX={vix:.1f}>25->0.6x")

        if win_streak > WIN_STREAK_LIMIT:
            size_modifier = min(size_modifier, WIN_STREAK_SIZE)
            applied_rules.append(f"WinStreak={win_streak}>5->0.85x")

        if portfolio_heat > PORTFOLIO_HEAT_MAX:
            size_modifier = min(size_modifier, PORTFOLIO_HEAT_SIZE)
            applied_rules.append(f"Heat={portfolio_heat:.1f}%>6%->0.7x")

        if not applied_rules:
            applied_rules.append('DEFAULT->1.0x')

        # Clamp
        size_modifier = max(MIN_SIZE_MODIFIER, min(MAX_SIZE_MODIFIER, size_modifier))

        return {
            'size_modifier': round(size_modifier, 2),
            'drawdown_pct': round(drawdown_pct, 2),
            'vix': round(vix, 2),
            'win_streak': win_streak,
            'win_rate': round(win_rate, 3),
            'portfolio_heat': round(portfolio_heat, 2),
            'applied_rules': applied_rules,
            'mode': 'FALLBACK',
        }

    # ----------------------------------------------------------
    # Feature construction (for ML mode)
    # ----------------------------------------------------------
    def _build_features(self, df: Any, dt: date) -> Optional[np.ndarray]:
        """Build state vector for RL model."""
        try:
            if not hasattr(df, 'loc'):
                return None

            if hasattr(df.index, 'date'):
                mask = df.index.date <= dt
            else:
                mask = df.index <= str(dt)
            sub = df.loc[mask]

            if sub.empty:
                return None

            # Drawdown
            close_col = None
            for col in ['equity', 'portfolio_value', 'close', 'Close']:
                if col in sub.columns:
                    close_col = col
                    break

            drawdown = 0.0
            if close_col:
                vals = sub[close_col].astype(float).values
                peak = np.maximum.accumulate(vals)
                drawdown = ((peak[-1] - vals[-1]) / peak[-1]) * 100 if peak[-1] > 0 else 0.0

            # VIX
            vix = 18.0
            for col in ['india_vix', 'vix', 'VIX']:
                if col in sub.columns:
                    vix = _safe_float(sub[col].iloc[-1], 18.0)
                    break

            # Win rate
            win_rate = 0.5
            pnl_col = None
            for col in ['trade_pnl', 'pnl', 'daily_pnl', 'returns']:
                if col in sub.columns:
                    pnl_col = col
                    break
            if pnl_col:
                pnls = sub[pnl_col].astype(float).dropna().values
                recent = pnls[-WIN_RATE_LOOKBACK:]
                win_rate = float(np.sum(recent > 0)) / len(recent) if len(recent) > 0 else 0.5

            # Portfolio heat
            heat = 0.0
            for col in ['portfolio_heat', 'total_risk', 'open_risk']:
                if col in sub.columns:
                    heat = _safe_float(sub[col].iloc[-1], 0.0)
                    break

            features = np.array([drawdown, vix, win_rate, heat], dtype=np.float64)
            return features
        except Exception as e:
            logger.debug('RL feature build failed: %s', e)
            return None

    # ----------------------------------------------------------
    # Main evaluate
    # ----------------------------------------------------------
    def evaluate(self, df: Any, dt: date) -> Optional[Dict]:
        """
        Evaluate the position sizer.

        Parameters
        ----------
        df : DataFrame with equity/PnL/VIX data.
        dt : Trade date.

        Returns
        -------
        dict with signal_id, direction, strength, price, reason, metadata.
        None if unable to compute sizing.
        """
        try:
            return self._evaluate_inner(df, dt)
        except Exception as e:
            logger.error('RLPositionSizer.evaluate error: %s', e, exc_info=True)
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

        size_modifier = result['size_modifier']

        # Direction is NEUTRAL — this is a sizing overlay, not directional
        direction = 'NEUTRAL'
        # Strength represents how much we're reducing (1.0 = full, 0.3 = minimal)
        strength = size_modifier

        # Extract current price
        price = 0.0
        try:
            if hasattr(df, 'loc'):
                close_col = None
                for col in ['close', 'Close', 'equity', 'portfolio_value']:
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
            'RL_POSITION_SIZER',
            f"Mode={result.get('mode', 'UNKNOWN')}",
            f"SizeModifier={size_modifier:.2f}",
        ]
        if 'applied_rules' in result:
            reason_parts.append(f"Rules={','.join(result['applied_rules'])}")

        return {
            'signal_id': self.SIGNAL_ID,
            'direction': direction,
            'strength': round(strength, 3),
            'price': price,
            'reason': ' | '.join(reason_parts),
            'metadata': {
                'mode': result.get('mode', 'UNKNOWN'),
                'size_modifier': size_modifier,
                'drawdown_pct': result.get('drawdown_pct'),
                'vix': result.get('vix'),
                'win_streak': result.get('win_streak'),
                'win_rate': result.get('win_rate'),
                'portfolio_heat': result.get('portfolio_heat'),
                'applied_rules': result.get('applied_rules', []),
            },
        }

    def reset(self) -> None:
        """Reset internal state."""
        pass

    def __repr__(self) -> str:
        mode = 'ML' if self._model is not None else 'FALLBACK'
        return f"RLPositionSizer(signal_id='{self.SIGNAL_ID}', mode='{mode}')"

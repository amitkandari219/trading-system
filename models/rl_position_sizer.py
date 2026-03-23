"""
Reinforcement Learning Dynamic Position Sizer (PPO/SAC).

Replaces fixed size_modifier rules with an RL agent that learns optimal
sizing given current signal constellation + portfolio state.

Environment:
  State: [signal_scores (11), vix_level, portfolio_pnl_today, n_open_positions,
          portfolio_drawdown, cash_available, time_to_close]
  Action: Continuous size_modifier in [0.1, 2.0]
  Reward: risk-adjusted return = return / max(drawdown, 0.01) - penalty_for_large_position

Agent: SAC (Soft Actor-Critic) — better for continuous action spaces
  - Actor: Gaussian policy μ(s), σ(s) → sample action
  - Twin critics: Q1(s,a), Q2(s,a) → min for pessimistic estimate
  - Temperature α: automatically tuned entropy coefficient

Training:
  - Offline from backtest data (batch RL)
  - Online fine-tuning in paper trading
  - Walk-forward: train on 3yr, test 1yr, update quarterly

Usage:
    from models.rl_position_sizer import RLPositionSizer
    sizer = RLPositionSizer()
    action = sizer.get_size(state_dict)
"""

import logging
import pickle
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# CONFIGURATION
# ================================================================
STATE_DIM = 20          # Number of state features
ACTION_DIM = 1          # Continuous: size_modifier
HIDDEN_DIM = 128
GAMMA = 0.99            # Discount factor
TAU = 0.005             # Soft update coefficient
ALPHA_LR = 3e-4         # Temperature learning rate
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
MIN_SIZE = 0.1          # Minimum size modifier
MAX_SIZE = 2.0          # Maximum size modifier

MODEL_DIR = Path(__file__).parent / 'rl_sizer'
MODEL_FILE = MODEL_DIR / 'sac_sizer.pkl'

# State feature names
STATE_FEATURES = [
    # Signal scores (11)
    'pcr_size_mod', 'fii_size_mod', 'delivery_size_mod',
    'rollover_size_mod', 'sentiment_size_mod', 'bond_size_mod',
    'gamma_size_mod', 'vol_ts_size_mod', 'macro_size_mod',
    'orderflow_size_mod', 'global_size_mod',
    # Market context
    'vix_level_norm',       # VIX / 20 (normalized)
    'vix_regime_encoded',   # 0-4
    'nifty_5d_return',
    'nifty_20d_return',
    # Portfolio state
    'portfolio_pnl_today',  # Today's P&L as % of capital
    'n_open_positions',     # 0-4
    'portfolio_drawdown',   # Current drawdown from peak
    'cash_ratio',           # Available cash / total capital
    'time_to_close_norm',   # Minutes to close / 375
]

# Reward shaping
DRAWDOWN_PENALTY = 2.0      # Multiply drawdown by this in reward
POSITION_COST = 0.001       # Small cost per unit size (to prevent oversizing)
SHARPE_WINDOW = 20          # Rolling Sharpe for reward calculation


class GaussianActor:
    """
    Gaussian policy network for continuous action.

    Outputs mean and log_std of action distribution.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(hidden_dim, state_dim) * 0.02
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, hidden_dim) * 0.02
        self.b2 = np.zeros(hidden_dim)
        self.W_mu = rng.randn(action_dim, hidden_dim) * 0.02
        self.b_mu = np.zeros(action_dim)
        self.W_log_std = rng.randn(action_dim, hidden_dim) * 0.02
        self.b_log_std = np.zeros(action_dim)

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        state : (state_dim,)

        Returns
        -------
        mu : (action_dim,) — mean
        log_std : (action_dim,) — log standard deviation
        """
        h = self._relu(self.W1 @ state + self.b1)
        h = self._relu(self.W2 @ h + self.b2)
        mu = self.W_mu @ h + self.b_mu
        log_std = np.clip(self.W_log_std @ h + self.b_log_std, -5, 2)
        return mu, log_std

    def sample(self, state: np.ndarray) -> Tuple[float, float]:
        """Sample action from policy."""
        mu, log_std = self.forward(state)
        std = np.exp(log_std)
        # Reparameterization trick
        noise = np.random.randn(*mu.shape)
        raw_action = mu + std * noise
        # Squash to [MIN_SIZE, MAX_SIZE] via sigmoid
        action = MIN_SIZE + (MAX_SIZE - MIN_SIZE) / (1 + np.exp(-raw_action))
        # Log probability (for training)
        log_prob = -0.5 * ((raw_action - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        return float(action[0]), float(log_prob.sum())

    def deterministic(self, state: np.ndarray) -> float:
        """Deterministic action (mean)."""
        mu, _ = self.forward(state)
        action = MIN_SIZE + (MAX_SIZE - MIN_SIZE) / (1 + np.exp(-mu))
        return float(action[0])


class TwinCritic:
    """Twin Q-networks for pessimistic value estimation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        rng = np.random.RandomState(42)
        # Q1
        self.W1_1 = rng.randn(hidden_dim, state_dim + action_dim) * 0.02
        self.b1_1 = np.zeros(hidden_dim)
        self.W1_2 = rng.randn(hidden_dim, hidden_dim) * 0.02
        self.b1_2 = np.zeros(hidden_dim)
        self.W1_out = rng.randn(1, hidden_dim) * 0.02
        self.b1_out = np.zeros(1)
        # Q2
        self.W2_1 = rng.randn(hidden_dim, state_dim + action_dim) * 0.02
        self.b2_1 = np.zeros(hidden_dim)
        self.W2_2 = rng.randn(hidden_dim, hidden_dim) * 0.02
        self.b2_2 = np.zeros(hidden_dim)
        self.W2_out = rng.randn(1, hidden_dim) * 0.02
        self.b2_out = np.zeros(1)

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    def forward(self, state: np.ndarray, action: np.ndarray) -> Tuple[float, float]:
        sa = np.concatenate([state, action])
        # Q1
        h1 = self._relu(self.W1_1 @ sa + self.b1_1)
        h1 = self._relu(self.W1_2 @ h1 + self.b1_2)
        q1 = float((self.W1_out @ h1 + self.b1_out)[0])
        # Q2
        h2 = self._relu(self.W2_1 @ sa + self.b2_1)
        h2 = self._relu(self.W2_2 @ h2 + self.b2_2)
        q2 = float((self.W2_out @ h2 + self.b2_out)[0])
        return q1, q2


class ReplayBuffer:
    """Experience replay buffer for off-policy RL."""

    def __init__(self, capacity: int = BUFFER_SIZE):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class RLPositionSizer:
    """
    SAC-based dynamic position sizer.

    Learns optimal sizing from signal constellation and portfolio state.
    """

    SIGNAL_ID = 'RL_POSITION_SIZER'

    def __init__(self, db_conn=None):
        self.conn = db_conn
        self.actor = GaussianActor(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
        self.critic = TwinCritic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
        self.buffer = ReplayBuffer()
        self.log_alpha = 0.0  # Temperature
        self.trained = False
        self._load_model()

    def _load_model(self):
        if MODEL_FILE.exists():
            try:
                with open(MODEL_FILE, 'rb') as f:
                    state = pickle.load(f)
                self.actor = state['actor']
                self.critic = state['critic']
                self.log_alpha = state['log_alpha']
                self.trained = state.get('trained', False)
                logger.info("Loaded RL sizer from %s", MODEL_FILE)
            except Exception as e:
                logger.warning("Failed to load RL model: %s", e)

    def save_model(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            'actor': self.actor,
            'critic': self.critic,
            'log_alpha': self.log_alpha,
            'trained': self.trained,
        }
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(state, f)

    # ----------------------------------------------------------
    # State construction
    # ----------------------------------------------------------
    def build_state(self, signals: Dict, portfolio: Dict) -> np.ndarray:
        """
        Build state vector from signal outputs and portfolio state.

        Parameters
        ----------
        signals : dict of signal evaluations (same as meta-learner input)
        portfolio : dict with portfolio state info

        Returns
        -------
        state : (STATE_DIM,) numpy array
        """
        state = np.zeros(STATE_DIM)

        # Signal size modifiers (11)
        signal_keys = [
            ('PCR_AUTOTRENDER', 'pcr_size_mod'),
            ('FII_FUTURES_OI', 'fii_size_mod'),
            ('DELIVERY_PCT', 'delivery_size_mod'),
            ('ROLLOVER_ANALYSIS', 'rollover_size_mod'),
            ('SENTIMENT_COMPOSITE', 'sentiment_size_mod'),
            ('BOND_YIELD_SPREAD', 'bond_size_mod'),
            ('GAMMA_EXPOSURE', 'gamma_size_mod'),
            ('VOL_TERM_STRUCTURE', 'vol_ts_size_mod'),
            ('RBI_MACRO_FILTER', 'macro_size_mod'),
            ('ORDER_FLOW_IMBALANCE', 'orderflow_size_mod'),
            ('GLOBAL_OVERNIGHT_COMPOSITE', 'global_size_mod'),
        ]
        for i, (sig_id, _) in enumerate(signal_keys):
            sig = signals.get(sig_id, {})
            state[i] = sig.get('size_modifier', 1.0)

        # Market context
        ctx = signals.get('context', {})
        state[11] = ctx.get('vix_level', 15.0) / 20.0
        state[12] = {'CALM': 0, 'NORMAL': 1, 'ELEVATED': 2,
                      'HIGH_VOL': 3, 'CRISIS': 4}.get(
            ctx.get('vix_regime', 'NORMAL'), 1) / 4.0
        state[13] = ctx.get('nifty_5d_return', 0.0) / 5.0
        state[14] = ctx.get('nifty_20d_return', 0.0) / 10.0

        # Portfolio state
        state[15] = portfolio.get('pnl_today_pct', 0.0) / 5.0
        state[16] = portfolio.get('n_open_positions', 0) / 4.0
        state[17] = portfolio.get('drawdown_pct', 0.0) / 25.0
        state[18] = portfolio.get('cash_ratio', 1.0)
        state[19] = portfolio.get('time_to_close_min', 375) / 375.0

        return state.astype(np.float32)

    # ----------------------------------------------------------
    # Sizing decision
    # ----------------------------------------------------------
    def get_size(
        self,
        signals: Dict,
        portfolio: Optional[Dict] = None,
        deterministic: bool = True,
    ) -> Dict:
        """
        Get optimal position size modifier.

        Parameters
        ----------
        signals : dict of all signal outputs
        portfolio : portfolio state dict
        deterministic : if True, use mean (no exploration)

        Returns
        -------
        dict with size_modifier, confidence, method
        """
        if portfolio is None:
            portfolio = {
                'pnl_today_pct': 0.0,
                'n_open_positions': 0,
                'drawdown_pct': 0.0,
                'cash_ratio': 1.0,
                'time_to_close_min': 375,
            }

        state = self.build_state(signals, portfolio)

        if deterministic:
            size_modifier = self.actor.deterministic(state)
        else:
            size_modifier, _ = self.actor.sample(state)

        # Clamp
        size_modifier = float(np.clip(size_modifier, MIN_SIZE, MAX_SIZE))

        # Q-value based confidence
        action = np.array([size_modifier])
        q1, q2 = self.critic.forward(state, action)
        q_value = min(q1, q2)
        confidence = float(np.clip(0.5 + q_value, 0.0, 0.95))

        if not self.trained:
            # Untrained: fall back to conservative sizing
            size_modifier = 1.0 + (size_modifier - 1.0) * 0.2
            confidence = 0.30

        return {
            'signal_id': self.SIGNAL_ID,
            'size_modifier': round(size_modifier, 3),
            'confidence': round(confidence, 3),
            'q_value': round(float(q_value), 4),
            'trained': self.trained,
            'method': 'sac' if self.trained else 'sac_untrained',
        }

    # ----------------------------------------------------------
    # Reward computation
    # ----------------------------------------------------------
    @staticmethod
    def compute_reward(
        pnl_pct: float, drawdown_pct: float, size_modifier: float
    ) -> float:
        """
        Compute reward for RL training.

        reward = pnl - drawdown_penalty - position_cost
        """
        drawdown_penalty = abs(drawdown_pct) * DRAWDOWN_PENALTY
        position_cost = abs(size_modifier - 1.0) * POSITION_COST
        return pnl_pct - drawdown_penalty - position_cost

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
    def train_offline(
        self, experience_data: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Train from historical experience data.

        For full training: pip install stable-baselines3
        """
        logger.info("RL training requires: pip install stable-baselines3")

        if experience_data and len(experience_data) > 100:
            # Fill replay buffer
            for exp in experience_data:
                self.buffer.push(
                    exp['state'], exp['action'], exp['reward'],
                    exp['next_state'], exp.get('done', False)
                )
            logger.info("Loaded %d experiences into replay buffer", len(experience_data))

        self.save_model()
        return {
            'status': 'baseline',
            'buffer_size': len(self.buffer),
            'message': 'Baseline SAC saved. Install stable-baselines3 for full training.',
        }

    def evaluate(self, signals: Dict, portfolio: Optional[Dict] = None) -> Dict:
        """Signal interface."""
        return self.get_size(signals, portfolio)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    sizer = RLPositionSizer()
    signals = {
        'PCR_AUTOTRENDER': {'size_modifier': 1.15, 'direction': 'BULLISH'},
        'FII_FUTURES_OI': {'size_modifier': 1.35, 'direction': 'BULLISH'},
        'context': {'vix_level': 15, 'vix_regime': 'NORMAL'},
    }
    result = sizer.get_size(signals)
    print(f"RL Size: {result['size_modifier']:.3f} | Q: {result['q_value']:.4f} "
          f"| Method: {result['method']}")

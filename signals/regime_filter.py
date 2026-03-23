"""
Intraday Regime Filter — adapts signal execution based on real-time volatility.

Problem: KAUFMAN_BB_MR and GUJRAL_RANGE are mean-reversion strategies.
They thrive in RANGING conditions but get destroyed in:
  - CRISIS (VIX>25): trends overwhelm mean-reversion
  - HIGH_VOL (VIX>18): stop losses hit more frequently
  - TRENDING (ADX>30): price doesn't revert to mean

Solution: detect regime from 5-min bar features and adjust behavior:

┌────────────┬───────────────────┬──────────────┬──────────────────────┐
│ Regime     │ Detection         │ Size Factor  │ Signal Policy        │
├────────────┼───────────────────┼──────────────┼──────────────────────┤
│ CALM       │ ATR_ratio < 0.8   │ 1.00         │ All signals active   │
│ NORMAL     │ 0.8 ≤ ATR_r < 1.2 │ 1.00         │ All signals active   │
│ ELEVATED   │ 1.2 ≤ ATR_r < 2.0 │ 0.60         │ Widen SL, skip RANGE │
│ HIGH_VOL   │ 2.0 ≤ ATR_r < 3.0 │ 0.30         │ BB_MR only, wider SL │
│ CRISIS     │ ATR_ratio ≥ 3.0   │ 0.00         │ NO TRADING           │
└────────────┴───────────────────┴──────────────┴──────────────────────┘

ATR_ratio = today's ATR / 20-day average ATR
  (measures "how volatile is today vs. normal?")

Additional filters:
  - Gap filter: skip first 30min if |overnight_gap| > 1%
  - Trend filter: if ADX > 30 on 5-min bars, disable mean-reversion
  - Squeeze filter: if BB_width < 50th percentile, boost BB_MR size 1.3x
  - Streak filter: after 3 consecutive losses, reduce size 50% for next 2 trades

Usage:
    from signals.regime_filter import IntradayRegimeFilter
    rf = IntradayRegimeFilter()
    decision = rf.evaluate(bar, session_bars, signal_name)
    # decision = {'allow': True, 'size_factor': 0.60, 'regime': 'ELEVATED', ...}
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# REGIME THRESHOLDS
# ═══════════════════════════════════════════════════════════

REGIMES = {
    'CALM':     {'atr_ratio_max': 0.8,  'size_factor': 1.00, 'sl_mult': 0.8},
    'NORMAL':   {'atr_ratio_max': 1.2,  'size_factor': 1.00, 'sl_mult': 1.0},
    'ELEVATED': {'atr_ratio_max': 2.0,  'size_factor': 0.60, 'sl_mult': 1.5},
    'HIGH_VOL': {'atr_ratio_max': 3.0,  'size_factor': 0.30, 'sl_mult': 2.0},
    'CRISIS':   {'atr_ratio_max': 999,  'size_factor': 0.00, 'sl_mult': 0.0},
}

# Signal-regime compatibility matrix
# True = signal allowed in this regime
SIGNAL_REGIME_MATRIX = {
    'KAUFMAN_BB_MR': {
        'CALM': True, 'NORMAL': True, 'ELEVATED': True,
        'HIGH_VOL': True, 'CRISIS': False,
    },
    'GUJRAL_RANGE': {
        'CALM': True, 'NORMAL': True, 'ELEVATED': False,
        'HIGH_VOL': False, 'CRISIS': False,
    },
    'GAMMA_BREAKOUT': {
        'CALM': False, 'NORMAL': True, 'ELEVATED': True,
        'HIGH_VOL': True, 'CRISIS': False,
    },
    'GAMMA_REVERSAL': {
        'CALM': True, 'NORMAL': True, 'ELEVATED': True,
        'HIGH_VOL': False, 'CRISIS': False,
    },
    'GAMMA_SQUEEZE': {
        'CALM': True, 'NORMAL': True, 'ELEVATED': False,
        'HIGH_VOL': False, 'CRISIS': False,
    },

    # ── BankNifty signals ──
    # BN is ~2.1x more volatile than Nifty, so stricter regime gates.
    'BN_KAUFMAN_BB_MR': {
        # Mean-reversion — works through HIGH_VOL (like Nifty KAUFMAN)
        'CALM': True, 'NORMAL': True, 'ELEVATED': True,
        'HIGH_VOL': True, 'CRISIS': False,
    },
    'BN_GUJRAL_RANGE': {
        # Range MR — blocked in ELEVATED+ (BN range expands violently)
        'CALM': True, 'NORMAL': True, 'ELEVATED': False,
        'HIGH_VOL': False, 'CRISIS': False,
    },
    'BN_ORB_BREAKOUT': {
        # Breakout — allowed through HIGH_VOL
        'CALM': True, 'NORMAL': True, 'ELEVATED': True,
        'HIGH_VOL': True, 'CRISIS': False,
    },
    'BN_TREND_BAR': {
        # Trend continuation — allowed through HIGH_VOL
        'CALM': True, 'NORMAL': True, 'ELEVATED': True,
        'HIGH_VOL': True, 'CRISIS': False,
    },
    'BN_FAILED_BREAKOUT': {
        # Failed breakout reversal — allowed through HIGH_VOL
        'CALM': True, 'NORMAL': True, 'ELEVATED': True,
        'HIGH_VOL': True, 'CRISIS': False,
    },
    'BN_VWAP_RECLAIM': {
        # VWAP reclaim — blocked in HIGH_VOL+ (too noisy)
        'CALM': True, 'NORMAL': True, 'ELEVATED': True,
        'HIGH_VOL': False, 'CRISIS': False,
    },
    'BN_VWAP_REJECTION': {
        # VWAP rejection — blocked in HIGH_VOL+
        'CALM': True, 'NORMAL': True, 'ELEVATED': True,
        'HIGH_VOL': False, 'CRISIS': False,
    },
    'BN_FIRST_PULLBACK': {
        # Pullback — blocked in HIGH_VOL+ (pullbacks overshoot)
        'CALM': True, 'NORMAL': True, 'ELEVATED': True,
        'HIGH_VOL': False, 'CRISIS': False,
    },
    'BN_GAP_FILL': {
        # Gap fill — blocked in HIGH_VOL+ (gaps don't fill in crashes)
        'CALM': True, 'NORMAL': True, 'ELEVATED': True,
        'HIGH_VOL': False, 'CRISIS': False,
    },
    'BN_EOD_TREND': {
        # EOD trend — blocked in HIGH_VOL+ (EOD moves unreliable)
        'CALM': True, 'NORMAL': True, 'ELEVATED': True,
        'HIGH_VOL': False, 'CRISIS': False,
    },
}

# Default: allow in CALM and NORMAL, block in HIGH_VOL and CRISIS
DEFAULT_REGIME_POLICY = {
    'CALM': True, 'NORMAL': True, 'ELEVATED': True,
    'HIGH_VOL': False, 'CRISIS': False,
}

# Streak filter
MAX_CONSECUTIVE_LOSSES = 3
STREAK_SIZE_FACTOR = 0.50
STREAK_RECOVERY_TRADES = 2


class IntradayRegimeFilter:
    """
    Real-time regime detection and signal filtering for intraday trading.
    """

    def __init__(self):
        # Streak tracking
        self._consecutive_losses = 0
        self._streak_cooldown = 0

        # Day-level regime cache
        self._cached_regime = None
        self._cached_date = None

        # ATR baseline (rolling 20-day average of daily ATR)
        self._atr_history = []
        self._atr_baseline = None

    def evaluate(self, bar: pd.Series, session_bars: pd.DataFrame,
                 signal_name: str, daily_atr: Optional[float] = None) -> Dict:
        """
        Evaluate whether a signal should fire given current regime.

        Args:
            bar: current 5-min bar with indicators
            session_bars: all bars today so far
            signal_name: e.g. 'KAUFMAN_BB_MR', 'GUJRAL_RANGE'
            daily_atr: today's daily-timeframe ATR (if available)

        Returns:
            dict with:
                allow: bool — should this signal fire?
                size_factor: float — multiply lot count by this
                regime: str — detected regime name
                sl_multiplier: float — multiply stop loss by this
                reason: str — explanation
        """
        # Detect regime
        regime = self._detect_regime(bar, session_bars, daily_atr)

        # Get regime config
        regime_config = REGIMES.get(regime, REGIMES['NORMAL'])
        size_factor = regime_config['size_factor']
        sl_mult = regime_config['sl_mult']

        # Check signal-regime compatibility
        signal_policy = SIGNAL_REGIME_MATRIX.get(signal_name, DEFAULT_REGIME_POLICY)
        signal_allowed = signal_policy.get(regime, True)

        if not signal_allowed:
            return {
                'allow': False,
                'size_factor': 0,
                'regime': regime,
                'sl_multiplier': sl_mult,
                'reason': f'{signal_name} blocked in {regime} regime',
            }

        if size_factor <= 0:
            return {
                'allow': False,
                'size_factor': 0,
                'regime': regime,
                'sl_multiplier': 0,
                'reason': f'Trading halted in {regime} regime',
            }

        # ── Additional filters ──

        # Gap filter: skip first 30 min if gap > 1%
        gap_pct = float(bar.get('overnight_gap_pct', 0)) if pd.notna(bar.get('overnight_gap_pct')) else 0
        bar_time = bar['datetime']
        if abs(gap_pct) > 0.01 and bar_time.hour == 9 and bar_time.minute < 45:
            return {
                'allow': False,
                'size_factor': 0,
                'regime': regime,
                'sl_multiplier': sl_mult,
                'reason': f'Gap filter: |{gap_pct:.1%}| gap, waiting for 9:45',
            }

        # Trend filter: ADX > 30 blocks mean-reversion signals
        adx_val = float(bar.get('adx_14', 0)) if pd.notna(bar.get('adx_14')) else 0
        if adx_val > 30 and signal_name in ('KAUFMAN_BB_MR', 'GUJRAL_RANGE'):
            size_factor *= 0.50  # don't fully block, but halve size

        # Squeeze boost: tight BBands = good for BB_MR
        if signal_name == 'KAUFMAN_BB_MR':
            bb_upper = float(bar.get('bb_upper', 0)) if pd.notna(bar.get('bb_upper')) else 0
            bb_lower = float(bar.get('bb_lower', 0)) if pd.notna(bar.get('bb_lower')) else 0
            close = float(bar['close'])
            if close > 0 and bb_upper > bb_lower:
                bb_width_pct = (bb_upper - bb_lower) / close
                if bb_width_pct < 0.015:  # tight bands = strong MR setup
                    size_factor *= 1.30

        # Streak filter
        if self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            if self._streak_cooldown > 0:
                size_factor *= STREAK_SIZE_FACTOR
                self._streak_cooldown -= 1
            else:
                self._streak_cooldown = STREAK_RECOVERY_TRADES

        return {
            'allow': True,
            'size_factor': round(size_factor, 2),
            'regime': regime,
            'sl_multiplier': round(sl_mult, 2),
            'reason': f'{signal_name} allowed in {regime} (size={size_factor:.0%})',
        }

    def record_trade_result(self, won: bool):
        """Update streak tracking after a trade completes."""
        if won:
            self._consecutive_losses = 0
            self._streak_cooldown = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
                self._streak_cooldown = STREAK_RECOVERY_TRADES

    def update_daily_atr(self, atr_val: float):
        """Feed daily ATR for baseline computation."""
        self._atr_history.append(atr_val)
        if len(self._atr_history) > 20:
            self._atr_history = self._atr_history[-20:]
        self._atr_baseline = np.mean(self._atr_history)

    def _detect_regime(self, bar: pd.Series, session_bars: pd.DataFrame,
                       daily_atr: Optional[float] = None) -> str:
        """
        Detect current regime from 5-min bar features.

        Primary metric: ATR ratio = current ATR / baseline ATR
        Secondary: session range expansion rate, bar body sizes
        """
        # Method 1: ATR ratio (daily timeframe)
        atr_14 = float(bar.get('atr_14', 0)) if pd.notna(bar.get('atr_14')) else 0

        if daily_atr and self._atr_baseline and self._atr_baseline > 0:
            atr_ratio = daily_atr / self._atr_baseline
        elif atr_14 > 0 and len(session_bars) > 20:
            # Fallback: compare current intraday ATR to rolling average
            recent_atr = session_bars['atr_14'].dropna().tail(20)
            if len(recent_atr) > 5:
                baseline = recent_atr.mean()
                atr_ratio = atr_14 / baseline if baseline > 0 else 1.0
            else:
                atr_ratio = 1.0
        else:
            atr_ratio = 1.0

        # Method 2: Session range expansion (backup)
        if len(session_bars) >= 10:
            sess_range = float(session_bars['high'].max() - session_bars['low'].min())
            close = float(bar['close'])
            if close > 0:
                sess_range_pct = sess_range / close
                # Normal Nifty day: ~1-1.5% range. Crisis: >3%
                if sess_range_pct > 0.03:
                    atr_ratio = max(atr_ratio, 3.0)
                elif sess_range_pct > 0.02:
                    atr_ratio = max(atr_ratio, 2.0)

        # Method 3: Large bar detection (spikes)
        if len(session_bars) >= 3:
            last_3_ranges = (session_bars.tail(3)['high'] - session_bars.tail(3)['low'])
            avg_range = last_3_ranges.mean()
            if atr_14 > 0 and avg_range > atr_14 * 2.5:
                atr_ratio = max(atr_ratio, 2.5)

        # Classify
        for regime_name in ['CALM', 'NORMAL', 'ELEVATED', 'HIGH_VOL', 'CRISIS']:
            if atr_ratio < REGIMES[regime_name]['atr_ratio_max']:
                return regime_name

        return 'CRISIS'

    def update_live_vix(self, vix: float):
        """
        Store real-time VIX value from VIXStreamer.

        This is used by DynamicRegimeManager to keep the regime filter
        aware of the current VIX level for signal-specific adjustments.

        Args:
            vix: current India VIX value
        """
        self.live_vix = vix

    def get_adjusted_size_factor(self, signal_name: str, regime: str) -> float:
        """
        Get position size factor adjusted for signal-specific characteristics.

        Base factor comes from the regime, then adjusted per signal:
        - KAUFMAN_BB_MR: +20% in CALM (thrives in low vol), -20% in ELEVATED
        - GUJRAL_RANGE: +10% in CALM, blocked above ELEVATED (handled elsewhere)
        - GAMMA_BREAKOUT: +30% in HIGH_VOL (breakouts work in high vol)
        - ORB_TUNED: -30% in ELEVATED (ORB less reliable in vol)

        Args:
            signal_name: signal identifier
            regime: current regime label

        Returns:
            float: adjusted size factor (clamped to [0.0, 1.50])
        """
        base_factor = REGIMES.get(regime, REGIMES['NORMAL'])['size_factor']

        # Signal-specific adjustments (multiplicative)
        adjustments = {
            'KAUFMAN_BB_MR': {
                'CALM': 1.20, 'NORMAL': 1.00, 'ELEVATED': 0.80,
                'HIGH_VOL': 1.00, 'CRISIS': 0.00,
            },
            'GUJRAL_RANGE': {
                'CALM': 1.10, 'NORMAL': 1.00, 'ELEVATED': 0.70,
                'HIGH_VOL': 0.00, 'CRISIS': 0.00,
            },
            'GAMMA_BREAKOUT': {
                'CALM': 0.80, 'NORMAL': 1.00, 'ELEVATED': 1.10,
                'HIGH_VOL': 1.30, 'CRISIS': 0.00,
            },
            'GAMMA_REVERSAL': {
                'CALM': 1.00, 'NORMAL': 1.00, 'ELEVATED': 0.90,
                'HIGH_VOL': 0.50, 'CRISIS': 0.00,
            },
            'ORB_TUNED': {
                'CALM': 1.10, 'NORMAL': 1.00, 'ELEVATED': 0.70,
                'HIGH_VOL': 0.50, 'CRISIS': 0.00,
            },
        }

        signal_adj = adjustments.get(signal_name, {})
        multiplier = signal_adj.get(regime, 1.00)

        adjusted = base_factor * multiplier
        return round(max(0.0, min(adjusted, 1.50)), 2)

    def get_status(self) -> Dict:
        """Return current filter state."""
        return {
            'regime': self._cached_regime,
            'consecutive_losses': self._consecutive_losses,
            'streak_cooldown': self._streak_cooldown,
            'atr_baseline': round(self._atr_baseline, 2) if self._atr_baseline else None,
            'atr_history_len': len(self._atr_history),
            'live_vix': getattr(self, 'live_vix', None),
        }


# ═══════════════════════════════════════════════════════════
# VECTORIZED VERSION for backtest (performance)
# ═══════════════════════════════════════════════════════════

def compute_regime_column(df5: pd.DataFrame) -> pd.Series:
    """
    Compute regime for every bar in a vectorized way (for backtesting).

    Uses:
      - Session range as % of close
      - ATR_14 vs rolling 20-bar ATR
      - 3-bar average range vs ATR

    Returns:
        pd.Series with regime labels: CALM, NORMAL, ELEVATED, HIGH_VOL, CRISIS
    """
    df = df5.copy()

    # ATR ratio: current ATR / rolling 100-bar ATR (longer window for stability)
    atr = df['atr_14'].fillna(method='ffill')
    atr_baseline = atr.rolling(100, min_periods=20).mean()
    atr_ratio = atr / atr_baseline.replace(0, np.nan)
    atr_ratio = atr_ratio.fillna(1.0)

    # Session range boost
    if 'sess_range' not in df.columns:
        date_col = df['datetime'].dt.date
        df['_sh'] = df.groupby(date_col)['high'].cummax()
        df['_sl'] = df.groupby(date_col)['low'].cummin()
        sess_range = df['_sh'] - df['_sl']
        df.drop(columns=['_sh', '_sl'], inplace=True)
    else:
        sess_range = df['sess_range']

    sess_range_pct = sess_range / df['close'].replace(0, np.nan)

    # Boost ATR ratio if session range is extreme
    atr_ratio = np.where(sess_range_pct > 0.03, np.maximum(atr_ratio, 3.0), atr_ratio)
    atr_ratio = np.where((sess_range_pct > 0.02) & (sess_range_pct <= 0.03),
                         np.maximum(atr_ratio, 2.0), atr_ratio)

    # Classify
    regime = pd.Series('NORMAL', index=df.index)
    regime = np.where(atr_ratio < 0.8, 'CALM', regime)
    regime = np.where((atr_ratio >= 0.8) & (atr_ratio < 1.2), 'NORMAL', regime)
    regime = np.where((atr_ratio >= 1.2) & (atr_ratio < 2.0), 'ELEVATED', regime)
    regime = np.where((atr_ratio >= 2.0) & (atr_ratio < 3.0), 'HIGH_VOL', regime)
    regime = np.where(atr_ratio >= 3.0, 'CRISIS', regime)

    return pd.Series(regime, index=df5.index, name='regime')


def compute_size_factor_column(regime_col: pd.Series) -> pd.Series:
    """Convert regime labels to size factors."""
    mapping = {r: REGIMES[r]['size_factor'] for r in REGIMES}
    return regime_col.map(mapping).fillna(1.0)


def compute_sl_multiplier_column(regime_col: pd.Series) -> pd.Series:
    """Convert regime labels to stop-loss multipliers."""
    mapping = {r: REGIMES[r]['sl_mult'] for r in REGIMES}
    return regime_col.map(mapping).fillna(1.0)


def signal_allowed_in_regime(signal_name: str, regime_col: pd.Series) -> pd.Series:
    """Check if signal is allowed in each bar's regime."""
    policy = SIGNAL_REGIME_MATRIX.get(signal_name, DEFAULT_REGIME_POLICY)
    return regime_col.map(policy).fillna(True)

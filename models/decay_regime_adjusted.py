"""
Regime-adjusted decay detection.

Key insight: signal appearing to decay in TRENDING market
is NOT decaying if it's a mean-reversion signal — it's just
in the wrong regime. Only flag decay when signal underperforms
in its TARGET regime.
"""

import numpy as np
from collections import defaultdict


# Target regimes per signal (where they should perform best)
SIGNAL_TARGET_REGIME = {
    'KAUFMAN_DRY_20': 'TRENDING',
    'KAUFMAN_DRY_16': 'TRENDING',
    'KAUFMAN_DRY_12': 'TRENDING',
    'GUJRAL_DRY_8':   'TRENDING',
    'GUJRAL_DRY_13':  'TRENDING',
    'CHAN_AT_DRY_4':   'RANGING',
    'CANDLESTICK_DRY_0': 'ANY',
    'KAUFMAN_DRY_7':  'TRENDING',
    'GUJRAL_DRY_9':   'TRENDING',
    'BULKOWSKI_CUP_HANDLE': 'RANGING',
    'BULKOWSKI_ADAM_AND_EVE_OR': 'RANGING',
}


class RegimeAdjustedDecay:
    """Tracks signal performance per regime, detects decay only in target regime."""

    def __init__(self, signal_id, lookback=60):
        self.signal_id = signal_id
        self.lookback = lookback
        self.target_regime = SIGNAL_TARGET_REGIME.get(signal_id, 'ANY')

        # Rolling P&L by regime
        self.pnl_by_regime = defaultdict(list)
        self.all_pnl = []

    def update(self, pnl, regime):
        """Record a trade's P&L with its regime context."""
        self.pnl_by_regime[regime].append(pnl)
        self.all_pnl.append(pnl)

        # Trim to lookback
        for r in self.pnl_by_regime:
            if len(self.pnl_by_regime[r]) > self.lookback:
                self.pnl_by_regime[r] = self.pnl_by_regime[r][-self.lookback:]
        if len(self.all_pnl) > self.lookback:
            self.all_pnl = self.all_pnl[-self.lookback:]

    def regime_sharpe(self, regime=None):
        """Compute Sharpe for trades in a specific regime."""
        if regime is None:
            regime = self.target_regime

        if regime == 'ANY':
            pnls = self.all_pnl
        else:
            pnls = self.pnl_by_regime.get(regime, [])

        if len(pnls) < 5:
            return None  # insufficient data

        mean = np.mean(pnls)
        std = np.std(pnls)
        if std == 0:
            return 0.0
        return (mean / std) * np.sqrt(252)

    def is_decaying(self):
        """
        Check if signal is decaying IN ITS TARGET REGIME.

        Returns:
            (is_decay, reason_string)
        """
        target_sharpe = self.regime_sharpe(self.target_regime)
        overall_sharpe = self.regime_sharpe('ANY')

        if target_sharpe is None:
            return False, "insufficient data in target regime"

        # Decay = negative Sharpe even in target regime
        if target_sharpe < 0:
            return True, f"negative Sharpe ({target_sharpe:.2f}) in target regime {self.target_regime}"

        # Not decaying — just in wrong regime
        if overall_sharpe is not None and overall_sharpe < 0 and target_sharpe > 0.5:
            return False, f"overall poor ({overall_sharpe:.2f}) but target regime OK ({target_sharpe:.2f})"

        return False, f"target regime Sharpe = {target_sharpe:.2f}"

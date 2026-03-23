"""
Adaptive Kelly — gear-shifting position sizing based on recent performance.

Instead of a fixed Kelly fraction, adapts based on drawdown, win streak,
VIX regime, and Mamba regime state.

Usage:
    from execution.adaptive_kelly import AdaptiveKelly
    kelly = AdaptiveKelly(base_fraction=0.75)
    frac = kelly.get_fraction(drawdown_pct=0.03, recent_wr=0.52, vix=16)
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Clamp range
MIN_KELLY = 0.20
MAX_KELLY = 1.00


class AdaptiveKelly:
    """
    Gear-shifting Kelly: adapts fraction based on market conditions
    and recent performance.

    Normal:       base_fraction (e.g., 0.75)
    Mild stress:  0.55 - 0.65
    Heavy stress: 0.20 - 0.40
    Hot streak:   0.85 - 1.00
    """

    def __init__(self, base_fraction: float = 0.75):
        self.base = max(MIN_KELLY, min(MAX_KELLY, base_fraction))
        self._history: List[Dict] = []

    def get_fraction(
        self,
        drawdown_pct: float = 0.0,
        recent_wr: float = 0.50,
        consecutive_losers: int = 0,
        vix: float = 15.0,
        regime: str = 'NEUTRAL',
    ) -> Dict:
        """
        Compute adaptive Kelly fraction.

        Args:
            drawdown_pct: current drawdown from peak (0.0 - 1.0)
            recent_wr: win rate of last 10 trades (0.0 - 1.0)
            consecutive_losers: current losing streak
            vix: India VIX
            regime: Mamba regime string

        Returns:
            dict with fraction, adjustments, reason
        """
        frac = self.base
        adjustments = []

        # Drawdown adjustment
        if drawdown_pct > 0.10:
            frac -= 0.30
            adjustments.append(f'DD>{drawdown_pct:.0%}: -0.30')
        elif drawdown_pct > 0.05:
            frac -= 0.15
            adjustments.append(f'DD>{drawdown_pct:.0%}: -0.15')

        # Win rate adjustment
        if recent_wr > 0.55:
            frac += 0.10
            adjustments.append(f'WR={recent_wr:.0%}: +0.10')
        elif recent_wr < 0.35:
            frac -= 0.15
            adjustments.append(f'WR={recent_wr:.0%}: -0.15')

        # Consecutive losers
        if consecutive_losers >= 4:
            frac -= 0.20
            adjustments.append(f'ConsecLoss={consecutive_losers}: -0.20')

        # VIX regime
        if vix > 25:
            frac -= 0.15
            adjustments.append(f'VIX={vix:.0f}: -0.15')
        elif vix > 20:
            frac -= 0.10
            adjustments.append(f'VIX={vix:.0f}: -0.10')

        # Mamba regime
        if regime in ('VOLATILE_BEAR', 'CALM_BEAR'):
            frac -= 0.20
            adjustments.append(f'Regime={regime}: -0.20')
        elif regime == 'VOLATILE_BULL':
            frac -= 0.10
            adjustments.append(f'Regime={regime}: -0.10')

        # Clamp
        frac = max(MIN_KELLY, min(MAX_KELLY, frac))

        result = {
            'fraction': round(frac, 2),
            'base': self.base,
            'adjustments': adjustments,
            'gear': self._classify_gear(frac),
        }

        self._history.append(result)
        return result

    @staticmethod
    def _classify_gear(frac):
        if frac >= 0.85:
            return 'AGGRESSIVE'
        if frac >= 0.65:
            return 'NORMAL'
        if frac >= 0.45:
            return 'CAUTIOUS'
        return 'DEFENSIVE'

    def compute_kelly_from_trades(self, trades: list) -> float:
        """Compute raw Kelly f* from trade history."""
        if len(trades) < 20:
            return 0.50

        wins = [t for t in trades if t.get('pnl_pct', t.get('net_pnl', 0)) > 0]
        losses = [t for t in trades if t.get('pnl_pct', t.get('net_pnl', 0)) <= 0]

        p = len(wins) / len(trades)
        if not wins or not losses:
            return 0.50

        avg_win = abs(sum(t.get('pnl_pct', t.get('net_pnl', 0)) for t in wins) / len(wins))
        avg_loss = abs(sum(t.get('pnl_pct', t.get('net_pnl', 0)) for t in losses) / len(losses))

        if avg_loss <= 0:
            return 0.50

        b = avg_win / avg_loss
        kelly = p - (1 - p) / b if b > 0 else 0
        return max(0, min(1.0, kelly))

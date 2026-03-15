"""
Position Sizing DSL Schema.

Different from entry/exit DSL — describes HOW MUCH to trade, not WHEN.
Used for THARP (R-multiple system) and VINCE (optimal-f) signals.
"""

from dataclasses import dataclass, field
from typing import Optional, List

SIZING_METHODS = [
    'R_MULTIPLE',         # Tharp: risk N% per trade, target NR profit
    'OPTIMAL_F',          # Vince: mathematically optimal fraction
    'FIXED_FRACTIONAL',   # Fixed % of equity per trade
    'VOLATILITY_SCALED',  # Size inversely proportional to volatility
    'KELLY',              # Kelly criterion (or half-Kelly)
    'ANTI_MARTINGALE',    # Increase after wins, decrease after losses
]


@dataclass
class ScaleCondition:
    trigger: str       # 'consecutive_wins', 'consecutive_losses', 'drawdown_pct', 'rolling_sharpe'
    threshold: float   # e.g. 3 wins, 0.10 drawdown
    multiplier: float  # e.g. 1.25 (increase 25%), 0.75 (decrease 25%)

    def to_dict(self):
        return {'trigger': self.trigger, 'threshold': self.threshold, 'multiplier': self.multiplier}

    @classmethod
    def from_dict(cls, d):
        return cls(trigger=d['trigger'], threshold=d['threshold'], multiplier=d['multiplier'])


@dataclass
class PositionSizingRule:
    signal_id: str
    signal_type: str = 'POSITION_SIZING'
    book: str = ''

    sizing_method: str = 'FIXED_FRACTIONAL'

    base_risk_pct: float = 0.01        # 1% of capital at risk per trade
    r_multiple_target: float = 3.0     # target profit as multiple of risk
    r_multiple_stop: float = 1.0       # stop as multiple of risk (1R)

    scale_up_condition: Optional[ScaleCondition] = None
    scale_down_condition: Optional[ScaleCondition] = None

    max_risk_pct: float = 0.02         # never risk more than 2%
    min_risk_pct: float = 0.005        # never risk less than 0.5%

    optimal_f: Optional[float] = None  # Vince optimal-f value
    kelly_fraction: Optional[float] = None  # Kelly or half-Kelly

    applies_to: str = 'ALL'            # ALL, TRENDING, VOLATILE, or specific signal_id

    source_text: str = ''
    confidence: str = 'MEDIUM'

    def to_dict(self):
        d = {
            'signal_id': self.signal_id,
            'signal_type': self.signal_type,
            'book': self.book,
            'sizing_method': self.sizing_method,
            'base_risk_pct': self.base_risk_pct,
            'r_multiple_target': self.r_multiple_target,
            'r_multiple_stop': self.r_multiple_stop,
            'max_risk_pct': self.max_risk_pct,
            'min_risk_pct': self.min_risk_pct,
            'applies_to': self.applies_to,
            'confidence': self.confidence,
        }
        if self.scale_up_condition:
            d['scale_up_condition'] = self.scale_up_condition.to_dict()
        if self.scale_down_condition:
            d['scale_down_condition'] = self.scale_down_condition.to_dict()
        if self.optimal_f is not None:
            d['optimal_f'] = self.optimal_f
        if self.kelly_fraction is not None:
            d['kelly_fraction'] = self.kelly_fraction
        return d

    def compute_lots(self, capital: float, stop_pts: float,
                     lot_value: float, lot_size: int = 75) -> int:
        """
        Compute number of lots based on this sizing rule.

        capital: total trading capital (₹)
        stop_pts: distance to stop loss in Nifty points
        lot_value: current value of 1 lot (close × lot_size)
        """
        risk_amount = capital * self.base_risk_pct
        risk_per_lot = stop_pts * lot_size

        if risk_per_lot <= 0:
            return 1

        lots = int(risk_amount / risk_per_lot)
        return max(1, min(lots, 50))  # floor 1, cap 50

    def compute_expectancy(self, win_rate: float, avg_win_r: float,
                            avg_loss_r: float = 1.0) -> float:
        """
        Tharp's expectancy formula.
        expectancy = (win_rate × avg_win_R) - (loss_rate × avg_loss_R)
        """
        loss_rate = 1 - win_rate
        return (win_rate * avg_win_r) - (loss_rate * avg_loss_r)

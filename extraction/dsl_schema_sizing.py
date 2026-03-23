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
    'DRAWDOWN_SCALED',    # Reduce size during drawdowns
]

SIZING_METHOD_SET = set(SIZING_METHODS)

ALLOWED_TRIGGERS = [
    'consecutive_wins', 'consecutive_losses',
    'drawdown_pct', 'rolling_profit_factor',
    'rolling_win_rate', 'rolling_volatility_20',
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

    def effective_risk_pct(self, consecutive_wins: int = 0,
                           consecutive_losses: int = 0,
                           drawdown_pct: float = 0.0,
                           rolling_volatility: float = None) -> float:
        """
        Compute effective risk % after applying scaling conditions.
        Returns clamped value between min_risk_pct and max_risk_pct.
        """
        risk = self.base_risk_pct

        # Apply scale-up condition
        if self.scale_up_condition:
            sc = self.scale_up_condition
            if sc.trigger == 'consecutive_wins' and consecutive_wins >= sc.threshold:
                risk *= sc.multiplier

        # Apply scale-down condition
        if self.scale_down_condition:
            sc = self.scale_down_condition
            if sc.trigger == 'drawdown_pct' and drawdown_pct >= sc.threshold:
                risk *= sc.multiplier
            elif sc.trigger == 'consecutive_losses' and consecutive_losses >= sc.threshold:
                risk *= sc.multiplier

        # Volatility scaling: lower vol → bigger size, higher vol → smaller
        if self.sizing_method == 'VOLATILITY_SCALED' and rolling_volatility is not None:
            baseline_vol = 0.20  # 20% annualized as 1x baseline
            if rolling_volatility > 0:
                risk *= baseline_vol / rolling_volatility

        # Kelly / optimal_f override
        if self.sizing_method == 'KELLY' and self.kelly_fraction is not None:
            risk = self.kelly_fraction
        elif self.sizing_method == 'OPTIMAL_F' and self.optimal_f is not None:
            risk = self.optimal_f

        return max(self.min_risk_pct, min(self.max_risk_pct, risk))

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

    @classmethod
    def from_dict(cls, d: dict) -> 'PositionSizingRule':
        scale_up = None
        if d.get('scale_up_condition'):
            scale_up = ScaleCondition.from_dict(d['scale_up_condition'])
        scale_down = None
        if d.get('scale_down_condition'):
            scale_down = ScaleCondition.from_dict(d['scale_down_condition'])

        return cls(
            signal_id=d.get('signal_id', 'unknown'),
            sizing_method=d.get('sizing_method', 'FIXED_FRACTIONAL'),
            base_risk_pct=float(d.get('base_risk_pct', 0.01)),
            r_multiple_target=float(d.get('r_multiple_target', 3.0)),
            r_multiple_stop=float(d.get('r_multiple_stop', 1.0)),
            optimal_f=d.get('optimal_f'),
            kelly_fraction=d.get('kelly_fraction'),
            scale_up_condition=scale_up,
            scale_down_condition=scale_down,
            max_risk_pct=float(d.get('max_risk_pct', 0.02)),
            min_risk_pct=float(d.get('min_risk_pct', 0.005)),
            applies_to=d.get('applies_to', 'ALL'),
            confidence=d.get('confidence', 'MEDIUM'),
            source_text=d.get('source_text', ''),
            book=d.get('book', ''),
        )

    @classmethod
    def from_llm_response(cls, signal_id: str, resp: dict) -> 'PositionSizingRule':
        """Parse an LLM sizing response into a PositionSizingRule."""
        scale_up = None
        scale_down = None
        trigger = resp.get('trigger_condition')

        if trigger and resp.get('scale_up_multiplier'):
            scale_up = ScaleCondition(
                trigger=trigger.get('metric', ''),
                threshold=float(trigger.get('threshold', 0)),
                multiplier=float(resp['scale_up_multiplier']),
            )
        if trigger and resp.get('scale_down_multiplier'):
            scale_down = ScaleCondition(
                trigger=trigger.get('metric', ''),
                threshold=float(trigger.get('threshold', 0)),
                multiplier=float(resp['scale_down_multiplier']),
            )

        return cls(
            signal_id=signal_id,
            sizing_method=resp.get('sizing_method', 'FIXED_FRACTIONAL'),
            base_risk_pct=float(resp.get('base_risk_pct', 0.01)),
            r_multiple_target=float(resp.get('r_multiple_target', 3.0)),
            r_multiple_stop=float(resp.get('r_multiple_stop', 1.0)),
            optimal_f=resp.get('base_risk_pct') if resp.get('sizing_method') == 'OPTIMAL_F' else None,
            kelly_fraction=resp.get('scale_factor'),
            scale_up_condition=scale_up,
            scale_down_condition=scale_down,
            max_risk_pct=float(resp.get('max_risk_pct', 0.02)),
            min_risk_pct=float(resp.get('min_risk_pct', 0.005)),
            applies_to=resp.get('applies_to', 'ALL'),
            confidence=resp.get('confidence', 'MEDIUM'),
        )


def validate_sizing_rule(rule: PositionSizingRule) -> tuple:
    """
    Validate a PositionSizingRule for structural correctness.
    Returns (passed: bool, issues: list[str])
    """
    issues = []

    if rule.sizing_method not in SIZING_METHOD_SET:
        issues.append(f"Unknown sizing_method: {rule.sizing_method}")

    if rule.base_risk_pct < 0 or rule.base_risk_pct > 1.0:
        issues.append(f"base_risk_pct={rule.base_risk_pct} outside [0, 1.0]")

    if rule.max_risk_pct < rule.min_risk_pct:
        issues.append(f"max_risk_pct ({rule.max_risk_pct}) < min_risk_pct ({rule.min_risk_pct})")

    if rule.r_multiple_stop <= 0:
        issues.append(f"r_multiple_stop must be positive, got {rule.r_multiple_stop}")

    if rule.scale_up_condition and rule.scale_up_condition.multiplier <= 1.0:
        issues.append(f"scale_up multiplier should be > 1.0, got {rule.scale_up_condition.multiplier}")

    if rule.scale_down_condition and rule.scale_down_condition.multiplier >= 1.0:
        issues.append(f"scale_down multiplier should be < 1.0, got {rule.scale_down_condition.multiplier}")

    if rule.optimal_f is not None and (rule.optimal_f <= 0 or rule.optimal_f >= 1.0):
        issues.append(f"optimal_f={rule.optimal_f} outside (0, 1.0)")

    if rule.kelly_fraction is not None and (rule.kelly_fraction <= 0 or rule.kelly_fraction >= 1.0):
        issues.append(f"kelly_fraction={rule.kelly_fraction} outside (0, 1.0)")

    return len(issues) == 0, issues

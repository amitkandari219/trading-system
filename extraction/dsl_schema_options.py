"""
Options DSL Schema for L8 signals.

Each options signal specifies:
- Strategy type (iron_condor, short_strangle, etc.)
- Entry conditions (IV rank, regime, VIX, DTE)
- Leg definitions (strike selection by delta or offset)
- Exit conditions (profit target, stop loss, DTE)
- Position sizing (max risk, margin)
"""

from dataclasses import dataclass, field
from typing import List, Optional


STRATEGY_TYPES = [
    'SHORT_STRANGLE',    # sell OTM call + OTM put
    'IRON_CONDOR',       # sell OTM strangle + buy wings
    'SHORT_STRADDLE',    # sell ATM call + ATM put
    'BULL_PUT_SPREAD',   # sell put + buy lower put
    'BEAR_CALL_SPREAD',  # sell call + buy higher call
    'CALENDAR_SPREAD',   # sell near month + buy far month
    'PROTECTIVE_PUT',    # buy put for downside protection
    'COVERED_CALL',      # sell OTM call against long futures
    'RATIO_SPREAD',      # buy 1 + sell 2 (or vice versa)
    'NAKED_STRANGLE',    # same as short strangle (alias)
]


@dataclass
class OptionLeg:
    """Definition of one option leg."""
    option_type: str             # CE or PE
    action: str                  # BUY or SELL
    strike_method: str = 'delta' # 'delta', 'offset', 'atm', 'otm_pct'
    delta_target: float = 0.0   # target delta (e.g., 0.25 for 25-delta)
    offset_points: float = 0.0  # offset from ATM in points (e.g., 200)
    otm_pct: float = 0.0        # OTM as % of spot (e.g., 0.02 = 2%)
    lots: int = 1

    def to_dict(self):
        return {
            'option_type': self.option_type,
            'action': self.action,
            'strike_method': self.strike_method,
            'delta_target': self.delta_target,
            'offset_points': self.offset_points,
            'otm_pct': self.otm_pct,
            'lots': self.lots,
        }


@dataclass
class OptionsSignalRule:
    """Complete options strategy signal definition."""
    signal_id: str
    strategy_type: str               # from STRATEGY_TYPES

    # Legs
    legs: List[OptionLeg] = field(default_factory=list)

    # Entry conditions
    iv_rank_min: float = 0.0         # minimum IV rank (0-100)
    iv_rank_max: float = 100.0
    vix_min: float = 0.0
    vix_max: float = 50.0
    dte_min: int = 1                 # minimum days to expiry
    dte_max: int = 45                # maximum days to expiry
    regime_filter: List[str] = field(default_factory=lambda: ['ANY'])

    # Exit conditions
    profit_target_pct: float = 0.50   # exit at 50% of max credit
    stop_loss_multiple: float = 2.0   # exit at 2x credit received (loss)
    exit_dte: int = 2                 # exit at N days to expiry
    max_hold_days: int = 30

    # Position sizing
    max_risk_pct: float = 0.02        # max 2% of capital at risk
    max_margin_pct: float = 0.15      # max 15% of capital as margin
    lots_per_trade: int = 1

    # Costs
    brokerage_per_lot: float = 40.0   # ₹40 per lot per leg
    stt_pct: float = 0.0005           # 0.05% STT on sell side
    slippage_pct: float = 0.001       # 0.1% of premium

    # Metadata
    signal_family: str = 'MCMILLAN'
    confidence: str = 'MEDIUM'
    expected_win_rate: float = 0.0
    notes: str = ''

    def to_dict(self):
        return {
            'signal_id': self.signal_id,
            'strategy_type': self.strategy_type,
            'legs': [l.to_dict() for l in self.legs],
            'iv_rank_min': self.iv_rank_min,
            'iv_rank_max': self.iv_rank_max,
            'vix_min': self.vix_min,
            'vix_max': self.vix_max,
            'dte_min': self.dte_min,
            'dte_max': self.dte_max,
            'regime_filter': self.regime_filter,
            'profit_target_pct': self.profit_target_pct,
            'stop_loss_multiple': self.stop_loss_multiple,
            'exit_dte': self.exit_dte,
            'max_risk_pct': self.max_risk_pct,
        }


# ================================================================
# PRE-BUILT STRATEGY TEMPLATES
# ================================================================

def short_strangle(call_delta=0.20, put_delta=0.20):
    """Standard short strangle legs."""
    return [
        OptionLeg(option_type='CE', action='SELL', strike_method='delta',
                  delta_target=call_delta),
        OptionLeg(option_type='PE', action='SELL', strike_method='delta',
                  delta_target=put_delta),
    ]


def iron_condor(call_delta=0.20, put_delta=0.20, wing_offset=50):
    """Iron condor: short strangle + long wings."""
    return [
        OptionLeg(option_type='CE', action='SELL', strike_method='delta',
                  delta_target=call_delta),
        OptionLeg(option_type='PE', action='SELL', strike_method='delta',
                  delta_target=put_delta),
        OptionLeg(option_type='CE', action='BUY', strike_method='offset',
                  offset_points=wing_offset),
        OptionLeg(option_type='PE', action='BUY', strike_method='offset',
                  offset_points=-wing_offset),
    ]


def bull_put_spread(sell_delta=0.30, width=100):
    """Bull put spread: sell put + buy lower put."""
    return [
        OptionLeg(option_type='PE', action='SELL', strike_method='delta',
                  delta_target=sell_delta),
        OptionLeg(option_type='PE', action='BUY', strike_method='offset',
                  offset_points=-width),
    ]


def bear_call_spread(sell_delta=0.30, width=100):
    """Bear call spread: sell call + buy higher call."""
    return [
        OptionLeg(option_type='CE', action='SELL', strike_method='delta',
                  delta_target=sell_delta),
        OptionLeg(option_type='CE', action='BUY', strike_method='offset',
                  offset_points=width),
    ]

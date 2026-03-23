"""
Intraday DSL Schema for L9 signals.

Extends the standard DSL with intraday-specific indicators, session filters,
and time-based exits. All positions must close by 15:20.

Key differences from standard DSL:
- Timeframe specification (5min/15min/60min)
- Session context (opening range, VWAP, time-to-close)
- Forced session close at 15:20
- No overnight holds
- Entry window filters (e.g., only enter 9:30-11:00)
"""

from dataclasses import dataclass, field
from typing import List, Optional


INTRADAY_INDICATORS = [
    # Session indicators
    "vwap", "vwap_deviation",
    "opening_range_high", "opening_range_low",
    "or_breakout", "or_breakdown",
    "session_bar", "time_to_close",
    "overnight_gap_pct",
    # Prior day levels
    "prev_day_high", "prev_day_low", "prev_day_close",
    # Standard (computed on intraday bars)
    "open", "high", "low", "close", "volume",
    "sma_20", "sma_50", "ema_9", "ema_20",
    "rsi_14", "atr_14", "adx_14",
    "bb_upper", "bb_middle", "bb_lower", "bb_pct_b", "bb_bandwidth",
    "stoch_k", "stoch_d",
    "body", "body_pct", "upper_wick", "lower_wick", "range",
    "prev_close", "prev_high", "prev_low", "prev_open",
    "returns", "vol_ratio_20",
]

INTRADAY_INDICATOR_SET = set(INTRADAY_INDICATORS)

ALLOWED_TIMEFRAMES = ["5min", "15min", "60min"]

ALLOWED_SESSION_FILTERS = [
    "ANY",              # no filter
    "FIRST_HOUR",       # 9:15-10:15
    "MORNING",          # 9:15-12:00
    "MIDDAY",           # 11:30-13:30
    "AFTERNOON",        # 13:00-15:30
    "LAST_HOUR",        # 14:30-15:30
    "NO_LAST_30MIN",    # exclude 15:00-15:30 (avoid close volatility)
]


@dataclass
class IntradayCondition:
    """A single condition in an intraday signal rule."""
    left: str          # from INTRADAY_INDICATORS
    operator: str      # >, <, >=, <=, ==, crosses_above, crosses_below
    right: str         # indicator name, number, or boolean

    def to_dict(self):
        return {"left": self.left, "operator": self.operator, "right": self.right}

    @classmethod
    def from_dict(cls, d):
        return cls(left=d["left"], operator=d["operator"], right=str(d["right"]))


@dataclass
class IntradaySignalRule:
    """Complete structured intraday signal rule."""
    signal_id: str
    timeframe: str = "5min"            # 5min, 15min, 60min

    entry_long: List[IntradayCondition] = field(default_factory=list)
    entry_short: List[IntradayCondition] = field(default_factory=list)
    exit_long: List[IntradayCondition] = field(default_factory=list)
    exit_short: List[IntradayCondition] = field(default_factory=list)

    entry_logic: str = "AND"
    exit_logic: str = "OR"

    stop_loss_pct: float = 0.005       # 0.5% (tighter for intraday)
    max_hold_bars: int = 50            # max bars before forced exit
    direction: str = "BOTH"

    # Session constraints
    session_filter: str = "ANY"        # from ALLOWED_SESSION_FILTERS
    session_close_time: str = "15:20"  # hard close — no overnight
    allow_overnight: bool = False      # always False for L9
    entry_start_time: Optional[str] = None   # e.g., "09:30"
    entry_end_time: Optional[str] = None     # e.g., "14:30"

    # Slippage
    slippage_pct: float = 0.0005       # 0.05% each way

    # Metadata
    signal_family: str = "BROOKS"
    confidence: str = "MEDIUM"
    expected_trades_per_week: float = 0.0

    def to_dict(self):
        return {
            "signal_id": self.signal_id,
            "timeframe": self.timeframe,
            "entry_long": [c.to_dict() for c in self.entry_long],
            "entry_short": [c.to_dict() for c in self.entry_short],
            "exit_long": [c.to_dict() for c in self.exit_long],
            "exit_short": [c.to_dict() for c in self.exit_short],
            "entry_logic": self.entry_logic,
            "exit_logic": self.exit_logic,
            "stop_loss_pct": self.stop_loss_pct,
            "max_hold_bars": self.max_hold_bars,
            "direction": self.direction,
            "session_filter": self.session_filter,
            "session_close_time": self.session_close_time,
            "allow_overnight": self.allow_overnight,
            "entry_start_time": self.entry_start_time,
            "entry_end_time": self.entry_end_time,
            "slippage_pct": self.slippage_pct,
            "signal_family": self.signal_family,
            "confidence": self.confidence,
        }

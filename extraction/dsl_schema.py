"""
DSL Schema for structured signal translation.

Forces Haiku to fill a strict template instead of generating free-form
indicator conditions. Every field is constrained to valid values — unknown
indicators, phantom conditions, and self-comparisons are structurally
impossible.
"""

from dataclasses import dataclass, field
from typing import List, Optional


ALLOWED_INDICATORS = [
    # Price
    "open", "high", "low", "close", "volume",
    # Moving averages
    "sma_5", "sma_10", "sma_20", "sma_40", "sma_50", "sma_80", "sma_100", "sma_200",
    "ema_5", "ema_10", "ema_20", "ema_50", "ema_100", "ema_200",
    # RSI
    "rsi_7", "rsi_14", "rsi_21",
    # ATR
    "atr_7", "atr_14", "atr_20",
    # Bollinger Bands (20-period default)
    "bb_upper", "bb_middle", "bb_lower", "bb_pct_b", "bb_bandwidth",
    # Bollinger Bands (30-period — Chan quantitative)
    "bb_upper_30", "bb_middle_30", "bb_lower_30", "bb_pct_b_30", "bb_bandwidth_30",
    # MACD
    "macd", "macd_signal", "macd_hist",
    # ADX
    "adx_14",
    # Stochastic
    "stoch_k", "stoch_d", "stoch_k_5", "stoch_d_5",
    # Donchian
    "dc_upper", "dc_lower", "dc_middle",
    # Pivots
    "pivot", "r1", "s1", "r2", "s2", "r3", "s3",
    # Volume
    "vol_ratio_20",
    # Volatility
    "hvol_6", "hvol_20", "hvol_100", "india_vix",
    # Price position
    "price_pos_20",
    # Mean reversion z-score (close vs 20d mean, normalized by 20d std)
    "zscore_20", "zscore_50",
    # Previous bar
    "prev_close", "prev_high", "prev_low", "prev_open", "prev_volume",
    # Returns
    "returns", "log_returns",
    # Bar properties
    "body", "body_pct", "upper_wick", "lower_wick", "range",
    # True range
    "true_range",
    # Regime (string column — use with "is" operator only)
    "regime",
]

ALLOWED_OPERATORS = [
    ">", "<", ">=", "<=", "==",
    "crosses_above",   # today > threshold AND prev_day <= threshold
    "crosses_below",   # today < threshold AND prev_day >= threshold
    "is",              # for regime string comparisons only
]

ALLOWED_DIRECTIONS = ["LONG", "SHORT", "BOTH"]

ALLOWED_REGIMES = [
    "TRENDING_UP", "TRENDING_DOWN", "RANGING", "HIGH_VOL", "ANY",
]

# Set for fast lookup
INDICATOR_SET = set(ALLOWED_INDICATORS)
OPERATOR_SET = set(ALLOWED_OPERATORS)


@dataclass
class DSLCondition:
    """A single condition in a signal rule."""
    left: str         # must be from ALLOWED_INDICATORS
    operator: str     # must be from ALLOWED_OPERATORS
    right: str        # ALLOWED_INDICATORS member, numeric string, or regime string

    def to_dict(self) -> dict:
        return {"left": self.left, "operator": self.operator, "right": self.right}

    @classmethod
    def from_dict(cls, d: dict) -> 'DSLCondition':
        return cls(left=d["left"], operator=d["operator"], right=str(d["right"]))


@dataclass
class DSLSignalRule:
    """Complete structured signal rule — output of the DSL translator."""
    signal_id: str
    entry_long: List[DSLCondition] = field(default_factory=list)
    entry_short: List[DSLCondition] = field(default_factory=list)
    exit_long: List[DSLCondition] = field(default_factory=list)
    exit_short: List[DSLCondition] = field(default_factory=list)
    entry_logic: str = "AND"       # "AND" or "OR"
    exit_logic: str = "AND"        # "AND" or "OR"
    stop_loss_pct: float = 2.0     # 0.5 to 10.0
    hold_days_max: int = 10        # 1 to 30
    direction: str = "BOTH"        # LONG, SHORT, BOTH
    target_regime: List[str] = field(default_factory=lambda: ["ANY"])

    # Translation metadata
    untranslatable: bool = False
    untranslatable_reason: Optional[str] = None
    translation_notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "entry_long": [c.to_dict() for c in self.entry_long],
            "entry_short": [c.to_dict() for c in self.entry_short],
            "exit_long": [c.to_dict() for c in self.exit_long],
            "exit_short": [c.to_dict() for c in self.exit_short],
            "entry_logic": self.entry_logic,
            "exit_logic": self.exit_logic,
            "stop_loss_pct": self.stop_loss_pct,
            "hold_days_max": self.hold_days_max,
            "direction": self.direction,
            "target_regime": self.target_regime,
            "untranslatable": self.untranslatable,
            "untranslatable_reason": self.untranslatable_reason,
            "translation_notes": self.translation_notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'DSLSignalRule':
        return cls(
            signal_id=d["signal_id"],
            entry_long=[DSLCondition.from_dict(c) for c in d.get("entry_long", [])],
            entry_short=[DSLCondition.from_dict(c) for c in d.get("entry_short", [])],
            exit_long=[DSLCondition.from_dict(c) for c in d.get("exit_long", [])],
            exit_short=[DSLCondition.from_dict(c) for c in d.get("exit_short", [])],
            entry_logic=d.get("entry_logic", "AND"),
            exit_logic=d.get("exit_logic", "AND"),
            stop_loss_pct=d.get("stop_loss_pct", 2.0),
            hold_days_max=d.get("hold_days_max", 10),
            direction=d.get("direction", "BOTH"),
            target_regime=d.get("target_regime", ["ANY"]),
            untranslatable=d.get("untranslatable", False),
            untranslatable_reason=d.get("untranslatable_reason"),
            translation_notes=d.get("translation_notes"),
        )


class DSLTranslationError(Exception):
    """Raised when DSL translation fails."""
    pass

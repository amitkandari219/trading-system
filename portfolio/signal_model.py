"""
Signal dataclass — runtime representation of one signal from the signal registry.
Used by SignalSelector, ExecutionEngine, and RiskEngine.
Loaded from signals table in PostgreSQL — all fields map to columns.
"""

import copy
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Signal:
    """
    Runtime representation of one signal from the signal registry.
    Loaded fresh at 8:50 AM from the signals table.
    """
    # Identity
    signal_id:            str
    name:                 str
    book_id:              str

    # Classification
    signal_category:      str    # TREND|REVERSION|VOL|PATTERN|EVENT
    direction:            str    # LONG|SHORT|NEUTRAL|CONTEXT_DEPENDENT
    instrument:           str    # FUTURES|OPTIONS_BUYING|OPTIONS_SELLING|SPREAD|COMBINED|ANY
    classification:       str    # PRIMARY|SECONDARY
    status:               str    # ACTIVE|WATCH|INACTIVE|...

    # Regime and calendar rules
    target_regimes:       List[str]  # e.g. ['TRENDING', 'ANY']
    expiry_week_behavior: str    # NORMAL|AVOID_MONDAY|REDUCE_SIZE_MONDAY|AVOID_EXPIRY_WEEK
    avoid_rbi_day:        bool = False

    # Performance metrics (from backtest / live tracking)
    sharpe_ratio:         float = 0.0
    rolling_sharpe_60d:   float = 0.0

    # Capital requirements
    required_margin:      int = 0  # ₹ approximate margin per trade

    # Runtime fields (set by SignalSelector during selection)
    efficiency_score:     float = 0.0  # computed each morning
    execution_priority:   int = 99     # 1 = execute first
    size_multiplier:      float = 1.0  # 1.0 = full size, 0.5 = half size

    # FII overnight signals only
    source:               str = 'BOOK'  # 'BOOK' or 'FII_OVERNIGHT'
    requires_confirmation: bool = False

    def copy(self) -> 'Signal':
        """Return a shallow copy for per-instance modifications (e.g. size_multiplier)."""
        return copy.copy(self)

    @classmethod
    def from_db_row(cls, row: dict) -> 'Signal':
        """Construct a Signal from a signals table row dict."""
        return cls(
            signal_id            = row['signal_id'],
            name                 = row['name'],
            book_id              = row['book_id'],
            signal_category      = row['signal_category'],
            direction            = row['direction'],
            instrument           = row['instrument'],
            classification       = row['classification'],
            status               = row['status'],
            target_regimes       = row['target_regimes'],  # TEXT[] from PG
            expiry_week_behavior = row['expiry_week_behavior'],
            avoid_rbi_day        = row.get('avoid_rbi_day', False),
            sharpe_ratio         = row.get('sharpe_ratio', 0.0) or 0.0,
            rolling_sharpe_60d   = row.get('rolling_sharpe_60d', 0.0) or 0.0,
            required_margin      = row.get('required_margin', 0) or 0,
        )

    @classmethod
    def from_fii_redis(cls, fields: dict) -> 'Signal':
        """Construct a Signal from a Redis FII_OVERNIGHT message."""
        return cls(
            signal_id            = fields['signal_id'],
            name                 = fields.get('pattern', fields['signal_id']),
            book_id              = 'NSE_EMPIRICAL',
            signal_category      = 'EVENT',
            direction            = fields['direction'],
            instrument           = 'FUTURES',
            classification       = 'SECONDARY',
            status               = 'ACTIVE',
            target_regimes       = ['ANY'],
            expiry_week_behavior = 'NORMAL',
            source               = 'FII_OVERNIGHT',
            requires_confirmation = True,
            sharpe_ratio         = float(fields.get('confidence', 0.65)),
            required_margin      = 120_000,   # ~₹1.2L for 1 futures lot
        )

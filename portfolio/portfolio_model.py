"""
Portfolio dataclass — runtime portfolio state.
Loaded at 8:50 AM from the latest portfolio_state snapshot + open trades.
Updated in Redis after every fill (PORTFOLIO_STATE key).
"""

from dataclasses import dataclass, field
from typing import List

from config.settings import TOTAL_CAPITAL, CAPITAL_RESERVE_FRACTION


@dataclass
class PortfolioGreeks:
    delta: float = 0.0
    vega:  float = 0.0
    gamma: float = 0.0
    theta: float = 0.0


@dataclass
class OpenPosition:
    trade_id:      int
    signal_id:     str
    instrument:    str
    direction:     str    # 'LONG' | 'SHORT'
    lots:          int
    entry_price:   float
    current_price: float = 0.0


@dataclass
class Portfolio:
    """
    Runtime portfolio state. Loaded fresh at 8:50 AM each day.
    Also updated in Redis after every fill (PORTFOLIO_STATE key).
    """
    total_capital:    float = TOTAL_CAPITAL
    deployed_capital: float = 0.0
    cash_reserve:     float = TOTAL_CAPITAL * CAPITAL_RESERVE_FRACTION
    open_positions:   List[OpenPosition] = field(default_factory=list)
    greeks:           PortfolioGreeks     = field(default_factory=PortfolioGreeks)
    daily_pnl:        float = 0.0
    mtd_pnl:          float = 0.0

    @classmethod
    def from_db(cls, db) -> 'Portfolio':
        """
        Load current portfolio from latest DAILY_CLOSE or POST_TRADE snapshot
        plus all open trades (exit_date IS NULL, trade_type='LIVE').
        """
        # Latest snapshot for capital and greeks
        snap = db.execute("""
            SELECT total_capital, deployed_capital, cash_reserve,
                   portfolio_delta, portfolio_vega,
                   portfolio_gamma, portfolio_theta,
                   daily_pnl, mtd_pnl
            FROM portfolio_state
            WHERE snapshot_type IN ('DAILY_CLOSE', 'POST_TRADE', 'RECONCILIATION')
            ORDER BY snapshot_time DESC LIMIT 1
        """).fetchone()

        # Open positions
        rows = db.execute("""
            SELECT trade_id, signal_id, instrument, direction,
                   lots, entry_price
            FROM trades
            WHERE exit_date IS NULL AND trade_type = 'LIVE'
        """).fetchall()

        positions = [
            OpenPosition(
                trade_id=r['trade_id'], signal_id=r['signal_id'],
                instrument=r['instrument'], direction=r['direction'],
                lots=r['lots'], entry_price=r['entry_price']
            )
            for r in rows
        ]

        if snap:
            return cls(
                total_capital    = snap['total_capital'],
                deployed_capital = snap['deployed_capital'],
                cash_reserve     = snap['cash_reserve'],
                open_positions   = positions,
                greeks           = PortfolioGreeks(
                    delta = snap['portfolio_delta'] or 0.0,
                    vega  = snap['portfolio_vega']  or 0.0,
                    gamma = snap['portfolio_gamma'] or 0.0,
                    theta = snap['portfolio_theta'] or 0.0,
                ),
                daily_pnl = snap['daily_pnl'] or 0.0,
                mtd_pnl   = snap['mtd_pnl']   or 0.0,
            )
        # No snapshot yet (first run of the day)
        return cls(open_positions=positions)

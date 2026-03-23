"""
Portfolio Greeks Monitor for L8 options positions.

Tracks aggregate delta, gamma, theta, vega across all open options.
Alerts when limits breached. Logs daily snapshot to greeks_history.

Hard limits (from peer review):
- Max delta: ±0.50 (portfolio level)
- Max vega: ₹3,000 per 1% IV move
- Max gamma: ₹30,000 per 1% spot move
- Min theta: -₹5,000/day (should be collecting theta, not paying)
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Optional

from config.settings import NIFTY_LOT_SIZE

logger = logging.getLogger(__name__)

# Portfolio-level Greek limits
GREEK_LIMITS = {
    'max_abs_delta': 0.50,
    'max_vega': 3000,       # ₹ per 1% IV move
    'max_gamma': 30000,     # ₹ per 1% spot move
    'min_theta': -5000,     # must be > this (collecting theta)
    'alert_delta': 0.70,    # auto-reduce trigger
    'alert_vega': 5000,     # auto-reduce trigger
}


@dataclass
class PositionGreeks:
    """Greeks for a single options position."""
    signal_id: str
    option_type: str     # CE or PE
    action: str          # BUY or SELL
    strike: float
    lots: int
    delta: float
    gamma: float
    theta: float
    vega: float
    premium: float       # current market value

    @property
    def position_delta(self):
        """Net delta (negative for short positions)."""
        mult = -1 if self.action == 'SELL' else 1
        return self.delta * mult * self.lots * NIFTY_LOT_SIZE

    @property
    def position_gamma(self):
        mult = -1 if self.action == 'SELL' else 1
        return self.gamma * mult * self.lots * NIFTY_LOT_SIZE

    @property
    def position_theta(self):
        """Theta in ₹/day. Positive for sellers (collecting decay)."""
        mult = -1 if self.action == 'SELL' else 1
        return self.theta * mult * self.lots * NIFTY_LOT_SIZE

    @property
    def position_vega(self):
        mult = -1 if self.action == 'SELL' else 1
        return self.vega * mult * self.lots * NIFTY_LOT_SIZE


class GreeksMonitor:
    """Monitors portfolio-level Greeks and enforces limits."""

    def __init__(self, limits=None):
        self.limits = limits or GREEK_LIMITS
        self.positions: List[PositionGreeks] = []
        self.alerts: List[Dict] = []

    def add_position(self, pos: PositionGreeks):
        self.positions.append(pos)

    def clear_positions(self):
        self.positions = []

    def portfolio_greeks(self) -> Dict:
        """Compute aggregate portfolio Greeks."""
        total_delta = sum(p.position_delta for p in self.positions)
        total_gamma = sum(p.position_gamma for p in self.positions)
        total_theta = sum(p.position_theta for p in self.positions)
        total_vega = sum(p.position_vega for p in self.positions)
        total_premium = sum(p.premium * p.lots * NIFTY_LOT_SIZE
                            * (-1 if p.action == 'SELL' else 1)
                            for p in self.positions)

        return {
            'delta': round(total_delta, 2),
            'gamma': round(total_gamma, 2),
            'theta': round(total_theta, 2),
            'vega': round(total_vega, 2),
            'net_premium': round(total_premium, 2),
            'n_positions': len(self.positions),
        }

    def check_limits(self) -> List[Dict]:
        """Check portfolio Greeks against limits. Returns list of breaches."""
        greeks = self.portfolio_greeks()
        self.alerts = []

        if abs(greeks['delta']) > self.limits['max_abs_delta']:
            self.alerts.append({
                'type': 'DELTA_BREACH',
                'level': 'WARNING',
                'value': greeks['delta'],
                'limit': self.limits['max_abs_delta'],
                'action': 'Reduce directional exposure',
            })

        if abs(greeks['delta']) > self.limits['alert_delta']:
            self.alerts.append({
                'type': 'DELTA_CRITICAL',
                'level': 'CRITICAL',
                'value': greeks['delta'],
                'limit': self.limits['alert_delta'],
                'action': 'AUTO-REDUCE: close most directional position',
            })

        if abs(greeks['vega']) > self.limits['max_vega']:
            self.alerts.append({
                'type': 'VEGA_BREACH',
                'level': 'WARNING',
                'value': greeks['vega'],
                'limit': self.limits['max_vega'],
                'action': 'Reduce vega exposure — close shortest DTE position',
            })

        if abs(greeks['vega']) > self.limits['alert_vega']:
            self.alerts.append({
                'type': 'VEGA_CRITICAL',
                'level': 'CRITICAL',
                'value': greeks['vega'],
                'limit': self.limits['alert_vega'],
                'action': 'AUTO-REDUCE: close highest vega position',
            })

        if abs(greeks['gamma']) > self.limits['max_gamma']:
            self.alerts.append({
                'type': 'GAMMA_BREACH',
                'level': 'WARNING',
                'value': greeks['gamma'],
                'limit': self.limits['max_gamma'],
                'action': 'Reduce gamma — close nearest expiry position',
            })

        if greeks['theta'] < self.limits['min_theta']:
            self.alerts.append({
                'type': 'THETA_NEGATIVE',
                'level': 'WARNING',
                'value': greeks['theta'],
                'limit': self.limits['min_theta'],
                'action': 'Portfolio is net paying theta — review bought positions',
            })

        return self.alerts

    def daily_snapshot(self, trade_date: date = None) -> Dict:
        """Generate daily Greeks snapshot for logging."""
        greeks = self.portfolio_greeks()
        alerts = self.check_limits()

        return {
            'date': trade_date or date.today(),
            **greeks,
            'alerts': len(alerts),
            'critical_alerts': sum(1 for a in alerts if a['level'] == 'CRITICAL'),
        }

    def save_snapshot(self, conn, trade_date: date = None):
        """Save daily snapshot to greeks_history table."""
        snap = self.daily_snapshot(trade_date)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO greeks_history (date, portfolio_delta, portfolio_gamma,
                portfolio_theta, portfolio_vega, n_options_positions, margin_used)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                portfolio_delta = EXCLUDED.portfolio_delta,
                portfolio_gamma = EXCLUDED.portfolio_gamma,
                portfolio_theta = EXCLUDED.portfolio_theta,
                portfolio_vega = EXCLUDED.portfolio_vega,
                n_options_positions = EXCLUDED.n_options_positions
        """, (snap['date'], snap['delta'], snap['gamma'],
              snap['theta'], snap['vega'], snap['n_positions'], 0))
        conn.commit()

    def print_status(self):
        """Print portfolio Greeks status."""
        greeks = self.portfolio_greeks()
        alerts = self.check_limits()

        print(f"\nPortfolio Greeks ({greeks['n_positions']} positions):")
        print(f"  Delta:  {greeks['delta']:>8.2f}  (limit: ±{self.limits['max_abs_delta']})")
        print(f"  Gamma:  {greeks['gamma']:>8.0f}  (limit: {self.limits['max_gamma']})")
        print(f"  Theta:  {greeks['theta']:>8.0f}  (min: {self.limits['min_theta']})")
        print(f"  Vega:   {greeks['vega']:>8.0f}  (limit: {self.limits['max_vega']})")
        print(f"  Premium:{greeks['net_premium']:>8.0f}")

        if alerts:
            print(f"\n  ALERTS ({len(alerts)}):")
            for a in alerts:
                print(f"    [{a['level']}] {a['type']}: {a['value']:.2f} > {a['limit']} — {a['action']}")
        else:
            print(f"\n  All limits OK")

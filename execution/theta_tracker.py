"""
Theta Tracker — monitors theta decay on all open options positions.

Computes per-position and portfolio-level theta using Black-Scholes,
checks for divergence between theoretical and actual P&L, and flags
exit conditions (profit target, gamma risk, diminishing returns).

Usage:
    from execution.theta_tracker import ThetaTracker
    tracker = ThetaTracker(db_conn)
    pos_theta = tracker.compute_position_theta(spot, strike, dte, iv, 'CE', 'SELL', 2)
    portfolio = tracker.compute_portfolio_theta(positions, spot)
    divergence = tracker.check_theta_divergence(position, actual_pnl, days_held)
    exit_info = tracker.check_exit_conditions(position, current_pnl, credit, dte)
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from config.settings import NIFTY_LOT_SIZE, RISK_FREE_RATE
from data.options_loader import bs_price

logger = logging.getLogger(__name__)


class ThetaTracker:
    """
    Tracks theta decay on open options positions.

    Computes daily theta per position and across the portfolio,
    detects divergence from theoretical decay, and recommends
    exit timing based on profit targets and gamma risk.
    """

    def __init__(self, db_conn, risk_free_rate: float = RISK_FREE_RATE):
        self.db_conn = db_conn
        self.risk_free_rate = risk_free_rate
        self._positions: Dict[str, Dict[str, Any]] = {}

    # ================================================================
    # POSITION-LEVEL THETA
    # ================================================================

    def compute_position_theta(
        self,
        spot: float,
        strike: float,
        dte: float,
        iv: float,
        option_type: str,
        action: str,
        lots: int,
        lot_size: int = NIFTY_LOT_SIZE,
    ) -> dict:
        """
        Compute theta for a single option leg.

        Theta = bs_price(T) - bs_price(T - 1/365).
        For SELL positions the sign is flipped (seller collects theta).

        Returns:
            {'theta_per_day': float, 'theta_per_day_rs': float}
        """
        T = dte / 365.0
        T_next = (dte - 1) / 365.0

        price_today = bs_price(spot, strike, T, self.risk_free_rate, iv, option_type)
        price_tomorrow = bs_price(spot, strike, T_next, self.risk_free_rate, iv, option_type)

        # Raw theta is the change in option price (negative for long holders)
        raw_theta = price_tomorrow - price_today  # typically negative

        # SELL positions benefit from decay, so flip sign
        direction = -1.0 if action.upper() == 'SELL' else 1.0
        theta_per_day = direction * raw_theta

        total_qty = lots * lot_size
        theta_per_day_rs = theta_per_day * total_qty

        return {
            'theta_per_day': round(theta_per_day, 4),
            'theta_per_day_rs': round(theta_per_day_rs, 2),
        }

    # ================================================================
    # PORTFOLIO-LEVEL THETA
    # ================================================================

    def compute_portfolio_theta(
        self,
        positions: List[Dict[str, Any]],
        spot: float,
    ) -> dict:
        """
        Aggregate theta across all open positions.

        Each position dict must contain:
            strike, dte, iv, option_type, action, lots
            (and optionally lot_size, signal_id)

        Returns:
            {
                'net_theta_per_day': float,
                'net_theta_rs': float,
                'positions': [per-position detail],
                'is_net_seller': bool,
                'alert': str | None,
            }
        """
        net_theta = 0.0
        net_theta_rs = 0.0
        pos_details: List[dict] = []

        for pos in positions:
            lot_size = pos.get('lot_size', NIFTY_LOT_SIZE)
            result = self.compute_position_theta(
                spot=spot,
                strike=pos['strike'],
                dte=pos['dte'],
                iv=pos['iv'],
                option_type=pos['option_type'],
                action=pos['action'],
                lots=pos['lots'],
                lot_size=lot_size,
            )
            net_theta += result['theta_per_day']
            net_theta_rs += result['theta_per_day_rs']

            pos_details.append({
                'signal_id': pos.get('signal_id', 'unknown'),
                'strike': pos['strike'],
                'option_type': pos['option_type'],
                'action': pos['action'],
                **result,
            })

        alert: Optional[str] = None
        is_net_seller = net_theta_rs > 0

        if not is_net_seller:
            alert = (
                f"ALERT: Portfolio net theta is NEGATIVE ({net_theta_rs:+.2f} Rs/day). "
                "You are paying theta — review positions."
            )
            logger.warning(alert)

        return {
            'net_theta_per_day': round(net_theta, 4),
            'net_theta_rs': round(net_theta_rs, 2),
            'positions': pos_details,
            'is_net_seller': is_net_seller,
            'alert': alert,
        }

    # ================================================================
    # THETA DIVERGENCE CHECK
    # ================================================================

    def check_theta_divergence(
        self,
        position: Dict[str, Any],
        actual_pnl: float,
        days_held: int,
    ) -> dict:
        """
        Compare actual P&L against theoretical theta-based P&L.

        Divergence is flagged when the actual P&L deviates from the
        expected theta-driven P&L by more than 2x the expected amount.

        Returns:
            {
                'diverged': bool,
                'actual_pnl': float,
                'expected_theta_pnl': float,
                'ratio': float,
                'alert': str,
            }
        """
        theta_per_day_rs = position.get('theta_per_day_rs', 0.0)
        expected_theta_pnl = theta_per_day_rs * days_held

        if abs(expected_theta_pnl) < 1e-6:
            ratio = 0.0
            diverged = abs(actual_pnl) > 100  # minimal fallback
        else:
            ratio = actual_pnl / expected_theta_pnl if expected_theta_pnl != 0 else 0.0
            diverged = abs(actual_pnl - expected_theta_pnl) > 2.0 * abs(expected_theta_pnl)

        alert = ""
        if diverged:
            alert = (
                f"THETA DIVERGENCE: actual P&L {actual_pnl:+.2f} vs "
                f"expected theta P&L {expected_theta_pnl:+.2f} "
                f"(ratio={ratio:.2f}x) over {days_held}d. "
                "Delta/gamma moves may dominate — review hedge."
            )
            logger.warning(alert)

        return {
            'diverged': diverged,
            'actual_pnl': round(actual_pnl, 2),
            'expected_theta_pnl': round(expected_theta_pnl, 2),
            'ratio': round(ratio, 2),
            'alert': alert,
        }

    # ================================================================
    # EXIT CONDITIONS
    # ================================================================

    def check_exit_conditions(
        self,
        position: Dict[str, Any],
        current_pnl: float,
        credit_received: float,
        dte: float,
    ) -> dict:
        """
        Evaluate whether a position should be closed.

        Conditions checked:
          1. 50% profit target reached
          2. DTE <= 5: gamma risk warning
          3. Theta/day < 0.1% of premium: diminishing returns

        Returns:
            {'should_exit': bool, 'reason': str, 'details': dict}
        """
        reasons: List[str] = []
        details: Dict[str, Any] = {}

        # --- 50 % profit target ---
        if credit_received > 0 and current_pnl >= 0.50 * credit_received:
            reasons.append('PROFIT_TARGET')
            details['profit_target'] = {
                'current_pnl': round(current_pnl, 2),
                'target': round(0.50 * credit_received, 2),
                'pct_captured': round(current_pnl / credit_received * 100, 1),
            }

        # --- Gamma risk (DTE <= 5) ---
        if dte <= 5:
            reasons.append('GAMMA_RISK')
            details['gamma_risk'] = {
                'dte': dte,
                'message': 'DTE <= 5 — gamma risk elevated, consider closing.',
            }

        # --- Diminishing theta returns ---
        theta_per_day_rs = position.get('theta_per_day_rs', 0.0)
        if credit_received > 0:
            theta_pct = abs(theta_per_day_rs) / credit_received
            if theta_pct < 0.001:  # < 0.1 % of premium per day
                reasons.append('DIMINISHING_THETA')
                details['diminishing_theta'] = {
                    'theta_per_day_rs': round(theta_per_day_rs, 2),
                    'credit_received': round(credit_received, 2),
                    'theta_pct_of_premium': round(theta_pct * 100, 4),
                }

        should_exit = len(reasons) > 0
        reason = ', '.join(reasons) if reasons else 'HOLD'

        return {
            'should_exit': should_exit,
            'reason': reason,
            'details': details,
        }

    # ================================================================
    # DAILY THETA LOG (DB)
    # ================================================================

    def log_daily_theta(
        self,
        db_conn,
        as_of: date,
        positions: List[Dict[str, Any]],
        spot: float,
    ) -> None:
        """
        Insert daily theta snapshot into `theta_daily_log` table.

        Gracefully skips if the table does not exist.

        Columns: date, signal_id, theta_per_day, cumulative_theta,
                 days_held, pnl_vs_theta
        """
        portfolio = self.compute_portfolio_theta(positions, spot)

        try:
            cur = db_conn.cursor()
            # Check table existence
            cur.execute(
                "SELECT to_regclass('public.theta_daily_log')"
            )
            if cur.fetchone()[0] is None:
                logger.info("theta_daily_log table does not exist — skipping log.")
                cur.close()
                return

            for pos_detail in portfolio['positions']:
                signal_id = pos_detail.get('signal_id', 'unknown')
                theta_per_day = pos_detail.get('theta_per_day', 0.0)

                # Pull running context from the position dicts
                matched = [p for p in positions if p.get('signal_id') == signal_id]
                if matched:
                    days_held = matched[0].get('days_held', 0)
                    actual_pnl = matched[0].get('actual_pnl', 0.0)
                    cumulative_theta = theta_per_day * days_held
                    pnl_vs_theta = actual_pnl - cumulative_theta
                else:
                    days_held = 0
                    cumulative_theta = 0.0
                    pnl_vs_theta = 0.0

                cur.execute(
                    """
                    INSERT INTO theta_daily_log
                        (date, signal_id, theta_per_day, cumulative_theta,
                         days_held, pnl_vs_theta)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (as_of, signal_id, theta_per_day, cumulative_theta,
                     days_held, pnl_vs_theta),
                )

            db_conn.commit()
            cur.close()
            logger.info("Logged daily theta for %d positions as of %s.",
                        len(portfolio['positions']), as_of)

        except Exception as exc:
            logger.error("Failed to log daily theta: %s", exc)
            try:
                db_conn.rollback()
            except Exception:
                pass

    # ================================================================
    # STATUS SUMMARY
    # ================================================================

    def get_status(self) -> dict:
        """
        Return a summary dict of the tracker's internal state.
        """
        tracked_ids = list(self._positions.keys())
        return {
            'tracked_count': len(tracked_ids),
            'tracked_signal_ids': tracked_ids,
            'risk_free_rate': self.risk_free_rate,
            'has_db': self.db_conn is not None,
        }

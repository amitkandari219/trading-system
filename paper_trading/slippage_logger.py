"""
Slippage Logger: tracks gap between paper trading fills and estimated live fills.
Builds a paper-to-live adjustment model over time.
"""

import logging
from datetime import date
from typing import Optional

import psycopg2

from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE

logger = logging.getLogger(__name__)

BROKERAGE_PER_LOT = 300  # ₹ approximate round-trip per lot


class SlippageLogger:

    def __init__(self, db_conn=None):
        self.conn = db_conn or psycopg2.connect(DATABASE_DSN)

    def log_paper_trade(self, signal_id: str, direction: str,
                        theoretical_fill: float, actual_open: float,
                        atr_14: float, vix: float, lots: int):
        """Log gap between paper fill and actual next-day open."""
        gap_pts = actual_open - theoretical_fill
        gap_pct = gap_pts / theoretical_fill if theoretical_fill else 0

        # For LONG: positive gap = adverse (paid more)
        # For SHORT: negative gap = adverse (sold lower)
        if direction == 'SHORT':
            gap_pts = -gap_pts
            gap_pct = -gap_pct

        est_slippage = atr_14 * 0.05 if atr_14 else 1.0
        adverse_sel = 1.0
        brokerage_pts = BROKERAGE_PER_LOT / (NIFTY_LOT_SIZE * max(lots, 1))
        total_cost = est_slippage + adverse_sel + brokerage_pts

        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO slippage_log
                    (signal_id, trade_date, direction, theoretical_fill,
                     actual_open, gap_pct, gap_pts, atr_14, vix, lots,
                     est_slippage_pts, adverse_sel_pts, total_cost_pts)
                VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (signal_id, direction, theoretical_fill, actual_open,
                  gap_pct, gap_pts, atr_14, vix, lots,
                  est_slippage, adverse_sel, total_cost))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to log slippage: {e}")
            self.conn.rollback()

    def get_paper_to_live_model(self, signal_id: str = None,
                                 min_trades: int = 20) -> dict:
        """Compute paper-to-live adjustment after enough trades logged."""
        try:
            cur = self.conn.cursor()
            where = "WHERE signal_id = %s" if signal_id else ""
            params = (signal_id,) if signal_id else ()

            cur.execute(f"""
                SELECT signal_id, COUNT(*) as n,
                       AVG(gap_pct) as avg_gap_pct,
                       AVG(gap_pts) as avg_gap_pts,
                       AVG(total_cost_pts) as avg_cost_pts
                FROM slippage_log {where}
                GROUP BY signal_id
            """, params)

            rows = cur.fetchall()
            if not rows:
                return {'trades_logged': 0, 'confidence': 'LOW'}

            results = []
            for row in rows:
                sid, n, avg_gap_pct, avg_gap_pts, avg_cost_pts = row

                if n < min_trades:
                    results.append({
                        'signal_id': sid, 'trades_logged': n,
                        'confidence': 'LOW',
                    })
                    continue

                # Estimate annual cost drag
                # Assume ~20-60 trades/year depending on signal
                trades_per_year = n * 252 / max(1, (self._days_of_data(sid) or 252))
                nifty_price = 23000  # approximate
                cost_per_trade_pct = avg_cost_pts / nifty_price
                annual_cost_drag = cost_per_trade_pct * trades_per_year
                slippage_factor = max(0, 1 - annual_cost_drag)

                confidence = 'HIGH' if n >= 50 else 'MEDIUM' if n >= min_trades else 'LOW'

                results.append({
                    'signal_id': sid,
                    'trades_logged': n,
                    'avg_gap_pct': round(avg_gap_pct, 6),
                    'avg_gap_pts': round(avg_gap_pts, 2),
                    'avg_cost_pts': round(avg_cost_pts, 2),
                    'annual_cost_drag_pct': round(annual_cost_drag * 100, 2),
                    'slippage_factor': round(slippage_factor, 4),
                    'confidence': confidence,
                })

            return results[0] if signal_id and results else results

        except Exception as e:
            logger.error(f"Failed to compute slippage model: {e}")
            return {'trades_logged': 0, 'confidence': 'LOW'}

    def _days_of_data(self, signal_id: str) -> int:
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT MAX(trade_date) - MIN(trade_date)
                FROM slippage_log WHERE signal_id = %s
            """, (signal_id,))
            row = cur.fetchone()
            return row[0].days if row and row[0] else 0
        except Exception:
            return 0

    def get_portfolio_slippage_model(self) -> dict:
        """Aggregate slippage model across all signals."""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT COUNT(*) as n,
                       AVG(gap_pct) as avg_gap_pct,
                       AVG(total_cost_pts) as avg_cost_pts
                FROM slippage_log
            """)
            row = cur.fetchone()
            if not row or row[0] == 0:
                return {'trades_logged': 0, 'confidence': 'LOW'}

            n, avg_gap_pct, avg_cost_pts = row
            nifty_price = 23000
            cost_pct = avg_cost_pts / nifty_price
            # Assume portfolio trades ~130/year (combined signals)
            annual_drag = cost_pct * 130
            factor = max(0, 1 - annual_drag)

            return {
                'trades_logged': n,
                'avg_gap_pct': round(avg_gap_pct, 6),
                'avg_cost_pts': round(avg_cost_pts, 2),
                'annual_cost_drag_pct': round(annual_drag * 100, 2),
                'slippage_factor': round(factor, 4),
                'confidence': 'HIGH' if n >= 100 else 'MEDIUM' if n >= 30 else 'LOW',
            }
        except Exception as e:
            logger.error(f"Failed to compute portfolio slippage: {e}")
            return {'trades_logged': 0, 'confidence': 'LOW'}

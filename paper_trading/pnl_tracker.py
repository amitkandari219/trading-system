"""
Paper trading P&L tracker.

Queries the trades table for PAPER trades, computes daily/weekly/monthly P&L,
tracks per-signal performance, and generates summary reports.

Usage:
    python -m paper_trading.pnl_tracker              # daily summary
    python -m paper_trading.pnl_tracker --detailed   # per-signal breakdown
    python -m paper_trading.pnl_tracker --history 30  # last 30 days
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Dict, Optional

import psycopg2

from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE

logger = logging.getLogger(__name__)


@dataclass
class SignalStats:
    signal_id: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    long_trades: int = 0
    long_wins: int = 0
    long_pnl: float = 0.0
    short_trades: int = 0
    short_wins: int = 0
    short_pnl: float = 0.0
    avg_hold_days: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    open_positions: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0

    @property
    def long_win_rate(self) -> float:
        return self.long_wins / self.long_trades if self.long_trades > 0 else 0

    @property
    def short_win_rate(self) -> float:
        return self.short_wins / self.short_trades if self.short_trades > 0 else 0


@dataclass
class DailyPnL:
    date: date
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    trades_closed: int = 0
    trades_opened: int = 0


class PnLTracker:
    """Tracks paper trading P&L from the trades table."""

    def __init__(self, db_conn=None):
        self.conn = db_conn or psycopg2.connect(DATABASE_DSN)

    def daily_summary(self, as_of: date = None) -> Dict:
        """Get P&L summary for a specific date."""
        as_of = as_of or date.today()
        cur = self.conn.cursor()

        # Trades closed today
        cur.execute("""
            SELECT signal_id, direction, entry_price, exit_price,
                   gross_pnl, net_pnl, return_pct, exit_reason,
                   entry_date
            FROM trades
            WHERE trade_type = 'PAPER' AND exit_date = %s
            ORDER BY signal_id
        """, (as_of,))
        closed_today = cur.fetchall()

        # Trades opened today
        cur.execute("""
            SELECT signal_id, direction, entry_price
            FROM trades
            WHERE trade_type = 'PAPER' AND entry_date = %s
            ORDER BY signal_id
        """, (as_of,))
        opened_today = cur.fetchall()

        # Open positions
        cur.execute("""
            SELECT signal_id, direction, entry_price, entry_date
            FROM trades
            WHERE trade_type = 'PAPER' AND exit_date IS NULL
            ORDER BY entry_date
        """)
        open_positions = cur.fetchall()

        # Day's P&L (points)
        day_pnl = sum(r[4] or 0 for r in closed_today)
        # P&L in rupees (1 lot of Nifty futures)
        day_pnl_rs = day_pnl * NIFTY_LOT_SIZE

        return {
            'date': str(as_of),
            'closed_trades': [{
                'signal_id': r[0], 'direction': r[1],
                'entry_price': r[2], 'exit_price': r[3],
                'gross_pnl_pts': r[4], 'return_pct': r[6],
                'exit_reason': r[7],
                'hold_days': (as_of - r[8]).days if r[8] else 0,
            } for r in closed_today],
            'opened_trades': [{
                'signal_id': r[0], 'direction': r[1], 'entry_price': r[2],
            } for r in opened_today],
            'open_positions': [{
                'signal_id': r[0], 'direction': r[1],
                'entry_price': r[2], 'entry_date': str(r[3]),
                'days_held': (as_of - r[3]).days if r[3] else 0,
            } for r in open_positions],
            'day_pnl_pts': round(day_pnl, 2),
            'day_pnl_rs': round(day_pnl_rs, 2),
            'trades_closed': len(closed_today),
            'trades_opened': len(opened_today),
            'positions_open': len(open_positions),
        }

    def signal_stats(self, since: date = None) -> Dict[str, SignalStats]:
        """Get per-signal statistics."""
        cur = self.conn.cursor()

        where_clause = "WHERE trade_type = 'PAPER' AND exit_date IS NOT NULL"
        params = ()
        if since:
            where_clause += " AND entry_date >= %s"
            params = (since,)

        cur.execute(f"""
            SELECT signal_id, direction, gross_pnl, entry_date, exit_date
            FROM trades
            {where_clause}
            ORDER BY signal_id, entry_date
        """, params)
        closed = cur.fetchall()

        # Open positions
        cur.execute("""
            SELECT signal_id FROM trades
            WHERE trade_type = 'PAPER' AND exit_date IS NULL
        """)
        open_pos = cur.fetchall()
        open_counts = {}
        for r in open_pos:
            open_counts[r[0]] = open_counts.get(r[0], 0) + 1

        stats = {}
        for r in closed:
            sid, direction, pnl, entry_dt, exit_dt = r
            if sid not in stats:
                stats[sid] = SignalStats(signal_id=sid)
            s = stats[sid]

            s.total_trades += 1
            s.total_pnl += pnl or 0
            hold = (exit_dt - entry_dt).days if entry_dt and exit_dt else 0

            if pnl and pnl > 0:
                s.wins += 1
                s.max_win = max(s.max_win, pnl)
            elif pnl and pnl < 0:
                s.losses += 1
                s.max_loss = min(s.max_loss, pnl)

            if direction == 'LONG':
                s.long_trades += 1
                s.long_pnl += pnl or 0
                if pnl and pnl > 0:
                    s.long_wins += 1
            else:
                s.short_trades += 1
                s.short_pnl += pnl or 0
                if pnl and pnl > 0:
                    s.short_wins += 1

            # Running average hold days
            s.avg_hold_days = ((s.avg_hold_days * (s.total_trades - 1) + hold)
                               / s.total_trades)

        # Add open position counts
        for sid, count in open_counts.items():
            if sid not in stats:
                stats[sid] = SignalStats(signal_id=sid)
            stats[sid].open_positions = count

        return stats

    def pnl_history(self, days: int = 30) -> List[DailyPnL]:
        """Get daily P&L history for the last N days."""
        cur = self.conn.cursor()
        since = date.today() - timedelta(days=days)

        cur.execute("""
            SELECT exit_date, SUM(gross_pnl), SUM(net_pnl), COUNT(*)
            FROM trades
            WHERE trade_type = 'PAPER'
              AND exit_date >= %s
              AND exit_date IS NOT NULL
            GROUP BY exit_date
            ORDER BY exit_date
        """, (since,))

        history = []
        for r in cur.fetchall():
            history.append(DailyPnL(
                date=r[0],
                gross_pnl=r[1] or 0,
                net_pnl=r[2] or 0,
                trades_closed=r[3],
            ))

        return history

    def cumulative_pnl(self, since: date = None) -> Dict:
        """Get cumulative P&L metrics."""
        cur = self.conn.cursor()

        where = "WHERE trade_type = 'PAPER' AND exit_date IS NOT NULL"
        params = ()
        if since:
            where += " AND entry_date >= %s"
            params = (since,)

        cur.execute(f"""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN gross_pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(gross_pnl) as total_pnl,
                AVG(gross_pnl) as avg_pnl,
                MAX(gross_pnl) as max_win,
                MIN(gross_pnl) as max_loss,
                SUM(CASE WHEN gross_pnl > 0 THEN gross_pnl ELSE 0 END) as gross_wins,
                SUM(CASE WHEN gross_pnl < 0 THEN ABS(gross_pnl) ELSE 0 END) as gross_losses
            FROM trades
            {where}
        """, params)
        r = cur.fetchone()

        total = r[0] or 0
        wins = r[1] or 0
        total_pnl = r[2] or 0
        gross_wins = r[6] or 0
        gross_losses = r[7] or 1e-9

        return {
            'total_trades': total,
            'wins': wins,
            'losses': total - wins,
            'win_rate': wins / total if total > 0 else 0,
            'total_pnl_pts': round(total_pnl, 2),
            'total_pnl_rs': round(total_pnl * NIFTY_LOT_SIZE, 2),
            'avg_pnl_pts': round((r[3] or 0), 2),
            'max_win_pts': round((r[4] or 0), 2),
            'max_loss_pts': round((r[5] or 0), 2),
            'profit_factor': round(gross_wins / gross_losses, 2) if gross_losses > 0 else 0,
        }

    def print_report(self, detailed: bool = False, history_days: int = 0):
        """Print formatted P&L report to console."""
        today = date.today()

        # Daily summary
        daily = self.daily_summary(today)
        print(f"\n{'='*60}")
        print(f"PAPER TRADING REPORT — {today}")
        print(f"{'='*60}")
        print(f"Day P&L:    {daily['day_pnl_pts']:+.2f} pts (₹{daily['day_pnl_rs']:+,.0f})")
        print(f"Closed:     {daily['trades_closed']}  Opened: {daily['trades_opened']}  "
              f"Open: {daily['positions_open']}")

        if daily['closed_trades']:
            print(f"\nClosed today:")
            for t in daily['closed_trades']:
                print(f"  {t['signal_id']:25s} {t['direction']:5s} "
                      f"{t['gross_pnl_pts']:+7.1f}pts ({t['return_pct']:+.2%}) "
                      f"{t['exit_reason']} [{t['hold_days']}d]")

        if daily['open_positions']:
            print(f"\nOpen positions:")
            for p in daily['open_positions']:
                print(f"  {p['signal_id']:25s} {p['direction']:5s} "
                      f"@ {p['entry_price']:.0f} [{p['days_held']}d held]")

        # Cumulative
        cumul = self.cumulative_pnl()
        print(f"\nCumulative:")
        print(f"  Total trades:  {cumul['total_trades']}")
        print(f"  Win rate:      {cumul['win_rate']:.1%}")
        print(f"  Total P&L:     {cumul['total_pnl_pts']:+.2f} pts (₹{cumul['total_pnl_rs']:+,.0f})")
        print(f"  Profit factor: {cumul['profit_factor']:.2f}")

        if detailed:
            stats = self.signal_stats()
            if stats:
                print(f"\n{'='*60}")
                print(f"PER-SIGNAL BREAKDOWN")
                print(f"{'='*60}")
                print(f"{'Signal':25s} {'Trades':>6s} {'WR':>6s} {'P&L':>8s} "
                      f"{'L-WR':>6s} {'S-WR':>6s} {'Hold':>5s}")
                print(f"{'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*5}")
                for sid, s in sorted(stats.items()):
                    print(f"{s.signal_id:25s} {s.total_trades:6d} "
                          f"{s.win_rate:5.0%} {s.total_pnl:+8.1f} "
                          f"{s.long_win_rate:5.0%} {s.short_win_rate:5.0%} "
                          f"{s.avg_hold_days:5.1f}d")

        if history_days > 0:
            history = self.pnl_history(history_days)
            if history:
                print(f"\n{'='*60}")
                print(f"DAILY P&L HISTORY (last {history_days} days)")
                print(f"{'='*60}")
                cumul_pnl = 0
                for d in history:
                    cumul_pnl += d.gross_pnl
                    print(f"  {d.date}  {d.gross_pnl:+8.1f}pts  "
                          f"cumul={cumul_pnl:+8.1f}  trades={d.trades_closed}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Paper trading P&L tracker')
    parser.add_argument('--detailed', action='store_true', help='Per-signal breakdown')
    parser.add_argument('--history', type=int, default=0, help='Show N days of P&L history')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    tracker = PnLTracker()
    tracker.print_report(detailed=args.detailed, history_days=args.history)


if __name__ == '__main__':
    main()

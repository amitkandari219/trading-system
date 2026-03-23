"""
Entry Window Optimizer — restricts signal entries to historically profitable time windows.

Intraday signals fire throughout 9:15-15:10, but not all hours are equal.
Some signals perform better at certain times (ORB in morning, EOD trend in afternoon).
The dead zone (12:00-13:30) typically has lower volume and worse signal quality.

This module analyzes historical trades by entry hour and restricts entries
to windows where each signal has demonstrated positive expectancy.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import time, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trading session constants
# ---------------------------------------------------------------------------
SESSION_START = time(9, 15)
SESSION_END = time(15, 10)
TRADING_HOURS = [9, 10, 11, 12, 13, 14, 15]
DEFAULT_DEAD_ZONE = [12, 13]  # lunch hours — low volume, poor fills

MIN_TRADES_PER_BUCKET = 10
PROFITABLE_PF_THRESHOLD = 1.0
DEAD_ZONE_PF_THRESHOLD = 0.8


class EntryWindowOptimizer:
    """Analyzes historical trade performance by hour and restricts entries
    to time windows where each signal has positive expectancy."""

    # Pre-configured windows based on Al Brooks price action theory.
    # Tuples are (start_time, end_time) — entry is allowed within this range
    # unless blocked by hourly analysis.
    EXPECTED_WINDOWS = {
        'L9_ORB_BREAKOUT':    (time(9, 30), time(10, 30)),
        'L9_VWAP_RECLAIM':    (time(9, 45), time(14, 30)),
        'L9_VWAP_REJECTION':  (time(9, 45), time(14, 30)),
        'L9_FIRST_PULLBACK':  (time(9, 30), time(13, 0)),
        'L9_FAILED_BREAKOUT': (time(9, 45), time(14, 0)),
        'L9_DOUBLE_BOTTOM':   (time(10, 0), time(14, 30)),
        'L9_DOUBLE_TOP':      (time(10, 0), time(14, 30)),
        'L9_GAP_FILL':        (time(9, 20), time(11, 0)),
        'L9_TREND_BAR':       (time(10, 0), time(14, 30)),
        'L9_EOD_TREND':       (time(14, 0), time(15, 10)),
        # BN signals — same windows (BN follows Nifty session structure)
        'BN_ORB_BREAKOUT':    (time(9, 30), time(10, 30)),
        'BN_VWAP_RECLAIM':    (time(9, 45), time(14, 30)),
        'BN_VWAP_REJECTION':  (time(9, 45), time(14, 30)),
        'BN_FIRST_PULLBACK':  (time(9, 30), time(13, 0)),
        'BN_FAILED_BREAKOUT': (time(9, 45), time(14, 0)),
        'BN_GAP_FILL':        (time(9, 20), time(11, 0)),
        'BN_TREND_BAR':       (time(10, 0), time(14, 30)),
        'BN_EOD_TREND':       (time(14, 0), time(15, 10)),
    }

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        # signal_id -> {start_time, end_time, blocked_hours}
        self._windows: dict[str, dict[str, Any]] = {}

        # Populate defaults from EXPECTED_WINDOWS
        for sig_id, (start, end) in self.EXPECTED_WINDOWS.items():
            self._windows[sig_id] = {
                'start_time': start,
                'end_time': end,
                'blocked_hours': list(DEFAULT_DEAD_ZONE),
            }

        # signal_id -> {hour: {pf, win_rate, avg_pnl, count}}
        self._hourly_stats: dict[str, dict[int, dict[str, float]]] = {}
        self._analysis_done: set[str] = set()

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------
    def analyze_trades(self, trades_by_signal: dict[str, list[dict]]) -> None:
        """Analyze historical trades bucketed by entry hour.

        Parameters
        ----------
        trades_by_signal : dict
            signal_id -> list of trade dicts.  Each trade must have:
                entry_time : datetime or str (ISO format)
                pnl        : float (absolute P&L)
                pnl_pct    : float (percent P&L)
        """
        for signal_id, trades in trades_by_signal.items():
            if not trades:
                continue

            buckets: dict[int, list[dict]] = defaultdict(list)
            for t in trades:
                entry = t.get('entry_time')
                if entry is None:
                    continue
                if isinstance(entry, str):
                    entry = datetime.fromisoformat(entry)
                hour = entry.hour
                if hour in TRADING_HOURS:
                    buckets[hour].append(t)

            hourly: dict[int, dict[str, float]] = {}
            for hour in TRADING_HOURS:
                bucket = buckets.get(hour, [])
                count = len(bucket)
                if count == 0:
                    hourly[hour] = {'pf': 0.0, 'win_rate': 0.0,
                                    'avg_pnl': 0.0, 'count': 0}
                    continue

                pnls = [t.get('pnl', 0.0) for t in bucket]
                wins = [p for p in pnls if p > 0]
                losses = [p for p in pnls if p <= 0]

                gross_profit = sum(wins) if wins else 0.0
                gross_loss = abs(sum(losses)) if losses else 0.0
                pf = gross_profit / gross_loss if gross_loss > 0 else (
                    float('inf') if gross_profit > 0 else 0.0)

                win_rate = len(wins) / count
                avg_pnl = sum(pnls) / count

                hourly[hour] = {
                    'pf': round(pf, 3),
                    'win_rate': round(win_rate, 4),
                    'avg_pnl': round(avg_pnl, 2),
                    'count': count,
                }

            self._hourly_stats[signal_id] = hourly

            # Derive optimal window from stats
            profitable_hours = sorted([
                h for h, s in hourly.items()
                if s['pf'] > PROFITABLE_PF_THRESHOLD
                and s['count'] >= MIN_TRADES_PER_BUCKET
            ])
            dead_hours = sorted([
                h for h, s in hourly.items()
                if s['pf'] < DEAD_ZONE_PF_THRESHOLD
                or s['count'] < MIN_TRADES_PER_BUCKET
            ])

            if profitable_hours:
                # Optimal = contiguous block starting from earliest profitable hour
                start_h = profitable_hours[0]
                end_h = profitable_hours[-1]
                start_t = time(start_h, 15) if start_h == 9 else time(start_h, 0)
                end_t = time(end_h, 59) if end_h < 15 else time(15, 10)
            elif signal_id in self.EXPECTED_WINDOWS:
                # Fall back to Al Brooks defaults
                start_t, end_t = self.EXPECTED_WINDOWS[signal_id]
            else:
                start_t, end_t = SESSION_START, SESSION_END

            self._windows[signal_id] = {
                'start_time': start_t,
                'end_time': end_t,
                'blocked_hours': dead_hours,
            }
            self._analysis_done.add(signal_id)

            logger.info(
                "EntryWindowOptimizer | %s | window=%s-%s | blocked=%s | "
                "profitable_hours=%s",
                signal_id, start_t, end_t, dead_hours, profitable_hours,
            )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------
    def get_optimal_window(self, signal_id: str) -> dict[str, Any]:
        """Return the optimal entry window for a signal.

        Returns dict with keys: start_time, end_time, blocked_hours, pf_by_hour.
        """
        window = self._windows.get(signal_id)
        if window is None:
            # Unknown signal — return full session
            return {
                'start_time': SESSION_START,
                'end_time': SESSION_END,
                'blocked_hours': list(DEFAULT_DEAD_ZONE),
                'pf_by_hour': {},
            }

        return {
            'start_time': window['start_time'],
            'end_time': window['end_time'],
            'blocked_hours': window['blocked_hours'],
            'pf_by_hour': self._hourly_stats.get(signal_id, {}),
        }

    def is_entry_allowed(self, signal_id: str, current_time: time | datetime) -> bool:
        """Check if entry is allowed for *signal_id* at *current_time*.

        Returns True if:
          1. current_time is within the signal's optimal window, AND
          2. current_time's hour is not in the blocked list.

        If no analysis has been done for this signal, defaults to True
        (permissive until data is available).
        """
        if isinstance(current_time, datetime):
            current_time = current_time.time()

        window = self._windows.get(signal_id)
        if window is None:
            # No config and no analysis — allow by default
            return True

        start = window['start_time']
        end = window['end_time']

        # Outside window -> blocked
        if current_time < start or current_time > end:
            return False

        # In a blocked hour -> blocked (only if analysis has been performed)
        if signal_id in self._analysis_done:
            if current_time.hour in window.get('blocked_hours', []):
                return False

        return True

    def detect_dead_zones(self, min_pf: float = 0.8) -> dict[str, list[int]]:
        """Find hours where each analyzed signal has PF below *min_pf*.

        Returns {signal_id: [dead_hour_ints]}.
        """
        result: dict[str, list[int]] = {}
        for signal_id, hourly in self._hourly_stats.items():
            dead = sorted([
                h for h, s in hourly.items()
                if s['pf'] < min_pf or s['count'] < MIN_TRADES_PER_BUCKET
            ])
            if dead:
                result[signal_id] = dead
        return result

    # ------------------------------------------------------------------
    # Status / reporting
    # ------------------------------------------------------------------
    def get_status(self) -> dict[str, Any]:
        """Per-signal summary of current windows and analysis state."""
        status: dict[str, Any] = {}
        all_signals = set(self._windows.keys()) | set(self._hourly_stats.keys())

        for sig_id in sorted(all_signals):
            w = self._windows.get(sig_id, {})
            hourly = self._hourly_stats.get(sig_id, {})
            analyzed = sig_id in self._analysis_done

            profitable_hours = sorted([
                h for h, s in hourly.items()
                if s['pf'] > PROFITABLE_PF_THRESHOLD
                and s['count'] >= MIN_TRADES_PER_BUCKET
            ]) if hourly else []

            dead_hours = sorted([
                h for h, s in hourly.items()
                if s['pf'] < DEAD_ZONE_PF_THRESHOLD
            ]) if hourly else []

            total_trades = sum(s.get('count', 0) for s in hourly.values())

            status[sig_id] = {
                'start_time': str(w.get('start_time', SESSION_START)),
                'end_time': str(w.get('end_time', SESSION_END)),
                'blocked_hours': w.get('blocked_hours', []),
                'analyzed': analyzed,
                'total_trades': total_trades,
                'profitable_hours': profitable_hours,
                'dead_hours': dead_hours,
                'pf_by_hour': {
                    h: s['pf'] for h, s in hourly.items()
                } if hourly else {},
            }
        return status

    def __repr__(self) -> str:
        analyzed = len(self._analysis_done)
        configured = len(self._windows)
        return (f"EntryWindowOptimizer(configured={configured}, "
                f"analyzed={analyzed})")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def _format_time(t: time) -> str:
    return t.strftime('%H:%M')


def main() -> None:
    optimizer = EntryWindowOptimizer()

    print("=" * 72)
    print("ENTRY WINDOW OPTIMIZER — Default Signal Windows")
    print("=" * 72)
    print(f"{'Signal':<25} {'Window':>15}  {'Blocked Hours'}")
    print("-" * 72)

    status = optimizer.get_status()
    for sig_id, info in status.items():
        start = info['start_time']
        end = info['end_time']
        blocked = info['blocked_hours']
        window_str = f"{start}-{end}"
        blocked_str = ', '.join(f"{h}:00" for h in blocked) if blocked else '—'
        tag = ''
        if info.get('analyzed'):
            tag = f"  [analyzed, {info['total_trades']} trades]"
        print(f"{sig_id:<25} {window_str:>15}  {blocked_str}{tag}")

    print("-" * 72)
    print(f"Total signals configured: {len(status)}")
    print(f"Session: {_format_time(SESSION_START)} - {_format_time(SESSION_END)}")
    print(f"Default dead zone: {', '.join(f'{h}:00' for h in DEFAULT_DEAD_ZONE)}")
    print("=" * 72)


if __name__ == '__main__':
    main()

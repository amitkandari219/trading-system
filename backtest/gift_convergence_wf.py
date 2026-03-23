"""
GIFT Nifty → NSE Convergence Signal Backtest.

Thesis: GIFT Nifty trades overnight on SGX/NSE IFSC. When NSE opens at 9:15,
the opening price often overshoots or undershoots where GIFT closed. This
deviation converges within 15-30 minutes as arbitrageurs close the gap.

We don't have direct GIFT Nifty data, but we CAN measure:
  gap = (NSE 9:15 open) - (previous NSE close)
This captures the same overnight deviation that GIFT Nifty represents.

Strategy:
  - If gap > +50 pts: SHORT at 9:15 (gap will compress)
  - If gap < -50 pts: LONG at 9:15 (gap will fill)
  - Exit at 9:30 (15 min), or SL at gap widens 50% more
  - Larger gaps (>100 pts) get higher conviction

Data: 493 real 9:15 AM bars from Kite (Mar 2024 - Mar 2026)

Usage:
    venv/bin/python3 -m backtest.gift_convergence_wf
"""

import logging
import math
import time as time_mod
from datetime import date, time, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import psycopg2

from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE

logger = logging.getLogger(__name__)

# Signal parameters
MIN_GAP_PTS = 50        # minimum gap to trade
LARGE_GAP_PTS = 100     # larger gap = higher conviction
SL_EXTENSION_PCT = 0.50 # SL if gap extends 50% more
EXIT_BARS = 3           # exit after 3 bars (15 min) from 9:15
SLIPPAGE_PTS = 1.0      # per side

# WF params
TRAIN_DAYS = 126   # 6 months
TEST_DAYS = 42     # 2 months
STEP_DAYS = 21     # 1 month


def load_data():
    conn = psycopg2.connect(DATABASE_DSN)

    # Load all 5-min bars
    bars = pd.read_sql(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM intraday_bars WHERE instrument='NIFTY' ORDER BY timestamp",
        conn, parse_dates=['timestamp'])
    bars['date'] = bars['timestamp'].dt.date
    bars['time'] = bars['timestamp'].dt.time

    # Load daily closes for prev_close
    daily = pd.read_sql(
        "SELECT date, close FROM nifty_daily ORDER BY date",
        conn, parse_dates=['date'])
    daily['date'] = daily['date'].dt.date

    conn.close()
    return bars, daily


def simulate_gift_convergence(bars: pd.DataFrame, daily: pd.DataFrame) -> List[Dict]:
    """Simulate the GIFT convergence strategy on all trading days."""
    trades = []

    # Build prev_close lookup
    daily_sorted = daily.sort_values('date')
    prev_close_map = {}
    dates = daily_sorted['date'].tolist()
    for i in range(1, len(dates)):
        prev_close_map[dates[i]] = float(daily_sorted.iloc[i-1]['close'])

    trade_dates = sorted(bars['date'].unique())

    for d in trade_dates:
        prev_close = prev_close_map.get(d)
        if prev_close is None or prev_close <= 0:
            continue

        session = bars[bars['date'] == d].sort_values('timestamp')
        if len(session) < EXIT_BARS + 1:
            continue

        # 9:15 bar = entry bar
        first_bar = session.iloc[0]
        day_open = float(first_bar['open'])
        gap_pts = day_open - prev_close
        gap_pct = gap_pts / prev_close

        if abs(gap_pts) < MIN_GAP_PTS:
            continue  # gap too small

        # Direction: FADE the gap
        if gap_pts > 0:
            direction = 'SHORT'  # gap up → short, expect compression
            entry_price = day_open - SLIPPAGE_PTS  # sell at open minus slip
            sl_price = day_open + abs(gap_pts) * SL_EXTENSION_PCT  # gap widens 50%
        else:
            direction = 'LONG'  # gap down → long, expect fill
            entry_price = day_open + SLIPPAGE_PTS  # buy at open plus slip
            sl_price = day_open - abs(gap_pts) * SL_EXTENSION_PCT

        # Exit at 9:30 (bar index EXIT_BARS) or SL hit
        exit_price = entry_price
        exit_reason = 'TIME'
        exit_bar_idx = min(EXIT_BARS, len(session) - 1)

        for j in range(1, exit_bar_idx + 1):
            bar = session.iloc[j]
            h = float(bar['high'])
            l = float(bar['low'])
            c = float(bar['close'])

            # Check SL
            if direction == 'SHORT' and h >= sl_price:
                exit_price = sl_price + SLIPPAGE_PTS
                exit_reason = 'SL'
                break
            if direction == 'LONG' and l <= sl_price:
                exit_price = sl_price - SLIPPAGE_PTS
                exit_reason = 'SL'
                break

            # At exit bar, use close
            if j == exit_bar_idx:
                exit_price = c + (SLIPPAGE_PTS if direction == 'SHORT' else -SLIPPAGE_PTS)
                exit_reason = 'TIME_15MIN'

        # P&L
        if direction == 'LONG':
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price

        pnl_rs = pnl_pts * NIFTY_LOT_SIZE * 1  # 1 lot
        costs = 40 * 2 + abs(exit_price * NIFTY_LOT_SIZE * 0.000125)  # brokerage + STT
        net_pnl = pnl_rs - costs

        trades.append({
            'date': d,
            'direction': direction,
            'gap_pts': round(gap_pts, 0),
            'gap_pct': round(gap_pct * 100, 2),
            'entry': round(entry_price, 1),
            'exit': round(exit_price, 1),
            'pnl_pts': round(pnl_pts, 1),
            'pnl_rs': round(net_pnl),
            'exit_reason': exit_reason,
            'is_large_gap': abs(gap_pts) >= LARGE_GAP_PTS,
        })

    return trades


def main():
    t0 = time_mod.perf_counter()

    print("=" * 80)
    print("  GIFT NIFTY → NSE CONVERGENCE — Walk-Forward Backtest")
    print("  Thesis: Fade overnight gaps > 50pts, exit in 15 min")
    print("  Data: 493 real trading days (Mar 2024 - Mar 2026)")
    print("=" * 80)

    bars, daily = load_data()
    print(f"\nLoaded {len(bars)} bars, {len(bars['date'].unique())} days")

    trades = simulate_gift_convergence(bars, daily)
    print(f"Total trades: {len(trades)}")

    if not trades:
        print("No trades generated!")
        return

    # ── PER-DIRECTION BREAKDOWN ──
    print(f"\n{'─' * 80}")
    for direction in ['LONG', 'SHORT']:
        dt = [t for t in trades if t['direction'] == direction]
        if not dt:
            continue
        wins = [t for t in dt if t['pnl_rs'] > 0]
        wr = len(wins) / len(dt)
        total = sum(t['pnl_rs'] for t in dt)
        avg = np.mean([t['pnl_rs'] for t in dt])
        print(f"  {direction:5s}: {len(dt):3d} trades, WR {wr:.0%}, "
              f"avg ₹{avg:+,.0f}, total ₹{total:+,.0f}")

    # ── BY GAP SIZE ──
    print(f"\n  By gap size:")
    small = [t for t in trades if not t['is_large_gap']]
    large = [t for t in trades if t['is_large_gap']]
    for label, subset in [('50-100 pts', small), ('>100 pts', large)]:
        if not subset:
            continue
        wins = [t for t in subset if t['pnl_rs'] > 0]
        wr = len(wins) / len(subset)
        total = sum(t['pnl_rs'] for t in subset)
        print(f"    {label:>10s}: {len(subset):3d} trades, WR {wr:.0%}, total ₹{total:+,.0f}")

    # ── BY EXIT REASON ──
    print(f"\n  By exit reason:")
    for reason in sorted(set(t['exit_reason'] for t in trades)):
        rt = [t for t in trades if t['exit_reason'] == reason]
        wins = [t for t in rt if t['pnl_rs'] > 0]
        wr = len(wins) / len(rt)
        total = sum(t['pnl_rs'] for t in rt)
        print(f"    {reason:>12s}: {len(rt):3d} trades, WR {wr:.0%}, total ₹{total:+,.0f}")

    # ── OVERALL METRICS ──
    pnls = [t['pnl_rs'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
    total_pnl = sum(pnls)
    avg_pnl = np.mean(pnls)

    # Sharpe from daily P&L
    daily_pnl = {}
    for t in trades:
        d = str(t['date'])
        daily_pnl[d] = daily_pnl.get(d, 0) + t['pnl_rs']
    rets = np.array(list(daily_pnl.values())) / 1_000_000
    sharpe = (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0

    # ── WF WINDOWS ──
    print(f"\n{'─' * 80}")
    print("  WALK-FORWARD WINDOWS")

    trade_dates = sorted(set(t['date'] for t in trades))
    if not trade_dates:
        print("  No trade dates")
        return

    all_dates = sorted(set(t['date'] for t in trades))
    min_d = min(all_dates)
    max_d = max(all_dates)

    windows = []
    start = min_d
    while True:
        test_start = start + timedelta(days=TRAIN_DAYS)
        test_end = test_start + timedelta(days=TEST_DAYS)
        if test_end > max_d:
            break
        windows.append((start, test_start, test_end))
        start = start + timedelta(days=STEP_DAYS)

    passes = 0
    for i, (_, ts, te) in enumerate(windows):
        wt = [t for t in trades if ts <= t['date'] <= te]
        if len(wt) < 3:
            continue
        w_pnls = [t['pnl_rs'] for t in wt]
        w_wins = [p for p in w_pnls if p > 0]
        w_losses = [p for p in w_pnls if p <= 0]
        w_wr = len(w_wins) / len(w_pnls)
        w_pf = sum(w_wins) / abs(sum(w_losses)) if w_losses and sum(w_losses) != 0 else 0

        w_daily = {}
        for t in wt:
            d = str(t['date'])
            w_daily[d] = w_daily.get(d, 0) + t['pnl_rs']
        w_rets = np.array(list(w_daily.values())) / 1_000_000
        w_sharpe = (np.mean(w_rets) / np.std(w_rets, ddof=1)) * np.sqrt(252) if len(w_rets) > 1 and np.std(w_rets) > 0 else 0

        passed = w_sharpe >= 0.8 and w_pf >= 1.3
        if passed:
            passes += 1
        marker = " *" if passed else ""
        print(f"  W{i:2d} {ts}–{te} {len(wt):3d}tr WR {w_wr:.0%} PF {w_pf:.2f} Sharpe {w_sharpe:+.2f}{marker}")

    wf_rate = passes / max(len(windows), 1)

    # ── FINAL TABLE ──
    trades_per_year = len(trades) / (len(set(t['date'] for t in trades)) / 252)

    print(f"\n{'=' * 80}")
    print(f"  GIFT CONVERGENCE RESULTS")
    print(f"{'=' * 80}")
    print(f"  Trades:        {len(trades)}")
    print(f"  Trades/year:   ~{trades_per_year:.0f}")
    print(f"  Win Rate:      {wr:.0%}")
    print(f"  Profit Factor: {pf:.2f}")
    print(f"  Sharpe:        {sharpe:.2f}")
    print(f"  Total P&L:     ₹{total_pnl:+,.0f}")
    print(f"  Avg P&L/trade: ₹{avg_pnl:+,.0f}")
    print(f"  WF Pass Rate:  {passes}/{len(windows)} ({wf_rate:.0%})")
    print(f"  Verdict:       {'PASS' if wf_rate >= 0.60 and pf >= 1.3 and sharpe >= 0.8 else 'FAIL'}")
    print(f"  Time:          {time_mod.perf_counter() - t0:.1f}s")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    main()

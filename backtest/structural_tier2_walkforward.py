"""
Walk-Forward Backtest for 4 Tier 2 Structural Signals.

Uses actual signal class backtest_evaluate() methods on real DB data.
WF: 252-day train / 63-day test / 63-day step.
Pass: WR >= 50%, PF >= 1.2, trades >= 3 per window, >= 60% windows pass.

Usage:
    venv/bin/python3 -m backtest.structural_tier2_walkforward
"""

import logging
import math
import time as time_mod
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE

logger = logging.getLogger(__name__)

# WF config
TRAIN_DAYS = 252
TEST_DAYS = 63
STEP_DAYS = 63
MIN_TRADES_PER_WINDOW = 3

# Pass criteria per window
MIN_WR = 0.50
MIN_PF = 1.20

# Overall pass
MIN_WINDOW_PASS_RATE = 0.60

# Trade mechanics
SL_PCT = 0.015      # 1.5%
TGT_PCT = 0.020     # 2.0%
SLIPPAGE_PTS = 2.0  # per side

# Max hold days per signal
MAX_HOLD = {
    'ROLLOVER_FLOW': 5,
    'INDEX_REBALANCE': 3,
    'QUARTER_WINDOW': 5,
    'DII_PUT_FLOOR': 3,
}


def load_data():
    """Load daily, options, and intraday data from Docker DB."""
    conn = psycopg2.connect(DATABASE_DSN)

    daily = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix "
        "FROM nifty_daily ORDER BY date",
        conn, parse_dates=['date'])
    daily['date'] = daily['date'].dt.date

    options = pd.read_sql(
        "SELECT date, expiry, strike, option_type, oi, close as premium "
        "FROM nifty_options WHERE oi > 0 ORDER BY date, strike",
        conn, parse_dates=['date', 'expiry'])
    options['date'] = options['date'].dt.date
    options['expiry'] = options['expiry'].dt.date

    conn.close()
    return daily, options


def generate_signals_for_period(
    signal_id: str,
    signal_obj,
    daily: pd.DataFrame,
    options: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> List[Dict]:
    """Generate all signal fires for a date range using backtest_evaluate()."""
    signals = []
    period_daily = daily[(daily['date'] >= start_date) & (daily['date'] <= end_date)]
    trade_dates = sorted(period_daily['date'].unique())

    for d in trade_dates:
        try:
            if signal_id == 'ROLLOVER_FLOW':
                result = signal_obj.backtest_evaluate(d, daily, options)
            elif signal_id == 'INDEX_REBALANCE':
                result = signal_obj.backtest_evaluate(d, daily)
            elif signal_id == 'QUARTER_WINDOW':
                result = signal_obj.backtest_evaluate(d, daily)
            elif signal_id == 'DII_PUT_FLOOR':
                spot = float(period_daily[period_daily['date'] == d].iloc[0]['close'])
                day_opts = options[options['date'] == d]
                result = signal_obj.backtest_evaluate(d, spot, day_opts if not day_opts.empty else None)
            else:
                result = None

            if result and result.get('direction'):
                result['trade_date'] = d
                result['signal_id'] = signal_id
                signals.append(result)
        except Exception as e:
            logger.debug(f"{signal_id} error on {d}: {e}")
            continue

    # Reset stateful signals between periods
    if hasattr(signal_obj, 'reset'):
        signal_obj.reset()

    return signals


def simulate_trades(
    signals: List[Dict],
    daily: pd.DataFrame,
    signal_id: str,
) -> List[Dict]:
    """Simulate trades from signal fires with SL/TGT/time exit."""
    trades = []
    close_map = {r['date']: float(r['close']) for _, r in daily.iterrows()}
    dates_list = sorted(close_map.keys())

    max_hold = MAX_HOLD.get(signal_id, 5)
    traded_dates = set()

    for sig in signals:
        d = sig.get('trade_date')
        if d is None or d in traded_dates:
            continue

        direction = sig.get('direction', '').upper()
        if direction not in ('LONG', 'SHORT', 'BULLISH', 'BEARISH'):
            # Map alternative names
            if direction in ('BUY', 'BULL'):
                direction = 'LONG'
            elif direction in ('SELL', 'BEAR'):
                direction = 'SHORT'
            else:
                continue

        if 'BEAR' in direction:
            direction = 'SHORT'
        if 'BULL' in direction:
            direction = 'LONG'

        entry_price = sig.get('entry_price') or sig.get('price') or close_map.get(d, 0)
        if not entry_price or entry_price <= 0:
            entry_price = close_map.get(d, 0)
        if entry_price <= 0:
            continue

        # Apply entry slippage
        if direction == 'LONG':
            entry_price += SLIPPAGE_PTS
        else:
            entry_price -= SLIPPAGE_PTS

        sl = entry_price * (1 - SL_PCT) if direction == 'LONG' else entry_price * (1 + SL_PCT)
        tgt = entry_price * (1 + TGT_PCT) if direction == 'LONG' else entry_price * (1 - TGT_PCT)

        # Use signal-provided SL/TGT if available
        if sig.get('stop_loss') and sig['stop_loss'] > 0:
            sl = sig['stop_loss']
        if sig.get('target') and sig['target'] > 0:
            tgt = sig['target']

        # Simulate holding period
        try:
            d_idx = dates_list.index(d)
        except ValueError:
            continue

        exit_price = entry_price
        exit_reason = 'TIME'

        for j in range(1, max_hold + 1):
            if d_idx + j >= len(dates_list):
                break
            next_d = dates_list[d_idx + j]
            next_row = daily[daily['date'] == next_d]
            if next_row.empty:
                continue

            h = float(next_row.iloc[0]['high'])
            l = float(next_row.iloc[0]['low'])
            c = float(next_row.iloc[0]['close'])

            if direction == 'LONG':
                if l <= sl:
                    exit_price = sl - SLIPPAGE_PTS
                    exit_reason = 'SL'
                    break
                if h >= tgt:
                    exit_price = tgt - SLIPPAGE_PTS
                    exit_reason = 'TGT'
                    break
            else:
                if h >= sl:
                    exit_price = sl + SLIPPAGE_PTS
                    exit_reason = 'SL'
                    break
                if l <= tgt:
                    exit_price = tgt + SLIPPAGE_PTS
                    exit_reason = 'TGT'
                    break

            if j == max_hold:
                exit_price = c + (SLIPPAGE_PTS if direction == 'SHORT' else -SLIPPAGE_PTS)
                exit_reason = 'TIME'

        # Lot size
        lot_size = 75 if d < date(2023, 7, 1) else 25

        if direction == 'LONG':
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price

        pnl_rs = pnl_pts * lot_size * 1  # 1 lot
        costs = 40 * 2 + abs(exit_price * lot_size * 0.000125)
        net_pnl = pnl_rs - costs

        trades.append({
            'signal_id': signal_id,
            'date': d,
            'direction': direction,
            'entry': round(entry_price, 1),
            'exit': round(exit_price, 1),
            'pnl_pts': round(pnl_pts, 1),
            'pnl_rs': round(net_pnl),
            'exit_reason': exit_reason,
            'lot_size': lot_size,
        })
        traded_dates.add(d)

    return trades


def run_walkforward(
    signal_id: str,
    signal_obj,
    daily: pd.DataFrame,
    options: pd.DataFrame,
) -> Dict:
    """Run full walk-forward validation for one signal."""
    dates = sorted(daily['date'].unique())
    min_d = dates[0]
    max_d = dates[-1]

    # Generate ALL signals first
    all_signals = generate_signals_for_period(
        signal_id, signal_obj, daily, options, min_d, max_d)

    # Simulate ALL trades
    all_trades = simulate_trades(all_signals, daily, signal_id)

    if not all_trades:
        return {
            'signal_id': signal_id,
            'windows': [],
            'total_trades': 0,
            'overall_wr': 0, 'overall_pf': 0,
            'total_pnl': 0,
            'pass_rate': 0,
            'verdict': 'NO_TRADES',
        }

    # Generate WF windows
    windows = []
    start_idx = 0
    while start_idx + TRAIN_DAYS + TEST_DAYS <= len(dates):
        test_start = dates[start_idx + TRAIN_DAYS]
        test_end_idx = min(start_idx + TRAIN_DAYS + TEST_DAYS - 1, len(dates) - 1)
        test_end = dates[test_end_idx]
        windows.append((test_start, test_end))
        start_idx += STEP_DAYS

    # Evaluate each window
    window_results = []
    for i, (ts, te) in enumerate(windows):
        w_trades = [t for t in all_trades if ts <= t['date'] <= te]

        if len(w_trades) < MIN_TRADES_PER_WINDOW:
            window_results.append({
                'window': i, 'period': f"{ts} to {te}",
                'trades': len(w_trades), 'wr': 0, 'pf': 0,
                'net_pnl': sum(t['pnl_rs'] for t in w_trades),
                'passed': False, 'reason': 'INSUFFICIENT_TRADES',
            })
            continue

        pnls = [t['pnl_rs'] for t in w_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(pnls)
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 99.0
        net = sum(pnls)

        passed = wr >= MIN_WR and pf >= MIN_PF
        window_results.append({
            'window': i, 'period': f"{ts} to {te}",
            'trades': len(w_trades), 'wr': wr, 'pf': pf,
            'net_pnl': net, 'passed': passed,
        })

    # Aggregate
    total_pnl = sum(t['pnl_rs'] for t in all_trades)
    all_pnls = [t['pnl_rs'] for t in all_trades]
    all_wins = [p for p in all_pnls if p > 0]
    all_losses = [p for p in all_pnls if p <= 0]
    overall_wr = len(all_wins) / len(all_pnls) if all_pnls else 0
    overall_pf = sum(all_wins) / abs(sum(all_losses)) if all_losses and sum(all_losses) != 0 else 0

    evaluated_windows = [w for w in window_results if w.get('reason') != 'INSUFFICIENT_TRADES']
    passes = sum(1 for w in evaluated_windows if w['passed'])
    pass_rate = passes / len(evaluated_windows) if evaluated_windows else 0
    verdict = 'PASS' if pass_rate >= MIN_WINDOW_PASS_RATE else 'FAIL'

    return {
        'signal_id': signal_id,
        'windows': window_results,
        'total_trades': len(all_trades),
        'overall_wr': overall_wr,
        'overall_pf': overall_pf,
        'total_pnl': total_pnl,
        'pass_rate': pass_rate,
        'passes': passes,
        'total_windows': len(evaluated_windows),
        'verdict': verdict,
    }


def print_results(result: Dict):
    """Print formatted WF results for one signal."""
    sig = result['signal_id']
    print(f"\n=== {sig} Walk-Forward Results ===")

    for w in result['windows']:
        status = "PASS" if w['passed'] else "FAIL"
        if w.get('reason') == 'INSUFFICIENT_TRADES':
            status = "SKIP (<3 trades)"
        print(f"  Window {w['window']:2d}: {w['period']} | "
              f"Trades: {w['trades']:2d} | WR: {w['wr']:.0%} | "
              f"PF: {w['pf']:.2f} | Net: ₹{w['net_pnl']:+,.0f} | {status}")

    v = result['verdict']
    pr = result['pass_rate']
    print(f"\n  OVERALL: {result.get('passes', 0)}/{result.get('total_windows', 0)} "
          f"windows pass ({pr:.0%}) → {v} {'✓' if v == 'PASS' else '✗'}")
    print(f"  Aggregate: {result['total_trades']} trades | "
          f"WR: {result['overall_wr']:.1%} | PF: {result['overall_pf']:.2f} | "
          f"Total PnL: ₹{result['total_pnl']:+,.0f}")


def main():
    logging.basicConfig(level=logging.WARNING)
    t0 = time_mod.perf_counter()

    print("=" * 85)
    print("  STRUCTURAL TIER 2 — Proper Walk-Forward Validation")
    print("  WF: 252-day train / 63-day test / 63-day step")
    print("  Pass: WR >= 50%, PF >= 1.2, trades >= 3, >= 60% windows pass")
    print("=" * 85)

    daily, options = load_data()
    print(f"\nLoaded {len(daily)} daily bars, {len(options)} option rows")
    print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")

    # Import and instantiate signals
    from signals.structural.rollover_flow import RolloverFlowSignal
    from signals.structural.index_rebalance import IndexRebalanceSignal
    from signals.structural.quarter_window import QuarterWindowSignal
    from signals.structural.dii_put_floor import DIIPutFloorSignal

    signals = [
        ('ROLLOVER_FLOW', RolloverFlowSignal()),
        ('INDEX_REBALANCE', IndexRebalanceSignal()),
        ('QUARTER_WINDOW', QuarterWindowSignal()),
        ('DII_PUT_FLOOR', DIIPutFloorSignal()),
    ]

    results = {}
    for sig_id, sig_obj in signals:
        print(f"\nRunning WF for {sig_id}...")
        result = run_walkforward(sig_id, sig_obj, daily, options)
        print_results(result)
        results[sig_id] = result

    # Summary
    print(f"\n{'=' * 85}")
    print("  SUMMARY")
    print(f"{'=' * 85}")
    print(f"  {'Signal':<20s} {'Verdict':>8s} {'WF%':>6s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} {'Total PnL':>12s}")
    print(f"  {'─' * 65}")

    for sig_id, r in results.items():
        v = r['verdict']
        marker = '✓' if v == 'PASS' else '✗'
        print(f"  {sig_id:<20s} {v+' '+marker:>8s} {r['pass_rate']:>5.0%} "
              f"{r['total_trades']:>7d} {r['overall_wr']:>5.1%} "
              f"{r['overall_pf']:>5.2f} ₹{r['total_pnl']:>10,.0f}")

    elapsed = time_mod.perf_counter() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    print("=" * 85)


if __name__ == '__main__':
    main()

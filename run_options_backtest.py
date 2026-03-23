"""
Run L8 options backtest on all 10 strategies.

Usage:
    python run_options_backtest.py
    python run_options_backtest.py --capital 5000000
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import psycopg2

from backtest.options_backtest import run_options_backtest
from signals.l8_signals import L8_SIGNALS
from config.settings import DATABASE_DSN


def load_options_data(conn):
    """Load historical options data grouped by date."""
    print("Loading options data...", flush=True)

    # Options chain per date
    df = pd.read_sql(
        "SELECT date, expiry, strike, option_type, close, volume, oi, "
        "implied_volatility, delta, gamma, theta, vega, days_to_expiry, moneyness "
        "FROM nifty_options ORDER BY date, strike",
        conn
    )
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['expiry'] = pd.to_datetime(df['expiry']).dt.date

    options_by_date = {dt: group for dt, group in df.groupby('date')}
    print(f"  {len(options_by_date)} trading days with options data")

    # Spot prices
    spot_df = pd.read_sql(
        "SELECT date, close, india_vix FROM nifty_daily ORDER BY date", conn
    )
    spot_df['date'] = pd.to_datetime(spot_df['date']).dt.date
    spot_history = dict(zip(spot_df['date'], spot_df['close']))
    vix_history = dict(zip(spot_df['date'],
                           spot_df['india_vix'].fillna(15)))

    # Regime (simple classification)
    regime_history = {}
    for _, row in spot_df.iterrows():
        vix = row['india_vix'] if pd.notna(row['india_vix']) else 15
        regime_history[row['date']] = 'RANGING'  # simplified
        # Could load from regime_labels table for better accuracy

    return options_by_date, spot_history, vix_history, regime_history


def main():
    parser = argparse.ArgumentParser(description='L8 Options Backtest')
    parser.add_argument('--capital', type=float, default=5_000_000)
    args = parser.parse_args()

    print(f"L8 Options Backtest — 10 McMillan/Natenberg Strategies")
    print(f"Capital: ₹{args.capital/1e5:.0f}L\n")

    conn = psycopg2.connect(DATABASE_DSN)

    # Check if we have options data
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM nifty_options")
    count, min_d, max_d = cur.fetchone()

    if count == 0:
        print("ERROR: No options data. Run: python -m data.options_loader --init --start 2024-01-01")
        conn.close()
        return

    print(f"Options data: {count:,} rows, {min_d} to {max_d}")

    options_by_date, spot_history, vix_history, regime_history = load_options_data(conn)
    conn.close()

    n_days = len(options_by_date)
    V = len(L8_SIGNALS)
    dsr_penalty = math.sqrt(math.log(max(V, 2)) / max(n_days, 1))
    print(f"DSR penalty: {dsr_penalty:.4f} (V={V}, T={n_days})")

    # Run each strategy
    print(f"\n{'='*90}")
    print(f"RUNNING 10 L8 OPTIONS STRATEGIES")
    print(f"{'='*90}\n")

    all_results = []

    for signal_id, rule in L8_SIGNALS.items():
        metrics, trades = run_options_backtest(
            signal_id, rule, options_by_date, spot_history,
            vix_history, regime_history, args.capital
        )

        dsr = metrics['sharpe'] - dsr_penalty
        status = 'PASS' if dsr > 0.90 else 'MARGINAL' if dsr > 0.50 else 'FAIL'
        metrics['dsr'] = round(dsr, 2)
        metrics['status'] = status

        all_results.append(metrics)

        print(f"  {signal_id:<30s} {rule.strategy_type:<18s} "
              f"trades={metrics['trades']:>4d}  WR={metrics['win_rate']:>5.1%}  "
              f"Sharpe={metrics['sharpe']:>6.2f}  DSR={dsr:>6.2f}  "
              f"DD={metrics['max_dd']:>6.1%}  P&L=₹{metrics['total_pnl']:>8,.0f}  "
              f"{status}")

    # Summary
    print(f"\n{'='*90}")
    print(f"L8 OPTIONS RESULTS SUMMARY")
    print(f"{'='*90}")

    passed = [r for r in all_results if r['status'] == 'PASS']
    marginal = [r for r in all_results if r['status'] == 'MARGINAL']
    failed = [r for r in all_results if r['status'] == 'FAIL']

    print(f"  PASS (DSR > 0.90):    {len(passed)}")
    print(f"  MARGINAL (0.50-0.90): {len(marginal)}")
    print(f"  FAIL (DSR < 0.50):    {len(failed)}")

    total_pnl = sum(r['total_pnl'] for r in all_results)
    total_trades = sum(r['trades'] for r in all_results)
    print(f"\n  Total P&L:    ₹{total_pnl:,.0f}")
    print(f"  Total trades: {total_trades}")
    print(f"  Avg P&L/trade: ₹{total_pnl/total_trades:,.0f}" if total_trades > 0 else "")

    if passed:
        print(f"\n  SURVIVORS (ready for SHADOW):")
        for r in sorted(passed, key=lambda x: -x['sharpe']):
            print(f"    {r['signal_id']:<30s} Sharpe={r['sharpe']:.2f} DSR={r['dsr']:.2f} "
                  f"WR={r['win_rate']:.0%} trades={r['trades']}")

    # Save
    os.makedirs('backtest_results/options', exist_ok=True)
    with open('backtest_results/options/l8_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: backtest_results/options/l8_results.json")


if __name__ == '__main__':
    main()

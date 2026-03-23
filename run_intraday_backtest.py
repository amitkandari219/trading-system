"""
Run L9 intraday backtest on all 10 Al Brooks signals.

Usage:
    python run_intraday_backtest.py
    python run_intraday_backtest.py --capital 5000000
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import psycopg2

from backtest.intraday_backtest import run_intraday_backtest
from data.intraday_loader import load_intraday_history
from signals.l9_signals import IntradaySignalComputer
from config.settings import DATABASE_DSN


def main():
    parser = argparse.ArgumentParser(description='L9 Intraday Backtest')
    parser.add_argument('--capital', type=float, default=1_000_000)
    args = parser.parse_args()

    print("L9 Intraday Backtest — 10 Al Brooks Signals")
    print(f"Capital: ₹{args.capital/1e5:.0f}L")

    # Load 5-min data
    print("\nLoading intraday data...", flush=True)
    conn = psycopg2.connect(DATABASE_DSN)
    df = load_intraday_history(conn, timeframe='5min')
    conn.close()

    if df.empty:
        print("ERROR: No intraday data. Run: python -m data.intraday_loader --generate-synthetic")
        return

    n_days = df['datetime'].dt.date.nunique()
    print(f"  {len(df):,} 5-min bars, {n_days} trading days")
    print(f"  {df['datetime'].min().date()} to {df['datetime'].max().date()}")

    # DSR parameters
    V = 10  # 10 signals tested
    T = n_days
    dsr_penalty = math.sqrt(math.log(max(V, 2)) / max(T, 1))
    print(f"  DSR penalty: {dsr_penalty:.4f} (V={V}, T={T})")

    # Run each signal
    computer = IntradaySignalComputer()
    all_results = []

    print(f"\n{'='*90}")
    print(f"RUNNING 10 L9 SIGNALS")
    print(f"{'='*90}\n")

    for signal_id, config in computer.SIGNALS.items():
        result, trades = run_intraday_backtest(signal_id, config, df, args.capital)

        dsr = result.sharpe - dsr_penalty
        status = 'PASS' if dsr > 0.90 else 'MARGINAL' if dsr > 0.50 else 'FAIL'

        all_results.append({
            'signal_id': signal_id,
            'trades': result.trade_count,
            'win_rate': result.win_rate,
            'sharpe': result.sharpe,
            'dsr': round(dsr, 2),
            'max_dd': result.max_drawdown,
            'profit_factor': result.profit_factor,
            'calmar': result.calmar_ratio,
            'status': status,
            'all_trades': trades,
        })

        print(f"  {signal_id:<25s} trades={result.trade_count:>5d}  WR={result.win_rate:>5.1%}  "
              f"Sharpe={result.sharpe:>6.2f}  DSR={dsr:>6.2f}  DD={result.max_drawdown:>6.1%}  "
              f"PF={result.profit_factor:>5.2f}  {status}")

    # Summary
    print(f"\n{'='*90}")
    print(f"L9 INTRADAY RESULTS SUMMARY")
    print(f"{'='*90}")

    passed = [r for r in all_results if r['status'] == 'PASS']
    marginal = [r for r in all_results if r['status'] == 'MARGINAL']
    failed = [r for r in all_results if r['status'] == 'FAIL']

    print(f"  PASS (DSR > 0.90):    {len(passed)}")
    print(f"  MARGINAL (0.50-0.90): {len(marginal)}")
    print(f"  FAIL (DSR < 0.50):    {len(failed)}")

    if passed:
        print(f"\n  SURVIVORS (ready for SHADOW):")
        for r in sorted(passed, key=lambda x: -x['sharpe']):
            print(f"    {r['signal_id']:<25s} Sharpe={r['sharpe']:.2f} DSR={r['dsr']:.2f} "
                  f"trades={r['trades']} WR={r['win_rate']:.1%}")

    # Total trade frequency
    total_trades = sum(r['trades'] for r in all_results)
    trades_per_day = total_trades / n_days if n_days > 0 else 0
    print(f"\n  Total trades: {total_trades:,}")
    print(f"  Trades per day: {trades_per_day:.1f}")
    print(f"  Slippage: 0.05% each way (₹{24000*0.0005*2*25:.0f} per round-trip at Nifty 24000)")

    # Slippage sensitivity
    if passed:
        best = max(passed, key=lambda x: x['sharpe'])
        best_trades = best['all_trades']
        if best_trades:
            gross_pnl = sum(t['pnl_rs'] for t in best_trades)
            # Estimate slippage impact
            slippage_total = len(best_trades) * 2 * 24000 * 0.0005 * 25  # 2 legs × price × slip × lot
            print(f"\n  Slippage sensitivity ({best['signal_id']}):")
            print(f"    Gross P&L:     ₹{gross_pnl:,.0f}")
            print(f"    Total slippage: ₹{slippage_total:,.0f} ({len(best_trades)} trades)")
            print(f"    Slippage as % of gross: {slippage_total/abs(gross_pnl)*100:.1f}%" if gross_pnl != 0 else "")

    # Save results
    os.makedirs('backtest_results/intraday', exist_ok=True)
    save = [{k: v for k, v in r.items() if k != 'all_trades'} for r in all_results]
    with open('backtest_results/intraday/l9_results.json', 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\n  Saved: backtest_results/intraday/l9_results.json")


if __name__ == '__main__':
    main()

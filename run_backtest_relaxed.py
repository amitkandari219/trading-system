"""
Relaxed backtest evaluation: run all translated signals on full dataset
(no walk-forward splitting) to find signals with promise.

Signals that pass relaxed criteria are candidates for full WF later
with tuned parameters.

Relaxed criteria:
  - Min 20 trades (vs 50 in WF)
  - Sharpe >= 0.5 (vs 1.2 in WF)
  - Profit factor >= 1.1 (vs 1.6 in WF)

Usage:
    python run_backtest_relaxed.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import run_generic_backtest
from config.settings import DATABASE_DSN


TRANSLATED_PATH = 'extraction_results/translated_signals.json'
RESULTS_DIR = 'backtest_results'

# Relaxed criteria
MIN_TRADES = 20
MIN_SHARPE = 0.5
MIN_PROFIT_FACTOR = 1.1


def main():
    # Load data
    print("Loading Nifty data...", flush=True)
    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix "
        "FROM nifty_daily ORDER BY date", conn
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    print(f"  {len(df)} trading days ({df['date'].min().date()} to {df['date'].max().date()})")

    # Load translated signals
    translated = json.load(open(TRANSLATED_PATH))
    backtestable = [t for t in translated if t.get('backtestable') and t.get('rules')]
    print(f"  {len(backtestable)} backtestable signals")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_all = []
    promising = []
    zero_trades = 0
    errors = 0
    start = time.time()

    for i, t in enumerate(backtestable):
        signal_id = t['signal_id']
        rules = t['rules']

        try:
            result = run_generic_backtest(rules, df, {})

            entry = {
                'signal_id': signal_id,
                'trade_count': result.trade_count,
                'sharpe': result.sharpe,
                'calmar': result.calmar_ratio,
                'max_dd': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'avg_wl_ratio': result.avg_win_loss_ratio,
                'annual_return': result.annual_return,
                'nifty_corr': result.nifty_correlation,
                'reason': t.get('reason', ''),
            }

            if result.trade_count == 0:
                zero_trades += 1
            elif (result.trade_count >= MIN_TRADES and
                  result.sharpe >= MIN_SHARPE and
                  result.profit_factor >= MIN_PROFIT_FACTOR):
                entry['promising'] = True
                promising.append(entry)

            results_all.append(entry)

        except Exception as e:
            errors += 1
            results_all.append({
                'signal_id': signal_id,
                'error': str(e)[:200],
            })

        done = i + 1
        if done % 100 == 0 or done == len(backtestable):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{len(backtestable)}] ({rate:.1f}/s) "
                  f"Promising:{len(promising)} Zero:{zero_trades} Err:{errors}",
                  flush=True)

    # Sort promising by Sharpe
    promising.sort(key=lambda x: x['sharpe'], reverse=True)

    # Save results
    with open(os.path.join(RESULTS_DIR, '_RELAXED_ALL.json'), 'w') as f:
        json.dump(results_all, f, indent=2)

    with open(os.path.join(RESULTS_DIR, '_PROMISING.json'), 'w') as f:
        json.dump(promising, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"RELAXED BACKTEST RESULTS")
    print(f"{'='*80}")
    print(f"  Total backtestable:  {len(backtestable)}")
    print(f"  Zero trades:         {zero_trades}")
    print(f"  Errors:              {errors}")
    print(f"  Active (>0 trades):  {len(backtestable) - zero_trades - errors}")
    print(f"  PROMISING:           {len(promising)} (Sharpe≥{MIN_SHARPE}, PF≥{MIN_PROFIT_FACTOR}, trades≥{MIN_TRADES})")

    if promising:
        print(f"\n{'='*80}")
        print(f"TOP PROMISING SIGNALS (sorted by Sharpe)")
        print(f"{'='*80}")
        print(f"{'Signal ID':<25} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'Trades':>7} {'AnnRet':>8}")
        print(f"{'-'*25} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*8}")
        for p in promising[:30]:
            print(f"{p['signal_id']:<25} {p['sharpe']:>7.2f} {p['win_rate']:>5.1%} "
                  f"{p['profit_factor']:>6.2f} {p['max_dd']:>6.1%} {p['trade_count']:>7} "
                  f"{p['annual_return']:>8.0f}")

    print(f"\n  Saved: {RESULTS_DIR}/_PROMISING.json ({len(promising)} signals)")
    print(f"  Saved: {RESULTS_DIR}/_RELAXED_ALL.json ({len(results_all)} signals)")


if __name__ == '__main__':
    main()

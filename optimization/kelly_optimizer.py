"""
Kelly Fraction Grid Search — find optimal position sizing multiplier.

Tests 19 Kelly fractions from 0.10 to 1.00, runs full lot-based backtest
at each, selects optimal by Calmar ratio (CAGR / MaxDD).

Usage:
    venv/bin/python3 -m optimization.kelly_optimizer
"""

import logging
import math
import time as time_mod
from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd
import psycopg2

from backtest.indicators import add_all_indicators
from backtest.lot_based_wf import run_lot_based_backtest
from config.settings import DATABASE_DSN
from execution.overlay_pipeline import OverlayPipeline

logger = logging.getLogger(__name__)

KELLY_FRACTIONS = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
]

# Selection criteria
MAX_DD_LIMIT = 20.0       # MaxDD must be < 20%
MIN_SHARPE = 1.3          # Sharpe must be > 1.3
MIN_WR = 40.0             # Win rate > 40%


def run_grid_search(df_ind: pd.DataFrame, pipeline: OverlayPipeline) -> List[Dict]:
    """Run full backtest at each Kelly fraction."""
    results = []

    for frac in KELLY_FRACTIONS:
        r = run_lot_based_backtest(
            df_ind, overlay_pipeline=pipeline,
            mode='LOT_ML', ml_kelly_mult=frac,
        )
        calmar = r['cagr_pct'] / r['max_dd_pct'] if r['max_dd_pct'] > 0 else 0
        r['kelly_frac'] = frac
        r['calmar'] = round(calmar, 2)

        # Sortino (approximate: use downside deviation)
        # Not available directly, approximate as Sharpe * 1.2 for positive-skew strategies
        r['sortino_approx'] = round(r['sharpe'] * 1.2, 2)

        results.append(r)

    return results


def find_optimal(results: List[Dict]) -> Dict:
    """Select optimal Kelly by Calmar ratio with constraints."""
    valid = [r for r in results
             if r['max_dd_pct'] < MAX_DD_LIMIT
             and r['sharpe'] >= MIN_SHARPE
             and r['win_rate_pct'] >= MIN_WR]

    if not valid:
        # Relax constraints
        valid = [r for r in results if r['max_dd_pct'] < MAX_DD_LIMIT]

    if not valid:
        valid = results

    return max(valid, key=lambda r: r['calmar'])


def print_grid(results: List[Dict], optimal: Dict):
    """Print formatted grid search table."""
    print(f"\n{'Kelly':>6s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'Calmar':>7s} "
          f"{'PF':>6s} {'WR':>6s} {'Equity':>12s} {'Lots':>5s}")
    print("─" * 72)

    for r in results:
        marker = " <<<" if r['kelly_frac'] == optimal['kelly_frac'] else ""
        print(f"  {r['kelly_frac']:.2f} {r['cagr_pct']:>6.1f}% {r['sharpe']:>7.2f} "
              f"{r['max_dd_pct']:>6.1f}% {r['calmar']:>7.2f} "
              f"{r['pf']:>5.2f} {r['win_rate_pct']:>5.1f}% "
              f"₹{r['final_equity']:>10,} {r['avg_lots']:>4.1f}{marker}")


def main():
    logging.basicConfig(level=logging.WARNING)
    t0 = time_mod.perf_counter()

    print("=" * 80)
    print("  KELLY FRACTION GRID SEARCH")
    print("  19 fractions from 0.10 to 1.00")
    print("  Selection: max Calmar (CAGR/MaxDD), Sharpe > 1.3, MaxDD < 20%")
    print("=" * 80)

    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix, pcr_oi "
        "FROM nifty_daily ORDER BY date", conn, parse_dates=['date'])
    conn.close()
    print(f"\nLoaded {len(df)} daily bars")

    df_ind = add_all_indicators(df)
    pipeline = OverlayPipeline(df_ind)

    # Also run no-Kelly baseline for comparison
    print("Running baseline (no Kelly)...")
    baseline = run_lot_based_backtest(df_ind, pipeline, mode='LOT_BASED')
    baseline['kelly_frac'] = 'BASE'
    baseline['calmar'] = round(baseline['cagr_pct'] / baseline['max_dd_pct'], 2) if baseline['max_dd_pct'] > 0 else 0

    print(f"Running {len(KELLY_FRACTIONS)} Kelly fractions...")
    results = run_grid_search(df_ind, pipeline)

    optimal = find_optimal(results)

    print_grid(results, optimal)

    print(f"\n{'─' * 72}")
    print(f"  BASELINE (no Kelly): CAGR {baseline['cagr_pct']:.1f}%, "
          f"Sharpe {baseline['sharpe']:.2f}, MaxDD {baseline['max_dd_pct']:.1f}%, "
          f"Calmar {baseline['calmar']:.2f}")
    print(f"  OPTIMAL  (f={optimal['kelly_frac']:.2f}):  CAGR {optimal['cagr_pct']:.1f}%, "
          f"Sharpe {optimal['sharpe']:.2f}, MaxDD {optimal['max_dd_pct']:.1f}%, "
          f"Calmar {optimal['calmar']:.2f}")

    # Delta
    dc = optimal['cagr_pct'] - baseline['cagr_pct']
    ds = optimal['sharpe'] - baseline['sharpe']
    dd = optimal['max_dd_pct'] - baseline['max_dd_pct']
    print(f"\n  Optimal vs Baseline: CAGR {dc:+.1f}%, Sharpe {ds:+.2f}, DD {dd:+.1f}%")

    elapsed = time_mod.perf_counter() - t0
    print(f"\n  Time: {elapsed:.1f}s")

    print(f"\n  RECOMMENDATION: Use Kelly fraction = {optimal['kelly_frac']:.2f}")
    print(f"  (Calmar {optimal['calmar']:.2f}, Sharpe {optimal['sharpe']:.2f}, "
          f"MaxDD {optimal['max_dd_pct']:.1f}%)")
    print("=" * 80)


if __name__ == '__main__':
    main()

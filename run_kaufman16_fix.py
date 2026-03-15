"""
KAUFMAN_DRY_16 fix: improve the pivot-based rule from near-miss DROP to Tier B or A.

Current translation (from _WF_ALPHA.json):
  Entry long:  close > r1 AND low >= pivot
  Entry short: close < s1 (only 1 condition — asymmetric with long)
  Exit long:   low < pivot
  Exit short:  high > r1
  stop_loss 2%, direction BOTH, no regime filter

Result: 16/26 windows, Sharpe 0.77 => DROP (sharpe < 1.0)

Original book concepts (KAUFMAN Ch.16):
  1. Volatility filter: "Enter trades only when 6-day historic volatility
     is less than 100-day historic volatility"
  2. The short entry lacks a symmetric pivot condition

Fix variants:
  ORIGINAL:   Current pivot rule as-is
  FIX_1:      Add missing short-side condition (high <= pivot) — symmetric
  FIX_2:      Add RANGING-only regime filter (pivots work best in ranges)
  FIX_3:      Both FIX_1 + FIX_2 combined
  FIX_4:      Add Kaufman's volatility filter (hvol_6 < hvol_100)
  FIX_5:      FIX_1 + volatility filter
"""

import json
import time

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import run_generic_backtest, make_generic_backtest_fn
from backtest.indicators import sma, adx, historical_volatility
from backtest.walk_forward import WalkForwardEngine
from backtest.types import harmonic_mean_sharpe
from config.settings import DATABASE_DSN


# ── Rule variants ───────────────────────────────────────────────────────────

ORIGINAL_RULES = {
    "backtestable": True,
    "entry_long": [
        {"indicator": "close", "op": ">", "value": "r1"},
        {"indicator": "low", "op": ">=", "value": "pivot"},
    ],
    "entry_short": [
        {"indicator": "close", "op": "<", "value": "s1"},
    ],
    "exit_long": [
        {"indicator": "low", "op": "<", "value": "pivot"},
    ],
    "exit_short": [
        {"indicator": "high", "op": ">", "value": "r1"},
    ],
    "regime_filter": [],
    "hold_days": 0,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0,
    "direction": "BOTH",
}

# FIX_1: Add symmetric short-side condition (high <= pivot)
FIX_1_RULES = {
    **ORIGINAL_RULES,
    "entry_short": [
        {"indicator": "close", "op": "<", "value": "s1"},
        {"indicator": "high", "op": "<=", "value": "pivot"},
    ],
}

# FIX_2: Add RANGING-only regime filter
FIX_2_RULES = {
    **ORIGINAL_RULES,
    "regime_filter": ["RANGING"],
}

# FIX_3: Both fixes
FIX_3_RULES = {
    **FIX_1_RULES,
    "regime_filter": ["RANGING"],
}

# FIX_4: Add Kaufman's volatility filter (hvol_6 < hvol_100)
FIX_4_RULES = {
    **ORIGINAL_RULES,
    "entry_long": [
        {"indicator": "close", "op": ">", "value": "r1"},
        {"indicator": "low", "op": ">=", "value": "pivot"},
        {"indicator": "hvol_6", "op": "<", "value": "hvol_100"},
    ],
    "entry_short": [
        {"indicator": "close", "op": "<", "value": "s1"},
        {"indicator": "hvol_6", "op": "<", "value": "hvol_100"},
    ],
}

# FIX_5: Symmetric short + volatility filter
FIX_5_RULES = {
    **ORIGINAL_RULES,
    "entry_long": [
        {"indicator": "close", "op": ">", "value": "r1"},
        {"indicator": "low", "op": ">=", "value": "pivot"},
        {"indicator": "hvol_6", "op": "<", "value": "hvol_100"},
    ],
    "entry_short": [
        {"indicator": "close", "op": "<", "value": "s1"},
        {"indicator": "high", "op": "<=", "value": "pivot"},
        {"indicator": "hvol_6", "op": "<", "value": "hvol_100"},
    ],
}


# ── Helper functions (from run_kaufman20_fix.py) ────────────────────────────

def classify_regime(history_df, start_date, end_date):
    mask = (history_df['date'] >= start_date) & (history_df['date'] <= end_date)
    window_df = history_df[mask].copy()
    if len(window_df) < 20:
        return 'UNKNOWN'
    close = window_df['close']
    sma_200 = sma(history_df['close'], 200)
    window_sma = sma_200[mask]
    above_sma = (close > window_sma).mean() if window_sma.notna().sum() > 0 else 0.5
    adx_vals = adx(history_df)
    window_adx = adx_vals[mask]
    mean_adx = window_adx.mean() if window_adx.notna().sum() > 0 else 20
    vix = window_df.get('india_vix')
    mean_vix = vix.mean() if vix is not None and vix.notna().sum() > 0 else 15
    net_return = (close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else 0
    if mean_vix > 22:
        return 'HIGH_VOL'
    elif mean_adx > 25:
        return 'TRENDING_UP' if net_return > 0.05 and above_sma > 0.6 else \
               'TRENDING_DOWN' if net_return < -0.05 and above_sma < 0.4 else \
               ('TRENDING_UP' if above_sma > 0.5 else 'TRENDING_DOWN')
    return 'RANGING'


def tier_signal(window_details):
    total = len(window_details)
    if total < 4:
        return 'DROP', 'INSUFFICIENT_WINDOWS'
    passed = sum(1 for w in window_details if w['passed'])
    pass_rate = passed / total
    last4 = window_details[-4:]
    last4_passed = sum(1 for w in last4 if w['passed'])
    last4_rate = last4_passed / 4
    agg_sharpe = harmonic_mean_sharpe(
        [{'result': w['result']} for w in window_details]
    )
    zero_windows = sum(1 for w in window_details if w['result'].trade_count == 0)
    zero_pct = zero_windows / total
    active_dds = [w['result'].max_drawdown for w in window_details
                  if w['result'].trade_count > 0]
    worst_dd = max(active_dds) if active_dds else 1.0

    if zero_pct > 0.6:
        return 'DROP', f'ZERO_TRADES_{zero_pct:.0%}_OF_WINDOWS'
    if last4_rate == 0:
        return 'DROP', 'LAST_4_ALL_FAIL'
    if passed <= 2:
        return 'DROP', f'ONLY_{passed}_WINDOWS_PASS'

    if (pass_rate >= 0.60 and agg_sharpe >= 1.5 and
            last4_rate >= 0.75 and worst_dd <= 0.20):
        return 'TIER_A', None
    if (pass_rate >= 0.40 and agg_sharpe >= 1.0 and
            last4_rate >= 0.50):
        return 'TIER_B', None

    reasons = []
    if pass_rate < 0.40:
        reasons.append(f'pass_rate={pass_rate:.0%}<40%')
    if agg_sharpe < 1.0:
        reasons.append(f'sharpe={agg_sharpe:.2f}<1.0')
    if last4_rate < 0.50:
        reasons.append(f'last4={last4_rate:.0%}<50%')
    return 'DROP', '; '.join(reasons) if reasons else 'BELOW_TIER_B'


def run_one(name, rules, df, cal, regime_labels):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # Full-period backtest
    full_result = run_generic_backtest(rules, df, {})
    print(f"\n  Full-period results:")
    print(f"    Sharpe:     {full_result.sharpe:.3f}")
    print(f"    Calmar:     {full_result.calmar_ratio:.3f}")
    print(f"    Win rate:   {full_result.win_rate:.1%}")
    print(f"    Trades:     {full_result.trade_count}")
    print(f"    Max DD:     {full_result.max_drawdown:.1%}")
    print(f"    Nifty corr: {full_result.nifty_correlation:.3f}")
    print(f"    Ann return: {full_result.annual_return:.2%}")

    if full_result.trade_count < 5:
        print("  SKIPPING WALK-FORWARD: too few trades")
        return 'DROP', None

    # Walk-forward
    wfe = WalkForwardEngine(cal)
    for criteria in wfe.CRITERIA.values():
        criteria['min_trades'] = 10
        criteria['min_sharpe'] = 0.8
        if 'min_calmar' in criteria:
            del criteria['min_calmar']
        if 'min_profit_factor' in criteria:
            del criteria['min_profit_factor']

    backtest_fn = make_generic_backtest_fn(rules)
    wf = wfe.run('KAUFMAN_DRY_16', backtest_fn, df, regime_labels, rules, 'FUTURES')

    test_windows = wfe._generate_windows(df)
    window_regimes = {i: classify_regime(df, w['test_start'], w['test_end'])
                      for i, w in enumerate(test_windows)}

    for wd in wf['window_details']:
        wd['regime'] = window_regimes.get(wd['window_index'], 'UNKNOWN')

    tier, reason = tier_signal(wf['window_details'])

    print(f"\n  Walk-forward results:")
    print(f"    Windows passed: {wf['windows_passed']}/{wf['total_windows']} "
          f"({wf['pass_rate']:.0%})")
    print(f"    Aggregate Sharpe: {wf['aggregate_sharpe']:.3f}")

    last4 = wf['window_details'][-4:]
    last4_passed = sum(1 for w in last4 if w['passed'])
    print(f"    Last 4: {last4_passed}/4")

    active_dds = [w['result'].max_drawdown for w in wf['window_details']
                  if w['result'].trade_count > 0]
    worst_dd = max(active_dds) if active_dds else 0
    print(f"    Worst window DD: {worst_dd:.1%}")

    print(f"\n    >>> TIER: {tier} {'(' + reason + ')' if reason else ''}")

    # Per-window detail
    print(f"\n  Per-window breakdown:")
    print(f"  {'Idx':>3s} {'Regime':>12s} {'Sharpe':>7s} {'WR':>6s} {'Trades':>6s} "
          f"{'MaxDD':>6s} {'Pass':>5s}")
    print(f"  {'-'*3} {'-'*12} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
    for wd in wf['window_details']:
        r = wd['result']
        print(f"  {wd['window_index']:3d} {wd.get('regime', '?'):>12s} "
              f"{r.sharpe:7.2f} {r.win_rate:6.1%} {r.trade_count:6d} "
              f"{r.max_drawdown:6.1%} {'PASS' if wd['passed'] else 'FAIL':>5s}")

    return tier, wf


def main():
    print("Loading data...", flush=True)
    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix "
        "FROM nifty_daily ORDER BY date", conn
    )
    cal = pd.read_sql(
        "SELECT trading_date AS date, is_trading_day "
        "FROM market_calendar ORDER BY trading_date", conn
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    cal['date'] = pd.to_datetime(cal['date'])
    print(f"  {len(df)} trading days", flush=True)

    # Add hvol_6 and hvol_100 for Kaufman's volatility filter
    df['hvol_6'] = historical_volatility(df['close'], period=6)
    df['hvol_100'] = historical_volatility(df['close'], period=100)

    from regime_labeler import RegimeLabeler
    labeler = RegimeLabeler()
    regime_labels = labeler.label_full_history(df)

    variants = [
        ("ORIGINAL (current pivot rule)", ORIGINAL_RULES),
        ("FIX_1 (symmetric short: + high <= pivot)", FIX_1_RULES),
        ("FIX_2 (RANGING-only regime filter)", FIX_2_RULES),
        ("FIX_3 (symmetric short + RANGING filter)", FIX_3_RULES),
        ("FIX_4 (volatility filter: hvol_6 < hvol_100)", FIX_4_RULES),
        ("FIX_5 (symmetric short + volatility filter)", FIX_5_RULES),
    ]

    results = {}
    for name, rules in variants:
        tier, wf = run_one(name, rules, df, cal, regime_labels)
        results[name] = {'tier': tier, 'wf': wf}

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — KAUFMAN_DRY_16 FIX VARIANTS")
    print(f"{'='*70}")
    for name, res in results.items():
        tier = res['tier']
        wf = res['wf']
        if wf:
            passed = wf['windows_passed']
            total = wf['total_windows']
            sharpe = wf['aggregate_sharpe']
            print(f"  {name:55s} => {tier}  ({passed}/{total}, Sharpe {sharpe:.2f})")
        else:
            print(f"  {name:55s} => {tier}  (no walk-forward)")

    # Find best
    best_name = None
    best_tier = None
    for name, res in results.items():
        tier = res['tier']
        if tier == 'TIER_A':
            if best_tier != 'TIER_A':
                best_name = name
                best_tier = tier
        elif tier == 'TIER_B' and best_tier not in ('TIER_A',):
            best_name = name
            best_tier = tier

    if best_tier in ('TIER_A', 'TIER_B'):
        print(f"\n  >>> BEST VARIANT: {best_name}")
        print(f"  >>> TIER: {best_tier}")
        print(f"  >>> KAUFMAN_DRY_16 CAN BE PROMOTED <<<")
    else:
        print(f"\n  No variant reached Tier B or above.")
        print(f"  KAUFMAN_DRY_16 remains DROP'd.")


if __name__ == '__main__':
    main()

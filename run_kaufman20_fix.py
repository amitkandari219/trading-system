"""
KAUFMAN_DRY_20 fix: correct the stochastic period and prev_close comparison.

Original book rule (Kaufman, Trading Systems and Methods, Ch.20, p.865):
  "Buy when a 10-day moving average is less than yesterday's close
   AND a 5-day stochastic is greater than 50."

Current (broken) translation:
  Entry: sma_10 < close AND stoch_k > 50
  Issues:
    1. stoch_k is 14-period, book says 5-period
    2. "yesterday's close" → should be prev_close, not close

Fixed translation:
  Entry: sma_10 < prev_close AND stoch_k_5 > 50
  Exit: stoch_k_5 <= 50 (Haiku's reasonable default for AUTHOR_SILENT)

Run walk-forward on both original and fixed rules to compare.
"""

import json
import time

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import run_generic_backtest, make_generic_backtest_fn
from backtest.indicators import sma, adx
from backtest.walk_forward import WalkForwardEngine
from backtest.types import harmonic_mean_sharpe
from config.settings import DATABASE_DSN


ORIGINAL_RULES = {
    "backtestable": True,
    "entry_long": [
        {"indicator": "sma_10", "op": "<", "value": "close"},
        {"indicator": "stoch_k", "op": ">", "value": 50}
    ],
    "entry_short": [],
    "exit_long": [
        {"indicator": "stoch_k", "op": "<=", "value": 50}
    ],
    "exit_short": [],
    "regime_filter": [],
    "hold_days": 0,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0,
    "direction": "LONG",
}

FIXED_RULES = {
    "backtestable": True,
    "entry_long": [
        {"indicator": "sma_10", "op": "<", "value": "prev_close"},
        {"indicator": "stoch_k_5", "op": ">", "value": 50}
    ],
    "entry_short": [],
    "exit_long": [
        {"indicator": "stoch_k_5", "op": "<=", "value": 50}
    ],
    "exit_short": [],
    "regime_filter": [],
    "hold_days": 0,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0,
    "direction": "LONG",
}

# Also test a variant with hold_days for comparison
FIXED_WITH_HOLD = {
    **FIXED_RULES,
    "hold_days": 5,
    "exit_long": [
        {"indicator": "stoch_k_5", "op": "<=", "value": 30}  # tighter exit
    ],
}


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
        return

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
    wf = wfe.run('KAUFMAN_DRY_20', backtest_fn, df, regime_labels, rules, 'FUTURES')

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

    from regime_labeler import RegimeLabeler
    labeler = RegimeLabeler()
    regime_labels = labeler.label_full_history(df)

    # Run original
    original_tier, _ = run_one("ORIGINAL (stoch_k 14-period, close)", ORIGINAL_RULES,
                                df, cal, regime_labels)

    # Run fixed
    fixed_tier, fixed_wf = run_one("FIXED (stoch_k_5 5-period, prev_close)", FIXED_RULES,
                                    df, cal, regime_labels)

    # Run variant with hold_days
    hold_tier, _ = run_one("FIXED + 5-day hold + tighter exit", FIXED_WITH_HOLD,
                            df, cal, regime_labels)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Original (14-period stoch): {original_tier}")
    print(f"  Fixed (5-period stoch):     {fixed_tier}")
    print(f"  Fixed + hold:               {hold_tier}")

    if fixed_tier == 'TIER_A':
        print(f"\n  >>> KAUFMAN_DRY_20 PROMOTED TO TIER A <<<")
        print(f"  >>> PAPER TRADING CANDIDATE <<<")

        # Save the fixed rules
        output = {
            'signal_id': 'KAUFMAN_DRY_20',
            'rules': FIXED_RULES,
            'tier': 'TIER_A',
            'fix_description': 'Corrected stochastic period from 14 to 5 (per book), '
                               'changed close to prev_close (per book: "yesterday\'s close")',
            'original_rule': 'Buy when a 10-day moving average is less than yesterday\'s close '
                             'AND a 5-day stochastic is greater than 50.',
            'source': 'Trading Systems and Methods, Perry Kaufman, Ch.20, p.865',
        }
        with open('validation_results/kaufman_dry_20_fixed.json', 'w') as f:
            json.dump(output, f, indent=2)
        print(f"  Saved to: validation_results/kaufman_dry_20_fixed.json")
    elif fixed_tier == 'TIER_B':
        print(f"\n  Fixed rule is TIER_B — improvement but not Tier A yet")
    else:
        print(f"\n  Fixed rule did not reach Tier B — the fix was not sufficient")


if __name__ == '__main__':
    main()

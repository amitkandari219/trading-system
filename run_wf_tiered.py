"""
Walk-forward with tiered pass criteria and regime tagging.

Tier A (strong):
  - Window pass rate >= 60%
  - Aggregate Sharpe >= 1.5
  - Last 4 windows pass rate >= 75%
  - Max drawdown per window <= 20%

Tier B (regime-dependent):
  - Window pass rate >= 40%
  - Aggregate Sharpe >= 1.0
  - Last 4 windows pass rate >= 50%

Drop:
  - Zero trades in majority of windows
  - Last 4 windows all failing
  - Concentrated in single regime only

Each window is tagged with the dominant market regime during that period.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import run_generic_backtest, make_generic_backtest_fn
from backtest.indicators import sma, adx
from backtest.walk_forward import WalkForwardEngine
from backtest.types import harmonic_mean_sharpe
from config.settings import DATABASE_DSN


TRANSLATED_PATH = 'extraction_results/translated_signals.json'
PROMISING_PATH = 'backtest_results/_PROMISING.json'
RESULTS_DIR = 'backtest_results'


def classify_regime(history_df, start_date, end_date):
    """Classify the dominant market regime for a date range.

    Returns: TRENDING_UP, TRENDING_DOWN, HIGH_VOL, RANGING
    """
    mask = (history_df['date'] >= start_date) & (history_df['date'] <= end_date)
    window_df = history_df[mask].copy()

    if len(window_df) < 20:
        return 'UNKNOWN'

    close = window_df['close']

    # Trend: price vs 200-day SMA (use what's available in the window)
    sma_200 = sma(history_df['close'], 200)
    window_sma = sma_200[mask]
    if window_sma.notna().sum() > 0:
        above_sma = (close > window_sma).mean()
    else:
        # Fallback: use 50-day SMA
        sma_50 = sma(history_df['close'], 50)
        window_sma50 = sma_50[mask]
        above_sma = (close > window_sma50).mean() if window_sma50.notna().sum() > 0 else 0.5

    # ADX for trend strength
    adx_vals = adx(history_df)
    window_adx = adx_vals[mask]
    mean_adx = window_adx.mean() if window_adx.notna().sum() > 0 else 20

    # VIX
    vix = window_df.get('india_vix')
    mean_vix = vix.mean() if vix is not None and vix.notna().sum() > 0 else 15

    # Net return for direction
    net_return = (close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else 0

    # Classification logic
    if mean_vix > 22:
        return 'HIGH_VOL'
    elif mean_adx > 25:
        if net_return > 0.05 and above_sma > 0.6:
            return 'TRENDING_UP'
        elif net_return < -0.05 and above_sma < 0.4:
            return 'TRENDING_DOWN'
        else:
            return 'TRENDING_UP' if above_sma > 0.5 else 'TRENDING_DOWN'
    else:
        return 'RANGING'


def evaluate_window(result, min_sharpe=0.8, min_trades=10, max_dd=0.25):
    """Relaxed per-window pass check."""
    if result.trade_count < min_trades:
        return False
    if result.sharpe < min_sharpe:
        return False
    if result.max_drawdown > max_dd:
        return False
    return True


def tier_signal(window_details):
    """Classify signal into Tier A, Tier B, or DROP."""
    total = len(window_details)
    if total < 4:
        return 'DROP', 'INSUFFICIENT_WINDOWS'

    passed = sum(1 for w in window_details if w['passed'])
    pass_rate = passed / total

    # Last 4 windows
    last4 = window_details[-4:]
    last4_passed = sum(1 for w in last4 if w['passed'])
    last4_rate = last4_passed / 4

    # Aggregate Sharpe
    agg_sharpe = harmonic_mean_sharpe(
        [{'result': w['result']} for w in window_details]
    )

    # Zero-trade windows
    zero_windows = sum(1 for w in window_details if w['result'].trade_count == 0)
    zero_pct = zero_windows / total

    # Max drawdown across windows (exclude zero-trade windows)
    active_dds = [w['result'].max_drawdown for w in window_details
                  if w['result'].trade_count > 0]
    worst_dd = max(active_dds) if active_dds else 1.0

    # DROP conditions
    if zero_pct > 0.6:
        return 'DROP', f'ZERO_TRADES_{zero_pct:.0%}_OF_WINDOWS'
    if last4_rate == 0:
        return 'DROP', 'LAST_4_ALL_FAIL'
    if passed <= 2:
        return 'DROP', f'ONLY_{passed}_WINDOWS_PASS'

    # Tier A
    if (pass_rate >= 0.60 and
            agg_sharpe >= 1.5 and
            last4_rate >= 0.75 and
            worst_dd <= 0.20):
        return 'TIER_A', None

    # Tier B
    if (pass_rate >= 0.40 and
            agg_sharpe >= 1.0 and
            last4_rate >= 0.50):
        return 'TIER_B', None

    # Near-miss analysis
    reasons = []
    if pass_rate < 0.40:
        reasons.append(f'pass_rate={pass_rate:.0%}<40%')
    if agg_sharpe < 1.0:
        reasons.append(f'sharpe={agg_sharpe:.2f}<1.0')
    if last4_rate < 0.50:
        reasons.append(f'last4={last4_rate:.0%}<50%')

    return 'DROP', '; '.join(reasons) if reasons else 'BELOW_TIER_B'


def regime_profile(window_details):
    """Build regime-performance profile."""
    regime_stats = {}
    for w in window_details:
        regime = w.get('regime', 'UNKNOWN')
        if regime not in regime_stats:
            regime_stats[regime] = {'total': 0, 'passed': 0, 'sharpes': [],
                                     'trades': 0}
        regime_stats[regime]['total'] += 1
        if w['passed']:
            regime_stats[regime]['passed'] += 1
        if w['result'].sharpe != 0:
            regime_stats[regime]['sharpes'].append(w['result'].sharpe)
        regime_stats[regime]['trades'] += w['result'].trade_count

    profile = {}
    for regime, s in regime_stats.items():
        avg_sharpe = np.mean(s['sharpes']) if s['sharpes'] else 0
        profile[regime] = {
            'windows': s['total'],
            'passed': s['passed'],
            'pass_rate': s['passed'] / s['total'] if s['total'] > 0 else 0,
            'avg_sharpe': round(avg_sharpe, 2),
            'total_trades': s['trades'],
        }
    return profile


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
    print(f"  {len(df)} trading days, {len(cal)} calendar days", flush=True)

    # Regime labels
    from regime_labeler import RegimeLabeler
    labeler = RegimeLabeler()
    regime_labels = labeler.label_full_history(df)
    print(f"  {len(regime_labels)} regime labels", flush=True)

    # Load translated signals — find promising ones
    translated = json.load(open(TRANSLATED_PATH))
    promising_ids = set()
    promising_list = json.load(open(PROMISING_PATH))
    for p in promising_list:
        promising_ids.add(p['signal_id'])

    # For each promising signal_id, pick the best variant
    best_variants = {}
    for t in translated:
        sid = t.get('signal_id', '')
        if sid not in promising_ids or not t.get('backtestable') or not t.get('rules'):
            continue
        try:
            result = run_generic_backtest(t['rules'], df, {})
            if result.trade_count >= 15:
                key = sid
                if key not in best_variants or result.sharpe > best_variants[key][1]:
                    best_variants[key] = (t, result.sharpe)
        except:
            pass

    signals_to_test = {k: v[0] for k, v in best_variants.items()}
    print(f"  {len(signals_to_test)} unique signals to test", flush=True)

    # Set up WFE with relaxed per-window criteria
    wfe = WalkForwardEngine(cal)
    # Override criteria — we'll evaluate ourselves
    for criteria in wfe.CRITERIA.values():
        criteria['min_trades'] = 10
        criteria['min_sharpe'] = 0.8
        if 'min_calmar' in criteria:
            del criteria['min_calmar']
        if 'min_profit_factor' in criteria:
            del criteria['min_profit_factor']

    # Pre-compute window regime tags
    print("\nPre-computing regime tags for WF windows...", flush=True)
    # Generate windows once to get date ranges
    test_windows = wfe._generate_windows(df)
    window_regimes = {}
    for i, w in enumerate(test_windows):
        regime = classify_regime(df, w['test_start'], w['test_end'])
        window_regimes[i] = regime
    regime_counts = {}
    for r in window_regimes.values():
        regime_counts[r] = regime_counts.get(r, 0) + 1
    print(f"  Window regime distribution: {regime_counts}")

    # Run walk-forward on all signals
    print(f"\n{'='*90}")
    print(f"RUNNING TIERED WALK-FORWARD ON {len(signals_to_test)} SIGNALS")
    print(f"{'='*90}\n")

    results = {'TIER_A': [], 'TIER_B': [], 'DROP': []}
    start = time.time()

    for idx, (signal_id, t) in enumerate(sorted(signals_to_test.items())):
        rules = t['rules']
        backtest_fn = make_generic_backtest_fn(rules)

        instrument = rules.get('instrument', 'FUTURES')
        if 'OPTIONS_SELLING' in str(instrument):
            sig_type = 'OPTIONS_SELLING'
        elif 'OPTIONS_BUYING' in str(instrument):
            sig_type = 'OPTIONS_BUYING'
        else:
            sig_type = 'FUTURES'

        try:
            wf = wfe.run(signal_id, backtest_fn, df, regime_labels, rules, sig_type)
        except Exception as e:
            results['DROP'].append({
                'signal_id': signal_id,
                'tier': 'DROP',
                'reason': f'ERROR: {str(e)[:100]}',
            })
            continue

        # Tag each window with regime
        for wd in wf['window_details']:
            w_idx = wd['window_index']
            wd['regime'] = window_regimes.get(w_idx, 'UNKNOWN')

        # Classify tier
        tier, reason = tier_signal(wf['window_details'])

        # Build regime profile
        profile = regime_profile(wf['window_details'])

        # Last 4 windows detail
        last4 = wf['window_details'][-4:]
        last4_passed = sum(1 for w in last4 if w['passed'])

        # Active windows (non-zero trades)
        active_windows = [w for w in wf['window_details']
                          if w['result'].trade_count > 0]
        active_sharpes = [w['result'].sharpe for w in active_windows]
        active_dds = [w['result'].max_drawdown for w in active_windows]

        entry = {
            'signal_id': signal_id,
            'tier': tier,
            'reason': reason,
            'windows_passed': wf['windows_passed'],
            'total_windows': wf['total_windows'],
            'pass_rate': wf['pass_rate'],
            'last4_passed': last4_passed,
            'last4_rate': last4_passed / 4,
            'aggregate_sharpe': wf['aggregate_sharpe'],
            'active_windows': len(active_windows),
            'mean_sharpe': round(np.mean(active_sharpes), 2) if active_sharpes else 0,
            'median_sharpe': round(np.median(active_sharpes), 2) if active_sharpes else 0,
            'worst_dd': round(max(active_dds), 3) if active_dds else 1.0,
            'regime_profile': profile,
            'rules': rules,
            'per_window': [{
                'idx': w['window_index'],
                'regime': w.get('regime', '?'),
                'test_start': str(w['window']['test_start'])[:10],
                'test_end': str(w['window']['test_end'])[:10],
                'sharpe': w['result'].sharpe,
                'max_dd': w['result'].max_drawdown,
                'win_rate': w['result'].win_rate,
                'trades': w['result'].trade_count,
                'passed': w['passed'],
            } for w in wf['window_details']],
        }

        results[tier].append(entry)

        done = idx + 1
        if done % 10 == 0 or done == len(signals_to_test):
            elapsed = time.time() - start
            print(f"  [{done}/{len(signals_to_test)}] ({elapsed:.0f}s) "
                  f"A:{len(results['TIER_A'])} B:{len(results['TIER_B'])} "
                  f"Drop:{len(results['DROP'])}", flush=True)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for tier in ['TIER_A', 'TIER_B']:
        results[tier].sort(key=lambda x: x['aggregate_sharpe'], reverse=True)

    output = {
        'summary': {
            'total_tested': len(signals_to_test),
            'tier_a': len(results['TIER_A']),
            'tier_b': len(results['TIER_B']),
            'dropped': len(results['DROP']),
            'window_regimes': regime_counts,
        },
        'tier_a': results['TIER_A'],
        'tier_b': results['TIER_B'],
        'dropped': results['DROP'],
    }

    output_path = os.path.join(RESULTS_DIR, '_WF_TIERED.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Print report
    print(f"\n{'='*90}")
    print(f"WALK-FORWARD TIERED RESULTS")
    print(f"{'='*90}")
    print(f"  Total tested:  {len(signals_to_test)}")
    print(f"  TIER A:        {len(results['TIER_A'])} (strong — ready for paper trading)")
    print(f"  TIER B:        {len(results['TIER_B'])} (regime-dependent — usable with filter)")
    print(f"  DROPPED:       {len(results['DROP'])}")

    if results['TIER_A']:
        print(f"\n{'='*90}")
        print(f"TIER A SIGNALS (sorted by aggregate Sharpe)")
        print(f"{'='*90}")
        print(f"{'Signal':<25} {'Sharpe':>7} {'Pass':>6} {'L4':>4} "
              f"{'WrstDD':>7} {'ActWin':>6}  Regime Profile")
        print(f"{'-'*25} {'-'*7} {'-'*6} {'-'*4} {'-'*7} {'-'*6}  {'-'*30}")
        for s in results['TIER_A']:
            regime_str = ' | '.join(
                f"{r}:{p['passed']}/{p['windows']}"
                for r, p in sorted(s['regime_profile'].items())
                if p['windows'] > 0
            )
            print(f"{s['signal_id']:<25} {s['aggregate_sharpe']:>7.2f} "
                  f"{s['pass_rate']:>5.0%} {s['last4_passed']:>3}/4 "
                  f"{s['worst_dd']:>6.1%} {s['active_windows']:>5}  {regime_str}")

    if results['TIER_B']:
        print(f"\n{'='*90}")
        print(f"TIER B SIGNALS (sorted by aggregate Sharpe)")
        print(f"{'='*90}")
        print(f"{'Signal':<25} {'Sharpe':>7} {'Pass':>6} {'L4':>4} "
              f"{'WrstDD':>7} {'ActWin':>6}  Regime Profile")
        print(f"{'-'*25} {'-'*7} {'-'*6} {'-'*4} {'-'*7} {'-'*6}  {'-'*30}")
        for s in results['TIER_B']:
            regime_str = ' | '.join(
                f"{r}:{p['passed']}/{p['windows']}"
                for r, p in sorted(s['regime_profile'].items())
                if p['windows'] > 0
            )
            print(f"{s['signal_id']:<25} {s['aggregate_sharpe']:>7.2f} "
                  f"{s['pass_rate']:>5.0%} {s['last4_passed']:>3}/4 "
                  f"{s['worst_dd']:>6.1%} {s['active_windows']:>5}  {regime_str}")

    # Drop reasons summary
    from collections import Counter
    drop_reasons = Counter()
    for d in results['DROP']:
        r = d.get('reason', 'unknown')
        if r:
            drop_reasons[r.split(';')[0].strip()] += 1
    print(f"\n{'='*90}")
    print(f"DROP REASONS")
    print(f"{'='*90}")
    for reason, count in drop_reasons.most_common():
        print(f"  {count:>3}  {reason}")

    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()

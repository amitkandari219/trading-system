"""
Walk-forward with alpha filter: reject signals with Nifty correlation > 0.5.

Path B approach: keep the translation pipeline as-is, but filter out
beta-dressed-as-alpha by requiring low correlation to the underlying.

Pipeline:
1. Run each signal on full dataset
2. REJECT if |nifty_correlation| > 0.5 (momentum proxy)
3. REJECT if direction=LONG only and no short entries (long-biased)
4. Run walk-forward on survivors with tiered criteria
5. Regime-tag results

This finds signals with genuine edge independent of market direction.
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
RESULTS_DIR = 'backtest_results'

MAX_NIFTY_CORR = 0.50


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


def regime_profile(window_details):
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
            'pass_rate': round(s['passed'] / s['total'], 2) if s['total'] > 0 else 0,
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
    print(f"  {len(df)} trading days", flush=True)

    from regime_labeler import RegimeLabeler
    labeler = RegimeLabeler()
    regime_labels = labeler.label_full_history(df)

    # Load ALL translated signals (not just promising — rescreen with correlation)
    translated = json.load(open(TRANSLATED_PATH))
    backtestable = [t for t in translated if t.get('backtestable') and t.get('rules')]
    print(f"  {len(backtestable)} backtestable signals", flush=True)

    # ================================================================
    # STEP 1: Full-dataset screen with correlation filter
    # ================================================================
    print(f"\n{'='*90}")
    print(f"STEP 1: ALPHA SCREEN — reject |Nifty correlation| > {MAX_NIFTY_CORR}")
    print(f"{'='*90}")

    alpha_candidates = []  # (signal_id, translated_entry, full_result)
    rejected_corr = 0
    rejected_long_bias = 0
    rejected_no_trades = 0
    rejected_error = 0
    seen_ids = {}  # signal_id -> best (by sharpe) that passes filters

    start = time.time()
    for i, t in enumerate(backtestable):
        sid = t['signal_id']
        rules = t['rules']

        try:
            result = run_generic_backtest(rules, df, {})
        except Exception:
            rejected_error += 1
            continue

        if result.trade_count < 15:
            rejected_no_trades += 1
            continue

        # Correlation filter — the key alpha gate
        if abs(result.nifty_correlation) > MAX_NIFTY_CORR:
            rejected_corr += 1
            continue

        # Long-bias filter: reject if long-only with no short entries
        # AND positive nifty correlation (even if under 0.5)
        entry_short = rules.get('entry_short', [])
        direction = rules.get('direction', '')
        if (not entry_short and direction == 'LONG' and
                result.nifty_correlation > 0.3):
            rejected_long_bias += 1
            continue

        # Keep best variant per signal_id
        if sid not in seen_ids or result.sharpe > seen_ids[sid][1].sharpe:
            seen_ids[sid] = (t, result)

        done = i + 1
        if done % 100 == 0 or done == len(backtestable):
            elapsed = time.time() - start
            print(f"  [{done}/{len(backtestable)}] ({elapsed:.0f}s) "
                  f"Alpha:{len(seen_ids)} CorrKill:{rejected_corr} "
                  f"LongBias:{rejected_long_bias} NoTrade:{rejected_no_trades}",
                  flush=True)

    alpha_candidates = list(seen_ids.values())
    print(f"\n  ALPHA SCREEN RESULTS:")
    print(f"    Input:              {len(backtestable)}")
    print(f"    Rejected (corr):    {rejected_corr}")
    print(f"    Rejected (long):    {rejected_long_bias}")
    print(f"    Rejected (trades):  {rejected_no_trades}")
    print(f"    Rejected (errors):  {rejected_error}")
    print(f"    ALPHA CANDIDATES:   {len(alpha_candidates)}")

    if not alpha_candidates:
        print("\n  NO ALPHA CANDIDATES SURVIVED. Pipeline needs deeper fix.")
        # Show the closest misses
        print("\n  Closest misses (lowest correlation among rejects):")
        all_results = []
        for t in backtestable:
            try:
                r = run_generic_backtest(t['rules'], df, {})
                if r.trade_count >= 15:
                    all_results.append((t['signal_id'], r.nifty_correlation,
                                       r.sharpe, r.trade_count, r.win_rate,
                                       t['rules'].get('direction', '?'),
                                       len(t['rules'].get('entry_short', [])) > 0))
            except:
                pass
        all_results.sort(key=lambda x: abs(x[1]))
        for sid, corr, sharpe, tc, wr, d, has_short in all_results[:15]:
            print(f"    {sid:<25} corr={corr:>6.3f} sharpe={sharpe:>6.2f} "
                  f"trades={tc:>4} WR={wr:.0%} dir={d} short={has_short}")

        # Save for analysis
        with open(os.path.join(RESULTS_DIR, '_ALPHA_MISSES.json'), 'w') as f:
            json.dump([{
                'signal_id': x[0], 'nifty_corr': x[1], 'sharpe': x[2],
                'trades': x[3], 'win_rate': x[4], 'direction': x[5],
                'has_short': x[6]
            } for x in all_results[:50]], f, indent=2)
        print(f"\n  Saved top 50 misses to {RESULTS_DIR}/_ALPHA_MISSES.json")
        return

    # ================================================================
    # STEP 2: Walk-forward on alpha candidates
    # ================================================================
    print(f"\n{'='*90}")
    print(f"STEP 2: WALK-FORWARD ON {len(alpha_candidates)} ALPHA CANDIDATES")
    print(f"{'='*90}")

    wfe = WalkForwardEngine(cal)
    for criteria in wfe.CRITERIA.values():
        criteria['min_trades'] = 10
        criteria['min_sharpe'] = 0.8
        if 'min_calmar' in criteria:
            del criteria['min_calmar']
        if 'min_profit_factor' in criteria:
            del criteria['min_profit_factor']

    test_windows = wfe._generate_windows(df)
    window_regimes = {i: classify_regime(df, w['test_start'], w['test_end'])
                      for i, w in enumerate(test_windows)}

    results = {'TIER_A': [], 'TIER_B': [], 'DROP': []}

    for idx, (t, full_result) in enumerate(alpha_candidates):
        signal_id = t['signal_id']
        rules = t['rules']
        backtest_fn = make_generic_backtest_fn(rules)

        instrument = rules.get('instrument', 'FUTURES')
        sig_type = ('OPTIONS_SELLING' if 'OPTIONS_SELLING' in str(instrument)
                    else 'OPTIONS_BUYING' if 'OPTIONS_BUYING' in str(instrument)
                    else 'FUTURES')

        try:
            wf = wfe.run(signal_id, backtest_fn, df, regime_labels, rules, sig_type)
        except Exception as e:
            results['DROP'].append({
                'signal_id': signal_id,
                'tier': 'DROP',
                'reason': f'ERROR: {str(e)[:100]}',
            })
            continue

        for wd in wf['window_details']:
            wd['regime'] = window_regimes.get(wd['window_index'], 'UNKNOWN')

        tier, reason = tier_signal(wf['window_details'])
        profile = regime_profile(wf['window_details'])
        last4 = wf['window_details'][-4:]
        last4_passed = sum(1 for w in last4 if w['passed'])
        active_windows = [w for w in wf['window_details']
                          if w['result'].trade_count > 0]
        active_dds = [w['result'].max_drawdown for w in active_windows]

        entry = {
            'signal_id': signal_id,
            'tier': tier,
            'reason': reason,
            'nifty_correlation': full_result.nifty_correlation,
            'full_sharpe': full_result.sharpe,
            'full_trades': full_result.trade_count,
            'full_win_rate': full_result.win_rate,
            'full_max_dd': full_result.max_drawdown,
            'full_annual_return': full_result.annual_return,
            'windows_passed': wf['windows_passed'],
            'total_windows': wf['total_windows'],
            'pass_rate': wf['pass_rate'],
            'last4_passed': last4_passed,
            'last4_rate': last4_passed / 4,
            'aggregate_sharpe': wf['aggregate_sharpe'],
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
                'nifty_corr': w['result'].nifty_correlation,
                'passed': w['passed'],
            } for w in wf['window_details']],
        }
        results[tier].append(entry)

        done = idx + 1
        if done % 5 == 0 or done == len(alpha_candidates):
            print(f"  [{done}/{len(alpha_candidates)}] "
                  f"A:{len(results['TIER_A'])} B:{len(results['TIER_B'])} "
                  f"Drop:{len(results['DROP'])}", flush=True)

    # Save and report
    for tier in ['TIER_A', 'TIER_B']:
        results[tier].sort(key=lambda x: x['aggregate_sharpe'], reverse=True)

    output = {
        'filters': {
            'max_nifty_correlation': MAX_NIFTY_CORR,
            'min_trades': 15,
            'long_bias_corr_threshold': 0.3,
        },
        'summary': {
            'input_backtestable': len(backtestable),
            'alpha_candidates': len(alpha_candidates),
            'tier_a': len(results['TIER_A']),
            'tier_b': len(results['TIER_B']),
            'dropped': len(results['DROP']),
        },
        'tier_a': results['TIER_A'],
        'tier_b': results['TIER_B'],
        'dropped': results['DROP'],
    }

    output_path = os.path.join(RESULTS_DIR, '_WF_ALPHA.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*90}")
    print(f"ALPHA WALK-FORWARD RESULTS")
    print(f"{'='*90}")
    print(f"  Input backtestable:  {len(backtestable)}")
    print(f"  Alpha candidates:    {len(alpha_candidates)} (|corr| <= {MAX_NIFTY_CORR})")
    print(f"  TIER A:              {len(results['TIER_A'])}")
    print(f"  TIER B:              {len(results['TIER_B'])}")
    print(f"  DROPPED:             {len(results['DROP'])}")

    for tier_name in ['TIER_A', 'TIER_B']:
        if results[tier_name]:
            print(f"\n{'='*90}")
            print(f"{tier_name} SIGNALS")
            print(f"{'='*90}")
            print(f"{'Signal':<25} {'Sharpe':>7} {'NCorr':>6} {'Pass':>6} {'L4':>4} "
                  f"{'DD':>6} {'Trades':>6}  Regime Profile")
            print(f"{'-'*25} {'-'*7} {'-'*6} {'-'*6} {'-'*4} {'-'*6} {'-'*6}  {'-'*30}")
            for s in results[tier_name]:
                regime_str = ' | '.join(
                    f"{r}:{p['passed']}/{p['windows']}"
                    for r, p in sorted(s['regime_profile'].items())
                    if p['windows'] > 0
                )
                print(f"{s['signal_id']:<25} {s['aggregate_sharpe']:>7.2f} "
                      f"{s['nifty_correlation']:>6.3f} {s['pass_rate']:>5.0%} "
                      f"{s['last4_passed']:>3}/4 {s['worst_dd']:>5.1%} "
                      f"{s['full_trades']:>5}  {regime_str}")

    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()

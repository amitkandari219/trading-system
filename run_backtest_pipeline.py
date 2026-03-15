"""
End-to-end backtest pipeline:
1. Load approved signals
2. Translate text conditions → structured rules (via Haiku)
3. Run walk-forward backtest on each translated signal
4. Save results

Usage:
    python run_backtest_pipeline.py --step translate   # Step 1: translate only
    python run_backtest_pipeline.py --step backtest    # Step 2: backtest translated
    python run_backtest_pipeline.py --step all         # Both steps
    python run_backtest_pipeline.py --step validate    # Validate engine first
"""

import argparse
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.WARNING)

from backtest.generic_backtest import make_generic_backtest_fn
from backtest.signal_translator import SignalTranslator
from backtest.types import BacktestResult
from backtest.validator import validate_backtest_engine, make_reference_backtest_fn
from backtest.walk_forward import WalkForwardEngine
from config.settings import DATABASE_DSN
from regime_labeler import RegimeLabeler


APPROVED_PATH = 'extraction_results/approved/_ALL.json'
TRANSLATED_PATH = 'extraction_results/translated_signals.json'
RESULTS_DIR = 'backtest_results'


def load_nifty_data():
    """Load Nifty OHLCV from database."""
    print("Loading Nifty data from database...", flush=True)
    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql("""
        SELECT date, open, high, low, close, volume, india_vix
        FROM nifty_daily
        ORDER BY date
    """, conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Loaded {len(df)} trading days "
          f"({df['date'].min()} to {df['date'].max()})", flush=True)
    return df


def load_regime_labels(history_df):
    """Compute regime labels for all dates."""
    print("Computing regime labels...", flush=True)
    labeler = RegimeLabeler()
    labels = labeler.label_full_history(history_df)
    print(f"  {len(labels)} regime labels computed", flush=True)
    return labels


def step_validate(history_df, regime_labels):
    """Step 0: Validate backtest engine."""
    print("\n" + "=" * 60)
    print("STEP 0: VALIDATING BACKTEST ENGINE")
    print("=" * 60)
    backtest_fn = make_reference_backtest_fn()
    passed = validate_backtest_engine(backtest_fn, history_df, regime_labels)
    if not passed:
        print("\nENGINE VALIDATION FAILED — fix before proceeding")
        return False
    return True


def step_translate():
    """Step 1: Translate approved signals to structured rules."""
    print("\n" + "=" * 60)
    print("STEP 1: TRANSLATING SIGNALS → STRUCTURED RULES")
    print("=" * 60)

    signals = json.load(open(APPROVED_PATH))
    print(f"Loaded {len(signals)} approved signals", flush=True)

    translator = SignalTranslator(num_workers=6)
    results = translator.translate_batch(signals, output_path=TRANSLATED_PATH)

    backtestable = [r for r in results if r.get('backtestable')]
    print(f"\nBacktestable signals: {len(backtestable)}/{len(results)}")
    return results


def _infer_signal_type(rules: dict) -> str:
    """Infer WFE signal type from translated rules."""
    instrument = rules.get('instrument', 'FUTURES')
    if 'OPTIONS_SELLING' in str(instrument):
        return 'OPTIONS_SELLING'
    elif 'OPTIONS_BUYING' in str(instrument):
        return 'OPTIONS_BUYING'
    elif 'SPREAD' in str(instrument) or 'COMBINED' in str(instrument):
        return 'COMBINED'
    return 'FUTURES'


def step_backtest(history_df, regime_labels, calendar_df):
    """Step 2: Run walk-forward backtest on translated signals."""
    print("\n" + "=" * 60)
    print("STEP 2: RUNNING WALK-FORWARD BACKTESTS")
    print("=" * 60)

    translated = json.load(open(TRANSLATED_PATH))
    backtestable = [t for t in translated if t.get('backtestable') and t.get('rules')]
    print(f"Backtestable signals: {len(backtestable)}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    wfe = WalkForwardEngine(calendar_df)

    results = []
    passed_count = 0
    failed_count = 0
    error_count = 0
    start = time.time()

    for i, t in enumerate(backtestable):
        signal_id = t['signal_id']
        rules = t['rules']

        try:
            backtest_fn = make_generic_backtest_fn(rules)
            signal_type = _infer_signal_type(rules)
            params = rules  # rules dict doubles as params

            wf_result = wfe.run(
                signal_id, backtest_fn, history_df,
                regime_labels, params, signal_type
            )

            result = {
                'signal_id': signal_id,
                'passed': wf_result['overall_pass'],
                'aggregate_sharpe': wf_result.get('aggregate_sharpe', 0),
                'windows_passed': wf_result.get('windows_passed', 0),
                'windows_total': wf_result.get('total_windows', 0),
                'reason': t.get('reason', ''),
            }

            if wf_result['overall_pass']:
                passed_count += 1
                detail_path = os.path.join(RESULTS_DIR, f'{signal_id}.json')
                with open(detail_path, 'w') as f:
                    json.dump({
                        'signal_id': signal_id,
                        'rules': rules,
                        'wf_result': _serialize_wf_result(wf_result),
                    }, f, indent=2)
            else:
                failed_count += 1

            results.append(result)

        except Exception as e:
            error_count += 1
            results.append({
                'signal_id': signal_id,
                'passed': False,
                'error': str(e)[:200],
            })

        # Progress
        done = i + 1
        if done % 50 == 0 or done == len(backtestable):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{len(backtestable)}] ({rate:.1f}/s) "
                  f"Pass:{passed_count} Fail:{failed_count} Err:{error_count}",
                  flush=True)

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, '_SUMMARY.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'total_translated': len(translated),
            'total_backtestable': len(backtestable),
            'passed': passed_count,
            'failed': failed_count,
            'errors': error_count,
            'results': results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"BACKTEST COMPLETE")
    print(f"  Backtestable: {len(backtestable)}")
    print(f"  Passed WF:    {passed_count}")
    print(f"  Failed WF:    {failed_count}")
    print(f"  Errors:       {error_count}")
    print(f"  Results:      {summary_path}")
    print(f"{'='*60}")

    return results


def _serialize_wf_result(wf_result: dict) -> dict:
    """Make walk-forward result JSON-serializable."""
    serialized = {}
    for k, v in wf_result.items():
        if k == 'window_results':
            serialized[k] = []
            for w in v:
                wr = dict(w)
                if 'result' in wr and isinstance(wr['result'], BacktestResult):
                    wr['result'] = {
                        'sharpe': wr['result'].sharpe,
                        'calmar_ratio': wr['result'].calmar_ratio,
                        'max_drawdown': wr['result'].max_drawdown,
                        'win_rate': wr['result'].win_rate,
                        'profit_factor': wr['result'].profit_factor,
                        'avg_win_loss_ratio': wr['result'].avg_win_loss_ratio,
                        'trade_count': wr['result'].trade_count,
                        'nifty_correlation': wr['result'].nifty_correlation,
                        'annual_return': wr['result'].annual_return,
                        'drawdown_2020': wr['result'].drawdown_2020,
                    }
                # Convert window dates to strings
                if 'window' in wr:
                    wr['window'] = {k2: str(v2) for k2, v2 in wr['window'].items()}
                serialized[k].append(wr)
        elif isinstance(v, (int, float, str, bool, type(None))):
            serialized[k] = v
        else:
            serialized[k] = str(v)
    return serialized


def load_calendar():
    """Load market calendar from database."""
    print("Loading market calendar...", flush=True)
    conn = psycopg2.connect(DATABASE_DSN)
    cal = pd.read_sql("SELECT trading_date AS date, is_trading_day FROM market_calendar ORDER BY trading_date", conn)
    conn.close()
    cal['date'] = pd.to_datetime(cal['date'])
    print(f"  Calendar: {len(cal)} days", flush=True)
    return cal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest pipeline')
    parser.add_argument('--step', choices=['translate', 'backtest', 'validate', 'all'],
                        default='all')
    args = parser.parse_args()

    if args.step == 'translate':
        step_translate()

    elif args.step == 'validate':
        history_df = load_nifty_data()
        regime_labels = load_regime_labels(history_df)
        step_validate(history_df, regime_labels)

    elif args.step == 'backtest':
        history_df = load_nifty_data()
        regime_labels = load_regime_labels(history_df)
        calendar_df = load_calendar()
        step_validate(history_df, regime_labels)
        step_backtest(history_df, regime_labels, calendar_df)

    elif args.step == 'all':
        step_translate()
        history_df = load_nifty_data()
        regime_labels = load_regime_labels(history_df)
        calendar_df = load_calendar()
        if step_validate(history_df, regime_labels):
            step_backtest(history_df, regime_labels, calendar_df)

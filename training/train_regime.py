"""
Train and evaluate ML regime classifiers.

Trains HMM + RF on 2014-2022, tests on 2023-2024 OOS.
Compares with fixed-rule baseline.

Usage:
    python -m training.train_regime
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import psycopg2

from config.settings import DATABASE_DSN
from models.regime_features import compute_regime_features
from models.regime_hmm import RegimeHMM, REGIME_NAMES
from models.regime_rf import RegimeRF, label_regime_rules
from models.regime_ensemble import RegimeEnsemble


def main():
    # Load data
    print("Loading market data...", flush=True)
    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix "
        "FROM nifty_daily ORDER BY date", conn
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    print(f"  {len(df)} trading days: {df['date'].min().date()} to {df['date'].max().date()}")

    # Feature engineering
    print("Computing features...", flush=True)
    features, feature_cols = compute_regime_features(df)
    print(f"  {len(features)} rows, {len(feature_cols)} features")

    # Split: train (2014-2022), test (2023+)
    train = features[features['date'] < '2023-01-01'].reset_index(drop=True)
    test = features[features['date'] >= '2023-01-01'].reset_index(drop=True)
    print(f"  Train: {len(train)} days ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"  Test:  {len(test)} days ({test['date'].min().date()} to {test['date'].max().date()})")

    # ── FIXED RULES BASELINE ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  BASELINE: Fixed Rules")
    print(f"{'='*70}")
    train_rules = train.apply(label_regime_rules, axis=1)
    test_rules = test.apply(label_regime_rules, axis=1)
    print(f"  Train regime distribution:")
    for r, c in Counter(train_rules).most_common():
        print(f"    {r:<16s} {c:>5d} ({c/len(train)*100:.1f}%)")
    print(f"  Test regime distribution:")
    for r, c in Counter(test_rules).most_common():
        print(f"    {r:<16s} {c:>5d} ({c/len(test)*100:.1f}%)")

    # ── HMM ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MODEL 1: Hidden Markov Model (5 states)")
    print(f"{'='*70}")
    hmm = RegimeHMM(n_states=5)
    hmm_train_states = hmm.fit(train, feature_cols)
    hmm_train_labels = [hmm.state_map.get(s, 'RANGING') for s in hmm_train_states]

    print(f"  State summary:")
    for name, stats in hmm.get_state_summary().items():
        print(f"    {name:<16s} n={stats['count']:>4d}  ret_20d={stats['avg_returns_20d']:>6.1f}%  "
              f"adx={stats['avg_adx']:>4.1f}  vix={stats['avg_vix']:>4.1f}")

    hmm_test_labels, hmm_test_probs, hmm_test_conf = hmm.predict(test, feature_cols)
    print(f"\n  Test predictions:")
    for r, c in Counter(hmm_test_labels).most_common():
        print(f"    {r:<16s} {c:>5d} ({c/len(test)*100:.1f}%)")
    print(f"  Avg confidence: {np.mean(hmm_test_conf):.2f}")

    # ── RANDOM FOREST ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MODEL 2: Random Forest")
    print(f"{'='*70}")
    rf = RegimeRF()
    rf_train_labels = rf.fit(train, feature_cols)

    rf_test_labels, rf_test_probs, rf_test_conf = rf.predict(test)
    print(f"  Test predictions:")
    for r, c in Counter(rf_test_labels).most_common():
        print(f"    {r:<16s} {c:>5d} ({c/len(test)*100:.1f}%)")
    print(f"  Avg confidence: {np.mean(rf_test_conf):.2f}")

    print(f"\n  Top feature importances:")
    for feat, imp in list(rf.feature_importances().items())[:10]:
        print(f"    {feat:<25s} {imp:.3f}")

    # ── ENSEMBLE ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MODEL 3: Ensemble (HMM 60% + RF 40%)")
    print(f"{'='*70}")
    ensemble = RegimeEnsemble()
    ensemble.hmm = hmm
    ensemble.rf = rf

    ens_labels, ens_conf, ens_details = ensemble.predict(test, feature_cols)
    print(f"  Test predictions:")
    for r, c in Counter(ens_labels).most_common():
        print(f"    {r:<16s} {c:>5d} ({c/len(test)*100:.1f}%)")
    print(f"  Avg confidence: {np.mean(ens_conf):.2f}")

    # Agreement between models
    agree = sum(1 for d in ens_details if d['agree'])
    print(f"  HMM-RF agreement: {agree}/{len(ens_details)} ({agree/len(ens_details)*100:.0f}%)")

    # ── COMPARISON: Fixed Rules vs HMM vs RF vs Ensemble ──────────
    print(f"\n{'='*70}")
    print(f"  COMPARISON: OOS Test Period ({test['date'].min().date()} to {test['date'].max().date()})")
    print(f"{'='*70}")

    # For each method, compute avg 20d-forward return per predicted regime
    test_copy = test.copy()
    test_copy['fwd_return_20d'] = test_copy['close'].pct_change(20).shift(-20)
    test_copy['rules'] = test_rules.values
    test_copy['hmm'] = hmm_test_labels
    test_copy['rf'] = rf_test_labels
    test_copy['ensemble'] = ens_labels

    print(f"\n  Avg 20-day forward return by predicted regime:")
    print(f"  {'Method':<12s}", end='')
    for regime in REGIME_NAMES:
        print(f"  {regime:>14s}", end='')
    print()
    print("  " + "-" * 84)

    for method in ['rules', 'hmm', 'rf', 'ensemble']:
        print(f"  {method:<12s}", end='')
        for regime in REGIME_NAMES:
            mask = test_copy[method] == regime
            if mask.sum() > 0:
                avg_ret = test_copy.loc[mask, 'fwd_return_20d'].mean()
                n = mask.sum()
                print(f"  {avg_ret*100:>6.2f}% ({n:>3d})", end='')
            else:
                print(f"  {'--':>14s}", end='')
        print()

    # Key metric: TRENDING_UP should have highest forward returns
    print(f"\n  Quality check — TRENDING_UP prediction accuracy:")
    for method in ['rules', 'hmm', 'rf', 'ensemble']:
        mask = test_copy[method] == 'TRENDING_UP'
        if mask.sum() > 0:
            avg_ret = test_copy.loc[mask, 'fwd_return_20d'].mean()
            print(f"    {method:<12s}: {mask.sum():>4d} days predicted, avg fwd return = {avg_ret*100:.2f}%")
        else:
            print(f"    {method:<12s}: 0 days predicted")

    # Save models
    os.makedirs('models/saved', exist_ok=True)
    hmm.save('models/saved/regime_hmm.pkl')
    rf.save('models/saved/regime_rf.pkl')
    print(f"\n  Models saved to models/saved/")

    # Save results
    results = {
        'train_period': f"{train['date'].min().date()} to {train['date'].max().date()}",
        'test_period': f"{test['date'].min().date()} to {test['date'].max().date()}",
        'hmm_state_summary': hmm.get_state_summary(),
        'rf_feature_importances': rf.feature_importances(),
        'hmm_rf_agreement': f"{agree}/{len(ens_details)}",
    }
    with open('models/saved/regime_training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()

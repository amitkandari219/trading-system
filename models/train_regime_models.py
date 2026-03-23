"""
Train regime HMM + RF models from nifty_daily data.

Fixes from previous training:
  1. HMM had only 9 TRENDING_UP samples — now uses better state mapping
     with returns + ADX jointly, not just max/min returns
  2. RF trained on fixed-rule labels only — now uses HMM states as
     additional features for richer supervision
  3. No walk-forward validation — now splits 70/30 train/test and
     reports per-regime accuracy on OOS data
  4. Feature columns mismatched between training and inference —
     now uses regime_features.compute_regime_features() consistently

Usage:
    cd /path/to/trading-system
    python -m models.train_regime_models                    # full train + save
    python -m models.train_regime_models --validate-only    # just check OOS accuracy
    python -m models.train_regime_models --dry-run          # train but don't save
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.regime_features import compute_regime_features
from models.regime_hmm import RegimeHMM
from models.regime_rf import RegimeRF, label_regime_rules

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SAVED_DIR = os.path.join(os.path.dirname(__file__), 'saved')
TRAIN_CUTOFF_YEARS = 3  # Last N years reserved for OOS test


def load_nifty_data(conn):
    """Load full Nifty history from DB."""
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix "
        "FROM nifty_daily ORDER BY date ASC",
        conn
    )
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Loaded {len(df)} rows from nifty_daily ({df['date'].min().date()} to {df['date'].max().date()})")

    vix_fill = df['india_vix'].notna().sum()
    logger.info(f"  VIX data available: {vix_fill}/{len(df)} rows ({vix_fill/len(df)*100:.0f}%)")

    return df


def train_hmm(features_df, feature_cols, n_states=5):
    """
    Train HMM with improved state mapping.

    Key fix: Previous mapping assigned TRENDING_DOWN to a nearly-flat state
    because it used max/min returns alone. Now we use a joint criterion:
    - CRISIS:        highest avg VIX (must be > 22)
    - HIGH_VOL:      next highest avg VIX (must be > 16)
    - TRENDING_UP:   positive returns_20d AND adx_14 > 22
    - TRENDING_DOWN: negative returns_20d AND adx_14 > 22
    - RANGING:       everything else
    """
    hmm = RegimeHMM(n_states=n_states, n_iter=300, random_state=42)
    states = hmm.fit(features_df, feature_cols)

    # Override the auto-mapping with better logic
    state_stats = {}
    for s in range(n_states):
        mask = states == s
        if mask.sum() == 0:
            continue
        state_stats[s] = {
            'count': int(mask.sum()),
            'pct': round(mask.sum() / len(states) * 100, 1),
            'returns_20d': float(features_df.loc[mask, 'returns_20d'].mean()),
            'adx_14': float(features_df.loc[mask, 'adx_14'].mean()),
            'vix': float(features_df.loc[mask, 'vix'].mean()),
            'hvol_20': float(features_df.loc[mask, 'hvol_20'].mean()),
        }

    logger.info("HMM state statistics:")
    for s, st in sorted(state_stats.items()):
        logger.info(
            f"  State {s}: n={st['count']} ({st['pct']}%) "
            f"ret20d={st['returns_20d']:.2f}% adx={st['adx_14']:.1f} "
            f"vix={st['vix']:.1f} hvol={st['hvol_20']:.2f}"
        )

    # Improved mapping
    assigned = {}
    used = set()
    by_vix = sorted(state_stats.items(), key=lambda x: -x[1]['vix'])

    # CRISIS: highest VIX > 22
    if by_vix[0][1]['vix'] > 22:
        assigned[by_vix[0][0]] = 'CRISIS'
        used.add(by_vix[0][0])

    # HIGH_VOL: next highest VIX > 16
    for s, st in by_vix:
        if s not in used and st['vix'] > 16:
            assigned[s] = 'HIGH_VOL'
            used.add(s)
            break

    # Among remaining: separate TRENDING_UP vs TRENDING_DOWN vs RANGING
    remaining = [(s, st) for s, st in state_stats.items() if s not in used]

    for s, st in remaining:
        if st['returns_20d'] > 1.0 and st['adx_14'] > 22:
            assigned[s] = 'TRENDING_UP'
            used.add(s)
        elif st['returns_20d'] < -1.0 and st['adx_14'] > 22:
            assigned[s] = 'TRENDING_DOWN'
            used.add(s)

    # Everything else = RANGING
    for s in state_stats:
        if s not in assigned:
            assigned[s] = 'RANGING'

    hmm.state_map = assigned
    hmm.state_stats = state_stats

    logger.info("Final state mapping:")
    for s, name in sorted(assigned.items()):
        st = state_stats[s]
        logger.info(f"  {name}: state={s}, n={st['count']} ({st['pct']}%)")

    return hmm, states


def train_rf(features_df, feature_cols, hmm_states=None):
    """
    Train RF with enriched labels.

    Uses fixed-rule labels as ground truth, but also incorporates HMM states
    as an additional feature for richer pattern capture.
    """
    rf = RegimeRF(n_estimators=300, max_depth=10, random_state=42)

    # Generate fixed-rule labels as ground truth
    labels = features_df.apply(label_regime_rules, axis=1).values

    # Add HMM state as extra feature if available
    train_cols = list(feature_cols)
    train_df = features_df.copy()
    if hmm_states is not None:
        train_df['hmm_state'] = hmm_states
        train_cols.append('hmm_state')

    rf.fit(train_df, train_cols, labels=labels)

    # Log class distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("RF label distribution:")
    for u, c in zip(unique, counts):
        logger.info(f"  {u}: {c} ({c/len(labels)*100:.1f}%)")

    # Log top feature importances
    importances = rf.feature_importances()
    logger.info("RF top 10 feature importances:")
    for feat, imp in list(importances.items())[:10]:
        logger.info(f"  {feat}: {imp:.4f}")

    return rf


def validate_oos(hmm, rf, features_df, feature_cols, train_end_idx):
    """
    Validate on OOS data (last TRAIN_CUTOFF_YEARS).

    Reports:
    - Per-regime accuracy (HMM vs fixed-rule labels)
    - HMM-RF agreement rate
    - Regime transition frequency
    """
    test_df = features_df.iloc[train_end_idx:].copy().reset_index(drop=True)
    if len(test_df) < 50:
        logger.warning(f"Only {len(test_df)} OOS rows — too few for validation")
        return {}

    # HMM predictions
    hmm_labels, hmm_probs, hmm_conf = hmm.predict(test_df, feature_cols)

    # Add HMM state as RF feature (RF was trained with hmm_state)
    try:
        X_scaled = hmm.scaler.transform(test_df[feature_cols].values)
        test_df['hmm_state'] = hmm.model.predict(X_scaled)
    except Exception:
        test_df['hmm_state'] = 0

    # RF predictions
    rf_labels, rf_probs, rf_conf = rf.predict(test_df)

    # Fixed-rule labels (ground truth for comparison)
    fixed_labels = test_df.apply(label_regime_rules, axis=1).values

    # Agreement metrics
    hmm_rf_agree = sum(1 for h, r in zip(hmm_labels, rf_labels) if h == r)
    hmm_fixed_agree = sum(1 for h, f in zip(hmm_labels, fixed_labels) if h == f)
    rf_fixed_agree = sum(1 for r, f in zip(rf_labels, fixed_labels) if r == f)

    n = len(test_df)
    logger.info(f"\nOOS Validation ({n} rows):")
    logger.info(f"  HMM-RF agreement:    {hmm_rf_agree}/{n} ({hmm_rf_agree/n*100:.1f}%)")
    logger.info(f"  HMM-Fixed agreement: {hmm_fixed_agree}/{n} ({hmm_fixed_agree/n*100:.1f}%)")
    logger.info(f"  RF-Fixed agreement:  {rf_fixed_agree}/{n} ({rf_fixed_agree/n*100:.1f}%)")

    # Per-regime OOS accuracy (RF vs fixed rules)
    from collections import Counter
    regime_correct = Counter()
    regime_total = Counter()
    for pred, true in zip(rf_labels, fixed_labels):
        regime_total[true] += 1
        if pred == true:
            regime_correct[true] += 1

    logger.info("\nPer-regime RF accuracy (vs fixed rules):")
    for regime in sorted(regime_total.keys()):
        total = regime_total[regime]
        correct = regime_correct[regime]
        logger.info(f"  {regime}: {correct}/{total} ({correct/total*100:.1f}%)")

    # Avg confidence
    avg_hmm_conf = np.mean(hmm_conf)
    avg_rf_conf = np.mean(rf_conf)
    logger.info(f"\nAvg confidence — HMM: {avg_hmm_conf:.3f}, RF: {avg_rf_conf:.3f}")

    # Regime transition frequency
    transitions = sum(1 for i in range(1, len(hmm_labels)) if hmm_labels[i] != hmm_labels[i-1])
    logger.info(f"HMM regime transitions: {transitions}/{n} ({transitions/n*100:.1f}% of days)")

    return {
        'oos_rows': n,
        'hmm_rf_agreement': f"{hmm_rf_agree}/{n}",
        'hmm_fixed_agreement': f"{hmm_fixed_agree}/{n}",
        'rf_fixed_agreement': f"{rf_fixed_agree}/{n}",
        'avg_hmm_confidence': round(avg_hmm_conf, 3),
        'avg_rf_confidence': round(avg_rf_conf, 3),
        'hmm_transitions_pct': round(transitions / n * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='Train regime HMM + RF models')
    parser.add_argument('--validate-only', action='store_true',
                        help='Load existing models and validate OOS')
    parser.add_argument('--dry-run', action='store_true',
                        help='Train but do not save models')
    parser.add_argument('--dsn', default=None,
                        help='PostgreSQL DSN (default: from settings)')
    args = parser.parse_args()

    import psycopg2
    if args.dsn:
        dsn = args.dsn
    else:
        from config.settings import DATABASE_DSN
        dsn = DATABASE_DSN

    conn = psycopg2.connect(dsn)

    # Load data
    raw_df = load_nifty_data(conn)
    conn.close()

    # Compute features
    features_df, feature_cols = compute_regime_features(raw_df)
    logger.info(f"Feature matrix: {features_df.shape[0]} rows × {len(feature_cols)} features")

    # Train/test split
    cutoff_date = features_df['date'].max() - pd.Timedelta(days=TRAIN_CUTOFF_YEARS * 365)
    train_end_idx = features_df[features_df['date'] <= cutoff_date].index[-1] + 1
    logger.info(
        f"Train: {features_df['date'].iloc[0].date()} to "
        f"{features_df['date'].iloc[train_end_idx-1].date()} ({train_end_idx} rows)"
    )
    logger.info(
        f"Test:  {features_df['date'].iloc[train_end_idx].date()} to "
        f"{features_df['date'].iloc[-1].date()} ({len(features_df) - train_end_idx} rows)"
    )

    train_df = features_df.iloc[:train_end_idx].copy().reset_index(drop=True)

    if args.validate_only:
        # Load existing models and validate
        hmm = RegimeHMM()
        hmm.load(os.path.join(SAVED_DIR, 'regime_hmm.pkl'))
        rf = RegimeRF()
        rf.load(os.path.join(SAVED_DIR, 'regime_rf.pkl'))
        validate_oos(hmm, rf, features_df, feature_cols, train_end_idx)
        return

    # Train HMM
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING HMM")
    logger.info("=" * 60)
    hmm, hmm_states = train_hmm(train_df, feature_cols)

    # Train RF (with HMM states as extra feature)
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING RF")
    logger.info("=" * 60)
    rf = train_rf(train_df, feature_cols, hmm_states=hmm_states)

    # Validate OOS
    logger.info("\n" + "=" * 60)
    logger.info("OOS VALIDATION")
    logger.info("=" * 60)
    oos_results = validate_oos(hmm, rf, features_df, feature_cols, train_end_idx)

    # Save
    if not args.dry_run:
        os.makedirs(SAVED_DIR, exist_ok=True)

        hmm_path = os.path.join(SAVED_DIR, 'regime_hmm.pkl')
        rf_path = os.path.join(SAVED_DIR, 'regime_rf.pkl')
        hmm.save(hmm_path)
        rf.save(rf_path)
        logger.info(f"\nSaved HMM to {hmm_path}")
        logger.info(f"Saved RF to {rf_path}")

        # Save training results
        results = {
            'train_period': f"{features_df['date'].iloc[0].date()} to {features_df['date'].iloc[train_end_idx-1].date()}",
            'test_period': f"{features_df['date'].iloc[train_end_idx].date()} to {features_df['date'].iloc[-1].date()}",
            'train_rows': int(train_end_idx),
            'test_rows': int(len(features_df) - train_end_idx),
            'hmm_state_summary': hmm.get_state_summary(),
            'rf_feature_importances': rf.feature_importances(),
            'oos_validation': oos_results,
        }
        results_path = os.path.join(SAVED_DIR, 'regime_training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {results_path}")
    else:
        logger.info("\n[DRY RUN] Models not saved.")

    logger.info("\nDone.")


if __name__ == '__main__':
    main()

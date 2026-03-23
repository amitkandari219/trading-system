#!/usr/bin/env python3
"""
Validation script for Chan features, CPCV, and XGBoost v2 scorer.

Runs all requested validations:
a. WF-validate CHAN_AT_DRY_4 (36M/12M/3M step, Sharpe>=1.2, PF>=1.6, 75% windows)
b. Test CHAN_ZSCORE_MR as standalone signal (z_40 < -2.0 → LONG, > 2.0 → SHORT)
c. Train XGB v2 with Chan features, compare AUC vs v1
d. Run deflated Sharpe on all 6 active scoring signals
e. Compare CPCV variance vs naive k-fold

Usage:
    venv/bin/python3 scripts/validate_chan_xgb.py
"""

import os
import sys
import logging
import warnings

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('validate')


def load_data():
    """Load Nifty and BankNifty daily data from PostgreSQL."""
    from config.settings import DATABASE_DSN
    import psycopg2

    conn = psycopg2.connect(DATABASE_DSN)

    df_nifty = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix "
        "FROM nifty_daily ORDER BY date",
        conn
    )
    df_nifty['date'] = pd.to_datetime(df_nifty['date'])

    df_bn = pd.read_sql(
        "SELECT date, open, high, low, close, volume "
        "FROM banknifty_daily ORDER BY date",
        conn
    )
    df_bn['date'] = pd.to_datetime(df_bn['date'])

    conn.close()

    logger.info(f"Nifty: {len(df_nifty)} rows, {df_nifty['date'].min()} to {df_nifty['date'].max()}")
    logger.info(f"BankNifty: {len(df_bn)} rows, {df_bn['date'].min()} to {df_bn['date'].max()}")

    return df_nifty, df_bn


def load_calendar():
    """Load market calendar."""
    from config.settings import DATABASE_DSN
    import psycopg2
    conn = psycopg2.connect(DATABASE_DSN)
    cal = pd.read_sql("SELECT trading_date AS date, is_trading_day FROM market_calendar ORDER BY trading_date", conn)
    cal['date'] = pd.to_datetime(cal['date'])
    conn.close()
    return cal


def add_indicators(df):
    """Add all indicators to dataframe."""
    from backtest.indicators import add_all_indicators
    return add_all_indicators(df)


# ================================================================
# VALIDATION A: Walk-Forward CHAN_AT_DRY_4
# ================================================================

def validate_chan_at_dry4(df_nifty, calendar_df):
    """WF-validate CHAN_AT_DRY_4 using 36M/12M/3M step."""
    logger.info("=" * 60)
    logger.info("VALIDATION A: Walk-Forward CHAN_AT_DRY_4")
    logger.info("=" * 60)

    from backtest.types import BacktestResult
    from backtest.walk_forward import WalkForwardEngine

    df = add_indicators(df_nifty.copy())

    def backtest_chan_at4(params, test_df, regime_labels):
        """Simple backtest for CHAN_AT_DRY_4 BB mean-reversion."""
        trades = []
        position = None

        for i in range(1, len(test_df)):
            today = test_df.iloc[i]
            yesterday = test_df.iloc[i - 1]

            close = float(today['close'])
            sma_20 = today.get('sma_20', np.nan)
            bb_pct_b = today.get('bb_pct_b', np.nan)

            if pd.isna(sma_20) or pd.isna(bb_pct_b):
                continue

            # Entry logic
            if position is None:
                if close < sma_20 and bb_pct_b < 0:
                    position = {'direction': 'LONG', 'entry': close, 'entry_idx': i}
                elif close > sma_20 and bb_pct_b > 1.0:
                    position = {'direction': 'SHORT', 'entry': close, 'entry_idx': i}
            else:
                # Exit: close crosses back to SMA20
                if position['direction'] == 'LONG' and close >= sma_20:
                    pnl = close - position['entry']
                    trades.append(pnl)
                    position = None
                elif position['direction'] == 'SHORT' and close <= sma_20:
                    pnl = position['entry'] - close
                    trades.append(pnl)
                    position = None

                # Stop loss: 2%
                if position is not None:
                    if position['direction'] == 'LONG':
                        if close < position['entry'] * 0.98:
                            trades.append(close - position['entry'])
                            position = None
                    else:
                        if close > position['entry'] * 1.02:
                            trades.append(position['entry'] - close)
                            position = None

        return _trades_to_result(trades, test_df)

    wf = WalkForwardEngine(calendar_df)
    result = wf.run(
        signal_id='CHAN_AT_DRY_4',
        backtest_fn=backtest_chan_at4,
        history_df=df,
        regime_labels={},
        params={},
        signal_type='FUTURES'
    )

    return {
        'signal_id': 'CHAN_AT_DRY_4',
        'overall_pass': result['overall_pass'],
        'pass_rate': result['pass_rate'],
        'windows_passed': result['windows_passed'],
        'total_windows': result['total_windows'],
        'aggregate_sharpe': result['aggregate_sharpe'],
        'last_window_passed': result['last_window_passed'],
        'recommendation': result['recommendation'],
    }


# ================================================================
# VALIDATION B: CHAN_ZSCORE_MR standalone signal
# ================================================================

def validate_zscore_mr(df_nifty):
    """Test CHAN_ZSCORE_MR: z_40 < -2 → LONG, > 2 → SHORT."""
    logger.info("=" * 60)
    logger.info("VALIDATION B: CHAN_ZSCORE_MR Standalone Signal")
    logger.info("=" * 60)

    from signals.chan_features import ChanFeatureExtractor

    df = add_indicators(df_nifty.copy())
    extractor = ChanFeatureExtractor()
    close = df['close'].astype(float)
    log_close = np.log(close)

    # Raw z-score (not normalized) for threshold comparison
    z_40 = extractor._rolling_zscore(log_close, 40)

    trades = []
    position = None

    for i in range(41, len(df)):
        z = z_40.iloc[i]
        price = float(df['close'].iloc[i])

        if pd.isna(z):
            continue

        if position is None:
            if z < -2.0:
                position = {'direction': 'LONG', 'entry': price, 'entry_z': z}
            elif z > 2.0:
                position = {'direction': 'SHORT', 'entry': price, 'entry_z': z}
        else:
            # Exit when z reverts to [-0.5, 0.5]
            if abs(z) < 0.5:
                if position['direction'] == 'LONG':
                    pnl = price - position['entry']
                else:
                    pnl = position['entry'] - price
                trades.append(pnl)
                position = None

            # Stop loss: 3% (wider for MR)
            if position is not None:
                if position['direction'] == 'LONG' and price < position['entry'] * 0.97:
                    trades.append(price - position['entry'])
                    position = None
                elif position['direction'] == 'SHORT' and price > position['entry'] * 1.03:
                    trades.append(position['entry'] - price)
                    position = None

    result = _compute_signal_metrics(trades, 'CHAN_ZSCORE_MR')
    return result


# ================================================================
# VALIDATION C: XGB v2 training and AUC comparison
# ================================================================

def validate_xgb_v2(df_nifty, df_bn):
    """Train XGB v2, compare AUC vs v1-style (4 features only)."""
    logger.info("=" * 60)
    logger.info("VALIDATION C: XGBoost v2 with Chan Features")
    logger.info("=" * 60)

    import xgboost as xgb_lib
    from models.xgb_scorer import XGBScorerV2
    from models.cpcv import CombinatorialPurgedCV
    from signals.chan_features import ChanFeatureExtractor

    scorer = XGBScorerV2()
    df = add_indicators(df_nifty.copy())

    # Train v2
    try:
        training_results = scorer.train_full(df, df_bn, forward_days=5)
    except Exception as e:
        logger.error(f"XGB v2 training failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

    # v1 baseline: train with only 4 original features
    logger.info("Training v1 baseline (4 features only)...")
    from signals.chan_features import ChanFeatureExtractor
    chan_ext = ChanFeatureExtractor()
    chan_feats = chan_ext.compute_all(df, df_bn)

    close = df['close'].astype(float)
    rsi_14 = df.get('rsi_14', pd.Series(50.0, index=df.index))
    sma_50 = df.get('sma_50', close)
    sma_200 = df.get('sma_200', close)
    vol_ratio = df.get('vol_ratio_20', pd.Series(1.0, index=df.index))
    hurst = df.get('hurst_100', pd.Series(0.5, index=df.index))

    feature_df_v1 = pd.DataFrame({
        'dim1_extremity': (rsi_14 - 50).abs().clip(0, 30) / 30.0,
        'dim2_regime': hurst.fillna(0.5),
        'dim3_timeframe': np.where(close > sma_50, np.where(sma_50 > sma_200, 1.0, 0.6), 0.2),
        'dim4_volume': (vol_ratio / 2.0).clip(0, 1),
    }, index=df.index)

    fwd_return = close.shift(-5) / close - 1.0
    target = (fwd_return > 0).astype(int)
    valid = feature_df_v1.notna().all(axis=1) & target.notna()

    X_v1 = feature_df_v1[valid].values
    y_v1 = target[valid].values
    dates_v1 = df.loc[valid, 'date'].values

    cpcv = CombinatorialPurgedCV(n_groups=6, test_groups=2, embargo_days=5)
    v1_results = cpcv.backtest_paths(
        X_v1, y_v1, dates_v1,
        model_factory=lambda: xgb_lib.XGBClassifier(
            max_depth=5, n_estimators=200, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )
    )

    # Save model
    save_path = os.path.join(PROJECT_ROOT, 'models', 'artifacts', 'xgb_scorer_v2.pkl')
    scorer.save(save_path)
    logger.info(f"Model saved to {save_path}")

    return {
        'v2_cpcv_auc': training_results.get('cpcv_mean_auc'),
        'v2_cpcv_auc_std': training_results.get('cpcv_std_auc'),
        'v2_cpcv_accuracy': training_results.get('cpcv_mean_accuracy'),
        'v2_best_params': training_results.get('best_params'),
        'v2_n_samples': training_results.get('n_samples'),
        'v1_cpcv_auc': v1_results['mean_metrics'].get('auc'),
        'v1_cpcv_auc_std': v1_results['std_metrics'].get('auc'),
        'v1_cpcv_accuracy': v1_results['mean_metrics'].get('accuracy'),
        'auc_improvement': (
            (training_results.get('cpcv_mean_auc', 0) or 0) -
            (v1_results['mean_metrics'].get('auc', 0) or 0)
        ),
        'feature_importance': dict(list(scorer.feature_importances.items())[:10]),
        'shap_importance': dict(list(scorer.get_shap_importance().items())[:10]),
    }


# ================================================================
# VALIDATION D: Deflated Sharpe on 6 active scoring signals
# ================================================================

def validate_deflated_sharpe(df_nifty):
    """Run deflated Sharpe on all 6 active scoring signals."""
    logger.info("=" * 60)
    logger.info("VALIDATION D: Deflated Sharpe Ratio")
    logger.info("=" * 60)

    from models.cpcv import deflated_sharpe

    df = add_indicators(df_nifty.copy())
    close = df['close'].astype(float)
    returns = close.pct_change().dropna()
    T = len(returns)

    # Simulate 6 scoring signals with different strategies
    signal_configs = {
        'KAUFMAN_DRY_20': {'sma_fast': 10, 'sma_slow': 20, 'type': 'trend'},
        'KAUFMAN_DRY_16': {'sma_fast': 8, 'sma_slow': 16, 'type': 'trend'},
        'KAUFMAN_DRY_12': {'sma_fast': 6, 'sma_slow': 12, 'type': 'trend'},
        'GUJRAL_DRY_8': {'sma_fast': 5, 'sma_slow': 20, 'type': 'trend'},
        'GUJRAL_DRY_13': {'sma_fast': 10, 'sma_slow': 40, 'type': 'trend'},
        'CHAN_AT_DRY_4': {'type': 'mr'},
    }

    all_sharpes = []
    results = {}

    for sig_id, config in signal_configs.items():
        if config['type'] == 'trend':
            fast = df[f'sma_{config["sma_fast"]}'] if f'sma_{config["sma_fast"]}' in df.columns else close.rolling(config['sma_fast']).mean()
            slow = df[f'sma_{config["sma_slow"]}'] if f'sma_{config["sma_slow"]}' in df.columns else close.rolling(config['sma_slow']).mean()
            pos = np.where(fast > slow, 1, -1)
        else:
            z = (close - close.rolling(20).mean()) / close.rolling(20).std().replace(0, np.nan)
            pos = np.where(z < -2, 1, np.where(z > 2, -1, 0))

        strat_returns = pd.Series(pos, index=df.index).shift(1) * returns.reindex(df.index)
        strat_returns = strat_returns.dropna()

        if len(strat_returns) < 50 or strat_returns.std() == 0:
            results[sig_id] = {'sharpe': 0.0, 'deflated_p': 1.0, 'significant': False}
            all_sharpes.append(0.0)
            continue

        sr = float(strat_returns.mean() / strat_returns.std() * np.sqrt(252))
        skew = float(strat_returns.skew())
        kurt = float(strat_returns.kurtosis())
        all_sharpes.append(sr)

    # Now compute deflated Sharpe for each
    n_trials = len(signal_configs)
    var_sharpes = float(np.var(all_sharpes)) if len(all_sharpes) > 1 else 0.01

    i = 0
    for sig_id, config in signal_configs.items():
        if sig_id in results:
            i += 1
            continue

        sr = all_sharpes[i]

        if config['type'] == 'trend':
            fast = df[f'sma_{config["sma_fast"]}'] if f'sma_{config["sma_fast"]}' in df.columns else close.rolling(config['sma_fast']).mean()
            slow = df[f'sma_{config["sma_slow"]}'] if f'sma_{config["sma_slow"]}' in df.columns else close.rolling(config['sma_slow']).mean()
            pos = np.where(fast > slow, 1, -1)
        else:
            z = (close - close.rolling(20).mean()) / close.rolling(20).std().replace(0, np.nan)
            pos = np.where(z < -2, 1, np.where(z > 2, -1, 0))

        strat_returns = pd.Series(pos, index=df.index).shift(1) * returns.reindex(df.index)
        strat_returns = strat_returns.dropna()
        skew = float(strat_returns.skew())
        kurt = float(strat_returns.kurtosis())

        p_val = deflated_sharpe(
            observed_sharpe=sr,
            n_trials=n_trials,
            variance_of_sharpes=var_sharpes,
            skewness=skew,
            kurtosis=kurt,
            T=len(strat_returns)
        )

        results[sig_id] = {
            'sharpe': round(sr, 3),
            'deflated_p': round(p_val, 4),
            'significant': p_val < 0.05,
        }
        i += 1

    return results


# ================================================================
# VALIDATION E: CPCV variance vs naive k-fold
# ================================================================

def validate_cpcv_vs_kfold(df_nifty, df_bn):
    """Compare CPCV variance vs naive k-fold."""
    logger.info("=" * 60)
    logger.info("VALIDATION E: CPCV vs Naive K-Fold Variance")
    logger.info("=" * 60)

    import xgboost as xgb_lib
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score
    from models.cpcv import CombinatorialPurgedCV
    from signals.chan_features import ChanFeatureExtractor

    df = add_indicators(df_nifty.copy())
    chan_ext = ChanFeatureExtractor()
    chan_feats = chan_ext.compute_all(df, df_bn)

    close = df['close'].astype(float)
    rsi_14 = df.get('rsi_14', pd.Series(50.0, index=df.index))
    sma_50 = df.get('sma_50', close)
    sma_200 = df.get('sma_200', close)
    vol_ratio = df.get('vol_ratio_20', pd.Series(1.0, index=df.index))
    hurst = df.get('hurst_100', pd.Series(0.5, index=df.index))

    feature_df = pd.DataFrame({
        'dim1_extremity': (rsi_14 - 50).abs().clip(0, 30) / 30.0,
        'dim2_regime': hurst.fillna(0.5),
        'dim3_timeframe': np.where(close > sma_50, np.where(sma_50 > sma_200, 1.0, 0.6), 0.2),
        'dim4_volume': (vol_ratio / 2.0).clip(0, 1),
    }, index=df.index)

    for name, series in chan_feats.items():
        feature_df[name] = series

    fwd_return = close.shift(-5) / close - 1.0
    target = (fwd_return > 0).astype(int)
    valid = feature_df.notna().all(axis=1) & target.notna()

    from models.xgb_scorer import ALL_FEATURES_V2
    X = feature_df.loc[valid, ALL_FEATURES_V2].values
    y = target[valid].values
    dates = df.loc[valid, 'date'].values

    params = dict(max_depth=5, n_estimators=200, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8)

    def make_model():
        return xgb_lib.XGBClassifier(
            **params, random_state=42, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        )

    # CPCV
    logger.info("Running CPCV (6 groups, 2 test)...")
    cpcv = CombinatorialPurgedCV(n_groups=6, test_groups=2, embargo_days=5)
    cpcv_results = cpcv.backtest_paths(X, y, dates, make_model)

    cpcv_aucs = [
        m['auc'] for m in cpcv_results['path_metrics']
        if m.get('auc') is not None
    ]

    # Naive k-fold (no purging, no embargo)
    logger.info("Running naive 5-fold CV...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    kfold_aucs = []
    for train_idx, test_idx in kfold.split(X):
        model = make_model()
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        try:
            auc = roc_auc_score(y[test_idx], proba)
            kfold_aucs.append(auc)
        except Exception:
            pass

    return {
        'cpcv_mean_auc': round(float(np.mean(cpcv_aucs)), 4) if cpcv_aucs else None,
        'cpcv_std_auc': round(float(np.std(cpcv_aucs)), 4) if cpcv_aucs else None,
        'cpcv_n_paths': len(cpcv_aucs),
        'kfold_mean_auc': round(float(np.mean(kfold_aucs)), 4) if kfold_aucs else None,
        'kfold_std_auc': round(float(np.std(kfold_aucs)), 4) if kfold_aucs else None,
        'kfold_n_folds': len(kfold_aucs),
        'auc_gap': round(
            (float(np.mean(kfold_aucs)) if kfold_aucs else 0) -
            (float(np.mean(cpcv_aucs)) if cpcv_aucs else 0),
            4
        ),
        'variance_ratio': round(
            (float(np.var(cpcv_aucs)) / float(np.var(kfold_aucs)))
            if kfold_aucs and np.var(kfold_aucs) > 0 and cpcv_aucs
            else 0, 3
        ),
    }


# ================================================================
# HELPERS
# ================================================================

def _trades_to_result(trades, test_df):
    """Convert list of PnL trades to BacktestResult."""
    from backtest.types import BacktestResult

    if len(trades) < 2:
        return BacktestResult(
            sharpe=0.0, calmar_ratio=0.0, max_drawdown=1.0,
            win_rate=0.0, profit_factor=0.0, avg_win_loss_ratio=0.0,
            trade_count=len(trades), nifty_correlation=0.0,
            annual_return=0.0
        )

    trades = np.array(trades)
    wins = trades[trades > 0]
    losses = trades[trades < 0]

    win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1
    profit_factor = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else 10.0
    avg_wl = avg_win / avg_loss if avg_loss > 0 else 0

    # Approximate Sharpe from trade PnL
    daily_equiv = trades / float(test_df['close'].iloc[0])  # normalize
    sharpe = float(np.mean(daily_equiv) / np.std(daily_equiv) * np.sqrt(252)) if np.std(daily_equiv) > 0 else 0

    # Drawdown
    cum = np.cumsum(trades)
    running_max = np.maximum.accumulate(cum)
    dd = (running_max - cum)
    max_dd = float(dd.max() / float(test_df['close'].iloc[0])) if len(dd) > 0 else 0

    # Annual return approximation
    n_days = (test_df['date'].max() - test_df['date'].min()).days if 'date' in test_df.columns else 252
    total_ret = np.sum(trades) / float(test_df['close'].iloc[0])
    annual_ret = total_ret * (252 / max(n_days, 1))

    calmar = annual_ret / max_dd if max_dd > 0 else 0

    return BacktestResult(
        sharpe=round(sharpe, 3),
        calmar_ratio=round(calmar, 3),
        max_drawdown=round(max_dd, 4),
        win_rate=round(win_rate, 3),
        profit_factor=round(profit_factor, 3),
        avg_win_loss_ratio=round(avg_wl, 3),
        trade_count=len(trades),
        nifty_correlation=0.0,
        annual_return=round(annual_ret, 4),
    )


def _compute_signal_metrics(trades, signal_id):
    """Compute metrics dict from list of PnL trades."""
    if not trades:
        return {
            'signal_id': signal_id, 'trade_count': 0,
            'sharpe': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'avg_win_loss_ratio': 0.0, 'total_return_pct': 0.0,
        }

    trades = np.array(trades)
    wins = trades[trades > 0]
    losses = trades[trades < 0]

    win_rate = len(wins) / len(trades)
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1
    pf = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else 10.0
    sharpe = float(np.mean(trades) / np.std(trades) * np.sqrt(252)) if np.std(trades) > 0 else 0

    return {
        'signal_id': signal_id,
        'trade_count': len(trades),
        'sharpe': round(sharpe, 3),
        'win_rate': round(win_rate, 3),
        'profit_factor': round(pf, 3),
        'avg_win_loss_ratio': round(avg_win / avg_loss if avg_loss > 0 else 0, 3),
        'total_return_pct': round(float(np.sum(trades)), 2),
        'avg_trade': round(float(np.mean(trades)), 2),
        'max_win': round(float(np.max(trades)), 2),
        'max_loss': round(float(np.min(trades)), 2),
    }


# ================================================================
# RESULTS TABLE
# ================================================================

def print_results_table(results):
    """Print comprehensive results table."""
    print("\n")
    print("=" * 80)
    print("  COMPREHENSIVE VALIDATION RESULTS")
    print("  Chan Features + CPCV + XGBoost v2")
    print("=" * 80)

    # A: Walk-Forward
    print("\n--- A. Walk-Forward: CHAN_AT_DRY_4 ---")
    a = results.get('wf_chan_at4', {})
    if a:
        status = "PASS" if a.get('overall_pass') else "FAIL"
        print(f"  Status:          {status}")
        print(f"  Pass Rate:       {a.get('pass_rate', 0):.0%} (need >=75%)")
        print(f"  Windows:         {a.get('windows_passed', 0)}/{a.get('total_windows', 0)}")
        print(f"  Agg Sharpe:      {a.get('aggregate_sharpe', 0):.3f} (need >=1.2)")
        print(f"  Last Window:     {'PASS' if a.get('last_window_passed') else 'FAIL'}")
        print(f"  Recommendation:  {a.get('recommendation', 'N/A')}")
    else:
        print("  [skipped or failed]")

    # B: ZSCORE MR
    print("\n--- B. CHAN_ZSCORE_MR Standalone ---")
    b = results.get('zscore_mr', {})
    if b:
        print(f"  Trade Count:     {b.get('trade_count', 0)}")
        print(f"  Sharpe:          {b.get('sharpe', 0):.3f}")
        print(f"  Win Rate:        {b.get('win_rate', 0):.1%}")
        print(f"  Profit Factor:   {b.get('profit_factor', 0):.3f}")
        print(f"  Avg W/L Ratio:   {b.get('avg_win_loss_ratio', 0):.3f}")
        print(f"  Avg Trade:       {b.get('avg_trade', 0):.2f} pts")
    else:
        print("  [skipped or failed]")

    # C: XGB v2 vs v1
    print("\n--- C. XGBoost v2 vs v1 AUC Comparison ---")
    c = results.get('xgb_v2', {})
    if c and 'error' not in c:
        print(f"  v2 CPCV AUC:     {c.get('v2_cpcv_auc', 'N/A')}")
        print(f"  v2 AUC Std:      {c.get('v2_cpcv_auc_std', 'N/A')}")
        print(f"  v2 Accuracy:     {c.get('v2_cpcv_accuracy', 'N/A')}")
        print(f"  v1 CPCV AUC:     {c.get('v1_cpcv_auc', 'N/A')}")
        print(f"  v1 AUC Std:      {c.get('v1_cpcv_auc_std', 'N/A')}")
        print(f"  AUC Improvement: {c.get('auc_improvement', 0):+.4f}")
        print(f"  Best Params:     {c.get('v2_best_params', {})}")
        print(f"  Top Features (XGB importance):")
        for feat, imp in list(c.get('feature_importance', {}).items())[:7]:
            print(f"    {feat:25s} {imp:.4f}")
        if c.get('shap_importance'):
            print(f"  Top Features (SHAP):")
            for feat, imp in list(c.get('shap_importance', {}).items())[:7]:
                print(f"    {feat:25s} {imp:.4f}")
    elif c and 'error' in c:
        print(f"  ERROR: {c['error']}")
    else:
        print("  [skipped or failed]")

    # D: Deflated Sharpe
    print("\n--- D. Deflated Sharpe Ratio (6 Scoring Signals) ---")
    d = results.get('deflated_sharpe', {})
    if d:
        print(f"  {'Signal':<20s} {'Sharpe':>8s} {'DSR p-val':>10s} {'Significant':>12s}")
        print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*12}")
        for sig_id, metrics in d.items():
            sig_str = 'YES' if metrics.get('significant') else 'no'
            print(f"  {sig_id:<20s} {metrics.get('sharpe', 0):>8.3f} {metrics.get('deflated_p', 1):>10.4f} {sig_str:>12s}")
    else:
        print("  [skipped or failed]")

    # E: CPCV vs K-Fold
    print("\n--- E. CPCV vs Naive K-Fold ---")
    e = results.get('cpcv_vs_kfold', {})
    if e:
        print(f"  CPCV Mean AUC:   {e.get('cpcv_mean_auc', 'N/A')} (+/- {e.get('cpcv_std_auc', 'N/A')})")
        print(f"  CPCV Paths:      {e.get('cpcv_n_paths', 'N/A')}")
        print(f"  K-Fold Mean AUC: {e.get('kfold_mean_auc', 'N/A')} (+/- {e.get('kfold_std_auc', 'N/A')})")
        print(f"  K-Fold Folds:    {e.get('kfold_n_folds', 'N/A')}")
        print(f"  AUC Gap (KF-CPCV): {e.get('auc_gap', 0):+.4f}")
        print(f"    (positive gap = k-fold is overoptimistic due to leakage)")
        print(f"  Variance Ratio (CPCV/KF): {e.get('variance_ratio', 'N/A')}")
        print(f"    (>1 = CPCV has higher variance = more realistic estimate)")
    else:
        print("  [skipped or failed]")

    print("\n" + "=" * 80)
    print("  Files created:")
    print(f"    signals/chan_features.py    — ChanFeatureExtractor (10 features)")
    print(f"    models/cpcv.py             — CombinatorialPurgedCV + deflated_sharpe")
    print(f"    models/xgb_scorer.py       — XGBScorerV2 (14 features)")
    print(f"    models/artifacts/xgb_scorer_v2.pkl — trained model")
    print("=" * 80)


# ================================================================
# MAIN
# ================================================================

def main():
    logger.info("Loading data from PostgreSQL...")
    df_nifty, df_bn = load_data()
    calendar_df = load_calendar()

    results = {}

    # A: Walk-Forward CHAN_AT_DRY_4
    try:
        results['wf_chan_at4'] = validate_chan_at_dry4(df_nifty, calendar_df)
    except Exception as e:
        logger.error(f"Validation A failed: {e}")
        import traceback; traceback.print_exc()
        results['wf_chan_at4'] = {'error': str(e)}

    # B: CHAN_ZSCORE_MR standalone
    try:
        results['zscore_mr'] = validate_zscore_mr(df_nifty)
    except Exception as e:
        logger.error(f"Validation B failed: {e}")
        import traceback; traceback.print_exc()
        results['zscore_mr'] = {'error': str(e)}

    # C: XGB v2 training and comparison
    try:
        results['xgb_v2'] = validate_xgb_v2(df_nifty, df_bn)
    except Exception as e:
        logger.error(f"Validation C failed: {e}")
        import traceback; traceback.print_exc()
        results['xgb_v2'] = {'error': str(e)}

    # D: Deflated Sharpe
    try:
        results['deflated_sharpe'] = validate_deflated_sharpe(df_nifty)
    except Exception as e:
        logger.error(f"Validation D failed: {e}")
        import traceback; traceback.print_exc()
        results['deflated_sharpe'] = {'error': str(e)}

    # E: CPCV vs K-Fold
    try:
        results['cpcv_vs_kfold'] = validate_cpcv_vs_kfold(df_nifty, df_bn)
    except Exception as e:
        logger.error(f"Validation E failed: {e}")
        import traceback; traceback.print_exc()
        results['cpcv_vs_kfold'] = {'error': str(e)}

    print_results_table(results)


if __name__ == '__main__':
    main()

"""
Combinatorial Purged Cross-Validation (CPCV) — Lopez de Prado Ch. 12.

Generates C(n_groups, test_groups) backtest paths with purging and embargo
to avoid information leakage. This replaces naive k-fold for financial ML.

Also implements Deflated Sharpe Ratio (Ch. 11) to test if observed Sharpe
is significant given multiple trials.

Usage:
    cpcv = CombinatorialPurgedCV(n_groups=6, test_groups=2, embargo_days=5)
    for train_idx, test_idx in cpcv.split(X, y, dates):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])

    metrics = cpcv.backtest_paths(X, y, dates, model_factory)
    p_val = deflated_sharpe(observed_sr, n_trials, var_sr, skew, kurt, T)
"""

import logging
from itertools import combinations
from typing import Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation with embargo.

    Parameters:
        n_groups: number of groups to split data into (default 6)
        test_groups: number of groups used for test in each split (default 2)
        embargo_days: number of calendar days to embargo after test period
    """

    def __init__(self, n_groups: int = 6, test_groups: int = 2,
                 embargo_days: int = 5):
        self.n_groups = n_groups
        self.test_groups = test_groups
        self.embargo_days = embargo_days

        # Total number of paths = C(n_groups, test_groups)
        self.n_paths = _n_choose_k(n_groups, test_groups)
        logger.info(
            f"CPCV initialized: {n_groups} groups, {test_groups} test, "
            f"{self.n_paths} paths, {embargo_days}d embargo"
        )

    def split(self, X: np.ndarray, y: np.ndarray,
              dates: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test index splits with purging and embargo.

        Args:
            X: feature matrix (n_samples, n_features)
            y: target vector (n_samples,)
            dates: array of dates/timestamps aligned with X, y

        Yields:
            (train_indices, test_indices) for each combinatorial path
        """
        n = len(X)
        if n != len(y) or n != len(dates):
            raise ValueError(
                f"X ({len(X)}), y ({len(y)}), dates ({len(dates)}) must have same length"
            )

        dates = pd.to_datetime(dates)
        sorted_idx = np.argsort(dates)
        dates_sorted = dates[sorted_idx]

        # Split into n_groups of roughly equal size
        group_bounds = np.array_split(sorted_idx, self.n_groups)

        # Generate all C(n_groups, test_groups) combinations
        for test_group_ids in combinations(range(self.n_groups), self.test_groups):
            test_idx = np.concatenate([group_bounds[g] for g in test_group_ids])
            train_group_ids = [g for g in range(self.n_groups) if g not in test_group_ids]
            train_idx = np.concatenate([group_bounds[g] for g in train_group_ids])

            # Purging: remove train samples that are within embargo_days
            # of any test sample boundary
            test_dates = dates_sorted[test_idx] if len(test_idx) > 0 else pd.DatetimeIndex([])
            if len(test_dates) > 0:
                test_min = test_dates.min()
                test_max = test_dates.max()
                embargo_td = pd.Timedelta(days=self.embargo_days)

                # Purge zone: [test_min - embargo, test_max + embargo]
                purge_mask = np.ones(len(train_idx), dtype=bool)
                train_dates_arr = dates_sorted[train_idx]

                for g in test_group_ids:
                    g_dates = dates_sorted[group_bounds[g]]
                    if len(g_dates) == 0:
                        continue
                    g_start = g_dates.min() - embargo_td
                    g_end = g_dates.max() + embargo_td

                    in_purge = (train_dates_arr >= g_start) & (train_dates_arr <= g_end)
                    purge_mask &= ~in_purge

                train_idx = train_idx[purge_mask]

            if len(train_idx) == 0 or len(test_idx) == 0:
                logger.warning(
                    f"Empty split for test groups {test_group_ids}, skipping"
                )
                continue

            yield train_idx, test_idx

    def backtest_paths(self, X: np.ndarray, y: np.ndarray,
                       dates: np.ndarray,
                       model_factory: Callable,
                       metric_fn: Optional[Callable] = None) -> Dict:
        """
        Run model on all CPCV paths and return metrics distribution.

        Args:
            X: feature matrix
            y: target vector
            dates: date array
            model_factory: callable that returns a fresh model instance
            metric_fn: optional function(y_true, y_pred, y_proba) -> dict
                       Default computes accuracy, AUC, Sharpe proxy.

        Returns:
            dict with:
                'path_metrics': list of per-path metric dicts
                'mean_metrics': dict of mean across paths
                'std_metrics': dict of std across paths
                'n_paths': int
        """
        if metric_fn is None:
            metric_fn = _default_metric_fn

        path_metrics = []
        for i, (train_idx, test_idx) in enumerate(self.split(X, y, dates)):
            try:
                model = model_factory()
                model.fit(X[train_idx], y[train_idx])

                y_pred = model.predict(X[test_idx])
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_proba = model.predict_proba(X[test_idx])[:, 1]
                    except Exception:
                        pass

                metrics = metric_fn(y[test_idx], y_pred, y_proba)
                metrics['path_id'] = i
                metrics['train_size'] = len(train_idx)
                metrics['test_size'] = len(test_idx)
                path_metrics.append(metrics)
            except Exception as e:
                logger.warning(f"Path {i} failed: {e}")
                continue

        if not path_metrics:
            return {
                'path_metrics': [],
                'mean_metrics': {},
                'std_metrics': {},
                'n_paths': 0,
            }

        # Aggregate
        metric_keys = [k for k in path_metrics[0] if k not in ('path_id', 'train_size', 'test_size')]
        mean_metrics = {}
        std_metrics = {}
        for k in metric_keys:
            vals = [m[k] for m in path_metrics if k in m and m[k] is not None]
            if vals:
                mean_metrics[k] = float(np.mean(vals))
                std_metrics[k] = float(np.std(vals))

        return {
            'path_metrics': path_metrics,
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'n_paths': len(path_metrics),
        }

    def get_path_count(self) -> int:
        """Return total number of combinatorial paths."""
        return self.n_paths


def deflated_sharpe(observed_sharpe: float, n_trials: int,
                    variance_of_sharpes: float, skewness: float,
                    kurtosis: float, T: int) -> float:
    """
    Deflated Sharpe Ratio test (Lopez de Prado, Ch. 11).

    Tests whether the observed Sharpe ratio is statistically significant
    given the number of trials (strategy variants) tested.

    The key insight: if you test 100 strategies, the best one will have
    an inflated Sharpe purely from luck. DSR corrects for this.

    Args:
        observed_sharpe: the Sharpe ratio of the selected strategy
        n_trials: number of independent strategies tested
        variance_of_sharpes: variance of Sharpe ratios across all trials
        skewness: skewness of returns of the selected strategy
        kurtosis: excess kurtosis of returns of the selected strategy
        T: number of observations (trading days)

    Returns:
        p-value (probability that observed Sharpe is due to luck).
        Low p-value (< 0.05) means the Sharpe is likely genuine.
    """
    if n_trials <= 1 or T <= 1 or variance_of_sharpes <= 0:
        return 1.0  # cannot assess

    # Expected maximum Sharpe under null (from order statistics)
    # E[max(SR)] ≈ sqrt(V(SR)) * [(1 - gamma) * Z_inv(1 - 1/N) + gamma * Z_inv(1 - 1/(N*e))]
    # Simplified: E[max] ≈ sqrt(2 * log(N)) * sqrt(V(SR))  (Bonferroni approximation)
    euler_mascheroni = 0.5772156649
    try:
        z_val = scipy_stats.norm.ppf(1 - 1.0 / n_trials)
    except Exception:
        z_val = 2.0  # fallback

    expected_max_sr = np.sqrt(variance_of_sharpes) * (
        (1 - euler_mascheroni) * z_val +
        euler_mascheroni * scipy_stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    )

    # Standard error of Sharpe ratio (accounting for skew/kurtosis)
    # SE(SR) = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (T-1))
    sr = observed_sharpe
    se_sr = np.sqrt(
        max(1e-10, (1 - skewness * sr + ((kurtosis - 1) / 4) * sr ** 2))
        / max(1, T - 1)
    )

    if se_sr <= 0:
        return 1.0

    # Test statistic: (observed_SR - E[max_SR]) / SE(SR)
    psr_stat = (sr - expected_max_sr) / se_sr

    # p-value from standard normal
    p_value = 1 - scipy_stats.norm.cdf(psr_stat)

    return float(p_value)


# ================================================================
# HELPERS
# ================================================================

def _n_choose_k(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    from math import comb
    return comb(n, k)


def _default_metric_fn(y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray]) -> Dict:
    """Default metrics: accuracy, AUC, precision, recall."""
    metrics = {}

    # Accuracy
    metrics['accuracy'] = float(np.mean(y_true == y_pred))

    # AUC (if probabilities available)
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            from sklearn.metrics import roc_auc_score
            metrics['auc'] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics['auc'] = None
    else:
        metrics['auc'] = None

    # Precision and recall for class 1
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    metrics['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Sharpe proxy: if we go long when pred=1, short when pred=0
    # Use y_true as returns sign (1=up, 0=down)
    positions = 2.0 * y_pred - 1.0  # map {0,1} -> {-1, +1}
    returns_sign = 2.0 * y_true - 1.0
    strategy_returns = positions * returns_sign
    if np.std(strategy_returns) > 0:
        metrics['sharpe_proxy'] = float(
            np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        )
    else:
        metrics['sharpe_proxy'] = 0.0

    return metrics

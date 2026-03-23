"""
Chan statistical feature extractor for mean-reversion and regime detection.

Implements features from Ernest Chan's quantitative trading books:
  1. Z-score of log(close) spread: rolling 20/40/60 periods
  2. Hurst exponent: rolling 100-bar R/S method on log returns
  3. Half-life of mean reversion: AR(1) fit
  4. Variance ratio: VR(q) for q=[2,5,10,20]
  5. Cointegration spread (Nifty vs BankNifty): Johansen test, spread z-score

All features normalized to [0,1] for ML input.
NaN filled with 0.5 (neutral) for normalized features.

Usage:
    extractor = ChanFeatureExtractor()
    features = extractor.compute_all(df_nifty, df_banknifty=df_bn)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ChanFeatureExtractor:
    """Compute Chan-style statistical features for mean-reversion trading."""

    # Normalization bounds for clipping before [0,1] mapping
    ZSCORE_CLIP = 4.0        # z-scores beyond +/-4 are clipped
    HURST_MIN = 0.0
    HURST_MAX = 1.0
    HALF_LIFE_MAX = 120.0    # days — beyond this, treat as no mean-reversion
    VR_CLIP = 3.0            # variance ratios clipped to [0, 3]

    def __init__(self):
        self._feature_names = None

    @property
    def feature_names(self):
        """Return ordered list of all feature names produced."""
        if self._feature_names is None:
            self._feature_names = (
                [f'chan_zscore_{p}' for p in [20, 40, 60]] +
                ['chan_hurst_100'] +
                ['chan_half_life'] +
                [f'chan_vr_{q}' for q in [2, 5, 10, 20]] +
                ['chan_coint_spread_z']
            )
        return self._feature_names

    def compute_all(self, df: pd.DataFrame,
                    df_banknifty: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """
        Compute all Chan features on a DataFrame with OHLCV columns.

        Args:
            df: Nifty daily DataFrame with 'close' column (and 'date')
            df_banknifty: BankNifty daily DataFrame for cointegration
                          (optional — if None, coint feature returns 0.5)

        Returns:
            dict of {feature_name: pd.Series} — each series aligned to df.index,
            values normalized to [0, 1].
        """
        features = {}
        close = df['close'].astype(float)
        log_close = np.log(close)

        # --- 1. Z-score of log(close) spread ---
        for period in [20, 40, 60]:
            z = self._rolling_zscore(log_close, period)
            features[f'chan_zscore_{period}'] = self._normalize_zscore(z)

        # --- 2. Hurst exponent (rolling 100-bar) ---
        hurst = self._rolling_hurst(close, window=100)
        features['chan_hurst_100'] = self._normalize_hurst(hurst)

        # --- 3. Half-life of mean reversion ---
        hl = self._rolling_half_life(log_close, window=100)
        features['chan_half_life'] = self._normalize_half_life(hl)

        # --- 4. Variance ratio ---
        log_returns = np.log(close / close.shift(1))
        for q in [2, 5, 10, 20]:
            vr = self._rolling_variance_ratio(log_returns, q=q, window=100)
            features[f'chan_vr_{q}'] = self._normalize_variance_ratio(vr)

        # --- 5. Cointegration spread (Nifty vs BankNifty) ---
        if df_banknifty is not None and len(df_banknifty) > 60:
            coint_z = self._cointegration_spread(df, df_banknifty)
            features['chan_coint_spread_z'] = self._normalize_zscore(coint_z)
        else:
            features['chan_coint_spread_z'] = pd.Series(
                0.5, index=df.index, name='chan_coint_spread_z'
            )

        # Fill NaN with 0.5 (neutral)
        for k in features:
            features[k] = features[k].fillna(0.5)

        return features

    def compute_single_row(self, df: pd.DataFrame, idx: int,
                           df_banknifty: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Compute features for a single row (for live scoring).
        Uses the full df up to idx for rolling calculations.

        Returns:
            dict of {feature_name: float} — values in [0, 1]
        """
        all_features = self.compute_all(df.iloc[:idx + 1], df_banknifty)
        return {k: float(v.iloc[-1]) for k, v in all_features.items()}

    # ================================================================
    # FEATURE COMPUTATIONS
    # ================================================================

    @staticmethod
    def _rolling_zscore(series: pd.Series, period: int) -> pd.Series:
        """Z-score of series relative to rolling mean/std."""
        roll_mean = series.rolling(period, min_periods=period).mean()
        roll_std = series.rolling(period, min_periods=period).std()
        return (series - roll_mean) / roll_std.replace(0, np.nan)

    @staticmethod
    def _rolling_hurst(close: pd.Series, window: int = 100) -> pd.Series:
        """
        Rolling Hurst exponent via R/S analysis on log returns.
        Reuses the algorithm from backtest/indicators.py.
        H > 0.55 = trending, H < 0.45 = mean-reverting, ~0.5 = random walk.
        """
        result = pd.Series(np.nan, index=close.index)
        log_returns = np.log(close / close.shift(1)).values

        for i in range(window + 1, len(log_returns)):
            segment = log_returns[i - window:i]
            if np.any(np.isnan(segment)):
                continue
            result.iloc[i] = _rs_hurst_compute(segment)

        return result

    @staticmethod
    def _rolling_half_life(log_price: pd.Series, window: int = 100) -> pd.Series:
        """
        Half-life of mean reversion from AR(1) model.
        Regress delta_y on y_lag: delta_y = lambda * y_lag + noise
        half_life = -log(2) / log(1 + lambda)
        Negative or very large half-life → no mean reversion.
        """
        result = pd.Series(np.nan, index=log_price.index)
        values = log_price.values

        for i in range(window, len(values)):
            segment = values[i - window:i]
            if np.any(np.isnan(segment)):
                continue

            y = segment[1:] - segment[:-1]  # delta_y
            x = segment[:-1]                # y_lag

            # De-mean for numerical stability
            x_dm = x - np.mean(x)
            denom = np.dot(x_dm, x_dm)
            if denom == 0:
                continue

            lam = np.dot(x_dm, y) / denom  # OLS slope

            # lambda must be negative for mean-reversion
            if lam >= 0:
                result.iloc[i] = np.nan  # no mean reversion
                continue

            denom_log = np.log(1 + lam)
            if denom_log == 0:
                continue

            hl = -np.log(2) / denom_log
            if hl > 0:
                result.iloc[i] = hl

        return result

    @staticmethod
    def _rolling_variance_ratio(log_returns: pd.Series, q: int,
                                 window: int = 100) -> pd.Series:
        """
        Variance ratio VR(q) = Var(q-period return) / (q * Var(1-period return)).
        VR = 1 → random walk, VR < 1 → mean reverting, VR > 1 → trending.
        """
        result = pd.Series(np.nan, index=log_returns.index)
        values = log_returns.values

        for i in range(window + q, len(values)):
            segment = values[i - window:i]
            if np.any(np.isnan(segment)):
                continue

            # 1-period variance
            var_1 = np.var(segment, ddof=1)
            if var_1 == 0:
                continue

            # q-period returns
            n = len(segment)
            if n < q + 1:
                continue
            q_returns = np.array([
                np.sum(segment[j:j + q])
                for j in range(n - q + 1)
            ])
            var_q = np.var(q_returns, ddof=1)

            vr = var_q / (q * var_1)
            result.iloc[i] = vr

        return result

    @staticmethod
    def _cointegration_spread(df_nifty: pd.DataFrame,
                               df_banknifty: pd.DataFrame) -> pd.Series:
        """
        Compute cointegration spread between Nifty and BankNifty.
        Uses rolling OLS hedge ratio (simpler than Johansen for rolling window).
        Spread = log(BN) - beta * log(Nifty)
        Then z-score the spread.
        """
        # Merge on date
        merged = pd.merge(
            df_nifty[['date', 'close']].rename(columns={'close': 'nifty'}),
            df_banknifty[['date', 'close']].rename(columns={'close': 'bn'}),
            on='date', how='inner'
        )

        if len(merged) < 60:
            return pd.Series(0.0, index=df_nifty.index)

        log_n = np.log(merged['nifty'].values)
        log_bn = np.log(merged['bn'].values)

        result = pd.Series(np.nan, index=merged.index)
        window = 60

        for i in range(window, len(merged)):
            seg_n = log_n[i - window:i]
            seg_bn = log_bn[i - window:i]

            # OLS: log_bn = beta * log_n + alpha
            x = seg_n - np.mean(seg_n)
            y = seg_bn - np.mean(seg_bn)
            denom = np.dot(x, x)
            if denom == 0:
                continue

            beta = np.dot(x, y) / denom
            alpha = np.mean(seg_bn) - beta * np.mean(seg_n)

            # Spread at current bar
            spread = log_bn[i] - beta * log_n[i] - alpha

            # Z-score using rolling spread stats
            spreads_hist = log_bn[i - window:i + 1] - beta * log_n[i - window:i + 1] - alpha
            s_mean = np.mean(spreads_hist[:-1])
            s_std = np.std(spreads_hist[:-1], ddof=1)

            if s_std > 0:
                result.iloc[i] = (spread - s_mean) / s_std

        # Re-index to match original nifty index
        if 'date' in df_nifty.columns:
            date_to_val = dict(zip(merged['date'], result))
            return df_nifty['date'].map(date_to_val).fillna(np.nan)

        return result.reindex(df_nifty.index, fill_value=np.nan)

    # ================================================================
    # NORMALIZATION (all map to [0, 1])
    # ================================================================

    def _normalize_zscore(self, z: pd.Series) -> pd.Series:
        """Map z-score to [0,1]: z=-4→0, z=0→0.5, z=+4→1."""
        clipped = z.clip(-self.ZSCORE_CLIP, self.ZSCORE_CLIP)
        return (clipped + self.ZSCORE_CLIP) / (2 * self.ZSCORE_CLIP)

    def _normalize_hurst(self, h: pd.Series) -> pd.Series:
        """Map Hurst [0,1] to [0,1] (already bounded)."""
        return h.clip(self.HURST_MIN, self.HURST_MAX)

    def _normalize_half_life(self, hl: pd.Series) -> pd.Series:
        """Map half-life to [0,1]: short HL→1 (strong MR), long→0."""
        clipped = hl.clip(1, self.HALF_LIFE_MAX)
        # Invert: short half-life = strong mean-reversion = high score
        return 1.0 - (clipped - 1) / (self.HALF_LIFE_MAX - 1)

    def _normalize_variance_ratio(self, vr: pd.Series) -> pd.Series:
        """Map VR to [0,1]: VR<1 (MR)→low, VR=1→0.5, VR>1 (trend)→high."""
        clipped = vr.clip(0, self.VR_CLIP)
        return clipped / self.VR_CLIP


# ================================================================
# HELPER: R/S Hurst computation (standalone for performance)
# ================================================================

def _rs_hurst_compute(ts: np.ndarray) -> float:
    """Compute Hurst exponent via rescaled range (R/S) method."""
    n = len(ts)
    if n < 20:
        return np.nan

    max_k = min(n // 2, 50)
    sizes = []
    rs_values = []

    for k in [20, 30, 40, 50]:
        if k > max_k:
            break
        n_chunks = n // k
        if n_chunks < 1:
            continue

        rs_list = []
        for j in range(n_chunks):
            chunk = ts[j * k:(j + 1) * k]
            mean_c = np.mean(chunk)
            devs = np.cumsum(chunk - mean_c)
            r = np.max(devs) - np.min(devs)
            s = np.std(chunk, ddof=1)
            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            sizes.append(np.log(k))
            rs_values.append(np.log(np.mean(rs_list)))

    if len(sizes) < 2:
        return np.nan

    coeffs = np.polyfit(sizes, rs_values, 1)
    return float(np.clip(coeffs[0], 0.0, 1.0))

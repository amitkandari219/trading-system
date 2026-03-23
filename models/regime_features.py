"""
Feature engineering for ML regime classification.

All features computed from data available at bar close — no lookahead.
Returns DataFrame ready for HMM/RF training.
"""

import numpy as np
import pandas as pd

from backtest.indicators import sma, ema, rsi, atr, adx, bollinger_bands, historical_volatility


def compute_regime_features(df):
    """
    Compute regime classification features from daily OHLCV + VIX.

    Args:
        df: DataFrame with [date, open, high, low, close, volume, india_vix]

    Returns:
        DataFrame with all features, no NaN rows (dropped).
    """
    df = df.copy()
    close = df['close']

    # Returns at multiple horizons
    df['returns_1d'] = close.pct_change(1)
    df['returns_5d'] = close.pct_change(5)
    df['returns_20d'] = close.pct_change(20)
    df['returns_60d'] = close.pct_change(60)

    # Volatility
    df['hvol_20'] = historical_volatility(close, 20)
    df['hvol_60'] = historical_volatility(close, 60)
    df['hvol_ratio'] = df['hvol_20'] / df['hvol_60'].replace(0, np.nan)

    # ADX (trend strength)
    df['adx_14'] = adx(df, period=14)

    # VIX features
    if 'india_vix' in df.columns:
        df['vix'] = df['india_vix'].fillna(method='ffill')
        df['vix_change_5d'] = df['vix'].pct_change(5)
        df['vix_zscore'] = (df['vix'] - df['vix'].rolling(60).mean()) / df['vix'].rolling(60).std()
    else:
        df['vix'] = 15.0
        df['vix_change_5d'] = 0.0
        df['vix_zscore'] = 0.0

    # Volume
    vol_avg = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / vol_avg.replace(0, np.nan)

    # Bollinger Band width (proxy for range-bound vs trending)
    bb = bollinger_bands(close, 20)
    df['bb_width'] = bb['bb_bandwidth']

    # RSI
    df['rsi_14'] = rsi(close, 14)

    # Price vs long-term trend
    df['close_vs_sma_200'] = close / sma(close, 200) - 1
    df['close_vs_sma_50'] = close / sma(close, 50) - 1

    # Momentum indicators
    df['macd_hist'] = ema(close, 12) - ema(close, 26)
    df['macd_hist_norm'] = df['macd_hist'] / close  # normalize

    # ATR as % of price
    df['atr_pct'] = atr(df, 14) / close

    # Rolling Sharpe (20-day)
    df['rolling_sharpe_20'] = (
        df['returns_1d'].rolling(20).mean() /
        df['returns_1d'].rolling(20).std().replace(0, np.nan)
    ) * np.sqrt(252)

    # Select feature columns
    FEATURE_COLS = [
        'returns_1d', 'returns_5d', 'returns_20d', 'returns_60d',
        'hvol_20', 'hvol_60', 'hvol_ratio',
        'adx_14',
        'vix', 'vix_change_5d', 'vix_zscore',
        'volume_ratio',
        'bb_width',
        'rsi_14',
        'close_vs_sma_200', 'close_vs_sma_50',
        'macd_hist_norm',
        'atr_pct',
        'rolling_sharpe_20',
    ]

    result = df[['date', 'close'] + FEATURE_COLS].copy()
    result = result.dropna().reset_index(drop=True)

    return result, FEATURE_COLS

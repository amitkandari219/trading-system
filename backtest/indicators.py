"""
Technical indicator library for backtest signal evaluation.

All functions take a pandas DataFrame with OHLCV columns and return
a Series or DataFrame with the indicator values. Column names follow
the convention: {indicator}_{period} (e.g., sma_20, rsi_14).

Usage:
    from backtest.indicators import add_all_indicators
    df = add_all_indicators(history_df)
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, histogram."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': histogram,
    })


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """Single-bar true range."""
    high = df['high']
    low = df['low']
    prev_close = df['close'].shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def bollinger_bands(series: pd.Series, period: int = 20,
                    num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: upper, middle, lower, %B, bandwidth."""
    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'bb_pct_b': pct_b,
        'bb_bandwidth': bandwidth,
    })


def stochastic(df: pd.DataFrame, k_period: int = 14,
               d_period: int = 3) -> pd.DataFrame:
    """Stochastic %K and %D."""
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100.0 * (df['close'] - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return pd.DataFrame({'stoch_k': k, 'stoch_d': d})


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (trend strength, 0-100)."""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = np.where((high - prev_high) > (prev_low - low),
                       np.maximum(high - prev_high, 0), 0)
    minus_dm = np.where((prev_low - low) > (high - prev_high),
                        np.maximum(prev_low - low, 0), 0)

    tr = true_range(df)
    atr_val = tr.ewm(alpha=1.0 / period, min_periods=period).mean()

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period).mean() / atr_val
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period).mean() / atr_val

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, min_periods=period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative VWAP (resets daily — assumes daily bars)."""
    typical = (df['high'] + df['low'] + df['close']) / 3.0
    return (typical * df['volume']).cumsum() / df['volume'].cumsum().replace(0, np.nan)


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Classic pivot points from previous bar's HLC."""
    prev_h = df['high'].shift(1)
    prev_l = df['low'].shift(1)
    prev_c = df['close'].shift(1)
    pp = (prev_h + prev_l + prev_c) / 3.0
    r1 = 2 * pp - prev_l
    s1 = 2 * pp - prev_h
    r2 = pp + (prev_h - prev_l)
    s2 = pp - (prev_h - prev_l)
    r3 = prev_h + 2 * (pp - prev_l)
    s3 = prev_l - 2 * (prev_h - pp)
    return pd.DataFrame({
        'pivot': pp, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2, 'r3': r3, 's3': s3,
    })


def donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian channel: upper (highest high), lower (lowest low)."""
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    middle = (upper + lower) / 2.0
    return pd.DataFrame({
        'dc_upper': upper, 'dc_lower': lower, 'dc_middle': middle,
    })


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume / average volume over period."""
    avg_vol = df['volume'].rolling(window=period, min_periods=period).mean()
    return df['volume'] / avg_vol.replace(0, np.nan)


def historical_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Annualized historical volatility (rolling std of log returns)."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window=period, min_periods=period).std() * np.sqrt(252)


def price_position_in_range(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Where close sits in the period range (0=low, 1=high)."""
    highest = df['high'].rolling(window=period).max()
    lowest = df['low'].rolling(window=period).min()
    denom = (highest - lowest).replace(0, np.nan)
    return (df['close'] - lowest) / denom


def higher_highs(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """True if current high > max of previous `lookback` highs."""
    return df['high'] > df['high'].shift(1).rolling(window=lookback).max()


def lower_lows(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """True if current low < min of previous `lookback` lows."""
    return df['low'] < df['low'].shift(1).rolling(window=lookback).min()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all standard indicators to a copy of the DataFrame."""
    df = df.copy()

    close = df['close']

    # Moving averages
    for p in [5, 10, 20, 40, 50, 80, 100, 200]:
        df[f'sma_{p}'] = sma(close, p)
    for p in [5, 10, 20, 50, 100, 200]:
        df[f'ema_{p}'] = ema(close, p)

    # RSI
    for p in [7, 14, 21]:
        df[f'rsi_{p}'] = rsi(close, p)

    # MACD
    macd_df = macd(close)
    df = pd.concat([df, macd_df], axis=1)

    # ATR
    for p in [7, 14, 20]:
        df[f'atr_{p}'] = atr(df, p)

    # True range
    df['true_range'] = true_range(df)

    # Bollinger Bands
    bb = bollinger_bands(close)
    df = pd.concat([df, bb], axis=1)

    # Stochastic (14-period default)
    stoch = stochastic(df)
    df = pd.concat([df, stoch], axis=1)

    # Stochastic (5-period — for Kaufman's genetic algo rules)
    stoch5 = stochastic(df, k_period=5, d_period=3)
    df['stoch_k_5'] = stoch5['stoch_k']
    df['stoch_d_5'] = stoch5['stoch_d']

    # ADX
    df['adx_14'] = adx(df)

    # Donchian
    dc = donchian(df)
    df = pd.concat([df, dc], axis=1)

    # Volume ratio
    df['vol_ratio_20'] = volume_ratio(df)

    # Volatility
    df['hvol_20'] = historical_volatility(close)
    df['hvol_6'] = historical_volatility(close, period=6)
    df['hvol_100'] = historical_volatility(close, period=100)

    # Pivot points
    pp = pivot_points(df)
    df = pd.concat([df, pp], axis=1)

    # Price position in range
    df['price_pos_20'] = price_position_in_range(df)

    # Daily returns
    df['returns'] = close.pct_change()
    df['log_returns'] = np.log(close / close.shift(1))

    # Previous bar values (for lagged comparisons)
    df['prev_close'] = close.shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_open'] = df['open'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)

    # Bar properties
    df['body'] = df['close'] - df['open']
    df['body_pct'] = df['body'] / df['open']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['range'] = df['high'] - df['low']

    return df

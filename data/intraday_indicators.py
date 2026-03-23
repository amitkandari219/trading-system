"""
Session-aware intraday indicators for Nifty futures.

All indicators reset at session open (9:15 IST).
No look-ahead bias — each bar only sees data up to and including itself.

Usage:
    from data.intraday_indicators import add_intraday_indicators
    df = add_intraday_indicators(df_5min)
"""

import numpy as np
import pandas as pd

from backtest.indicators import sma, ema, rsi, atr, adx, bollinger_bands, stochastic


SESSION_OPEN_TIME = '09:15'
SESSION_CLOSE_TIME = '15:30'


def _trading_date(dt):
    """Extract trading date from datetime (session date, not calendar date)."""
    return dt.date()


def session_vwap(df):
    """
    Compute VWAP that resets at 9:15 each session.
    VWAP = cumulative(price × volume) / cumulative(volume)
    """
    df = df.copy()
    df['_date'] = df['datetime'].dt.date
    df['_typical'] = (df['high'] + df['low'] + df['close']) / 3
    df['_tpv'] = df['_typical'] * df['volume']

    # Cumulative within each session
    df['_cum_tpv'] = df.groupby('_date')['_tpv'].cumsum()
    df['_cum_vol'] = df.groupby('_date')['volume'].cumsum()

    df['vwap'] = df['_cum_tpv'] / df['_cum_vol'].replace(0, np.nan)
    df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'].replace(0, np.nan)

    return df.drop(columns=['_date', '_typical', '_tpv', '_cum_tpv', '_cum_vol'])


def opening_range(df, n_bars=3):
    """
    Compute opening range high/low from first N bars of each session.
    Default: 3 bars of 5-min = first 15 minutes.

    Returns df with opening_range_high, opening_range_low, or_breakout, or_breakdown.
    """
    df = df.copy()
    df['_date'] = df['datetime'].dt.date
    df['_bar_num'] = df.groupby('_date').cumcount()

    # First N bars high/low per session
    first_n = df[df['_bar_num'] < n_bars].groupby('_date').agg(
        opening_range_high=('high', 'max'),
        opening_range_low=('low', 'min'),
    )

    df = df.merge(first_n, left_on='_date', right_index=True, how='left')

    # Days with too few bars → leave as NaN (skip day, don't fake the range)
    # The old fillna(df['high']) made OR_high == bar high, preventing breakouts

    # Breakout/breakdown flags (only valid after opening range is established)
    df['or_breakout'] = (df['_bar_num'] >= n_bars) & (df['close'] > df['opening_range_high'])
    df['or_breakdown'] = (df['_bar_num'] >= n_bars) & (df['close'] < df['opening_range_low'])

    return df.drop(columns=['_date', '_bar_num'])


def session_features(df):
    """
    Add session context features:
    - session_bar: bar number within session (0-indexed)
    - session_half: 'AM' or 'PM'
    - is_first_hour: True for bars 9:15-10:15
    - is_last_hour: True for bars 14:30-15:30
    - time_to_close: minutes until 15:30
    - overnight_gap_pct: (today open - yesterday close) / yesterday close
    """
    df = df.copy()
    df['_date'] = df['datetime'].dt.date
    df['_time'] = df['datetime'].dt.time

    df['session_bar'] = df.groupby('_date').cumcount()

    df['session_half'] = np.where(
        df['datetime'].dt.hour < 12, 'AM', 'PM'
    )

    df['is_first_hour'] = (df['datetime'].dt.hour == 9) | (
        (df['datetime'].dt.hour == 10) & (df['datetime'].dt.minute <= 15)
    )

    df['is_last_hour'] = (
        (df['datetime'].dt.hour == 14) & (df['datetime'].dt.minute >= 30)
    ) | (df['datetime'].dt.hour == 15)

    # Minutes to close (15:30)
    df['time_to_close'] = (
        15 * 60 + 30 - df['datetime'].dt.hour * 60 - df['datetime'].dt.minute
    ).clip(lower=0)

    # Overnight gap
    session_open = df.groupby('_date')['open'].first()
    session_prev_close = df.groupby('_date')['close'].last().shift(1)
    gap = (session_open - session_prev_close) / session_prev_close.replace(0, np.nan)
    gap_map = gap.to_dict()
    df['overnight_gap_pct'] = df['_date'].map(gap_map)

    # Prior day high/low/close
    daily = df.groupby('_date').agg(
        _day_high=('high', 'max'),
        _day_low=('low', 'min'),
        _day_close=('close', 'last'),
    )
    daily['prev_day_high'] = daily['_day_high'].shift(1)
    daily['prev_day_low'] = daily['_day_low'].shift(1)
    daily['prev_day_close'] = daily['_day_close'].shift(1)

    df = df.merge(
        daily[['prev_day_high', 'prev_day_low', 'prev_day_close']],
        left_on='_date', right_index=True, how='left'
    )

    return df.drop(columns=['_date', '_time'])


def intraday_technicals(df):
    """
    Add standard technical indicators computed on intraday bars.
    Periods are adjusted for bar frequency (assumes 5-min bars).
    """
    df = df.copy()
    close = df['close']

    # SMAs — 20 bars = 100 min, 50 bars = ~4 hours
    df['sma_20'] = sma(close, 20)
    df['sma_50'] = sma(close, 50)

    # EMAs
    df['ema_9'] = ema(close, 9)
    df['ema_20'] = ema(close, 20)

    # RSI (14 bars = 70 min)
    df['rsi_14'] = rsi(close, 14)

    # ATR (14 bars)
    if 'high' in df.columns and 'low' in df.columns:
        df['atr_14'] = atr(df, 14)

    # ADX (14 bars)
    if 'high' in df.columns and 'low' in df.columns:
        df['adx_14'] = adx(df)

    # Bollinger Bands (20-period)
    bb = bollinger_bands(close, 20)
    for col in bb.columns:
        df[col] = bb[col].values

    # Stochastic (14-period)
    stoch = stochastic(df, k_period=14, d_period=3)
    for col in stoch.columns:
        df[col] = stoch[col].values

    # Bar properties
    df['body'] = df['close'] - df['open']
    df['body_pct'] = abs(df['body']) / (df['high'] - df['low']).replace(0, np.nan)
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['range'] = df['high'] - df['low']

    # Previous bar values
    df['prev_close'] = close.shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_open'] = df['open'].shift(1)

    # Returns
    df['returns'] = close.pct_change()

    # Volume ratio (20-bar average)
    vol_avg = df['volume'].rolling(20, min_periods=5).mean()
    df['vol_ratio_20'] = df['volume'] / vol_avg.replace(0, np.nan)

    return df


def add_intraday_indicators(df):
    """
    Add all intraday indicators to a DataFrame of 5-min bars.
    Mirrors add_all_indicators() from backtest/indicators.py.

    Args:
        df: DataFrame with columns [datetime, open, high, low, close, volume]

    Returns:
        DataFrame with all indicators added.
    """
    df = df.copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    # Drop DB-loaded placeholder columns (will be recomputed)
    for col in ['vwap', 'vwap_deviation', 'opening_range_high', 'opening_range_low']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Session-aware indicators
    df = session_vwap(df)
    df = opening_range(df, n_bars=3)  # 15-min opening range
    df = session_features(df)

    # Standard technicals
    df = intraday_technicals(df)

    return df

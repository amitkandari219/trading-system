#!/usr/bin/env python3
"""
Offline intraday walk-forward backtest for 6 failed-daily candidates.

Reconstructs daily OHLCV from trade log prices (no DB needed).
Generates synthetic 5-min bars and runs walk-forward validation.

Usage:
    python run_intraday_offline.py
"""

import json
import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd

# ── Import indicator functions from backtest module ──
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.indicators import (
    sma, ema, rsi, atr, adx, bollinger_bands, stochastic, pivot_points,
    volume_ratio, historical_volatility
)

SLIPPAGE_PCT = 0.0005
SESSION_CLOSE_HOUR = 15
SESSION_CLOSE_MIN = 20
NIFTY_LOT_SIZE = 25
CAPITAL = 1_000_000

# Walk-forward parameters
WF_TRAIN_MONTHS = 12
WF_TEST_MONTHS = 4
WF_STEP_MONTHS = 2
WF_PASS_THRESHOLD = 0.70

# Per-window criteria
MIN_SHARPE = 0.80
MIN_TRADES = 12
MIN_WIN_RATE = 0.38
MIN_PROFIT_FACTOR = 1.15

# ── Volume profile for synthetic 5-min bars ──
VOLUME_PROFILE_75 = np.array([
    3.5, 3.0, 2.5, 2.2, 2.0, 1.8, 1.7, 1.6, 1.5,
    1.4, 1.3, 1.3, 1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 1.0, 0.9, 0.9,
    0.9, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6,
    0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1,
    1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3,
    2.5, 2.7, 3.0, 3.2, 3.5, 4.0,
])
VOLUME_PROFILE_75 = VOLUME_PROFILE_75 / VOLUME_PROFILE_75.sum()


# ════════════════════════════════════════════════════════════════
# DATA LOADING — reconstruct OHLCV from trade logs
# ════════════════════════════════════════════════════════════════

def load_daily_from_trade_logs():
    """Reconstruct daily OHLCV from trade log prices."""
    dfs = []
    trade_dir = os.path.join(os.path.dirname(__file__), 'trade_analysis')

    for d in os.listdir(trade_dir):
        p = os.path.join(trade_dir, d, 'trade_log.csv')
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if 'entry_price' not in df.columns:
            continue

        # Collect entry and exit prices
        for _, row in df.iterrows():
            for date_col, price_col in [('entry_date', 'entry_price'), ('exit_date', 'exit_price')]:
                if pd.notna(row.get(date_col)) and pd.notna(row.get(price_col)):
                    vix_val = float(row.get('vix', 15.0)) if pd.notna(row.get('vix')) else 15.0
                    dfs.append({
                        'date': row[date_col],
                        'price': float(row[price_col]),
                        'vix': vix_val,
                    })

    if not dfs:
        return pd.DataFrame()

    prices = pd.DataFrame(dfs)
    prices['date'] = pd.to_datetime(prices['date'])

    # Build daily OHLCV approximation
    daily = prices.groupby('date').agg(
        close=('price', 'mean'),
        india_vix=('vix', 'mean'),
    ).reset_index().sort_values('date').reset_index(drop=True)

    # Reconstruct OHLC from close + synthetic noise
    rng = np.random.default_rng(42)
    closes = daily['close'].values
    daily['open'] = np.roll(closes, 1)
    daily.loc[0, 'open'] = closes[0]

    # High/Low from realistic intraday range (~1.2% avg for Nifty)
    daily_range = closes * 0.012
    daily['high'] = np.maximum(daily['close'], daily['open']) + daily_range * rng.uniform(0.1, 0.5, len(daily))
    daily['low'] = np.minimum(daily['close'], daily['open']) - daily_range * rng.uniform(0.1, 0.5, len(daily))
    daily['volume'] = (rng.uniform(0.8, 1.2, len(daily)) * 200_000_000).astype(int)

    return daily


def generate_5min_bars_simple(daily_row, rng):
    """Generate 75 synthetic 5-min bars from daily OHLCV."""
    day_open = float(daily_row['open'])
    day_high = float(daily_row['high'])
    day_low = float(daily_row['low'])
    day_close = float(daily_row['close'])
    day_volume = int(daily_row['volume'])
    day_date = pd.Timestamp(daily_row['date'])

    n = 75
    day_range = max(day_high - day_low, day_open * 0.005)

    steps = rng.normal(0, 1, n)
    cum = np.cumsum(steps)
    cum = cum - cum[-1]
    path = day_open + cum * (day_range / (4 * np.std(cum) + 1e-9))
    path = path - (path[-1] - day_close) * np.linspace(0, 1, n)

    high_bar = rng.integers(0, n)
    low_bar = rng.integers(0, n)
    while low_bar == high_bar:
        low_bar = rng.integers(0, n)
    path[high_bar] = day_high - day_range * 0.01
    path[low_bar] = day_low + day_range * 0.01

    bars = []
    for i in range(n):
        bar_time = day_date + timedelta(hours=9, minutes=15 + i * 5)
        bar_close = float(np.clip(path[i], day_low, day_high))
        bar_open = float(np.clip(path[i-1] if i > 0 else day_open, day_low, day_high))
        noise = day_range * 0.003 * rng.uniform(0.5, 1.5)
        bar_high = min(max(bar_open, bar_close) + noise, day_high)
        bar_low = max(min(bar_open, bar_close) - noise, day_low)

        if i == 0: bar_open = day_open
        if i == n - 1: bar_close = day_close
        if i == high_bar: bar_high = day_high
        if i == low_bar: bar_low = day_low

        bars.append({
            'datetime': bar_time,
            'open': round(bar_open, 2),
            'high': round(bar_high, 2),
            'low': round(bar_low, 2),
            'close': round(bar_close, 2),
            'volume': int(day_volume * VOLUME_PROFILE_75[i]),
        })
    return bars


def add_intraday_indicators_standalone(df):
    """Add all intraday indicators (standalone, no imports from data/)."""
    df = df.copy().sort_values('datetime').reset_index(drop=True)
    close = df['close']

    # Session VWAP
    df['_date'] = df['datetime'].dt.date
    df['_typical'] = (df['high'] + df['low'] + df['close']) / 3
    df['_tpv'] = df['_typical'] * df['volume']
    df['_cum_tpv'] = df.groupby('_date')['_tpv'].cumsum()
    df['_cum_vol'] = df.groupby('_date')['volume'].cumsum()
    df['vwap'] = df['_cum_tpv'] / df['_cum_vol'].replace(0, np.nan)
    df['vwap_deviation'] = (close - df['vwap']) / df['vwap'].replace(0, np.nan)

    # Opening range (15-min = first 3 bars)
    df['_bar_num'] = df.groupby('_date').cumcount()
    first_3 = df[df['_bar_num'] < 3].groupby('_date').agg(
        opening_range_high=('high', 'max'), opening_range_low=('low', 'min'))
    df = df.merge(first_3, left_on='_date', right_index=True, how='left')
    df['or_breakout'] = (df['_bar_num'] >= 3) & (close > df['opening_range_high'])
    df['or_breakdown'] = (df['_bar_num'] >= 3) & (close < df['opening_range_low'])

    # Session features
    df['session_bar'] = df.groupby('_date').cumcount()
    df['time_to_close'] = (15*60 + 30 - df['datetime'].dt.hour * 60 - df['datetime'].dt.minute).clip(lower=0)

    # Prior day high/low/close
    daily_agg = df.groupby('_date').agg(
        _day_high=('high', 'max'), _day_low=('low', 'min'), _day_close=('close', 'last'))
    daily_agg['prev_day_high'] = daily_agg['_day_high'].shift(1)
    daily_agg['prev_day_low'] = daily_agg['_day_low'].shift(1)
    daily_agg['prev_day_close'] = daily_agg['_day_close'].shift(1)
    df = df.merge(daily_agg[['prev_day_high', 'prev_day_low', 'prev_day_close']],
                  left_on='_date', right_index=True, how='left')

    # Overnight gap
    sess_open = df.groupby('_date')['open'].first()
    sess_prev_close = df.groupby('_date')['close'].last().shift(1)
    gap = (sess_open - sess_prev_close) / sess_prev_close.replace(0, np.nan)
    df['overnight_gap_pct'] = df['_date'].map(gap.to_dict())

    # Technical indicators on 5-min bars
    df['sma_20'] = sma(close, 20)
    df['sma_50'] = sma(close, 50)
    df['ema_9'] = ema(close, 9)
    df['ema_20'] = ema(close, 20)
    df['rsi_14'] = rsi(close, 14)
    df['atr_14'] = atr(df, 14)
    df['adx_14'] = adx(df)
    bb = bollinger_bands(close, 20)
    for col in bb.columns:
        df[col] = bb[col].values
    stoch = stochastic(df, k_period=14, d_period=3)
    for col in stoch.columns:
        df[col] = stoch[col].values

    df['body'] = close - df['open']
    df['body_pct'] = abs(df['body']) / (df['high'] - df['low']).replace(0, np.nan)
    df['prev_close'] = close.shift(1)
    df['returns'] = close.pct_change()

    vol_avg = df['volume'].rolling(20, min_periods=5).mean()
    df['vol_ratio_20'] = df['volume'] / vol_avg.replace(0, np.nan)

    # Clean temp columns
    for c in ['_date', '_typical', '_tpv', '_cum_tpv', '_cum_vol', '_bar_num']:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df


# ════════════════════════════════════════════════════════════════
# SIGNAL DEFINITIONS
# ════════════════════════════════════════════════════════════════

INTRADAY_CANDIDATES = {
    'ID_KAUFMAN_BB_MR': {
        'source': 'KAUFMAN_DRY_22',
        'description': 'BB mean-reversion at band extremes (ADX<25)',
        'direction': 'BOTH',
        'stop_loss_pct': 0.004,
        'max_hold_bars': 30,
        'entry_start': '09:45',
        'entry_end': '14:30',
    },
    'ID_GUJRAL_RANGE': {
        'source': 'GUJRAL_DRY_12',
        'description': 'Range boundary trades with ADX filter',
        'direction': 'BOTH',
        'stop_loss_pct': 0.005,
        'max_hold_bars': 35,
        'entry_start': '10:00',
        'entry_end': '14:30',
    },
    'ID_GUJRAL_PIVOT_R1': {
        'source': 'GUJRAL_DRY_15',
        'description': 'Pivot R1 breakout in trending markets (ADX>25)',
        'direction': 'LONG',
        'stop_loss_pct': 0.004,
        'max_hold_bars': 30,
        'entry_start': '09:30',
        'entry_end': '14:00',
    },
    'ID_KAUFMAN_SMA_MR': {
        'source': 'KAUFMAN_DRY_19',
        'description': 'Mean-reversion at SMA(50) zones',
        'direction': 'BOTH',
        'stop_loss_pct': 0.005,
        'max_hold_bars': 25,
        'entry_start': '09:45',
        'entry_end': '14:30',
    },
    'ID_KAUFMAN_IMPULSE': {
        'source': 'KAUFMAN_DRY_15',
        'description': 'Momentum impulse on 0.8%+ intraday move',
        'direction': 'BOTH',
        'stop_loss_pct': 0.004,
        'max_hold_bars': 20,
        'entry_start': '09:30',
        'entry_end': '14:00',
    },
    'ID_GRIMES_ATR_BURST': {
        'source': 'GRIMES_DRY_8_0',
        'description': 'Volatility impulse: returns > 2x ATR + volume',
        'direction': 'BOTH',
        'stop_loss_pct': 0.005,
        'max_hold_bars': 25,
        'entry_start': '09:30',
        'entry_end': '14:30',
    },
}


def _is_session_close(dt):
    return dt.hour > SESSION_CLOSE_HOUR or (dt.hour == SESSION_CLOSE_HOUR and dt.minute >= SESSION_CLOSE_MIN)

def _time_in_range(dt, start_str, end_str):
    h1, m1 = map(int, start_str.split(':'))
    h2, m2 = map(int, end_str.split(':'))
    bar_mins = dt.hour * 60 + dt.minute
    return (h1*60+m1) <= bar_mins <= (h2*60+m2)


# ════════════════════════════════════════════════════════════════
# SIGNAL CHECK FUNCTIONS
# ════════════════════════════════════════════════════════════════

def check_kaufman_bb_mr(bar, prev_bar, session_bars, config):
    if prev_bar is None: return None
    close = float(bar['close'])
    adx_val = float(bar['adx_14']) if pd.notna(bar.get('adx_14')) else 30
    if adx_val >= 25: return None
    bb_u, bb_l = bar.get('bb_upper'), bar.get('bb_lower')
    if pd.isna(bb_u) or pd.isna(bb_l): return None
    bb_u, bb_l = float(bb_u), float(bb_l)
    pc = float(prev_bar['close'])
    if close <= bb_l and pc > bb_l:
        return {'direction': 'LONG', 'price': close, 'reason': f'BB_MR long ADX={adx_val:.0f}'}
    if close >= bb_u and pc < bb_u:
        return {'direction': 'SHORT', 'price': close, 'reason': f'BB_MR short ADX={adx_val:.0f}'}
    return None

def check_gujral_range(bar, prev_bar, session_bars, config):
    if len(session_bars) < 20: return None
    close = float(bar['close'])
    adx_val = float(bar['adx_14']) if pd.notna(bar.get('adx_14')) else 30
    if adx_val >= 25: return None
    sh = float(session_bars['high'].max())
    sl = float(session_bars['low'].min())
    sr = sh - sl
    if sr <= 0: return None
    th = sr * 0.20
    if close <= sl + th and close > sl and close > float(bar['open']):
        return {'direction': 'LONG', 'price': close, 'reason': 'RANGE long near sess_low'}
    if close >= sh - th and close < sh and close < float(bar['open']):
        return {'direction': 'SHORT', 'price': close, 'reason': 'RANGE short near sess_high'}
    return None

def check_gujral_pivot_r1(bar, prev_bar, session_bars, config):
    if prev_bar is None: return None
    close = float(bar['close'])
    pc = float(prev_bar['close'])
    adx_val = float(bar['adx_14']) if pd.notna(bar.get('adx_14')) else 20
    if adx_val < 25: return None
    pdh, pdl, pdc = bar.get('prev_day_high'), bar.get('prev_day_low'), bar.get('prev_day_close')
    if pd.isna(pdh) or pd.isna(pdl) or pd.isna(pdc): return None
    pivot = (float(pdh) + float(pdl) + float(pdc)) / 3.0
    r1 = 2 * pivot - float(pdl)
    vr = float(bar['vol_ratio_20']) if pd.notna(bar.get('vol_ratio_20')) else 0.8
    if vr < 1.1: return None
    if close > r1 and pc <= r1:
        return {'direction': 'LONG', 'price': close, 'reason': f'PIVOT_R1 long R1={r1:.0f}'}
    return None

def check_kaufman_sma_mr(bar, prev_bar, session_bars, config):
    if prev_bar is None: return None
    close = float(bar['close'])
    pc = float(prev_bar['close'])
    s50 = bar.get('sma_50')
    if pd.isna(s50): return None
    s50 = float(s50)
    off = s50 * 0.003
    lo, hi = s50 - off, s50 + off
    if close < lo and pc >= lo:
        return {'direction': 'LONG', 'price': close, 'reason': 'SMA_MR long below zone'}
    if close > hi and pc <= hi:
        return {'direction': 'SHORT', 'price': close, 'reason': 'SMA_MR short above zone'}
    return None

def check_kaufman_impulse(bar, prev_bar, session_bars, config):
    if len(session_bars) < 5 or prev_bar is None: return None
    close = float(bar['close'])
    so = float(session_bars.iloc[0]['open'])
    if so <= 0: return None
    sr = (close - so) / so
    pr = (float(prev_bar['close']) - so) / so
    vr = float(bar['vol_ratio_20']) if pd.notna(bar.get('vol_ratio_20')) else 0.8
    if vr < 1.2: return None
    if sr > 0.008 and pr <= 0.008:
        return {'direction': 'LONG', 'price': close, 'reason': f'IMPULSE long ret={sr:.2%}'}
    if sr < -0.008 and pr >= -0.008:
        return {'direction': 'SHORT', 'price': close, 'reason': f'IMPULSE short ret={sr:.2%}'}
    return None

def check_grimes_atr_burst(bar, prev_bar, session_bars, config):
    if prev_bar is None: return None
    close = float(bar['close'])
    pc = float(prev_bar['close'])
    a = bar.get('atr_14')
    vr = bar.get('vol_ratio_20')
    if pd.isna(a) or pd.isna(vr): return None
    a, vr = float(a), float(vr)
    if a <= 0 or vr < 1.3: return None
    ret = close - pc
    if ret > 2.0 * a:
        return {'direction': 'LONG', 'price': close, 'reason': f'ATR_BURST long move={ret:.0f}'}
    if ret < -2.0 * a:
        return {'direction': 'SHORT', 'price': close, 'reason': f'ATR_BURST short move={ret:.0f}'}
    return None

SIGNAL_CHECKS = {
    'ID_KAUFMAN_BB_MR': check_kaufman_bb_mr,
    'ID_GUJRAL_RANGE': check_gujral_range,
    'ID_GUJRAL_PIVOT_R1': check_gujral_pivot_r1,
    'ID_KAUFMAN_SMA_MR': check_kaufman_sma_mr,
    'ID_KAUFMAN_IMPULSE': check_kaufman_impulse,
    'ID_GRIMES_ATR_BURST': check_grimes_atr_burst,
}


# ════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════

def run_signal_on_period(signal_id, config, df_daily_period):
    check_fn = SIGNAL_CHECKS[signal_id]
    rng = np.random.default_rng(42)
    trades = []
    position = None
    bars_held = 0
    daily_pnl = {}

    for i in range(len(df_daily_period)):
        daily_row = df_daily_period.iloc[i]
        trading_date = pd.Timestamp(daily_row['date']).date()

        bars_data = generate_5min_bars_simple(daily_row, rng)
        df_bars = pd.DataFrame(bars_data)
        df_bars = add_intraday_indicators_standalone(df_bars)

        session_bars_so_far = pd.DataFrame()

        for idx in range(len(df_bars)):
            bar = df_bars.iloc[idx]
            prev_bar = df_bars.iloc[idx-1] if idx > 0 else None
            bar_dt = bar['datetime']
            close = float(bar['close'])

            session_bars_so_far = pd.concat(
                [session_bars_so_far, bar.to_frame().T], ignore_index=True)

            if position is not None:
                bars_held += 1
                d = position['direction']
                ep = position['entry_price']
                exit_reason = None

                if _is_session_close(bar_dt):
                    exit_reason = 'session_close'
                if not exit_reason:
                    loss = (ep - close)/ep if d == 'LONG' else (close - ep)/ep
                    if loss >= config['stop_loss_pct']:
                        exit_reason = 'stop_loss'
                if not exit_reason and bars_held >= config['max_hold_bars']:
                    exit_reason = 'max_hold'
                if not exit_reason and signal_id == 'ID_KAUFMAN_BB_MR':
                    bb_mid = bar.get('bb_middle')
                    if pd.notna(bb_mid):
                        bb_mid = float(bb_mid)
                        if (d == 'LONG' and close >= bb_mid) or (d == 'SHORT' and close <= bb_mid):
                            exit_reason = 'target_bb_mid'

                if exit_reason:
                    if d == 'LONG':
                        xp = close * (1 - SLIPPAGE_PCT)
                        pnl_pts = xp - ep
                    else:
                        xp = close * (1 + SLIPPAGE_PCT)
                        pnl_pts = ep - xp
                    pnl_rs = pnl_pts * NIFTY_LOT_SIZE
                    trades.append({'pnl_rs': pnl_rs, 'exit_reason': exit_reason, 'date': trading_date})
                    daily_pnl.setdefault(trading_date, 0)
                    daily_pnl[trading_date] += pnl_rs
                    position = None
                    bars_held = 0

            if position is None and not _is_session_close(bar_dt):
                if _time_in_range(bar_dt, config['entry_start'], config['entry_end']):
                    result = check_fn(bar, prev_bar, session_bars_so_far, config)
                    if result:
                        adj = close * (1 + SLIPPAGE_PCT) if result['direction'] == 'LONG' else close * (1 - SLIPPAGE_PCT)
                        position = {'direction': result['direction'], 'entry_price': round(adj, 2)}
                        bars_held = 0

        # Force close at session end
        if position is not None:
            last = df_bars.iloc[-1]
            c = float(last['close'])
            d = position['direction']
            ep = position['entry_price']
            if d == 'LONG':
                xp = c * (1-SLIPPAGE_PCT); pnl_pts = xp - ep
            else:
                xp = c * (1+SLIPPAGE_PCT); pnl_pts = ep - xp
            pnl_rs = pnl_pts * NIFTY_LOT_SIZE
            trades.append({'pnl_rs': pnl_rs, 'exit_reason': 'forced_session_end', 'date': trading_date})
            daily_pnl.setdefault(trading_date, 0)
            daily_pnl[trading_date] += pnl_rs
            position = None

    if not trades:
        return {'sharpe': 0, 'win_rate': 0, 'profit_factor': 0, 'trade_count': 0, 'total_pnl': 0, 'max_dd': 1.0}

    pnls = [t['pnl_rs'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins)/len(pnls)
    pf = sum(wins)/abs(sum(losses)) if losses and sum(losses) != 0 else 0
    dr = pd.Series(daily_pnl) / CAPITAL
    sharpe = dr.mean()/dr.std()*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0
    cum = np.cumsum(pnls)
    pk = np.maximum.accumulate(cum)
    dd = cum - pk
    max_dd = abs(dd.min())/CAPITAL if CAPITAL > 0 else 0

    return {
        'sharpe': round(float(sharpe), 3),
        'win_rate': round(win_rate, 3),
        'profit_factor': round(float(pf), 3),
        'trade_count': len(trades),
        'total_pnl': round(sum(pnls), 0),
        'max_dd': round(float(max_dd), 4),
    }


# ════════════════════════════════════════════════════════════════
# WALK-FORWARD
# ════════════════════════════════════════════════════════════════

def walk_forward(signal_id, config, df_daily):
    dates = pd.to_datetime(df_daily['date'])
    min_d, max_d = dates.min(), dates.max()

    windows = []
    ts = min_d
    while True:
        te = ts + pd.DateOffset(months=WF_TRAIN_MONTHS)
        test_s = te + pd.DateOffset(days=10)
        test_e = test_s + pd.DateOffset(months=WF_TEST_MONTHS)
        if test_e > max_d:
            break
        windows.append({'test_start': test_s, 'test_end': test_e})
        ts += pd.DateOffset(months=WF_STEP_MONTHS)

    if not windows:
        return {'pass': False, 'reason': 'insufficient_data', 'windows': []}

    results = []
    for i, w in enumerate(windows):
        mask = (dates >= w['test_start']) & (dates <= w['test_end'])
        df_test = df_daily[mask].copy()
        if len(df_test) < 15:
            results.append({'window': i, 'pass': False, 'sharpe': 0, 'trades': 0, 'pnl': 0,
                            'win_rate': 0, 'profit_factor': 0, 'max_dd': 0,
                            'test_period': f"{w['test_start'].strftime('%Y-%m')} to {w['test_end'].strftime('%Y-%m')}"})
            continue

        m = run_signal_on_period(signal_id, config, df_test)
        wp = (m['sharpe'] >= MIN_SHARPE and m['trade_count'] >= MIN_TRADES and
              m['win_rate'] >= MIN_WIN_RATE and m['profit_factor'] >= MIN_PROFIT_FACTOR)

        results.append({
            'window': i, 'pass': wp,
            'test_period': f"{w['test_start'].strftime('%Y-%m')} to {w['test_end'].strftime('%Y-%m')}",
            'sharpe': m['sharpe'], 'win_rate': m['win_rate'],
            'profit_factor': m['profit_factor'], 'trades': m['trade_count'],
            'pnl': m['total_pnl'], 'max_dd': m['max_dd'],
        })

    n_p = sum(1 for r in results if r['pass'])
    n_t = len(results)
    pr = n_p/n_t if n_t > 0 else 0
    recent = results[-1]['pass'] if results else False

    return {
        'pass': pr >= WF_PASS_THRESHOLD and recent,
        'pass_rate': round(pr, 3), 'n_passed': n_p, 'n_total': n_t,
        'recent_pass': recent, 'windows': results,
        'avg_sharpe': round(np.mean([r['sharpe'] for r in results]), 3),
        'avg_pnl': round(np.mean([r['pnl'] for r in results]), 0),
        'total_trades': sum(r['trades'] for r in results),
    }


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("INTRADAY WALK-FORWARD BACKTEST — 6 Failed-Daily Candidates")
    print("=" * 80)
    print(f"WF: {WF_TRAIN_MONTHS}mo train / {WF_TEST_MONTHS}mo test / {WF_STEP_MONTHS}mo step")
    print(f"Pass: Sharpe≥{MIN_SHARPE}, Trades≥{MIN_TRADES}, WR≥{MIN_WIN_RATE:.0%}, PF≥{MIN_PROFIT_FACTOR}")
    print(f"Overall: ≥{WF_PASS_THRESHOLD:.0%} windows + recent must pass")
    print()

    print("Loading daily data from trade logs...", flush=True)
    df_daily = load_daily_from_trade_logs()
    if df_daily.empty:
        print("ERROR: No data found in trade_analysis/")
        return

    # Filter from 2019 onwards for enough data
    df_daily = df_daily[df_daily['date'] >= '2019-01-01'].reset_index(drop=True)
    print(f"  {len(df_daily)} trading days: {df_daily['date'].min().date()} to {df_daily['date'].max().date()}")
    print()

    all_results = {}
    for signal_id, config in INTRADAY_CANDIDATES.items():
        print(f"{'─' * 80}")
        print(f"Testing: {signal_id} ({config['source']})")
        print(f"  {config['description']}")
        print(f"  Dir: {config['direction']}  SL: {config['stop_loss_pct']:.1%}  "
              f"MaxHold: {config['max_hold_bars']}bars  Time: {config['entry_start']}-{config['entry_end']}")
        print()

        wf = walk_forward(signal_id, config, df_daily)
        all_results[signal_id] = wf

        status = "PASS" if wf['pass'] else "FAIL"
        print(f"  Result: {status}")
        print(f"  Windows: {wf['n_passed']}/{wf['n_total']} ({wf['pass_rate']:.0%})")
        print(f"  Recent: {'PASS' if wf['recent_pass'] else 'FAIL'}")
        print(f"  Avg Sharpe: {wf['avg_sharpe']:.2f}  Avg P&L: Rs{wf['avg_pnl']:,.0f}  Total trades: {wf['total_trades']}")

        print(f"\n  {'Win':<4} {'Period':<22} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'Tr':>5} {'P&L':>10} {'DD':>7} {'Res':>5}")
        for w in wf['windows']:
            p = w.get('test_period', 'N/A')
            r = 'PASS' if w['pass'] else 'FAIL'
            print(f"  {w['window']:<4} {p:<22} {w['sharpe']:>7.2f} {w.get('win_rate',0):>5.0%} "
                  f"{w.get('profit_factor',0):>6.2f} {w['trades']:>5} {w['pnl']:>9,.0f} "
                  f"{w.get('max_dd',0):>6.2%} {r:>5}")
        print()

    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    passed = {k: v for k, v in all_results.items() if v['pass']}
    failed = {k: v for k, v in all_results.items() if not v['pass']}

    print(f"\n  PASSED ({len(passed)}/{len(all_results)}):")
    if passed:
        for s, r in sorted(passed.items(), key=lambda x: -x[1]['avg_sharpe']):
            c = INTRADAY_CANDIDATES[s]
            print(f"    {s:<25s} ({c['source']:<18s}) WF={r['pass_rate']:.0%}  "
                  f"Sharpe={r['avg_sharpe']:.2f}  trades={r['total_trades']}")
    else:
        print("    None")

    print(f"\n  FAILED ({len(failed)}/{len(all_results)}):")
    for s, r in failed.items():
        c = INTRADAY_CANDIDATES[s]
        reason = f"WF={r['pass_rate']:.0%}"
        if not r['recent_pass']:
            reason += " (recent FAIL)"
        print(f"    {s:<25s} ({c['source']:<18s}) {reason}  Sharpe={r['avg_sharpe']:.2f}")

    os.makedirs('backtest_results/intraday', exist_ok=True)
    save = {s: {
        'source': INTRADAY_CANDIDATES[s]['source'],
        'pass': r['pass'], 'pass_rate': r['pass_rate'],
        'n_passed': r['n_passed'], 'n_total': r['n_total'],
        'recent_pass': r['recent_pass'],
        'avg_sharpe': r['avg_sharpe'], 'avg_pnl': r['avg_pnl'],
        'total_trades': r['total_trades'],
        'windows': r['windows'],
    } for s, r in all_results.items()}

    with open('backtest_results/intraday/candidate_wf_results.json', 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\n  Saved: backtest_results/intraday/candidate_wf_results.json")
    print("=" * 80)


if __name__ == '__main__':
    main()

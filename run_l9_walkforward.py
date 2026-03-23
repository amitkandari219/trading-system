#!/usr/bin/env python3
"""
Walk-forward validation for L9 intraday signals.

Tests the 3 most promising L9 signals (ORB_BREAKOUT, GAP_FILL, VWAP_RECLAIM)
using the same offline methodology as run_intraday_offline.py.

Parameters:
    Train: 12 months, Test: 4 months, Step: 2 months
    Pass: Sharpe≥0.80, Trades≥8, WR≥35%, PF≥1.10 (relaxed vs daily)

Usage:
    python run_l9_walkforward.py
"""

import json
import math
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.indicators import (
    sma, ema, rsi, atr, adx, bollinger_bands, stochastic, volume_ratio
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

# Relaxed per-window criteria for L9
MIN_SHARPE = 0.80
MIN_TRADES = 8
MIN_WIN_RATE = 0.35
MIN_PROFIT_FACTOR = 1.10

# Volume profile for synthetic 5-min bars
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
# DATA LOADING
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

        for _, row in df.iterrows():
            for date_col, price_col in [('entry_date', 'entry_price'), ('exit_date', 'exit_price')]:
                if pd.notna(row.get(date_col)) and pd.notna(row.get(price_col)):
                    try:
                        dt = pd.to_datetime(row[date_col])
                        price = float(row[price_col])
                        if 5000 < price < 50000:
                            dfs.append({'date': dt.date(), 'price': price})
                    except (ValueError, TypeError):
                        continue

    if not dfs:
        raise RuntimeError("No trade log data found")

    pdf = pd.DataFrame(dfs)
    daily = pdf.groupby('date')['price'].agg(['mean', 'min', 'max', 'count']).reset_index()
    daily.columns = ['date', 'close', 'low_raw', 'high_raw', 'obs_count']
    daily = daily.sort_values('date').reset_index(drop=True)

    rng = np.random.RandomState(42)
    daily['range_pct'] = 0.012  # 1.2% daily range
    daily['open'] = daily['close'].shift(1).fillna(daily['close'])
    daily['high'] = daily[['close', 'open']].max(axis=1) * (
        1 + daily['range_pct'] * rng.uniform(0.3, 0.7, len(daily)))
    daily['low'] = daily[['close', 'open']].min(axis=1) * (
        1 - daily['range_pct'] * rng.uniform(0.3, 0.7, len(daily)))
    daily['volume'] = (rng.uniform(5e6, 15e6, len(daily))).astype(int)
    daily['date'] = pd.to_datetime(daily['date'])

    return daily[['date', 'open', 'high', 'low', 'close', 'volume']].copy()


def generate_5min_bars(daily_df, rng_seed=42):
    """Generate synthetic 5-min bars from daily OHLCV using GBM."""
    rng = np.random.RandomState(rng_seed)
    all_bars = []

    for _, day in daily_df.iterrows():
        dt = day['date']
        o, h, l, c = day['open'], day['high'], day['low'], day['close']
        vol = day['volume']
        n_bars = 75

        # GBM path from open to close
        returns = rng.normal(0, 0.001, n_bars)
        drift = (c / o - 1) / n_bars
        returns += drift
        prices = o * np.cumprod(1 + returns)
        prices[-1] = c

        # Scale to fit high/low
        raw_max = prices.max()
        raw_min = prices.min()
        raw_range = raw_max - raw_min
        if raw_range > 0:
            prices = l + (prices - raw_min) / raw_range * (h - l)

        # Volume distribution
        vol_bars = (vol * VOLUME_PROFILE_75).astype(int)
        vol_bars = np.maximum(vol_bars, 100)

        times = pd.date_range(f"{dt.date()} 09:15", periods=n_bars, freq='5min')

        for i in range(n_bars):
            bar_h = prices[i] * (1 + abs(rng.normal(0, 0.0005)))
            bar_l = prices[i] * (1 - abs(rng.normal(0, 0.0005)))
            bar_o = prices[i - 1] if i > 0 else o
            bar_c = prices[i]

            all_bars.append({
                'datetime': times[i],
                'open': round(bar_o, 2),
                'high': round(max(bar_h, bar_o, bar_c), 2),
                'low': round(min(bar_l, bar_o, bar_c), 2),
                'close': round(bar_c, 2),
                'volume': int(vol_bars[i]),
            })

    return pd.DataFrame(all_bars)


def add_indicators(df):
    """Add all indicators needed for L9 signals."""
    df = df.copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    close = df['close']

    # Session grouping
    df['_date'] = df['datetime'].dt.date

    # VWAP
    df['_typical'] = (df['high'] + df['low'] + df['close']) / 3
    df['_tpv'] = df['_typical'] * df['volume']
    df['_cum_tpv'] = df.groupby('_date')['_tpv'].cumsum()
    df['_cum_vol'] = df.groupby('_date')['volume'].cumsum()
    df['vwap'] = df['_cum_tpv'] / df['_cum_vol'].replace(0, np.nan)

    # Opening range (first 3 bars = 15 min)
    df['_bar_num'] = df.groupby('_date').cumcount()
    first_3 = df[df['_bar_num'] < 3].groupby('_date').agg(
        opening_range_high=('high', 'max'),
        opening_range_low=('low', 'min'),
    )
    df = df.merge(first_3, left_on='_date', right_index=True, how='left')

    # Session features
    session_open = df.groupby('_date')['open'].first()
    session_prev_close = df.groupby('_date')['close'].last().shift(1)
    gap = (session_open - session_prev_close) / session_prev_close.replace(0, np.nan)
    gap_map = gap.to_dict()
    df['overnight_gap_pct'] = df['_date'].map(gap_map)

    # Prior day close
    daily_close = df.groupby('_date')['close'].last()
    prev_day_close_map = daily_close.shift(1).to_dict()
    df['prev_day_close'] = df['_date'].map(prev_day_close_map)

    # Technicals
    df['ema_20'] = ema(close, 20)
    df['sma_20'] = sma(close, 20)

    # ATR
    if 'high' in df.columns and 'low' in df.columns:
        df['atr_14'] = atr(df, 14)
        df['adx_14'] = adx(df)

    # Bollinger Bands
    bb = bollinger_bands(close, 20)
    for col in bb.columns:
        df[col] = bb[col].values

    # Volume ratio
    vol_avg = df['volume'].rolling(20, min_periods=5).mean()
    df['vol_ratio_20'] = df['volume'] / vol_avg.replace(0, np.nan)

    # Bar properties
    df['body'] = df['close'] - df['open']
    df['body_pct'] = abs(df['body']) / (df['high'] - df['low']).replace(0, np.nan)

    # Cleanup
    drop_cols = [c for c in df.columns if c.startswith('_')]
    df = df.drop(columns=drop_cols)

    return df


# ════════════════════════════════════════════════════════════════
# L9 SIGNAL CHECKS (3 candidates)
# ════════════════════════════════════════════════════════════════

def check_orb_breakout(bar, prev_bar, session_bars):
    """L9 ORB breakout with volume confirmation."""
    if pd.isna(bar.get('opening_range_high')) or pd.isna(bar.get('vol_ratio_20')):
        return None
    close = float(bar['close'])
    or_high = float(bar['opening_range_high'])
    or_low = float(bar['opening_range_low'])
    vol_ratio = float(bar['vol_ratio_20'])
    prev_close = float(prev_bar['close']) if prev_bar is not None else close

    if vol_ratio < 1.3:
        return None

    if close > or_high and prev_close <= or_high:
        return {'signal_id': 'L9_ORB_BREAKOUT', 'direction': 'LONG', 'price': close,
                'reason': f'ORB long: close={close:.0f} > OR_high={or_high:.0f}'}
    if close < or_low and prev_close >= or_low:
        return {'signal_id': 'L9_ORB_BREAKOUT', 'direction': 'SHORT', 'price': close,
                'reason': f'ORB short: close={close:.0f} < OR_low={or_low:.0f}'}
    return None


def check_gap_fill(bar, prev_bar, session_bars):
    """L9 Gap fill — fade gaps > 0.3%."""
    if pd.isna(bar.get('overnight_gap_pct')):
        return None
    gap = float(bar['overnight_gap_pct'])
    close = float(bar['close'])
    prev_day_close = float(bar['prev_day_close']) if pd.notna(bar.get('prev_day_close')) else None

    if prev_day_close is None or abs(gap) < 0.003:
        return None

    if gap > 0.003 and close > prev_day_close:
        return {'signal_id': 'L9_GAP_FILL', 'direction': 'SHORT', 'price': close,
                'reason': f'Gap fill short: gap={gap:.2%}'}
    if gap < -0.003 and close < prev_day_close:
        return {'signal_id': 'L9_GAP_FILL', 'direction': 'LONG', 'price': close,
                'reason': f'Gap fill long: gap={gap:.2%}'}
    return None


def check_vwap_reclaim(bar, prev_bar, session_bars):
    """L9 VWAP reclaim — cross above VWAP from below."""
    if pd.isna(bar.get('vwap')) or prev_bar is None:
        return None
    close = float(bar['close'])
    vwap = float(bar['vwap'])
    prev_close = float(prev_bar['close'])
    prev_vwap = float(prev_bar['vwap']) if pd.notna(prev_bar.get('vwap')) else vwap

    if prev_close < prev_vwap and close > vwap:
        return {'signal_id': 'L9_VWAP_RECLAIM', 'direction': 'LONG', 'price': close,
                'reason': f'VWAP reclaim: {close:.0f} > vwap={vwap:.0f}'}
    return None


# Also test EOD_TREND and TREND_BAR as bonus
def check_eod_trend(bar, prev_bar, session_bars):
    """L9 EOD trend resumption with volume."""
    if len(session_bars) < 30 or pd.isna(bar.get('vol_ratio_20')):
        return None
    close = float(bar['close'])
    vol_ratio = float(bar['vol_ratio_20'])
    ema_20 = float(bar['ema_20']) if pd.notna(bar.get('ema_20')) else close

    if vol_ratio < 1.2:
        return None

    if close > ema_20 * 1.001:
        return {'signal_id': 'L9_EOD_TREND', 'direction': 'LONG', 'price': close,
                'reason': f'EOD trend long: vol_r={vol_ratio:.1f}'}
    elif close < ema_20 * 0.999:
        return {'signal_id': 'L9_EOD_TREND', 'direction': 'SHORT', 'price': close,
                'reason': f'EOD trend short: vol_r={vol_ratio:.1f}'}
    return None


def check_trend_bar(bar, prev_bar, session_bars):
    """L9 Trend bar after dojis."""
    if len(session_bars) < 5:
        return None
    body_pct = float(bar['body_pct']) if pd.notna(bar.get('body_pct')) else 0
    if body_pct < 0.7:
        return None

    recent = session_bars.tail(4).head(3)
    if len(recent) < 3:
        return None
    small = sum(1 for _, r in recent.iterrows()
                if pd.notna(r.get('body_pct')) and float(r['body_pct']) < 0.3)
    if small < 3:
        return None

    body = float(bar['close']) - float(bar['open'])
    d = 'LONG' if body > 0 else 'SHORT'
    return {'signal_id': 'L9_TREND_BAR', 'direction': d, 'price': float(bar['close']),
            'reason': f'Trend bar {d.lower()}: body={body_pct:.0%} after {small} dojis'}


L9_SIGNALS = {
    'L9_ORB_BREAKOUT': {
        'check': check_orb_breakout,
        'entry_start': '09:30', 'entry_end': '14:00',
        'stop_loss_pct': 0.004, 'max_hold_bars': 40,
    },
    'L9_GAP_FILL': {
        'check': check_gap_fill,
        'entry_start': '09:20', 'entry_end': '11:00',
        'stop_loss_pct': 0.003, 'max_hold_bars': 20,
    },
    'L9_VWAP_RECLAIM': {
        'check': check_vwap_reclaim,
        'entry_start': '09:45', 'entry_end': '14:30',
        'stop_loss_pct': 0.003, 'max_hold_bars': 30,
    },
    'L9_EOD_TREND': {
        'check': check_eod_trend,
        'entry_start': '14:00', 'entry_end': '15:10',
        'stop_loss_pct': 0.003, 'max_hold_bars': 15,
    },
    'L9_TREND_BAR': {
        'check': check_trend_bar,
        'entry_start': '09:30', 'entry_end': '14:30',
        'stop_loss_pct': 0.004, 'max_hold_bars': 30,
    },
}


# ════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ════════════════════════════════════════════════════════════════

def backtest_signal(df_5min, signal_config, signal_check_fn, start_date, end_date):
    """Run a single signal on 5-min bars within date range."""
    mask = (df_5min['datetime'].dt.date >= start_date) & (df_5min['datetime'].dt.date <= end_date)
    df = df_5min[mask].copy()

    if len(df) == 0:
        return {'trades': 0}

    trades = []
    position = None
    bars_held = 0

    dates = sorted(df['datetime'].dt.date.unique())

    for dt in dates:
        day_bars = df[df['datetime'].dt.date == dt].reset_index(drop=True)
        session_bars_accum = []

        for i in range(len(day_bars)):
            bar = day_bars.iloc[i]
            prev_bar = day_bars.iloc[i - 1] if i > 0 else None
            session_bars_accum.append(bar)
            session_df = pd.DataFrame(session_bars_accum)

            bar_time = bar['datetime']
            close = float(bar['close'])

            # Check time window
            entry_start = pd.Timestamp(f"{dt} {signal_config['entry_start']}")
            entry_end = pd.Timestamp(f"{dt} {signal_config['entry_end']}")

            # Force exit at 15:20
            if bar_time.hour == SESSION_CLOSE_HOUR and bar_time.minute >= SESSION_CLOSE_MIN:
                if position:
                    exit_price = close * (1 - SLIPPAGE_PCT if position['direction'] == 'LONG' else 1 + SLIPPAGE_PCT)
                    pnl = (exit_price - position['entry_price']) if position['direction'] == 'LONG' else (position['entry_price'] - exit_price)
                    trades.append({
                        'entry_date': str(position['entry_date']),
                        'exit_date': str(dt),
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': round(exit_price, 2),
                        'pnl_pts': round(pnl, 2),
                        'exit_reason': 'EOD',
                    })
                    position = None
                    bars_held = 0
                continue

            # Check stop loss / max hold
            if position:
                bars_held += 1
                sl_price = position['entry_price'] * (
                    1 - signal_config['stop_loss_pct'] if position['direction'] == 'LONG'
                    else 1 + signal_config['stop_loss_pct']
                )
                sl_hit = (position['direction'] == 'LONG' and close <= sl_price) or \
                         (position['direction'] == 'SHORT' and close >= sl_price)
                max_hold = bars_held >= signal_config['max_hold_bars']

                if sl_hit or max_hold:
                    slip = SLIPPAGE_PCT if position['direction'] == 'LONG' else -SLIPPAGE_PCT
                    exit_price = close * (1 - slip)
                    pnl = (exit_price - position['entry_price']) if position['direction'] == 'LONG' else (position['entry_price'] - exit_price)
                    trades.append({
                        'entry_date': str(position['entry_date']),
                        'exit_date': str(dt),
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': round(exit_price, 2),
                        'pnl_pts': round(pnl, 2),
                        'exit_reason': 'SL' if sl_hit else 'MAX_HOLD',
                    })
                    position = None
                    bars_held = 0

            # Signal check (only if no position)
            if position is None and entry_start <= bar_time <= entry_end:
                result = signal_check_fn(bar, prev_bar, session_df)
                if result:
                    entry_price = close * (1 + SLIPPAGE_PCT if result['direction'] == 'LONG' else 1 - SLIPPAGE_PCT)
                    position = {
                        'direction': result['direction'],
                        'entry_price': round(entry_price, 2),
                        'entry_date': dt,
                    }
                    bars_held = 0

    # Compute metrics
    if not trades:
        return {'trades': 0}

    pnls = [t['pnl_pts'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    n_trades = len(trades)
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 99

    # Daily P&L for Sharpe
    daily_pnl = {}
    for t in trades:
        d = t['exit_date']
        daily_pnl[d] = daily_pnl.get(d, 0) + t['pnl_pts']

    daily_vals = list(daily_pnl.values())
    sharpe = (np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(250)) if len(daily_vals) > 1 and np.std(daily_vals) > 0 else 0

    return {
        'trades': n_trades,
        'total_pnl': round(total_pnl, 2),
        'win_rate': round(win_rate, 4),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(pf, 2),
        'sharpe': round(sharpe, 2),
    }


def walk_forward(df_5min, signal_id, signal_config, signal_check_fn):
    """Run walk-forward validation for a single L9 signal."""
    dates = sorted(df_5min['datetime'].dt.date.unique())
    total_days = len(dates)

    print(f"\n{'─' * 70}")
    print(f"  {signal_id}")
    print(f"  {total_days} trading days, {dates[0]} → {dates[-1]}")
    print(f"{'─' * 70}")

    train_days = WF_TRAIN_MONTHS * 21
    test_days = WF_TEST_MONTHS * 21
    step_days = WF_STEP_MONTHS * 21

    windows = []
    start_idx = 0

    while start_idx + train_days + test_days <= total_days:
        train_start = dates[start_idx]
        train_end = dates[start_idx + train_days - 1]
        test_start = dates[start_idx + train_days]
        test_end_idx = min(start_idx + train_days + test_days - 1, total_days - 1)
        test_end = dates[test_end_idx]

        # Train metrics
        train_m = backtest_signal(df_5min, signal_config, signal_check_fn, train_start, train_end)

        # Test metrics
        test_m = backtest_signal(df_5min, signal_config, signal_check_fn, test_start, test_end)

        passed = (
            test_m['trades'] >= MIN_TRADES and
            test_m.get('sharpe', 0) >= MIN_SHARPE and
            test_m.get('win_rate', 0) >= MIN_WIN_RATE and
            test_m.get('profit_factor', 0) >= MIN_PROFIT_FACTOR
        )

        windows.append({
            'train': f"{train_start}→{train_end}",
            'test': f"{test_start}→{test_end}",
            'train_trades': train_m['trades'],
            'test_trades': test_m['trades'],
            'test_sharpe': test_m.get('sharpe', 0),
            'test_wr': test_m.get('win_rate', 0),
            'test_pf': test_m.get('profit_factor', 0),
            'test_pnl': test_m.get('total_pnl', 0),
            'passed': passed,
        })

        tag = "PASS" if passed else "FAIL"
        print(f"  Window {len(windows):2d}: test {test_start}→{test_end} "
              f"| {test_m['trades']:3d} trades | Sharpe {test_m.get('sharpe', 0):5.2f} "
              f"| WR {test_m.get('win_rate', 0):.0%} | PF {test_m.get('profit_factor', 0):5.2f} "
              f"| [{tag}]")

        start_idx += step_days

    n_pass = sum(1 for w in windows if w['passed'])
    n_total = len(windows)
    pass_rate = n_pass / n_total if n_total > 0 else 0
    overall = "PASS" if pass_rate >= WF_PASS_THRESHOLD else "FAIL"

    print(f"\n  RESULT: {n_pass}/{n_total} windows passed ({pass_rate:.0%}) → {overall}")

    return {
        'signal_id': signal_id,
        'windows': windows,
        'n_pass': n_pass,
        'n_total': n_total,
        'pass_rate': round(pass_rate, 4),
        'overall': overall,
    }


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  L9 INTRADAY WALK-FORWARD VALIDATION")
    print("  Testing: ORB_BREAKOUT, GAP_FILL, VWAP_RECLAIM, EOD_TREND, TREND_BAR")
    print("=" * 70)

    # Load data
    print("\nLoading daily OHLCV from trade logs...")
    daily = load_daily_from_trade_logs()
    print(f"  {len(daily)} daily bars: {daily['date'].iloc[0].date()} → {daily['date'].iloc[-1].date()}")

    print("\nGenerating synthetic 5-min bars...")
    df_5min = generate_5min_bars(daily)
    print(f"  {len(df_5min)} 5-min bars")

    print("\nComputing indicators...")
    df_5min = add_indicators(df_5min)
    print(f"  Indicators added: {[c for c in df_5min.columns if c not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]}")

    # Run walk-forward for each signal
    results = {}
    for sig_id, sig_config in L9_SIGNALS.items():
        wf_result = walk_forward(df_5min, sig_id, sig_config, sig_config['check'])
        results[sig_id] = wf_result

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Signal':<25s} {'Pass Rate':>10s} {'Windows':>10s} {'Result':>8s}")
    print(f"  {'─' * 55}")

    passed_signals = []
    for sig_id, r in results.items():
        print(f"  {sig_id:<25s} {r['pass_rate']:>9.0%} {r['n_pass']}/{r['n_total']:>7s} {r['overall']:>8s}")
        if r['overall'] == 'PASS':
            passed_signals.append(sig_id)

    if passed_signals:
        print(f"\n  PASSED L9 signals ready for deployment: {passed_signals}")
        print(f"  These can be added to the options execution pipeline alongside")
        print(f"  KAUFMAN_BB_MR and GUJRAL_RANGE for additional trade frequency.")
    else:
        print(f"\n  No L9 signals passed walk-forward at current thresholds.")
        print(f"  Consider: relaxing criteria or testing with real tick data.")

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), 'backtest_results', 'intraday')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'l9_wf_results.json')

    # Convert for JSON serialization
    json_results = {}
    for sig_id, r in results.items():
        json_results[sig_id] = {
            'pass_rate': r['pass_rate'],
            'n_pass': r['n_pass'],
            'n_total': r['n_total'],
            'overall': r['overall'],
        }

    with open(out_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    return results


if __name__ == '__main__':
    main()

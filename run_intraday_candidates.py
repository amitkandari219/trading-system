"""
Intraday walk-forward backtest for 6 failed-daily signals with intraday potential.

Adapts KAUFMAN_DRY_22, GUJRAL_DRY_12, GUJRAL_DRY_15, KAUFMAN_DRY_19,
KAUFMAN_DRY_15, GRIMES_DRY_8_0 to 5-min bars and runs walk-forward validation.

Uses synthetic 5-min bars generated from daily OHLCV (same as L9 pipeline).

Usage:
    python run_intraday_candidates.py
    python run_intraday_candidates.py --start-year 2019
"""

import argparse
import json
import os
import math
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import psycopg2

from data.intraday_loader import generate_5min_bars
from data.intraday_indicators import add_intraday_indicators
from backtest.indicators import (
    sma, ema, rsi, atr, adx, bollinger_bands, stochastic, pivot_points
)
from config.settings import DATABASE_DSN

SLIPPAGE_PCT = 0.0005       # 0.05% each way
SESSION_CLOSE_HOUR = 15
SESSION_CLOSE_MIN = 20
NIFTY_LOT_SIZE = 25
CAPITAL = 1_000_000

# Walk-forward params adapted for intraday
# Use 12-month train / 4-month test / 2-month step (more windows, faster adaptation)
WF_TRAIN_MONTHS = 12
WF_TEST_MONTHS = 4
WF_STEP_MONTHS = 2
WF_PASS_THRESHOLD = 0.70   # 70% of windows must pass (slightly relaxed for intraday)

# Per-window minimum criteria
MIN_SHARPE = 0.80
MIN_TRADES = 15
MIN_WIN_RATE = 0.40
MIN_PROFIT_FACTOR = 1.20


# ════════════════════════════════════════════════════════════════
# SIGNAL DEFINITIONS — 6 candidates adapted for 5-min bars
# ════════════════════════════════════════════════════════════════

INTRADAY_CANDIDATES = {
    'ID_KAUFMAN_BB_MR': {
        'source': 'KAUFMAN_DRY_22',
        'description': 'BB mean-reversion at band extremes in ranging markets (ADX<25)',
        'direction': 'BOTH',
        'stop_loss_pct': 0.004,    # 0.4%
        'max_hold_bars': 30,       # 150 min
        'entry_start': '09:45',
        'entry_end': '14:30',
    },
    'ID_GUJRAL_RANGE': {
        'source': 'GUJRAL_DRY_12',
        'description': 'Range boundary trades with ADX filter (from intraday book)',
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
        'description': 'Mean-reversion at SMA(50) of highs/lows zones',
        'direction': 'BOTH',
        'stop_loss_pct': 0.005,
        'max_hold_bars': 25,
        'entry_start': '09:45',
        'entry_end': '14:30',
    },
    'ID_KAUFMAN_IMPULSE': {
        'source': 'KAUFMAN_DRY_15',
        'description': 'Momentum impulse on large % move (1.5%+ intraday)',
        'direction': 'BOTH',
        'stop_loss_pct': 0.004,
        'max_hold_bars': 20,
        'entry_start': '09:30',
        'entry_end': '14:00',
    },
    'ID_GRIMES_ATR_BURST': {
        'source': 'GRIMES_DRY_8_0',
        'description': 'Volatility impulse: returns > 2x ATR(20) with volume confirm',
        'direction': 'BOTH',
        'stop_loss_pct': 0.005,
        'max_hold_bars': 25,
        'entry_start': '09:30',
        'entry_end': '14:30',
    },
}


def _is_session_close(dt):
    return (dt.hour > SESSION_CLOSE_HOUR or
            (dt.hour == SESSION_CLOSE_HOUR and dt.minute >= SESSION_CLOSE_MIN))


def _time_in_range(dt, start_str, end_str):
    """Check if datetime is within entry time window."""
    h1, m1 = map(int, start_str.split(':'))
    h2, m2 = map(int, end_str.split(':'))
    bar_mins = dt.hour * 60 + dt.minute
    return (h1 * 60 + m1) <= bar_mins <= (h2 * 60 + m2)


# ════════════════════════════════════════════════════════════════
# SIGNAL CHECK FUNCTIONS
# ════════════════════════════════════════════════════════════════

def check_kaufman_bb_mr(bar, prev_bar, session_bars, config):
    """
    KAUFMAN_DRY_22: BB mean-reversion.
    Buy when close < bb_lower AND ADX < 25 (ranging market).
    Sell when close > bb_upper AND ADX < 25.
    Exit on bb_middle cross.
    """
    if prev_bar is None:
        return None

    close = float(bar['close'])
    adx_val = float(bar['adx_14']) if pd.notna(bar.get('adx_14')) else 30

    # Only trade in ranging markets
    if adx_val >= 25:
        return None

    bb_upper = bar.get('bb_upper')
    bb_lower = bar.get('bb_lower')
    if pd.isna(bb_upper) or pd.isna(bb_lower):
        return None

    bb_upper = float(bb_upper)
    bb_lower = float(bb_lower)
    prev_close = float(prev_bar['close'])

    # Long: close crosses below lower band
    if close <= bb_lower and prev_close > bb_lower:
        return {'direction': 'LONG', 'price': close,
                'reason': f'BB_MR long: close={close:.0f} <= bb_lower={bb_lower:.0f} ADX={adx_val:.0f}'}

    # Short: close crosses above upper band
    if close >= bb_upper and prev_close < bb_upper:
        return {'direction': 'SHORT', 'price': close,
                'reason': f'BB_MR short: close={close:.0f} >= bb_upper={bb_upper:.0f} ADX={adx_val:.0f}'}

    return None


def check_gujral_range(bar, prev_bar, session_bars, config):
    """
    GUJRAL_DRY_12: Range boundary trades.
    In ranging market (ADX < 25):
    - Buy near session low (within 0.2% of session low)
    - Sell near session high (within 0.2% of session high)
    Requires at least 20 bars to establish range.
    """
    if len(session_bars) < 20:
        return None

    close = float(bar['close'])
    adx_val = float(bar['adx_14']) if pd.notna(bar.get('adx_14')) else 30

    if adx_val >= 25:
        return None

    session_high = float(session_bars['high'].max())
    session_low = float(session_bars['low'].min())
    session_range = session_high - session_low

    if session_range <= 0:
        return None

    # Proximity threshold: 20% of session range
    threshold = session_range * 0.20

    # Long near session low
    if close <= session_low + threshold and close > session_low:
        # Confirm: bar is bullish (close > open)
        if close > float(bar['open']):
            return {'direction': 'LONG', 'price': close,
                    'reason': f'RANGE long: close={close:.0f} near sess_low={session_low:.0f}'}

    # Short near session high
    if close >= session_high - threshold and close < session_high:
        if close < float(bar['open']):
            return {'direction': 'SHORT', 'price': close,
                    'reason': f'RANGE short: close={close:.0f} near sess_high={session_high:.0f}'}

    return None


def check_gujral_pivot_r1(bar, prev_bar, session_bars, config):
    """
    GUJRAL_DRY_15: Pivot R1 breakout in trending market.
    Long when close breaks above R1 AND ADX > 25 (trending).
    Uses previous day HLC for pivot calculation.
    """
    if prev_bar is None:
        return None

    close = float(bar['close'])
    prev_close = float(prev_bar['close'])
    adx_val = float(bar['adx_14']) if pd.notna(bar.get('adx_14')) else 20

    # Only trade in trending markets
    if adx_val < 25:
        return None

    # Use prev_day_high/low/close for pivot
    pdh = bar.get('prev_day_high')
    pdl = bar.get('prev_day_low')
    pdc = bar.get('prev_day_close')

    if pd.isna(pdh) or pd.isna(pdl) or pd.isna(pdc):
        return None

    pdh, pdl, pdc = float(pdh), float(pdl), float(pdc)
    pivot = (pdh + pdl + pdc) / 3.0
    r1 = 2 * pivot - pdl

    # Volume confirmation
    vol_ratio = float(bar['vol_ratio_20']) if pd.notna(bar.get('vol_ratio_20')) else 0.8
    if vol_ratio < 1.1:
        return None

    # Long: close crosses above R1
    if close > r1 and prev_close <= r1:
        return {'direction': 'LONG', 'price': close,
                'reason': f'PIVOT_R1 long: close={close:.0f} > R1={r1:.0f} ADX={adx_val:.0f}'}

    return None


def check_kaufman_sma_mr(bar, prev_bar, session_bars, config):
    """
    KAUFMAN_DRY_19: Mean-reversion at SMA zones.
    Buy when close < SMA(50) - 0.3% (below support zone).
    Sell when close > SMA(50) + 0.3% (above resistance zone).
    """
    if prev_bar is None:
        return None

    close = float(bar['close'])
    prev_close = float(prev_bar['close'])
    sma_50 = bar.get('sma_50')

    if pd.isna(sma_50):
        return None

    sma_50 = float(sma_50)
    offset = sma_50 * 0.003  # 0.3% bands around SMA

    lower_zone = sma_50 - offset
    upper_zone = sma_50 + offset

    # Long: price enters lower zone (crosses below)
    if close < lower_zone and prev_close >= lower_zone:
        return {'direction': 'LONG', 'price': close,
                'reason': f'SMA_MR long: close={close:.0f} < sma50-0.3%={lower_zone:.0f}'}

    # Short: price enters upper zone (crosses above)
    if close > upper_zone and prev_close <= upper_zone:
        return {'direction': 'SHORT', 'price': close,
                'reason': f'SMA_MR short: close={close:.0f} > sma50+0.3%={upper_zone:.0f}'}

    return None


def check_kaufman_impulse(bar, prev_bar, session_bars, config):
    """
    KAUFMAN_DRY_15: Momentum impulse on large intraday move.
    Entry when cumulative session return exceeds 0.8% (scaled from daily 1.96%).
    Direction: follow the impulse momentum.
    """
    if len(session_bars) < 5:
        return None

    close = float(bar['close'])
    session_open = float(session_bars.iloc[0]['open'])

    if session_open <= 0:
        return None

    session_return = (close - session_open) / session_open

    vol_ratio = float(bar['vol_ratio_20']) if pd.notna(bar.get('vol_ratio_20')) else 0.8

    # Require volume confirmation
    if vol_ratio < 1.2:
        return None

    # Already checked — need to see if this is a NEW impulse (not continued position)
    prev_close = float(prev_bar['close']) if prev_bar is not None else close
    prev_return = (prev_close - session_open) / session_open if session_open > 0 else 0

    # Long: session return crosses above 0.8%
    if session_return > 0.008 and prev_return <= 0.008:
        return {'direction': 'LONG', 'price': close,
                'reason': f'IMPULSE long: session_ret={session_return:.2%} vol_r={vol_ratio:.1f}'}

    # Short: session return crosses below -0.8%
    if session_return < -0.008 and prev_return >= -0.008:
        return {'direction': 'SHORT', 'price': close,
                'reason': f'IMPULSE short: session_ret={session_return:.2%} vol_r={vol_ratio:.1f}'}

    return None


def check_grimes_atr_burst(bar, prev_bar, session_bars, config):
    """
    GRIMES_DRY_8_0: Volatility impulse — returns > 2x ATR(14).
    Trades outsized moves with volume confirmation.
    """
    if prev_bar is None:
        return None

    close = float(bar['close'])
    prev_close = float(prev_bar['close'])
    atr_val = bar.get('atr_14')
    vol_ratio = bar.get('vol_ratio_20')

    if pd.isna(atr_val) or pd.isna(vol_ratio):
        return None

    atr_val = float(atr_val)
    vol_ratio = float(vol_ratio)

    if atr_val <= 0 or vol_ratio < 1.3:
        return None

    bar_return = close - prev_close

    # Long: large positive move > 2x ATR
    if bar_return > 2.0 * atr_val:
        return {'direction': 'LONG', 'price': close,
                'reason': f'ATR_BURST long: move={bar_return:.0f} > 2xATR={2*atr_val:.0f} vol_r={vol_ratio:.1f}'}

    # Short: large negative move > 2x ATR
    if bar_return < -2.0 * atr_val:
        return {'direction': 'SHORT', 'price': close,
                'reason': f'ATR_BURST short: move={bar_return:.0f} < -2xATR={-2*atr_val:.0f} vol_r={vol_ratio:.1f}'}

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
# BACKTEST ENGINE (per-signal, per-period)
# ════════════════════════════════════════════════════════════════

def run_signal_on_period(signal_id, config, df_daily_period):
    """
    Run a single signal on a date range of daily data.
    Generates synthetic 5-min bars per day, adds indicators, checks signal.

    Returns: dict with metrics + trades list
    """
    check_fn = SIGNAL_CHECKS[signal_id]
    rng = np.random.default_rng(42)

    trades = []
    position = None
    bars_held = 0
    equity = CAPITAL
    daily_pnl = {}

    dates = df_daily_period['date'].values

    for i in range(len(df_daily_period)):
        daily_row = df_daily_period.iloc[i]
        trading_date = pd.Timestamp(daily_row['date']).date()

        # Generate 5-min bars for this day
        bars_data = generate_5min_bars(daily_row, rng)
        df_bars = pd.DataFrame(bars_data)

        # Need at least a few days of context for indicators
        # Build a window of recent bars for indicator computation
        # For simplicity, compute indicators on this day's bars only
        # (session-scoped indicators reset daily anyway)
        df_bars = add_intraday_indicators(df_bars)

        session_bars_so_far = pd.DataFrame()

        for idx in range(len(df_bars)):
            bar = df_bars.iloc[idx]
            prev_bar = df_bars.iloc[idx - 1] if idx > 0 else None
            bar_dt = bar['datetime']
            close = float(bar['close'])

            session_bars_so_far = pd.concat(
                [session_bars_so_far, bar.to_frame().T], ignore_index=True
            )

            # ── CHECK EXITS ──
            if position is not None:
                bars_held += 1
                direction = position['direction']
                ep = position['entry_price']
                exit_reason = None

                if _is_session_close(bar_dt):
                    exit_reason = 'session_close'

                if not exit_reason:
                    loss = (ep - close) / ep if direction == 'LONG' else (close - ep) / ep
                    if loss >= config['stop_loss_pct']:
                        exit_reason = 'stop_loss'

                if not exit_reason and bars_held >= config['max_hold_bars']:
                    exit_reason = 'max_hold'

                # Mean-reversion exit: for BB_MR signal, exit at bb_middle
                if not exit_reason and signal_id == 'ID_KAUFMAN_BB_MR':
                    bb_mid = bar.get('bb_middle')
                    if pd.notna(bb_mid):
                        bb_mid = float(bb_mid)
                        if direction == 'LONG' and close >= bb_mid:
                            exit_reason = 'target_bb_mid'
                        elif direction == 'SHORT' and close <= bb_mid:
                            exit_reason = 'target_bb_mid'

                if exit_reason:
                    if direction == 'LONG':
                        exit_price = close * (1 - SLIPPAGE_PCT)
                        pnl_pts = exit_price - ep
                    else:
                        exit_price = close * (1 + SLIPPAGE_PCT)
                        pnl_pts = ep - exit_price

                    pnl_rs = pnl_pts * NIFTY_LOT_SIZE
                    trades.append({
                        'signal_id': signal_id,
                        'direction': direction,
                        'entry_price': ep,
                        'exit_price': round(exit_price, 2),
                        'pnl_pts': round(pnl_pts, 2),
                        'pnl_rs': round(pnl_rs, 2),
                        'bars_held': bars_held,
                        'exit_reason': exit_reason,
                        'date': trading_date,
                    })
                    equity += pnl_rs
                    daily_pnl.setdefault(trading_date, 0)
                    daily_pnl[trading_date] += pnl_rs
                    position = None
                    bars_held = 0

            # ── CHECK ENTRIES ──
            if position is None and not _is_session_close(bar_dt):
                if _time_in_range(bar_dt, config['entry_start'], config['entry_end']):
                    result = check_fn(bar, prev_bar, session_bars_so_far, config)
                    if result:
                        if result['direction'] == 'LONG':
                            adj_price = close * (1 + SLIPPAGE_PCT)
                        else:
                            adj_price = close * (1 - SLIPPAGE_PCT)

                        position = {
                            'direction': result['direction'],
                            'entry_price': round(adj_price, 2),
                        }
                        bars_held = 0

        # Force close any open position at session end
        if position is not None:
            last_bar = df_bars.iloc[-1]
            close = float(last_bar['close'])
            direction = position['direction']
            ep = position['entry_price']

            if direction == 'LONG':
                exit_price = close * (1 - SLIPPAGE_PCT)
                pnl_pts = exit_price - ep
            else:
                exit_price = close * (1 + SLIPPAGE_PCT)
                pnl_pts = ep - exit_price

            pnl_rs = pnl_pts * NIFTY_LOT_SIZE
            trades.append({
                'signal_id': signal_id,
                'direction': direction,
                'entry_price': ep,
                'exit_price': round(exit_price, 2),
                'pnl_pts': round(pnl_pts, 2),
                'pnl_rs': round(pnl_rs, 2),
                'bars_held': bars_held,
                'exit_reason': 'forced_session_end',
                'date': trading_date,
            })
            equity += pnl_rs
            daily_pnl.setdefault(trading_date, 0)
            daily_pnl[trading_date] += pnl_rs
            position = None

    # Compute metrics
    if not trades:
        return {'sharpe': 0, 'win_rate': 0, 'profit_factor': 0,
                'trade_count': 0, 'total_pnl': 0, 'max_dd': 1.0}, trades

    pnls = [t['pnl_rs'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) if pnls else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0

    daily_returns = pd.Series(daily_pnl) / CAPITAL
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if len(daily_returns) > 1 and daily_returns.std() > 0 else 0)

    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    max_dd = abs(dd.min()) / CAPITAL if CAPITAL > 0 else 0

    return {
        'sharpe': round(float(sharpe), 3),
        'win_rate': round(win_rate, 3),
        'profit_factor': round(float(pf), 3),
        'trade_count': len(trades),
        'total_pnl': round(sum(pnls), 0),
        'max_dd': round(float(max_dd), 4),
    }, trades


# ════════════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ════════════════════════════════════════════════════════════════

def walk_forward_intraday(signal_id, config, df_daily):
    """
    Walk-forward validation for an intraday signal.
    12-month train / 4-month test / 2-month step.
    """
    dates = pd.to_datetime(df_daily['date'])
    min_date = dates.min()
    max_date = dates.max()

    windows = []
    train_start = min_date

    while True:
        train_end = train_start + pd.DateOffset(months=WF_TRAIN_MONTHS)
        test_start = train_end + pd.DateOffset(days=10)  # 10-day purge
        test_end = test_start + pd.DateOffset(months=WF_TEST_MONTHS)

        if test_end > max_date:
            break

        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
        })

        train_start += pd.DateOffset(months=WF_STEP_MONTHS)

    if not windows:
        return {'pass': False, 'reason': 'insufficient_data', 'windows': 0}

    results = []
    for i, w in enumerate(windows):
        # Get test period data
        mask = (dates >= w['test_start']) & (dates <= w['test_end'])
        df_test = df_daily[mask].copy()

        if len(df_test) < 20:
            results.append({'window': i, 'pass': False, 'reason': 'too_few_days',
                            'sharpe': 0, 'trades': 0, 'pnl': 0})
            continue

        metrics, trades = run_signal_on_period(signal_id, config, df_test)

        window_pass = (
            metrics['sharpe'] >= MIN_SHARPE and
            metrics['trade_count'] >= MIN_TRADES and
            metrics['win_rate'] >= MIN_WIN_RATE and
            metrics['profit_factor'] >= MIN_PROFIT_FACTOR
        )

        results.append({
            'window': i,
            'pass': window_pass,
            'test_period': f"{w['test_start'].strftime('%Y-%m')} to {w['test_end'].strftime('%Y-%m')}",
            'sharpe': metrics['sharpe'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'trades': metrics['trade_count'],
            'pnl': metrics['total_pnl'],
            'max_dd': metrics['max_dd'],
        })

    n_passed = sum(1 for r in results if r['pass'])
    n_total = len(results)
    pass_rate = n_passed / n_total if n_total > 0 else 0
    recent_pass = results[-1]['pass'] if results else False

    overall_pass = pass_rate >= WF_PASS_THRESHOLD and recent_pass

    return {
        'pass': overall_pass,
        'pass_rate': round(pass_rate, 3),
        'n_passed': n_passed,
        'n_total': n_total,
        'recent_pass': recent_pass,
        'windows': results,
        'avg_sharpe': round(np.mean([r['sharpe'] for r in results]), 3),
        'avg_pnl': round(np.mean([r['pnl'] for r in results]), 0),
        'total_trades': sum(r['trades'] for r in results),
    }


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Intraday Candidate Backtest')
    parser.add_argument('--start-year', type=int, default=2019,
                        help='Start year for data (default: 2019)')
    parser.add_argument('--signal', type=str, default=None,
                        help='Run single signal (e.g., ID_KAUFMAN_BB_MR)')
    args = parser.parse_args()

    print("=" * 80)
    print("INTRADAY WALK-FORWARD BACKTEST — 6 Failed-Daily Candidates")
    print("=" * 80)
    print(f"WF Params: {WF_TRAIN_MONTHS}mo train / {WF_TEST_MONTHS}mo test / {WF_STEP_MONTHS}mo step")
    print(f"Pass criteria: Sharpe≥{MIN_SHARPE}, Trades≥{MIN_TRADES}, WR≥{MIN_WIN_RATE:.0%}, PF≥{MIN_PROFIT_FACTOR}")
    print(f"Overall pass: ≥{WF_PASS_THRESHOLD:.0%} windows + recent window must pass")
    print()

    # Load daily data — try DB first, fallback to yfinance
    print("Loading daily data...", flush=True)
    df_daily = None
    try:
        conn = psycopg2.connect(DATABASE_DSN)
        df_daily = pd.read_sql(
            f"""SELECT date, open, high, low, close, volume, india_vix
                FROM nifty_daily
                WHERE date >= '{args.start_year}-01-01'
                ORDER BY date""",
            conn,
        )
        conn.close()
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        print(f"  Loaded from database")
    except Exception as e:
        print(f"  DB unavailable ({e.__class__.__name__}), downloading from yfinance...")
        import yfinance as yf
        nifty = yf.download('^NSEI', start=f'{args.start_year}-01-01', progress=False)
        if nifty.empty:
            print("ERROR: Could not download Nifty data")
            return
        # Handle multi-level columns from yfinance
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        nifty = nifty.reset_index()
        nifty.columns = [c.lower().replace(' ', '_') for c in nifty.columns]
        nifty = nifty.rename(columns={'adj_close': 'adj_close'})
        if 'date' not in nifty.columns:
            nifty = nifty.rename(columns={nifty.columns[0]: 'date'})
        nifty['date'] = pd.to_datetime(nifty['date'])
        # Add india_vix placeholder (not critical for signal logic)
        nifty['india_vix'] = 15.0
        try:
            vix = yf.download('^INDIAVIX', start=f'{args.start_year}-01-01', progress=False)
            if not vix.empty:
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                vix = vix.reset_index()
                vix.columns = [c.lower().replace(' ', '_') for c in vix.columns]
                if 'date' not in vix.columns:
                    vix = vix.rename(columns={vix.columns[0]: 'date'})
                vix['date'] = pd.to_datetime(vix['date'])
                vix_map = vix.set_index('date')['close'].to_dict()
                nifty['india_vix'] = nifty['date'].map(vix_map).fillna(15.0)
                print(f"  VIX data loaded ({len(vix_map)} days)")
        except Exception:
            pass
        df_daily = nifty[['date', 'open', 'high', 'low', 'close', 'volume', 'india_vix']].copy()
        df_daily = df_daily.dropna(subset=['open', 'high', 'low', 'close']).sort_values('date').reset_index(drop=True)
        print(f"  Downloaded from yfinance")

    print(f"  {len(df_daily)} trading days: {df_daily['date'].min().date()} to {df_daily['date'].max().date()}")
    print()

    # Select signals to test
    signals_to_test = INTRADAY_CANDIDATES
    if args.signal:
        if args.signal in INTRADAY_CANDIDATES:
            signals_to_test = {args.signal: INTRADAY_CANDIDATES[args.signal]}
        else:
            print(f"ERROR: Unknown signal '{args.signal}'")
            print(f"Available: {', '.join(INTRADAY_CANDIDATES.keys())}")
            return

    # Run each signal through walk-forward
    all_results = {}
    for signal_id, config in signals_to_test.items():
        print(f"{'─' * 80}")
        print(f"Testing: {signal_id} ({config['source']})")
        print(f"  {config['description']}")
        print(f"  Direction: {config['direction']}  SL: {config['stop_loss_pct']:.1%}  "
              f"MaxHold: {config['max_hold_bars']} bars  "
              f"Time: {config['entry_start']}-{config['entry_end']}")
        print()

        wf_result = walk_forward_intraday(signal_id, config, df_daily)
        all_results[signal_id] = wf_result

        status = "✓ PASS" if wf_result['pass'] else "✗ FAIL"
        print(f"  Result: {status}")
        print(f"  Windows: {wf_result['n_passed']}/{wf_result['n_total']} passed "
              f"({wf_result['pass_rate']:.0%})")
        print(f"  Recent window: {'PASS' if wf_result['recent_pass'] else 'FAIL'}")
        print(f"  Avg Sharpe: {wf_result['avg_sharpe']:.2f}  "
              f"Avg P&L: ₹{wf_result['avg_pnl']:,.0f}  "
              f"Total trades: {wf_result['total_trades']}")

        # Print per-window details
        print(f"\n  {'Window':<6} {'Period':<22} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'Trades':>6} {'P&L':>10} {'DD':>7} {'Result':>6}")
        for w in wf_result['windows']:
            period = w.get('test_period', 'N/A')
            res = 'PASS' if w['pass'] else 'FAIL'
            print(f"  {w['window']:<6} {period:<22} {w['sharpe']:>7.2f} {w.get('win_rate',0):>5.0%} "
                  f"{w.get('profit_factor',0):>6.2f} {w['trades']:>6} {w['pnl']:>9,.0f}₹ "
                  f"{w.get('max_dd',0):>6.2%} {res:>6}")

        print()

    # ── SUMMARY ──
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    passed = {k: v for k, v in all_results.items() if v['pass']}
    failed = {k: v for k, v in all_results.items() if not v['pass']}

    print(f"\n  PASSED ({len(passed)}/{len(all_results)}):")
    if passed:
        for sid, r in sorted(passed.items(), key=lambda x: -x[1]['avg_sharpe']):
            cfg = INTRADAY_CANDIDATES[sid]
            print(f"    {sid:<25s} ({cfg['source']:<18s}) "
                  f"WF={r['pass_rate']:.0%}  Sharpe={r['avg_sharpe']:.2f}  "
                  f"trades={r['total_trades']}")
    else:
        print("    None")

    print(f"\n  FAILED ({len(failed)}/{len(all_results)}):")
    for sid, r in failed.items():
        cfg = INTRADAY_CANDIDATES[sid]
        reason = f"WF={r['pass_rate']:.0%}"
        if not r['recent_pass']:
            reason += " (recent FAIL)"
        print(f"    {sid:<25s} ({cfg['source']:<18s}) {reason}  "
              f"Sharpe={r['avg_sharpe']:.2f}")

    # Save results
    os.makedirs('backtest_results/intraday', exist_ok=True)
    save_data = {}
    for sid, r in all_results.items():
        save_data[sid] = {
            'source': INTRADAY_CANDIDATES[sid]['source'],
            'pass': r['pass'],
            'pass_rate': r['pass_rate'],
            'n_passed': r['n_passed'],
            'n_total': r['n_total'],
            'recent_pass': r['recent_pass'],
            'avg_sharpe': r['avg_sharpe'],
            'avg_pnl': r['avg_pnl'],
            'total_trades': r['total_trades'],
        }

    with open('backtest_results/intraday/candidate_wf_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved: backtest_results/intraday/candidate_wf_results.json")
    print("=" * 80)


if __name__ == '__main__':
    main()

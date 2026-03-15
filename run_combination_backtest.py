"""
Backtest signal combinations for the 3 confirmed Kaufman signals.

Tests AND-logic pairs, triple AND, scoring system, and regime-conditional stacking.
Reports 10-year and 6-month results side by side.
"""

import json
import os
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd
import psycopg2

from backtest.indicators import add_all_indicators, historical_volatility, sma, adx
from backtest.types import BacktestResult
from backtest.walk_forward import WalkForwardEngine
from backtest.types import harmonic_mean_sharpe
from config.settings import DATABASE_DSN

RESULTS_DIR = 'backtest_results'


# ================================================================
# SIGNAL CONDITION CHECKERS (same logic as paper_trading/signal_compute.py)
# ================================================================

def check_dry20_long(row, prev_row):
    """sma_10 < prev_close AND stoch_k_5 > 50"""
    if pd.isna(row['sma_10']) or pd.isna(row['stoch_k_5']):
        return False
    return row['sma_10'] < prev_row['close'] and row['stoch_k_5'] > 50

def check_dry20_exit_long(row, prev_row):
    """stoch_k_5 <= 50"""
    if pd.isna(row['stoch_k_5']):
        return False
    return row['stoch_k_5'] <= 50

def check_dry16_long(row, prev_row):
    """close > r1 AND low >= pivot AND hvol_6 < hvol_100"""
    if pd.isna(row.get('hvol_6')) or pd.isna(row.get('hvol_100')):
        return False
    return (row['close'] > row['r1'] and
            row['low'] >= row['pivot'] and
            row['hvol_6'] < row['hvol_100'])

def check_dry16_short(row, prev_row):
    """close < s1 AND hvol_6 < hvol_100"""
    if pd.isna(row.get('hvol_6')) or pd.isna(row.get('hvol_100')):
        return False
    return row['close'] < row['s1'] and row['hvol_6'] < row['hvol_100']

def check_dry16_exit_long(row, prev_row):
    """low < pivot"""
    return row['low'] < row['pivot']

def check_dry16_exit_short(row, prev_row):
    """high > r1"""
    return row['high'] > row['r1']

def check_dry12_long(row, prev_row):
    """close > prev_close AND volume < prev_volume"""
    return row['close'] > prev_row['close'] and row['volume'] < prev_row['volume']

def check_dry12_short(row, prev_row):
    """close < prev_close AND volume < prev_volume"""
    return row['close'] < prev_row['close'] and row['volume'] < prev_row['volume']

def check_dry12_exit_long(row, prev_row):
    """close < prev_close"""
    return row['close'] < prev_row['close']

def check_dry12_exit_short(row, prev_row):
    """close > prev_close"""
    return row['close'] > prev_row['close']


# ================================================================
# GENERIC COMBINATION BACKTESTER
# ================================================================

def run_combination_backtest(df, entry_fn, exit_fn, stop_pct=0.02,
                             tp_pct=0.03, hold_max=0, size_fn=None):
    """
    Run a combination backtest.

    entry_fn(row, prev_row) -> 'LONG', 'SHORT', or None
    exit_fn(row, prev_row, position_dir) -> bool
    size_fn(row, prev_row) -> float (position size multiplier, default 1.0)
    """
    closes = df['close'].values
    dates = df['date'].values if 'date' in df.columns else df.index.values
    n = len(df)

    trades = []
    position = None
    entry_price = 0.0
    entry_idx = 0
    size = 1.0

    for i in range(1, n):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        price = closes[i]

        if position is None:
            direction = entry_fn(row, prev_row)
            if direction:
                position = direction
                entry_price = price
                entry_idx = i
                size = size_fn(row, prev_row) if size_fn else 1.0
                continue
        else:
            days_held = i - entry_idx

            # Stop loss
            if position == 'LONG':
                loss_pct = (entry_price - price) / entry_price
            else:
                loss_pct = (price - entry_price) / entry_price
            if loss_pct >= stop_pct:
                pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
                trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                               'direction': position, 'exit_reason': 'stop_loss',
                               'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                position = None
                continue

            # Take profit
            if tp_pct > 0:
                if position == 'LONG':
                    gain_pct = (price - entry_price) / entry_price
                else:
                    gain_pct = (entry_price - price) / entry_price
                if gain_pct >= tp_pct:
                    pnl = gain_pct * entry_price
                    trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                                   'direction': position, 'exit_reason': 'take_profit',
                                   'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                    position = None
                    continue

            # Hold max
            if hold_max > 0 and days_held >= hold_max:
                pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
                trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                               'direction': position, 'exit_reason': 'hold_max',
                               'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                position = None
                continue

            # Signal exit
            if exit_fn(row, prev_row, position):
                pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
                trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                               'direction': position, 'exit_reason': 'signal',
                               'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                position = None

    # Close open position
    if position is not None:
        price = closes[-1]
        pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
        trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                       'direction': position, 'exit_reason': 'end_of_data',
                       'entry_date': dates[entry_idx], 'exit_date': dates[-1]})

    return compute_metrics(trades, df)


def compute_metrics(trades, df):
    """Compute standard metrics from trade list."""
    if len(trades) < 5:
        return {
            'trades': len(trades), 'sharpe': 0, 'win_rate': 0, 'profit_factor': 0,
            'max_dd': 1.0, 'nifty_corr': 0, 'annual_return': 0,
            'long_trades': 0, 'long_wr': 0, 'short_trades': 0, 'short_wr': 0,
            'pnl_pts': 0, 'insufficient': len(trades) < 20,
        }

    notional = trades[0]['entry_price']
    pnls = [t['pnl'] for t in trades]
    pct_returns = [p / notional for p in pnls]

    wins = [r for r in pct_returns if r > 0]
    losses = [r for r in pct_returns if r < 0]
    win_rate = len(wins) / len(pct_returns)

    ret_series = pd.Series(pct_returns)
    std = ret_series.std()
    sharpe = (ret_series.mean() / std * np.sqrt(252)) if std > 0 else 0

    equity = [1.0]
    for r in pct_returns:
        equity.append(equity[-1] * (1 + r))
    eq = pd.Series(equity)
    dd = (eq - eq.cummax()) / eq.cummax()
    max_dd = abs(dd.min())

    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else 99

    # Nifty correlation
    trade_ret_by_date = {}
    for t in trades:
        d = t['exit_date']
        trade_ret_by_date[d] = trade_ret_by_date.get(d, 0) + t['pnl'] / notional
    nifty_ret = df.set_index('date')['close'].pct_change()
    strat_ret = pd.Series(trade_ret_by_date)
    common = strat_ret.index.intersection(nifty_ret.index)
    if len(common) > 10:
        corr = float(strat_ret.reindex(common).corr(nifty_ret.reindex(common)))
        if np.isnan(corr):
            corr = 0
    else:
        corr = 0

    date_range = (pd.Timestamp(df['date'].max()) - pd.Timestamp(df['date'].min())).days
    years = max(1, date_range / 365.25)
    if equity[-1] > 0:
        annual_ret = equity[-1] ** (1 / years) - 1
    else:
        annual_ret = -1

    long_trades = [t for t in trades if t['direction'] == 'LONG']
    short_trades = [t for t in trades if t['direction'] == 'SHORT']
    long_wr = sum(1 for t in long_trades if t['pnl'] > 0) / len(long_trades) if long_trades else 0
    short_wr = sum(1 for t in short_trades if t['pnl'] > 0) / len(short_trades) if short_trades else 0

    return {
        'trades': len(trades),
        'sharpe': round(sharpe, 3),
        'win_rate': round(win_rate, 3),
        'profit_factor': round(min(pf, 99), 2),
        'max_dd': round(max_dd, 3),
        'nifty_corr': round(corr, 3),
        'annual_return': round(annual_ret, 4),
        'long_trades': len(long_trades),
        'long_wr': round(long_wr, 3),
        'short_trades': len(short_trades),
        'short_wr': round(short_wr, 3),
        'pnl_pts': round(sum(pnls), 1),
        'insufficient': len(trades) < 20,
    }


# ================================================================
# COMBINATION DEFINITIONS
# ================================================================

def combo_dry20_dry12_and(row, prev, df=None):
    """Entry: DRY_20 long AND DRY_12 long same day"""
    if check_dry20_long(row, prev) and check_dry12_long(row, prev):
        return 'LONG'
    return None

def combo_dry20_dry12_exit(row, prev, pos):
    if pos == 'LONG':
        return check_dry20_exit_long(row, prev) or check_dry12_exit_long(row, prev)
    return False

def combo_dry20_dry16_and(row, prev, df=None):
    if check_dry20_long(row, prev) and check_dry16_long(row, prev):
        return 'LONG'
    if check_dry16_short(row, prev):
        return 'SHORT'  # single-signal short
    return None

def combo_dry20_dry16_exit(row, prev, pos):
    if pos == 'LONG':
        return check_dry20_exit_long(row, prev) or check_dry16_exit_long(row, prev)
    if pos == 'SHORT':
        return check_dry16_exit_short(row, prev)
    return False

def combo_dry12_dry16_and(row, prev, df=None):
    if check_dry12_long(row, prev) and check_dry16_long(row, prev):
        return 'LONG'
    if check_dry12_short(row, prev) and check_dry16_short(row, prev):
        return 'SHORT'
    return None

def combo_dry12_dry16_exit(row, prev, pos):
    if pos == 'LONG':
        return check_dry12_exit_long(row, prev) or check_dry16_exit_long(row, prev)
    if pos == 'SHORT':
        return check_dry12_exit_short(row, prev) or check_dry16_exit_short(row, prev)
    return False

def combo_all_three_and(row, prev, df=None):
    if (check_dry20_long(row, prev) and check_dry12_long(row, prev) and
            check_dry16_long(row, prev)):
        return 'LONG'
    return None

def combo_all_three_exit(row, prev, pos):
    if pos == 'LONG':
        return (check_dry20_exit_long(row, prev) or
                check_dry12_exit_long(row, prev) or
                check_dry16_exit_long(row, prev))
    return False


# ================================================================
# SCORING SYSTEM
# ================================================================

def make_scoring_entry_exit(df):
    """Pre-compute daily scores for the scoring system."""
    scores = np.zeros(len(df))
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        score = 0
        if check_dry20_long(row, prev):
            score += 2
        if check_dry12_long(row, prev):
            score += 1
        if check_dry12_short(row, prev):
            score -= 1
        if check_dry16_long(row, prev):
            score += 1
        if check_dry16_short(row, prev):
            score -= 1
        scores[i] = score

    return scores


def run_scoring_backtest(df, stop_pct=0.02, tp_pct=0.03):
    """Scoring system backtest with variable position sizing."""
    scores = make_scoring_entry_exit(df)
    closes = df['close'].values
    dates = df['date'].values if 'date' in df.columns else df.index.values
    n = len(df)

    trades = []
    position = None
    entry_price = 0.0
    entry_idx = 0
    entry_threshold = 0
    size = 1.0

    for i in range(1, n):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = closes[i]
        score = scores[i]

        if position is None:
            if score >= 3:
                position = 'LONG'
                entry_price = price
                entry_idx = i
                entry_threshold = 3
                size = 1.0
            elif score >= 2:
                position = 'LONG'
                entry_price = price
                entry_idx = i
                entry_threshold = 2
                size = 0.5
            elif score <= -2:
                position = 'SHORT'
                entry_price = price
                entry_idx = i
                entry_threshold = -2
                size = 0.5
        else:
            # Stop loss
            if position == 'LONG':
                loss_pct = (entry_price - price) / entry_price
            else:
                loss_pct = (price - entry_price) / entry_price
            if loss_pct >= stop_pct:
                pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
                trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                               'direction': position, 'exit_reason': 'stop_loss',
                               'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                position = None
                continue

            # Take profit
            if tp_pct > 0:
                if position == 'LONG':
                    gain_pct = (price - entry_price) / entry_price
                else:
                    gain_pct = (entry_price - price) / entry_price
                if gain_pct >= tp_pct:
                    pnl = gain_pct * entry_price
                    trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                                   'direction': position, 'exit_reason': 'take_profit',
                                   'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                    position = None
                    continue

            # Score-based exit
            if position == 'LONG' and score < entry_threshold:
                pnl = price - entry_price
                trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                               'direction': position, 'exit_reason': 'score_drop',
                               'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                position = None
            elif position == 'SHORT' and score > entry_threshold:
                pnl = entry_price - price
                trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                               'direction': position, 'exit_reason': 'score_rise',
                               'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                position = None

    if position is not None:
        price = closes[-1]
        pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
        trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                       'direction': position, 'exit_reason': 'end_of_data',
                       'entry_date': dates[entry_idx], 'exit_date': dates[-1]})

    return compute_metrics(trades, df)


# ================================================================
# REGIME-CONDITIONAL STACKING
# ================================================================

def run_regime_conditional(df, regime_labels):
    """Regime-conditional signal stacking."""
    closes = df['close'].values
    dates = df['date'].values if 'date' in df.columns else df.index.values
    n = len(df)

    trades = []
    position = None
    entry_price = 0.0
    entry_idx = 0
    size = 1.0
    stop_pct = 0.02
    tp_pct = 0.03

    for i in range(1, n):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        price = closes[i]
        dt = dates[i]

        regime = (regime_labels.get(dt) or
                  regime_labels.get(pd.Timestamp(dt)) or
                  regime_labels.get(str(dt)[:10]) or 'UNKNOWN')

        if position is None:
            direction = None

            if regime in ('HIGH_VOL', 'CRISIS'):
                # No new entries — capital preservation
                # Force close DRY_20 longs handled below
                pass

            elif regime == 'TRENDING':
                # DRY_20 alone is sufficient in trends
                if check_dry20_long(row, prev):
                    direction = 'LONG'
                    size = 0.5
                    if check_dry16_long(row, prev):
                        size = 1.0  # DRY_16 confirmation → full size
                # Trend shorts via DRY_16 only when market looks weak
                elif check_dry16_short(row, prev):
                    direction = 'SHORT'
                    size = 0.5

            elif regime == 'RANGING':
                # Require DRY_12 confirmation for longs
                if check_dry12_long(row, prev) and check_dry20_long(row, prev):
                    direction = 'LONG'
                    size = 1.0
                elif check_dry12_short(row, prev) and check_dry16_short(row, prev):
                    direction = 'SHORT'
                    size = 1.0

            if direction:
                position = direction
                entry_price = price
                entry_idx = i
        else:
            # HIGH_VOL/CRISIS force exit for longs
            if regime in ('HIGH_VOL', 'CRISIS') and position == 'LONG':
                vix = row.get('india_vix', 0)
                if pd.notna(vix) and vix > 22:
                    pnl = price - entry_price
                    trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                                   'direction': position, 'exit_reason': 'vix_force',
                                   'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                    position = None
                    continue

            # Stop loss
            if position == 'LONG':
                loss_pct = (entry_price - price) / entry_price
            else:
                loss_pct = (price - entry_price) / entry_price
            if loss_pct >= stop_pct:
                pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
                trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                               'direction': position, 'exit_reason': 'stop_loss',
                               'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                position = None
                continue

            # Take profit
            if tp_pct > 0:
                if position == 'LONG':
                    gain_pct = (price - entry_price) / entry_price
                else:
                    gain_pct = (entry_price - price) / entry_price
                if gain_pct >= tp_pct:
                    pnl = gain_pct * entry_price
                    trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                                   'direction': position, 'exit_reason': 'take_profit',
                                   'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                    position = None
                    continue

            # Signal exit (most conservative — any exit fires)
            exit_fired = False
            if position == 'LONG':
                exit_fired = (check_dry20_exit_long(row, prev) or
                              check_dry12_exit_long(row, prev) or
                              check_dry16_exit_long(row, prev))
            else:
                exit_fired = (check_dry12_exit_short(row, prev) or
                              check_dry16_exit_short(row, prev))

            if exit_fired:
                pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
                trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                               'direction': position, 'exit_reason': 'signal',
                               'entry_date': dates[entry_idx], 'exit_date': dates[i]})
                position = None

    if position is not None:
        price = closes[-1]
        pnl = (price - entry_price) if position == 'LONG' else (entry_price - price)
        trades.append({'pnl': pnl * size, 'entry_price': entry_price,
                       'direction': position, 'exit_reason': 'end_of_data',
                       'entry_date': dates[entry_idx], 'exit_date': dates[-1]})

    return compute_metrics(trades, df)


# ================================================================
# INDIVIDUAL SIGNAL BASELINES (using same engine for consistency)
# ================================================================

def run_dry20_alone(df):
    def entry(row, prev):
        return 'LONG' if check_dry20_long(row, prev) else None
    def exit_fn(row, prev, pos):
        return check_dry20_exit_long(row, prev) if pos == 'LONG' else False
    return run_combination_backtest(df, entry, exit_fn, stop_pct=0.02, tp_pct=0, hold_max=0)

def run_dry16_alone(df):
    def entry(row, prev):
        if check_dry16_long(row, prev): return 'LONG'
        if check_dry16_short(row, prev): return 'SHORT'
        return None
    def exit_fn(row, prev, pos):
        if pos == 'LONG': return check_dry16_exit_long(row, prev)
        if pos == 'SHORT': return check_dry16_exit_short(row, prev)
        return False
    return run_combination_backtest(df, entry, exit_fn, stop_pct=0.02, tp_pct=0.03, hold_max=0)

def run_dry12_alone(df):
    def entry(row, prev):
        if check_dry12_long(row, prev): return 'LONG'
        if check_dry12_short(row, prev): return 'SHORT'
        return None
    def exit_fn(row, prev, pos):
        if pos == 'LONG': return check_dry12_exit_long(row, prev)
        if pos == 'SHORT': return check_dry12_exit_short(row, prev)
        return False
    return run_combination_backtest(df, entry, exit_fn, stop_pct=0.02, tp_pct=0.03, hold_max=7)


# ================================================================
# MAIN
# ================================================================

def main():
    print("Loading data...", flush=True)
    conn = psycopg2.connect(DATABASE_DSN)
    df_raw = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix "
        "FROM nifty_daily ORDER BY date", conn)
    conn.close()
    df_raw['date'] = pd.to_datetime(df_raw['date'])

    # Add indicators
    df = add_all_indicators(df_raw)
    df['hvol_6'] = historical_volatility(df['close'], period=6)
    df['hvol_100'] = historical_volatility(df['close'], period=100)
    df['date'] = df_raw['date']
    print(f"  {len(df)} trading days, indicators computed", flush=True)

    # Regime labels
    from regime_labeler import RegimeLabeler
    regime_labels = RegimeLabeler().label_full_history(df_raw)

    # Slices
    df_10y = df.copy()
    df_6mo = df[df['date'] >= '2025-09-15'].reset_index(drop=True)
    print(f"  10yr: {len(df_10y)} days, 6mo: {len(df_6mo)} days", flush=True)

    # Define all combinations
    combos = [
        ('DRY_20 alone', lambda d: run_dry20_alone(d), None),
        ('DRY_16 alone', lambda d: run_dry16_alone(d), None),
        ('DRY_12 alone', lambda d: run_dry12_alone(d), None),
        ('DRY20+DRY12 AND', lambda d: run_combination_backtest(
            d, combo_dry20_dry12_and, combo_dry20_dry12_exit, 0.02, 0.03), None),
        ('DRY20+DRY16 AND', lambda d: run_combination_backtest(
            d, combo_dry20_dry16_and, combo_dry20_dry16_exit, 0.02, 0.03), None),
        ('DRY12+DRY16 AND', lambda d: run_combination_backtest(
            d, combo_dry12_dry16_and, combo_dry12_dry16_exit, 0.02, 0.03), None),
        ('ALL THREE AND', lambda d: run_combination_backtest(
            d, combo_all_three_and, combo_all_three_exit, 0.02, 0.03), None),
        ('SCORING', lambda d: run_scoring_backtest(d, 0.02, 0.03), None),
        ('REGIME-COND', None, 'regime'),  # special handling
    ]

    results = {}
    print(f"\nRunning backtests...", flush=True)

    for name, fn, special in combos:
        print(f"  {name}...", end=' ', flush=True)

        if special == 'regime':
            r10 = run_regime_conditional(df_10y, regime_labels)
            r6 = run_regime_conditional(df_6mo, regime_labels)
        else:
            r10 = fn(df_10y)
            r6 = fn(df_6mo)

        results[name] = {'10yr': r10, '6mo': r6}
        flag = ''
        if r10['insufficient']:
            flag = ' [INSUFFICIENT_TRADES]'
        print(f"10yr: Sh={r10['sharpe']:.2f} Tr={r10['trades']} | "
              f"6mo: Sh={r6['sharpe']:.2f} Tr={r6['trades']}{flag}", flush=True)

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print(f"\n{'='*120}")
    print(f"COMBINATION COMPARISON TABLE")
    print(f"{'='*120}")
    print(f"{'Combination':20s} | {'--- 10-YEAR ---':^52s} | {'--- 6-MONTH ---':^30s} |")
    print(f"{'':20s} | {'Trades':>6s} {'WR':>6s} {'Sharpe':>7s} {'PF':>6s} {'MaxDD':>6s} {'Corr':>6s} {'AnnRet':>7s} {'Flag':>8s} | "
          f"{'Trades':>6s} {'WR':>6s} {'Sharpe':>7s} {'P&L':>8s} |")
    print(f"{'-'*20}-+-{'-'*52}-+-{'-'*30}-+")

    for name in [c[0] for c in combos]:
        r10 = results[name]['10yr']
        r6 = results[name]['6mo']
        flag = 'INSUFF' if r10['insufficient'] else ''
        print(f"{name:20s} | {r10['trades']:6d} {r10['win_rate']:5.0%} {r10['sharpe']:7.2f} "
              f"{r10['profit_factor']:6.2f} {r10['max_dd']:5.1%} {r10['nifty_corr']:6.3f} "
              f"{r10['annual_return']:6.2%} {flag:>8s} | "
              f"{r6['trades']:6d} {r6['win_rate']:5.0%} {r6['sharpe']:7.2f} {r6['pnl_pts']:+8.1f} |")

    # ================================================================
    # WINNER ANALYSIS
    # ================================================================
    print(f"\n{'='*120}")
    print(f"WINNER ANALYSIS")
    print(f"{'='*120}")

    dry20_10 = results['DRY_20 alone']['10yr']
    dry20_6 = results['DRY_20 alone']['6mo']

    for name in [c[0] for c in combos]:
        if name == 'DRY_20 alone':
            continue
        r10 = results[name]['10yr']
        r6 = results[name]['6mo']

        criteria = []
        passes = 0

        # 1. WF proxy: use trade count + sharpe as proxy (no walk-forward in this run)
        if r10['trades'] >= 20 * 10:  # ~20/year minimum
            criteria.append(('Sufficient trades', True))
            passes += 1
        else:
            criteria.append(('Sufficient trades', False))

        # 2. Recent 6mo P&L positive
        if r6['pnl_pts'] > 0:
            criteria.append(('6mo P&L positive', True))
            passes += 1
        else:
            criteria.append(('6mo P&L positive', False))

        # 3. Sharpe >= 1.5
        if r10['sharpe'] >= 1.5:
            criteria.append(('Sharpe >= 1.5', True))
            passes += 1
        else:
            criteria.append(('Sharpe >= 1.5', False))

        # 4. Max DD <= 20%
        if r10['max_dd'] <= 0.20:
            criteria.append(('MaxDD <= 20%', True))
            passes += 1
        else:
            criteria.append(('MaxDD <= 20%', False))

        # 5. Beats DRY_20 Sharpe
        if r10['sharpe'] > dry20_10['sharpe']:
            criteria.append(('Beats DRY_20 Sharpe', True))
            passes += 1
        else:
            criteria.append(('Beats DRY_20 Sharpe', False))

        status = 'COMBINATION_WINNER' if passes == 5 else f'{passes}/5 criteria'
        print(f"\n  {name}: {status}")
        for c, ok in criteria:
            print(f"    {'✓' if ok else '✗'} {c}")

    # ================================================================
    # SAVE
    # ================================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_data = {}
    for name, data in results.items():
        save_data[name] = data
    with open(os.path.join(RESULTS_DIR, 'combination_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_DIR}/combination_results.json")


if __name__ == '__main__':
    main()

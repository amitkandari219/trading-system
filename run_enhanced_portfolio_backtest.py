"""
Enhanced portfolio backtest: baseline vs cross-asset overlays vs Kelly sizing.

Runs all 7 scoring signals across 3 sizing modes over the past year:
  A) BASELINE  — fixed 1% risk, drawdown scaling (existing system)
  B) CROSS-ASSET — baseline + cross-asset multiplier (0.2x-2.0x)
  C) FULL ENHANCED — cross-asset + Kelly quarter-fraction sizing + FII overlay

Usage:
    venv/bin/python3 run_enhanced_portfolio_backtest.py
    venv/bin/python3 run_enhanced_portfolio_backtest.py --start 2025-03-21
"""

import argparse
import logging
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import _eval_conditions
from backtest.indicators import add_all_indicators, historical_volatility
from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE, MAX_POSITIONS, MAX_SAME_DIRECTION
from paper_trading.kelly_sizer import KellySizer
from signals.cross_asset_signals import CrossAssetOverlays

logger = logging.getLogger(__name__)

# ================================================================
# SIGNAL DEFINITIONS — same as run_portfolio_backtest.py
# ================================================================
PORTFOLIO_SIGNALS = {
    'KAUFMAN_DRY_20': {
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'entry_long': [
            {'indicator': 'sma_10', 'op': '<', 'value': 'prev_close'},
            {'indicator': 'stoch_k_5', 'op': '>', 'value': 50},
        ],
        'exit_long': [{'indicator': 'stoch_k_5', 'op': '<=', 'value': 50}],
    },
    'KAUFMAN_DRY_16': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.03,
        'hold_days_max': 0,
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'r1'},
            {'indicator': 'low', 'op': '>=', 'value': 'pivot'},
            {'indicator': 'hvol_6', 'op': '<', 'value': 'hvol_100'},
        ],
        'entry_short': [
            {'indicator': 'close', 'op': '<', 'value': 's1'},
            {'indicator': 'hvol_6', 'op': '<', 'value': 'hvol_100'},
        ],
        'exit_long': [{'indicator': 'low', 'op': '<', 'value': 'pivot'}],
        'exit_short': [{'indicator': 'high', 'op': '>', 'value': 'r1'}],
    },
    'KAUFMAN_DRY_12': {
        'direction': 'BOTH',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.03,
        'hold_days_max': 7,
        'adaptive_sizing': True,  # Reduce to 0.5x when rolling 60-day PnL < 0
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'prev_close'},
            {'indicator': 'volume', 'op': '<', 'value': 'prev_volume'},
        ],
        'entry_short': [
            {'indicator': 'close', 'op': '<', 'value': 'prev_close'},
            {'indicator': 'volume', 'op': '<', 'value': 'prev_volume'},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'prev_close'}],
        'exit_short': [{'indicator': 'close', 'op': '>', 'value': 'prev_close'}],
    },
    'GUJRAL_DRY_8': {
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 0,
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'pivot'},
            {'indicator': 'open', 'op': '>', 'value': 'pivot'},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'pivot'}],
    },
    'GUJRAL_DRY_13': {
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 10,
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'prev_high'},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'prev_low'}],
    },
    'PCR_ELEVATED_FEAR': {
        'direction': 'LONG',
        'stop_loss_pct': 0.02,
        'take_profit_pct': 0.0,
        'hold_days_max': 5,
        'entry_long': [
            {'indicator': 'pcr_oi', 'op': '>', 'value': 1.3},
        ],
        'exit_long': [{'indicator': 'pcr_oi', 'op': '<', 'value': 1.1}],
    },
}

# Drawdown scaling levels
DD_LEVELS = [(0.05, 0.75), (0.10, 0.50), (0.15, 0.25), (0.20, 0.0)]


def load_cross_asset_data(conn, start_date):
    """Load cross-asset data from DB, pivoted by date."""
    df = pd.read_sql(
        f"""SELECT date, instrument, close, daily_return, weekly_return, zscore_60
            FROM cross_asset_daily
            WHERE date >= '{start_date}'
            ORDER BY date""",
        conn,
    )
    if df.empty:
        return {}

    # Pivot: {date_str: {instrument: {close, daily_return, weekly_return, zscore_60}}}
    cross_data = {}
    for dt, group in df.groupby('date'):
        day_data = {}
        for _, row in group.iterrows():
            day_data[row['instrument']] = {
                'close': row['close'],
                'daily_return': row['daily_return'],
                'weekly_return': row['weekly_return'],
                'zscore_60': row['zscore_60'],
            }
        cross_data[str(dt)] = day_data

    return cross_data


def compute_cross_asset_mult(cross_data, bar_date):
    """Get cross-asset composite multiplier for a given date."""
    date_str = str(bar_date.date()) if hasattr(bar_date, 'date') else str(bar_date)
    today_data = cross_data.get(date_str, {})
    if not today_data:
        return 1.0, {}

    overlays = CrossAssetOverlays()
    result = overlays.compute(today_data)
    return result['composite_multiplier'], result.get('signals', {})


def run_single_backtest(df, capital, cross_data, mode='baseline'):
    """
    Run portfolio backtest in one of three modes.

    mode:
      'baseline'    — fixed 1% risk + drawdown scaling
      'cross_asset' — baseline + cross-asset multiplier
      'full'        — cross-asset + Kelly sizing + FII
    """
    kelly = KellySizer() if mode == 'full' else None
    n_bars = len(df)

    equity = capital
    peak_equity = capital
    positions = []
    closed_trades = []
    daily_equity = []
    daily_pnl_list = []

    # Track rolling P&L per signal for adaptive sizing (last 60 days of trades)
    signal_recent_pnl = defaultdict(list)  # signal_id -> [(bar_idx, pnl_rs)]

    for i in range(1, n_bars):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        bar_date = row['date']
        close = float(row['close'])
        vix = float(row['india_vix']) if pd.notna(row.get('india_vix')) else 15.0

        day_pnl = 0.0

        # --- EXITS ---
        still_open = []
        for pos in positions:
            entry_price = pos['entry_price']
            direction = pos['direction']
            signal_id = pos['signal_id']
            config = PORTFOLIO_SIGNALS[signal_id]
            days_held = i - pos['entry_idx']

            if direction == 'LONG':
                loss_pct = (entry_price - close) / entry_price
            else:
                loss_pct = (close - entry_price) / entry_price

            exit_reason = None

            if loss_pct >= config['stop_loss_pct']:
                exit_reason = 'stop_loss'
            elif config['take_profit_pct'] > 0:
                gain = ((close - entry_price) / entry_price if direction == 'LONG'
                        else (entry_price - close) / entry_price)
                if gain >= config['take_profit_pct']:
                    exit_reason = 'take_profit'

            if not exit_reason and config['hold_days_max'] > 0 and days_held >= config['hold_days_max']:
                exit_reason = 'hold_days_max'

            if not exit_reason:
                # Min hold: don't signal-exit before min_hold_days (SL/TP still apply)
                min_hold = config.get('min_hold_days', 0)
                if days_held >= min_hold:
                    exit_key = 'exit_long' if direction == 'LONG' else 'exit_short'
                    exit_conds = config.get(exit_key, [])
                    if exit_conds and _eval_conditions(row, prev, exit_conds):
                        exit_reason = 'signal_exit'

            if exit_reason:
                pnl_pts = (close - entry_price) if direction == 'LONG' else (entry_price - close)
                pnl_rs = pnl_pts * NIFTY_LOT_SIZE * pos['lots']
                day_pnl += pnl_rs
                closed_trades.append({
                    'signal_id': signal_id, 'direction': direction,
                    'entry_date': pos['entry_date'], 'exit_date': bar_date,
                    'entry_price': entry_price, 'exit_price': close,
                    'pnl_pts': round(pnl_pts, 2), 'pnl_rs': round(pnl_rs, 2),
                    'lots': pos['lots'], 'days_held': days_held,
                    'exit_reason': exit_reason,
                    'size_mult': pos.get('size_mult', 1.0),
                })
                # Track for adaptive sizing
                signal_recent_pnl[signal_id].append((i, pnl_rs))
            else:
                still_open.append(pos)

        positions = still_open

        # --- DRAWDOWN SCALING ---
        dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        dd_mult = 1.0
        for dd_threshold, mult in DD_LEVELS:
            if dd_pct >= dd_threshold:
                dd_mult = mult

        if dd_mult == 0:
            equity += day_pnl
            peak_equity = max(peak_equity, equity)
            daily_equity.append({'date': bar_date, 'equity': equity})
            daily_pnl_list.append(day_pnl)
            continue

        # --- CROSS-ASSET MULTIPLIER ---
        ca_mult = 1.0
        if mode in ('cross_asset', 'full'):
            ca_mult, _ = compute_cross_asset_mult(cross_data, bar_date)

        # --- ENTRIES ---
        active_sids = {p['signal_id'] for p in positions}
        n_long = sum(1 for p in positions if p['direction'] == 'LONG')
        n_short = sum(1 for p in positions if p['direction'] == 'SHORT')

        for signal_id, config in PORTFOLIO_SIGNALS.items():
            if signal_id in active_sids:
                continue
            if len(positions) >= MAX_POSITIONS:
                break

            direction = config['direction']

            # Volatility floor: skip entry when daily range too compressed
            min_range = config.get('min_range_pct', 0)
            if min_range > 0:
                day_range = (float(row['high']) - float(row['low'])) / close
                if day_range < min_range:
                    continue

            fired_direction = None
            if direction in ('LONG', 'BOTH'):
                entry_conds = config.get('entry_long', [])
                if entry_conds and _eval_conditions(row, prev, entry_conds):
                    if n_long < MAX_SAME_DIRECTION:
                        fired_direction = 'LONG'

            if not fired_direction and direction in ('SHORT', 'BOTH'):
                entry_conds = config.get('entry_short', [])
                if entry_conds and _eval_conditions(row, prev, entry_conds):
                    if n_short < MAX_SAME_DIRECTION:
                        fired_direction = 'SHORT'

            if fired_direction:
                # --- POSITION SIZING ---
                base_risk_pct = 0.01  # 1% baseline

                # Kelly override
                kelly_mult = 1.0
                kelly_grade = '-'
                if mode == 'full' and kelly:
                    kelly_result = kelly.compute_size(
                        signal_id, features={}, regime=None, vix=vix
                    )
                    if kelly_result['skip_trade']:
                        continue  # Kelly says skip
                    kelly_mult = kelly_result['size_multiplier']
                    kelly_grade = kelly_result['grade']

                # Adaptive sizing: reduce to 0.5x when signal's rolling P&L is negative
                adaptive_mult = 1.0
                if config.get('adaptive_sizing'):
                    # Look at last 60 days of trades for this signal
                    recent = [(idx, pnl) for idx, pnl in signal_recent_pnl.get(signal_id, [])
                              if i - idx <= 60]
                    if len(recent) >= 5:
                        rolling_pnl = sum(pnl for _, pnl in recent)
                        if rolling_pnl < 0:
                            adaptive_mult = 0.5  # Halve size during losing streaks

                # Composite multiplier
                size_mult = dd_mult * ca_mult * kelly_mult * adaptive_mult

                # Cross-asset gate: skip trade if composite mult < 0.3 (severe stress only)
                if mode in ('cross_asset', 'full') and ca_mult < 0.3:
                    continue

                # At 1-lot minimum, multiplier can't reduce lots below 1.
                # Apply multiplier as P&L scaling factor instead.
                risk_amount = equity * base_risk_pct * dd_mult
                stop_pts = close * config['stop_loss_pct']
                risk_per_lot = stop_pts * NIFTY_LOT_SIZE
                lots = max(1, int(risk_amount / risk_per_lot)) if risk_per_lot > 0 else 1
                lots = min(lots, 50)

                # Scale lots up when composite > 1.0 and capital permits
                if size_mult > 1.3 and lots < 50:
                    lots = min(50, max(lots, int(lots * size_mult)))

                positions.append({
                    'signal_id': signal_id,
                    'direction': fired_direction,
                    'entry_price': close,
                    'entry_date': bar_date,
                    'entry_idx': i,
                    'lots': lots,
                    'size_mult': round(size_mult, 3),
                    'kelly_grade': kelly_grade,
                })

                if fired_direction == 'LONG':
                    n_long += 1
                else:
                    n_short += 1

        equity += day_pnl
        peak_equity = max(peak_equity, equity)
        daily_equity.append({'date': bar_date, 'equity': equity})
        daily_pnl_list.append(day_pnl)

    return compute_metrics(capital, equity, closed_trades, daily_equity, daily_pnl_list, df)


def compute_metrics(capital, equity, trades, daily_equity, daily_pnl_list, df):
    """Compute standard portfolio metrics."""
    n_bars = len(df)
    equity_series = pd.Series([capital] + [d['equity'] for d in daily_equity])
    returns = equity_series.pct_change().dropna()

    total_return = (equity - capital) / capital
    years = n_bars / 252
    cagr = (equity / capital) ** (1 / max(years, 0.01)) - 1

    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    rolling_max = equity_series.cummax()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_dd = abs(drawdowns.min())
    calmar = cagr / max_dd if max_dd > 0 else 0

    pnls = [t['pnl_rs'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0

    # Monthly returns
    eq_df = pd.DataFrame(daily_equity)
    winning_months = 0
    total_months = 0
    monthly_returns = pd.Series(dtype=float)
    if not eq_df.empty:
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        eq_df['month'] = eq_df['date'].dt.to_period('M')
        monthly = eq_df.groupby('month')['equity'].last()
        monthly_returns = monthly.pct_change().dropna()
        winning_months = int((monthly_returns > 0).sum())
        total_months = len(monthly_returns)

    # Per-signal stats
    signal_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0, 'pnl_list': []})
    for t in trades:
        sid = t['signal_id']
        signal_stats[sid]['trades'] += 1
        signal_stats[sid]['total_pnl'] += t['pnl_rs']
        signal_stats[sid]['pnl_list'].append(t['pnl_rs'])
        if t['pnl_rs'] > 0:
            signal_stats[sid]['wins'] += 1

    return {
        'capital': capital,
        'final_equity': round(equity),
        'total_return_pct': round(total_return * 100, 2),
        'cagr_pct': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'max_dd_pct': round(max_dd * 100, 2),
        'calmar': round(calmar, 2),
        'total_trades': len(trades),
        'win_rate': round(win_rate, 1),
        'profit_factor': round(pf, 2),
        'total_pnl': round(sum(pnls)),
        'avg_trade_pnl': round(np.mean(pnls)) if pnls else 0,
        'winning_months': winning_months,
        'total_months': total_months,
        'worst_month_pct': round(monthly_returns.min() * 100, 1) if len(monthly_returns) > 0 else 0,
        'best_month_pct': round(monthly_returns.max() * 100, 1) if len(monthly_returns) > 0 else 0,
        'years': round(years, 1),
        'signal_stats': {
            sid: {
                'trades': s['trades'],
                'wins': s['wins'],
                'win_rate': round(s['wins'] / s['trades'] * 100, 1) if s['trades'] > 0 else 0,
                'total_pnl': round(s['total_pnl']),
                'avg_pnl': round(np.mean(s['pnl_list'])) if s['pnl_list'] else 0,
            }
            for sid, s in signal_stats.items()
        },
        'trades': trades,
    }


def print_comparison(results):
    """Print side-by-side comparison of all modes."""
    modes = list(results.keys())

    print(f"\n{'=' * 80}")
    print(f"ENHANCED PORTFOLIO BACKTEST — SIDE-BY-SIDE COMPARISON")
    print(f"{'=' * 80}")

    # Header
    header = f"{'Metric':<25}"
    for mode in modes:
        header += f" {mode:>16}"
    print(header)
    print('─' * (25 + 17 * len(modes)))

    # Rows
    metrics = [
        ('Final Equity', 'final_equity', '₹{:>12,.0f}'),
        ('Total Return', 'total_return_pct', '{:>12.2f}%'),
        ('CAGR', 'cagr_pct', '{:>12.2f}%'),
        ('Sharpe', 'sharpe', '{:>13.2f}'),
        ('Max Drawdown', 'max_dd_pct', '{:>12.2f}%'),
        ('Calmar', 'calmar', '{:>13.2f}'),
        ('Profit Factor', 'profit_factor', '{:>13.2f}'),
        ('Total Trades', 'total_trades', '{:>13d}'),
        ('Win Rate', 'win_rate', '{:>12.1f}%'),
        ('Avg Trade P&L', 'avg_trade_pnl', '₹{:>12,.0f}'),
        ('Total P&L', 'total_pnl', '₹{:>12,.0f}'),
        ('Winning Months', 'winning_months', '{:>9d}/{:d}'),
        ('Best Month', 'best_month_pct', '{:>12.1f}%'),
        ('Worst Month', 'worst_month_pct', '{:>12.1f}%'),
    ]

    for label, key, fmt in metrics:
        row = f"{label:<25}"
        for mode in modes:
            r = results[mode]
            if key == 'winning_months':
                val = f"{r['winning_months']:>9d}/{r['total_months']}"
                row += f" {val:>16}"
            else:
                val = r[key]
                row += f" {fmt.format(val):>16}"
        print(row)

    # Per-signal breakdown for each mode
    for mode in modes:
        r = results[mode]
        print(f"\n{'─' * 80}")
        print(f"Signal breakdown — {mode}")
        print(f"{'Signal':<22} {'Trades':>7} {'Wins':>5} {'WR':>6} {'Total P&L':>12} {'Avg':>10}")
        print(f"{'─' * 22} {'─' * 7} {'─' * 5} {'─' * 6} {'─' * 12} {'─' * 10}")

        for sid in PORTFOLIO_SIGNALS:
            ss = r['signal_stats'].get(sid, {})
            if ss.get('trades', 0) > 0:
                print(f"{sid:<22} {ss['trades']:>7} {ss['wins']:>5} {ss['win_rate']:>5.1f}% "
                      f"₹{ss['total_pnl']:>11,} ₹{ss['avg_pnl']:>9,}")
            else:
                print(f"{sid:<22}       0     -      -            -          -")

    # Delta analysis
    if len(modes) > 1:
        base = results[modes[0]]
        print(f"\n{'─' * 80}")
        print(f"Impact vs BASELINE:")
        for mode in modes[1:]:
            r = results[mode]
            sharpe_delta = r['sharpe'] - base['sharpe']
            dd_delta = r['max_dd_pct'] - base['max_dd_pct']
            pnl_delta = r['total_pnl'] - base['total_pnl']
            print(f"  {mode}:  Sharpe {sharpe_delta:+.2f}  |  MaxDD {dd_delta:+.2f}%  |  P&L {'+' if pnl_delta >= 0 else ''}₹{pnl_delta:,.0f}")

    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced portfolio backtest')
    parser.add_argument('--start', type=str, default='2025-03-21',
                        help='Start date (default: 2025-03-21 = 1 year)')
    parser.add_argument('--full-history', action='store_true',
                        help='Run on full history (since 2021, when cross-asset data starts)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    start = '2021-03-16' if args.full_history else args.start
    capital = 1_000_000

    print("Loading data...", flush=True)
    conn = psycopg2.connect(DATABASE_DSN)

    df = pd.read_sql(
        f"SELECT date, open, high, low, close, volume, india_vix "
        f"FROM nifty_daily WHERE date >= '{start}' ORDER BY date", conn
    )
    df['date'] = pd.to_datetime(df['date'])

    # PCR data for PCR_ELEVATED_FEAR signal
    try:
        pcr = pd.read_sql(
            f"SELECT date, pcr_oi FROM pcr_daily WHERE date >= '{start}' ORDER BY date", conn
        )
        pcr['date'] = pd.to_datetime(pcr['date'])
        df = df.merge(pcr, on='date', how='left')
    except Exception:
        df['pcr_oi'] = np.nan

    cross_data = load_cross_asset_data(conn, start)
    conn.close()

    print(f"  {len(df)} trading days ({df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()})")
    print(f"  {len(cross_data)} days of cross-asset data")

    print("Computing indicators...", flush=True)
    df = add_all_indicators(df)
    df['hvol_6'] = historical_volatility(df['close'], period=6)
    df['hvol_100'] = historical_volatility(df['close'], period=100)

    # Fill PCR NaN for signal eval
    if 'pcr_oi' not in df.columns:
        df['pcr_oi'] = np.nan

    print(f"\nRunning 3 backtest modes on {len(PORTFOLIO_SIGNALS)} signals...")
    print(f"Capital: ₹{capital:,.0f}\n")

    results = {}

    for mode, label in [('baseline', 'BASELINE'), ('cross_asset', 'CROSS-ASSET'), ('full', 'FULL ENHANCED')]:
        print(f"  Running {label}...", flush=True)
        r = run_single_backtest(df, capital, cross_data, mode=mode)
        results[label] = r
        print(f"    Sharpe={r['sharpe']:.2f}  Return={r['total_return_pct']:.1f}%  "
              f"MaxDD={r['max_dd_pct']:.1f}%  Trades={r['total_trades']}")

    print_comparison(results)


if __name__ == '__main__':
    main()

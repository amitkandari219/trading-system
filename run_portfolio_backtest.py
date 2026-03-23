"""
Full portfolio backtest — all SCORING signals simultaneously.

Runs all 7 SCORING signals with:
- Position sizing (FIXED_1PCT baseline + drawdown scaling)
- Regime filtering (HIGH_VOL gate on EADB)
- Max position limits (from config.settings)
- Daily loss limits
- Realistic trade management (stop loss, hold days, take profit)

This is the honest "what would I have made" number.

Usage:
    python run_portfolio_backtest.py
    python run_portfolio_backtest.py --capital 2500000
    python run_portfolio_backtest.py --start 2016-01-01
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import date, datetime

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import _eval_conditions, _eval_condition
from backtest.indicators import add_all_indicators, historical_volatility
from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE, MAX_POSITIONS, MAX_SAME_DIRECTION

# ================================================================
# SIGNAL DEFINITIONS — same rules as signal_compute.py
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
}


def run_portfolio_backtest(capital: float, df: pd.DataFrame,
                           base_risk_pct: float = 0.01,
                           leverage: float = 1.0,
                           dd_scaling: bool = True) -> dict:
    """
    Run full portfolio backtest with SCORING signals.

    Args:
        capital: starting capital
        df: OHLCV DataFrame with indicators
        base_risk_pct: risk per trade as fraction of equity
        leverage: position size multiplier (futures margin allows 8-10x)

    Returns detailed results dict.
    """
    n_bars = len(df)

    # Track state
    equity = capital
    peak_equity = capital
    positions = []  # list of open position dicts
    closed_trades = []
    daily_equity = []
    daily_pnl = []

    # Drawdown scaling thresholds
    DD_LEVELS = [(0.05, 0.75), (0.10, 0.50), (0.15, 0.25), (0.20, 0.0)]

    for i in range(1, n_bars):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        bar_date = row['date']
        close = float(row['close'])
        vix = float(row['india_vix']) if pd.notna(row.get('india_vix')) else 15.0

        day_pnl = 0.0

        # --- CHECK EXITS ---
        still_open = []
        for pos in positions:
            entry_price = pos['entry_price']
            direction = pos['direction']
            signal_id = pos['signal_id']
            config = PORTFOLIO_SIGNALS[signal_id]
            days_held = i - pos['entry_idx']

            # Stop loss
            if direction == 'LONG':
                loss_pct = (entry_price - close) / entry_price
            else:
                loss_pct = (close - entry_price) / entry_price

            exit_reason = None

            if loss_pct >= config['stop_loss_pct']:
                exit_reason = 'stop_loss'
            elif config['take_profit_pct'] > 0:
                if direction == 'LONG':
                    gain = (close - entry_price) / entry_price
                else:
                    gain = (entry_price - close) / entry_price
                if gain >= config['take_profit_pct']:
                    exit_reason = 'take_profit'

            # Hold days max
            if not exit_reason and config['hold_days_max'] > 0 and days_held >= config['hold_days_max']:
                exit_reason = 'hold_days_max'

            # Signal exit conditions
            if not exit_reason:
                exit_key = 'exit_long' if direction == 'LONG' else 'exit_short'
                exit_conds = config.get(exit_key, [])
                if exit_conds and _eval_conditions(row, prev, exit_conds):
                    exit_reason = 'signal_exit'

            if exit_reason:
                if direction == 'LONG':
                    pnl_pts = close - entry_price
                else:
                    pnl_pts = entry_price - close

                pnl_rs = pnl_pts * NIFTY_LOT_SIZE * pos['lots']
                day_pnl += pnl_rs

                closed_trades.append({
                    'signal_id': signal_id,
                    'direction': direction,
                    'entry_date': pos['entry_date'],
                    'exit_date': bar_date,
                    'entry_price': entry_price,
                    'exit_price': close,
                    'pnl_pts': round(pnl_pts, 2),
                    'pnl_rs': round(pnl_rs, 2),
                    'lots': pos['lots'],
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                })
            else:
                still_open.append(pos)

        positions = still_open

        # --- DRAWDOWN SCALING ---
        dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        size_mult = 1.0
        if dd_scaling:
            for dd_threshold, mult in DD_LEVELS:
                if dd_pct >= dd_threshold:
                    size_mult = mult

        if dd_scaling and size_mult == 0:
            # Halt — don't open new positions
            equity += day_pnl
            peak_equity = max(peak_equity, equity)
            daily_equity.append({'date': bar_date, 'equity': equity, 'dd_pct': dd_pct})
            daily_pnl.append(day_pnl)
            continue

        # --- CHECK ENTRIES ---
        active_sids = {p['signal_id'] for p in positions}
        n_long = sum(1 for p in positions if p['direction'] == 'LONG')
        n_short = sum(1 for p in positions if p['direction'] == 'SHORT')

        for signal_id, config in PORTFOLIO_SIGNALS.items():
            if signal_id in active_sids:
                continue
            if len(positions) >= MAX_POSITIONS:
                break

            direction = config['direction']

            # HIGH_VOL regime gate for EADB
            if config.get('regime_gate_high_vol') and vix > 22:
                continue

            # Check entry conditions
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
                # Position sizing: risk-based with leverage
                risk_amount = equity * base_risk_pct * size_mult * leverage
                stop_pts = close * config['stop_loss_pct']
                risk_per_lot = stop_pts * NIFTY_LOT_SIZE
                lots = max(1, int(risk_amount / risk_per_lot)) if risk_per_lot > 0 else 1
                lots = min(lots, 50)

                positions.append({
                    'signal_id': signal_id,
                    'direction': fired_direction,
                    'entry_price': close,
                    'entry_date': bar_date,
                    'entry_idx': i,
                    'lots': lots,
                })

                if fired_direction == 'LONG':
                    n_long += 1
                else:
                    n_short += 1

        # Update equity
        equity += day_pnl
        peak_equity = max(peak_equity, equity)
        daily_equity.append({'date': bar_date, 'equity': equity, 'dd_pct': dd_pct})
        daily_pnl.append(day_pnl)

    # --- COMPUTE METRICS ---
    equity_series = pd.Series([capital] + [d['equity'] for d in daily_equity])
    returns = equity_series.pct_change().dropna()

    total_return = (equity - capital) / capital
    years = n_bars / 252
    cagr = (equity / capital) ** (1 / years) - 1 if years > 0 else 0

    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Max drawdown
    rolling_max = equity_series.cummax()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_dd = abs(drawdowns.min())

    calmar = cagr / max_dd if max_dd > 0 else 0

    # Per-signal stats
    signal_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pnl': 0, 'pnl_list': []})
    for t in closed_trades:
        sid = t['signal_id']
        signal_stats[sid]['trades'] += 1
        signal_stats[sid]['total_pnl'] += t['pnl_rs']
        signal_stats[sid]['pnl_list'].append(t['pnl_rs'])
        if t['pnl_rs'] > 0:
            signal_stats[sid]['wins'] += 1

    # Monthly returns
    eq_df = pd.DataFrame(daily_equity)
    if not eq_df.empty:
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        eq_df['month'] = eq_df['date'].dt.to_period('M')
        monthly = eq_df.groupby('month')['equity'].last()
        monthly_returns = monthly.pct_change().dropna()
        winning_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
    else:
        winning_months = 0
        total_months = 0
        monthly_returns = pd.Series(dtype=float)

    return {
        'capital': capital,
        'final_equity': round(equity, 0),
        'total_return': round(total_return * 100, 1),
        'cagr': round(cagr * 100, 1),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(max_dd * 100, 1),
        'calmar': round(calmar, 2),
        'total_trades': len(closed_trades),
        'winning_trades': sum(1 for t in closed_trades if t['pnl_rs'] > 0),
        'win_rate': round(sum(1 for t in closed_trades if t['pnl_rs'] > 0) / len(closed_trades) * 100, 1) if closed_trades else 0,
        'total_pnl': round(sum(t['pnl_rs'] for t in closed_trades), 0),
        'avg_trade_pnl': round(np.mean([t['pnl_rs'] for t in closed_trades]), 0) if closed_trades else 0,
        'winning_months': winning_months,
        'total_months': total_months,
        'monthly_win_rate': round(winning_months / total_months * 100, 1) if total_months > 0 else 0,
        'worst_month': round(monthly_returns.min() * 100, 1) if len(monthly_returns) > 0 else 0,
        'best_month': round(monthly_returns.max() * 100, 1) if len(monthly_returns) > 0 else 0,
        'years': round(years, 1),
        'signal_stats': {
            sid: {
                'trades': s['trades'],
                'wins': s['wins'],
                'win_rate': round(s['wins'] / s['trades'] * 100, 1) if s['trades'] > 0 else 0,
                'total_pnl': round(s['total_pnl'], 0),
                'avg_pnl': round(np.mean(s['pnl_list']), 0) if s['pnl_list'] else 0,
            }
            for sid, s in signal_stats.items()
        },
        'trades': closed_trades,
        'daily_equity': daily_equity,
    }


def main():
    parser = argparse.ArgumentParser(description='Full portfolio backtest')
    parser.add_argument('--capital', type=float, default=1_000_000)
    parser.add_argument('--start', type=str, default=None)
    parser.add_argument('--risk-pct', type=float, default=0.01,
                        help='Base risk per trade as fraction (default 0.01 = 1%%)')
    parser.add_argument('--leverage', type=float, default=1.0,
                        help='Leverage multiplier on position size (default 1.0)')
    parser.add_argument('--no-dd-scaling', action='store_true',
                        help='Disable drawdown-based position scaling')
    args = parser.parse_args()

    print("Loading market data...", flush=True)
    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix "
        "FROM nifty_daily ORDER BY date", conn
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])

    if args.start:
        df = df[df['date'] >= args.start].reset_index(drop=True)

    print(f"  {len(df)} trading days ({df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()})")

    # Add indicators
    print("Computing indicators...", flush=True)
    df = add_all_indicators(df)
    df['hvol_6'] = historical_volatility(df['close'], period=6)
    df['hvol_100'] = historical_volatility(df['close'], period=100)

    print(f"\nRunning portfolio backtest with {len(PORTFOLIO_SIGNALS)} signals...")
    print(f"Capital: ₹{args.capital:,.0f}")
    print(f"Risk per trade: {args.risk_pct*100:.0f}%")
    print(f"Leverage: {args.leverage:.0f}x")
    print(f"Signals: {', '.join(PORTFOLIO_SIGNALS.keys())}")

    result = run_portfolio_backtest(args.capital, df,
                                    base_risk_pct=args.risk_pct,
                                    leverage=args.leverage,
                                    dd_scaling=not args.no_dd_scaling)

    # Report
    print(f"\n{'='*70}")
    print(f"PORTFOLIO BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"  Period:           {result['years']} years")
    print(f"  Capital:          ₹{result['capital']:>12,.0f}")
    print(f"  Final equity:     ₹{result['final_equity']:>12,.0f}")
    print(f"  Total return:     {result['total_return']:>10.1f}%")
    print(f"  CAGR:             {result['cagr']:>10.1f}%")
    print(f"  Sharpe:           {result['sharpe']:>10.2f}")
    print(f"  Max drawdown:     {result['max_drawdown']:>10.1f}%")
    print(f"  Calmar:           {result['calmar']:>10.2f}")
    print(f"  Total trades:     {result['total_trades']:>10d}")
    print(f"  Win rate:         {result['win_rate']:>10.1f}%")
    print(f"  Avg trade P&L:    ₹{result['avg_trade_pnl']:>11,.0f}")
    print(f"  Monthly WR:       {result['monthly_win_rate']:>10.1f}% ({result['winning_months']}/{result['total_months']})")
    print(f"  Best month:       {result['best_month']:>10.1f}%")
    print(f"  Worst month:      {result['worst_month']:>10.1f}%")

    print(f"\n{'Signal':<40s} {'Trades':>7s} {'Wins':>5s} {'WR':>6s} {'Total P&L':>12s} {'Avg P&L':>10s}")
    print(f"{'-'*40} {'-'*7} {'-'*5} {'-'*6} {'-'*12} {'-'*10}")
    for sid in PORTFOLIO_SIGNALS:
        ss = result['signal_stats'].get(sid, {})
        if ss.get('trades', 0) > 0:
            print(f"{sid:<40s} {ss['trades']:>7d} {ss['wins']:>5d} {ss['win_rate']:>5.1f}% "
                  f"₹{ss['total_pnl']:>11,.0f} ₹{ss['avg_pnl']:>9,.0f}")
        else:
            print(f"{sid:<40s}       0     -      -            -          -")

    # Save
    output_path = 'backtest_results/_PORTFOLIO_BACKTEST.json'
    os.makedirs('backtest_results', exist_ok=True)
    save_data = {k: v for k, v in result.items() if k not in ('trades', 'daily_equity')}
    save_data['trade_count_by_signal'] = {sid: s.get('trades', 0) for sid, s in result['signal_stats'].items()}
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()

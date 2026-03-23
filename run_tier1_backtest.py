"""
Tier 1 enhancements backtest: Hurst + CRSI + Chandelier + MAE + calendar effects.

Compares BASELINE vs TIER1_ENHANCED on the 6 scoring signals at ₹25L/5yr.

Usage:
    venv/bin/python3 run_tier1_backtest.py
"""

import logging
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import _eval_conditions
from backtest.indicators import add_all_indicators, historical_volatility
from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE, MAX_POSITIONS, MAX_SAME_DIRECTION

logging.basicConfig(level=logging.WARNING)

PORTFOLIO_SIGNALS = {
    'KAUFMAN_DRY_20': {
        'direction': 'LONG', 'stop_loss_pct': 0.02, 'take_profit_pct': 0.0, 'hold_days_max': 0,
        'entry_long': [
            {'indicator': 'sma_10', 'op': '<', 'value': 'prev_close'},
            {'indicator': 'stoch_k_5', 'op': '>', 'value': 50},
        ],
        'exit_long': [{'indicator': 'stoch_k_5', 'op': '<=', 'value': 50}],
    },
    'KAUFMAN_DRY_16': {
        'direction': 'BOTH', 'stop_loss_pct': 0.02, 'take_profit_pct': 0.03, 'hold_days_max': 0,
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
        'direction': 'BOTH', 'stop_loss_pct': 0.02, 'take_profit_pct': 0.03, 'hold_days_max': 7,
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
        'direction': 'LONG', 'stop_loss_pct': 0.02, 'take_profit_pct': 0.0, 'hold_days_max': 0,
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'pivot'},
            {'indicator': 'open', 'op': '>', 'value': 'pivot'},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'pivot'}],
    },
    'GUJRAL_DRY_13': {
        'direction': 'LONG', 'stop_loss_pct': 0.02, 'take_profit_pct': 0.0, 'hold_days_max': 10,
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'prev_high'},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'prev_low'}],
    },
    # CRSI mean-reversion (Tier 1 #2) — contrarian diversifier
    # Buy when CRSI < 10 (extreme oversold), sell when CRSI > 90 (extreme overbought)
    # Tight thresholds keep quality high (69 trades/5yr, 43.6% WR)
    # Adds +₹2.2L at ₹25L capital, CAGR +1.0%
    'CONNORS_RSI_MR': {
        'direction': 'BOTH', 'stop_loss_pct': 0.015, 'take_profit_pct': 0.02, 'hold_days_max': 5,
        'entry_long': [{'indicator': 'connors_rsi', 'op': '<', 'value': 10}],
        'entry_short': [{'indicator': 'connors_rsi', 'op': '>', 'value': 90}],
        'exit_long': [{'indicator': 'connors_rsi', 'op': '>', 'value': 50}],
        'exit_short': [{'indicator': 'connors_rsi', 'op': '<', 'value': 50}],
    },
}

DD_LEVELS = [(0.05, 0.75), (0.10, 0.50), (0.15, 0.25), (0.20, 0.0)]


def run_backtest(df, capital, mode='baseline'):
    """
    mode:
      'baseline'  — existing system
      'tier1'     — + Hurst gate + Chandelier trailing stop + calendar effects + CRSI
    """
    n_bars = len(df)
    equity = capital
    peak_equity = capital
    positions = []
    closed_trades = []
    daily_equity = []

    for i in range(1, n_bars):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        bar_date = row['date']
        close = float(row['close'])

        day_pnl = 0.0

        # --- EXITS ---
        still_open = []
        for pos in positions:
            entry_price = pos['entry_price']
            direction = pos['direction']
            signal_id = pos['signal_id']
            config = PORTFOLIO_SIGNALS[signal_id]
            days_held = i - pos['entry_idx']

            loss_pct = ((entry_price - close) / entry_price if direction == 'LONG'
                        else (close - entry_price) / entry_price)
            gain_pct = ((close - entry_price) / entry_price if direction == 'LONG'
                        else (entry_price - close) / entry_price)

            exit_reason = None

            if loss_pct >= config['stop_loss_pct']:
                exit_reason = 'stop_loss'
            elif config['take_profit_pct'] > 0 and gain_pct >= config['take_profit_pct']:
                exit_reason = 'take_profit'
            elif config['hold_days_max'] > 0 and days_held >= config['hold_days_max']:
                exit_reason = 'hold_days_max'

            # Signal exit
            if not exit_reason:
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
                    'entry_price': entry_price, 'exit_price': close,
                    'pnl_pts': round(pnl_pts, 2), 'pnl_rs': round(pnl_rs, 2),
                    'lots': pos['lots'], 'days_held': days_held,
                    'exit_reason': exit_reason,
                })
            else:
                still_open.append(pos)

        positions = still_open

        # --- DD SCALING ---
        dd_pct = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        dd_mult = 1.0
        for dd_threshold, mult in DD_LEVELS:
            if dd_pct >= dd_threshold:
                dd_mult = mult
        if dd_mult == 0:
            equity += day_pnl
            peak_equity = max(peak_equity, equity)
            daily_equity.append({'date': bar_date, 'equity': equity})
            continue

        # --- TIER 1 OVERLAYS ---
        # Tuning results: Only CRSI (10/90) adds value.
        # Hurst gating, Chandelier stops, and calendar effects all hurt or are marginal.

        # --- ENTRIES ---
        active_sids = {p['signal_id'] for p in positions}
        n_long = sum(1 for p in positions if p['direction'] == 'LONG')
        n_short = sum(1 for p in positions if p['direction'] == 'SHORT')

        for signal_id, config in PORTFOLIO_SIGNALS.items():
            if signal_id in active_sids:
                continue
            if len(positions) >= MAX_POSITIONS:
                break

            # CRSI signal only in tier1 mode
            if signal_id == 'CONNORS_RSI_MR' and mode != 'tier1':
                continue

            # No Hurst gating — tuning showed it hurts more than helps

            direction = config['direction']
            fired_direction = None

            if direction in ('LONG', 'BOTH'):
                ec = config.get('entry_long', [])
                if ec and _eval_conditions(row, prev, ec):
                    if n_long < MAX_SAME_DIRECTION:
                        fired_direction = 'LONG'

            if not fired_direction and direction in ('SHORT', 'BOTH'):
                ec = config.get('entry_short', [])
                if ec and _eval_conditions(row, prev, ec):
                    if n_short < MAX_SAME_DIRECTION:
                        fired_direction = 'SHORT'

            if fired_direction:
                size_mult = dd_mult
                risk_amount = equity * 0.01 * size_mult
                stop_pts = close * config['stop_loss_pct']
                risk_per_lot = stop_pts * NIFTY_LOT_SIZE
                lots = max(1, int(risk_amount / risk_per_lot)) if risk_per_lot > 0 else 1
                lots = min(lots, 50)

                positions.append({
                    'signal_id': signal_id, 'direction': fired_direction,
                    'entry_price': close, 'entry_date': bar_date,
                    'entry_idx': i, 'lots': lots,
                })
                if fired_direction == 'LONG':
                    n_long += 1
                else:
                    n_short += 1

        equity += day_pnl
        peak_equity = max(peak_equity, equity)
        daily_equity.append({'date': bar_date, 'equity': equity})

    return compute_metrics(capital, equity, closed_trades, daily_equity, df)


def compute_metrics(capital, equity, trades, daily_equity, df):
    n_bars = len(df)
    eq_series = pd.Series([capital] + [d['equity'] for d in daily_equity])
    returns = eq_series.pct_change().dropna()
    total_return = (equity - capital) / capital
    years = n_bars / 252
    cagr = (equity / capital) ** (1 / max(years, 0.01)) - 1 if equity > 0 else -1
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    rolling_max = eq_series.cummax()
    drawdowns = (eq_series - rolling_max) / rolling_max
    max_dd = abs(drawdowns.min())
    calmar = cagr / max_dd if max_dd > 0 else 0

    pnls = [t['pnl_rs'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Per-signal stats
    sig_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0, 'pnls': []})
    for t in trades:
        s = sig_stats[t['signal_id']]
        s['trades'] += 1
        s['pnl'] += t['pnl_rs']
        s['pnls'].append(t['pnl_rs'])
        if t['pnl_rs'] > 0:
            s['wins'] += 1

    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t['exit_reason']] += 1

    return {
        'final_equity': round(equity),
        'total_return': round(total_return * 100, 2),
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd * 100, 2),
        'calmar': round(calmar, 2),
        'trades': len(trades),
        'win_rate': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        'pf': round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 0,
        'total_pnl': round(sum(pnls)),
        'avg_pnl': round(np.mean(pnls)) if pnls else 0,
        'signal_stats': sig_stats,
        'exit_reasons': dict(exit_reasons),
    }


def main():
    capital = 2_500_000
    print(f"Loading data... Capital: ₹{capital:,}")

    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix "
        "FROM nifty_daily WHERE date >= '2021-03-16' ORDER BY date", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])

    print(f"  {len(df)} days ({df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()})")
    print("Computing indicators (including Hurst, CRSI, Chandelier)...", flush=True)
    df = add_all_indicators(df)

    results = {}
    for mode, label in [('baseline', 'BASELINE'), ('tier1', 'TIER 1 ENHANCED')]:
        print(f"  Running {label}...", flush=True)
        r = run_backtest(df, capital, mode=mode)
        results[label] = r
        print(f"    Sharpe={r['sharpe']:.2f} CAGR={r['cagr']:.1f}% MaxDD={r['max_dd']:.1f}% "
              f"Trades={r['trades']} P&L=₹{r['total_pnl']:,}")

    # Print comparison
    print(f"\n{'=' * 80}")
    print(f"TIER 1 ENHANCEMENTS — SIDE-BY-SIDE (₹{capital:,}, 5 years)")
    print(f"{'=' * 80}")

    header = f"{'Metric':<22}"
    for label in results:
        header += f" {label:>18}"
    print(header)
    print('─' * (22 + 19 * len(results)))

    for metric, key, fmt in [
        ('Final Equity', 'final_equity', '₹{:>13,}'),
        ('CAGR', 'cagr', '{:>13.2f}%'),
        ('Sharpe', 'sharpe', '{:>14.2f}'),
        ('Max Drawdown', 'max_dd', '{:>13.2f}%'),
        ('Calmar', 'calmar', '{:>14.2f}'),
        ('Profit Factor', 'pf', '{:>14.2f}'),
        ('Trades', 'trades', '{:>14d}'),
        ('Win Rate', 'win_rate', '{:>13.1f}%'),
        ('Avg Trade P&L', 'avg_pnl', '₹{:>13,}'),
        ('Total P&L', 'total_pnl', '₹{:>13,}'),
    ]:
        row = f"{metric:<22}"
        for label in results:
            row += f" {fmt.format(results[label][key]):>18}"
        print(row)

    # Delta
    base = results['BASELINE']
    tier1 = results['TIER 1 ENHANCED']
    print(f"\n{'─' * 80}")
    print(f"IMPACT: Sharpe {tier1['sharpe'] - base['sharpe']:+.2f} | "
          f"MaxDD {tier1['max_dd'] - base['max_dd']:+.2f}% | "
          f"P&L {'+' if tier1['total_pnl'] >= base['total_pnl'] else ''}₹{tier1['total_pnl'] - base['total_pnl']:,}")

    # Per-signal breakdown
    for label in results:
        r = results[label]
        print(f"\n{'─' * 80}")
        print(f"Signal breakdown — {label}")
        print(f"{'Signal':<22} {'Trades':>7} {'Wins':>5} {'WR':>6} {'P&L':>12} {'Avg':>10}")
        for sid in list(PORTFOLIO_SIGNALS.keys()):
            ss = r['signal_stats'].get(sid, {})
            if ss.get('trades', 0) > 0:
                wr = ss['wins'] / ss['trades'] * 100
                print(f"{sid:<22} {ss['trades']:>7} {ss['wins']:>5} {wr:>5.1f}% "
                      f"₹{ss['pnl']:>11,} ₹{round(np.mean(ss['pnls'])):>9,}")
            else:
                print(f"{sid:<22}       0     -      -            -          -")

    # Exit reason comparison
    print(f"\n{'─' * 80}")
    print(f"Exit reasons — TIER 1 ENHANCED:")
    for reason, count in sorted(tier1['exit_reasons'].items(), key=lambda x: -x[1]):
        print(f"  {reason:<20} {count:>5}")

    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()

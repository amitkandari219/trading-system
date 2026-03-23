"""
Forward-test the intraday framework signals over synthetic 5-min bars.

Runs ORB_15MIN, INST_FLOW_WINDOW, and EXPIRY_GAMMA over the past year
of synthetic intraday data generated from daily OHLCV.

Usage:
    venv/bin/python3 -m backtest.intraday_forward_test
    venv/bin/python3 -m backtest.intraday_forward_test --start 2025-06-01
"""

import argparse
import logging
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import psycopg2

from data.intraday_loader import generate_5min_bars
from signals.intraday_framework import (
    ExpiryDayGamma,
    InstitutionalFlowWindow,
    IntradayBar,
    IntradayEngine,
)

logger = logging.getLogger(__name__)

NIFTY_LOT_SIZE = 25
SLIPPAGE_PCT = 0.0005  # 0.05% each way
CAPITAL = 1_000_000


def is_expiry_day(d):
    """Weekly Nifty expiry is Thursday (weekday 3)."""
    return d.weekday() == 3


def load_daily_data(conn, start_date):
    """Load daily OHLCV + indicators for the test period."""
    df = pd.read_sql(
        f"""SELECT date, open, high, low, close, volume, india_vix
            FROM nifty_daily
            WHERE date >= '{start_date}'
            ORDER BY date""",
        conn,
    )
    df['date'] = pd.to_datetime(df['date'])

    # Compute daily ATR(14) and ADX(14) inline
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # True Range
    tr = np.zeros(len(df))
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(df)):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    # ATR(14)
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    df['atr_14'] = atr

    # ADX(14) — simplified using DM+/DM- smoothing
    dm_plus = np.zeros(len(df))
    dm_minus = np.zeros(len(df))
    for i in range(1, len(df)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        dm_plus[i] = up if (up > down and up > 0) else 0
        dm_minus[i] = down if (down > up and down > 0) else 0

    smooth_tr = pd.Series(tr).rolling(14, min_periods=1).sum().values
    smooth_dmp = pd.Series(dm_plus).rolling(14, min_periods=1).sum().values
    smooth_dmn = pd.Series(dm_minus).rolling(14, min_periods=1).sum().values

    di_plus = 100 * smooth_dmp / np.where(smooth_tr > 0, smooth_tr, 1)
    di_minus = 100 * smooth_dmn / np.where(smooth_tr > 0, smooth_tr, 1)
    di_sum = di_plus + di_minus
    dx = 100 * np.abs(di_plus - di_minus) / np.where(di_sum > 0, di_sum, 1)
    df['adx_14'] = pd.Series(dx).rolling(14, min_periods=1).mean().values

    return df


def run_forward_test(conn, start_date='2025-03-21'):
    """Run the intraday forward test using real 5-min data from DB."""
    df_daily = load_daily_data(conn, start_date)
    print(f"Loaded {len(df_daily)} trading days ({df_daily['date'].min().date()} to {df_daily['date'].max().date()})")

    # Load real 5-min data from DB
    df_intraday = pd.read_sql(
        f"SELECT datetime, open, high, low, close, volume FROM nifty_intraday "
        f"WHERE timeframe='5min' AND datetime >= '{start_date}' ORDER BY datetime",
        conn,
    )
    df_intraday['datetime'] = pd.to_datetime(df_intraday['datetime'])
    df_intraday['date'] = df_intraday['datetime'].dt.date
    intraday_by_date = {d: group for d, group in df_intraday.groupby('date')}

    n_real_days = len(intraday_by_date)
    print(f"Loaded {len(df_intraday)} real 5-min bars ({n_real_days} trading days)")

    engine = IntradayEngine()
    rng = np.random.default_rng(42)

    all_trades = []
    daily_pnl = {}

    for i, daily_row in df_daily.iterrows():
        trading_date = daily_row['date'].date()
        daily_atr = float(daily_row['atr_14'])
        daily_vix = float(daily_row['india_vix']) if pd.notna(daily_row.get('india_vix')) else 15.0
        daily_adx = float(daily_row['adx_14']) if pd.notna(daily_row.get('adx_14')) else 20.0

        # Use real 5-min bars from DB, fall back to synthetic if not available
        if trading_date in intraday_by_date:
            day_bars = intraday_by_date[trading_date]
            bars_data = [
                {'datetime': row['datetime'], 'open': row['open'], 'high': row['high'],
                 'low': row['low'], 'close': row['close'], 'volume': int(row['volume'] or 0)}
                for _, row in day_bars.iterrows()
            ]
        else:
            bars_data = generate_5min_bars(daily_row, rng)

        # Reset engine for new day
        engine.on_new_day(trading_date)

        # Build context
        context = {
            'daily_atr': daily_atr,
            'daily_vix': daily_vix,
            'daily_regime': 'NORMAL',
            'daily_adx': daily_adx,
            'bars_today': [],
            'is_expiry': is_expiry_day(trading_date),
            'day_of_week': trading_date.weekday(),
            'max_pain_strike': _estimate_max_pain(float(daily_row['close'])),
        }

        # Process each 5-min bar
        session_trades = []
        for bar_dict in bars_data:
            bar = IntradayBar(
                timestamp=bar_dict['datetime'],
                open=bar_dict['open'],
                high=bar_dict['high'],
                low=bar_dict['low'],
                close=bar_dict['close'],
                volume=bar_dict['volume'],
            )
            context['bars_today'].append(bar)

            results = engine.process_bar(bar, context)
            for r in results:
                session_trades.append({
                    'date': trading_date,
                    'signal_id': r.signal_id,
                    'action': r.action,
                    'price': r.price,
                    'stop_loss': r.stop_loss,
                    'take_profit': r.take_profit,
                    'reason': r.reason,
                    'timestamp': bar.timestamp,
                })

        # Pair entries with exits to compute PnL
        day_trades = _pair_trades(session_trades, bars_data)
        all_trades.extend(day_trades)

        day_pnl = sum(t['pnl_rs'] for t in day_trades)
        if day_pnl != 0:
            daily_pnl[trading_date] = day_pnl

    return all_trades, daily_pnl, df_daily


def _estimate_max_pain(nifty_close):
    """Estimate max pain strike — nearest 50-point strike."""
    return round(nifty_close / 50) * 50


def _pair_trades(session_events, bars_data):
    """Pair entry/exit events into complete trades with PnL."""
    trades = []
    open_positions = {}  # signal_id -> entry event

    for event in session_events:
        sig = event['signal_id']

        if event['action'] in ('ENTER_LONG', 'ENTER_SHORT'):
            direction = 'LONG' if event['action'] == 'ENTER_LONG' else 'SHORT'
            # Apply entry slippage
            if direction == 'LONG':
                entry_price = event['price'] * (1 + SLIPPAGE_PCT)
            else:
                entry_price = event['price'] * (1 - SLIPPAGE_PCT)

            open_positions[sig] = {
                'direction': direction,
                'entry_price': entry_price,
                'entry_time': event['timestamp'],
                'stop_loss': event['stop_loss'],
                'take_profit': event['take_profit'],
                'entry_reason': event['reason'],
            }

        elif event['action'] == 'EXIT' and sig in open_positions:
            pos = open_positions.pop(sig)
            direction = pos['direction']

            # Apply exit slippage
            if direction == 'LONG':
                exit_price = event['price'] * (1 - SLIPPAGE_PCT)
                pnl_pts = exit_price - pos['entry_price']
            else:
                exit_price = event['price'] * (1 + SLIPPAGE_PCT)
                pnl_pts = pos['entry_price'] - exit_price

            pnl_rs = pnl_pts * NIFTY_LOT_SIZE

            trades.append({
                'date': event['date'],
                'signal_id': sig,
                'direction': direction,
                'entry_time': pos['entry_time'],
                'exit_time': event['timestamp'],
                'entry_price': round(pos['entry_price'], 2),
                'exit_price': round(exit_price, 2),
                'pnl_pts': round(pnl_pts, 2),
                'pnl_rs': round(pnl_rs, 2),
                'exit_reason': event['reason'],
                'entry_reason': pos['entry_reason'],
            })

    # Force-close any remaining open positions at session end
    if open_positions and bars_data:
        last_bar = bars_data[-1]
        last_close = last_bar['close']
        last_time = last_bar['datetime']

        for sig, pos in open_positions.items():
            direction = pos['direction']
            if direction == 'LONG':
                exit_price = last_close * (1 - SLIPPAGE_PCT)
                pnl_pts = exit_price - pos['entry_price']
            else:
                exit_price = last_close * (1 + SLIPPAGE_PCT)
                pnl_pts = pos['entry_price'] - exit_price

            pnl_rs = pnl_pts * NIFTY_LOT_SIZE
            trades.append({
                'date': last_bar['datetime'].date() if isinstance(last_bar['datetime'], datetime) else last_bar['datetime'],
                'signal_id': sig,
                'direction': direction,
                'entry_time': pos['entry_time'],
                'exit_time': last_time,
                'entry_price': round(pos['entry_price'], 2),
                'exit_price': round(exit_price, 2),
                'pnl_pts': round(pnl_pts, 2),
                'pnl_rs': round(pnl_rs, 2),
                'exit_reason': 'forced_session_end',
                'entry_reason': pos['entry_reason'],
            })

    return trades


def print_report(all_trades, daily_pnl, df_daily):
    """Print comprehensive forward test results."""
    if not all_trades:
        print("\nNo trades generated.")
        return

    df_trades = pd.DataFrame(all_trades)
    n_days = len(df_daily)
    n_trade_days = len(daily_pnl)

    print(f"\n{'=' * 70}")
    print(f"INTRADAY FORWARD TEST RESULTS")
    print(f"{'=' * 70}")
    print(f"Period: {df_daily['date'].min().date()} to {df_daily['date'].max().date()} ({n_days} days)")
    print(f"Trading days with trades: {n_trade_days}")
    print(f"Total trades: {len(all_trades)}")

    # Per-signal breakdown
    print(f"\n{'─' * 70}")
    print(f"{'Signal':<20} {'Trades':>6} {'Win%':>6} {'PF':>6} {'Avg PnL':>10} {'Total PnL':>12} {'Max DD':>10}")
    print(f"{'─' * 70}")

    for sig_id in sorted(df_trades['signal_id'].unique()):
        sig_trades = df_trades[df_trades['signal_id'] == sig_id]
        pnls = sig_trades['pnl_rs'].values
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        win_rate = len(wins) / len(pnls) * 100 if len(pnls) > 0 else 0
        pf = sum(wins) / abs(sum(losses)) if len(losses) > 0 and sum(losses) != 0 else float('inf')
        avg_pnl = np.mean(pnls)
        total_pnl = np.sum(pnls)

        # Max drawdown
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        max_dd = abs(dd.min()) if len(dd) > 0 else 0

        print(f"{sig_id:<20} {len(pnls):>6} {win_rate:>5.1f}% {pf:>6.2f} {avg_pnl:>9.0f}₹ {total_pnl:>11,.0f}₹ {max_dd:>9,.0f}₹")

    # Portfolio summary
    total_pnl = sum(t['pnl_rs'] for t in all_trades)
    all_pnls = [t['pnl_rs'] for t in all_trades]
    wins = [p for p in all_pnls if p > 0]
    losses = [p for p in all_pnls if p <= 0]

    print(f"{'─' * 70}")
    print(f"{'PORTFOLIO':<20} {len(all_pnls):>6} {len(wins)/len(all_pnls)*100:>5.1f}% "
          f"{sum(wins)/abs(sum(losses)) if losses and sum(losses) != 0 else 0:>6.2f} "
          f"{np.mean(all_pnls):>9.0f}₹ {total_pnl:>11,.0f}₹")

    # Daily Sharpe
    if daily_pnl:
        daily_rets = pd.Series(daily_pnl) / CAPITAL
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
        annual_ret = daily_rets.mean() * 252
        cum_pnl = np.cumsum(list(daily_pnl.values()))
        peak = np.maximum.accumulate(cum_pnl)
        max_dd_port = abs((cum_pnl - peak).min())
        calmar = (annual_ret * CAPITAL) / max_dd_port if max_dd_port > 0 else 0

        print(f"\n{'─' * 70}")
        print(f"Portfolio Metrics:")
        print(f"  Sharpe Ratio:     {sharpe:.2f}")
        print(f"  Annual Return:    {annual_ret*100:.2f}%")
        print(f"  Max Drawdown:     ₹{max_dd_port:,.0f} ({max_dd_port/CAPITAL*100:.2f}%)")
        print(f"  Calmar Ratio:     {calmar:.2f}")
        print(f"  Total P&L:        ₹{total_pnl:,.0f} ({total_pnl/CAPITAL*100:.2f}%)")

    # Exit reason breakdown
    print(f"\n{'─' * 70}")
    print("Exit reasons:")
    for reason in df_trades['exit_reason'].value_counts().items():
        print(f"  {reason[0]:<30} {reason[1]:>5}")

    # Monthly breakdown
    print(f"\n{'─' * 70}")
    print("Monthly P&L:")
    df_trades['month'] = pd.to_datetime(df_trades['date']).dt.to_period('M')
    monthly = df_trades.groupby('month')['pnl_rs'].agg(['sum', 'count'])
    for month, row in monthly.iterrows():
        bar = '█' * max(1, int(abs(row['sum']) / 1000))
        sign = '+' if row['sum'] >= 0 else '-'
        color = '' if row['sum'] >= 0 else ''
        print(f"  {month}  {sign}₹{abs(row['sum']):>8,.0f}  ({int(row['count']):>3} trades)  {bar}")

    # Sample trades
    print(f"\n{'─' * 70}")
    print("Last 10 trades:")
    for t in all_trades[-10:]:
        print(f"  {t['date']} {t['signal_id']:<20} {t['direction']:<5} "
              f"entry={t['entry_price']:.0f} exit={t['exit_price']:.0f} "
              f"pnl={t['pnl_rs']:>+8.0f}₹  [{t['exit_reason']}]")

    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description='Intraday forward test')
    parser.add_argument('--start', type=str, default='2025-03-21',
                        help='Start date (default: 2025-03-21)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    conn = psycopg2.connect('postgresql://trader:trader123@localhost:5450/trading')
    try:
        all_trades, daily_pnl, df_daily = run_forward_test(conn, args.start)
        print_report(all_trades, daily_pnl, df_daily)
    finally:
        conn.close()


if __name__ == '__main__':
    main()

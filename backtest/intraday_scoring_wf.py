"""
Walk-Forward Backtest for 5 Intraday Scoring Signals on real 5-min Kite data.

Signals: ORB, VWAP Crossover, Momentum Candles, GIFT Gap, RSI Divergence
Data: 495 trading days (Mar 2024 - Mar 2026), 36,945 real 5-min bars
WF: 6mo train / 2mo test / 1mo step
Costs: 0.05% slippage + 0.0125% STT + ₹40/lot brokerage per side

Usage:
    venv/bin/python3 -m backtest.intraday_scoring_wf
"""

import logging
import math
import time as time_mod
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2

from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE

logger = logging.getLogger(__name__)

# Costs
SLIPPAGE_PER_SIDE = 1.0  # 1 Nifty point per side
STT_SELL_PCT = 0.000125
BROKERAGE_PER_LOT = 40

# WF params
TRAIN_MONTHS = 6
TEST_MONTHS = 2
STEP_MONTHS = 1

# Pass criteria
MIN_SHARPE = 0.8
MIN_PF = 1.3
MIN_WF_PASS = 0.60

# Force exit
FORCE_EXIT_TIME = time(15, 20)
MAX_TRADES_PER_DAY = 3


# ================================================================
# DATA LOADING
# ================================================================

def load_bars() -> pd.DataFrame:
    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM intraday_bars WHERE instrument='NIFTY' ORDER BY timestamp",
        conn, parse_dates=['timestamp'])
    conn.close()
    df['date'] = df['timestamp'].dt.date
    df['time'] = df['timestamp'].dt.time
    return df


def load_daily() -> pd.DataFrame:
    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql(
        "SELECT date, close as prev_close, india_vix FROM nifty_daily ORDER BY date",
        conn, parse_dates=['date'])
    conn.close()
    df['date'] = df['date'].dt.date
    return df


# ================================================================
# INDICATOR COMPUTATION (per session)
# ================================================================

def add_session_indicators(session: pd.DataFrame) -> pd.DataFrame:
    """Add indicators needed by the 5 signals to a single session."""
    df = session.copy()
    n = len(df)
    if n < 5:
        return df

    closes = df['close'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    opens = df['open'].values.astype(float)
    volumes = df['volume'].values.astype(float)

    # ── Opening Range (first 3 bars = 15 min) ──
    or_high = highs[:3].max()
    or_low = lows[:3].min()
    df['or_high'] = or_high
    df['or_low'] = or_low
    df['or_range_pct'] = (or_high - or_low) / or_low if or_low > 0 else 0

    # ── VWAP ──
    cum_vol = np.cumsum(volumes)
    cum_tp_vol = np.cumsum(closes * volumes)
    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, closes)
    df['vwap'] = vwap

    # VWAP bands (rolling std dev)
    vwap_dev = np.zeros(n)
    for i in range(1, n):
        diffs = (closes[:i+1] - vwap[:i+1]) ** 2
        vols = volumes[:i+1]
        if vols.sum() > 0:
            vwap_dev[i] = math.sqrt(np.sum(diffs * vols) / vols.sum())
    df['vwap_upper_1s'] = vwap + vwap_dev
    df['vwap_lower_1s'] = vwap - vwap_dev
    df['vwap_upper_2s'] = vwap + 2 * vwap_dev
    df['vwap_lower_2s'] = vwap - 2 * vwap_dev

    # ── ATR (20-bar) ──
    tr = np.maximum(highs - lows,
                    np.maximum(np.abs(highs - np.roll(closes, 1)),
                               np.abs(lows - np.roll(closes, 1))))
    tr[0] = highs[0] - lows[0]
    atr = pd.Series(tr).rolling(20, min_periods=5).mean().values
    df['atr_20'] = atr

    # ── RSI (14) ──
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - 100 / (1 + rs)
    df['rsi_14'] = rsi

    # ── Body pct ──
    bar_range = highs - lows
    body = np.abs(closes - opens)
    df['body_pct'] = np.where(bar_range > 0, body / bar_range, 0)

    # ── Bar direction ──
    df['bar_dir'] = np.where(closes > opens, 1, np.where(closes < opens, -1, 0))

    return df


# ================================================================
# SIGNAL EVALUATION
# ================================================================

def evaluate_orb(bar_idx: int, df: pd.DataFrame, context: dict) -> Optional[dict]:
    """ORB breakout signal."""
    if bar_idx < 3:
        return None

    bar = df.iloc[bar_idx]
    t = bar['time']
    if t < time(9, 30) or t > time(14, 0):
        return None

    c = float(bar['close'])
    or_high = float(bar['or_high'])
    or_low = float(bar['or_low'])
    or_range_pct = float(bar['or_range_pct'])

    # Range filter
    if or_range_pct < 0.003 or or_range_pct > 0.012:
        return None

    prev_close = float(df.iloc[bar_idx - 1]['close'])

    if c > or_high and prev_close <= or_high:
        return {
            'signal_id': 'INTRA_ORB', 'direction': 'LONG', 'price': c,
            'sl': or_low, 'tgt': c + 1.5 * (or_high - or_low),
        }
    if c < or_low and prev_close >= or_low:
        return {
            'signal_id': 'INTRA_ORB', 'direction': 'SHORT', 'price': c,
            'sl': or_high, 'tgt': c - 1.5 * (or_high - or_low),
        }
    return None


def evaluate_vwap(bar_idx: int, df: pd.DataFrame, context: dict) -> Optional[dict]:
    """VWAP crossover signal."""
    if bar_idx < 10:
        return None

    bar = df.iloc[bar_idx]
    t = bar['time']
    if t < time(9, 45) or t > time(14, 30):
        return None

    c = float(bar['close'])
    vwap = float(bar['vwap'])
    prev_close = float(df.iloc[bar_idx - 1]['close'])
    prev_vwap = float(df.iloc[bar_idx - 1]['vwap'])
    atr = float(bar['atr_20']) if not np.isnan(bar['atr_20']) else c * 0.005

    # Crossover
    if prev_close < prev_vwap and c > vwap:
        return {
            'signal_id': 'INTRA_VWAP', 'direction': 'LONG', 'price': c,
            'sl': c - 1.5 * atr, 'tgt': c + 2.0 * atr,
        }
    if prev_close > prev_vwap and c < vwap:
        return {
            'signal_id': 'INTRA_VWAP', 'direction': 'SHORT', 'price': c,
            'sl': c + 1.5 * atr, 'tgt': c - 2.0 * atr,
        }

    # Mean reversion at 2-sigma bands
    upper_2s = float(bar['vwap_upper_2s'])
    lower_2s = float(bar['vwap_lower_2s'])
    if c <= lower_2s and lower_2s > 0:
        return {
            'signal_id': 'INTRA_VWAP_MR', 'direction': 'LONG', 'price': c,
            'sl': c - atr, 'tgt': vwap,
        }
    if c >= upper_2s and upper_2s > 0:
        return {
            'signal_id': 'INTRA_VWAP_MR', 'direction': 'SHORT', 'price': c,
            'sl': c + atr, 'tgt': vwap,
        }
    return None


def evaluate_momentum(bar_idx: int, df: pd.DataFrame, context: dict) -> Optional[dict]:
    """Momentum candle signal."""
    if bar_idx < 20:
        return None

    bar = df.iloc[bar_idx]
    t = bar['time']
    if t < time(10, 0) or t > time(14, 30):
        return None

    c = float(bar['close'])
    o = float(bar['open'])
    h = float(bar['high'])
    l = float(bar['low'])
    body = abs(c - o)
    atr = float(bar['atr_20']) if not np.isnan(bar['atr_20']) else (h - l)

    if atr <= 0:
        return None

    # Wide-range bar: body > 2x ATR
    if body > 2 * atr:
        direction = 'LONG' if c > o else 'SHORT'
        sl = l if direction == 'LONG' else h
        risk = abs(c - sl)
        return {
            'signal_id': 'INTRA_MOMENTUM', 'direction': direction, 'price': c,
            'sl': sl, 'tgt': c + 2 * risk * (1 if direction == 'LONG' else -1),
        }

    # 3-bar momentum
    if bar_idx >= 3:
        dirs = [int(df.iloc[bar_idx - j]['bar_dir']) for j in range(3)]
        if all(d == 1 for d in dirs):
            return {
                'signal_id': 'INTRA_MOMENTUM_3BAR', 'direction': 'LONG', 'price': c,
                'sl': c - 1.5 * atr, 'tgt': c + 2.0 * atr,
            }
        if all(d == -1 for d in dirs):
            return {
                'signal_id': 'INTRA_MOMENTUM_3BAR', 'direction': 'SHORT', 'price': c,
                'sl': c + 1.5 * atr, 'tgt': c - 2.0 * atr,
            }
    return None


def evaluate_gift_gap(bar_idx: int, df: pd.DataFrame, context: dict) -> Optional[dict]:
    """GIFT gap signal — first bar only."""
    if bar_idx != 0:
        return None

    prev_close = context.get('prev_close', 0)
    if prev_close <= 0:
        return None

    bar = df.iloc[0]
    day_open = float(bar['open'])
    gap_pct = (day_open - prev_close) / prev_close

    if abs(gap_pct) < 0.005:
        return None  # < 0.5% gap — skip

    c = float(bar['close'])

    if 0.005 < abs(gap_pct) < 0.015:
        # Fade the gap
        direction = 'SHORT' if gap_pct > 0 else 'LONG'
        sl_dist = abs(gap_pct) * prev_close * 0.5
        tgt_dist = abs(gap_pct) * prev_close * 0.6
        return {
            'signal_id': 'INTRA_GAP_FADE', 'direction': direction, 'price': c,
            'sl': c + sl_dist * (1 if direction == 'SHORT' else -1),
            'tgt': c - tgt_dist * (1 if direction == 'SHORT' else -1),
        }
    elif abs(gap_pct) >= 0.015:
        # Follow the gap
        direction = 'LONG' if gap_pct > 0 else 'SHORT'
        sl_dist = abs(gap_pct) * prev_close * 0.3
        tgt_dist = abs(gap_pct) * prev_close * 0.5
        return {
            'signal_id': 'INTRA_GAP_FOLLOW', 'direction': direction, 'price': c,
            'sl': c - sl_dist * (1 if direction == 'LONG' else -1),
            'tgt': c + tgt_dist * (1 if direction == 'LONG' else -1),
        }
    return None


def evaluate_rsi_div(bar_idx: int, df: pd.DataFrame, context: dict) -> Optional[dict]:
    """RSI divergence signal."""
    if bar_idx < 30:
        return None

    bar = df.iloc[bar_idx]
    t = bar['time']
    if t < time(10, 0) or t > time(14, 30):
        return None

    c = float(bar['close'])
    rsi_now = float(bar['rsi_14'])
    atr = float(bar['atr_20']) if not np.isnan(bar['atr_20']) else c * 0.005
    if np.isnan(rsi_now) or atr <= 0:
        return None

    # Find last 2 swing lows (for bullish div)
    lows = df['low'].values[max(0, bar_idx-20):bar_idx+1].astype(float)
    rsis = df['rsi_14'].values[max(0, bar_idx-20):bar_idx+1].astype(float)

    if len(lows) < 10:
        return None

    # Simple divergence: compare current bar vs bar 10 ago
    lookback = 10
    prev_low = float(lows[-lookback])
    prev_rsi = float(rsis[-lookback])
    curr_low = float(lows[-1])
    curr_rsi = float(rsis[-1])

    # Bullish: price lower low, RSI higher low
    if curr_low < prev_low and curr_rsi > prev_rsi and curr_rsi < 40:
        return {
            'signal_id': 'INTRA_RSI_DIV', 'direction': 'LONG', 'price': c,
            'sl': c - 1.5 * atr, 'tgt': c + 2.0 * atr,
        }

    # Bearish: price higher high, RSI lower high
    highs = df['high'].values[max(0, bar_idx-20):bar_idx+1].astype(float)
    prev_high = float(highs[-lookback])
    curr_high = float(highs[-1])
    if curr_high > prev_high and curr_rsi < prev_rsi and curr_rsi > 60:
        return {
            'signal_id': 'INTRA_RSI_DIV', 'direction': 'SHORT', 'price': c,
            'sl': c + 1.5 * atr, 'tgt': c - 2.0 * atr,
        }
    return None


SIGNAL_EVALUATORS = {
    'INTRA_ORB': evaluate_orb,
    'INTRA_VWAP': evaluate_vwap,
    'INTRA_MOMENTUM': evaluate_momentum,
    'INTRA_GAP': evaluate_gift_gap,
    'INTRA_RSI_DIV': evaluate_rsi_div,
}


# ================================================================
# TRADE SIMULATION
# ================================================================

def simulate_day(session_df: pd.DataFrame, daily_ctx: dict) -> List[dict]:
    """Simulate all 5 signals on one trading day."""
    if len(session_df) < 5:
        return []

    df = add_session_indicators(session_df)
    trades = []
    open_pos = {}  # sig_id -> position
    day_trade_count = 0

    for i in range(len(df)):
        bar = df.iloc[i]
        t = bar['time']
        c = float(bar['close'])

        # ── Check exits ──
        for sig_id, pos in list(open_pos.items()):
            exit_reason = None
            exit_price = c

            if pos['direction'] == 'LONG':
                if c <= pos['sl']:
                    exit_reason, exit_price = 'SL', pos['sl']
                elif c >= pos['tgt']:
                    exit_reason, exit_price = 'TGT', pos['tgt']
            else:
                if c >= pos['sl']:
                    exit_reason, exit_price = 'SL', pos['sl']
                elif c <= pos['tgt']:
                    exit_reason, exit_price = 'TGT', pos['tgt']

            # Force exit at 15:20
            if t >= FORCE_EXIT_TIME:
                exit_reason = 'TIME'

            if exit_reason:
                pnl_pts = (exit_price - pos['entry']) if pos['direction'] == 'LONG' else (pos['entry'] - exit_price)
                pnl_pts -= 2 * SLIPPAGE_PER_SIDE  # round-trip slippage
                pnl_rs = pnl_pts * NIFTY_LOT_SIZE * 1  # 1 lot
                costs = BROKERAGE_PER_LOT * 2 + abs(exit_price * NIFTY_LOT_SIZE * STT_SELL_PCT)
                net_pnl = pnl_rs - costs

                trades.append({
                    'signal_id': sig_id,
                    'date': bar['date'],
                    'entry_time': pos['entry_time'],
                    'exit_time': t,
                    'direction': pos['direction'],
                    'entry_price': pos['entry'],
                    'exit_price': exit_price,
                    'pnl_pts': round(pnl_pts, 1),
                    'pnl_rs': round(net_pnl),
                    'pnl_pct': round(net_pnl / 1_000_000 * 100, 4),  # vs 10L capital
                    'exit_reason': exit_reason,
                })
                del open_pos[sig_id]

        # ── Check entries (max 3/day, not after 15:00) ──
        if day_trade_count >= MAX_TRADES_PER_DAY or t >= time(15, 0):
            continue

        for sig_name, evaluator in SIGNAL_EVALUATORS.items():
            if sig_name in open_pos:
                continue

            signal = evaluator(i, df, daily_ctx)
            if signal and signal.get('direction'):
                open_pos[signal['signal_id']] = {
                    'direction': signal['direction'],
                    'entry': signal['price'],
                    'sl': signal['sl'],
                    'tgt': signal['tgt'],
                    'entry_time': t,
                }
                day_trade_count += 1
                if day_trade_count >= MAX_TRADES_PER_DAY:
                    break

    return trades


# ================================================================
# WF ENGINE
# ================================================================

def run_wf():
    t0 = time_mod.perf_counter()
    print("=" * 85)
    print("  INTRADAY SCORING SIGNALS — WALK-FORWARD BACKTEST")
    print("  5 signals: ORB, VWAP, Momentum, GIFT Gap, RSI Divergence")
    print("  Data: 495 days real Kite 5-min bars (Mar 2024 - Mar 2026)")
    print("  WF: 6mo train / 2mo test / 1mo step")
    print("  Costs: 1pt slippage/side + STT + ₹40/lot brokerage")
    print("=" * 85)

    bars = load_bars()
    daily = load_daily()
    print(f"\nLoaded {len(bars)} bars, {len(bars['date'].unique())} trading days")

    # Build daily context
    daily_ctx_map = {}
    prev_close_map = {}
    for _, row in daily.iterrows():
        d = row['date']
        daily_ctx_map[d] = {
            'prev_close': float(row['prev_close']) if pd.notna(row['prev_close']) else 0,
            'vix': float(row['india_vix']) if pd.notna(row['india_vix']) else 15,
        }
    # Shift prev_close by 1 day
    daily_dates = sorted(daily_ctx_map.keys())
    for i in range(1, len(daily_dates)):
        daily_ctx_map[daily_dates[i]]['prev_close'] = daily_ctx_map[daily_dates[i-1]].get('prev_close', 0)

    # Simulate all trades
    print("Simulating trades across all days...")
    all_trades = []
    trade_dates = sorted(bars['date'].unique())

    for d in trade_dates:
        session = bars[bars['date'] == d].copy()
        ctx = daily_ctx_map.get(d, {'prev_close': 0, 'vix': 15})
        day_trades = simulate_day(session, ctx)
        all_trades.extend(day_trades)

    print(f"Total trades: {len(all_trades)}")

    # Per-signal summary
    print(f"\n{'Signal':<20s} {'Trades':>6s} {'WR':>6s} {'PF':>6s} {'Avg PnL':>10s} {'Total':>10s}")
    print("─" * 60)

    sig_stats = {}
    for sig_id in sorted(set(t['signal_id'] for t in all_trades)):
        sig_trades = [t for t in all_trades if t['signal_id'] == sig_id]
        wins = [t for t in sig_trades if t['pnl_rs'] > 0]
        losses = [t for t in sig_trades if t['pnl_rs'] <= 0]
        wr = len(wins) / max(len(sig_trades), 1)
        pf = sum(t['pnl_rs'] for t in wins) / abs(sum(t['pnl_rs'] for t in losses)) if losses and sum(t['pnl_rs'] for t in losses) != 0 else 0
        avg_pnl = np.mean([t['pnl_rs'] for t in sig_trades])
        total = sum(t['pnl_rs'] for t in sig_trades)
        print(f"  {sig_id:<18s} {len(sig_trades):>6d} {wr:>5.0%} {pf:>5.2f} ₹{avg_pnl:>8,.0f} ₹{total:>8,.0f}")
        sig_stats[sig_id] = {'trades': len(sig_trades), 'wr': wr, 'pf': pf, 'total': total}

    # WF windows
    print(f"\n{'─' * 85}")
    print("  WALK-FORWARD WINDOWS")
    print(f"{'─' * 85}")

    min_d = min(trade_dates)
    max_d = max(trade_dates)
    windows = []
    start = min_d
    while True:
        train_end = start + timedelta(days=TRAIN_MONTHS * 30)
        test_start = train_end
        test_end = test_start + timedelta(days=TEST_MONTHS * 30)
        if test_end > max_d:
            break
        windows.append((start, train_end, test_start, test_end))
        start = start + timedelta(days=STEP_MONTHS * 30)

    print(f"  {len(windows)} windows generated")
    passes = 0

    for i, (_, _, ts, te) in enumerate(windows):
        w_trades = [t for t in all_trades if ts <= t['date'] <= te]
        if len(w_trades) < 5:
            continue

        pnls = [t['pnl_rs'] for t in w_trades]
        wins_r = [p for p in pnls if p > 0]
        losses_r = [p for p in pnls if p <= 0]
        wr = len(wins_r) / len(pnls)
        pf = sum(wins_r) / abs(sum(losses_r)) if losses_r and sum(losses_r) != 0 else 0

        daily_pnl = {}
        for t in w_trades:
            d = str(t['date'])
            daily_pnl[d] = daily_pnl.get(d, 0) + t['pnl_rs']
        rets = np.array(list(daily_pnl.values())) / 1_000_000
        sharpe = (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0

        passed = sharpe >= MIN_SHARPE and pf >= MIN_PF
        if passed:
            passes += 1
        marker = " *" if passed else ""
        print(f"  W{i:2d} {ts}–{te} {len(w_trades):4d}tr Sharpe {sharpe:5.2f} PF {pf:5.2f} WR {wr:.0%}{marker}")

    wf_rate = passes / max(len(windows), 1)

    # Overall metrics
    total_pnl = sum(t['pnl_rs'] for t in all_trades)
    total_wins = sum(t['pnl_rs'] for t in all_trades if t['pnl_rs'] > 0)
    total_losses = abs(sum(t['pnl_rs'] for t in all_trades if t['pnl_rs'] <= 0))
    overall_pf = total_wins / total_losses if total_losses > 0 else 0
    overall_wr = len([t for t in all_trades if t['pnl_rs'] > 0]) / max(len(all_trades), 1)

    daily_pnl_all = {}
    for t in all_trades:
        d = str(t['date'])
        daily_pnl_all[d] = daily_pnl_all.get(d, 0) + t['pnl_rs']
    rets_all = np.array(list(daily_pnl_all.values())) / 1_000_000
    sharpe_all = (np.mean(rets_all) / np.std(rets_all, ddof=1)) * np.sqrt(252) if len(rets_all) > 1 else 0

    print(f"\n{'=' * 85}")
    print(f"  OVERALL: {len(all_trades)} trades, WR {overall_wr:.0%}, PF {overall_pf:.2f}, "
          f"Sharpe {sharpe_all:.2f}")
    print(f"  Total P&L: ₹{total_pnl:,.0f} | WF pass: {passes}/{len(windows)} ({wf_rate:.0%})")
    print(f"  Trades/year: ~{len(all_trades) / (len(trade_dates) / 252):.0f}")
    print(f"  Time: {time_mod.perf_counter() - t0:.1f}s")

    # Per-signal verdict
    print(f"\n{'─' * 85}")
    print(f"  {'Signal':<20s} {'Trades':>6s} {'WR':>6s} {'PF':>6s} {'Total P&L':>10s} {'Verdict':>8s}")
    print(f"{'─' * 85}")
    for sig_id, s in sorted(sig_stats.items()):
        verdict = 'PASS' if s['pf'] >= 1.3 and s['wr'] >= 0.40 else 'FAIL'
        print(f"  {sig_id:<20s} {s['trades']:>6d} {s['wr']:>5.0%} {s['pf']:>5.2f} ₹{s['total']:>8,.0f} {verdict:>8s}")

    print("=" * 85)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    run_wf()

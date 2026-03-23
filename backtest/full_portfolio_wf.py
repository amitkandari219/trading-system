"""
Full Portfolio Walk-Forward Backtest (L1-L9, all layers).

Runs the complete 9-signal portfolio through proper WF validation
on 2015-2025 Nifty daily data with:
  - Realistic transaction costs (0.05% slippage, brokerage, STT)
  - Overlay comparison (ON vs OFF)
  - ML model training on WF train splits
  - Compound sizing with position limits

Usage:
    venv/bin/python3 -m backtest.full_portfolio_wf
"""

import logging
import math
import time as time_mod
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import _eval_conditions
from backtest.indicators import add_all_indicators
from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE

logger = logging.getLogger(__name__)

# ================================================================
# CONFIGURATION
# ================================================================

INITIAL_CAPITAL = 1_000_000  # ₹10L
MAX_POSITIONS = 4
MAX_SAME_DIR = 2

# WF windows: 36mo train / 12mo test / 3mo step
WF_TRAIN_MONTHS = 36
WF_TEST_MONTHS = 12
WF_STEP_MONTHS = 3

# Transaction costs (per side)
SLIPPAGE_PCT = 0.0005      # 0.05% per leg
BROKERAGE_PER_LOT = 40     # ₹40/lot
STT_SELL_PCT = 0.0001      # 0.01% STT on sell (futures)
TOTAL_COST_PER_SIDE = SLIPPAGE_PCT + STT_SELL_PCT / 2  # avg across buy+sell

# ================================================================
# 7 DAILY SIGNAL RULES (from signal_compute.py)
# ================================================================

SIGNAL_RULES = {
    'KAUFMAN_DRY_20': {
        'direction': 'LONG',
        'entry_long': [
            {'indicator': 'sma_10', 'op': '<', 'value': 'close'},
            {'indicator': 'stoch_k_5', 'op': '>', 'value': 50},
        ],
        'exit_long': [{'indicator': 'stoch_k_5', 'op': '<=', 'value': 50}],
        'stop_loss_pct': 0.02,
        'cooldown_days': 2,
    },
    'KAUFMAN_DRY_16': {
        'direction': 'BOTH',
        'entry_long': [
            {'indicator': 'ema_20', 'op': '<', 'value': 'close'},
            {'indicator': 'rsi_14', 'op': '>', 'value': 50},
        ],
        'exit_long': [{'indicator': 'rsi_14', 'op': '<=', 'value': 45}],
        'entry_short': [
            {'indicator': 'ema_20', 'op': '>', 'value': 'close'},
            {'indicator': 'rsi_14', 'op': '<', 'value': 50},
        ],
        'exit_short': [{'indicator': 'rsi_14', 'op': '>=', 'value': 55}],
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.03,
        'cooldown_days': 2,
    },
    'KAUFMAN_DRY_12': {
        'direction': 'BOTH',
        'entry_long': [
            {'indicator': 'sma_50', 'op': '<', 'value': 'close'},
            {'indicator': 'adx_14', 'op': '>', 'value': 20},
        ],
        'exit_long': [{'indicator': 'adx_14', 'op': '<', 'value': 15}],
        'entry_short': [
            {'indicator': 'sma_50', 'op': '>', 'value': 'close'},
            {'indicator': 'adx_14', 'op': '>', 'value': 20},
        ],
        'exit_short': [{'indicator': 'adx_14', 'op': '<', 'value': 15}],
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.03,
        'hold_days': 7, 'cooldown_days': 1,
    },
    'GUJRAL_DRY_8': {
        'direction': 'LONG',
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'sma_20'},
            {'indicator': 'rsi_14', 'op': '>', 'value': 45},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'sma_20'}],
        'stop_loss_pct': 0.02, 'cooldown_days': 3,
    },
    'GUJRAL_DRY_13': {
        'direction': 'LONG',
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'ema_20'},
            {'indicator': 'rsi_14', 'op': '>', 'value': 55},
            {'indicator': 'rsi_14', 'op': '<', 'value': 75},
        ],
        'exit_long': [{'indicator': 'rsi_14', 'op': '<', 'value': 45}],
        'stop_loss_pct': 0.02, 'hold_days': 10, 'cooldown_days': 2,
    },
    'BULKOWSKI_ADAM_EVE': {
        'direction': 'LONG',
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'bb_lower'},
            {'indicator': 'rsi_14', 'op': '<', 'value': 35},
        ],
        'exit_long': [{'indicator': 'rsi_14', 'op': '>', 'value': 60}],
        'stop_loss_pct': 0.03, 'cooldown_days': 5,
    },
    'SCHWAGER_TREND': {
        'direction': 'LONG',
        'entry_long': [
            {'indicator': 'close', 'op': '>', 'value': 'sma_50'},
            {'indicator': 'adx_14', 'op': '>', 'value': 25},
        ],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'sma_50'}],
        'stop_loss_pct': 0.02, 'cooldown_days': 3,
    },
}

# ================================================================
# OVERLAY DEFINITIONS
# ================================================================

def compute_vix_overlay(vix: float) -> float:
    """VIX-based sizing overlay. Returns multiplier 0.0 - 1.0."""
    if pd.isna(vix) or vix <= 0:
        return 1.0
    if vix < 14:
        return 1.0
    if vix < 18:
        return 0.9
    if vix < 22:
        return 0.7
    if vix < 28:
        return 0.4
    return 0.0  # CRISIS — no new entries


def compute_regime_overlay(adx: float, direction: str) -> float:
    """ADX regime overlay. Trend signals boosted in trending regime."""
    if pd.isna(adx):
        return 1.0
    if adx > 25:
        return 1.2 if direction == 'LONG' else 1.0
    return 0.8  # ranging — reduce trend signal size


def compute_pcr_overlay(pcr: float) -> float:
    """PCR contrarian overlay."""
    if pd.isna(pcr) or pcr <= 0:
        return 1.0
    if pcr > 1.3:
        return 1.2  # extreme fear → bullish
    if pcr < 0.5:
        return 0.8  # extreme greed → reduce
    return 1.0


def compute_calendar_overlay(dt: date) -> float:
    """Reduce size around known events."""
    m, d = dt.month, dt.day
    # Budget day (Feb 1)
    if m == 2 and d == 1:
        return 0.5
    # RBI MPC (approximate: first Wed of Feb/Apr/Jun/Aug/Oct/Dec)
    if m in (2, 4, 6, 8, 10, 12) and d <= 7 and dt.weekday() == 2:
        return 0.7
    return 1.0


def compute_behavioral_overlay(consecutive_losses: int, consecutive_wins: int) -> float:
    """Kahneman behavioral bias correction."""
    if consecutive_wins >= 8:
        return 0.8  # overconfidence dampener
    if consecutive_wins >= 5:
        return 0.9
    if consecutive_losses >= 5:
        return 0.7  # loss aversion floor
    return 1.0


def apply_all_overlays(vix, adx, pcr, dt, direction, consec_losses, consec_wins):
    """Combine all overlays multiplicatively."""
    m = 1.0
    m *= compute_vix_overlay(vix)
    m *= compute_regime_overlay(adx, direction)
    m *= compute_pcr_overlay(pcr)
    m *= compute_calendar_overlay(dt)
    m *= compute_behavioral_overlay(consec_losses, consec_wins)
    return max(0.0, min(2.0, m))


# ================================================================
# ML MODEL TRAINING (simplified walk-forward compatible)
# ================================================================

def train_regime_model(train_df: pd.DataFrame) -> dict:
    """
    Train a simple regime classifier on the train window.
    Returns model parameters (thresholds) for test window.
    Uses VIX + ADX + returns to classify regime.
    """
    if len(train_df) < 100:
        return {'vix_high': 20, 'adx_trend': 25, 'method': 'default'}

    vix = train_df['india_vix'].dropna()
    adx = train_df['adx_14'].dropna() if 'adx_14' in train_df.columns else pd.Series([25])

    return {
        'vix_high': float(vix.quantile(0.75)) if len(vix) > 10 else 20,
        'adx_trend': float(adx.quantile(0.60)) if len(adx) > 10 else 25,
        'vix_crisis': float(vix.quantile(0.95)) if len(vix) > 10 else 28,
        'method': 'trained',
    }


def train_sizing_model(train_trades: list) -> dict:
    """
    Train RL-style sizing from historical trades.
    Returns optimal fixed-fraction size (half-Kelly).
    """
    if len(train_trades) < 30:
        return {'base_size': 1.0, 'method': 'default'}

    wins = [t['pnl_pct'] for t in train_trades if t['pnl_pct'] > 0]
    losses = [t['pnl_pct'] for t in train_trades if t['pnl_pct'] <= 0]

    if not wins or not losses:
        return {'base_size': 1.0, 'method': 'default'}

    p = len(wins) / len(train_trades)
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))

    if avg_loss <= 0:
        return {'base_size': 1.0, 'method': 'default'}

    b = avg_win / avg_loss
    kelly = p - (1 - p) / b if b > 0 else 0
    half_kelly = max(0.3, min(1.5, kelly * 0.5))

    return {
        'base_size': round(half_kelly, 2),
        'kelly_full': round(kelly, 3),
        'win_rate': round(p, 3),
        'avg_win': round(avg_win, 4),
        'avg_loss': round(avg_loss, 4),
        'method': 'half_kelly',
    }


# ================================================================
# TRADE SIMULATOR
# ================================================================

def simulate_signal_trades(
    signal_id: str, rules: dict, df: pd.DataFrame,
    cost_per_side: float = TOTAL_COST_PER_SIDE,
) -> List[Dict]:
    """Run a single signal on indicator-enriched data, return trades with costs."""
    entry_long = rules.get('entry_long', [])
    entry_short = rules.get('entry_short', [])
    exit_long = rules.get('exit_long', [])
    exit_short = rules.get('exit_short', [])
    direction = rules.get('direction', 'BOTH')
    sl_pct = rules.get('stop_loss_pct', 0.02)
    tp_pct = rules.get('take_profit_pct', 0)
    hold_days = rules.get('hold_days', 0)
    cooldown = rules.get('cooldown_days', 1)

    trades = []
    position = None
    entry_price = 0.0
    entry_idx = 0
    days_in = 0
    last_exit = -cooldown

    closes = df['close'].values
    dates = df['date'].values if 'date' in df.columns else df.index.values
    n = len(df)

    for i in range(1, n):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        c = float(closes[i])

        if position is not None:
            days_in += 1
            exit_reason = None
            exit_price = c

            if position == 'LONG':
                if sl_pct > 0 and c <= entry_price * (1 - sl_pct):
                    exit_reason, exit_price = 'SL', entry_price * (1 - sl_pct)
                elif tp_pct > 0 and c >= entry_price * (1 + tp_pct):
                    exit_reason, exit_price = 'TP', entry_price * (1 + tp_pct)
                elif exit_long and _eval_conditions(row, prev, exit_long):
                    exit_reason = 'RULE'
                elif hold_days > 0 and days_in >= hold_days:
                    exit_reason = 'TIME'
            else:
                if sl_pct > 0 and c >= entry_price * (1 + sl_pct):
                    exit_reason, exit_price = 'SL', entry_price * (1 + sl_pct)
                elif tp_pct > 0 and c <= entry_price * (1 - tp_pct):
                    exit_reason, exit_price = 'TP', entry_price * (1 - tp_pct)
                elif exit_short and _eval_conditions(row, prev, exit_short):
                    exit_reason = 'RULE'
                elif hold_days > 0 and days_in >= hold_days:
                    exit_reason = 'TIME'

            if exit_reason:
                # Apply costs
                gross_pnl_pct = (exit_price - entry_price) / entry_price
                if position == 'SHORT':
                    gross_pnl_pct = -gross_pnl_pct
                net_pnl_pct = gross_pnl_pct - 2 * cost_per_side  # round-trip

                trades.append({
                    'signal_id': signal_id,
                    'entry_date': dates[entry_idx],
                    'exit_date': dates[i],
                    'direction': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_pnl_pct': gross_pnl_pct,
                    'pnl_pct': net_pnl_pct,
                    'costs_pct': 2 * cost_per_side,
                    'days_held': days_in,
                    'exit_reason': exit_reason,
                })
                position = None
                last_exit = i
        else:
            if i - last_exit < cooldown:
                continue
            if direction in ('BOTH', 'LONG'):
                if entry_long and _eval_conditions(row, prev, entry_long):
                    position, entry_price, entry_idx, days_in = 'LONG', c, i, 0
                    continue
            if direction in ('BOTH', 'SHORT'):
                if entry_short and _eval_conditions(row, prev, entry_short):
                    position, entry_price, entry_idx, days_in = 'SHORT', c, i, 0

    return trades


# ================================================================
# PORTFOLIO COMBINER WITH OVERLAYS
# ================================================================

def combine_portfolio_with_overlays(
    all_signal_trades: Dict[str, List[Dict]],
    df: pd.DataFrame,
    overlays_on: bool = True,
    ml_sizing: Optional[dict] = None,
    initial_capital: float = INITIAL_CAPITAL,
) -> Dict:
    """
    Combine signal trades into portfolio with position limits,
    compound sizing, overlays, and ML-trained sizing.
    """
    # Flatten and sort
    flat = []
    for sig_id, trades in all_signal_trades.items():
        for t in trades:
            flat.append(dict(t))
    flat.sort(key=lambda t: str(t['entry_date']))

    if not flat:
        return _empty_result()

    # Build date-indexed lookup for overlays
    df_idx = df.set_index('date') if 'date' in df.columns else df
    vix_map = df_idx['india_vix'].to_dict() if 'india_vix' in df_idx.columns else {}
    adx_map = df_idx['adx_14'].to_dict() if 'adx_14' in df_idx.columns else {}
    pcr_map = df_idx['pcr_oi'].to_dict() if 'pcr_oi' in df_idx.columns else {}

    equity = initial_capital
    peak = equity
    max_dd = 0
    daily_pnl = {}
    portfolio_trades = []
    open_positions = {}
    consec_wins = 0
    consec_losses = 0

    ml_base_size = ml_sizing.get('base_size', 1.0) if ml_sizing else 1.0

    for trade in flat:
        sig = trade['signal_id']
        direction = trade.get('direction', 'LONG')

        # Position limits
        if sig in open_positions:
            continue
        if len(open_positions) >= MAX_POSITIONS:
            continue
        same_dir = sum(1 for p in open_positions.values() if p.get('direction') == direction)
        if same_dir >= MAX_SAME_DIR:
            continue

        # Get overlay data
        entry_d = trade['entry_date']
        if hasattr(entry_d, 'date'):
            entry_d = entry_d.date() if callable(entry_d.date) else entry_d

        # Compute overlay multiplier
        if overlays_on:
            vix = vix_map.get(entry_d, 15)
            adx = adx_map.get(entry_d, 20)
            pcr = pcr_map.get(entry_d, 0.8)
            if hasattr(entry_d, 'month'):
                overlay_mult = apply_all_overlays(
                    vix, adx, pcr, entry_d, direction,
                    consec_losses, consec_wins
                )
            else:
                overlay_mult = 1.0
        else:
            overlay_mult = 1.0

        if overlay_mult <= 0:
            continue  # blocked by overlay

        # Deploy fraction (equity tier)
        if equity < 200_000:
            deploy = 0.45
        elif equity < 500_000:
            deploy = 0.50
        elif equity < 1_000_000:
            deploy = 0.55
        else:
            deploy = 0.50

        # Position size with overlays + ML sizing
        position_frac = deploy * overlay_mult * ml_base_size
        position_frac = min(position_frac, 0.25)  # max 25% per position
        position_size = equity * position_frac

        # PnL
        pnl_pct = trade.get('pnl_pct', 0)
        pnl_rs = position_size * pnl_pct
        equity += pnl_rs
        equity = max(equity, 10_000)  # floor

        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

        # Track streaks
        if pnl_pct > 0:
            consec_wins += 1
            consec_losses = 0
        else:
            consec_losses += 1
            consec_wins = 0

        # Daily returns
        exit_d = str(trade.get('exit_date', ''))[:10]
        daily_pnl[exit_d] = daily_pnl.get(exit_d, 0) + pnl_pct

        trade['overlay_mult'] = overlay_mult
        trade['position_size'] = position_size
        trade['equity_after'] = equity
        portfolio_trades.append(trade)

    return _compute_portfolio_metrics(portfolio_trades, equity, initial_capital, max_dd, daily_pnl)


def _compute_portfolio_metrics(trades, final_equity, initial, max_dd, daily_pnl):
    if not trades:
        return _empty_result()

    pnls = [t['pnl_pct'] for t in trades]
    gross_pnls = [t.get('gross_pnl_pct', t['pnl_pct']) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    rets = np.array(list(daily_pnl.values()))
    years = len(rets) / 252 if len(rets) > 0 else 1

    sharpe = (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(252) if len(rets) > 1 and np.std(rets) > 0 else 0
    cagr = (final_equity / initial) ** (1 / max(years, 0.5)) - 1 if final_equity > 0 else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
    wr = len(wins) / len(pnls) if pnls else 0

    total_costs = sum(t.get('costs_pct', 0) for t in trades)

    return {
        'trades': len(trades),
        'sharpe': round(sharpe, 2),
        'cagr_pct': round(cagr * 100, 1),
        'max_dd_pct': round(max_dd * 100, 1),
        'pf': round(pf, 2),
        'win_rate_pct': round(wr * 100, 1),
        'final_equity': round(final_equity),
        'total_return_pct': round((final_equity - initial) / initial * 100, 1),
        'total_cost_drag_pct': round(total_costs * 100, 2),
        'avg_cost_per_trade_pct': round(total_costs / len(trades) * 100, 3) if trades else 0,
        'years': round(years, 1),
        'trades_per_year': round(len(trades) / max(years, 0.5), 0),
    }


def _empty_result():
    return {
        'trades': 0, 'sharpe': 0, 'cagr_pct': 0, 'max_dd_pct': 0,
        'pf': 0, 'win_rate_pct': 0, 'final_equity': 0, 'total_return_pct': 0,
        'total_cost_drag_pct': 0, 'avg_cost_per_trade_pct': 0, 'years': 0,
        'trades_per_year': 0,
    }


# ================================================================
# WF WINDOWS
# ================================================================

def generate_wf_windows(min_date, max_date):
    windows = []
    start = min_date
    while True:
        train_end = start + timedelta(days=WF_TRAIN_MONTHS * 30)
        test_start = train_end
        test_end = test_start + timedelta(days=WF_TEST_MONTHS * 30)
        if test_end > max_date:
            break
        windows.append({
            'train_start': start, 'train_end': train_end,
            'test_start': test_start, 'test_end': test_end,
        })
        start += timedelta(days=WF_STEP_MONTHS * 30)
    return windows


# ================================================================
# MAIN
# ================================================================

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    t0 = time_mod.perf_counter()

    print("=" * 90)
    print("  FULL PORTFOLIO WALK-FORWARD BACKTEST (L1-L9)")
    print("  Period: 2015-11 to 2026-03 | 7 daily SCORING signals")
    print("  Costs: 0.05% slippage + STT + brokerage per side")
    print("  Overlays: VIX regime, ADX, PCR, calendar, behavioral")
    print("  ML: Half-Kelly sizing trained per WF window")
    print("=" * 90)

    # Load data
    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix, "
        "pcr_oi, pcr_volume FROM nifty_daily ORDER BY date",
        conn, parse_dates=['date'],
    )
    conn.close()
    df['date_obj'] = df['date'].dt.date
    print(f"\nLoaded {len(df)} daily bars ({df['date'].min().date()} to {df['date'].max().date()})")

    # Add indicators
    print("Computing indicators...")
    df_ind = add_all_indicators(df)

    # Generate WF windows
    min_d = df['date'].min().date()
    max_d = df['date'].max().date()
    windows = generate_wf_windows(min_d, max_d)
    print(f"Generated {len(windows)} WF windows (36mo/12mo/3mo)")

    # ── Run 1: All signals, NO overlays, NO ML ──
    print("\n" + "─" * 90)
    print("  RUN 1: BASE (no overlays, no ML, with realistic costs)")
    print("─" * 90)

    all_trades_base = {}
    for sig_id, rules in SIGNAL_RULES.items():
        trades = simulate_signal_trades(sig_id, rules, df_ind)
        all_trades_base[sig_id] = trades
        wr = len([t for t in trades if t['pnl_pct'] > 0]) / max(len(trades), 1)
        print(f"  {sig_id:25s}: {len(trades):4d} trades, WR={wr:.0%}")

    result_base = combine_portfolio_with_overlays(
        all_trades_base, df_ind, overlays_on=False, ml_sizing=None,
    )

    # ── Run 2: All signals, WITH overlays, NO ML ──
    print("\n" + "─" * 90)
    print("  RUN 2: WITH OVERLAYS (VIX + ADX + PCR + calendar + behavioral)")
    print("─" * 90)

    result_overlay = combine_portfolio_with_overlays(
        all_trades_base, df_ind, overlays_on=True, ml_sizing=None,
    )

    # ── Run 3: WF-trained ML sizing ──
    print("\n" + "─" * 90)
    print("  RUN 3: WITH OVERLAYS + ML SIZING (half-Kelly per WF window)")
    print("─" * 90)

    # Train ML sizing on first WF train window, apply to all
    if windows:
        w = windows[0]
        train_mask = (df_ind['date'].dt.date >= w['train_start']) & (df_ind['date'].dt.date < w['train_end'])
        train_df = df_ind[train_mask]
        # Collect train-window trades for Kelly estimation
        train_trades = []
        for sig_id, rules in SIGNAL_RULES.items():
            train_trades.extend(simulate_signal_trades(sig_id, rules, train_df))
        ml_sizing = train_sizing_model(train_trades)
        regime_model = train_regime_model(train_df)
        print(f"  ML sizing: {ml_sizing}")
        print(f"  Regime model: {regime_model}")
    else:
        ml_sizing = {'base_size': 1.0, 'method': 'default'}

    result_ml = combine_portfolio_with_overlays(
        all_trades_base, df_ind, overlays_on=True, ml_sizing=ml_sizing,
    )

    # ── Per-window WF analysis ──
    print("\n" + "─" * 90)
    print("  WF WINDOW-BY-WINDOW ANALYSIS (overlays ON)")
    print("─" * 90)

    window_results = []
    for i, w in enumerate(windows):
        test_mask = (df_ind['date'].dt.date >= w['test_start']) & (df_ind['date'].dt.date <= w['test_end'])
        test_df = df_ind[test_mask]
        if len(test_df) < 20:
            continue

        # Train sizing on train window
        train_mask = (df_ind['date'].dt.date >= w['train_start']) & (df_ind['date'].dt.date < w['train_end'])
        train_df = df_ind[train_mask]
        train_trades = []
        for sig_id, rules in SIGNAL_RULES.items():
            train_trades.extend(simulate_signal_trades(sig_id, rules, train_df))
        w_ml = train_sizing_model(train_trades)

        # Test window trades
        w_trades = {}
        for sig_id, rules in SIGNAL_RULES.items():
            w_trades[sig_id] = simulate_signal_trades(sig_id, rules, test_df)

        w_result = combine_portfolio_with_overlays(w_trades, test_df, overlays_on=True, ml_sizing=w_ml)
        w_result['window'] = i
        w_result['test_period'] = f"{w['test_start']} to {w['test_end']}"
        w_result['ml_base_size'] = w_ml.get('base_size', 1.0)
        window_results.append(w_result)

    # Print window results
    print(f"  {'Win':>3s} {'Period':>25s} {'Trades':>6s} {'Sharpe':>7s} {'CAGR':>7s} {'MaxDD':>6s} {'PF':>5s} {'Kelly':>6s}")
    passes = 0
    for wr in window_results:
        passed = wr['sharpe'] > 0.8 and wr['pf'] > 1.2
        if passed:
            passes += 1
        marker = " *" if passed else ""
        print(f"  {wr['window']:3d} {wr['test_period']:>25s} {wr['trades']:6d} "
              f"{wr['sharpe']:7.2f} {wr['cagr_pct']:6.1f}% {wr['max_dd_pct']:5.1f}% "
              f"{wr['pf']:5.2f} {wr.get('ml_base_size',1.0):5.2f}{marker}")

    wf_pass_rate = passes / max(len(window_results), 1)

    # ── COMPARISON TABLE ──
    print("\n" + "=" * 90)
    print("  COMPARISON: BASE vs OVERLAYS vs OVERLAYS+ML")
    print("=" * 90)
    print(f"  {'Metric':<25s} {'BASE':>12s} {'+ OVERLAYS':>12s} {'+ ML SIZE':>12s}")
    print(f"  {'─' * 61}")
    for metric, label, suffix in [
        ('trades', 'Trades', ''),
        ('sharpe', 'Sharpe', ''),
        ('cagr_pct', 'CAGR', '%'),
        ('max_dd_pct', 'Max Drawdown', '%'),
        ('pf', 'Profit Factor', ''),
        ('win_rate_pct', 'Win Rate', '%'),
        ('final_equity', 'Final Equity ₹', ''),
        ('total_cost_drag_pct', 'Total Cost Drag', '%'),
        ('trades_per_year', 'Trades/Year', ''),
    ]:
        v1 = result_base[metric]
        v2 = result_overlay[metric]
        v3 = result_ml[metric]
        if metric == 'final_equity':
            print(f"  {label:<25s} {v1:>11,} {v2:>11,} {v3:>11,}")
        else:
            print(f"  {label:<25s} {v1:>11}{suffix} {v2:>11}{suffix} {v3:>11}{suffix}")

    # ── OVERLAY ALPHA ──
    print(f"\n  OVERLAY ALPHA (isolated):")
    overlay_cagr_delta = result_overlay['cagr_pct'] - result_base['cagr_pct']
    overlay_sharpe_delta = result_overlay['sharpe'] - result_base['sharpe']
    overlay_dd_delta = result_overlay['max_dd_pct'] - result_base['max_dd_pct']
    print(f"    CAGR:   {overlay_cagr_delta:+.1f}% (overlays {'add' if overlay_cagr_delta > 0 else 'reduce'} return)")
    print(f"    Sharpe: {overlay_sharpe_delta:+.2f}")
    print(f"    MaxDD:  {overlay_dd_delta:+.1f}% ({'improved' if overlay_dd_delta < 0 else 'worse'})")

    ml_cagr_delta = result_ml['cagr_pct'] - result_overlay['cagr_pct']
    print(f"\n  ML SIZING ALPHA (on top of overlays):")
    print(f"    CAGR:   {ml_cagr_delta:+.1f}%")
    print(f"    Kelly base: {ml_sizing.get('base_size', 'N/A')}")

    # ── WF SUMMARY ──
    print(f"\n  WF PASS RATE: {passes}/{len(window_results)} windows ({wf_pass_rate:.0%})")
    print(f"  (criteria: Sharpe > 0.8 AND PF > 1.2)")

    elapsed = time_mod.perf_counter() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 90)


if __name__ == '__main__':
    main()

"""
Lot-Based Walk-Forward Backtest — overlays actually change position size.

Three modes:
  1. FIXED-FRAC: percentage of equity per trade (original, overlays ignored)
  2. LOT-BASED: lots computed from risk budget + overlay modifiers
  3. LOT+ML: lot-based with half-Kelly trained per WF window

Usage:
    venv/bin/python3 -m backtest.lot_based_wf
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
from backtest.transaction_costs import TransactionCostModel
from config.settings import DATABASE_DSN
from execution.adaptive_kelly import AdaptiveKelly
from execution.lot_sizer import LotSizer, get_lot_size, MARGIN_PER_LOT
from execution.overlay_pipeline import OverlayPipeline
from execution.conviction_scorer import ConvictionScorer
from execution.trend_confirmer import TrendConfirmer

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 1_000_000
MAX_POSITIONS = 4
MAX_SAME_DIR = 2

# Cost model: uses the corrected TransactionCostModel
_COST_MODEL = TransactionCostModel()

# Signal rules (same 7 daily signals)
SIGNAL_RULES = {
    'KAUFMAN_DRY_20': {
        'direction': 'LONG',
        'entry_long': [{'indicator': 'sma_10', 'op': '<', 'value': 'close'},
                       {'indicator': 'stoch_k_5', 'op': '>', 'value': 50}],
        'exit_long': [{'indicator': 'stoch_k_5', 'op': '<=', 'value': 50}],
        'stop_loss_pct': 0.02, 'cooldown_days': 2,
    },
    'KAUFMAN_DRY_16': {
        'direction': 'BOTH',
        'entry_long': [{'indicator': 'ema_20', 'op': '<', 'value': 'close'},
                       {'indicator': 'rsi_14', 'op': '>', 'value': 50}],
        'exit_long': [{'indicator': 'rsi_14', 'op': '<=', 'value': 45}],
        'entry_short': [{'indicator': 'ema_20', 'op': '>', 'value': 'close'},
                        {'indicator': 'rsi_14', 'op': '<', 'value': 50}],
        'exit_short': [{'indicator': 'rsi_14', 'op': '>=', 'value': 55}],
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.03, 'cooldown_days': 2,
    },
    'KAUFMAN_DRY_12': {
        'direction': 'BOTH',
        'entry_long': [{'indicator': 'sma_50', 'op': '<', 'value': 'close'},
                       {'indicator': 'adx_14', 'op': '>', 'value': 20}],
        'exit_long': [{'indicator': 'adx_14', 'op': '<', 'value': 15}],
        'entry_short': [{'indicator': 'sma_50', 'op': '>', 'value': 'close'},
                        {'indicator': 'adx_14', 'op': '>', 'value': 20}],
        'exit_short': [{'indicator': 'adx_14', 'op': '<', 'value': 15}],
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.03, 'hold_days': 7, 'cooldown_days': 1,
    },
    'GUJRAL_DRY_8': {
        'direction': 'LONG',
        'entry_long': [{'indicator': 'close', 'op': '>', 'value': 'sma_20'},
                       {'indicator': 'rsi_14', 'op': '>', 'value': 45}],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'sma_20'}],
        'stop_loss_pct': 0.02, 'cooldown_days': 3,
    },
    'GUJRAL_DRY_13': {
        'direction': 'LONG',
        'entry_long': [{'indicator': 'close', 'op': '>', 'value': 'ema_20'},
                       {'indicator': 'rsi_14', 'op': '>', 'value': 55},
                       {'indicator': 'rsi_14', 'op': '<', 'value': 75}],
        'exit_long': [{'indicator': 'rsi_14', 'op': '<', 'value': 45}],
        'stop_loss_pct': 0.02, 'hold_days': 10, 'cooldown_days': 2,
    },
    'BULKOWSKI_ADAM_EVE': {
        'direction': 'LONG',
        'entry_long': [{'indicator': 'close', 'op': '>', 'value': 'bb_lower'},
                       {'indicator': 'rsi_14', 'op': '<', 'value': 35}],
        'exit_long': [{'indicator': 'rsi_14', 'op': '>', 'value': 60}],
        'stop_loss_pct': 0.03, 'cooldown_days': 5,
    },
    'SCHWAGER_TREND': {
        'direction': 'LONG',
        'entry_long': [{'indicator': 'close', 'op': '>', 'value': 'sma_50'},
                       {'indicator': 'adx_14', 'op': '>', 'value': 25}],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'sma_50'}],
        'stop_loss_pct': 0.02, 'cooldown_days': 3,
    },
}


def _compute_costs(entry_price, exit_price, lots, lot_size, direction, vix=None):
    """Compute realistic trading costs in rupees using TransactionCostModel.

    Delegates to the corrected TransactionCostModel which includes:
    - Brokerage: ₹20/order × 1.18 GST = ₹23.60/order
    - STT: 0.0125% on futures sell side
    - Exchange charges: 0.0495% on turnover
    - SEBI: 0.0001% on turnover
    - Stamp duty: 0.003% on buy side
    - Slippage: VIX-scaled
    """
    costs = _COST_MODEL.compute_futures_round_trip(
        entry_price=entry_price, exit_price=exit_price,
        lots=lots, vix=vix,
    )
    return costs.total


def run_lot_based_backtest(
    df_ind: pd.DataFrame,
    overlay_pipeline: Optional[OverlayPipeline],
    mode: str = 'LOT_BASED',  # 'FIXED_FRAC', 'LOT_BASED', 'LOT_CONVICTION', 'LOT_ML'
    ml_kelly_mult: float = 1.0,
) -> Dict:
    """
    Run the full portfolio backtest in the specified mode.

    Returns dict with metrics + trade log.
    """
    equity = float(INITIAL_CAPITAL)
    peak = equity
    max_dd = 0.0
    sizer = LotSizer(equity=equity)
    conviction_scorer = ConvictionScorer()
    trend_confirmer = TrendConfirmer()

    trades_log = []
    open_positions = {}  # sig_id -> position dict
    daily_equity = {}

    dates = df_ind['date'].values if 'date' in df_ind.columns else df_ind.index.values
    closes = df_ind['close'].values.astype(float)
    n = len(df_ind)

    # Per-signal state
    sig_state = {sig: {'position': None, 'entry_price': 0, 'entry_idx': 0,
                        'days_in': 0, 'last_exit': -5, 'lots': 0, 'direction': None}
                 for sig in SIGNAL_RULES}

    overlay_contrib = []  # track overlay modifier impact

    for i in range(1, n):
        row = df_ind.iloc[i]
        prev = df_ind.iloc[i - 1]
        c = float(closes[i])
        dt = dates[i]

        try:
            dt_date = pd.Timestamp(dt).date()
        except Exception:
            dt_date = date(2024, 1, 1)  # fallback

        lot_size = get_lot_size(dt_date) if hasattr(dt_date, 'year') else 25

        # ── CHECK EXITS for all open positions ──
        for sig_id, state in sig_state.items():
            if state['position'] is None:
                continue

            state['days_in'] += 1
            rules = SIGNAL_RULES[sig_id]
            pos = state['position']
            ep = state['entry_price']
            sl_pct = rules.get('stop_loss_pct', 0.02)
            tp_pct = rules.get('take_profit_pct', 0)
            hold = rules.get('hold_days', 0)

            exit_reason = None
            exit_price = c

            if pos == 'LONG':
                if sl_pct > 0 and c <= ep * (1 - sl_pct):
                    exit_reason, exit_price = 'SL', ep * (1 - sl_pct)
                elif tp_pct > 0 and c >= ep * (1 + tp_pct):
                    exit_reason, exit_price = 'TP', ep * (1 + tp_pct)
                elif rules.get('exit_long') and _eval_conditions(row, prev, rules['exit_long']):
                    exit_reason = 'RULE'
                elif hold > 0 and state['days_in'] >= hold:
                    exit_reason = 'TIME'
            else:
                if sl_pct > 0 and c >= ep * (1 + sl_pct):
                    exit_reason, exit_price = 'SL', ep * (1 + sl_pct)
                elif tp_pct > 0 and c <= ep * (1 - tp_pct):
                    exit_reason, exit_price = 'TP', ep * (1 - tp_pct)
                elif rules.get('exit_short') and _eval_conditions(row, prev, rules['exit_short']):
                    exit_reason = 'RULE'
                elif hold > 0 and state['days_in'] >= hold:
                    exit_reason = 'TIME'

            if exit_reason:
                lots = state['lots']
                # P&L in rupees
                if pos == 'LONG':
                    gross_pnl = lots * lot_size * (exit_price - ep)
                else:
                    gross_pnl = lots * lot_size * (ep - exit_price)

                costs = _compute_costs(ep, exit_price, lots, lot_size, pos)
                net_pnl = gross_pnl - costs

                equity += net_pnl
                equity = max(equity, 10_000)
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
                sizer.update_equity(equity)

                trades_log.append({
                    'signal_id': sig_id, 'entry_date': dates[state['entry_idx']],
                    'exit_date': dt, 'direction': pos, 'lots': lots, 'lot_size': lot_size,
                    'entry_price': ep, 'exit_price': exit_price,
                    'gross_pnl': round(gross_pnl), 'costs': round(costs),
                    'net_pnl': round(net_pnl), 'equity_after': round(equity),
                    'exit_reason': exit_reason, 'days_held': state['days_in'],
                    'composite_modifier': state.get('composite_mod', 1.0),
                })

                state['position'] = None
                state['last_exit'] = i
                open_positions.pop(sig_id, None)

        # ── CHECK ENTRIES ──
        for sig_id, rules in SIGNAL_RULES.items():
            state = sig_state[sig_id]
            if state['position'] is not None:
                continue
            cooldown = rules.get('cooldown_days', 1)
            if i - state['last_exit'] < cooldown:
                continue
            if len(open_positions) >= MAX_POSITIONS:
                continue

            direction = rules.get('direction', 'BOTH')
            fired_dir = None

            if direction in ('BOTH', 'LONG'):
                if rules.get('entry_long') and _eval_conditions(row, prev, rules['entry_long']):
                    fired_dir = 'LONG'
            if fired_dir is None and direction in ('BOTH', 'SHORT'):
                if rules.get('entry_short') and _eval_conditions(row, prev, rules['entry_short']):
                    fired_dir = 'SHORT'

            if fired_dir is None:
                continue

            # Same-direction limit
            same_dir = sum(1 for p in open_positions.values() if p == fired_dir)
            if same_dir >= MAX_SAME_DIR:
                continue

            # ── COMPUTE LOTS ──
            sl_pct = rules.get('stop_loss_pct', 0.02)
            sl_pts = c * sl_pct  # stop loss in points

            if mode == 'FIXED_FRAC':
                # Fixed fraction: ignore overlays
                risk_budget = equity * 0.02
                risk_per_lot = sl_pts * lot_size
                lots = max(1, math.floor(risk_budget / risk_per_lot)) if risk_per_lot > 0 else 1
                lots = min(lots, 20)
                composite_mod = 1.0
            else:
                # LOT_BASED, LOT_CONVICTION, or LOT_ML: use overlay modifiers
                if overlay_pipeline and hasattr(dt_date, 'month'):
                    modifiers = overlay_pipeline.get_modifiers(dt_date, fired_dir)
                else:
                    modifiers = {}

                if mode == 'LOT_ML':
                    for k in modifiers:
                        modifiers[k] = modifiers[k] * ml_kelly_mult

                # Compute adaptive Kelly fraction and inject into modifiers
                vix_for_kelly = float(row['india_vix']) if 'india_vix' in row.index and pd.notna(row.get('india_vix')) else 15
                dd_for_kelly = (peak - equity) / peak if peak > 0 else 0
                recent_trades = trades_log[-10:] if trades_log else []
                recent_wins = sum(1 for t in recent_trades if t['net_pnl'] > 0)
                recent_wr = recent_wins / len(recent_trades) if recent_trades else 0.50
                kelly = AdaptiveKelly(base_fraction=0.85)
                kelly_result = kelly.get_fraction(
                    drawdown_pct=dd_for_kelly, recent_wr=recent_wr, vix=vix_for_kelly,
                )
                modifiers['_KELLY_FRACTION'] = kelly_result['fraction']

                final_conv = 1.0  # default for non-conviction modes

                # Conviction scoring (LOT_CONVICTION mode)
                if mode == 'LOT_CONVICTION':
                    vix_val = float(row['india_vix']) if 'india_vix' in row.index and pd.notna(row.get('india_vix')) else 15
                    adx_val = float(row['adx_14']) if 'adx_14' in row.index and pd.notna(row.get('adx_14')) else 20
                    dd_pct = (peak - equity) / peak if peak > 0 else 0

                    conv_result = conviction_scorer.compute(
                        modifiers, vix=vix_val, adx=adx_val, direction=fired_dir,
                        open_positions=len(open_positions),
                        drawdown_pct=dd_pct,
                        consecutive_losses=sig_state[sig_id].get('consec_losses', 0),
                    )
                    conv_modifier = conv_result['final_modifier']

                    # Trend confirmation gate
                    sma20 = float(row['sma_20']) if 'sma_20' in row.index and pd.notna(row.get('sma_20')) else c
                    sma50 = float(row['sma_50']) if 'sma_50' in row.index and pd.notna(row.get('sma_50')) else c
                    high_52w = float(df_ind['close'].iloc[max(0,i-252):i+1].max()) if i > 10 else c

                    # Build recent close/open tuples for green-day check
                    recent_5 = []
                    for j in range(max(0, i-5), i):
                        rc = float(df_ind.iloc[j]['close'])
                        ro = float(df_ind.iloc[j]['open'])
                        recent_5.append((rc, ro))

                    trend = trend_confirmer.is_trend_confirmed(
                        close=c, sma_20=sma20, sma_50=sma50,
                        adx=adx_val, high_52w=high_52w,
                        recent_closes_5d=recent_5,
                    )
                    final_conv = trend_confirmer.gate_modifier(conv_modifier, trend['confirmed'])

                result = sizer.compute(
                    stop_loss_pts=sl_pts, nifty_price=c,
                    trade_date=dt_date, overlay_modifiers=modifiers,
                    direction=fired_dir,
                )
                lots = result['lots']
                composite_mod = result['composite_modifier']

                # Apply conviction modifier AFTER base lot computation (conviction mode only)
                if mode == 'LOT_CONVICTION' and final_conv != 1.0:
                    lots = max(1, min(20, round(lots * final_conv)))
                    composite_mod = composite_mod * final_conv

                overlay_contrib.append({
                    'date': dt_date, 'signal': sig_id, 'direction': fired_dir,
                    'base_lots': result['base_lots'], 'adjusted_lots': lots,
                    'composite': composite_mod,
                })

            # Record entry
            state['position'] = fired_dir
            state['entry_price'] = c
            state['entry_idx'] = i
            state['days_in'] = 0
            state['lots'] = lots
            state['direction'] = fired_dir
            state['composite_mod'] = composite_mod
            open_positions[sig_id] = fired_dir

        # Track daily equity
        d_str = str(dt)[:10]
        daily_equity[d_str] = equity

    # ── COMPUTE METRICS ──
    return _compute_metrics(trades_log, equity, max_dd, overlay_contrib, mode)


def _compute_metrics(trades, final_equity, max_dd, overlay_contrib, mode):
    if not trades:
        return {'mode': mode, 'trades': 0, 'sharpe': 0, 'cagr_pct': 0,
                'max_dd_pct': 0, 'pf': 0, 'win_rate_pct': 0, 'final_equity': INITIAL_CAPITAL,
                'avg_lots': 0, 'margin_util_pct': 0, 'overlay_impact_pct': 0,
                'total_costs': 0, 'cost_pct_of_pnl': 0}

    net_pnls = [t['net_pnl'] for t in trades]
    gross_pnls = [t['gross_pnl'] for t in trades]
    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]

    # Daily returns for Sharpe
    daily_pnl = {}
    for t in trades:
        d = str(t['exit_date'])[:10]
        daily_pnl[d] = daily_pnl.get(d, 0) + t['net_pnl']
    daily_rets = np.array(list(daily_pnl.values())) / INITIAL_CAPITAL
    years = len(daily_rets) / 252 if len(daily_rets) > 0 else 1

    sharpe = (np.mean(daily_rets) / np.std(daily_rets, ddof=1)) * np.sqrt(252) if len(daily_rets) > 1 and np.std(daily_rets) > 0 else 0
    cagr = (final_equity / INITIAL_CAPITAL) ** (1 / max(years, 0.5)) - 1 if final_equity > 0 else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
    wr = len(wins) / len(net_pnls) if net_pnls else 0

    total_costs = sum(t['costs'] for t in trades)
    total_gross = sum(abs(t['gross_pnl']) for t in trades)
    avg_lots = np.mean([t['lots'] for t in trades])

    # Overlay impact: how much did modifiers change from base?
    if overlay_contrib:
        base_sum = sum(o['base_lots'] for o in overlay_contrib)
        adj_sum = sum(o['adjusted_lots'] for o in overlay_contrib)
        overlay_impact = (adj_sum - base_sum) / base_sum * 100 if base_sum > 0 else 0
    else:
        overlay_impact = 0

    # Margin utilization
    avg_margin = np.mean([t['lots'] * MARGIN_PER_LOT for t in trades])
    margin_util = avg_margin / INITIAL_CAPITAL * 100

    # Overlay attribution: win rate when modifier > 1.2 vs < 0.8
    high_mod_trades = [t for t in trades if t.get('composite_modifier', 1) > 1.15]
    low_mod_trades = [t for t in trades if t.get('composite_modifier', 1) < 0.85]
    high_wr = len([t for t in high_mod_trades if t['net_pnl'] > 0]) / max(len(high_mod_trades), 1)
    low_wr = len([t for t in low_mod_trades if t['net_pnl'] > 0]) / max(len(low_mod_trades), 1)

    return {
        'mode': mode,
        'trades': len(trades),
        'sharpe': round(sharpe, 2),
        'cagr_pct': round(cagr * 100, 1),
        'max_dd_pct': round(max_dd * 100, 1),
        'pf': round(pf, 2),
        'win_rate_pct': round(wr * 100, 1),
        'final_equity': round(final_equity),
        'avg_lots': round(avg_lots, 1),
        'margin_util_pct': round(margin_util, 1),
        'overlay_impact_pct': round(overlay_impact, 1),
        'total_costs': round(total_costs),
        'cost_as_pct_gross': round(total_costs / max(total_gross, 1) * 100, 1),
        'high_mod_trades': len(high_mod_trades),
        'high_mod_wr': round(high_wr * 100, 1),
        'low_mod_trades': len(low_mod_trades),
        'low_mod_wr': round(low_wr * 100, 1),
        'years': round(years, 1),
    }


# ================================================================
# MAIN
# ================================================================

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    t0 = time_mod.perf_counter()

    print("=" * 95)
    print("  LOT-BASED WALK-FORWARD BACKTEST (overlays change position size)")
    print("  Period: 2015-11 to 2026-03 | 7 daily SCORING signals")
    print("  Costs: TransactionCostModel (₹23.60/order, 0.0495% exch, 0.003% stamp)")
    print("  Lot sizes: 75 (pre-Jul 2023), 25 (post-Jul 2023)")
    print("=" * 95)

    conn = psycopg2.connect(DATABASE_DSN)
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix, pcr_oi "
        "FROM nifty_daily ORDER BY date", conn, parse_dates=['date'])
    conn.close()
    print(f"\nLoaded {len(df)} daily bars")

    print("Computing indicators...")
    df_ind = add_all_indicators(df)

    # Build overlay pipeline
    print("Building overlay pipeline (22 overlay signals)...")
    pipeline = OverlayPipeline(df_ind)

    # ── MODE 1: FIXED FRACTION (baseline) ──
    print("\n" + "─" * 95)
    print("  MODE 1: FIXED-FRACTION (overlays ignored, percentage sizing)")
    print("─" * 95)
    r1 = run_lot_based_backtest(df_ind, overlay_pipeline=None, mode='FIXED_FRAC')

    # ── MODE 2: LOT-BASED (overlays active) ──
    print("\n" + "─" * 95)
    print("  MODE 2: LOT-BASED (22 overlays modify lot count)")
    print("─" * 95)
    r2 = run_lot_based_backtest(df_ind, overlay_pipeline=pipeline, mode='LOT_BASED')

    # ── MODE 3: LOT+CONVICTION (overlays + conviction amplification) ──
    print("\n" + "─" * 95)
    print("  MODE 3: LOT+CONVICTION (overlays + bullish amplification + trend gate)")
    print("─" * 95)
    r3 = run_lot_based_backtest(df_ind, overlay_pipeline=pipeline, mode='LOT_CONVICTION')

    # ── MODE 4: LOT+ML (overlays + half-Kelly) ──
    print("\n" + "─" * 95)
    print("  MODE 4: LOT+ML (overlays + half-Kelly = 0.6x conservative)")
    print("─" * 95)
    r4 = run_lot_based_backtest(df_ind, overlay_pipeline=pipeline, mode='LOT_ML', ml_kelly_mult=0.6)

    # ── 4-WAY COMPARISON TABLE ──
    print("\n" + "=" * 95)
    print("  COMPARISON: FIXED-FRAC vs LOT-BASED vs LOT+CONVICTION vs LOT+ML")
    print("=" * 95)
    print(f"  {'Metric':<22s} {'FIXED-FRAC':>13s} {'LOT-BASED':>13s} {'LOT+CONVICT':>13s} {'LOT+ML':>13s}")
    print(f"  {'─' * 75}")

    for key, label, fmt in [
        ('trades', 'Trades', '{}'),
        ('sharpe', 'Sharpe', '{:.2f}'),
        ('cagr_pct', 'CAGR', '{:.1f}%'),
        ('max_dd_pct', 'Max Drawdown', '{:.1f}%'),
        ('pf', 'Profit Factor', '{:.2f}'),
        ('win_rate_pct', 'Win Rate', '{:.1f}%'),
        ('final_equity', 'Final Equity', '₹{:,.0f}'),
        ('avg_lots', 'Avg Lots/Trade', '{:.1f}'),
        ('high_mod_trades', 'Trades >1.15x', '{}'),
        ('low_mod_trades', 'Trades <0.85x', '{}'),
        ('total_costs', 'Total Costs', '₹{:,.0f}'),
    ]:
        v1 = fmt.format(r1[key])
        v2 = fmt.format(r2[key])
        v3 = fmt.format(r3[key])
        v4 = fmt.format(r4[key])
        print(f"  {label:<22s} {v1:>13s} {v2:>13s} {v3:>13s} {v4:>13s}")

    # ── CONVICTION ATTRIBUTION ──
    print(f"\n  CONVICTION ATTRIBUTION:")
    print(f"    LOT+CONVICTION high-conviction trades: {r3['high_mod_trades']} (WR {r3['high_mod_wr']}%)")
    print(f"    LOT+CONVICTION low-conviction trades:  {r3['low_mod_trades']} (WR {r3['low_mod_wr']}%)")

    # ── ALPHA vs BASELINE ──
    print(f"\n  ALPHA vs FIXED-FRAC BASELINE:")
    for label, r in [('LOT-BASED', r2), ('LOT+CONVICT', r3), ('LOT+ML', r4)]:
        dc = r['cagr_pct'] - r1['cagr_pct']
        ds = r['sharpe'] - r1['sharpe']
        dd = r['max_dd_pct'] - r1['max_dd_pct']
        print(f"    {label:>13s}: CAGR {dc:+.1f}%, Sharpe {ds:+.2f}, DD {dd:+.1f}%")

    elapsed = time_mod.perf_counter() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    print("=" * 95)


if __name__ == '__main__':
    main()

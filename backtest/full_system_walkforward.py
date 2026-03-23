"""
COMPLETE Walk-Forward Validation — All 12 Scoring Signals + Portfolio + Overlays + Capital Sensitivity.

Phase 1: Individual signal WF (7 daily + 5 structural/intraday)
Phase 2: Combined portfolio from passing signals
Phase 3: Overlay impact (with vs without 28 overlays)
Phase 4: Capital sensitivity (₹5L / ₹10L / ₹30L)

Usage:
    venv/bin/python3 -m backtest.full_system_walkforward
"""

import logging
import math
import time as time_mod
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import _eval_conditions
from backtest.indicators import add_all_indicators
from config.settings import DATABASE_DSN, NIFTY_LOT_SIZE
from execution.lot_sizer import LotSizer, get_lot_size, MARGIN_PER_LOT
from execution.overlay_pipeline import OverlayPipeline

logger = logging.getLogger(__name__)

# ── Config ──
COST_PER_TRADE = 100  # ₹100 round-trip (brokerage + STT + slippage)
SL_PCT = 0.015
TGT_PCT = 0.025
DAILY_MAX_HOLD = 5
INTRA_MAX_HOLD = 1
RISK_FREE = 0.065

# ── Daily signal rules (from signal_compute.py) ──
DAILY_SIGNALS = {
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


def load_data():
    conn = psycopg2.connect(DATABASE_DSN)
    daily = pd.read_sql(
        "SELECT date, open, high, low, close, volume, india_vix, pcr_oi "
        "FROM nifty_daily ORDER BY date", conn, parse_dates=['date'])
    bars = pd.read_sql(
        "SELECT timestamp, open, high, low, close, volume FROM intraday_bars "
        "WHERE instrument='NIFTY' ORDER BY timestamp", conn, parse_dates=['timestamp'])
    options = pd.read_sql(
        "SELECT date, expiry, strike, option_type, oi, close as premium "
        "FROM nifty_options WHERE oi > 0 ORDER BY date, strike",
        conn, parse_dates=['date', 'expiry'])
    conn.close()
    daily['date'] = daily['date'].dt.date
    bars['date'] = bars['timestamp'].dt.date
    options['date'] = options['date'].dt.date
    options['expiry'] = options['expiry'].dt.date
    return daily, bars, options


def simulate_daily_signal(sig_id, rules, df_ind):
    """Backtest one daily signal, return trade list."""
    entry_long = rules.get('entry_long', [])
    entry_short = rules.get('entry_short', [])
    exit_long = rules.get('exit_long', [])
    exit_short = rules.get('exit_short', [])
    direction = rules.get('direction', 'BOTH')
    sl = rules.get('stop_loss_pct', 0.02)
    tp = rules.get('take_profit_pct', 0)
    hold = rules.get('hold_days', 0)
    cd = rules.get('cooldown_days', 1)

    trades = []
    pos = None
    ep = 0; ei = 0; days = 0; last_exit = -cd
    closes = df_ind['close'].values
    dates = df_ind['date'].values if 'date' in df_ind.columns else df_ind.index.values
    n = len(df_ind)

    for i in range(1, n):
        row = df_ind.iloc[i]; prev = df_ind.iloc[i-1]; c = float(closes[i])
        lot_size = 75 if pd.Timestamp(dates[i]).year < 2023 or (pd.Timestamp(dates[i]).year == 2023 and pd.Timestamp(dates[i]).month < 7) else 25

        if pos:
            days += 1
            xr = None; xp = c
            if pos == 'LONG':
                if sl > 0 and c <= ep*(1-sl): xr, xp = 'SL', ep*(1-sl)
                elif tp > 0 and c >= ep*(1+tp): xr, xp = 'TP', ep*(1+tp)
                elif exit_long and _eval_conditions(row, prev, exit_long): xr = 'RULE'
                elif hold > 0 and days >= hold: xr = 'TIME'
            else:
                if sl > 0 and c >= ep*(1+sl): xr, xp = 'SL', ep*(1+sl)
                elif tp > 0 and c <= ep*(1-tp): xr, xp = 'TP', ep*(1-tp)
                elif exit_short and _eval_conditions(row, prev, exit_short): xr = 'RULE'
                elif hold > 0 and days >= hold: xr = 'TIME'
            if xr:
                pnl_pct = (xp-ep)/ep if pos=='LONG' else (ep-xp)/ep
                pnl_rs = pnl_pct * lot_size * ep - COST_PER_TRADE
                trades.append({'sig': sig_id, 'date': dates[ei], 'exit_date': dates[i],
                               'dir': pos, 'pnl_rs': round(pnl_rs), 'pnl_pct': pnl_pct,
                               'lot_size': lot_size, 'xr': xr})
                pos = None; last_exit = i
        else:
            if i - last_exit < cd: continue
            if direction in ('BOTH','LONG') and entry_long and _eval_conditions(row, prev, entry_long):
                pos='LONG'; ep=c; ei=i; days=0; continue
            if direction in ('BOTH','SHORT') and entry_short and _eval_conditions(row, prev, entry_short):
                pos='SHORT'; ep=c; ei=i; days=0
    return trades


def simulate_structural_signals(daily, bars, options):
    """Backtest structural/intraday signals using their backtest_evaluate() methods."""
    results = {}
    close_map = {r['date']: float(r['close']) for _, r in daily.iterrows()}
    vix_map = {r['date']: float(r['india_vix']) if pd.notna(r['india_vix']) else 15 for _, r in daily.iterrows()}
    dates = sorted(daily['date'].unique())

    # GIFT Convergence
    gift_trades = []
    prev_close_map = {}
    for i in range(1, len(dates)): prev_close_map[dates[i]] = close_map.get(dates[i-1], 0)

    for d in sorted(bars['date'].unique()):
        pc = prev_close_map.get(d, 0)
        if pc <= 0: continue
        vix = vix_map.get(d, 15)
        if vix > 25: continue
        session = bars[bars['date']==d].sort_values('timestamp')
        if len(session) < 6: continue
        gap = float(session.iloc[0]['open']) - pc
        if abs(gap) < 50 or abs(gap) > pc*0.02: continue
        d_dir = 'SHORT' if gap > 0 else 'LONG'
        entry = float(session.iloc[0]['open']) + (1 if d_dir=='SHORT' else -1)
        sl_d = abs(gap)*0.5; tgt_d = abs(gap)*0.7
        xp = entry; xr = 'TIME'
        for j in range(1, min(6, len(session))):
            b = session.iloc[j]
            if d_dir=='LONG' and float(b['low'])<=entry-sl_d: xp=entry-sl_d; xr='SL'; break
            if d_dir=='LONG' and float(b['high'])>=entry+tgt_d: xp=entry+tgt_d; xr='TGT'; break
            if d_dir=='SHORT' and float(b['high'])>=entry+sl_d: xp=entry+sl_d; xr='SL'; break
            if d_dir=='SHORT' and float(b['low'])<=entry-tgt_d: xp=entry-tgt_d; xr='TGT'; break
            if j==min(5,len(session)-1): xp=float(session.iloc[j]['close'])
        pnl = ((xp-entry) if d_dir=='LONG' else (entry-xp)) * 25 - COST_PER_TRADE
        gift_trades.append({'sig':'GIFT_CONVERGENCE','date':d,'pnl_rs':round(pnl),'dir':d_dir,'xr':xr})
    results['GIFT_CONVERGENCE'] = gift_trades

    # EXPIRY_PIN_FADE + ORR_REVERSION (from india_intraday_wf results)
    # Use the already-validated results
    results['EXPIRY_PIN_FADE'] = []  # 45 trades, but need actual bars
    results['ORR_REVERSION'] = []

    for d in sorted(bars['date'].unique()):
        pc = prev_close_map.get(d, 0)
        if pc <= 0: continue
        session = bars[bars['date']==d].sort_values('timestamp')
        if len(session) < 10: continue

        # ORR: gap fade
        gap_pct = (float(session.iloc[0]['open']) - pc) / pc
        if abs(gap_pct) > 0.005:
            entry = float(session.iloc[0]['close'])
            d_dir = 'SHORT' if gap_pct > 0 else 'LONG'
            mid = session.iloc[len(session)//4:len(session)//2]
            if len(mid) > 0:
                xp = float(mid.iloc[-1]['close'])
                pnl = ((xp-entry) if d_dir=='LONG' else (entry-xp)) * 25 - COST_PER_TRADE
                results.setdefault('ORR_REVERSION',[]).append({'sig':'ORR_REVERSION','date':d,'pnl_rs':round(pnl),'dir':d_dir,'xr':'REVERT'})

        # PIN_FADE: Thursday expiry pinning
        if hasattr(d,'weekday') and d.weekday()==3:
            afternoon = session[session['timestamp'].dt.hour >= 14]
            if len(afternoon) >= 3:
                pin_entry = float(afternoon.iloc[0]['close'])
                pin_exit = float(afternoon.iloc[-1]['close'])
                nearest = round(pin_entry/100)*100
                if abs(pin_entry-nearest)/pin_entry < 0.002:
                    pnl = abs(pin_entry-pin_exit)*25 - COST_PER_TRADE if abs(pin_exit-nearest) < abs(pin_entry-nearest) else -abs(pin_entry-pin_exit)*25 - COST_PER_TRADE
                    results.setdefault('EXPIRY_PIN_FADE',[]).append({'sig':'EXPIRY_PIN_FADE','date':d,'pnl_rs':round(pnl),'dir':'FADE','xr':'PIN'})

    # MAX_OI_BARRIER
    oi_trades = []
    from datetime import time
    for d in sorted(bars['date'].unique()):
        day_opts = options[options['date']==d]
        if day_opts.empty: continue
        spot = close_map.get(d, 0)
        if spot <= 0: continue
        puts = day_opts[day_opts['option_type']=='PE'].groupby('strike')['oi'].sum()
        calls = day_opts[day_opts['option_type']=='CE'].groupby('strike')['oi'].sum()
        if puts.empty or calls.empty: continue
        valid_puts = puts[(puts.index >= spot*0.97) & (puts.index <= spot)]
        if valid_puts.empty or valid_puts.max() < 5_000_000: continue
        floor_strike = valid_puts.idxmax()
        session = bars[bars['date']==d]
        low = float(session['low'].min())
        if low > floor_strike + 30: continue
        entry = float(floor_strike + 15)
        xp = float(session.iloc[-5]['close']) if len(session) > 5 else spot
        pnl = (xp - entry - 2) * 25 - COST_PER_TRADE
        oi_trades.append({'sig':'MAX_OI_BARRIER','date':d,'pnl_rs':round(pnl),'dir':'LONG','xr':'OI_BOUNCE'})
    results['MAX_OI_BARRIER'] = oi_trades

    # MONDAY_STRADDLE
    straddle_trades = []
    for d in dates:
        if not hasattr(d,'weekday') or d.weekday() != 0: continue
        vix = vix_map.get(d, 15)
        if vix > 20: continue
        day_opts = options[options['date']==d]
        if day_opts.empty: continue
        spot = close_map.get(d, 0)
        if spot <= 0: continue
        atm = round(spot/50)*50
        ce = day_opts[(day_opts['strike']==atm) & (day_opts['option_type']=='CE')]
        pe = day_opts[(day_opts['strike']==atm) & (day_opts['option_type']=='PE')]
        if ce.empty or pe.empty: continue
        credit = float(ce.iloc[0]['premium']) + float(pe.iloc[0]['premium'])
        if credit < 150: continue
        # Check Monday close vs open for move size
        day_bars = bars[bars['date']==d]
        if day_bars.empty:
            move = abs(close_map.get(d, spot) - spot) / spot
        else:
            move = max(abs(float(day_bars['high'].max())-spot), abs(float(day_bars['low'].min())-spot)) / spot
        if move < 0.01: pnl = credit * 0.50 * 25 - 160
        elif move < 0.015: pnl = credit * 0.20 * 25 - 160
        else: pnl = -credit * 0.50 * 25 - 160
        straddle_trades.append({'sig':'MONDAY_STRADDLE','date':d,'pnl_rs':round(pnl),'dir':'STRADDLE','xr':'THETA'})
    results['MONDAY_STRADDLE'] = straddle_trades

    return results


def wf_evaluate(trades, train_days=252, test_days=63, step_days=63, min_trades=3):
    """Walk-forward evaluate a trade list."""
    if not trades: return {'trades':0,'wr':0,'pf':0,'total':0,'pass_rate':0,'verdict':'NO_TRADES','windows':[]}
    trade_dates = sorted(set(t['date'] for t in trades))
    if not trade_dates: return {'trades':0,'wr':0,'pf':0,'total':0,'pass_rate':0,'verdict':'NO_TRADES','windows':[]}

    all_dates = trade_dates
    windows = []
    idx = 0
    while idx + train_days < len(all_dates):
        ts_idx = min(idx + train_days, len(all_dates)-1)
        te_idx = min(ts_idx + test_days, len(all_dates)-1)
        if ts_idx >= te_idx: break
        windows.append((all_dates[ts_idx], all_dates[te_idx]))
        idx += step_days

    if not windows:
        pnls = [t['pnl_rs'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins)/len(pnls); pf = sum(wins)/abs(sum(losses)) if losses and sum(losses)!=0 else 0
        return {'trades':len(trades),'wr':wr,'pf':pf,'total':sum(pnls),'pass_rate':0,'passes':0,'windows':0,'verdict':'NO_WINDOWS'}

    passes = 0; evaluated = 0
    for ts, te in windows:
        wt = [t for t in trades if ts <= t['date'] <= te]
        if len(wt) < min_trades: continue
        evaluated += 1
        pnls = [t['pnl_rs'] for t in wt]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins)/len(pnls); pf = sum(wins)/abs(sum(losses)) if losses and sum(losses)!=0 else 0
        if wr >= 0.50 and pf >= 1.20: passes += 1

    all_pnls = [t['pnl_rs'] for t in trades]
    all_wins = [p for p in all_pnls if p > 0]
    all_losses = [p for p in all_pnls if p <= 0]
    rate = passes/evaluated if evaluated > 0 else 0

    return {
        'trades': len(trades),
        'wr': len(all_wins)/len(all_pnls) if all_pnls else 0,
        'pf': sum(all_wins)/abs(sum(all_losses)) if all_losses and sum(all_losses)!=0 else 0,
        'total': sum(all_pnls),
        'pass_rate': rate,
        'passes': passes,
        'windows': evaluated,
        'verdict': 'PASS' if rate >= 0.60 else 'FAIL',
    }


def portfolio_metrics(trades, capital):
    """Compute portfolio-level metrics from combined trade list."""
    if not trades: return {}
    equity = float(capital); peak = equity; max_dd = 0
    monthly = {}
    for t in sorted(trades, key=lambda x: str(x['date'])):
        equity += t['pnl_rs']; peak = max(peak, equity)
        dd = (peak-equity)/peak; max_dd = max(max_dd, dd)
        m = str(t['date'])[:7]
        monthly[m] = monthly.get(m, 0) + t['pnl_rs']

    total = sum(t['pnl_rs'] for t in trades)
    years = len(set(str(t['date'])[:4] for t in trades))
    years = max(years, 1)
    cagr = ((equity/capital)**(1/years) - 1) * 100 if equity > 0 else 0
    calmar = cagr / (max_dd*100) if max_dd > 0 else 0

    daily_pnl = {}
    for t in trades:
        d = str(t['date'])[:10]
        daily_pnl[d] = daily_pnl.get(d, 0) + t['pnl_rs']
    rets = np.array(list(daily_pnl.values())) / capital
    sharpe = (np.mean(rets)-RISK_FREE/252)/(np.std(rets,ddof=1))*np.sqrt(252) if len(rets)>1 and np.std(rets)>0 else 0

    wins = [t for t in trades if t['pnl_rs'] > 0]
    monthly_vals = list(monthly.values())

    return {
        'cagr': round(cagr, 1),
        'max_dd': round(max_dd*100, 1),
        'calmar': round(calmar, 2),
        'sharpe': round(sharpe, 2),
        'wr': round(len(wins)/len(trades)*100, 1),
        'pf': round(sum(t['pnl_rs'] for t in wins)/abs(sum(t['pnl_rs'] for t in trades if t['pnl_rs']<=0)), 2) if any(t['pnl_rs']<=0 for t in trades) else 0,
        'trades': len(trades),
        'trades_yr': round(len(trades)/max(years,1)),
        'total_pnl': round(total),
        'final_equity': round(equity),
        'worst_month': round(min(monthly_vals)) if monthly_vals else 0,
        'best_month': round(max(monthly_vals)) if monthly_vals else 0,
    }


def main():
    logging.basicConfig(level=logging.WARNING)
    t0 = time_mod.perf_counter()

    print("╔" + "═"*78 + "╗")
    print("║" + "FULL SYSTEM WALK-FORWARD VALIDATION".center(78) + "║")
    print("╠" + "═"*78 + "╣")
    print()

    daily, bars, options = load_data()
    df_ind = add_all_indicators(daily)
    pipeline = OverlayPipeline(df_ind)
    print(f"  Data: {len(daily)} daily, {len(bars)} intraday, {len(options)} options rows")

    # ═══ PHASE 1: Individual Signal WF ═══
    print(f"\n{'─'*78}")
    print("  PHASE 1: Individual Signal Walk-Forward")
    print(f"{'─'*78}")

    all_signal_trades = {}

    # Daily signals
    for sig_id, rules in DAILY_SIGNALS.items():
        trades = simulate_daily_signal(sig_id, rules, df_ind)
        all_signal_trades[sig_id] = trades
        wf = wf_evaluate(trades, train_days=252, test_days=63, step_days=63, min_trades=5)
        all_signal_trades[sig_id + '_wf'] = wf

    # Structural signals
    struct = simulate_structural_signals(daily, bars, options)
    for sig_id, trades in struct.items():
        all_signal_trades[sig_id] = trades
        wf = wf_evaluate(trades, train_days=126, test_days=63, step_days=63, min_trades=3)
        all_signal_trades[sig_id + '_wf'] = wf

    # Print Phase 1 table
    print(f"\n  {'Signal':<22s} {'Trades':>6s} {'WR':>5s} {'PF':>6s} {'Total PnL':>11s} {'WF%':>5s} {'W/T':>5s} {'Verdict':>8s}")
    print(f"  {'─'*68}")

    passing_signals = []
    for sig_id in list(DAILY_SIGNALS.keys()) + list(struct.keys()):
        wf = all_signal_trades.get(sig_id + '_wf', {})
        trades = all_signal_trades.get(sig_id, [])
        v = wf.get('verdict', 'N/A')
        marker = '✓' if v == 'PASS' else '✗'
        pr = wf.get('pass_rate', 0)
        w = wf.get('passes', 0)
        t = wf.get('windows', 0)
        print(f"  {sig_id:<22s} {wf.get('trades',0):>6d} {wf.get('wr',0):>4.0%} {wf.get('pf',0):>5.2f} "
              f"₹{wf.get('total',0):>9,} {pr:>4.0%} {w:>2d}/{t:<2d} {v+' '+marker:>8s}")
        if v == 'PASS':
            passing_signals.append(sig_id)

    print(f"\n  Passing: {len(passing_signals)}/{len(DAILY_SIGNALS)+len(struct)} signals")

    # ═══ PHASE 2: Combined Portfolio ═══
    print(f"\n{'─'*78}")
    print("  PHASE 2: Combined Portfolio (all passing signals)")
    print(f"{'─'*78}")

    combined = []
    for sig_id in passing_signals:
        combined.extend(all_signal_trades.get(sig_id, []))
    combined.sort(key=lambda t: str(t['date']))

    for capital in [10_00_000]:
        pm = portfolio_metrics(combined, capital)
        print(f"\n  ┌{'─'*35}┬{'─'*20}┐")
        print(f"  │ {'Metric':<33s} │ {'Value':>18s} │")
        print(f"  ├{'─'*35}┼{'─'*20}┤")
        for k, label in [('cagr','CAGR'), ('max_dd','Max Drawdown'), ('calmar','Calmar'),
                          ('sharpe','Sharpe'), ('wr','Win Rate'), ('pf','Profit Factor'),
                          ('trades','Total Trades'), ('trades_yr','Trades/Year'),
                          ('total_pnl','Total P&L'), ('worst_month','Worst Month')]:
            v = pm.get(k, 0)
            if k in ('cagr','max_dd','wr'): vs = f"{v}%"
            elif k in ('total_pnl','worst_month','best_month','final_equity'): vs = f"₹{v:,}"
            else: vs = str(v)
            print(f"  │ {label:<33s} │ {vs:>18s} │")
        print(f"  └{'─'*35}┴{'─'*20}┘")

    # ═══ PHASE 3: Overlay Impact ═══
    print(f"\n{'─'*78}")
    print("  PHASE 3: Overlay Impact (with vs without)")
    print(f"{'─'*78}")

    # Without overlays
    pm_no = portfolio_metrics(combined, 10_00_000)

    # With overlays: apply overlay modifiers to scale P&L
    combined_overlay = []
    for t in combined:
        d = t['date']
        try:
            mods = pipeline.get_modifiers(d, t.get('dir', 'LONG'))
            composite = 1.0
            for v in mods.values():
                composite *= max(0.5, min(1.5, v))
            composite = max(0.5, min(1.5, composite ** 0.3))  # dampened
        except: composite = 1.0
        t_copy = dict(t)
        t_copy['pnl_rs'] = round(t['pnl_rs'] * composite)
        combined_overlay.append(t_copy)

    pm_ov = portfolio_metrics(combined_overlay, 10_00_000)

    sized_up = sum(1 for t1, t2 in zip(combined, combined_overlay) if t2['pnl_rs'] > t1['pnl_rs'])
    sized_down = sum(1 for t1, t2 in zip(combined, combined_overlay) if t2['pnl_rs'] < t1['pnl_rs'])

    print(f"\n  ┌{'─'*14}┬{'─'*14}┬{'─'*14}┬{'─'*10}┐")
    print(f"  │ {'Metric':<12s} │ {'No Overlay':>12s} │ {'With Overlay':>12s} │ {'Delta':>8s} │")
    print(f"  ├{'─'*14}┼{'─'*14}┼{'─'*14}┼{'─'*10}┤")
    for k, label in [('cagr','CAGR'), ('max_dd','Max DD'), ('calmar','Calmar'), ('sharpe','Sharpe')]:
        v1 = pm_no.get(k, 0); v2 = pm_ov.get(k, 0)
        delta = v2 - v1
        s = '%' if k in ('cagr','max_dd') else ''
        print(f"  │ {label:<12s} │ {v1:>11}{s} │ {v2:>11}{s} │ {delta:>+7.1f}{s} │")
    print(f"  └{'─'*14}┴{'─'*14}┴{'─'*14}┴{'─'*10}┘")
    print(f"  Trades sized UP: {sized_up} | Trades sized DOWN: {sized_down}")

    # ═══ PHASE 4: Capital Sensitivity ═══
    print(f"\n{'─'*78}")
    print("  PHASE 4: Capital Sensitivity")
    print(f"{'─'*78}")

    print(f"\n  ┌{'─'*11}┬{'─'*8}┬{'─'*8}┬{'─'*8}┬{'─'*18}┬{'─'*14}┐")
    print(f"  │ {'Capital':>9s} │ {'CAGR':>6s} │ {'MaxDD':>6s} │ {'Calmar':>6s} │ {'Total P&L':>16s} │ {'Final Equity':>12s} │")
    print(f"  ├{'─'*11}┼{'─'*8}┼{'─'*8}┼{'─'*8}┼{'─'*18}┼{'─'*14}┤")
    for cap in [5_00_000, 10_00_000, 30_00_000]:
        # Scale P&L by capital ratio (lot quantization)
        scale = cap / 10_00_000
        lots = max(1, math.floor(scale * 1.6))  # base 1.6 lots at 10L
        lot_scale = lots / 1.6
        scaled = [dict(t, pnl_rs=round(t['pnl_rs'] * lot_scale)) for t in combined]
        pm = portfolio_metrics(scaled, cap)
        print(f"  │ ₹{cap/100000:>5.0f}L │ {pm['cagr']:>5.1f}% │ {pm['max_dd']:>5.1f}% │ {pm['calmar']:>6.2f} │ ₹{pm['total_pnl']:>14,} │ ₹{pm['final_equity']:>10,} │")
    print(f"  └{'─'*11}┴{'─'*8}┴{'─'*8}┴{'─'*8}┴{'─'*18}┴{'─'*14}┘")

    elapsed = time_mod.perf_counter() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("╚" + "═"*78 + "╝")


if __name__ == '__main__':
    main()

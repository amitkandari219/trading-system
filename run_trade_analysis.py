"""
Complete trade-level analysis across all signals and combinations.
Every single trade, with compounding, realistic costs, and full statistics.
"""

import json
import os
import csv
from collections import defaultdict
from datetime import date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import psycopg2

from backtest.indicators import add_all_indicators, historical_volatility
from config.settings import DATABASE_DSN

# ================================================================
# CONSTANTS
# ================================================================
CAPITAL = 1_000_000
LOT_SIZE = 75
LOT_MARGIN = 100_000
BASE_LOTS = 7
MAX_LOTS = 50
COMPOUND_EVERY = 25
SLIPPAGE_PTS = 2  # 1 entry + 1 exit


def compute_cost(entry_price, exit_price, lots):
    """Exact transaction cost for one round trip."""
    buy_turnover = entry_price * lots * LOT_SIZE
    sell_turnover = exit_price * lots * LOT_SIZE
    total_turnover = buy_turnover + sell_turnover

    brokerage = 40  # ₹20 each way
    stt = sell_turnover * 0.000125
    exchange = total_turnover * 0.000019
    sebi = total_turnover * 0.000001
    stamp = buy_turnover * 0.00002
    gst = brokerage * 0.18

    return round(brokerage + stt + exchange + sebi + stamp + gst, 2)


# ================================================================
# SIGNAL CONDITION ENGINE (reused from run_multi_period.py)
# ================================================================

def _eval_cond(row, prev, cond):
    ind = cond.get('indicator', '')
    op = cond.get('op', '>')
    val = cond.get('value')
    if ind not in row.index: return False
    iv = row[ind]
    if pd.isna(iv): return False
    if isinstance(val, str) and val in row.index:
        tv = row[val]
        if pd.isna(tv): return False
    elif val is None: return False
    else:
        try: tv = float(val)
        except: return False
    if op == '>': return iv > tv
    elif op == '<': return iv < tv
    elif op == '>=': return iv >= tv
    elif op == '<=': return iv <= tv
    elif op == 'crosses_above':
        if prev is None: return False
        pi = prev.get(ind, np.nan)
        pt = prev.get(val, tv) if isinstance(val, str) else tv
        return not pd.isna(pi) and not pd.isna(pt) and pi <= pt and iv > tv
    elif op == 'crosses_below':
        if prev is None: return False
        pi = prev.get(ind, np.nan)
        pt = prev.get(val, tv) if isinstance(val, str) else tv
        return not pd.isna(pi) and not pd.isna(pt) and pi >= pt and iv < tv
    return False

def _eval_conds(row, prev, conds):
    return bool(conds) and all(_eval_cond(row, prev, c) for c in conds)


# ================================================================
# BACKTESTER WITH COMPOUNDING
# ================================================================

def backtest_with_compounding(df, rules, capital=CAPITAL, base_lots=BASE_LOTS,
                               stop_pct=0.02, tp_pct=0.0, hold_max=0):
    """Full backtest returning detailed trade log with compounding."""
    n = len(df)
    if n < 20: return []

    entry_long = rules.get('entry_long', [])
    entry_short = rules.get('entry_short', [])
    exit_long = rules.get('exit_long', [])
    exit_short = rules.get('exit_short', [])

    closes = df['close'].values
    opens = df['open'].values
    dates = df['date'].values
    regimes = df['regime'].values if 'regime' in df.columns else ['UNKNOWN'] * n
    vix_vals = df['india_vix'].values if 'india_vix' in df.columns else [0] * n

    trades = []
    pos = None
    ep = 0; ei = 0
    current_capital = capital
    lots = base_lots
    trade_count_since_compound = 0

    for i in range(1, n):
        row = df.iloc[i]; prev = df.iloc[i-1]

        if pos is None:
            direction = None
            if _eval_conds(row, prev, entry_long):
                direction = 'LONG'
            elif _eval_conds(row, prev, entry_short):
                direction = 'SHORT'

            if direction:
                pos = direction
                ep = closes[i]  # entry on close
                ei = i
        else:
            p = closes[i]; dh = i - ei

            # Stop loss
            if pos == 'LONG': lp = (ep - p) / ep
            else: lp = (p - ep) / ep

            exit_reason = None
            if lp >= stop_pct:
                exit_reason = 'stop_loss'
            elif tp_pct > 0:
                gp = (p - ep) / ep if pos == 'LONG' else (ep - p) / ep
                if gp >= tp_pct:
                    exit_reason = 'take_profit'
            if not exit_reason and hold_max > 0 and dh >= hold_max:
                exit_reason = 'hold_max'
            if not exit_reason:
                ex = exit_long if pos == 'LONG' else exit_short
                if _eval_conds(row, prev, ex):
                    exit_reason = 'signal_exit'

            if exit_reason:
                exit_price = p
                if pos == 'LONG':
                    pts = exit_price - ep - SLIPPAGE_PTS
                else:
                    pts = ep - exit_price - SLIPPAGE_PTS

                cost = compute_cost(ep, exit_price, lots)
                gross_pnl = pts * LOT_SIZE * lots
                net_pnl = gross_pnl - cost
                current_capital += net_pnl

                trades.append({
                    'trade_num': len(trades) + 1,
                    'entry_date': str(dates[ei])[:10],
                    'exit_date': str(dates[i])[:10],
                    'hold_days': dh,
                    'direction': pos,
                    'entry_price': round(ep, 2),
                    'exit_price': round(exit_price, 2),
                    'pts': round(pts, 1),
                    'gross_pnl': round(gross_pnl, 0),
                    'cost': round(cost, 0),
                    'net_pnl': round(net_pnl, 0),
                    'lots': lots,
                    'cumul_pnl': round(current_capital - capital, 0),
                    'capital': round(current_capital, 0),
                    'reason': exit_reason,
                    'regime': str(regimes[ei]),
                    'vix': round(float(vix_vals[ei]), 1) if not np.isnan(vix_vals[ei]) else 0,
                })

                pos = None
                trade_count_since_compound += 1

                # Compound every N trades
                if trade_count_since_compound >= COMPOUND_EVERY:
                    new_lots = max(1, min(MAX_LOTS, int(current_capital / LOT_MARGIN)))
                    if new_lots != lots:
                        lots = new_lots
                    trade_count_since_compound = 0

    return trades


def backtest_scoring_compound(df, sig_rules, capital=CAPITAL, base_lots=BASE_LOTS):
    """Scoring system with compounding."""
    n = len(df); closes = df['close'].values; dates = df['date'].values
    regimes = df['regime'].values if 'regime' in df.columns else ['UNKNOWN'] * n
    vix_vals = df['india_vix'].values if 'india_vix' in df.columns else [0] * n
    trades = []; pos = None; ep = 0; ei = 0; entry_thresh = 0; size = 1.0
    current_capital = capital; lots = base_lots; tsc = 0

    for i in range(1, n):
        row = df.iloc[i]; prev = df.iloc[i-1]
        score = 0
        d20l = _eval_conds(row, prev, sig_rules['DRY_20']['entry_long'])
        d20x = _eval_conds(row, prev, sig_rules['DRY_20']['exit_long'])
        d12l = _eval_conds(row, prev, sig_rules['DRY_12']['entry_long'])
        d12s = _eval_conds(row, prev, sig_rules['DRY_12']['entry_short'])
        d16l = _eval_conds(row, prev, sig_rules['DRY_16']['entry_long'])
        d16s = _eval_conds(row, prev, sig_rules['DRY_16']['entry_short'])
        if d20x: score = 0
        else:
            if d20l: score += 2
            if d12l: score += 1
            if d12s: score -= 1
            if d16l: score += 1
            if d16s: score -= 1

        if pos is None:
            if score >= 3: pos='LONG'; ep=closes[i]; ei=i; entry_thresh=3; size=1.0
            elif score >= 2: pos='LONG'; ep=closes[i]; ei=i; entry_thresh=2; size=0.5
            elif score <= -2: pos='SHORT'; ep=closes[i]; ei=i; entry_thresh=-2; size=0.5
        else:
            p = closes[i]
            if pos=='LONG': lp=(ep-p)/ep
            else: lp=(p-ep)/ep
            exit_reason = None
            if lp >= 0.02: exit_reason = 'stop_loss'
            elif pos=='LONG' and score < entry_thresh: exit_reason = 'score_drop'
            elif pos=='SHORT' and score > entry_thresh: exit_reason = 'score_rise'

            if exit_reason:
                if pos=='LONG': pts = p - ep - SLIPPAGE_PTS
                else: pts = ep - p - SLIPPAGE_PTS
                effective_lots = max(1, int(lots * size))
                cost = compute_cost(ep, p, effective_lots)
                gross = pts * LOT_SIZE * effective_lots
                net = gross - cost
                current_capital += net
                trades.append({
                    'trade_num': len(trades)+1,
                    'entry_date': str(dates[ei])[:10], 'exit_date': str(dates[i])[:10],
                    'hold_days': i-ei, 'direction': pos,
                    'entry_price': round(ep,2), 'exit_price': round(p,2),
                    'pts': round(pts,1), 'gross_pnl': round(gross,0),
                    'cost': round(cost,0), 'net_pnl': round(net,0),
                    'lots': effective_lots, 'cumul_pnl': round(current_capital-capital,0),
                    'capital': round(current_capital,0), 'reason': exit_reason,
                    'regime': str(regimes[ei]), 'vix': round(float(vix_vals[ei]),1) if not np.isnan(vix_vals[ei]) else 0,
                })
                pos = None; tsc += 1
                if tsc >= COMPOUND_EVERY:
                    lots = max(1, min(MAX_LOTS, int(current_capital / LOT_MARGIN)))
                    tsc = 0
    return trades


def backtest_seq5_compound(df, grimes_rules, kaufman_rules, capital=CAPITAL, base_lots=BASE_LOTS):
    """GRIMES → KAUFMAN SEQ_5 with compounding."""
    n = len(df); closes = df['close'].values; dates = df['date'].values
    regimes = df['regime'].values if 'regime' in df.columns else ['UNKNOWN'] * n
    vix_vals = df['india_vix'].values if 'india_vix' in df.columns else [0] * n
    trades = []; pos = None; ep = 0; ei = 0; pending_l = -999; pending_s = -999
    current_capital = capital; lots = base_lots; tsc = 0; pos_days = 0

    for i in range(1, n):
        row = df.iloc[i]; prev = df.iloc[i-1]
        regime = str(regimes[i]) if i < len(regimes) else 'UNKNOWN'
        trending = regime in ('TRENDING_UP', 'TRENDING_DOWN', 'TRENDING')

        if pos:
            pos_days += 1; p = closes[i]
            if pos == 'LONG': lp = (ep-p)/ep
            else: lp = (p-ep)/ep
            exit_reason = None
            if lp >= 0.02: exit_reason = 'stop_loss'
            elif pos_days >= 10: exit_reason = 'hold_max'
            elif pos == 'LONG' and row['low'] < prev['low']: exit_reason = 'structure'
            elif pos == 'SHORT' and row['high'] > prev['high']: exit_reason = 'structure'
            if exit_reason:
                pts = (p-ep-SLIPPAGE_PTS) if pos=='LONG' else (ep-p-SLIPPAGE_PTS)
                cost = compute_cost(ep, p, lots)
                gross = pts * LOT_SIZE * lots; net = gross - cost
                current_capital += net
                trades.append({
                    'trade_num': len(trades)+1,
                    'entry_date': str(dates[ei])[:10], 'exit_date': str(dates[i])[:10],
                    'hold_days': pos_days, 'direction': pos,
                    'entry_price': round(ep,2), 'exit_price': round(p,2),
                    'pts': round(pts,1), 'gross_pnl': round(gross,0),
                    'cost': round(cost,0), 'net_pnl': round(net,0),
                    'lots': lots, 'cumul_pnl': round(current_capital-capital,0),
                    'capital': round(current_capital,0), 'reason': exit_reason,
                    'regime': str(regimes[ei]), 'vix': round(float(vix_vals[ei]),1) if not np.isnan(vix_vals[ei]) else 0,
                })
                pos = None; tsc += 1
                if tsc >= COMPOUND_EVERY:
                    lots = max(1, min(MAX_LOTS, int(current_capital / LOT_MARGIN)))
                    tsc = 0
            continue

        adx_ok = pd.notna(row.get('adx_14')) and float(row['adx_14']) > 25
        gl = row['high']>prev['high'] and row['low']>prev['low'] and adx_ok and trending
        gs = row['low']<prev['low'] and row['high']<prev['high'] and adx_ok and trending
        if gl: pending_l = i; pending_s = -999
        if gs: pending_s = i; pending_l = -999
        if pending_l >= 0 and (i - pending_l) > 5: pending_l = -999
        if pending_s >= 0 and (i - pending_s) > 5: pending_s = -999
        kl = row['close'] > prev['close'] and row['volume'] < prev['volume']
        ks = row['close'] < prev['close'] and row['volume'] > prev['volume']
        if pending_l >= 0 and pending_l != i and kl:
            pos = 'LONG'; ep = closes[i]; ei = i; pos_days = 0; pending_l = -999
        elif pending_s >= 0 and pending_s != i and ks:
            pos = 'SHORT'; ep = closes[i]; ei = i; pos_days = 0; pending_s = -999
    return trades


def backtest_and_compound(df, rules_a, rules_b, capital=CAPITAL, base_lots=BASE_LOTS):
    """AND combination with compounding."""
    n = len(df); closes = df['close'].values; dates = df['date'].values
    regimes = df['regime'].values if 'regime' in df.columns else ['UNKNOWN'] * n
    vix_vals = df['india_vix'].values if 'india_vix' in df.columns else [0] * n
    trades = []; pos = None; ep = 0; ei = 0
    current_capital = capital; lots = base_lots; tsc = 0

    for i in range(1, n):
        row = df.iloc[i]; prev = df.iloc[i-1]
        if pos is None:
            al = _eval_conds(row, prev, rules_a.get('entry_long', []))
            bl = _eval_conds(row, prev, rules_b.get('entry_long', []))
            ash = _eval_conds(row, prev, rules_a.get('entry_short', []))
            bsh = _eval_conds(row, prev, rules_b.get('entry_short', []))
            if al and bl: pos = 'LONG'; ep = closes[i]; ei = i
            elif ash and bsh: pos = 'SHORT'; ep = closes[i]; ei = i
        else:
            p = closes[i]
            if pos=='LONG': lp=(ep-p)/ep
            else: lp=(p-ep)/ep
            exit_reason = None
            if lp >= 0.02: exit_reason = 'stop_loss'
            else:
                exl = _eval_conds(row, prev, rules_a.get('exit_long',[])) or _eval_conds(row, prev, rules_b.get('exit_long',[]))
                exs = _eval_conds(row, prev, rules_a.get('exit_short',[])) or _eval_conds(row, prev, rules_b.get('exit_short',[]))
                if (pos=='LONG' and exl) or (pos=='SHORT' and exs):
                    exit_reason = 'signal_exit'
            if exit_reason:
                pts = (p-ep-SLIPPAGE_PTS) if pos=='LONG' else (ep-p-SLIPPAGE_PTS)
                cost = compute_cost(ep, p, lots); gross = pts*LOT_SIZE*lots; net = gross-cost
                current_capital += net
                trades.append({
                    'trade_num': len(trades)+1,
                    'entry_date': str(dates[ei])[:10], 'exit_date': str(dates[i])[:10],
                    'hold_days': i-ei, 'direction': pos,
                    'entry_price': round(ep,2), 'exit_price': round(p,2),
                    'pts': round(pts,1), 'gross_pnl': round(gross,0),
                    'cost': round(cost,0), 'net_pnl': round(net,0),
                    'lots': lots, 'cumul_pnl': round(current_capital-capital,0),
                    'capital': round(current_capital,0), 'reason': exit_reason,
                    'regime': str(regimes[ei]), 'vix': round(float(vix_vals[ei]),1) if not np.isnan(vix_vals[ei]) else 0,
                })
                pos = None; tsc += 1
                if tsc >= COMPOUND_EVERY:
                    lots = max(1, min(MAX_LOTS, int(current_capital / LOT_MARGIN)))
                    tsc = 0
    return trades


# ================================================================
# STATISTICS
# ================================================================

def compute_stats(trades, data_years, capital=CAPITAL):
    """Compute all statistics from trade list."""
    if not trades:
        return {'total_trades': 0, 'final_capital': capital}

    net_pnls = [t['net_pnl'] for t in trades]
    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p < 0]

    # Capital curve
    cap = [capital]
    for t in trades:
        cap.append(cap[-1] + t['net_pnl'])
    cap_series = pd.Series(cap)
    peak = cap_series.cummax()
    dd = (cap_series - peak) / peak
    max_dd = abs(dd.min())
    max_dd_idx = dd.idxmin()

    # Find drawdown periods
    in_dd = False; dd_start = 0
    drawdowns = []
    for i in range(len(cap)):
        if cap[i] < peak.iloc[i] and not in_dd:
            in_dd = True; dd_start = i
        elif cap[i] >= peak.iloc[i] and in_dd:
            in_dd = False
            trough_idx = dd.iloc[dd_start:i].idxmin()
            drawdowns.append({
                'start_trade': dd_start,
                'trough_trade': trough_idx,
                'end_trade': i,
                'depth_pct': round(abs(dd.iloc[trough_idx]) * 100, 1),
            })

    final_cap = trades[-1]['capital']
    total_return = (final_cap / capital - 1) * 100
    cagr = ((final_cap / capital) ** (1 / data_years) - 1) * 100 if final_cap > 0 else -100

    # Sharpe from trade returns
    ret = pd.Series([p / capital for p in net_pnls])
    std = ret.std()
    sharpe = (ret.mean() / std * np.sqrt(252)) if std > 0 else 0
    down_std = ret[ret < 0].std()
    sortino = (ret.mean() / down_std * np.sqrt(252)) if down_std and down_std > 0 else 0
    calmar = cagr / (max_dd * 100) if max_dd > 0 else 0

    # Streaks
    streak_w = 0; streak_l = 0; max_w = 0; max_l = 0
    for p in net_pnls:
        if p > 0: streak_w += 1; streak_l = 0; max_w = max(max_w, streak_w)
        else: streak_l += 1; streak_w = 0; max_l = max(max_l, streak_l)

    total_costs = sum(t['cost'] for t in trades)

    return {
        'total_trades': len(trades),
        'winners': len(wins),
        'losers': len(losses),
        'win_rate': round(len(wins) / len(trades) * 100, 1),
        'avg_winner': round(np.mean(wins), 0) if wins else 0,
        'avg_loser': round(np.mean(losses), 0) if losses else 0,
        'best_trade': round(max(net_pnls), 0),
        'worst_trade': round(min(net_pnls), 0),
        'best_trade_idx': net_pnls.index(max(net_pnls)),
        'worst_trade_idx': net_pnls.index(min(net_pnls)),
        'avg_hold_wins': round(np.mean([t['hold_days'] for t in trades if t['net_pnl'] > 0]), 1) if wins else 0,
        'avg_hold_losses': round(np.mean([t['hold_days'] for t in trades if t['net_pnl'] <= 0]), 1) if losses else 0,
        'profit_factor': round(sum(wins) / abs(sum(losses)), 2) if losses else 99,
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'max_dd_pct': round(max_dd * 100, 1),
        'max_dd_rupees': round(max_dd * capital, 0),
        'max_win_streak': max_w,
        'max_loss_streak': max_l,
        'total_pnl_pts': round(sum(t['pts'] for t in trades), 0),
        'total_pnl_rs': round(sum(net_pnls), 0),
        'total_costs': round(total_costs, 0),
        'final_capital': round(final_cap, 0),
        'total_return': round(total_return, 1),
        'cagr': round(cagr, 1),
        'drawdowns': drawdowns,
    }


def annual_summary(trades, capital=CAPITAL):
    """Compute annual breakdown."""
    by_year = defaultdict(list)
    for t in trades:
        yr = t['exit_date'][:4]
        by_year[yr].append(t)

    summaries = []
    for yr in sorted(by_year.keys()):
        tt = by_year[yr]
        wins = [t for t in tt if t['net_pnl'] > 0]
        pnl = sum(t['net_pnl'] for t in tt)
        cap_start = tt[0]['capital'] - tt[0]['net_pnl']
        cap_end = tt[-1]['capital']
        ann_ret = (pnl / cap_start * 100) if cap_start > 0 else 0
        summaries.append({
            'year': yr,
            'trades': len(tt),
            'winners': len(wins),
            'losers': len(tt) - len(wins),
            'win_pct': round(len(wins) / len(tt) * 100, 0) if tt else 0,
            'pts': round(sum(t['pts'] for t in tt), 0),
            'pnl_rs': round(pnl, 0),
            'ann_ret': round(ann_ret, 1),
            'capital_eoy': round(cap_end, 0),
        })
    return summaries


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

    df = add_all_indicators(df_raw)
    df['hvol_6'] = historical_volatility(df['close'], period=6)
    df['hvol_100'] = historical_volatility(df['close'], period=100)
    df['date'] = df_raw['date']
    df['india_vix'] = df_raw['india_vix']

    from regime_labeler import RegimeLabeler
    regime_dict = RegimeLabeler().label_full_history(df_raw)
    df['regime'] = df['date'].map(regime_dict).fillna('UNKNOWN')

    data_start = df['date'].min()
    data_end = df['date'].max()
    data_years = (data_end - data_start).days / 365.25
    print(f"  {len(df)} days from {data_start.date()} to {data_end.date()} ({data_years:.1f} years)")

    # Nifty benchmark
    nifty_start = df.iloc[0]['close']
    nifty_end = df.iloc[-1]['close']
    nifty_return = (nifty_end / nifty_start - 1) * 100
    nifty_cagr = ((nifty_end / nifty_start) ** (1/data_years) - 1) * 100

    # Load signal rules
    def load_rules(path):
        with open(path) as f:
            d = json.load(f)
        return d.get('rules', d.get('backtest_rule', d.get('dsl_rule', d)))

    sig_rules = {}
    for sid, fname in [
        ('DRY_20', 'validation_results/kaufman_dry_20_fixed.json'),
        ('DRY_16', 'validation_results/kaufman_dry_16_fixed.json'),
        ('DRY_12', 'validation_results/kaufman_dry_12_fixed.json'),
    ]:
        sig_rules[sid] = load_rules(fname)

    for sid in ['GUJRAL_DRY_7', 'GUJRAL_DRY_8', 'GUJRAL_DRY_9', 'GRIMES_DRY_3_2', 'KAUFMAN_DRY_8']:
        path = f'dsl_results/BEST/{sid}.json'
        if os.path.exists(path):
            sig_rules[sid] = load_rules(path)

    # Build all configs
    configs = {}
    for sid in ['DRY_20', 'DRY_16', 'DRY_12', 'GUJRAL_DRY_7', 'GUJRAL_DRY_8', 'GUJRAL_DRY_9', 'GRIMES_DRY_3_2']:
        if sid in sig_rules:
            r = sig_rules[sid]
            configs[sid] = ('single', r, r.get('stop_loss_pct', 0.02), r.get('take_profit_pct', 0), r.get('hold_days', 0))

    configs['SCORING'] = ('scoring', sig_rules)
    configs['CONTROL'] = configs.get('DRY_20', configs.get('DRY_20'))
    if 'KAUFMAN_DRY_8' in sig_rules:
        configs['KAU16+KAU8'] = ('and', sig_rules.get('DRY_16', {}), sig_rules['KAUFMAN_DRY_8'])
    if 'GRIMES_DRY_3_2' in sig_rules:
        configs['COMBINATION'] = ('seq5', sig_rules.get('GRIMES_DRY_3_2', {}), sig_rules.get('DRY_12', {}))

    os.makedirs('trade_analysis', exist_ok=True)

    # Run all signals
    all_stats = {}
    all_trades = {}

    for name, config in configs.items():
        if name == 'CONTROL' and name != 'DRY_20':
            continue  # skip duplicate
        print(f"\n  Running {name}...", end=' ', flush=True)

        if config[0] == 'single':
            _, rules, stop, tp, hold = config
            trades = backtest_with_compounding(df, rules, stop_pct=stop, tp_pct=tp, hold_max=hold)
        elif config[0] == 'scoring':
            trades = backtest_scoring_compound(df, config[1])
        elif config[0] == 'seq5':
            trades = backtest_seq5_compound(df, config[1], config[2])
        elif config[0] == 'and':
            trades = backtest_and_compound(df, config[1], config[2])
        else:
            trades = []

        stats = compute_stats(trades, data_years)
        ann = annual_summary(trades) if trades else []
        all_stats[name] = stats
        all_trades[name] = trades

        print(f"Trades={stats['total_trades']} WR={stats.get('win_rate',0):.0f}% "
              f"CAGR={stats.get('cagr',0):.1f}% MaxDD={stats.get('max_dd_pct',0):.1f}% "
              f"Final=₹{stats.get('final_capital', CAPITAL):,.0f}")

        # Save trade log CSV
        sig_dir = os.path.join('trade_analysis', name)
        os.makedirs(sig_dir, exist_ok=True)
        if trades:
            with open(os.path.join(sig_dir, 'trade_log.csv'), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                writer.writeheader()
                writer.writerows(trades)

    # ================================================================
    # MASTER COMPARISON TABLE
    # ================================================================
    print(f"\n{'='*130}")
    print("MASTER COMPARISON TABLE")
    print(f"{'='*130}")
    print(f"{'Signal':16s} {'Trades':>6s} {'Win%':>6s} {'CAGR':>7s} {'MaxDD':>6s} {'Sharpe':>7s} {'PF':>6s} {'Final ₹':>14s} {'vs Nifty':>10s}")
    print("-" * 130)
    for name in configs:
        if name == 'CONTROL' and name != 'DRY_20': continue
        s = all_stats[name]
        alpha = s.get('cagr', 0) - nifty_cagr
        print(f"{name:16s} {s['total_trades']:6d} {s.get('win_rate',0):5.0f}% {s.get('cagr',0):6.1f}% "
              f"{s.get('max_dd_pct',0):5.1f}% {s.get('sharpe',0):7.2f} {s.get('profit_factor',0):6.2f} "
              f"₹{s.get('final_capital', CAPITAL):>12,.0f} {alpha:+9.1f}%")
    print(f"{'NIFTY B&H':16s} {'—':>6s} {'—':>6s} {nifty_cagr:6.1f}% {'—':>6s} {'—':>7s} {'—':>6s} "
          f"₹{CAPITAL * (1 + nifty_return/100):>12,.0f} {'baseline':>10s}")

    # ================================================================
    # YEAR-BY-YEAR WINNER
    # ================================================================
    print(f"\n{'='*130}")
    print("YEAR-BY-YEAR: BEST SIGNAL PER YEAR")
    print(f"{'='*130}")
    years = sorted(set(t['exit_date'][:4] for trades in all_trades.values() for t in trades))
    print(f"{'Year':>6s} {'Best Signal':>16s} {'Return':>8s} {'2nd Best':>16s} {'Return':>8s}")
    print("-" * 60)
    for yr in years:
        yr_returns = {}
        for name, trades in all_trades.items():
            if name == 'CONTROL' and name != 'DRY_20': continue
            tt = [t for t in trades if t['exit_date'][:4] == yr]
            if len(tt) >= 3:
                pnl = sum(t['net_pnl'] for t in tt)
                cap = tt[0]['capital'] - tt[0]['net_pnl']
                yr_returns[name] = round(pnl / cap * 100, 1) if cap > 0 else 0
        if yr_returns:
            ranked = sorted(yr_returns.items(), key=lambda x: -x[1])
            best = ranked[0]
            second = ranked[1] if len(ranked) > 1 else ('—', 0)
            print(f"{yr:>6s} {best[0]:>16s} {best[1]:+7.1f}% {second[0]:>16s} {second[1]:+7.1f}%")

    # ================================================================
    # TOP/BOTTOM 5 TRADES ACROSS ALL SIGNALS
    # ================================================================
    all_flat = []
    for name, trades in all_trades.items():
        if name == 'CONTROL' and name != 'DRY_20': continue
        for t in trades:
            all_flat.append({**t, 'signal': name})

    if all_flat:
        by_pnl = sorted(all_flat, key=lambda x: -x['net_pnl'])
        print(f"\n{'='*130}")
        print("TOP 5 MOST PROFITABLE TRADES (across all signals)")
        print(f"{'='*130}")
        for t in by_pnl[:5]:
            print(f"  {t['signal']:16s} {t['entry_date']} → {t['exit_date']} {t['direction']:5s} "
                  f"Entry={t['entry_price']:.0f} Exit={t['exit_price']:.0f} "
                  f"P&L=₹{t['net_pnl']:+,.0f} ({t['pts']:+.0f}pts) [{t['reason']}]")

        print(f"\n{'='*130}")
        print("TOP 5 WORST TRADES (across all signals)")
        print(f"{'='*130}")
        for t in by_pnl[-5:]:
            print(f"  {t['signal']:16s} {t['entry_date']} → {t['exit_date']} {t['direction']:5s} "
                  f"Entry={t['entry_price']:.0f} Exit={t['exit_price']:.0f} "
                  f"P&L=₹{t['net_pnl']:+,.0f} ({t['pts']:+.0f}pts) [{t['reason']}]")

    # ================================================================
    # TOTAL COSTS
    # ================================================================
    print(f"\n{'='*130}")
    print("TOTAL TRANSACTION COSTS PAID (what the broker earned)")
    print(f"{'='*130}")
    total_all_costs = 0
    for name, s in all_stats.items():
        if name == 'CONTROL' and name != 'DRY_20': continue
        costs = s.get('total_costs', 0)
        total_all_costs += costs
        print(f"  {name:16s} ₹{costs:>10,.0f} ({s['total_trades']} trades)")
    print(f"  {'TOTAL':16s} ₹{total_all_costs:>10,.0f}")

    # ================================================================
    # CAPITAL GROWTH
    # ================================================================
    print(f"\n{'='*130}")
    print("CAPITAL GROWTH (₹10L → ?)")
    print(f"{'='*130}")
    ranked_cap = sorted(all_stats.items(), key=lambda x: -x[1].get('final_capital', 0))
    for name, s in ranked_cap:
        if name == 'CONTROL' and name != 'DRY_20': continue
        fc = s.get('final_capital', CAPITAL)
        ret = s.get('total_return', 0)
        cagr = s.get('cagr', 0)
        print(f"  {name:16s} ₹{fc:>14,.0f}  ({ret:+.0f}%, CAGR {cagr:.1f}%)")
    nifty_final = CAPITAL * (1 + nifty_return/100)
    print(f"  {'NIFTY B&H':16s} ₹{nifty_final:>14,.0f}  ({nifty_return:+.0f}%, CAGR {nifty_cagr:.1f}%)")

    # Save all results
    save_data = {}
    for name, s in all_stats.items():
        save_data[name] = {k: v for k, v in s.items() if k != 'drawdowns'}
    with open('trade_analysis/all_stats.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nAll results saved to trade_analysis/")


if __name__ == '__main__':
    main()

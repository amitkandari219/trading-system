"""
Multi-period performance analysis across 9 time horizons.

Tests all confirmed signals and combinations with realistic
transaction costs, slippage, and capital assumptions.
"""

import json
import os
from collections import defaultdict
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import psycopg2

from backtest.indicators import add_all_indicators, historical_volatility
from config.settings import DATABASE_DSN

# Constants
CAPITAL = 1_000_000       # ₹10 lakh
LOT_SIZE = 75             # Nifty lot size (units)
LOT_MARGIN = 100_000      # approx margin per lot
LOTS = min(CAPITAL // LOT_MARGIN, 10)  # 10 lots max
SLIPPAGE_PTS = 2          # per trade (entry+exit combined)
COST_PER_TRADE = 250      # ₹ per round trip (brokerage + STT + exchange)

PERIODS = {
    '6mo': 6, '1yr': 12, '2yr': 24, '3yr': 36,
    '4yr': 48, '5yr': 60, '6yr': 72, '10yr': 120, 'max': None,
}


# ================================================================
# SIGNAL CONDITION EVALUATORS
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


def backtest_signal(rules, df, stop_pct=0.02, tp_pct=0.0, hold_max=0):
    """Run a single signal backtest. Returns trades list."""
    n = len(df)
    if n < 20: return []
    entry_long = rules.get('entry_long', [])
    entry_short = rules.get('entry_short', [])
    exit_long = rules.get('exit_long', [])
    exit_short = rules.get('exit_short', [])
    closes = df['close'].values
    dates = df['date'].values
    trades = []
    pos = None
    ep = 0; ei = 0

    for i in range(1, n):
        row = df.iloc[i]; prev = df.iloc[i-1]
        if pos is None:
            if _eval_conds(row, prev, entry_long):
                pos = 'LONG'; ep = closes[i]; ei = i
            elif _eval_conds(row, prev, entry_short):
                pos = 'SHORT'; ep = closes[i]; ei = i
        else:
            p = closes[i]; dh = i - ei
            if pos == 'LONG': lp = (ep - p) / ep
            else: lp = (p - ep) / ep
            if lp >= stop_pct:
                pnl = (p - ep) if pos == 'LONG' else (ep - p)
                trades.append({'pnl': pnl, 'dir': pos, 'entry_date': dates[ei], 'exit_date': dates[i], 'days': dh, 'reason': 'stop'})
                pos = None; continue
            if tp_pct > 0:
                gp = (p - ep) / ep if pos == 'LONG' else (ep - p) / ep
                if gp >= tp_pct:
                    trades.append({'pnl': gp * ep, 'dir': pos, 'entry_date': dates[ei], 'exit_date': dates[i], 'days': dh, 'reason': 'tp'})
                    pos = None; continue
            if hold_max > 0 and dh >= hold_max:
                pnl = (p - ep) if pos == 'LONG' else (ep - p)
                trades.append({'pnl': pnl, 'dir': pos, 'entry_date': dates[ei], 'exit_date': dates[i], 'days': dh, 'reason': 'hold'})
                pos = None; continue
            ex = exit_long if pos == 'LONG' else exit_short
            if _eval_conds(row, prev, ex):
                pnl = (p - ep) if pos == 'LONG' else (ep - p)
                trades.append({'pnl': pnl, 'dir': pos, 'entry_date': dates[ei], 'exit_date': dates[i], 'days': dh, 'reason': 'signal'})
                pos = None
    if pos:
        p = closes[-1]
        pnl = (p - ep) if pos == 'LONG' else (ep - p)
        trades.append({'pnl': pnl, 'dir': pos, 'entry_date': dates[ei], 'exit_date': dates[-1], 'days': len(df)-1-ei, 'reason': 'eod'})
    return trades


def backtest_scoring(df, sig_rules):
    """Scoring system backtest."""
    n = len(df); closes = df['close'].values; dates = df['date'].values
    trades = []; pos = None; ep = 0; ei = 0; entry_thresh = 0; size = 1.0
    for i in range(1, n):
        row = df.iloc[i]; prev = df.iloc[i-1]
        score = 0
        dry20_long = _eval_conds(row, prev, sig_rules['DRY_20']['entry_long'])
        dry20_exit = _eval_conds(row, prev, sig_rules['DRY_20']['exit_long'])
        dry12_long = _eval_conds(row, prev, sig_rules['DRY_12']['entry_long'])
        dry12_short = _eval_conds(row, prev, sig_rules['DRY_12']['entry_short'])
        dry16_long = _eval_conds(row, prev, sig_rules['DRY_16']['entry_long'])
        dry16_short = _eval_conds(row, prev, sig_rules['DRY_16']['entry_short'])
        if dry20_exit: score = 0
        else:
            if dry20_long: score += 2
            if dry12_long: score += 1
            if dry12_short: score -= 1
            if dry16_long: score += 1
            if dry16_short: score -= 1
        if pos is None:
            if score >= 3: pos='LONG'; ep=closes[i]; ei=i; entry_thresh=3; size=1.0
            elif score >= 2: pos='LONG'; ep=closes[i]; ei=i; entry_thresh=2; size=0.5
            elif score <= -2: pos='SHORT'; ep=closes[i]; ei=i; entry_thresh=-2; size=0.5
        else:
            p = closes[i]
            if pos=='LONG': lp=(ep-p)/ep
            else: lp=(p-ep)/ep
            if lp >= 0.02:
                pnl = ((p-ep) if pos=='LONG' else (ep-p)) * size
                trades.append({'pnl':pnl,'dir':pos,'entry_date':dates[ei],'exit_date':dates[i],'days':i-ei,'reason':'stop'})
                pos=None; continue
            if pos=='LONG' and score < entry_thresh:
                pnl = (p-ep)*size
                trades.append({'pnl':pnl,'dir':pos,'entry_date':dates[ei],'exit_date':dates[i],'days':i-ei,'reason':'score'})
                pos=None
            elif pos=='SHORT' and score > entry_thresh:
                pnl = (ep-p)*size
                trades.append({'pnl':pnl,'dir':pos,'entry_date':dates[ei],'exit_date':dates[i],'days':i-ei,'reason':'score'})
                pos=None
    return trades


def backtest_combination_seq5(df, grimes_rules, kaufman_rules):
    """GRIMES → KAUFMAN SEQ_5 backtest."""
    n = len(df); closes = df['close'].values; dates = df['date'].values
    trades = []; pos = None; ep = 0; ei = 0; pending_long = -999; pending_short = -999
    for i in range(1, n):
        row = df.iloc[i]; prev = df.iloc[i-1]
        regime = str(row.get('regime', ''))
        trending = regime in ('TRENDING_UP', 'TRENDING_DOWN', 'TRENDING')
        if pos:
            pos['days'] += 1
            p = closes[i]
            if pos['dir']=='LONG': lp=(ep-p)/ep
            else: lp=(p-ep)/ep
            if lp >= 0.02:
                pnl=(p-ep) if pos['dir']=='LONG' else (ep-p)
                trades.append({'pnl':pnl,'dir':pos['dir'],'entry_date':dates[ei],'exit_date':dates[i],'days':pos['days'],'reason':'stop'})
                pos=None; continue
            if pos['days'] >= 10:
                pnl=(p-ep) if pos['dir']=='LONG' else (ep-p)
                trades.append({'pnl':pnl,'dir':pos['dir'],'entry_date':dates[ei],'exit_date':dates[i],'days':pos['days'],'reason':'hold'})
                pos=None; continue
            if pos['dir']=='LONG' and row['low'] < prev['low']:
                pnl=p-ep
                trades.append({'pnl':pnl,'dir':'LONG','entry_date':dates[ei],'exit_date':dates[i],'days':pos['days'],'reason':'struct'})
                pos=None; continue
            if pos['dir']=='SHORT' and row['high'] > prev['high']:
                pnl=ep-p
                trades.append({'pnl':pnl,'dir':'SHORT','entry_date':dates[ei],'exit_date':dates[i],'days':pos['days'],'reason':'struct'})
                pos=None; continue
            continue
        adx_ok = pd.notna(row.get('adx_14')) and float(row['adx_14']) > 25
        gl = row['high']>prev['high'] and row['low']>prev['low'] and adx_ok and trending
        gs = row['low']<prev['low'] and row['high']<prev['high'] and adx_ok and trending
        if gl: pending_long=i; pending_short=-999
        if gs: pending_short=i; pending_long=-999
        if pending_long>=0 and (i-pending_long)>5: pending_long=-999
        if pending_short>=0 and (i-pending_short)>5: pending_short=-999
        kl = row['close']>prev['close'] and row['volume']<prev['volume']
        ks = row['close']<prev['close'] and row['volume']>prev['volume']
        if pending_long>=0 and pending_long!=i and kl:
            pos={'dir':'LONG','days':0}; ep=closes[i]; ei=i; pending_long=-999
        elif pending_short>=0 and pending_short!=i and ks:
            pos={'dir':'SHORT','days':0}; ep=closes[i]; ei=i; pending_short=-999
    return trades


def backtest_and_combo(df, rules_a, rules_b):
    """AND combination backtest."""
    n = len(df); closes = df['close'].values; dates = df['date'].values
    trades = []; pos = None; ep = 0; ei = 0
    for i in range(1, n):
        row = df.iloc[i]; prev = df.iloc[i-1]
        if pos is None:
            al = _eval_conds(row, prev, rules_a.get('entry_long', []))
            bl = _eval_conds(row, prev, rules_b.get('entry_long', []))
            if al and bl:
                pos = 'LONG'; ep = closes[i]; ei = i; continue
            ash = _eval_conds(row, prev, rules_a.get('entry_short', []))
            bsh = _eval_conds(row, prev, rules_b.get('entry_short', []))
            if ash and bsh:
                pos = 'SHORT'; ep = closes[i]; ei = i
        else:
            p = closes[i]
            if pos=='LONG': lp=(ep-p)/ep
            else: lp=(p-ep)/ep
            if lp >= 0.02:
                pnl=(p-ep) if pos=='LONG' else (ep-p)
                trades.append({'pnl':pnl,'dir':pos,'entry_date':dates[ei],'exit_date':dates[i],'days':i-ei,'reason':'stop'})
                pos=None; continue
            exl = _eval_conds(row, prev, rules_a.get('exit_long',[])) or _eval_conds(row, prev, rules_b.get('exit_long',[]))
            exs = _eval_conds(row, prev, rules_a.get('exit_short',[])) or _eval_conds(row, prev, rules_b.get('exit_short',[]))
            if (pos=='LONG' and exl) or (pos=='SHORT' and exs):
                pnl=(p-ep) if pos=='LONG' else (ep-p)
                trades.append({'pnl':pnl,'dir':pos,'entry_date':dates[ei],'exit_date':dates[i],'days':i-ei,'reason':'signal'})
                pos=None
    return trades


def compute_metrics(trades, df, capital=CAPITAL, lots=LOTS):
    """Compute all metrics from trades list."""
    date_range = (pd.Timestamp(df['date'].max()) - pd.Timestamp(df['date'].min())).days
    years = max(0.25, date_range / 365.25)

    if len(trades) < 2:
        return {'trades': len(trades), 'trades_per_year': len(trades)/years,
                'win_rate': 0, 'sharpe': 0, 'profit_factor': 0,
                'max_drawdown': 0, 'nifty_corr': 0,
                'pnl_points': 0, 'pnl_rupees': 0, 'pnl_pct': 0,
                'ann_return_pct': 0, 'sortino': 0,
                'best_trade_pts': 0, 'worst_trade_pts': 0,
                'avg_hold_days': 0, 'long_trades': 0, 'long_wr': 0,
                'short_trades': 0, 'short_wr': 0,
                'pnl_per_trade': 0, 'max_dd_rupees': 0,
                'insufficient': True}

    # Apply slippage and costs
    net_pnls = []
    for t in trades:
        raw_pnl = t['pnl']
        net_pnl = raw_pnl - SLIPPAGE_PTS  # slippage in points
        cost_pts = COST_PER_TRADE / (LOT_SIZE * lots)  # convert ₹ cost to points
        net_pnl -= cost_pts
        net_pnls.append(net_pnl)

    total_pnl = sum(net_pnls)
    pnl_rupees = total_pnl * LOT_SIZE * lots

    notional = abs(trades[0].get('pnl', 1)) + 1e-9
    # Use first close as notional for pct calculations
    first_close = 10000  # fallback
    pct_returns = [p / first_close for p in net_pnls]

    wins = [r for r in pct_returns if r > 0]
    losses = [r for r in pct_returns if r < 0]
    win_rate = len(wins) / len(pct_returns) if pct_returns else 0

    ret = pd.Series(pct_returns)
    std = ret.std()
    sharpe = (ret.mean() / std * np.sqrt(252)) if std > 0 else 0

    down_std = ret[ret < 0].std()
    sortino = (ret.mean() / down_std * np.sqrt(252)) if down_std and down_std > 0 else 0

    eq = [1.0]
    for r in pct_returns:
        eq.append(eq[-1] * (1 + r))
    eqs = pd.Series(eq)
    dd = (eqs - eqs.cummax()) / eqs.cummax()
    max_dd = abs(dd.min())

    gw = sum(p for p in net_pnls if p > 0)
    gl = abs(sum(p for p in net_pnls if p < 0))
    pf = gw / gl if gl > 0 else 99

    # Nifty correlation
    trade_ret = {}
    for t, pnl in zip(trades, net_pnls):
        d = t['exit_date']
        trade_ret[d] = trade_ret.get(d, 0) + pnl / first_close
    nifty_ret = df.set_index('date')['close'].pct_change()
    strat_ret = pd.Series(trade_ret)
    common = strat_ret.index.intersection(nifty_ret.index)
    corr = float(strat_ret.reindex(common).corr(nifty_ret.reindex(common))) if len(common) > 10 else 0
    if np.isnan(corr): corr = 0

    ann_return = (eq[-1] ** (1/years) - 1) * 100 if eq[-1] > 0 else -100

    long_t = [i for i, t in enumerate(trades) if t['dir'] == 'LONG']
    short_t = [i for i, t in enumerate(trades) if t['dir'] == 'SHORT']
    long_wr = sum(1 for i in long_t if net_pnls[i] > 0) / len(long_t) * 100 if long_t else 0
    short_wr = sum(1 for i in short_t if net_pnls[i] > 0) / len(short_t) * 100 if short_t else 0

    return {
        'trades': len(trades),
        'trades_per_year': round(len(trades) / years, 1),
        'win_rate': round(win_rate * 100, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'profit_factor': round(min(pf, 99), 2),
        'max_drawdown': round(max_dd * 100, 1),
        'nifty_corr': round(corr, 3),
        'pnl_points': round(total_pnl, 0),
        'pnl_rupees': round(pnl_rupees, 0),
        'pnl_pct': round(pnl_rupees / capital * 100, 1),
        'ann_return_pct': round(ann_return, 1),
        'pnl_per_trade': round(total_pnl / len(trades), 1) if trades else 0,
        'best_trade_pts': round(max(net_pnls), 1),
        'worst_trade_pts': round(min(net_pnls), 1),
        'avg_hold_days': round(np.mean([t['days'] for t in trades]), 1),
        'long_trades': len(long_t),
        'long_wr': round(long_wr, 1),
        'short_trades': len(short_t),
        'short_wr': round(short_wr, 1),
        'max_dd_rupees': round(max_dd * capital, 0),
        'insufficient': len(trades) < 10,
    }


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

    end_date = df['date'].max()
    data_start = df['date'].min()
    print(f"  {len(df)} days from {data_start.date()} to {end_date.date()}", flush=True)

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

    for sid in ['GUJRAL_DRY_7', 'GUJRAL_DRY_8', 'GUJRAL_DRY_9']:
        path = f'dsl_results/BEST/{sid}.json'
        if os.path.exists(path):
            sig_rules[sid] = load_rules(path)

    grimes_path = 'dsl_results/BEST/GRIMES_DRY_3_2.json'
    grimes_rules = load_rules(grimes_path) if os.path.exists(grimes_path) else {}

    kau8_path = 'dsl_results/BEST/KAUFMAN_DRY_8.json'
    kau8_rules = load_rules(kau8_path) if os.path.exists(kau8_path) else {}

    # Define all test configs
    configs = {}
    for sid in ['DRY_20', 'DRY_16', 'DRY_12', 'GUJRAL_DRY_7', 'GUJRAL_DRY_8', 'GUJRAL_DRY_9']:
        if sid in sig_rules:
            configs[sid] = ('single', sig_rules[sid])

    configs['CONTROL'] = ('single', sig_rules['DRY_20'])
    configs['SCORING'] = ('scoring', sig_rules)
    configs['COMBINATION'] = ('seq5', grimes_rules, sig_rules.get('DRY_12', {}))
    if kau8_rules:
        configs['KAU16+KAU8'] = ('and', sig_rules.get('DRY_16', {}), kau8_rules)

    # Nifty benchmark
    configs['NIFTY_BH'] = ('benchmark', None)

    # Run all periods
    os.makedirs('multi_period_results', exist_ok=True)
    all_results = {}

    period_names = list(PERIODS.keys())
    signal_names = list(configs.keys())

    print(f"\nRunning {len(signal_names)} signals × {len(period_names)} periods = {len(signal_names)*len(period_names)} backtests...\n", flush=True)

    for sig_name in signal_names:
        all_results[sig_name] = {}
        for period_name, months in PERIODS.items():
            if months:
                start = end_date - relativedelta(months=months)
            else:
                start = data_start

            df_slice = df[(df['date'] >= start) & (df['date'] <= end_date)].reset_index(drop=True)
            if len(df_slice) < 20:
                all_results[sig_name][period_name] = {'trades': 0, 'insufficient': True}
                continue

            config = configs[sig_name]

            if config[0] == 'benchmark':
                first_close = df_slice.iloc[0]['close']
                last_close = df_slice.iloc[-1]['close']
                nifty_ret = (last_close / first_close - 1) * 100
                years = max(0.25, (df_slice['date'].max() - df_slice['date'].min()).days / 365.25)
                ann_ret = ((last_close / first_close) ** (1/years) - 1) * 100
                # Max drawdown
                eq = df_slice['close'] / df_slice['close'].iloc[0]
                dd = (eq - eq.cummax()) / eq.cummax()
                max_dd = abs(dd.min()) * 100
                all_results[sig_name][period_name] = {
                    'trades': 1, 'pnl_points': round(last_close - first_close, 0),
                    'pnl_pct': round(nifty_ret, 1), 'ann_return_pct': round(ann_ret, 1),
                    'max_drawdown': round(max_dd, 1), 'sharpe': 0, 'win_rate': 0,
                    'pnl_rupees': round((last_close - first_close) * LOT_SIZE * LOTS, 0),
                }
                continue

            if config[0] == 'single':
                rules = config[1]
                trades = backtest_signal(rules, df_slice,
                                          rules.get('stop_loss_pct', 0.02),
                                          rules.get('take_profit_pct', 0),
                                          rules.get('hold_days', 0))
            elif config[0] == 'scoring':
                trades = backtest_scoring(df_slice, config[1])
            elif config[0] == 'seq5':
                trades = backtest_combination_seq5(df_slice, config[1], config[2])
            elif config[0] == 'and':
                trades = backtest_and_combo(df_slice, config[1], config[2])
            else:
                trades = []

            metrics = compute_metrics(trades, df_slice)

            # Add Nifty return for comparison
            first_close = df_slice.iloc[0]['close']
            last_close = df_slice.iloc[-1]['close']
            nifty_ret = (last_close / first_close - 1) * 100
            metrics['nifty_return'] = round(nifty_ret, 1)
            metrics['alpha'] = round(metrics['ann_return_pct'] - metrics.get('nifty_return', 0) / max(0.25, (df_slice['date'].max() - df_slice['date'].min()).days / 365.25), 1)

            all_results[sig_name][period_name] = metrics

    # ================================================================
    # PRINT TABLES
    # ================================================================

    def print_table(title, metric, fmt='{:>8.0f}', suffix=''):
        print(f"\n{'='*120}")
        print(f"  {title}")
        print(f"{'='*120}")
        header = f"{'Signal':16s}"
        for p in period_names:
            header += f" {p:>10s}"
        print(header)
        print("-" * 120)
        for sig in signal_names:
            row = f"{sig:16s}"
            for p in period_names:
                m = all_results.get(sig, {}).get(p, {})
                val = m.get(metric, 0)
                insuf = m.get('insufficient', False)
                if insuf and metric not in ('pnl_points', 'pnl_pct', 'ann_return_pct'):
                    row += f" {'LOW':>10s}"
                else:
                    try:
                        row += f" {fmt.format(val)}"
                    except:
                        row += f" {str(val):>10s}"
            print(row)

    print_table("TABLE 1: P&L IN NIFTY POINTS (after costs & slippage)", 'pnl_points', '{:>10.0f}')
    print_table("TABLE 2: P&L IN RUPEES (₹10L capital, 10 lots)", 'pnl_rupees', '{:>10,.0f}')
    print_table("TABLE 3: SHARPE RATIO", 'sharpe', '{:>10.2f}')
    print_table("TABLE 4: WIN RATE %", 'win_rate', '{:>10.1f}')
    print_table("TABLE 5: ANNUAL RETURN %", 'ann_return_pct', '{:>10.1f}')
    print_table("TABLE 6: TRADES (trades/year)", 'trades_per_year', '{:>10.1f}')
    print_table("TABLE 7: MAX DRAWDOWN %", 'max_drawdown', '{:>10.1f}')

    # Consistency scores
    print(f"\n{'='*120}")
    print("  CONSISTENCY SCORE (profitable periods / 9)")
    print(f"{'='*120}")
    for sig in signal_names:
        if sig == 'NIFTY_BH': continue
        profitable = sum(1 for p in period_names
                        if all_results.get(sig, {}).get(p, {}).get('pnl_points', 0) > 0)
        status = "✓ CONSISTENT" if profitable >= 7 else "○ MIXED" if profitable >= 5 else "✗ WEAK"
        print(f"  {sig:16s} {profitable}/9 {status}")

    # Current state
    print(f"\n{'='*120}")
    print("  CURRENT STATE (6mo and 1yr)")
    print(f"{'='*120}")
    for sig in signal_names:
        if sig == 'NIFTY_BH': continue
        m6 = all_results.get(sig, {}).get('6mo', {})
        m1 = all_results.get(sig, {}).get('1yr', {})
        p6 = m6.get('pnl_points', 0)
        p1 = m1.get('pnl_points', 0)
        state = "ACTIVE ✓" if p6 > 0 and p1 > 0 else "RECOVERING" if p1 > 0 else "IN DRAWDOWN ✗"
        print(f"  {sig:16s} 6mo={p6:+8.0f}pts  1yr={p1:+8.0f}pts  → {state}")

    # Capital growth
    print(f"\n{'='*120}")
    print("  CAPITAL GROWTH: ₹10L invested at data start")
    print(f"{'='*120}")
    for sig in signal_names:
        m = all_results.get(sig, {}).get('max', {})
        pnl_pct = m.get('pnl_pct', 0)
        final = CAPITAL * (1 + pnl_pct / 100)
        print(f"  {sig:16s} ₹{final:>12,.0f}  ({pnl_pct:+.1f}%)")

    # Save
    with open('multi_period_results/results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to multi_period_results/results.json")


if __name__ == '__main__':
    main()

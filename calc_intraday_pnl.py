#!/usr/bin/env python3
"""
Calculate 10-year P&L, CAGR, and detailed metrics for the 2 passed
intraday signals: ID_GUJRAL_RANGE and ID_KAUFMAN_BB_MR.

Runs full backtest from 2016 to 2026 (10 years).
"""

import os, sys, json
import numpy as np
import pandas as pd
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest.indicators import sma, ema, rsi, atr, adx, bollinger_bands, stochastic

SLIPPAGE_PCT = 0.0005
NIFTY_LOT_SIZE = 25
CAPITAL = 1_000_000  # ₹10L

VOLUME_PROFILE_75 = np.array([
    3.5, 3.0, 2.5, 2.2, 2.0, 1.8, 1.7, 1.6, 1.5,
    1.4, 1.3, 1.3, 1.2, 1.2, 1.1, 1.1, 1.0, 1.0, 1.0, 0.9, 0.9,
    0.9, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6,
    0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1,
    1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3,
    2.5, 2.7, 3.0, 3.2, 3.5, 4.0,
])
VOLUME_PROFILE_75 = VOLUME_PROFILE_75 / VOLUME_PROFILE_75.sum()


def load_daily_from_trade_logs():
    dfs = []
    trade_dir = os.path.join(os.path.dirname(__file__), 'trade_analysis')
    for d in os.listdir(trade_dir):
        p = os.path.join(trade_dir, d, 'trade_log.csv')
        if not os.path.exists(p): continue
        df = pd.read_csv(p)
        if 'entry_price' not in df.columns: continue
        for _, row in df.iterrows():
            for dc, pc in [('entry_date','entry_price'), ('exit_date','exit_price')]:
                if pd.notna(row.get(dc)) and pd.notna(row.get(pc)):
                    dfs.append({'date': row[dc], 'price': float(row[pc]),
                                'vix': float(row.get('vix', 15)) if pd.notna(row.get('vix')) else 15})
    prices = pd.DataFrame(dfs)
    prices['date'] = pd.to_datetime(prices['date'])
    daily = prices.groupby('date').agg(close=('price','mean'), india_vix=('vix','mean')).reset_index().sort_values('date').reset_index(drop=True)
    rng = np.random.default_rng(42)
    c = daily['close'].values
    daily['open'] = np.roll(c, 1); daily.loc[0,'open'] = c[0]
    dr = c * 0.012
    daily['high'] = np.maximum(daily['close'], daily['open']) + dr * rng.uniform(0.1, 0.5, len(daily))
    daily['low'] = np.minimum(daily['close'], daily['open']) - dr * rng.uniform(0.1, 0.5, len(daily))
    daily['volume'] = (rng.uniform(0.8, 1.2, len(daily)) * 200_000_000).astype(int)
    return daily


def gen_5min(daily_row, rng):
    do, dh, dl, dc = float(daily_row['open']), float(daily_row['high']), float(daily_row['low']), float(daily_row['close'])
    dv = int(daily_row['volume'])
    dd = pd.Timestamp(daily_row['date'])
    n = 75; dr = max(dh - dl, do * 0.005)
    steps = rng.normal(0, 1, n); cum = np.cumsum(steps); cum = cum - cum[-1]
    path = do + cum * (dr / (4*np.std(cum)+1e-9))
    path = path - (path[-1] - dc) * np.linspace(0, 1, n)
    hb, lb = rng.integers(0, n), rng.integers(0, n)
    while lb == hb: lb = rng.integers(0, n)
    path[hb] = dh - dr*0.01; path[lb] = dl + dr*0.01
    bars = []
    for i in range(n):
        bt = dd + timedelta(hours=9, minutes=15+i*5)
        bc = float(np.clip(path[i], dl, dh))
        bo = float(np.clip(path[i-1] if i>0 else do, dl, dh))
        noise = dr * 0.003 * rng.uniform(0.5, 1.5)
        bh = min(max(bo, bc)+noise, dh); bl = max(min(bo, bc)-noise, dl)
        if i == 0: bo = do
        if i == n-1: bc = dc
        if i == hb: bh = dh
        if i == lb: bl = dl
        bars.append({'datetime': bt, 'open': round(bo,2), 'high': round(bh,2),
                     'low': round(bl,2), 'close': round(bc,2), 'volume': int(dv*VOLUME_PROFILE_75[i])})
    return bars


def add_indicators(df):
    df = df.copy().sort_values('datetime').reset_index(drop=True)
    c = df['close']
    df['_date'] = df['datetime'].dt.date
    # VWAP
    df['_typ'] = (df['high']+df['low']+c)/3
    df['_tpv'] = df['_typ']*df['volume']
    df['_ctpv'] = df.groupby('_date')['_tpv'].cumsum()
    df['_cvol'] = df.groupby('_date')['volume'].cumsum()
    df['vwap'] = df['_ctpv'] / df['_cvol'].replace(0, np.nan)
    # OR
    df['_bn'] = df.groupby('_date').cumcount()
    f3 = df[df['_bn']<3].groupby('_date').agg(opening_range_high=('high','max'), opening_range_low=('low','min'))
    df = df.merge(f3, left_on='_date', right_index=True, how='left')
    # Session
    df['session_bar'] = df.groupby('_date').cumcount()
    # Prev day
    da = df.groupby('_date').agg(_dh=('high','max'), _dl=('low','min'), _dc=('close','last'))
    da['prev_day_high'] = da['_dh'].shift(1); da['prev_day_low'] = da['_dl'].shift(1); da['prev_day_close'] = da['_dc'].shift(1)
    df = df.merge(da[['prev_day_high','prev_day_low','prev_day_close']], left_on='_date', right_index=True, how='left')
    # Technicals
    df['sma_20'] = sma(c,20); df['sma_50'] = sma(c,50)
    df['ema_9'] = ema(c,9); df['ema_20'] = ema(c,20)
    df['rsi_14'] = rsi(c,14); df['atr_14'] = atr(df,14); df['adx_14'] = adx(df)
    bb = bollinger_bands(c, 20)
    for col in bb.columns: df[col] = bb[col].values
    va = df['volume'].rolling(20, min_periods=5).mean()
    df['vol_ratio_20'] = df['volume'] / va.replace(0, np.nan)
    df['prev_close'] = c.shift(1)
    for col in ['_date','_typ','_tpv','_ctpv','_cvol','_bn']:
        if col in df.columns: df = df.drop(columns=[col])
    return df


def _sc(dt):
    return dt.hour > 15 or (dt.hour == 15 and dt.minute >= 20)

def _tr(dt, s, e):
    h1,m1 = map(int, s.split(':')); h2,m2 = map(int, e.split(':'))
    bm = dt.hour*60+dt.minute
    return (h1*60+m1) <= bm <= (h2*60+m2)


# ── SIGNAL CHECKS ──

def check_bb_mr(bar, prev_bar, session_bars):
    if prev_bar is None: return None
    c = float(bar['close']); a = float(bar['adx_14']) if pd.notna(bar.get('adx_14')) else 30
    if a >= 25: return None
    bu, bl = bar.get('bb_upper'), bar.get('bb_lower')
    if pd.isna(bu) or pd.isna(bl): return None
    bu, bl = float(bu), float(bl); pc = float(prev_bar['close'])
    if c <= bl and pc > bl: return 'LONG'
    if c >= bu and pc < bu: return 'SHORT'
    return None

def check_range(bar, prev_bar, session_bars):
    if len(session_bars) < 20: return None
    c = float(bar['close']); a = float(bar['adx_14']) if pd.notna(bar.get('adx_14')) else 30
    if a >= 25: return None
    sh = float(session_bars['high'].max()); sl = float(session_bars['low'].min())
    sr = sh - sl
    if sr <= 0: return None
    th = sr * 0.20
    if c <= sl+th and c > sl and c > float(bar['open']): return 'LONG'
    if c >= sh-th and c < sh and c < float(bar['open']): return 'SHORT'
    return None


SIGNALS = {
    'ID_KAUFMAN_BB_MR': {'check': check_bb_mr, 'sl': 0.004, 'max_hold': 30,
                          'entry_start': '09:45', 'entry_end': '14:30', 'has_target': True},
    'ID_GUJRAL_RANGE':  {'check': check_range, 'sl': 0.005, 'max_hold': 35,
                          'entry_start': '10:00', 'entry_end': '14:30', 'has_target': False},
}


def run_full_backtest(signal_id, cfg, df_daily):
    """Run full 10-year backtest, return detailed trades + yearly stats."""
    check_fn = cfg['check']
    rng = np.random.default_rng(42)
    trades = []
    position = None
    bars_held = 0
    equity = CAPITAL
    daily_pnl = {}

    for i in range(len(df_daily)):
        row = df_daily.iloc[i]
        td = pd.Timestamp(row['date']).date()
        bars = gen_5min(row, rng)
        df_b = pd.DataFrame(bars)
        df_b = add_indicators(df_b)
        sb = pd.DataFrame()

        for idx in range(len(df_b)):
            bar = df_b.iloc[idx]
            pb = df_b.iloc[idx-1] if idx > 0 else None
            bdt = bar['datetime']; c = float(bar['close'])
            sb = pd.concat([sb, bar.to_frame().T], ignore_index=True)

            if position is not None:
                bars_held += 1
                d = position['dir']; ep = position['ep']
                xr = None
                if _sc(bdt): xr = 'session_close'
                if not xr:
                    loss = (ep-c)/ep if d=='LONG' else (c-ep)/ep
                    if loss >= cfg['sl']: xr = 'stop_loss'
                if not xr and bars_held >= cfg['max_hold']: xr = 'max_hold'
                if not xr and signal_id == 'ID_KAUFMAN_BB_MR':
                    bm = bar.get('bb_middle')
                    if pd.notna(bm):
                        bm = float(bm)
                        if (d=='LONG' and c >= bm) or (d=='SHORT' and c <= bm):
                            xr = 'target'
                if xr:
                    if d=='LONG': xp=c*(1-SLIPPAGE_PCT); pp=xp-ep
                    else: xp=c*(1+SLIPPAGE_PCT); pp=ep-xp
                    pr = pp * NIFTY_LOT_SIZE
                    trades.append({'date': td, 'dir': d, 'entry': ep, 'exit': round(xp,2),
                                   'pts': round(pp,2), 'pnl': round(pr,2), 'reason': xr,
                                   'bars': bars_held})
                    equity += pr
                    daily_pnl.setdefault(td, 0); daily_pnl[td] += pr
                    position = None; bars_held = 0

            if position is None and not _sc(bdt):
                if _tr(bdt, cfg['entry_start'], cfg['entry_end']):
                    sig = check_fn(bar, pb, sb)
                    if sig:
                        adj = c*(1+SLIPPAGE_PCT) if sig=='LONG' else c*(1-SLIPPAGE_PCT)
                        position = {'dir': sig, 'ep': round(adj,2)}
                        bars_held = 0

        if position is not None:
            last = df_b.iloc[-1]; c = float(last['close'])
            d = position['dir']; ep = position['ep']
            if d=='LONG': xp=c*(1-SLIPPAGE_PCT); pp=xp-ep
            else: xp=c*(1+SLIPPAGE_PCT); pp=ep-xp
            pr = pp * NIFTY_LOT_SIZE
            trades.append({'date': td, 'dir': d, 'entry': ep, 'exit': round(xp,2),
                           'pts': round(pp,2), 'pnl': round(pr,2), 'reason': 'forced_close',
                           'bars': bars_held})
            equity += pr
            daily_pnl.setdefault(td, 0); daily_pnl[td] += pr
            position = None

    return trades, daily_pnl, equity


def compute_metrics(trades, daily_pnl, final_equity):
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins)/len(pnls) if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 1
    pf = sum(wins)/abs(sum(losses)) if losses and sum(losses)!=0 else 0
    wl_ratio = avg_win/avg_loss if avg_loss > 0 else 0

    dr = pd.Series(daily_pnl)/CAPITAL
    sharpe = dr.mean()/dr.std()*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0

    cum = np.cumsum(pnls)
    pk = np.maximum.accumulate(cum)
    dd = cum - pk
    max_dd = abs(dd.min())
    max_dd_pct = max_dd/CAPITAL

    # CAGR
    df_dates = pd.Series(list(daily_pnl.keys()))
    years = (df_dates.max() - df_dates.min()).days / 365.25
    cagr = (final_equity/CAPITAL)**(1/years) - 1 if years > 0 else 0
    calmar = cagr / max_dd_pct if max_dd_pct > 0 else 0

    # Per-year breakdown
    tdf = pd.DataFrame(trades)
    tdf['date'] = pd.to_datetime(tdf['date'])
    tdf['year'] = tdf['date'].dt.year
    yearly = tdf.groupby('year').agg(
        trades=('pnl','count'),
        gross_pnl=('pnl','sum'),
        wins=('pnl', lambda x: (x>0).sum()),
        avg_pts=('pts','mean'),
        max_win=('pnl','max'),
        max_loss=('pnl','min'),
    )
    yearly['win_rate'] = yearly['wins']/yearly['trades']

    # Yearly equity
    cum_pnl = 0
    yearly_equity = {}
    for yr in sorted(yearly.index):
        cum_pnl += yearly.loc[yr, 'gross_pnl']
        yearly_equity[yr] = CAPITAL + cum_pnl
    yearly['ending_equity'] = yearly.index.map(yearly_equity)
    yearly['return_pct'] = yearly['gross_pnl'] / (yearly['ending_equity'] - yearly['gross_pnl']) * 100

    return {
        'total_pnl': total_pnl,
        'final_equity': final_equity,
        'cagr': cagr,
        'sharpe': float(sharpe),
        'calmar': calmar,
        'max_dd': max_dd,
        'max_dd_pct': max_dd_pct,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': pf,
        'avg_win': avg_win,
        'avg_loss': abs(np.mean(losses)) if losses else 0,
        'wl_ratio': wl_ratio,
        'years': years,
        'yearly': yearly,
    }


def main():
    print("=" * 90)
    print("10-YEAR INTRADAY P&L / CAGR CALCULATION — 2 Passed Signals")
    print("=" * 90)
    print(f"Capital: ₹{CAPITAL/1e5:.0f}L  |  Lot size: {NIFTY_LOT_SIZE}  |  Slippage: {SLIPPAGE_PCT:.2%}/side")
    print()

    print("Loading data...", flush=True)
    df_daily = load_daily_from_trade_logs()
    # Use from 2016 onwards for ~10 years
    df_daily = df_daily[df_daily['date'] >= '2016-01-01'].reset_index(drop=True)
    print(f"  {len(df_daily)} trading days: {df_daily['date'].min().date()} to {df_daily['date'].max().date()}")
    total_years = (df_daily['date'].max() - df_daily['date'].min()).days / 365.25
    print(f"  Span: {total_years:.1f} years")
    print()

    combined_daily_pnl = {}

    for signal_id, cfg in SIGNALS.items():
        print(f"{'━' * 90}")
        print(f"  {signal_id}")
        print(f"{'━' * 90}")

        trades, daily_pnl, equity = run_full_backtest(signal_id, cfg, df_daily)
        m = compute_metrics(trades, daily_pnl, equity)

        # Accumulate for combined
        for d, p in daily_pnl.items():
            combined_daily_pnl.setdefault(d, 0)
            combined_daily_pnl[d] += p

        fe = m['final_equity']; tp = m['total_pnl']; mdd = m['max_dd']
        aw = m['avg_win']; al = m['avg_loss']
        print(f"\n  {'METRIC':<28s} {'VALUE':>15s}")
        print(f"  {'─'*44}")
        print(f"  {'Starting Capital':<28s} Rs{CAPITAL:>13,.0f}")
        print(f"  {'Final Equity':<28s} Rs{fe:>13,.0f}")
        print(f"  {'Total P&L':<28s} Rs{tp:>13,.0f}")
        print(f"  {'CAGR':<28s} {m['cagr']:>14.1%}")
        print(f"  {'Sharpe Ratio':<28s} {m['sharpe']:>15.2f}")
        print(f"  {'Calmar Ratio':<28s} {m['calmar']:>15.2f}")
        print(f"  {'Max Drawdown':<28s} Rs{mdd:>13,.0f} ({m['max_dd_pct']:.1%})")
        print(f"  {'Total Trades':<28s} {m['total_trades']:>15,d}")
        print(f"  {'Win Rate':<28s} {m['win_rate']:>14.1%}")
        print(f"  {'Profit Factor':<28s} {m['profit_factor']:>15.2f}")
        print(f"  {'Avg Win':<28s} Rs{aw:>13,.0f}")
        print(f"  {'Avg Loss':<28s} Rs{al:>13,.0f}")
        print(f"  {'Win/Loss Ratio':<28s} {m['wl_ratio']:>15.2f}")
        print(f"  {'Period':<28s} {m['years']:>14.1f} yrs")

        print(f"\n  YEARLY BREAKDOWN:")
        print(f"  {'Year':<6} {'Trades':>7} {'P&L':>12} {'WR':>6} {'Avg Pts':>8} {'Return%':>8} {'Equity':>14}")
        print(f"  {'─'*63}")
        for yr, row in m['yearly'].iterrows():
            print(f"  {yr:<6} {int(row['trades']):>7} {row['gross_pnl']:>11,.0f}₹ "
                  f"{row['win_rate']:>5.0%} {row['avg_pts']:>7.1f} "
                  f"{row['return_pct']:>7.1f}% {row['ending_equity']:>13,.0f}₹")

        print()

    # ── COMBINED PORTFOLIO ──
    print(f"\n{'━' * 90}")
    print(f"  COMBINED PORTFOLIO (Both Signals Trading Together)")
    print(f"{'━' * 90}")

    combined_total = sum(combined_daily_pnl.values())
    combined_equity = CAPITAL + combined_total
    dr = pd.Series(combined_daily_pnl) / CAPITAL
    combined_sharpe = dr.mean()/dr.std()*np.sqrt(252) if len(dr)>1 and dr.std()>0 else 0

    sorted_dates = sorted(combined_daily_pnl.keys())
    cum = 0; peak = 0; max_dd = 0
    for d in sorted_dates:
        cum += combined_daily_pnl[d]
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > max_dd: max_dd = dd

    years = (max(sorted_dates) - min(sorted_dates)).days / 365.25
    cagr = (combined_equity/CAPITAL)**(1/years) - 1 if years > 0 else 0
    max_dd_pct = max_dd / CAPITAL

    # Yearly combined
    yearly_pnl = {}
    for d, p in combined_daily_pnl.items():
        yr = d.year
        yearly_pnl.setdefault(yr, 0)
        yearly_pnl[yr] += p

    calmar_c = cagr/max_dd_pct if max_dd_pct > 0 else 0
    print(f"\n  {'Starting Capital':<28s} Rs{CAPITAL:>13,.0f}")
    print(f"  {'Final Equity':<28s} Rs{combined_equity:>13,.0f}")
    print(f"  {'Total P&L':<28s} Rs{combined_total:>13,.0f}")
    print(f"  {'CAGR':<28s} {cagr:>14.1%}")
    print(f"  {'Sharpe Ratio':<28s} {combined_sharpe:>15.2f}")
    print(f"  {'Calmar (CAGR/MaxDD%)':<28s} {calmar_c:>15.2f}")
    print(f"  {'Max Drawdown':<28s} Rs{max_dd:>13,.0f} ({max_dd_pct:.1%})")
    print(f"  {'Period':<28s} {years:>14.1f} yrs")

    print(f"\n  YEARLY COMBINED P&L:")
    print(f"  {'Year':<6} {'P&L':>12} {'Cum P&L':>12} {'Equity':>14} {'Return%':>8}")
    print(f"  {'─'*54}")
    cum_p = 0
    for yr in sorted(yearly_pnl.keys()):
        p = yearly_pnl[yr]
        prev_eq = CAPITAL + cum_p
        cum_p += p
        eq = CAPITAL + cum_p
        ret = p/prev_eq*100
        print(f"  {yr:<6} {p:>11,.0f}₹ {cum_p:>11,.0f}₹ {eq:>13,.0f}₹ {ret:>7.1f}%")

    print(f"\n{'━' * 90}")
    print(f"  10L → ₹{combined_equity:,.0f} in {years:.1f} years = {cagr:.1%} CAGR")
    print(f"{'━' * 90}")


if __name__ == '__main__':
    main()

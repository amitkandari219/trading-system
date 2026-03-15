"""
Simple trade summary. No compounding, no costs, no lots. Just points.
"""

import json, os, csv
from collections import defaultdict
import numpy as np
import pandas as pd
import psycopg2
from backtest.indicators import add_all_indicators, historical_volatility
from config.settings import DATABASE_DSN

SLIPPAGE = 0  # zero — raw signal performance

def _ec(row, prev, cond):
    ind=cond.get('indicator',''); op=cond.get('op','>'); val=cond.get('value')
    if ind not in row.index: return False
    iv=row[ind]
    if pd.isna(iv): return False
    if isinstance(val,str) and val in row.index:
        tv=row[val]
        if pd.isna(tv): return False
    elif val is None: return False
    else:
        try: tv=float(val)
        except: return False
    if op=='>': return iv>tv
    elif op=='<': return iv<tv
    elif op=='>=': return iv>=tv
    elif op=='<=': return iv<=tv
    elif op=='crosses_above':
        if prev is None: return False
        pi=prev.get(ind,np.nan); pt=prev.get(val,tv) if isinstance(val,str) else tv
        return not pd.isna(pi) and not pd.isna(pt) and pi<=pt and iv>tv
    elif op=='crosses_below':
        if prev is None: return False
        pi=prev.get(ind,np.nan); pt=prev.get(val,tv) if isinstance(val,str) else tv
        return not pd.isna(pi) and not pd.isna(pt) and pi>=pt and iv<tv
    return False

def _ecs(row, prev, conds):
    return bool(conds) and all(_ec(row, prev, c) for c in conds)


def bt(df, rules, stop=0.02, tp=0.0, hold=0):
    n=len(df); closes=df['close'].values; dates=df['date'].values
    el=rules.get('entry_long',[]); es=rules.get('entry_short',[])
    xl=rules.get('exit_long',[]); xs=rules.get('exit_short',[])
    trades=[]; pos=None; ep=0; ei=0
    for i in range(1,n):
        r=df.iloc[i]; p=df.iloc[i-1]
        if pos is None:
            if _ecs(r,p,el): pos='LONG'; ep=closes[i]; ei=i
            elif _ecs(r,p,es): pos='SHORT'; ep=closes[i]; ei=i
        else:
            c=closes[i]; dh=i-ei
            if pos=='LONG': lp=(ep-c)/ep
            else: lp=(c-ep)/ep
            xr=None
            if lp>=stop: xr='stop'
            elif tp>0:
                gp=(c-ep)/ep if pos=='LONG' else (ep-c)/ep
                if gp>=tp: xr='tp'
            if not xr and hold>0 and dh>=hold: xr='hold'
            if not xr:
                ex=xl if pos=='LONG' else xs
                if _ecs(r,p,ex): xr='signal'
            if xr:
                pts=(c-ep) if pos=='LONG' else (ep-c)
                trades.append({'entry_date':str(dates[ei])[:10],'exit_date':str(dates[i])[:10],
                               'dir':pos,'entry':round(ep,1),'exit':round(c,1),
                               'pts':round(pts,1),'reason':xr,'days':dh})
                pos=None
    return trades


def bt_scoring(df, sr):
    n=len(df); closes=df['close'].values; dates=df['date'].values
    trades=[]; pos=None; ep=0; ei=0; eth=0; sz=1.0
    for i in range(1,n):
        r=df.iloc[i]; p=df.iloc[i-1]; score=0
        d20l=_ecs(r,p,sr['DRY_20']['entry_long']); d20x=_ecs(r,p,sr['DRY_20']['exit_long'])
        d12l=_ecs(r,p,sr['DRY_12']['entry_long']); d12s=_ecs(r,p,sr['DRY_12']['entry_short'])
        d16l=_ecs(r,p,sr['DRY_16']['entry_long']); d16s=_ecs(r,p,sr['DRY_16']['entry_short'])
        if d20x: score=0
        else:
            if d20l: score+=2
            if d12l: score+=1
            if d12s: score-=1
            if d16l: score+=1
            if d16s: score-=1
        if pos is None:
            if score>=3: pos='LONG'; ep=closes[i]; ei=i; eth=3; sz=1.0
            elif score>=2: pos='LONG'; ep=closes[i]; ei=i; eth=2; sz=0.5
            elif score<=-2: pos='SHORT'; ep=closes[i]; ei=i; eth=-2; sz=0.5
        else:
            c=closes[i]
            if pos=='LONG': lp=(ep-c)/ep
            else: lp=(c-ep)/ep
            xr=None
            if lp>=0.02: xr='stop'
            elif pos=='LONG' and score<eth: xr='score'
            elif pos=='SHORT' and score>eth: xr='score'
            if xr:
                pts=((c-ep) if pos=='LONG' else (ep-c))*sz
                trades.append({'entry_date':str(dates[ei])[:10],'exit_date':str(dates[i])[:10],
                               'dir':pos,'entry':round(ep,1),'exit':round(c,1),
                               'pts':round(pts,1),'reason':xr,'days':i-ei})
                pos=None
    return trades


def bt_seq5(df, gr, kr):
    n=len(df); closes=df['close'].values; dates=df['date'].values
    regimes=df['regime'].values if 'regime' in df.columns else ['UNKNOWN']*n
    trades=[]; pos=None; ep=0; ei=0; pl=-999; ps=-999; pd2=0
    for i in range(1,n):
        r=df.iloc[i]; p=df.iloc[i-1]; regime=str(regimes[i])
        trending=regime in ('TRENDING_UP','TRENDING_DOWN','TRENDING')
        if pos:
            pd2+=1; c=closes[i]
            if pos=='LONG': lp=(ep-c)/ep
            else: lp=(c-ep)/ep
            xr=None
            if lp>=0.02: xr='stop'
            elif pd2>=10: xr='hold'
            elif pos=='LONG' and r['low']<p['low']: xr='struct'
            elif pos=='SHORT' and r['high']>p['high']: xr='struct'
            if xr:
                pts=(c-ep) if pos=='LONG' else (ep-c)
                trades.append({'entry_date':str(dates[ei])[:10],'exit_date':str(dates[i])[:10],
                               'dir':pos,'entry':round(ep,1),'exit':round(c,1),
                               'pts':round(pts,1),'reason':xr,'days':pd2})
                pos=None
            continue
        adx_ok=pd.notna(r.get('adx_14')) and float(r['adx_14'])>25
        gl=r['high']>p['high'] and r['low']>p['low'] and adx_ok and trending
        gs=r['low']<p['low'] and r['high']<p['high'] and adx_ok and trending
        if gl: pl=i; ps=-999
        if gs: ps=i; pl=-999
        if pl>=0 and (i-pl)>5: pl=-999
        if ps>=0 and (i-ps)>5: ps=-999
        kl=r['close']>p['close'] and r['volume']<p['volume']
        ks=r['close']<p['close'] and r['volume']>p['volume']
        if pl>=0 and pl!=i and kl: pos='LONG'; ep=closes[i]; ei=i; pd2=0; pl=-999
        elif ps>=0 and ps!=i and ks: pos='SHORT'; ep=closes[i]; ei=i; pd2=0; ps=-999
    return trades


def bt_and(df, ra, rb):
    n=len(df); closes=df['close'].values; dates=df['date'].values
    trades=[]; pos=None; ep=0; ei=0
    for i in range(1,n):
        r=df.iloc[i]; p=df.iloc[i-1]
        if pos is None:
            if _ecs(r,p,ra.get('entry_long',[])) and _ecs(r,p,rb.get('entry_long',[])): pos='LONG'; ep=closes[i]; ei=i
            elif _ecs(r,p,ra.get('entry_short',[])) and _ecs(r,p,rb.get('entry_short',[])): pos='SHORT'; ep=closes[i]; ei=i
        else:
            c=closes[i]
            if pos=='LONG': lp=(ep-c)/ep
            else: lp=(c-ep)/ep
            xr=None
            if lp>=0.02: xr='stop'
            else:
                exl=_ecs(r,p,ra.get('exit_long',[])) or _ecs(r,p,rb.get('exit_long',[]))
                exs=_ecs(r,p,ra.get('exit_short',[])) or _ecs(r,p,rb.get('exit_short',[]))
                if (pos=='LONG' and exl) or (pos=='SHORT' and exs): xr='signal'
            if xr:
                pts=(c-ep) if pos=='LONG' else (ep-c)
                trades.append({'entry_date':str(dates[ei])[:10],'exit_date':str(dates[i])[:10],
                               'dir':pos,'entry':round(ep,1),'exit':round(c,1),
                               'pts':round(pts,1),'reason':xr,'days':i-ei})
                pos=None
    return trades


def main():
    conn=psycopg2.connect(DATABASE_DSN)
    df_raw=pd.read_sql("SELECT date,open,high,low,close,volume,india_vix FROM nifty_daily ORDER BY date",conn)
    conn.close()
    df_raw['date']=pd.to_datetime(df_raw['date'])
    df=add_all_indicators(df_raw)
    df['hvol_6']=historical_volatility(df['close'],period=6)
    df['hvol_100']=historical_volatility(df['close'],period=100)
    df['date']=df_raw['date']; df['india_vix']=df_raw['india_vix']
    from regime_labeler import RegimeLabeler
    rd=RegimeLabeler().label_full_history(df_raw)
    df['regime']=df['date'].map(rd).fillna('UNKNOWN')
    print(f"Data: {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}\n")

    def lr(path):
        with open(path) as f: d=json.load(f)
        return d.get('rules',d.get('backtest_rule',d.get('dsl_rule',d)))

    sr={}
    for sid,fn in [('DRY_20','validation_results/kaufman_dry_20_fixed.json'),
                    ('DRY_16','validation_results/kaufman_dry_16_fixed.json'),
                    ('DRY_12','validation_results/kaufman_dry_12_fixed.json')]:
        sr[sid]=lr(fn)
    for sid in ['GUJRAL_DRY_7','GUJRAL_DRY_8','GUJRAL_DRY_9','GRIMES_DRY_3_2','KAUFMAN_DRY_8']:
        p=f'dsl_results/BEST/{sid}.json'
        if os.path.exists(p): sr[sid]=lr(p)

    configs={
        'DRY_20':('s',sr['DRY_20'],0.02,0,0),
        'DRY_16':('s',sr['DRY_16'],0.02,0.03,0),
        'DRY_12':('s',sr['DRY_12'],0.02,0.03,7),
        'GUJRAL_DRY_7':('s',sr.get('GUJRAL_DRY_7',{}),0.02,0,0),
        'GUJRAL_DRY_8':('s',sr.get('GUJRAL_DRY_8',{}),0.02,0,0),
        'GUJRAL_DRY_9':('s',sr.get('GUJRAL_DRY_9',{}),0.02,0,0),
        'GRIMES_DRY_3_2':('s',sr.get('GRIMES_DRY_3_2',{}),0.02,0,10),
        'SCORING':('sc',sr),
        'KAU16+KAU8':('a',sr.get('DRY_16',{}),sr.get('KAUFMAN_DRY_8',{})),
        'COMBINATION':('q',sr.get('GRIMES_DRY_3_2',{}),sr.get('DRY_12',{})),
    }

    os.makedirs('trade_summary',exist_ok=True)
    master=[]

    for name,cfg in configs.items():
        if cfg[0]=='s': trades=bt(df,cfg[1],cfg[2],cfg[3],cfg[4])
        elif cfg[0]=='sc': trades=bt_scoring(df,cfg[1])
        elif cfg[0]=='a': trades=bt_and(df,cfg[1],cfg[2])
        elif cfg[0]=='q': trades=bt_seq5(df,cfg[1],cfg[2])
        else: trades=[]

        # Running total
        run=0
        for t in trades:
            run+=t['pts']
            t['running']=round(run,1)

        # Save CSV
        if trades:
            with open(f'trade_summary/{name}_trades.csv','w',newline='') as f:
                w=csv.DictWriter(f,['entry_date','exit_date','dir','entry','exit','pts','running','reason','days'])
                w.writeheader(); w.writerows(trades)

        # Annual summary
        by_yr=defaultdict(list)
        for t in trades: by_yr[t['exit_date'][:4]].append(t)
        annual=[]; yr_run=0
        for yr in sorted(by_yr):
            tt=by_yr[yr]; wins=sum(1 for t in tt if t['pts']>0)
            pts=round(sum(t['pts'] for t in tt),1); yr_run+=pts
            annual.append({'year':yr,'trades':len(tt),'wins':wins,'losses':len(tt)-wins,
                          'pts':pts,'running':round(yr_run,1)})

        if annual:
            with open(f'trade_summary/{name}_annual.csv','w',newline='') as f:
                w=csv.DictWriter(f,['year','trades','wins','losses','pts','running'])
                w.writeheader(); w.writerows(annual)

        # Print
        total_pts=round(sum(t['pts'] for t in trades),1) if trades else 0
        total_tr=len(trades)
        wr=round(sum(1 for t in trades if t['pts']>0)/total_tr*100,1) if total_tr else 0
        best_t=max(trades,key=lambda t:t['pts']) if trades else None
        worst_t=min(trades,key=lambda t:t['pts']) if trades else None

        best_yr=max(annual,key=lambda a:a['pts']) if annual else None
        worst_yr=min(annual,key=lambda a:a['pts']) if annual else None

        print(f"{'='*80}")
        print(f"  {name}")
        print(f"{'='*80}")
        if annual:
            print(f"  {'Year':>6s} {'Trades':>6s} {'Wins':>5s} {'Losses':>6s} {'Points':>8s} {'Running':>10s}")
            print(f"  {'-'*6} {'-'*6} {'-'*5} {'-'*6} {'-'*8} {'-'*10}")
            for a in annual:
                print(f"  {a['year']:>6s} {a['trades']:>6d} {a['wins']:>5d} {a['losses']:>6d} {a['pts']:>+8.1f} {a['running']:>+10.1f}")
        print(f"\n  Total trades:    {total_tr}")
        print(f"  Win rate:        {wr}%")
        print(f"  Total points:    {total_pts:+,.1f}")
        if best_t:
            print(f"  Best trade:      {best_t['pts']:+.1f} pts ({best_t['entry_date']} → {best_t['exit_date']})")
        if worst_t:
            print(f"  Worst trade:     {worst_t['pts']:+.1f} pts ({worst_t['entry_date']} → {worst_t['exit_date']})")
        print()

        master.append({
            'signal':name,'trades':total_tr,'win_pct':wr,'total_pts':total_pts,
            'best_yr':f"{best_yr['year']} ({best_yr['pts']:+.0f})" if best_yr else '—',
            'worst_yr':f"{worst_yr['year']} ({worst_yr['pts']:+.0f})" if worst_yr else '—',
        })

    # Final comparison
    print(f"\n{'='*100}")
    print("FINAL COMPARISON")
    print(f"{'='*100}")
    print(f"{'Signal':16s} {'Trades':>6s} {'Win%':>6s} {'Total Pts':>10s} {'Best Year':>18s} {'Worst Year':>18s}")
    print(f"{'-'*16} {'-'*6} {'-'*6} {'-'*10} {'-'*18} {'-'*18}")
    for m in master:
        print(f"{m['signal']:16s} {m['trades']:>6d} {m['win_pct']:>5.1f}% {m['total_pts']:>+10,.1f} {m['best_yr']:>18s} {m['worst_yr']:>18s}")

    # Save master
    with open('trade_summary/MASTER_SUMMARY.txt','w') as f:
        f.write("SIMPLE TRADE SUMMARY — ALL SIGNALS\n")
        f.write(f"Data: {df['date'].min().date()} to {df['date'].max().date()}\n\n")
        f.write(f"{'Signal':16s} {'Trades':>6s} {'Win%':>6s} {'Total Pts':>10s} {'Best Year':>18s} {'Worst Year':>18s}\n")
        f.write(f"{'-'*16} {'-'*6} {'-'*6} {'-'*10} {'-'*18} {'-'*18}\n")
        for m in master:
            f.write(f"{m['signal']:16s} {m['trades']:>6d} {m['win_pct']:>5.1f}% {m['total_pts']:>+10,.1f} {m['best_yr']:>18s} {m['worst_yr']:>18s}\n")

    print(f"\nAll files saved to trade_summary/")


if __name__=='__main__':
    main()

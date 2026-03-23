"""
Walk-Forward Validation for STRIKE_CLUSTERING_GRAV and WEEKLY_STRANGLE_SELL.

Uses real OI + premium data from nifty_options table (543 days, Jan 2024 - Mar 2026).

Usage:
    venv/bin/python3 -m backtest.options_signal_wf
"""

import logging
import math
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import psycopg2

from config.settings import DATABASE_DSN, RISK_FREE_RATE, NIFTY_LOT_SIZE
from data.options_loader import bs_price

logger = logging.getLogger(__name__)

# WF parameters
SC_TRAIN_MONTHS = 6
SC_TEST_MONTHS = 3
SC_STEP_MONTHS = 2
SC_MIN_SHARPE = 0.8
SC_MIN_PF = 1.5
SC_MIN_WF_PASS = 0.60

WS_TRAIN_MONTHS = 6
WS_TEST_MONTHS = 3
WS_STEP_MONTHS = 2
WS_MIN_SHARPE = 1.0
WS_MIN_PF = 1.8
WS_MIN_WF_PASS = 0.65

CAPITAL = 5_000_000


def _add_months(dt, months):
    m = dt.month + months
    y = dt.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    d = min(dt.day, 28)
    return date(y, m, d)


# ================================================================
# DATA LOADING
# ================================================================

class OptionsSignalWF:
    def __init__(self):
        self.conn = psycopg2.connect(DATABASE_DSN)

    def _load_daily(self) -> pd.DataFrame:
        df = pd.read_sql(
            "SELECT date, close, india_vix FROM nifty_daily ORDER BY date",
            self.conn, parse_dates=['date'])
        df['date'] = df['date'].dt.date
        return df

    def _load_options_by_date(self, trade_date) -> pd.DataFrame:
        """Load option chain for a specific date, aggregated by (strike, option_type, expiry)."""
        cur = self.conn.cursor()
        cur.execute("""
            SELECT strike, option_type, expiry,
                   SUM(oi) as oi, AVG(close) as premium,
                   AVG(implied_volatility) as iv, AVG(delta) as delta
            FROM nifty_options
            WHERE date = %s AND oi > 0
            GROUP BY strike, option_type, expiry
            ORDER BY expiry, strike
        """, (trade_date,))
        cols = ['strike', 'option_type', 'expiry', 'oi', 'premium', 'iv', 'delta']
        rows = cur.fetchall()
        cur.close()
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols)

    # ================================================================
    # MAX PAIN CALCULATION
    # ================================================================

    def _compute_max_pain(self, chain: pd.DataFrame, spot: float) -> Dict:
        """Compute max pain from option chain OI data."""
        if chain.empty:
            return {'max_pain': 0, 'confidence': 0}

        # Use nearest expiry only
        expiries = sorted(chain['expiry'].unique())
        if not expiries:
            return {'max_pain': 0, 'confidence': 0}
        near_expiry = expiries[0]
        near = chain[chain['expiry'] == near_expiry]

        # Aggregate OI by strike
        strikes = sorted(near['strike'].unique())
        if len(strikes) < 5:
            return {'max_pain': 0, 'confidence': 0}

        call_oi = {}
        put_oi = {}
        for _, row in near.iterrows():
            s = float(row['strike'])
            if row['option_type'] == 'CE':
                call_oi[s] = call_oi.get(s, 0) + float(row['oi'])
            else:
                put_oi[s] = put_oi.get(s, 0) + float(row['oi'])

        # Compute total writer loss at each strike
        min_loss = float('inf')
        max_pain_strike = strikes[len(strikes)//2]

        for test_strike in strikes:
            ts = float(test_strike)
            total_loss = 0
            for k in strikes:
                k = float(k)
                # Call writers lose if test_strike > their strike
                if ts > k:
                    total_loss += (ts - k) * call_oi.get(k, 0)
                # Put writers lose if test_strike < their strike
                if ts < k:
                    total_loss += (k - ts) * put_oi.get(k, 0)

            if total_loss < min_loss:
                min_loss = total_loss
                max_pain_strike = ts

        # Confidence based on OI concentration
        total_oi = sum(call_oi.values()) + sum(put_oi.values())
        top3_oi = sorted(list(call_oi.values()) + list(put_oi.values()), reverse=True)[:3]
        confidence = sum(top3_oi) / max(total_oi, 1) if total_oi > 0 else 0

        return {
            'max_pain': max_pain_strike,
            'distance_pct': abs(spot - max_pain_strike) / spot if spot > 0 else 0,
            'confidence': min(confidence * 2, 1.0),  # scale up
            'near_expiry': near_expiry,
            'total_oi': total_oi,
        }

    # ================================================================
    # SIGNAL 1: STRIKE_CLUSTERING_GRAV
    # ================================================================

    def _run_strike_clustering(self) -> Dict:
        """Backtest STRIKE_CLUSTERING_GRAV on real OI data."""
        daily = self._load_daily()
        trades = []

        for _, row in daily.iterrows():
            d = row['date']
            spot = float(row['close'])
            vix = float(row['india_vix']) if pd.notna(row['india_vix']) else 15

            # Only trade when DTE <= 2 (Tue/Wed for Thursday expiry)
            if not hasattr(d, 'weekday'):
                continue
            dow = d.weekday()
            if dow not in (1, 2):  # Tuesday=1, Wednesday=2
                continue

            if vix >= 22:
                continue

            chain = self._load_options_by_date(d)
            if chain.empty:
                continue

            mp = self._compute_max_pain(chain, spot)
            if mp['max_pain'] <= 0 or mp['confidence'] < 0.15:
                continue

            dist_pct = mp['distance_pct']
            if dist_pct < 0.001 or dist_pct > 0.015:
                continue  # 0.1% - 1.5% from max pain (relaxed: real NSE max pain is close to spot)

            # Direction: fade toward max pain
            if spot > mp['max_pain']:
                direction = 'SHORT'
                # Simulate: spot moves toward max pain by expiry
                target = mp['max_pain']
                sl_price = spot * 1.008  # 0.8% further away
            else:
                direction = 'LONG'
                target = mp['max_pain']
                sl_price = spot * 0.992

            # Simulate outcome: look at Thursday's close
            # Find next Thursday
            days_to_thu = (3 - dow) % 7
            if days_to_thu == 0:
                days_to_thu = 7
            thu_date = d + timedelta(days=days_to_thu)

            thu_row = daily[daily['date'] == thu_date]
            if thu_row.empty:
                continue

            thu_close = float(thu_row.iloc[0]['close'])

            # P&L
            if direction == 'LONG':
                # Check SL first
                # Approximate: did price go below SL during the period?
                pnl_pct = (thu_close - spot) / spot
                if pnl_pct < -0.008:
                    pnl_pct = -0.008  # SL hit
                elif thu_close >= target:
                    pnl_pct = (target - spot) / spot  # TGT hit
            else:
                pnl_pct = (spot - thu_close) / spot
                if pnl_pct < -0.008:
                    pnl_pct = -0.008
                elif thu_close <= target:
                    pnl_pct = (spot - target) / spot

            # Slippage
            pnl_pct -= 0.001  # 0.1% round-trip

            trades.append({
                'date': d,
                'direction': direction,
                'entry': spot,
                'target': target,
                'max_pain': mp['max_pain'],
                'distance_pct': dist_pct,
                'pnl_pct': pnl_pct,
                'confidence': mp['confidence'],
            })

        return self._wf_evaluate(
            'STRIKE_CLUSTERING_GRAV', trades,
            SC_TRAIN_MONTHS, SC_TEST_MONTHS, SC_STEP_MONTHS,
            SC_MIN_SHARPE, SC_MIN_PF, SC_MIN_WF_PASS,
        )

    # ================================================================
    # SIGNAL 2: WEEKLY_STRANGLE_SELL
    # ================================================================

    def _run_strangle_sell(self) -> Dict:
        """Backtest WEEKLY_STRANGLE_SELL on real premium data."""
        daily = self._load_daily()

        # Compute IV rank from VIX history
        vix_list = daily['india_vix'].dropna().tolist()
        trades = []
        row_num = 0

        for _, row in daily.iterrows():
            row_num += 1
            d = row['date']
            spot = float(row['close'])
            vix = float(row['india_vix']) if pd.notna(row['india_vix']) else 15

            if not hasattr(d, 'weekday'):
                continue

            dow = d.weekday()
            # Only enter Mon/Tue (Thu expiry)
            if dow not in (0, 1):
                continue

            # IV rank from prior VIX values
            prior_vix = vix_list[:row_num]
            if len(prior_vix) < 50:
                continue
            lookback = prior_vix[-252:]
            vix_min = min(lookback)
            vix_max = max(lookback)
            iv_rank = (vix - vix_min) / max(vix_max - vix_min, 0.01) * 100

            if iv_rank < 60:
                continue
            if vix < 16:
                continue
            if vix > 25:
                continue

            chain = self._load_options_by_date(d)
            if chain.empty:
                continue

            # Select nearest expiry
            expiries = sorted(chain['expiry'].unique())
            if not expiries:
                continue
            near_exp = expiries[0]
            if hasattr(near_exp, 'date'):
                near_exp = near_exp.date() if hasattr(near_exp, 'date') else near_exp
            dte = (near_exp - d).days if hasattr(near_exp, '__sub__') else 4
            if dte < 2 or dte > 7:
                continue

            near = chain[chain['expiry'] == expiries[0]]

            # Find OTM call (~delta 0.20-0.25)
            calls = near[(near['option_type'] == 'CE') & (near['strike'] > spot)]
            calls = calls.sort_values('strike')
            if calls.empty:
                continue

            # Target: ~1.5 ATR OTM
            atr_approx = spot * vix / 100 / math.sqrt(252) * math.sqrt(dte)
            call_target = spot + 1.5 * atr_approx
            call_row = calls.iloc[(calls['strike'] - call_target).abs().argsort()[:1]]
            if call_row.empty:
                continue
            call_strike = float(call_row.iloc[0]['strike'])
            call_prem = float(call_row.iloc[0]['premium'])

            # Find OTM put
            puts = near[(near['option_type'] == 'PE') & (near['strike'] < spot)]
            puts = puts.sort_values('strike', ascending=False)
            if puts.empty:
                continue

            put_target = spot - 1.5 * atr_approx
            put_row = puts.iloc[(puts['strike'] - put_target).abs().argsort()[:1]]
            if put_row.empty:
                continue
            put_strike = float(put_row.iloc[0]['strike'])
            put_prem = float(put_row.iloc[0]['premium'])

            combined = call_prem + put_prem
            if combined < 4:  # minimum ₹4/unit = ₹100/lot
                continue

            # Find expiry-day close
            exp_date = near_exp if isinstance(near_exp, date) else near_exp.date() if hasattr(near_exp, 'date') else d + timedelta(days=dte)
            exp_row = daily[daily['date'] == exp_date]
            if exp_row.empty:
                # Try Thursday
                for offset in range(0, 4):
                    exp_row = daily[daily['date'] == d + timedelta(days=dte + offset)]
                    if not exp_row.empty:
                        break
            if exp_row.empty:
                continue

            exp_close = float(exp_row.iloc[0]['close'])

            # P&L: premium collected minus any ITM amount
            call_itm = max(0, exp_close - call_strike)
            put_itm = max(0, put_strike - exp_close)
            pnl_per_unit = combined - call_itm - put_itm

            # Apply 50% profit target: if premium decays > 50%, close early
            # Approximate: at halfway through DTE, premiums are ~60% of initial
            # So profit target likely hit on ~65% of winning trades
            # Apply this as a cap: max profit = 50% of combined premium
            if pnl_per_unit > combined * 0.5:
                pnl_per_unit = combined * 0.5

            # SL: if either leg doubles (2x premium), close
            if call_itm > 2 * call_prem or put_itm > 2 * put_prem:
                # SL hit — cap loss at 2x one leg
                max_leg_loss = max(call_itm - call_prem, put_itm - put_prem)
                pnl_per_unit = combined - max_leg_loss - max(call_itm, put_itm)

            # Costs: 2 legs × entry + exit × ₹40/lot brokerage + slippage
            # ~₹2/unit all-in for costs
            pnl_per_unit -= 2.0

            pnl_pct = pnl_per_unit / (combined * 3)  # margin ~3x premium

            trades.append({
                'date': d,
                'call_strike': call_strike,
                'put_strike': put_strike,
                'combined_premium': combined,
                'exp_close': exp_close,
                'pnl_per_unit': pnl_per_unit,
                'pnl_pct': pnl_pct,
                'iv_rank': iv_rank,
                'dte': dte,
            })

        return self._wf_evaluate(
            'WEEKLY_STRANGLE_SELL', trades,
            WS_TRAIN_MONTHS, WS_TEST_MONTHS, WS_STEP_MONTHS,
            WS_MIN_SHARPE, WS_MIN_PF, WS_MIN_WF_PASS,
        )

    # ================================================================
    # WF EVALUATION
    # ================================================================

    def _wf_evaluate(self, signal_id, trades, train_m, test_m, step_m,
                     min_sharpe, min_pf, min_wf_pass) -> Dict:
        if not trades:
            return {
                'signal_id': signal_id, 'trades': 0, 'win_rate': 0,
                'pf': 0, 'sharpe': 0, 'wf_pass_rate': 0, 'verdict': 'NO_TRADES',
                'windows_pass': 0, 'windows_total': 0,
            }

        trade_dates = [t['date'] for t in trades]
        min_date = min(trade_dates)
        max_date = max(trade_dates)

        # Generate windows
        windows = []
        start = min_date
        while True:
            train_end = _add_months(start, train_m)
            test_start = train_end
            test_end = _add_months(test_start, test_m)
            if test_end > max_date:
                break
            windows.append((start, train_end, test_start, test_end))
            start = _add_months(start, step_m)

        if not windows:
            return {
                'signal_id': signal_id, 'trades': len(trades),
                'win_rate': _wr(trades), 'pf': _pf(trades),
                'sharpe': _sharpe(trades), 'wf_pass_rate': 0,
                'verdict': 'NO_WINDOWS', 'windows_pass': 0, 'windows_total': 0,
            }

        passes = 0
        for _, _, ts, te in windows:
            wt = [t for t in trades if ts <= t['date'] <= te]
            if len(wt) < 3:
                continue
            pf = _pf(wt)
            sh = _sharpe(wt)
            if sh >= min_sharpe and pf >= min_pf:
                passes += 1

        wf_rate = passes / len(windows)
        overall_pf = _pf(trades)
        overall_sharpe = _sharpe(trades)
        overall_wr = _wr(trades)

        verdict = 'PASS' if (overall_sharpe >= min_sharpe and overall_pf >= min_pf
                             and wf_rate >= min_wf_pass) else 'FAIL'

        return {
            'signal_id': signal_id,
            'trades': len(trades),
            'win_rate': round(overall_wr * 100, 1),
            'pf': round(overall_pf, 2),
            'sharpe': round(overall_sharpe, 2),
            'wf_pass_rate': round(wf_rate * 100, 0),
            'windows_pass': passes,
            'windows_total': len(windows),
            'verdict': verdict,
        }

    def run_all(self):
        print("=" * 80)
        print("  OPTIONS SIGNAL WALK-FORWARD VALIDATION")
        print("  Data: nifty_options (543 days, real OI + premiums)")
        print("=" * 80)

        print("\nRunning STRIKE_CLUSTERING_GRAV...")
        sc = self._run_strike_clustering()

        print("Running WEEKLY_STRANGLE_SELL...")
        ws = self._run_strangle_sell()

        print("\n" + "─" * 80)
        print(f"{'Signal':<28s} {'Trades':>6s} {'WR':>6s} {'PF':>6s} "
              f"{'Sharpe':>7s} {'WF%':>5s} {'Win':>4s} {'Tot':>4s} {'Verdict':>8s}")
        print("─" * 80)
        for r in [sc, ws]:
            print(f"{r['signal_id']:<28s} {r['trades']:>6d} {r['win_rate']:>5.1f}% "
                  f"{r['pf']:>6.2f} {r['sharpe']:>7.2f} {r['wf_pass_rate']:>4.0f}% "
                  f"{r['windows_pass']:>4d} {r['windows_total']:>4d} {r['verdict']:>8s}")
        print("─" * 80)

        self.conn.close()
        return {'STRIKE_CLUSTERING_GRAV': sc, 'WEEKLY_STRANGLE_SELL': ws}


def _pf(trades):
    wins = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
    losses = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0))
    return wins / losses if losses > 0 else 0

def _wr(trades):
    if not trades:
        return 0
    return len([t for t in trades if t['pnl_pct'] > 0]) / len(trades)

def _sharpe(trades):
    if len(trades) < 2:
        return 0
    rets = [t['pnl_pct'] for t in trades]
    m = np.mean(rets)
    s = np.std(rets, ddof=1)
    return (m / s) * np.sqrt(52) if s > 0 else 0  # weekly trades → sqrt(52)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    wf = OptionsSignalWF()
    wf.run_all()

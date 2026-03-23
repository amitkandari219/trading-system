"""
Walk-forward validation for global market signals.

Tests GIFT Nifty gap, US overnight, and global composite signals
using the same WF framework as other system signals:
  - 36-month train / 12-month test / 3-month step
  - Pass criteria: Sharpe >= 1.2, PF >= 1.6, 75% windows pass
  - Purge: 21 days, Embargo: 5 days

Data approach:
  - Downloads 5+ years of S&P 500, VIX, DXY, Brent via yfinance
  - Computes synthetic GIFT Nifty gaps using S&P overnight * 0.7 correlation
  - Aligns global data with Nifty daily for per-date evaluation
  - Uses actual Nifty open/high/low/close for trade simulation

Usage:
    python -m backtest.global_walk_forward              # full run
    python -m backtest.global_walk_forward --signal gift # single signal
    python -m backtest.global_walk_forward --dry-run     # print windows only
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ================================================================
# WF PARAMETERS (match system defaults from config/settings.py)
# ================================================================
WF_TRAIN_MONTHS = 36
WF_TEST_MONTHS = 12
WF_STEP_MONTHS = 3
WF_PURGE_DAYS = 21
WF_EMBARGO_DAYS = 5

# Pass criteria
MIN_SHARPE = 1.2
MIN_PROFIT_FACTOR = 1.6
MIN_TRADES = 30
MIN_WIN_RATE = 0.45
MIN_PASS_RATE = 0.75  # 75% of windows must pass

RISK_FREE_RATE = 0.065  # RBI repo rate


class GlobalWalkForward:
    """
    Walk-forward engine for global market signals.

    Downloads and aligns global data with Nifty, then runs
    expanding-window WF on each signal independently.
    """

    def __init__(self, db_conn=None):
        self.db = db_conn
        self._nifty_df = None
        self._global_df = None

    # ================================================================
    # MAIN RUNNER
    # ================================================================
    def run_all(self, signals: List[str] = None) -> Dict:
        """
        Run walk-forward for all global signals.

        Args:
            signals: List of signal names to test. Default: all three.
                     Options: 'gift_gap', 'us_overnight', 'global_composite'

        Returns:
            Dict with per-signal results:
              {signal_name: {windows: [...], pass_rate: float, overall_pass: bool}}
        """
        signals = signals or ['gift_gap', 'us_overnight', 'global_composite']

        # ── Load data ─────────────────────────────────────────────
        logger.info("Loading Nifty and global market data...")
        nifty_df = self._load_nifty_data()
        global_df = self._load_global_data(nifty_df)

        if nifty_df is None or len(nifty_df) < 252:
            return {'error': f'Insufficient Nifty data: {len(nifty_df) if nifty_df is not None else 0} rows'}
        if global_df is None or len(global_df) < 252:
            return {'error': f'Insufficient global data: {len(global_df) if global_df is not None else 0} rows'}

        logger.info(f"Data loaded: Nifty {len(nifty_df)} days, Global {len(global_df)} days")
        logger.info(f"Date range: {nifty_df['date'].min()} to {nifty_df['date'].max()}")

        # ── Generate WF windows ───────────────────────────────────
        windows = self._generate_windows(nifty_df)
        logger.info(f"Generated {len(windows)} walk-forward windows")

        # ── Run each signal ───────────────────────────────────────
        results = {}
        for sig_name in signals:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing signal: {sig_name}")
            logger.info(f"{'='*60}")

            sig_results = self._run_signal(sig_name, nifty_df, global_df, windows)
            results[sig_name] = sig_results

            # Print summary
            pass_count = sum(1 for w in sig_results['windows'] if w['pass'])
            total = len(sig_results['windows'])
            rate = pass_count / total if total > 0 else 0
            logger.info(f"\n{sig_name}: {pass_count}/{total} windows pass ({rate:.0%})")
            logger.info(f"Overall: {'PASS' if sig_results['overall_pass'] else 'FAIL'}")

        return results

    # ================================================================
    # RUN SINGLE SIGNAL
    # ================================================================
    def _run_signal(self, signal_name: str, nifty_df: pd.DataFrame,
                    global_df: pd.DataFrame,
                    windows: List[Dict]) -> Dict:
        """Run walk-forward for a single signal."""
        window_results = []

        for i, window in enumerate(windows):
            train_start = window['train_start']
            train_end = window['train_end']
            test_start = window['test_start']
            test_end = window['test_end']

            # ── Filter data for test window ───────────────────────
            # (These signals don't have trainable parameters,
            #  so we only evaluate on the test window)
            test_nifty = nifty_df[
                (nifty_df['date'] >= test_start) &
                (nifty_df['date'] <= test_end)
            ].copy()
            test_global = global_df[
                (global_df['snapshot_date'] >= test_start) &
                (global_df['snapshot_date'] <= test_end)
            ].copy()

            if len(test_nifty) < 20:
                logger.warning(f"Window {i}: insufficient test data ({len(test_nifty)} days)")
                continue

            # ── Generate trades ───────────────────────────────────
            trades_df = self._generate_trades(signal_name, test_nifty, test_global)

            if trades_df is None or len(trades_df) == 0:
                window_results.append({
                    'window': i,
                    'test_start': str(test_start),
                    'test_end': str(test_end),
                    'trades': 0,
                    'sharpe': 0,
                    'profit_factor': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'max_dd': 0,
                    'pass': False,
                    'reason': 'No trades generated',
                })
                continue

            # ── Compute metrics ───────────────────────────────────
            metrics = self._compute_metrics(trades_df)

            # ── Check pass criteria ───────────────────────────────
            passes = (
                metrics['sharpe'] >= MIN_SHARPE and
                metrics['profit_factor'] >= MIN_PROFIT_FACTOR and
                metrics['trades'] >= MIN_TRADES and
                metrics['win_rate'] >= MIN_WIN_RATE
            )

            reasons = []
            if metrics['sharpe'] < MIN_SHARPE:
                reasons.append(f"Sharpe {metrics['sharpe']:.2f} < {MIN_SHARPE}")
            if metrics['profit_factor'] < MIN_PROFIT_FACTOR:
                reasons.append(f"PF {metrics['profit_factor']:.2f} < {MIN_PROFIT_FACTOR}")
            if metrics['trades'] < MIN_TRADES:
                reasons.append(f"Trades {metrics['trades']} < {MIN_TRADES}")
            if metrics['win_rate'] < MIN_WIN_RATE:
                reasons.append(f"WR {metrics['win_rate']:.0%} < {MIN_WIN_RATE:.0%}")

            window_results.append({
                'window': i,
                'test_start': str(test_start),
                'test_end': str(test_end),
                'trades': metrics['trades'],
                'sharpe': round(metrics['sharpe'], 3),
                'profit_factor': round(metrics['profit_factor'], 3),
                'win_rate': round(metrics['win_rate'], 4),
                'total_return': round(metrics['total_return'], 2),
                'max_dd': round(metrics['max_dd'], 2),
                'avg_return': round(metrics['avg_return'], 4),
                'pass': passes,
                'reason': '; '.join(reasons) if reasons else 'PASS',
            })

            logger.info(
                f"  Window {i} [{test_start} → {test_end}]: "
                f"trades={metrics['trades']}, Sharpe={metrics['sharpe']:.2f}, "
                f"PF={metrics['profit_factor']:.2f}, WR={metrics['win_rate']:.0%} "
                f"{'✓' if passes else '✗'}"
            )

        # ── Overall assessment ────────────────────────────────────
        pass_count = sum(1 for w in window_results if w['pass'])
        total = len(window_results)
        pass_rate = pass_count / total if total > 0 else 0
        last_pass = window_results[-1]['pass'] if window_results else False

        overall = bool(pass_rate >= MIN_PASS_RATE and last_pass)

        return {
            'signal_name': signal_name,
            'windows': window_results,
            'pass_count': int(pass_count),
            'total_windows': int(total),
            'pass_rate': round(float(pass_rate), 4),
            'overall_pass': overall,
            'last_window_pass': bool(last_pass),
        }

    # ================================================================
    # GENERATE TRADES FOR A SIGNAL
    # ================================================================
    def _generate_trades(self, signal_name: str,
                         nifty_df: pd.DataFrame,
                         global_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Dispatch to appropriate signal backtester."""
        try:
            if signal_name == 'gift_gap':
                from signals.gift_nifty_gap import GiftNiftyGapSignal
                sig = GiftNiftyGapSignal()
                return sig.evaluate_backtest(nifty_df, global_df)

            elif signal_name == 'us_overnight':
                from signals.us_overnight import USOvernightSignal
                sig = USOvernightSignal()
                return sig.evaluate_backtest(nifty_df, global_df)

            elif signal_name == 'global_composite':
                from signals.global_composite import GlobalCompositeSignal
                sig = GlobalCompositeSignal()
                return sig.evaluate_backtest(nifty_df, global_df)

            else:
                logger.error(f"Unknown signal: {signal_name}")
                return None
        except Exception as e:
            logger.error(f"Error generating trades for {signal_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ================================================================
    # COMPUTE PERFORMANCE METRICS
    # ================================================================
    def _compute_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Compute standard performance metrics from trade DataFrame."""
        returns = trades_df['return_pct'].values
        n = len(returns)

        if n == 0:
            return {
                'trades': 0, 'sharpe': 0, 'profit_factor': 0,
                'win_rate': 0, 'total_return': 0, 'max_dd': 0,
                'avg_return': 0,
            }

        # Win rate
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        win_rate = len(wins) / n if n > 0 else 0

        # Profit factor
        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.001
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (annualized, assuming ~252 trading days)
        daily_rf = RISK_FREE_RATE / 252
        excess = returns / 100 - daily_rf
        sharpe = (np.mean(excess) / np.std(excess) * np.sqrt(252)
                  if np.std(excess) > 0 else 0)

        # Total return
        cumulative = np.cumprod(1 + returns / 100) - 1
        total_return = cumulative[-1] * 100 if len(cumulative) > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(np.cumprod(1 + returns / 100))
        dd = (np.cumprod(1 + returns / 100) - peak) / peak * 100
        max_dd = abs(np.min(dd)) if len(dd) > 0 else 0

        return {
            'trades': n,
            'sharpe': sharpe,
            'profit_factor': pf,
            'win_rate': win_rate,
            'total_return': total_return,
            'max_dd': max_dd,
            'avg_return': np.mean(returns),
        }

    # ================================================================
    # GENERATE WF WINDOWS
    # ================================================================
    def _generate_windows(self, nifty_df: pd.DataFrame) -> List[Dict]:
        """Generate expanding walk-forward windows."""
        dates = sorted(nifty_df['date'].unique())
        min_date = dates[0]
        max_date = dates[-1]

        windows = []
        train_start = min_date

        while True:
            if isinstance(train_start, np.datetime64):
                train_start = pd.Timestamp(train_start).date()

            train_end_dt = datetime(train_start.year, train_start.month, 1) + timedelta(days=WF_TRAIN_MONTHS * 30)
            train_end = train_end_dt.date()

            # Purge gap
            test_start_dt = train_end_dt + timedelta(days=WF_PURGE_DAYS)
            test_start = test_start_dt.date()

            test_end_dt = test_start_dt + timedelta(days=WF_TEST_MONTHS * 30)
            test_end = test_end_dt.date()

            if test_end > max_date:
                # Last window: use remaining data
                if test_start < max_date:
                    windows.append({
                        'train_start': train_start,
                        'train_end': train_end,
                        'test_start': test_start,
                        'test_end': max_date,
                    })
                break

            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            })

            # Step forward
            step_dt = datetime(train_start.year, train_start.month, 1) + timedelta(days=WF_STEP_MONTHS * 30)
            train_start = step_dt.date()

        return windows

    # ================================================================
    # DATA LOADING
    # ================================================================
    def _load_nifty_data(self) -> Optional[pd.DataFrame]:
        """Load Nifty daily data from DB or yfinance."""
        # Try DB first
        if self.db:
            try:
                df = pd.read_sql("""
                    SELECT date, open, high, low, close, volume, india_vix
                    FROM nifty_daily ORDER BY date
                """, self.db)
                if len(df) > 252:
                    df['date'] = pd.to_datetime(df['date']).dt.date
                    return df
            except Exception as e:
                logger.warning(f"DB load failed, falling back to yfinance: {e}")

        # Fallback: yfinance
        try:
            import yfinance as yf
            nifty = yf.Ticker('^NSEI')
            data = nifty.history(period='10y')
            if data is None or data.empty:
                return None

            df = data.reset_index()
            df.columns = [c.lower() for c in df.columns]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date

            # Try to get India VIX
            try:
                vix = yf.Ticker('^INDIAVIX')
                vix_data = vix.history(period='10y')
                if vix_data is not None and not vix_data.empty:
                    vix_df = vix_data.reset_index()[['Date', 'Close']].rename(
                        columns={'Date': 'date', 'Close': 'india_vix'}
                    )
                    vix_df['date'] = pd.to_datetime(vix_df['date']).dt.date
                    df = df.merge(vix_df, on='date', how='left')
                    df['india_vix'] = df['india_vix'].ffill()
            except Exception:
                df['india_vix'] = 16.0

            return df
        except Exception as e:
            logger.error(f"Failed to load Nifty data: {e}")
            return None

    def _load_global_data(self, nifty_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Load global market data aligned with Nifty dates.

        For backtesting, we construct synthetic snapshots from yfinance data.
        """
        # Try DB first
        if self.db:
            try:
                df = pd.read_sql("""
                    SELECT * FROM global_market_snapshots ORDER BY snapshot_date
                """, self.db)
                if len(df) > 100:
                    df['snapshot_date'] = pd.to_datetime(df['snapshot_date']).dt.date
                    return df
            except Exception:
                pass

        # Build from yfinance
        try:
            import yfinance as yf
            logger.info("Building global dataset from yfinance (this may take a minute)...")

            tickers = {
                'sp500': '^GSPC',
                'nasdaq': '^IXIC',
                'us_vix': '^VIX',
                'dxy': 'DX-Y.NYB',
                'brent': 'BZ=F',
                'hang_seng': '^HSI',
                'nikkei': '^N225',
            }

            dfs = {}
            for key, ticker in tickers.items():
                try:
                    t = yf.Ticker(ticker)
                    data = t.history(period='10y')
                    if data is not None and not data.empty:
                        d = data.reset_index()[['Date', 'Close']].copy()
                        d.columns = ['date', f'{key}_close']
                        d['date'] = pd.to_datetime(d['date']).dt.date
                        # Compute daily change
                        d[f'{key}_change_pct'] = d[f'{key}_close'].pct_change() * 100
                        dfs[key] = d
                        logger.info(f"  {key}: {len(d)} rows")
                except Exception as e:
                    logger.warning(f"  {key}: failed ({e})")

            if not dfs:
                return None

            # Merge all on date
            base = dfs.get('sp500')
            if base is None:
                return None

            result = base.copy()
            for key, df in dfs.items():
                if key != 'sp500':
                    result = result.merge(df, on='date', how='outer')

            result = result.sort_values('date').reset_index(drop=True)

            # Compute derived fields
            result['snapshot_date'] = result['date']

            # US overnight return (60% S&P + 40% Nasdaq)
            sp = result.get('sp500_change_pct', pd.Series(0, index=result.index))
            nq = result.get('nasdaq_change_pct', pd.Series(0, index=result.index))
            result['us_overnight_return'] = (0.6 * sp.fillna(0) + 0.4 * nq.fillna(0))

            # VIX change
            result['us_vix_change_pct'] = result.get('us_vix_change_pct', pd.Series(0, index=result.index))

            # DXY change
            result['dxy_change_pct'] = result.get('dxy_change_pct', pd.Series(0, index=result.index))

            # Brent change
            result['brent_change_pct'] = result.get('brent_change_pct', pd.Series(0, index=result.index))

            # Hang Seng / Nikkei
            result['hang_seng_change_pct'] = result.get('hang_seng_change_pct', pd.Series(0, index=result.index))
            result['nikkei_change_pct'] = result.get('nikkei_change_pct', pd.Series(0, index=result.index))

            # GIFT Nifty proxy: S&P change * 0.7 correlation
            result['sp500_change_pct'] = sp.fillna(0)
            result['gift_nifty_gap_pct'] = None  # Will be computed per-signal using SP proxy

            # Forward fill NaNs
            result = result.ffill()

            logger.info(f"Global dataset built: {len(result)} rows")
            return result

        except ImportError:
            logger.error("yfinance not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to build global dataset: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ================================================================
    # STORE WF RESULTS
    # ================================================================
    def store_results(self, results: Dict) -> bool:
        """Store walk-forward results to DB."""
        if not self.db:
            return False
        try:
            cur = self.db.cursor()
            for sig_name, sig_result in results.items():
                if isinstance(sig_result, str):
                    continue  # Skip error messages
                cur.execute("""
                    INSERT INTO walk_forward_results (
                        signal_name, run_date, pass_rate, overall_pass,
                        total_windows, pass_count, details_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    sig_name, date.today(),
                    sig_result['pass_rate'],
                    sig_result['overall_pass'],
                    sig_result['total_windows'],
                    sig_result['pass_count'],
                    json.dumps(sig_result['windows']),
                ))
            self.db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store WF results: {e}")
            self.db.rollback()
            return False


# ================================================================
# CLI ENTRY POINT
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='Walk-forward validation for global signals')
    parser.add_argument('--signal', type=str, help='Single signal to test')
    parser.add_argument('--dry-run', action='store_true', help='Print windows only')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    # Try DB connection
    db = None
    try:
        import psycopg2
        from config.settings import DATABASE_DSN
        db = psycopg2.connect(DATABASE_DSN)
    except Exception:
        logger.info("No DB connection — using yfinance data only")

    wf = GlobalWalkForward(db_conn=db)

    signals = [args.signal] if args.signal else None
    results = wf.run_all(signals=signals)

    # Print summary
    print("\n" + "=" * 70)
    print("GLOBAL SIGNALS WALK-FORWARD SUMMARY")
    print("=" * 70)
    for sig_name, sig_result in results.items():
        if isinstance(sig_result, str):
            print(f"\n{sig_name}: ERROR — {sig_result}")
            continue
        status = '✓ PASS' if sig_result['overall_pass'] else '✗ FAIL'
        print(f"\n{sig_name}: {status}")
        print(f"  Windows: {sig_result['pass_count']}/{sig_result['total_windows']} pass ({sig_result['pass_rate']:.0%})")
        if sig_result['windows']:
            sharpes = [w['sharpe'] for w in sig_result['windows'] if w['trades'] > 0]
            pfs = [w['profit_factor'] for w in sig_result['windows'] if w['trades'] > 0]
            if sharpes:
                print(f"  Avg Sharpe: {np.mean(sharpes):.2f}, Avg PF: {np.mean(pfs):.2f}")

    if db:
        wf.store_results(results)
        db.close()


if __name__ == '__main__':
    main()

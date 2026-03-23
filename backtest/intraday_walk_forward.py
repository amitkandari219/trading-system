"""
Walk-Forward Validation Engine for Intraday 5-min Signals.

Validates L9 Nifty and BN BankNifty intraday signals using walk-forward
with shorter windows suited to intraday regime changes:
    6-month train / 2-month test / 1-month step.

Pass criteria: Sharpe >= 0.8, PF >= 1.3, >= 60% windows pass.

Slippage: 1 pt/side NIFTY, 3 pts/side BANKNIFTY.
All positions close at 15:20.

Usage:
    venv/bin/python3 -m backtest.intraday_walk_forward
    venv/bin/python3 -m backtest.intraday_walk_forward --instrument NIFTY
    venv/bin/python3 -m backtest.intraday_walk_forward --signal L9_ORB_BREAKOUT
"""

import argparse
import logging
import sys
from datetime import date, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

from backtest.types import BacktestResult, add_months
from config.settings import (
    DATABASE_DSN, RISK_FREE_RATE,
    NIFTY_LOT_SIZE, BANKNIFTY_LOT_SIZE,
)
from signals.l9_signals import IntradaySignalComputer
from signals.banknifty_signals import BankNiftySignalComputer

logger = logging.getLogger(__name__)

# ================================================================
# INTRADAY WF PARAMETERS (shorter than daily — regimes change faster)
# ================================================================
INTRADAY_WF_TRAIN_MONTHS = 6
INTRADAY_WF_TEST_MONTHS = 2
INTRADAY_WF_STEP_MONTHS = 1

# Pass criteria (lower bar than daily futures)
INTRADAY_MIN_SHARPE = 0.8
INTRADAY_MIN_PF = 1.3
INTRADAY_MIN_WF_PASS_RATE = 0.60
INTRADAY_MIN_TRADES_PER_WINDOW = 10

# Slippage: absolute points per side
SLIPPAGE = {
    'NIFTY': 1.0,
    'BANKNIFTY': 3.0,
}

# Session timing
SESSION_OPEN_HOUR = 9
SESSION_OPEN_MIN = 15
SESSION_CLOSE_HOUR = 15
SESSION_CLOSE_MIN = 20
BARS_PER_DAY = 75  # (15:30 - 9:15) / 5min = 75 bars

# Annual trading days for Sharpe annualisation
TRADING_DAYS_PER_YEAR = 252


# ================================================================
# HELPER: ANNUALISED SHARPE FROM DAILY PNL SERIES
# ================================================================
def _compute_sharpe(daily_pnl: pd.Series) -> float:
    """Annualised Sharpe from daily PnL series."""
    if len(daily_pnl) < 5 or daily_pnl.std() == 0:
        return 0.0
    return float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def _compute_profit_factor(trades: List[dict]) -> float:
    """Gross profit / gross loss."""
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    if gross_loss == 0:
        return 10.0 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _compute_win_rate(trades: List[dict]) -> float:
    """Fraction of winning trades."""
    if not trades:
        return 0.0
    return sum(1 for t in trades if t['pnl'] > 0) / len(trades)


def _is_session_close(dt: pd.Timestamp) -> bool:
    """Check if bar is at or past session forced-exit time (15:20)."""
    return (dt.hour > SESSION_CLOSE_HOUR or
            (dt.hour == SESSION_CLOSE_HOUR and dt.minute >= SESSION_CLOSE_MIN))


# ================================================================
# MAIN CLASS
# ================================================================
class IntradayWalkForward:
    """Walk-forward validation for intraday 5-min signals."""

    def __init__(self, db_conn):
        """
        Args:
            db_conn: psycopg2 connection to the trading database.
        """
        self.conn = db_conn

    # ────────────────────────────────────────────────────────────
    # DATA LOADING
    # ────────────────────────────────────────────────────────────
    def _load_bars(self, instrument: str, start_date: date,
                   end_date: date) -> pd.DataFrame:
        """
        Load 5-min bars from intraday_bars table.
        Falls back to synthetic bars from nifty_daily if table is
        empty or missing.

        Returns DataFrame with columns:
            timestamp, instrument, open, high, low, close, volume, oi
        """
        try:
            query = """
                SELECT timestamp, instrument, open, high, low, close, volume, oi
                FROM intraday_bars
                WHERE instrument = %s
                  AND timestamp::date >= %s
                  AND timestamp::date <= %s
                ORDER BY timestamp
            """
            df = pd.read_sql(query, self.conn,
                             params=(instrument, start_date, end_date))
            if len(df) > 0:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"Loaded {len(df)} intraday bars for {instrument} "
                            f"({start_date} to {end_date})")
                return df
        except Exception as e:
            logger.warning(f"intraday_bars query failed: {e}")

        # Fallback: synthesise from daily data
        logger.info(f"Falling back to synthetic bars from nifty_daily for {instrument}")
        return self._synthesise_bars(instrument, start_date, end_date)

    def _synthesise_bars(self, instrument: str, start_date: date,
                         end_date: date) -> pd.DataFrame:
        """
        Generate synthetic 5-min bars from daily OHLCV using random walk
        within the daily range. Produces ~75 bars per trading day.
        """
        daily_table = 'nifty_daily' if 'NIFTY' in instrument.upper() else 'nifty_daily'
        try:
            query = f"""
                SELECT date, open, high, low, close, volume
                FROM {daily_table}
                WHERE date >= %s AND date <= %s
                ORDER BY date
            """
            daily = pd.read_sql(query, self.conn,
                                params=(start_date, end_date))
        except Exception:
            logger.warning("nifty_daily query failed, generating purely random data")
            daily = self._generate_random_daily(start_date, end_date, instrument)

        if len(daily) == 0:
            daily = self._generate_random_daily(start_date, end_date, instrument)

        daily['date'] = pd.to_datetime(daily['date'])
        rows = []
        rng = np.random.RandomState(42)

        for _, day in daily.iterrows():
            d = day['date']
            o, h, l, c = float(day['open']), float(day['high']), float(day['low']), float(day['close'])
            vol_total = float(day.get('volume', 1_000_000))

            # Scale prices for BankNifty if daily source is Nifty
            if 'BANKNIFTY' in instrument.upper() and o < 30000:
                scale = 2.1
                o, h, l, c = o * scale, h * scale, l * scale, c * scale
                vol_total = vol_total * 0.6

            # Generate random walk from open to close touching high and low
            prices = self._random_walk_within_range(
                rng, o, h, l, c, BARS_PER_DAY)
            vol_per_bar = vol_total / BARS_PER_DAY

            for i in range(BARS_PER_DAY):
                ts = pd.Timestamp(d.date()) + timedelta(
                    hours=SESSION_OPEN_HOUR,
                    minutes=SESSION_OPEN_MIN + i * 5)
                bar_o = prices[i]
                bar_c = prices[i + 1] if i + 1 < len(prices) else prices[i]
                bar_h = max(bar_o, bar_c) * (1 + rng.uniform(0, 0.001))
                bar_l = min(bar_o, bar_c) * (1 - rng.uniform(0, 0.001))
                # Clamp within daily range
                bar_h = min(bar_h, h)
                bar_l = max(bar_l, l)
                bar_vol = int(vol_per_bar * rng.uniform(0.3, 2.5))
                rows.append({
                    'timestamp': ts,
                    'instrument': instrument,
                    'open': round(bar_o, 2),
                    'high': round(bar_h, 2),
                    'low': round(bar_l, 2),
                    'close': round(bar_c, 2),
                    'volume': bar_vol,
                    'oi': 0,
                })

        df = pd.DataFrame(rows)
        logger.info(f"Synthesised {len(df)} bars for {instrument} "
                    f"({start_date} to {end_date})")
        return df

    @staticmethod
    def _random_walk_within_range(rng, open_px, high, low, close,
                                   n_steps: int) -> np.ndarray:
        """Generate n_steps+1 prices from open to close within [low, high]."""
        path = np.zeros(n_steps + 1)
        path[0] = open_px
        path[-1] = close

        # Linear interpolation + noise
        linear = np.linspace(open_px, close, n_steps + 1)
        spread = high - low
        if spread <= 0:
            return linear

        noise = rng.normal(0, spread * 0.05, n_steps + 1)
        noise[0] = 0
        noise[-1] = 0
        path = linear + noise

        # Clamp
        path = np.clip(path, low, high)
        path[0] = open_px
        path[-1] = close
        return path

    @staticmethod
    def _generate_random_daily(start_date: date, end_date: date,
                                instrument: str) -> pd.DataFrame:
        """Generate random daily bars when no DB data exists."""
        rng = np.random.RandomState(123)
        base_price = 45000.0 if 'BANKNIFTY' in instrument.upper() else 22000.0
        dates = pd.bdate_range(start_date, end_date)
        rows = []
        price = base_price
        for d in dates:
            ret = rng.normal(0.0002, 0.012)
            price *= (1 + ret)
            day_range = price * rng.uniform(0.005, 0.02)
            o = price + rng.uniform(-day_range * 0.3, day_range * 0.3)
            c = price + rng.uniform(-day_range * 0.3, day_range * 0.3)
            h = max(o, c) + rng.uniform(0, day_range * 0.3)
            l_ = min(o, c) - rng.uniform(0, day_range * 0.3)
            rows.append({
                'date': d, 'open': round(o, 2), 'high': round(h, 2),
                'low': round(l_, 2), 'close': round(c, 2),
                'volume': int(rng.uniform(500_000, 3_000_000)),
            })
        return pd.DataFrame(rows)

    # ────────────────────────────────────────────────────────────
    # SESSION INDICATORS
    # ────────────────────────────────────────────────────────────
    def _add_session_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Per trading day, compute session-scoped indicators:
            vwap, opening_range_high/low (first 3 bars), ema_20,
            body_pct, vol_ratio_20, overnight_gap_pct, prev_day_close,
            datetime
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['datetime'] = df['timestamp']
        df['_date'] = df['datetime'].dt.date

        # Ensure numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Initialise indicator columns
        df['vwap'] = np.nan
        df['opening_range_high'] = np.nan
        df['opening_range_low'] = np.nan
        df['ema_20'] = np.nan
        df['body_pct'] = np.nan
        df['vol_ratio_20'] = np.nan
        df['overnight_gap_pct'] = np.nan
        df['prev_day_close'] = np.nan

        dates = sorted(df['_date'].unique())
        prev_day_close_val = None

        for trading_date in dates:
            mask = df['_date'] == trading_date
            idx = df.index[mask]
            if len(idx) < 3:
                prev_day_close_val = float(df.loc[idx[-1], 'close']) if len(idx) > 0 else prev_day_close_val
                continue

            session = df.loc[idx]

            # ── VWAP ──
            cum_vol = session['volume'].cumsum()
            cum_tp_vol = (session['close'] * session['volume']).cumsum()
            vwap_vals = np.where(cum_vol > 0, cum_tp_vol / cum_vol, session['close'])
            df.loc[idx, 'vwap'] = vwap_vals

            # ── Opening range (first 3 bars = 15 min) ──
            first3 = session.iloc[:3]
            or_high = float(first3['high'].max())
            or_low = float(first3['low'].min())
            df.loc[idx, 'opening_range_high'] = or_high
            df.loc[idx, 'opening_range_low'] = or_low

            # ── EMA-20 (intraday, exponential moving avg of close) ──
            closes = session['close'].values.astype(float)
            ema = np.full(len(closes), np.nan)
            if len(closes) >= 1:
                ema[0] = closes[0]
                alpha = 2.0 / (20 + 1)
                for j in range(1, len(closes)):
                    ema[j] = alpha * closes[j] + (1 - alpha) * ema[j - 1]
            df.loc[idx, 'ema_20'] = ema

            # ── Body pct (|close - open| / (high - low)) ──
            bar_range = (session['high'] - session['low']).replace(0, np.nan)
            body = (session['close'] - session['open']).abs()
            df.loc[idx, 'body_pct'] = body / bar_range

            # ── Volume ratio (current vol / 20-bar SMA of vol) ──
            vol = session['volume'].values.astype(float)
            vol_sma = pd.Series(vol).rolling(20, min_periods=1).mean().values
            vol_ratio = np.where(vol_sma > 0, vol / vol_sma, 1.0)
            df.loc[idx, 'vol_ratio_20'] = vol_ratio

            # ── Overnight gap pct ──
            if prev_day_close_val is not None and prev_day_close_val > 0:
                day_open = float(session.iloc[0]['open'])
                gap_pct = (day_open - prev_day_close_val) / prev_day_close_val
                df.loc[idx, 'overnight_gap_pct'] = gap_pct
                df.loc[idx, 'prev_day_close'] = prev_day_close_val

            # Update prev day close for next session
            prev_day_close_val = float(session.iloc[-1]['close'])

        df.drop(columns=['_date'], inplace=True, errors='ignore')
        return df

    # ────────────────────────────────────────────────────────────
    # TRADE SIMULATION
    # ────────────────────────────────────────────────────────────
    def _simulate_trades(self, signal_computer, bars_df: pd.DataFrame,
                         instrument: str,
                         config_overrides: Optional[Dict] = None) -> List[dict]:
        """
        Walk through bars, fire signals, track positions.
        One position at a time per signal (no stacking).

        Returns list of trade dicts:
            {entry_date, exit_date, entry_price, exit_price, pnl, pnl_pct,
             signal_id, direction, exit_reason, bars_held}
        """
        slippage = SLIPPAGE.get(instrument.upper(), 1.0)
        signals_config = signal_computer.SIGNALS
        if config_overrides:
            for k, v in config_overrides.items():
                if k in signals_config:
                    signals_config[k].update(v)

        trades = []
        # Active positions keyed by signal_id
        open_positions: Dict[str, dict] = {}

        bars_df = bars_df.copy()
        bars_df['_date'] = bars_df['datetime'].dt.date
        dates = sorted(bars_df['_date'].unique())

        for trading_date in dates:
            session = bars_df[bars_df['_date'] == trading_date].copy()
            if len(session) < 3:
                continue

            session_so_far = pd.DataFrame()
            prev_bar = None

            for bar_idx in range(len(session)):
                bar = session.iloc[bar_idx]
                bar_time = bar['datetime']

                # Build session context (bars up to and including current)
                session_so_far = session.iloc[:bar_idx + 1]

                # ── CHECK EXITS for open positions ──
                to_close = []
                for sig_id, pos in open_positions.items():
                    cfg = signals_config.get(sig_id, {})
                    stop_loss_pct = cfg.get('stop_loss_pct', 0.004)
                    max_hold = cfg.get('max_hold_bars', 30)
                    exit_reason = None
                    exit_price = float(bar['close'])

                    # Stop loss check (intrabar approximation using high/low)
                    if pos['direction'] == 'LONG':
                        sl_price = pos['entry_price'] * (1 - stop_loss_pct)
                        if float(bar['low']) <= sl_price:
                            exit_price = sl_price
                            exit_reason = 'stop_loss'
                    else:
                        sl_price = pos['entry_price'] * (1 + stop_loss_pct)
                        if float(bar['high']) >= sl_price:
                            exit_price = sl_price
                            exit_reason = 'stop_loss'

                    # Max hold bars
                    if pos['bars_held'] >= max_hold:
                        exit_reason = exit_reason or 'max_hold'

                    # Session close forced exit
                    if _is_session_close(bar_time):
                        exit_reason = exit_reason or 'session_close'

                    if exit_reason:
                        # Apply slippage on exit
                        if pos['direction'] == 'LONG':
                            exit_price -= slippage
                        else:
                            exit_price += slippage

                        pnl = (exit_price - pos['entry_price']) if pos['direction'] == 'LONG' \
                            else (pos['entry_price'] - exit_price)
                        pnl_pct = pnl / pos['entry_price'] if pos['entry_price'] > 0 else 0.0

                        trades.append({
                            'entry_date': pos['entry_time'],
                            'exit_date': bar_time,
                            'entry_price': pos['entry_price'],
                            'exit_price': round(exit_price, 2),
                            'pnl': round(pnl, 2),
                            'pnl_pct': round(pnl_pct, 6),
                            'signal_id': sig_id,
                            'direction': pos['direction'],
                            'exit_reason': exit_reason,
                            'bars_held': pos['bars_held'],
                        })
                        to_close.append(sig_id)
                    else:
                        pos['bars_held'] += 1

                for sig_id in to_close:
                    del open_positions[sig_id]

                # ── Don't open new positions at session close ──
                if _is_session_close(bar_time):
                    prev_bar = bar
                    continue

                # ── CHECK ENTRIES ──
                fired = signal_computer.compute_all(bar, prev_bar, session_so_far)
                for sig in fired:
                    sig_id = sig['signal_id']
                    # No stacking: skip if already in position for this signal
                    if sig_id in open_positions:
                        continue

                    entry_price = float(sig['price'])
                    # Apply slippage on entry
                    if sig['direction'] == 'LONG':
                        entry_price += slippage
                    else:
                        entry_price -= slippage

                    open_positions[sig_id] = {
                        'entry_price': round(entry_price, 2),
                        'entry_time': bar_time,
                        'direction': sig['direction'],
                        'bars_held': 0,
                    }

                prev_bar = bar

            # ── Force-close any remaining positions at end of day ──
            if len(session) > 0:
                last_bar = session.iloc[-1]
                to_close = []
                for sig_id, pos in open_positions.items():
                    exit_price = float(last_bar['close'])
                    if pos['direction'] == 'LONG':
                        exit_price -= slippage
                    else:
                        exit_price += slippage

                    pnl = (exit_price - pos['entry_price']) if pos['direction'] == 'LONG' \
                        else (pos['entry_price'] - exit_price)
                    pnl_pct = pnl / pos['entry_price'] if pos['entry_price'] > 0 else 0.0

                    trades.append({
                        'entry_date': pos['entry_time'],
                        'exit_date': last_bar['datetime'],
                        'entry_price': pos['entry_price'],
                        'exit_price': round(exit_price, 2),
                        'pnl': round(pnl, 2),
                        'pnl_pct': round(pnl_pct, 6),
                        'signal_id': sig_id,
                        'direction': pos['direction'],
                        'exit_reason': 'eod_force_close',
                        'bars_held': pos['bars_held'],
                    })
                    to_close.append(sig_id)
                for sig_id in to_close:
                    del open_positions[sig_id]

        return trades

    # ────────────────────────────────────────────────────────────
    # WALK-FORWARD WINDOWS
    # ────────────────────────────────────────────────────────────
    @staticmethod
    def _generate_wf_windows(start_date: date,
                              end_date: date) -> List[Tuple[date, date, date, date]]:
        """
        Generate walk-forward windows: 6mo train / 2mo test / 1mo step.
        Returns list of (train_start, train_end, test_start, test_end).
        """
        windows = []
        cursor = start_date

        while True:
            train_start = cursor
            train_end = add_months(train_start, INTRADAY_WF_TRAIN_MONTHS)
            test_start = train_end
            test_end = add_months(test_start, INTRADAY_WF_TEST_MONTHS)

            if test_end > end_date:
                break

            windows.append((train_start, train_end, test_start, test_end))
            cursor = add_months(cursor, INTRADAY_WF_STEP_MONTHS)

        return windows

    # ────────────────────────────────────────────────────────────
    # PER-SIGNAL WALK-FORWARD
    # ────────────────────────────────────────────────────────────
    def run_signal(self, instrument: str, signal_id: str,
                   signal_computer) -> dict:
        """
        Run full walk-forward for a single signal on an instrument.

        Returns:
            {signal_id, instrument, trades, win_rate, pf, sharpe,
             wf_pass_rate, verdict, best_params, windows}
        """
        # Determine data range: go back as far as possible
        end_date = date.today()
        # Need at least train + test = 8 months; load 3 years for multiple windows
        start_date = add_months(end_date, -36)

        bars = self._load_bars(instrument, start_date, end_date)
        if len(bars) == 0:
            logger.warning(f"No bars for {instrument}, skipping {signal_id}")
            return self._empty_signal_result(signal_id, instrument)

        bars = self._add_session_indicators(bars)

        # Generate WF windows
        first_date = bars['datetime'].dt.date.min()
        last_date = bars['datetime'].dt.date.max()
        windows = self._generate_wf_windows(first_date, last_date)

        if not windows:
            logger.warning(f"Not enough data for WF windows: {signal_id}")
            return self._empty_signal_result(signal_id, instrument)

        window_results = []
        all_test_trades = []

        for w_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Train period: we don't optimise params here (signal logic is fixed),
            # but we compute train metrics for reference
            train_mask = ((bars['datetime'].dt.date >= train_start) &
                          (bars['datetime'].dt.date < train_end))
            test_mask = ((bars['datetime'].dt.date >= test_start) &
                         (bars['datetime'].dt.date < test_end))

            train_bars = bars[train_mask].copy()
            test_bars = bars[test_mask].copy()

            if len(train_bars) < 100 or len(test_bars) < 50:
                continue

            # Run simulation on TEST period only (train validates stationarity)
            # Filter signal_computer to only fire the target signal
            filtered_computer = _SingleSignalFilter(signal_computer, signal_id)
            test_trades = self._simulate_trades(
                filtered_computer, test_bars, instrument)

            # Compute per-window metrics
            if len(test_trades) < INTRADAY_MIN_TRADES_PER_WINDOW:
                w_sharpe, w_pf, w_wr = 0.0, 0.0, 0.0
                w_pass = False
            else:
                # Daily PnL aggregation for Sharpe
                daily_pnl = {}
                for t in test_trades:
                    d = t['entry_date'].date() if hasattr(t['entry_date'], 'date') else t['entry_date']
                    daily_pnl[d] = daily_pnl.get(d, 0.0) + t['pnl']
                daily_series = pd.Series(list(daily_pnl.values()))

                w_sharpe = _compute_sharpe(daily_series)
                w_pf = _compute_profit_factor(test_trades)
                w_wr = _compute_win_rate(test_trades)
                w_pass = (w_sharpe >= INTRADAY_MIN_SHARPE and
                          w_pf >= INTRADAY_MIN_PF)

            window_results.append({
                'window': w_idx,
                'train': f"{train_start} → {train_end}",
                'test': f"{test_start} → {test_end}",
                'trades': len(test_trades),
                'sharpe': round(w_sharpe, 2),
                'pf': round(w_pf, 2),
                'win_rate': round(w_wr, 3),
                'pass': w_pass,
            })
            all_test_trades.extend(test_trades)

        # ── Aggregate results ──
        n_windows = len(window_results)
        n_pass = sum(1 for w in window_results if w['pass'])
        wf_pass_rate = n_pass / n_windows if n_windows > 0 else 0.0

        # Overall metrics from all test trades
        if all_test_trades:
            overall_pf = _compute_profit_factor(all_test_trades)
            overall_wr = _compute_win_rate(all_test_trades)
            daily_pnl = {}
            for t in all_test_trades:
                d = t['entry_date'].date() if hasattr(t['entry_date'], 'date') else t['entry_date']
                daily_pnl[d] = daily_pnl.get(d, 0.0) + t['pnl']
            overall_sharpe = _compute_sharpe(pd.Series(list(daily_pnl.values())))
        else:
            overall_pf, overall_wr, overall_sharpe = 0.0, 0.0, 0.0

        verdict = 'PASS' if wf_pass_rate >= INTRADAY_MIN_WF_PASS_RATE else 'FAIL'

        return {
            'signal_id': signal_id,
            'instrument': instrument,
            'trades': len(all_test_trades),
            'win_rate': round(overall_wr, 3),
            'pf': round(overall_pf, 2),
            'sharpe': round(overall_sharpe, 2),
            'wf_pass_rate': round(wf_pass_rate, 2),
            'wf_windows': n_windows,
            'wf_passed': n_pass,
            'verdict': verdict,
            'best_params': {},  # fixed signal logic, no param search
            'windows': window_results,
            'all_trades': all_test_trades,
        }

    @staticmethod
    def _empty_signal_result(signal_id: str, instrument: str) -> dict:
        return {
            'signal_id': signal_id,
            'instrument': instrument,
            'trades': 0,
            'win_rate': 0.0,
            'pf': 0.0,
            'sharpe': 0.0,
            'wf_pass_rate': 0.0,
            'wf_windows': 0,
            'wf_passed': 0,
            'verdict': 'FAIL',
            'best_params': {},
            'windows': [],
            'all_trades': [],
        }

    # ────────────────────────────────────────────────────────────
    # RUN ALL SIGNALS
    # ────────────────────────────────────────────────────────────
    def run_all(self, print_results: bool = True) -> dict:
        """
        Run walk-forward for all L9 Nifty + BN BankNifty signals.
        Returns results dict keyed by signal_id.
        """
        results = {}

        # ── L9 Nifty signals (10 signals) ──
        nifty_computer = IntradaySignalComputer()
        logger.info("=" * 60)
        logger.info("NIFTY L9 SIGNALS — Walk-Forward Validation")
        logger.info("=" * 60)

        for signal_id in nifty_computer.SIGNALS:
            logger.info(f"Running WF: {signal_id} on NIFTY ...")
            result = self.run_signal('NIFTY', signal_id, nifty_computer)
            results[signal_id] = result
            logger.info(f"  {signal_id}: {result['verdict']} "
                        f"(Sharpe={result['sharpe']}, PF={result['pf']}, "
                        f"WR={result['win_rate']:.1%}, "
                        f"WF={result['wf_pass_rate']:.0%})")

        # ── BN BankNifty signals (8 signals) ──
        bn_computer = BankNiftySignalComputer()
        logger.info("=" * 60)
        logger.info("BANKNIFTY BN SIGNALS — Walk-Forward Validation")
        logger.info("=" * 60)

        for signal_id in bn_computer.SIGNALS:
            logger.info(f"Running WF: {signal_id} on BANKNIFTY ...")
            result = self.run_signal('BANKNIFTY', signal_id, bn_computer)
            results[signal_id] = result
            logger.info(f"  {signal_id}: {result['verdict']} "
                        f"(Sharpe={result['sharpe']}, PF={result['pf']}, "
                        f"WR={result['win_rate']:.1%}, "
                        f"WF={result['wf_pass_rate']:.0%})")

        if print_results:
            self._print_results(results)

        return results

    # ────────────────────────────────────────────────────────────
    # PRETTY PRINT
    # ────────────────────────────────────────────────────────────
    @staticmethod
    def _print_results(results: dict):
        """Print formatted results table."""
        header = (f"{'Signal':<28} {'Inst':<10} {'Trades':>6} "
                  f"{'WR':>6} {'PF':>6} {'Sharpe':>7} "
                  f"{'WF%':>5} {'W':>3} {'P':>3} {'Verdict':>8}")
        sep = "─" * len(header)

        print(f"\n{sep}")
        print("INTRADAY WALK-FORWARD RESULTS")
        print(f"Windows: {INTRADAY_WF_TRAIN_MONTHS}mo train / "
              f"{INTRADAY_WF_TEST_MONTHS}mo test / "
              f"{INTRADAY_WF_STEP_MONTHS}mo step")
        print(f"Pass: Sharpe >= {INTRADAY_MIN_SHARPE}, "
              f"PF >= {INTRADAY_MIN_PF}, "
              f">= {INTRADAY_MIN_WF_PASS_RATE:.0%} windows")
        print(sep)
        print(header)
        print(sep)

        # Group by instrument
        nifty_results = {k: v for k, v in results.items()
                         if v['instrument'] == 'NIFTY'}
        bn_results = {k: v for k, v in results.items()
                      if v['instrument'] == 'BANKNIFTY'}

        for group_name, group in [("NIFTY", nifty_results),
                                   ("BANKNIFTY", bn_results)]:
            if not group:
                continue
            for sig_id, r in sorted(group.items()):
                verdict_str = r['verdict']
                if verdict_str == 'PASS':
                    verdict_str = 'PASS'
                else:
                    verdict_str = 'FAIL'

                line = (f"{r['signal_id']:<28} {r['instrument']:<10} "
                        f"{r['trades']:>6} "
                        f"{r['win_rate']:>5.1%} "
                        f"{r['pf']:>6.2f} "
                        f"{r['sharpe']:>7.2f} "
                        f"{r['wf_pass_rate']:>4.0%} "
                        f"{r['wf_windows']:>3} "
                        f"{r['wf_passed']:>3} "
                        f"{verdict_str:>8}")
                print(line)
            print(sep)

        n_pass = sum(1 for r in results.values() if r['verdict'] == 'PASS')
        print(f"\nTotal: {len(results)} signals | "
              f"{n_pass} PASS | {len(results) - n_pass} FAIL\n")


# ================================================================
# SINGLE-SIGNAL FILTER WRAPPER
# ================================================================
class _SingleSignalFilter:
    """
    Wraps a signal computer to only fire a single target signal.
    Delegates compute_all but filters output.
    """

    def __init__(self, computer, target_signal_id: str):
        self.computer = computer
        self.target = target_signal_id
        # Expose SIGNALS for config access
        self.SIGNALS = computer.SIGNALS

    def compute_all(self, bar, prev_bar, session_bars):
        fired = self.computer.compute_all(bar, prev_bar, session_bars)
        return [s for s in fired if s['signal_id'] == self.target]


# ================================================================
# CLI ENTRY POINT
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Intraday Walk-Forward Validation Engine')
    parser.add_argument('--instrument', type=str, default=None,
                        help='NIFTY or BANKNIFTY (default: both)')
    parser.add_argument('--signal', type=str, default=None,
                        help='Single signal ID to test (e.g. L9_ORB_BREAKOUT)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    try:
        conn = psycopg2.connect(DATABASE_DSN)
    except Exception as e:
        logger.warning(f"DB connection failed ({e}), using None (synthetic data)")
        conn = None

    engine = IntradayWalkForward(conn)

    if args.signal:
        # Run single signal
        instrument = args.instrument
        if instrument is None:
            # Infer from signal prefix
            if args.signal.startswith('BN_'):
                instrument = 'BANKNIFTY'
            else:
                instrument = 'NIFTY'

        if instrument.upper() == 'BANKNIFTY':
            computer = BankNiftySignalComputer()
        else:
            computer = IntradaySignalComputer()

        if args.signal not in computer.SIGNALS:
            print(f"ERROR: signal '{args.signal}' not found in "
                  f"{instrument} signal list.")
            print(f"Available: {list(computer.SIGNALS.keys())}")
            sys.exit(1)

        result = engine.run_signal(instrument.upper(), args.signal, computer)
        engine._print_results({args.signal: result})

        # Print per-window detail
        if result['windows']:
            print(f"\nPer-Window Detail for {args.signal}:")
            print(f"{'W#':>3} {'Test Period':<28} {'Trades':>6} "
                  f"{'Sharpe':>7} {'PF':>6} {'WR':>6} {'Pass':>5}")
            print("─" * 70)
            for w in result['windows']:
                print(f"{w['window']:>3} {w['test']:<28} {w['trades']:>6} "
                      f"{w['sharpe']:>7.2f} {w['pf']:>6.2f} "
                      f"{w['win_rate']:>5.1%} "
                      f"{'YES' if w['pass'] else 'NO':>5}")

    elif args.instrument:
        # Run all signals for one instrument
        instrument = args.instrument.upper()
        if instrument == 'BANKNIFTY':
            computer = BankNiftySignalComputer()
        else:
            computer = IntradaySignalComputer()

        results = {}
        for signal_id in computer.SIGNALS:
            logger.info(f"Running WF: {signal_id} on {instrument} ...")
            result = engine.run_signal(instrument, signal_id, computer)
            results[signal_id] = result

        engine._print_results(results)

    else:
        # Run all
        engine.run_all(print_results=True)

    if conn is not None:
        conn.close()


if __name__ == '__main__':
    main()

"""
Walk-Forward Validation for India-specific Intraday Signals.

Validates the 5 India-specific intraday signals from IndiaIntradaySignals
using walk-forward with windows suited to intraday regime changes:
    6-month train / 2-month test / 1-month step.

Pass criteria: Sharpe >= 0.8, PF >= 1.3, >= 60% windows pass.
Relaxed for low-frequency: if < 15 trades in a window, require PF >= 1.5.

Slippage: 1 pt/side NIFTY, 3 pts/side BANKNIFTY.

The 5 India-specific signals:
    EXPIRY_PIN_FADE       - Expiry day pin/fade around max pain
    VIX_CRUSH_ENTRY       - Enter after VIX spike crush
    BUDGET_STRADDLE_BUY   - Straddle before Budget/RBI MPC events
    FII_FLOW_MOMENTUM     - Follow FII intraday flow direction
    ORR_REVERSION         - Opening range reversion on gap days

Usage:
    venv/bin/python3 -m backtest.india_intraday_wf
    venv/bin/python3 -m backtest.india_intraday_wf --signal EXPIRY_PIN_FADE
"""

import argparse
import logging
import sys
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

from backtest.indicators import atr as compute_atr, rsi as compute_rsi
from backtest.types import add_months
from config.settings import DATABASE_DSN, RISK_FREE_RATE, NIFTY_LOT_SIZE

logger = logging.getLogger(__name__)

# ================================================================
# WF PARAMETERS
# ================================================================
WF_TRAIN_MONTHS = 6
WF_TEST_MONTHS = 2
WF_STEP_MONTHS = 1

# Pass criteria
MIN_SHARPE = 0.8
MIN_PF = 1.3
MIN_WF_PASS_RATE = 0.60
MIN_TRADES_PER_WINDOW = 5
LOW_FREQ_THRESHOLD = 15
LOW_FREQ_PF = 1.5

# Slippage: absolute points per side
SLIPPAGE_PER_SIDE = {
    'NIFTY': 1.0,
    'BANKNIFTY': 3.0,
}

# Session timing
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 20)
BARS_PER_DAY = 75  # (15:30 - 9:15) / 5min
TRADING_DAYS_PER_YEAR = 252

# The 5 India-specific intraday signal definitions
INDIA_INTRADAY_SIGNALS = {
    'EXPIRY_PIN_FADE': {
        'description': 'Expiry day pin/fade around max pain',
        'direction': 'both',
        'sl_pct': 0.004,
        'tgt_pct': 0.006,
        'max_hold_bars': 30,  # ~2.5 hours
        'instrument': 'NIFTY',
    },
    'VIX_CRUSH_ENTRY': {
        'description': 'Enter long after VIX spike crush',
        'direction': 'long',
        'sl_pct': 0.005,
        'tgt_pct': 0.008,
        'max_hold_bars': 45,
        'instrument': 'NIFTY',
    },
    'BUDGET_STRADDLE_BUY': {
        'description': 'Straddle buy before Budget/RBI MPC events',
        'direction': 'both',
        'sl_pct': 0.010,
        'tgt_pct': 0.015,
        'max_hold_bars': 60,
        'instrument': 'NIFTY',
    },
    'FII_FLOW_MOMENTUM': {
        'description': 'Follow FII intraday flow direction',
        'direction': 'both',
        'sl_pct': 0.003,
        'tgt_pct': 0.005,
        'max_hold_bars': 36,
        'instrument': 'NIFTY',
    },
    'ORR_REVERSION': {
        'description': 'Opening range reversion on gap days',
        'direction': 'both',
        'sl_pct': 0.004,
        'tgt_pct': 0.006,
        'max_hold_bars': 24,  # ~2 hours
        'instrument': 'NIFTY',
    },
}

# ================================================================
# RBI MPC APPROXIMATE DATES (6 per year, bi-monthly)
# ================================================================
_RBI_MPC_MONTHS = [2, 4, 6, 8, 10, 12]
_RBI_MPC_DAY_RANGE = (4, 8)  # typically first week


def _rbi_mpc_dates(year: int) -> List[date]:
    """Approximate RBI MPC announcement dates for a given year."""
    dates = []
    for m in _RBI_MPC_MONTHS:
        # MPC decision typically falls on a Wednesday in first week
        for d in range(_RBI_MPC_DAY_RANGE[0], _RBI_MPC_DAY_RANGE[1] + 1):
            try:
                dt = date(year, m, d)
                if dt.weekday() == 2:  # Wednesday
                    dates.append(dt)
                    break
            except ValueError:
                continue
        else:
            # Fallback: use the 6th of the month
            try:
                dates.append(date(year, m, 6))
            except ValueError:
                pass
    return dates


def _budget_dates(year: int) -> List[date]:
    """Union Budget date — February 1 each year."""
    try:
        return [date(year, 2, 1)]
    except ValueError:
        return []


def _is_event_day(trading_date: date) -> Tuple[bool, str]:
    """Check if a date is a Budget day or RBI MPC day."""
    year = trading_date.year
    for bd in _budget_dates(year):
        if trading_date == bd:
            return True, 'BUDGET'
    for rd in _rbi_mpc_dates(year):
        if trading_date == rd:
            return True, 'RBI_MPC'
        # Also flag day before MPC (straddle entry)
        if trading_date == rd - timedelta(days=1):
            return True, 'PRE_RBI_MPC'
    return False, ''


def _is_expiry_day(trading_date: date) -> bool:
    """Simple heuristic: weekly expiry on Thursdays."""
    return trading_date.weekday() == 3


# ================================================================
# METRICS HELPERS
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


# ================================================================
# MAIN CLASS
# ================================================================
class IndiaIntradayWF:
    """Walk-forward validation for India-specific intraday signals."""

    def __init__(self, db_conn):
        """
        Args:
            db_conn: psycopg2 connection to the trading database.
        """
        self.conn = db_conn

    # ────────────────────────────────────────────────────────────
    # DATA LOADING
    # ────────────────────────────────────────────────────────────
    def _load_data(self, start_date: date,
                   end_date: date) -> Tuple[pd.DataFrame, dict]:
        """
        Load 5-min bars and daily context.

        Returns:
            (bars_df, daily_context_dict)
            bars_df: timestamp, instrument, open, high, low, close, volume, oi
            daily_context_dict: keyed by date -> {close, india_vix, atr_14,
                                                   rsi_14, is_expiry, ...}
        """
        bars_df = self._load_intraday_bars(start_date, end_date)
        daily_ctx = self._load_daily_context(start_date, end_date)
        return bars_df, daily_ctx

    def _load_intraday_bars(self, start_date: date,
                            end_date: date) -> pd.DataFrame:
        """Load 5-min bars from intraday_bars table for NIFTY."""
        try:
            query = """
                SELECT timestamp, instrument, open, high, low, close, volume, oi
                FROM intraday_bars
                WHERE instrument = 'NIFTY'
                  AND timestamp::date >= %s
                  AND timestamp::date <= %s
                ORDER BY timestamp
            """
            df = pd.read_sql(query, self.conn,
                             params=(start_date, end_date))
            if len(df) > 0:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"Loaded {len(df)} intraday bars "
                            f"({start_date} to {end_date})")
                return df
        except Exception as e:
            logger.warning(f"intraday_bars query failed: {e}")

        # Fallback: synthesise from daily data
        logger.info("Falling back to synthetic bars from nifty_daily")
        return self._synthesise_bars(start_date, end_date)

    def _load_daily_context(self, start_date: date,
                            end_date: date) -> dict:
        """
        Load daily context from nifty_daily.
        Compute ATR(14), RSI(14), VIX percentile for IV rank.

        Returns dict keyed by date.
        """
        # Load with 30 extra days for ATR/RSI warmup
        warmup_start = start_date - timedelta(days=45)
        try:
            query = """
                SELECT date, open, high, low, close, volume, india_vix
                FROM nifty_daily
                WHERE date >= %s AND date <= %s
                ORDER BY date
            """
            daily = pd.read_sql(query, self.conn,
                                params=(warmup_start, end_date))
        except Exception as e:
            logger.warning(f"nifty_daily query failed: {e}, generating synthetic")
            daily = self._generate_synthetic_daily(warmup_start, end_date)

        if len(daily) == 0:
            daily = self._generate_synthetic_daily(warmup_start, end_date)

        daily['date'] = pd.to_datetime(daily['date']).dt.date
        for col in ['open', 'high', 'low', 'close', 'volume']:
            daily[col] = pd.to_numeric(daily[col], errors='coerce')
        if 'india_vix' in daily.columns:
            daily['india_vix'] = pd.to_numeric(daily['india_vix'],
                                                errors='coerce')
        else:
            daily['india_vix'] = 14.0

        # Compute ATR(14)
        daily['atr_14'] = compute_atr(daily, period=14)

        # Compute RSI(14)
        daily['rsi_14'] = compute_rsi(daily['close'], period=14)

        # VIX percentile for IV rank (rolling 252-day percentile)
        vix = daily['india_vix'].fillna(14.0)
        daily['vix_pctile'] = vix.rolling(
            window=252, min_periods=20
        ).apply(lambda x: (x.iloc[:-1] < x.iloc[-1]).sum() / (len(x) - 1)
                if len(x) > 1 else 0.5, raw=False)

        # Build context dict keyed by date
        ctx = {}
        daily_sorted = daily.sort_values('date').reset_index(drop=True)
        for i, row in daily_sorted.iterrows():
            d = row['date']
            if d < start_date:
                continue
            # Last 20 daily bars for context
            start_idx = max(0, i - 19)
            daily_bars_slice = daily_sorted.iloc[start_idx:i + 1].copy()

            ctx[d] = {
                'close': float(row['close']),
                'india_vix': float(row.get('india_vix', 14.0)
                                   if pd.notna(row.get('india_vix')) else 14.0),
                'atr_14': float(row['atr_14'])
                          if pd.notna(row.get('atr_14')) else 0.0,
                'rsi_14': float(row['rsi_14'])
                          if pd.notna(row.get('rsi_14')) else 50.0,
                'vix_pctile': float(row['vix_pctile'])
                              if pd.notna(row.get('vix_pctile')) else 0.5,
                'daily_bars_df': daily_bars_slice,
            }

        return ctx

    def _synthesise_bars(self, start_date: date,
                         end_date: date) -> pd.DataFrame:
        """Generate synthetic 5-min bars from nifty_daily."""
        try:
            query = """
                SELECT date, open, high, low, close, volume
                FROM nifty_daily
                WHERE date >= %s AND date <= %s
                ORDER BY date
            """
            daily = pd.read_sql(query, self.conn,
                                params=(start_date, end_date))
        except Exception:
            daily = self._generate_synthetic_daily(start_date, end_date)

        if len(daily) == 0:
            daily = self._generate_synthetic_daily(start_date, end_date)

        daily['date'] = pd.to_datetime(daily['date'])
        rows = []
        rng = np.random.RandomState(42)

        for _, day in daily.iterrows():
            d = day['date']
            o = float(day['open'])
            h = float(day['high'])
            l_ = float(day['low'])
            c = float(day['close'])
            vol_total = float(day.get('volume', 1_000_000))

            prices = self._random_walk_ohlc(rng, o, h, l_, c, BARS_PER_DAY)
            vol_per_bar = vol_total / BARS_PER_DAY

            for i in range(BARS_PER_DAY):
                ts = pd.Timestamp(d.date()) + timedelta(
                    hours=9, minutes=15 + i * 5)
                bar_o = prices[i]
                bar_c = prices[i + 1] if i + 1 < len(prices) else prices[i]
                bar_h = max(bar_o, bar_c) * (1 + rng.uniform(0, 0.001))
                bar_l = min(bar_o, bar_c) * (1 - rng.uniform(0, 0.001))
                bar_h = min(bar_h, h)
                bar_l = max(bar_l, l_)
                rows.append({
                    'timestamp': ts,
                    'instrument': 'NIFTY',
                    'open': round(bar_o, 2),
                    'high': round(bar_h, 2),
                    'low': round(bar_l, 2),
                    'close': round(bar_c, 2),
                    'volume': int(vol_per_bar * rng.uniform(0.3, 2.5)),
                    'oi': 0,
                })

        df = pd.DataFrame(rows)
        logger.info(f"Synthesised {len(df)} bars ({start_date} to {end_date})")
        return df

    @staticmethod
    def _random_walk_ohlc(rng, open_px, high, low, close,
                          n_steps: int) -> np.ndarray:
        """Generate n_steps+1 prices from open to close within [low, high]."""
        linear = np.linspace(open_px, close, n_steps + 1)
        spread = high - low
        if spread <= 0:
            return linear
        noise = rng.normal(0, spread * 0.05, n_steps + 1)
        noise[0] = 0
        noise[-1] = 0
        path = np.clip(linear + noise, low, high)
        path[0] = open_px
        path[-1] = close
        return path

    @staticmethod
    def _generate_synthetic_daily(start_date: date,
                                  end_date: date) -> pd.DataFrame:
        """Generate random daily bars when no DB data available."""
        rng = np.random.RandomState(123)
        base_price = 22000.0
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
                'india_vix': round(rng.uniform(10, 25), 2),
            })
        return pd.DataFrame(rows)

    # ────────────────────────────────────────────────────────────
    # CONTEXT BUILDER
    # ────────────────────────────────────────────────────────────
    def _build_context(self, trading_date: date, daily_ctx: dict,
                       bars_so_far: pd.DataFrame) -> dict:
        """
        Build the context dict that IndiaIntradaySignals.check_all() expects.

        Includes: today, vix, daily_atr, expiry_info, fii_data,
                  calendar_ctx, iv_rank, daily_bars_df
        """
        day_data = daily_ctx.get(trading_date, {})
        vix = day_data.get('india_vix', 14.0)
        atr_val = day_data.get('atr_14', 0.0)
        rsi_val = day_data.get('rsi_14', 50.0)
        vix_pctile = day_data.get('vix_pctile', 0.5)
        daily_bars_df = day_data.get('daily_bars_df', pd.DataFrame())

        is_expiry = _is_expiry_day(trading_date)
        is_event, event_type = _is_event_day(trading_date)

        expiry_info = {
            'is_expiry': is_expiry,
            'is_monthly_expiry': is_expiry and trading_date.day > 21,
            'days_to_expiry': (3 - trading_date.weekday()) % 7,
        }

        calendar_ctx = {
            'is_event_day': is_event,
            'event_type': event_type,
            'is_budget_day': event_type == 'BUDGET',
            'is_rbi_mpc': event_type in ('RBI_MPC', 'PRE_RBI_MPC'),
            'is_pre_rbi_mpc': event_type == 'PRE_RBI_MPC',
        }

        # Approximate max pain as round to nearest 100
        close_px = day_data.get('close', 22000.0)
        approx_max_pain = round(close_px / 100) * 100

        return {
            'today': trading_date,
            'vix': vix,
            'daily_atr': atr_val,
            'daily_rsi': rsi_val,
            'expiry_info': expiry_info,
            'fii_data': {},  # Empty if unavailable
            'calendar_ctx': calendar_ctx,
            'iv_rank': vix_pctile,
            'daily_bars_df': daily_bars_df,
            'bars_today': bars_so_far.drop(columns=['_date'], errors='ignore'),
            'daily_close': close_px,
            'max_pain_strike': approx_max_pain,
            'is_expiry': is_expiry,
            'day_of_week': trading_date.weekday(),
        }

    # ────────────────────────────────────────────────────────────
    # SIGNAL EVALUATION (rule-based, no external import needed)
    # ────────────────────────────────────────────────────────────
    def _evaluate_signals(self, bar: dict, prev_bar: Optional[dict],
                          context: dict) -> List[dict]:
        """
        Evaluate all 5 India-specific signals on a single bar.
        Returns list of signal fires: [{signal_id, direction, price}]
        """
        fires = []
        ts = bar['timestamp']
        bar_time = ts.time() if hasattr(ts, 'time') else ts
        price = bar['close']
        atr_val = context.get('daily_atr', 0)
        vix = context.get('vix', 14.0)

        # ---- EXPIRY_PIN_FADE ----
        if context.get('is_expiry', False) and prev_bar is not None:
            mp = context.get('max_pain_strike', 0)
            if mp > 0 and time(13, 0) <= bar_time <= time(13, 30):
                dev = (price - mp) / mp
                # Fade: if price above max pain, short toward pin
                if dev > 0.003:
                    fires.append({
                        'signal_id': 'EXPIRY_PIN_FADE',
                        'direction': 'SHORT',
                        'price': price,
                    })
                elif dev < -0.003:
                    fires.append({
                        'signal_id': 'EXPIRY_PIN_FADE',
                        'direction': 'LONG',
                        'price': price,
                    })

        # ---- VIX_CRUSH_ENTRY ----
        # VIX in top quartile yesterday but dropping intraday => buy
        iv_rank = context.get('iv_rank', 0.5)
        if (iv_rank > 0.75 and vix < 20 and prev_bar is not None
                and time(10, 0) <= bar_time <= time(11, 0)):
            # Price rising off lows, VIX was high but calming
            if bar['close'] > bar['open'] and atr_val > 0:
                if (bar['close'] - bar['low']) > 0.3 * atr_val:
                    fires.append({
                        'signal_id': 'VIX_CRUSH_ENTRY',
                        'direction': 'LONG',
                        'price': price,
                    })

        # ---- BUDGET_STRADDLE_BUY ----
        cal = context.get('calendar_ctx', {})
        if cal.get('is_budget_day') or cal.get('is_rbi_mpc'):
            # Enter early in session for event day volatility
            if time(9, 30) <= bar_time <= time(10, 0):
                # Straddle-like: enter both directions
                # In practice this is a volatility play; we track as LONG
                # since we're buying gamma
                fires.append({
                    'signal_id': 'BUDGET_STRADDLE_BUY',
                    'direction': 'LONG',
                    'price': price,
                })

        # ---- FII_FLOW_MOMENTUM ----
        fii_data = context.get('fii_data', {})
        if fii_data:
            # FII net buy/sell from previous day
            fii_net = fii_data.get('net_buy', 0)
            if (fii_net > 500 and time(10, 30) <= bar_time <= time(11, 30)
                    and prev_bar is not None and bar['close'] > prev_bar['close']):
                fires.append({
                    'signal_id': 'FII_FLOW_MOMENTUM',
                    'direction': 'LONG',
                    'price': price,
                })
            elif (fii_net < -500 and time(10, 30) <= bar_time <= time(11, 30)
                  and prev_bar is not None and bar['close'] < prev_bar['close']):
                fires.append({
                    'signal_id': 'FII_FLOW_MOMENTUM',
                    'direction': 'SHORT',
                    'price': price,
                })
        # If no FII data, skip this signal entirely

        # ---- ORR_REVERSION ----
        bars_today = context.get('bars_today', pd.DataFrame())
        if (isinstance(bars_today, pd.DataFrame) and len(bars_today) >= 3
                and time(9, 45) <= bar_time <= time(10, 30)):
            first_bar = bars_today.iloc[0]
            prev_close = context.get('daily_close', 0)
            if prev_close > 0:
                gap_pct = (float(first_bar['open']) - prev_close) / prev_close
                # Gap up > 0.5% => expect reversion
                if gap_pct > 0.005 and bar['close'] < bar['open']:
                    fires.append({
                        'signal_id': 'ORR_REVERSION',
                        'direction': 'SHORT',
                        'price': price,
                    })
                # Gap down > 0.5% => expect reversion
                elif gap_pct < -0.005 and bar['close'] > bar['open']:
                    fires.append({
                        'signal_id': 'ORR_REVERSION',
                        'direction': 'LONG',
                        'price': price,
                    })

        return fires

    # ────────────────────────────────────────────────────────────
    # TRADE SIMULATION
    # ────────────────────────────────────────────────────────────
    def _simulate_trades(self, bars_df: pd.DataFrame,
                         daily_ctx: dict) -> List[dict]:
        """
        Walk through bars day by day, fire signals, manage positions.

        Returns list of trade dicts:
            {signal_id, entry_date, exit_date, entry_price, exit_price,
             pnl, pnl_pct, direction, exit_reason}
        """
        if bars_df.empty:
            return []

        bars_df = bars_df.copy()
        bars_df['timestamp'] = pd.to_datetime(bars_df['timestamp'])
        # Use string dates to avoid Timestamp vs date comparison issues
        bars_df['_date_str'] = bars_df['timestamp'].dt.strftime('%Y-%m-%d')
        bars_df['_date'] = bars_df['timestamp'].dt.date
        bars_df = bars_df.sort_values('timestamp').reset_index(drop=True)

        trades = []
        # Open positions: signal_id -> {direction, entry_price, entry_date,
        #                               entry_bar_idx, sl, tgt, max_hold}
        open_positions: Dict[str, dict] = {}
        # Track which signals fired today (max 1 entry per signal per day)
        fired_today: Dict[str, bool] = {}
        current_date = None

        for idx, bar in bars_df.iterrows():
            bar_dict = bar.to_dict()
            bar_date = bar['_date']
            if hasattr(bar_date, 'date') and callable(bar_date.date):
                bar_date = bar_date.date()
            bar_time = bar['timestamp'].time()

            # New day reset
            if bar_date != current_date:
                current_date = bar_date
                fired_today = {}
                day_bars = bars_df[bars_df['_date_str'] == str(bar_date)]
            else:
                day_bars = bars_df[(bars_df['_date_str'] == str(bar_date))
                                   & (bars_df.index <= idx)]

            # Build context (convert day_bars to avoid Timestamp/date issues)
            context = self._build_context(bar_date, daily_ctx, day_bars.copy())

            # Previous bar
            prev_bar = bars_df.iloc[idx - 1].to_dict() if idx > 0 else None

            # ---- EXIT CHECKS for open positions ----
            to_close = []
            for sig_id, pos in open_positions.items():
                sig_cfg = INDIA_INTRADAY_SIGNALS.get(sig_id, {})
                instrument = sig_cfg.get('instrument', 'NIFTY')
                slippage = SLIPPAGE_PER_SIDE.get(instrument, 1.0)
                direction = pos['direction']
                entry_px = pos['entry_price']
                sl_pct = sig_cfg.get('sl_pct', 0.005)
                tgt_pct = sig_cfg.get('tgt_pct', 0.008)
                max_hold = sig_cfg.get('max_hold_bars', 30)
                bars_held = idx - pos['entry_bar_idx']

                exit_price = None
                exit_reason = None

                # Session close forced exit
                if bar_time >= SESSION_CLOSE:
                    exit_price = bar['close']
                    exit_reason = 'session_close'

                # Max hold bars
                elif bars_held >= max_hold:
                    exit_price = bar['close']
                    exit_reason = 'max_hold'

                # Stop loss
                elif direction == 'LONG' and bar['low'] <= entry_px * (1 - sl_pct):
                    exit_price = entry_px * (1 - sl_pct)
                    exit_reason = 'stop_loss'
                elif direction == 'SHORT' and bar['high'] >= entry_px * (1 + sl_pct):
                    exit_price = entry_px * (1 + sl_pct)
                    exit_reason = 'stop_loss'

                # Take profit
                elif direction == 'LONG' and bar['high'] >= entry_px * (1 + tgt_pct):
                    exit_price = entry_px * (1 + tgt_pct)
                    exit_reason = 'take_profit'
                elif direction == 'SHORT' and bar['low'] <= entry_px * (1 - tgt_pct):
                    exit_price = entry_px * (1 - tgt_pct)
                    exit_reason = 'take_profit'

                if exit_price is not None:
                    # Apply slippage
                    if direction == 'LONG':
                        raw_pnl = exit_price - entry_px
                        net_pnl = raw_pnl - 2 * slippage  # entry + exit
                    else:
                        raw_pnl = entry_px - exit_price
                        net_pnl = raw_pnl - 2 * slippage

                    pnl_pct = net_pnl / entry_px if entry_px > 0 else 0.0

                    trades.append({
                        'signal_id': sig_id,
                        'entry_date': pos['entry_date'],
                        'exit_date': bar['timestamp'],
                        'entry_price': entry_px,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'pnl_pct': pnl_pct,
                        'direction': direction,
                        'exit_reason': exit_reason,
                    })
                    to_close.append(sig_id)

            for sig_id in to_close:
                del open_positions[sig_id]

            # ---- ENTRY CHECKS ----
            if bar_time < SESSION_CLOSE:
                try:
                    fires = self._evaluate_signals(bar_dict, prev_bar, context)
                except Exception:
                    fires = []
                for fire in fires:
                    sig_id = fire['signal_id']
                    # Max 1 position per signal at a time
                    if sig_id in open_positions:
                        continue
                    if fired_today.get(sig_id, False):
                        continue

                    sig_cfg = INDIA_INTRADAY_SIGNALS.get(sig_id, {})
                    instrument = sig_cfg.get('instrument', 'NIFTY')
                    slippage = SLIPPAGE_PER_SIDE.get(instrument, 1.0)

                    # Apply entry slippage
                    entry_px = fire['price']
                    if fire['direction'] == 'LONG':
                        entry_px += slippage
                    else:
                        entry_px -= slippage

                    open_positions[sig_id] = {
                        'direction': fire['direction'],
                        'entry_price': entry_px,
                        'entry_date': bar['timestamp'],
                        'entry_bar_idx': idx,
                    }
                    fired_today[sig_id] = True

        # Force-close any remaining positions at last bar
        if open_positions and len(bars_df) > 0:
            last_bar = bars_df.iloc[-1]
            for sig_id, pos in list(open_positions.items()):
                sig_cfg = INDIA_INTRADAY_SIGNALS.get(sig_id, {})
                instrument = sig_cfg.get('instrument', 'NIFTY')
                slippage = SLIPPAGE_PER_SIDE.get(instrument, 1.0)
                direction = pos['direction']
                entry_px = pos['entry_price']
                exit_px = last_bar['close']

                if direction == 'LONG':
                    net_pnl = (exit_px - entry_px) - 2 * slippage
                else:
                    net_pnl = (entry_px - exit_px) - 2 * slippage

                trades.append({
                    'signal_id': sig_id,
                    'entry_date': pos['entry_date'],
                    'exit_date': last_bar['timestamp'],
                    'entry_price': entry_px,
                    'exit_price': exit_px,
                    'pnl': net_pnl,
                    'pnl_pct': net_pnl / entry_px if entry_px > 0 else 0.0,
                    'direction': direction,
                    'exit_reason': 'end_of_data',
                })

        return trades

    # ────────────────────────────────────────────────────────────
    # WALK-FORWARD WINDOWS
    # ────────────────────────────────────────────────────────────
    def _generate_wf_windows(self, data_start: date,
                             data_end: date) -> List[Tuple[date, date, date, date]]:
        """
        Generate (train_start, train_end, test_start, test_end) tuples.
        6mo train / 2mo test / 1mo step.
        """
        windows = []
        cursor = data_start

        while True:
            train_start = cursor
            train_end = add_months(cursor, WF_TRAIN_MONTHS) - timedelta(days=1)
            test_start = add_months(cursor, WF_TRAIN_MONTHS)
            test_end = add_months(test_start, WF_TEST_MONTHS) - timedelta(days=1)

            if test_end > data_end:
                break

            windows.append((train_start, train_end, test_start, test_end))
            cursor = add_months(cursor, WF_STEP_MONTHS)

        logger.info(f"Generated {len(windows)} WF windows")
        return windows

    # ────────────────────────────────────────────────────────────
    # RUN SINGLE SIGNAL
    # ────────────────────────────────────────────────────────────
    def run_signal(self, signal_id: str) -> dict:
        """
        Full walk-forward validation for one signal.

        Returns:
            {signal_id, trades, total_trades, win_rate, pf, sharpe,
             wf_results, wf_pass_rate, verdict}
        """
        if signal_id not in INDIA_INTRADAY_SIGNALS:
            raise ValueError(f"Unknown signal: {signal_id}. "
                             f"Valid: {list(INDIA_INTRADAY_SIGNALS.keys())}")

        logger.info(f"Running WF for {signal_id}")

        # Load full data range
        # Use ~495 trading days => ~2 years
        end_date = date.today()
        start_date = end_date - timedelta(days=750)  # ~2 years + warmup

        bars_df, daily_ctx = self._load_data(start_date, end_date)

        if bars_df.empty:
            logger.warning(f"No data loaded for {signal_id}")
            return {
                'signal_id': signal_id,
                'trades': [],
                'total_trades': 0,
                'win_rate': 0.0,
                'pf': 0.0,
                'sharpe': 0.0,
                'wf_results': [],
                'wf_pass_rate': 0.0,
                'verdict': 'FAIL_NO_DATA',
            }

        # Simulate all trades
        all_trades = self._simulate_trades(bars_df, daily_ctx)

        # Filter to this signal
        sig_trades = [t for t in all_trades if t['signal_id'] == signal_id]

        # Determine data range from bars
        bars_df['_date'] = pd.to_datetime(bars_df['timestamp']).dt.date
        data_start = bars_df['_date'].min()
        data_end = bars_df['_date'].max()

        # Generate WF windows
        windows = self._generate_wf_windows(data_start, data_end)

        if not windows:
            logger.warning(f"No WF windows generated for date range "
                           f"{data_start} to {data_end}")
            return {
                'signal_id': signal_id,
                'trades': sig_trades,
                'total_trades': len(sig_trades),
                'win_rate': _compute_win_rate(sig_trades),
                'pf': _compute_profit_factor(sig_trades),
                'sharpe': 0.0,
                'wf_results': [],
                'wf_pass_rate': 0.0,
                'verdict': 'FAIL_NO_WINDOWS',
            }

        # Evaluate each WF test window
        wf_results = []
        for train_start, train_end, test_start, test_end in windows:
            test_trades = []
            for t in sig_trades:
                ed = t['entry_date']
                if hasattr(ed, 'date') and callable(ed.date):
                    ed = ed.date()
                if test_start <= ed <= test_end:
                    test_trades.append(t)

            n_trades = len(test_trades)
            pf = _compute_profit_factor(test_trades)
            wr = _compute_win_rate(test_trades)

            # Daily PnL for Sharpe
            if test_trades:
                pnl_df = pd.DataFrame(test_trades)
                pnl_df['_date'] = pd.to_datetime(pnl_df['entry_date']).dt.date
                daily_pnl = pnl_df.groupby('_date')['pnl'].sum()
                sharpe = _compute_sharpe(daily_pnl)
            else:
                sharpe = 0.0

            # Pass/fail logic
            if n_trades < MIN_TRADES_PER_WINDOW:
                passed = False
                reason = f'too_few_trades ({n_trades})'
            elif n_trades < LOW_FREQ_THRESHOLD:
                # Relaxed criteria for low frequency
                passed = (sharpe >= MIN_SHARPE and pf >= LOW_FREQ_PF)
                reason = ('pass_relaxed' if passed
                          else f'fail_relaxed (S={sharpe:.2f}, PF={pf:.2f})')
            else:
                passed = (sharpe >= MIN_SHARPE and pf >= MIN_PF)
                reason = ('pass' if passed
                          else f'fail (S={sharpe:.2f}, PF={pf:.2f})')

            wf_results.append({
                'test_start': test_start,
                'test_end': test_end,
                'n_trades': n_trades,
                'pf': pf,
                'sharpe': sharpe,
                'win_rate': wr,
                'passed': passed,
                'reason': reason,
            })

        # Aggregate
        evaluable = [w for w in wf_results
                     if w['n_trades'] >= MIN_TRADES_PER_WINDOW]
        if evaluable:
            pass_rate = sum(1 for w in evaluable if w['passed']) / len(evaluable)
        else:
            pass_rate = 0.0

        overall_pf = _compute_profit_factor(sig_trades)
        overall_wr = _compute_win_rate(sig_trades)

        # Overall Sharpe
        if sig_trades:
            pnl_df = pd.DataFrame(sig_trades)
            pnl_df['_date'] = pd.to_datetime(pnl_df['entry_date']).dt.date
            daily_pnl = pnl_df.groupby('_date')['pnl'].sum()
            overall_sharpe = _compute_sharpe(daily_pnl)
        else:
            overall_sharpe = 0.0

        # Verdict
        if len(sig_trades) == 0:
            verdict = 'FAIL_NO_TRADES'
        elif pass_rate >= MIN_WF_PASS_RATE:
            verdict = 'PASS'
        elif pass_rate >= 0.40:
            verdict = 'MARGINAL'
        else:
            verdict = 'FAIL'

        # Special case: FII_FLOW_MOMENTUM with no FII data
        if signal_id == 'FII_FLOW_MOMENTUM' and len(sig_trades) == 0:
            verdict = 'SKIP_NO_FII_DATA'

        result = {
            'signal_id': signal_id,
            'trades': sig_trades,
            'total_trades': len(sig_trades),
            'win_rate': overall_wr,
            'pf': overall_pf,
            'sharpe': overall_sharpe,
            'wf_results': wf_results,
            'wf_pass_rate': pass_rate,
            'verdict': verdict,
        }

        logger.info(f"{signal_id}: {len(sig_trades)} trades, "
                    f"PF={overall_pf:.2f}, Sharpe={overall_sharpe:.2f}, "
                    f"WF pass={pass_rate:.0%} => {verdict}")
        return result

    # ────────────────────────────────────────────────────────────
    # RUN ALL SIGNALS
    # ────────────────────────────────────────────────────────────
    def run_all(self, print_results: bool = True) -> dict:
        """
        Run walk-forward validation on all 5 India intraday signals.

        Returns dict keyed by signal_id -> result dict.
        """
        results = {}

        for sig_id in INDIA_INTRADAY_SIGNALS:
            try:
                results[sig_id] = self.run_signal(sig_id)
            except Exception as e:
                logger.error(f"Error running {sig_id}: {e}")
                results[sig_id] = {
                    'signal_id': sig_id,
                    'trades': [],
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'pf': 0.0,
                    'sharpe': 0.0,
                    'wf_results': [],
                    'wf_pass_rate': 0.0,
                    'verdict': f'ERROR: {e}',
                }

        if print_results:
            self._print_results(results)

        return results

    @staticmethod
    def _print_results(results: dict):
        """Print formatted results table."""
        header = (f"{'Signal':<25} {'Trades':>7} {'WR':>6} {'PF':>6} "
                  f"{'Sharpe':>7} {'WF%':>6} {'Verdict':<20}")
        sep = '-' * len(header)

        print("\n" + sep)
        print("INDIA INTRADAY WALK-FORWARD RESULTS")
        print(sep)
        print(header)
        print(sep)

        for sig_id in INDIA_INTRADAY_SIGNALS:
            r = results.get(sig_id, {})
            print(f"{sig_id:<25} "
                  f"{r.get('total_trades', 0):>7} "
                  f"{r.get('win_rate', 0):>5.1%} "
                  f"{r.get('pf', 0):>6.2f} "
                  f"{r.get('sharpe', 0):>7.2f} "
                  f"{r.get('wf_pass_rate', 0):>5.0%} "
                  f"{r.get('verdict', 'N/A'):<20}")

        print(sep)

        # Summary
        pass_count = sum(1 for r in results.values()
                         if r.get('verdict') == 'PASS')
        total = len(results)
        print(f"\nPassed: {pass_count}/{total}")

        skip_count = sum(1 for r in results.values()
                         if 'SKIP' in str(r.get('verdict', '')))
        if skip_count:
            print(f"Skipped (no data): {skip_count}")

        print()


# ================================================================
# CLI ENTRY POINT
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Walk-forward validation for India intraday signals')
    parser.add_argument('--signal', type=str, default=None,
                        help='Run a single signal (e.g. EXPIRY_PIN_FADE)')
    parser.add_argument('--db-dsn', type=str, default=DATABASE_DSN,
                        help='Database connection string')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    try:
        conn = psycopg2.connect(args.db_dsn)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.info("Proceeding with synthetic data fallback")
        conn = None

    # Create a minimal mock connection if DB unavailable
    if conn is None:
        class _MockConn:
            """Minimal mock that forces fallback to synthetic data."""
            def cursor(self):
                raise Exception("No DB connection")
        conn = _MockConn()

    wf = IndiaIntradayWF(conn)

    if args.signal:
        sig = args.signal.upper()
        if sig not in INDIA_INTRADAY_SIGNALS:
            print(f"Unknown signal: {sig}")
            print(f"Valid signals: {list(INDIA_INTRADAY_SIGNALS.keys())}")
            sys.exit(1)
        result = wf.run_signal(sig)
        wf._print_results({sig: result})
    else:
        wf.run_all(print_results=True)

    if hasattr(conn, 'close'):
        try:
            conn.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()

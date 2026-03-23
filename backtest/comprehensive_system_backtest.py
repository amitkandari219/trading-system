"""
Comprehensive System Backtest — DEFINITIVE test of the entire trading system.

Evaluates ALL signal layers (7 daily scoring + 13 structural + 4 overlays)
against real DB data (nifty_daily 2015-2026).

Database: postgresql://trader:trader123@localhost:5450/trading
Tables:   nifty_daily (2552 rows), nifty_options (510K), intraday_bars (36K)

Usage:
    venv/bin/python3 -m backtest.comprehensive_system_backtest
"""

import importlib
import logging
import math
import os
import sys
import time as time_mod
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

from backtest.generic_backtest import _eval_conditions
from backtest.indicators import add_all_indicators
from backtest.transaction_costs import TransactionCostModel
from config.settings import DATABASE_DSN, RISK_FREE_RATE

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

INITIAL_CAPITAL = 1_000_000          # 10 lakh
MAX_POSITIONS = 4
MAX_SAME_DIR = 2
LOT_SIZE_CHANGE_DATE = date(2023, 7, 1)
LOT_SIZE_PRE = 75
LOT_SIZE_POST = 25
MARGIN_PER_LOT = 120_000
NIFTY_PT_VALUE = 1                   # 1 pt per share

# Transaction cost model
COST_MODEL = TransactionCostModel()

# ================================================================
# SIGNAL DEFINITIONS
# ================================================================

SIGNAL_RULES = {
    'KAUFMAN_DRY_20': {
        'direction': 'LONG',
        'entry_long': [{'indicator': 'sma_10', 'op': '<', 'value': 'close'},
                       {'indicator': 'stoch_k_5', 'op': '>', 'value': 50}],
        'exit_long': [{'indicator': 'stoch_k_5', 'op': '<=', 'value': 50}],
        'stop_loss_pct': 0.02, 'cooldown_days': 2, 'max_hold': 15,
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
        'max_hold': 15,
    },
    'KAUFMAN_DRY_12': {
        'direction': 'BOTH',
        'entry_long': [{'indicator': 'sma_50', 'op': '<', 'value': 'close'},
                       {'indicator': 'adx_14', 'op': '>', 'value': 20}],
        'exit_long': [{'indicator': 'adx_14', 'op': '<', 'value': 15}],
        'entry_short': [{'indicator': 'sma_50', 'op': '>', 'value': 'close'},
                        {'indicator': 'adx_14', 'op': '>', 'value': 20}],
        'exit_short': [{'indicator': 'adx_14', 'op': '<', 'value': 15}],
        'stop_loss_pct': 0.02, 'take_profit_pct': 0.03, 'hold_days': 7,
        'cooldown_days': 1, 'max_hold': 10,
    },
    'GUJRAL_DRY_8': {
        'direction': 'LONG',
        'entry_long': [{'indicator': 'close', 'op': '>', 'value': 'sma_20'},
                       {'indicator': 'rsi_14', 'op': '>', 'value': 45}],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'sma_20'}],
        'stop_loss_pct': 0.02, 'cooldown_days': 3, 'max_hold': 20,
    },
    'GUJRAL_DRY_13': {
        'direction': 'LONG',
        'entry_long': [{'indicator': 'close', 'op': '>', 'value': 'ema_20'},
                       {'indicator': 'rsi_14', 'op': '>', 'value': 55},
                       {'indicator': 'rsi_14', 'op': '<', 'value': 75}],
        'exit_long': [{'indicator': 'rsi_14', 'op': '<', 'value': 45}],
        'stop_loss_pct': 0.02, 'hold_days': 10, 'cooldown_days': 2,
        'max_hold': 15,
    },
    'BULKOWSKI_ADAM_EVE': {
        'direction': 'LONG',
        'entry_long': [{'indicator': 'close', 'op': '>', 'value': 'bb_lower'},
                       {'indicator': 'rsi_14', 'op': '<', 'value': 35}],
        'exit_long': [{'indicator': 'rsi_14', 'op': '>', 'value': 60}],
        'stop_loss_pct': 0.03, 'cooldown_days': 5, 'max_hold': 20,
    },
    'SCHWAGER_TREND': {
        'direction': 'LONG',
        'entry_long': [{'indicator': 'close', 'op': '>', 'value': 'sma_50'},
                       {'indicator': 'adx_14', 'op': '>', 'value': 25}],
        'exit_long': [{'indicator': 'close', 'op': '<', 'value': 'sma_50'}],
        'stop_loss_pct': 0.02, 'cooldown_days': 3, 'max_hold': 25,
    },
}

STRUCTURAL_SIGNALS = {
    'MAX_OI_BARRIER': ('signals.structural.max_oi_barrier', 'MaxOIBarrierSignal'),
    'MONDAY_STRADDLE': ('signals.structural.monday_straddle', 'MondayStraddle'),
    'GIFT_CONVERGENCE': ('signals.structural.gift_convergence', 'GiftConvergenceSignal'),
    'EOD_INSTITUTIONAL_FLOW': ('signals.structural.eod_institutional_flow', 'EODInstitutionalFlowSignal'),
    'GAMMA_SQUEEZE': ('signals.structural.gamma_squeeze', 'GammaSqueezeSignal'),
    'OPENING_CANDLE': ('signals.structural.opening_candle', 'OpeningCandleSignal'),
    'SIP_FLOW': ('signals.structural.sip_flow', 'SIPFlowSignal'),
    'SKEW_REVERSAL': ('signals.structural.skew_reversal', 'SkewReversalSignal'),
    'THURSDAY_PIN_SETUP': ('signals.structural.thursday_pin_setup', 'ThursdayPinSetupSignal'),
    'RBI_DRIFT': ('signals.structural.rbi_drift', 'RBIDriftSignal'),
    'QUARTER_WINDOW': ('signals.structural.quarter_window', 'QuarterWindowSignal'),
    'DII_PUT_FLOOR': ('signals.structural.dii_put_floor', 'DIIPutFloorSignal'),
    'ROLLOVER_FLOW': ('signals.structural.rollover_flow', 'RolloverFlowSignal'),
}

OVERLAY_SIGNALS = {
    'VIX_TRANSMISSION': ('signals.structural.vix_transmission', 'VIXTransmissionSignal'),
    'RBI_INTERVENTION': ('signals.structural.rbi_intervention', 'RBIInterventionSignal'),
    'PREOPEN_AUCTION': ('signals.structural.preopen_auction', 'PreOpenAuctionSignal'),
    'FII_DIVERGENCE': ('signals.structural.fii_divergence', 'FIIDivergenceSignal'),
}


# ================================================================
# TRADE DATACLASS
# ================================================================

@dataclass
class Trade:
    trade_id: int = 0
    signal_id: str = ''
    signal_type: str = ''              # DAILY / STRUCTURAL
    entry_date: str = ''
    exit_date: str = ''
    direction: str = ''                # LONG / SHORT
    entry_price: float = 0.0
    exit_price: float = 0.0
    lots: int = 1
    lot_size: int = 25
    notional_entry: float = 0.0
    notional_exit: float = 0.0
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    net_pnl_pct: float = 0.0
    equity_before: float = 0.0
    equity_after: float = 0.0
    exit_reason: str = ''
    days_held: int = 0
    regime: str = ''
    vix_at_entry: float = 0.0
    adx_at_entry: float = 0.0
    rsi_at_entry: float = 0.0
    overlay_modifier: float = 1.0
    overlay_detail: str = ''
    conviction_score: float = 0.0
    day_of_week: int = 0
    month: int = 0
    year: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    is_synthetic: bool = False
    structural_confidence: float = 0.0
    structural_strength: str = ''


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def get_lot_size(trade_date: date) -> int:
    if trade_date >= LOT_SIZE_CHANGE_DATE:
        return LOT_SIZE_POST
    return LOT_SIZE_PRE


def detect_regime(row: pd.Series) -> str:
    """Detect market regime from VIX, ADX, and SMA alignment."""
    vix = float(row.get('india_vix', 15)) if pd.notna(row.get('india_vix')) else 15.0
    adx_val = float(row.get('adx_14', 20)) if pd.notna(row.get('adx_14')) else 20.0
    close = float(row['close'])
    sma50 = float(row.get('sma_50', close)) if pd.notna(row.get('sma_50')) else close
    sma200 = float(row.get('sma_200', close)) if pd.notna(row.get('sma_200')) else close

    if vix > 25:
        return 'CRISIS'
    elif vix > 18 and adx_val > 25:
        return 'HIGH_VOL_TRENDING'
    elif adx_val > 25 and close > sma50 > sma200:
        return 'TRENDING_UP'
    elif adx_val > 25 and close < sma50 < sma200:
        return 'TRENDING_DOWN'
    elif adx_val < 20:
        return 'RANGE_BOUND'
    else:
        return 'TRANSITIONAL'


def compute_costs(entry_price: float, exit_price: float, lots: int,
                  lot_size: int, vix: float = None) -> float:
    """Compute round-trip transaction costs in rupees."""
    costs = COST_MODEL.compute_futures_round_trip(
        entry_price=entry_price, exit_price=exit_price,
        lots=lots, vix=vix,
    )
    return costs.total


def build_synthetic_market_data(row: pd.Series, prev_row: pd.Series,
                                df: pd.DataFrame, idx: int) -> Dict[str, Any]:
    """
    Build a comprehensive market_data dict for structural signal evaluation.
    Uses daily OHLCV to synthesize intraday/options fields where needed.
    Synthetic fields are marked with _synthetic suffix in comments.
    """
    close = float(row['close'])
    open_ = float(row['open'])
    high = float(row['high'])
    low = float(row['low'])
    volume = float(row.get('volume', 0))
    prev_close = float(prev_row['close']) if prev_row is not None else close

    vix = float(row.get('india_vix', 15)) if pd.notna(row.get('india_vix')) else 15.0
    pcr = float(row.get('pcr_oi', 1.0)) if pd.notna(row.get('pcr_oi')) else 1.0

    try:
        trade_date = pd.Timestamp(row['date']).date() if 'date' in row.index else date.today()
    except Exception:
        trade_date = date.today()

    day_of_week = trade_date.weekday()

    # Synthetic intraday fields from daily OHLCV
    gap_pct = (open_ - prev_close) / prev_close if prev_close > 0 else 0
    body_pct = (close - open_) / open_ if open_ > 0 else 0
    range_pct = (high - low) / close if close > 0 else 0

    # Synthetic first 15-min candle (from open and direction of day)
    first_15min_close_est = open_ + (close - open_) * 0.08  # ~8% of day's move
    first_15min_high_est = max(open_, first_15min_close_est) + (high - low) * 0.05
    first_15min_low_est = min(open_, first_15min_close_est) - (high - low) * 0.03
    first_15min_vol_est = volume * 0.12  # ~12% of daily volume in first 15 min

    # Average first-15min volume (20d rolling)
    if idx >= 20 and 'volume' in df.columns:
        avg_15min_vol = float(df['volume'].iloc[max(0, idx-20):idx].mean()) * 0.12
    else:
        avg_15min_vol = first_15min_vol_est * 0.9

    # Synthetic morning moves
    first_30min_move = body_pct * 0.15   # ~15% of day's body in first 30 min
    next_45min_move = body_pct * 0.10    # ~10% of day's body in next 45 min

    # Last hour volume fraction (synthetic)
    last_hour_vol = volume * 0.25

    # 5-day return
    if idx >= 5:
        ret_5d = (close / float(df['close'].iloc[idx - 5]) - 1) * 100
    else:
        ret_5d = 0.0

    # Days to weekly expiry (Tuesday expiry)
    # 0=Mon -> 1 day, 1=Tue -> 0 days, 2=Wed -> 6 days, etc.
    days_to_expiry_map = {0: 1, 1: 0, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2}
    days_to_weekly_expiry = days_to_expiry_map.get(day_of_week, 7)

    # ATM OI concentration (synthetic — use PCR as proxy)
    atm_oi_pct = 0.08 + (pcr - 1.0) * 0.03  # ~8-11% range based on PCR

    # Monthly expiry proximity
    # Monthly expiry = last Thursday of month
    last_day = date(trade_date.year, trade_date.month, 28)
    try:
        import calendar
        last_day = date(trade_date.year, trade_date.month,
                        calendar.monthrange(trade_date.year, trade_date.month)[1])
        while last_day.weekday() != 3:  # Thursday
            last_day -= timedelta(days=1)
    except Exception:
        pass
    days_to_monthly_expiry = (last_day - trade_date).days

    # Rollover percentage (synthetic — higher near expiry)
    rollover_pct = max(0, min(100, 50 + (20 - days_to_monthly_expiry) * 3))

    # Max OI strikes (synthetic from daily levels)
    atm_strike = round(close / 50) * 50
    max_put_oi_strike = atm_strike - 200
    max_call_oi_strike = atm_strike + 200

    # US VIX (synthetic — approximate from India VIX)
    us_vix_est = vix * 1.15 + 2.0  # rough linear approx

    return {
        # Core fields
        'date': trade_date,
        'trade_date': trade_date,
        'day_of_week': day_of_week,
        'close': close,
        'open': open_,
        'high': high,
        'low': low,
        'volume': volume,
        'prev_close': prev_close,
        'current_price': close,
        'day_open': open_,

        # VIX / PCR
        'india_vix': vix,
        'india_vix_current': vix,
        'india_vix_prev': vix * (1 + np.random.normal(0, 0.03)),  # synthetic
        'pcr_oi': pcr,
        'vix': vix,

        # US VIX (synthetic)
        'us_vix_current': us_vix_est,
        'us_vix_prev': us_vix_est * (1 + np.random.normal(0, 0.04)),

        # Gap
        'gap_pct': gap_pct,
        'overnight_gap_pct': gap_pct,

        # Intraday synthetic
        'first_15min_open': open_,
        'first_15min_close': first_15min_close_est,
        'first_15min_high': first_15min_high_est,
        'first_15min_low': first_15min_low_est,
        'first_15min_volume': first_15min_vol_est,
        'avg_first_15min_volume_20d': avg_15min_vol,
        'first_30min_move_pct': first_30min_move,
        'next_45min_move_pct': next_45min_move,
        'last_hour_volume': last_hour_vol,
        'last_hour_volume_ratio': 0.25,

        # Option-chain synthetic
        'max_put_oi_strike': max_put_oi_strike,
        'max_call_oi_strike': max_call_oi_strike,
        'max_put_oi': 8_000_000,       # synthetic default
        'max_call_oi': 7_500_000,
        'atm_oi_pct_of_total': atm_oi_pct,
        'atm_strike': atm_strike,

        # Expiry
        'days_to_weekly_expiry': days_to_weekly_expiry,
        'days_to_monthly_expiry': days_to_monthly_expiry,
        'is_expiry_day': days_to_weekly_expiry == 0,
        'is_monthly_expiry': days_to_monthly_expiry == 0,

        # Returns
        'nifty_5d_return_pct': ret_5d,
        'returns_1d_pct': body_pct * 100,

        # Flow (synthetic)
        'fii_net_buy_crore': np.random.normal(0, 2000),
        'dii_net_buy_crore': np.random.normal(500, 1500),
        'monthly_sip_flow_crore': 20000,
        'rollover_pct': rollover_pct,

        # Straddle (synthetic)
        'atm_ce_premium': vix * close / 2400,   # rough IV -> premium
        'atm_pe_premium': vix * close / 2600,
        'is_event_day': False,

        # RBI
        'usdinr_current': 83.5,         # synthetic fixed
        'usdinr_prev': 83.5,
        'rbi_policy_date': False,

        # Skew (synthetic)
        'iv_25d_put': vix + 2,
        'iv_25d_call': vix - 1,
        'iv_atm': vix,

        # Pre-open auction (synthetic)
        'preopen_price': open_ * (1 + np.random.normal(0, 0.001)),
        'preopen_volume': volume * 0.02,
        'avg_preopen_volume_20d': volume * 0.018,

        # Flags
        '_synthetic': True,
    }


# ================================================================
# COMPREHENSIVE BACKTEST CLASS
# ================================================================

class ComprehensiveBacktest:
    """Definitive backtest of the entire trading system."""

    def __init__(self, capital: float = INITIAL_CAPITAL):
        self.initial_capital = capital
        self.equity = capital
        self.peak_equity = capital
        self.max_dd = 0.0
        self.trades: List[Trade] = []
        self.daily_equity: Dict[str, float] = {}
        self.daily_pnl: Dict[str, float] = {}
        self.trade_counter = 0

        # Signal instances
        self.structural_instances: Dict[str, Any] = {}
        self.overlay_instances: Dict[str, Any] = {}

        # Per-signal state for daily rules
        self.sig_state: Dict[str, Dict] = {}
        for sig_id in SIGNAL_RULES:
            self.sig_state[sig_id] = {
                'position': None, 'entry_price': 0, 'entry_idx': 0,
                'days_in': 0, 'last_exit': -10, 'lots': 0,
                'direction': None, 'regime': '', 'vix': 0, 'adx': 0,
                'rsi': 0, 'overlay_mod': 1.0,
            }

        # Open positions tracker
        self.open_positions: Dict[str, str] = {}

    # ----------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Load nifty_daily from DB and add all indicators."""
        print("  Loading nifty_daily from PostgreSQL...")
        conn = psycopg2.connect(DATABASE_DSN)
        df = pd.read_sql(
            "SELECT date, open, high, low, close, volume, india_vix, pcr_oi "
            "FROM nifty_daily ORDER BY date",
            conn, parse_dates=['date'],
        )
        conn.close()
        print(f"  Loaded {len(df)} daily bars ({df['date'].min().date()} to {df['date'].max().date()})")

        print("  Computing indicators...")
        df = add_all_indicators(df)
        print(f"  Added {len([c for c in df.columns if c not in ('date','open','high','low','close','volume')])} indicator columns")
        return df

    # ----------------------------------------------------------
    # LOAD SIGNAL CLASSES
    # ----------------------------------------------------------
    def load_signals(self):
        """Dynamically import and instantiate all structural + overlay signals."""
        print("\n  Loading structural signals...")
        for sig_id, (module_path, class_name) in STRUCTURAL_SIGNALS.items():
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                self.structural_instances[sig_id] = cls()
                print(f"    [OK] {sig_id}")
            except Exception as e:
                print(f"    [FAIL] {sig_id}: {e}")

        print("  Loading overlay signals...")
        for sig_id, (module_path, class_name) in OVERLAY_SIGNALS.items():
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                self.overlay_instances[sig_id] = cls()
                print(f"    [OK] {sig_id}")
            except Exception as e:
                print(f"    [FAIL] {sig_id}: {e}")

        print(f"  Loaded: {len(self.structural_instances)} structural, "
              f"{len(self.overlay_instances)} overlay signals")

    # ----------------------------------------------------------
    # EVALUATE OVERLAYS
    # ----------------------------------------------------------
    def evaluate_overlays(self, market_data: Dict) -> Tuple[float, str]:
        """Evaluate all overlay signals, return composite modifier and detail."""
        modifiers = {}
        details = []
        for sig_id, instance in self.overlay_instances.items():
            try:
                result = instance.evaluate(market_data)
                if result and isinstance(result, dict):
                    mod = result.get('size_modifier', 1.0)
                    if mod is None:
                        mod = 1.0
                    modifiers[sig_id] = mod
                    bias = result.get('bias', 'NEUTRAL')
                    details.append(f"{sig_id}={mod:.2f}({bias})")
            except Exception:
                pass

        if not modifiers:
            return 1.0, 'NO_OVERLAYS'

        # Geometric mean of all overlay modifiers
        vals = list(modifiers.values())
        geo_mean = math.exp(sum(math.log(max(0.1, v)) for v in vals) / len(vals))
        composite = max(0.3, min(2.0, geo_mean))
        detail_str = ', '.join(details)
        return composite, detail_str

    # ----------------------------------------------------------
    # EVALUATE STRUCTURAL SIGNALS
    # ----------------------------------------------------------
    def evaluate_structural(self, market_data: Dict) -> List[Dict]:
        """Evaluate all structural signals, return list of fired signals."""
        fired = []
        for sig_id, instance in self.structural_instances.items():
            try:
                result = instance.evaluate(market_data)
                if result and isinstance(result, dict):
                    direction = result.get('direction')
                    if direction and direction in ('LONG', 'SHORT', 'BULLISH', 'BEARISH'):
                        # Normalize direction
                        if direction == 'BULLISH':
                            direction = 'LONG'
                        elif direction == 'BEARISH':
                            direction = 'SHORT'
                        fired.append({
                            'signal_id': sig_id,
                            'direction': direction,
                            'confidence': result.get('confidence', 0.5),
                            'size_modifier': result.get('size_modifier', 1.0),
                            'strength': result.get('strength', 'MODERATE'),
                        })
            except Exception:
                pass
        return fired

    # ----------------------------------------------------------
    # POSITION SIZING
    # ----------------------------------------------------------
    def compute_lots(self, price: float, stop_loss_pct: float,
                     trade_date: date, overlay_mod: float = 1.0) -> int:
        """Compute lot count from risk budget."""
        lot_size = get_lot_size(trade_date)
        sl_pts = price * stop_loss_pct
        risk_per_lot = sl_pts * lot_size
        risk_budget = self.equity * 0.02 / MAX_POSITIONS
        if risk_per_lot <= 0:
            return 1
        base_lots = max(1, math.floor(risk_budget / risk_per_lot))
        adjusted = max(1, round(base_lots * overlay_mod))
        # Margin constraint
        margin_cap = math.floor(self.equity * 0.60 / MARGIN_PER_LOT)
        adjusted = min(adjusted, margin_cap, 20)
        return max(1, adjusted)

    # ----------------------------------------------------------
    # SIMULATE EXIT
    # ----------------------------------------------------------
    def simulate_exit(self, df: pd.DataFrame, entry_idx: int, direction: str,
                      entry_price: float, rules: Dict) -> Tuple[int, float, str, float, float]:
        """
        Simulate trade exit using subsequent daily bars.
        Returns (exit_idx, exit_price, exit_reason, max_favorable, max_adverse).
        """
        sl_pct = rules.get('stop_loss_pct', 0.02)
        tp_pct = rules.get('take_profit_pct', 0)
        hold_days = rules.get('hold_days', 0)
        max_hold = rules.get('max_hold', 25)
        exit_long_conds = rules.get('exit_long', [])
        exit_short_conds = rules.get('exit_short', [])

        max_fav = 0.0
        max_adv = 0.0
        n = len(df)

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
            row_j = df.iloc[j]
            prev_j = df.iloc[j - 1]
            c = float(row_j['close'])
            h = float(row_j['high'])
            l_ = float(row_j['low'])
            days = j - entry_idx

            if direction == 'LONG':
                fav = (h - entry_price) / entry_price
                adv = (entry_price - l_) / entry_price
                cur_pnl_pct = (c - entry_price) / entry_price
            else:
                fav = (entry_price - l_) / entry_price
                adv = (h - entry_price) / entry_price
                cur_pnl_pct = (entry_price - c) / entry_price

            max_fav = max(max_fav, fav)
            max_adv = max(max_adv, adv)

            # Stop loss (check against intraday extreme)
            if sl_pct > 0 and adv >= sl_pct:
                if direction == 'LONG':
                    exit_p = entry_price * (1 - sl_pct)
                else:
                    exit_p = entry_price * (1 + sl_pct)
                return j, exit_p, 'SL', max_fav, max_adv

            # Take profit
            if tp_pct > 0 and fav >= tp_pct:
                if direction == 'LONG':
                    exit_p = entry_price * (1 + tp_pct)
                else:
                    exit_p = entry_price * (1 - tp_pct)
                return j, exit_p, 'TP', max_fav, max_adv

            # Hold days
            if hold_days > 0 and days >= hold_days:
                return j, c, 'TIME', max_fav, max_adv

            # Rule-based exit
            if direction == 'LONG' and exit_long_conds:
                if _eval_conditions(row_j, prev_j, exit_long_conds):
                    return j, c, 'RULE', max_fav, max_adv
            elif direction == 'SHORT' and exit_short_conds:
                if _eval_conditions(row_j, prev_j, exit_short_conds):
                    return j, c, 'RULE', max_fav, max_adv

        # Max hold reached or end of data
        exit_idx = min(entry_idx + max_hold, n - 1)
        return exit_idx, float(df.iloc[exit_idx]['close']), 'MAX_HOLD', max_fav, max_adv

    # ----------------------------------------------------------
    # RECORD TRADE
    # ----------------------------------------------------------
    def record_trade(self, sig_id: str, sig_type: str, entry_date, exit_date,
                     direction: str, entry_price: float, exit_price: float,
                     lots: int, lot_size: int, exit_reason: str, days_held: int,
                     regime: str, vix: float, adx: float, rsi_val: float,
                     overlay_mod: float, overlay_detail: str,
                     max_fav: float, max_adv: float,
                     is_synthetic: bool = False,
                     structural_conf: float = 0.0,
                     structural_str: str = '') -> Trade:
        """Record a completed trade and update equity."""
        self.trade_counter += 1

        if direction == 'LONG':
            gross_pnl = lots * lot_size * (exit_price - entry_price)
        else:
            gross_pnl = lots * lot_size * (entry_price - exit_price)

        costs = compute_costs(entry_price, exit_price, lots, lot_size, vix)
        net_pnl = gross_pnl - costs

        equity_before = self.equity
        self.equity += net_pnl
        self.equity = max(self.equity, 10_000)
        self.peak_equity = max(self.peak_equity, self.equity)
        dd = (self.peak_equity - self.equity) / self.peak_equity
        self.max_dd = max(self.max_dd, dd)

        try:
            td = pd.Timestamp(entry_date).date()
        except Exception:
            td = date.today()

        trade = Trade(
            trade_id=self.trade_counter,
            signal_id=sig_id,
            signal_type=sig_type,
            entry_date=str(entry_date)[:10],
            exit_date=str(exit_date)[:10],
            direction=direction,
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            lots=lots,
            lot_size=lot_size,
            notional_entry=round(lots * lot_size * entry_price),
            notional_exit=round(lots * lot_size * exit_price),
            gross_pnl=round(gross_pnl),
            costs=round(costs),
            net_pnl=round(net_pnl),
            net_pnl_pct=round(net_pnl / (lots * lot_size * entry_price) * 100, 3)
                        if entry_price > 0 else 0,
            equity_before=round(equity_before),
            equity_after=round(self.equity),
            exit_reason=exit_reason,
            days_held=days_held,
            regime=regime,
            vix_at_entry=round(vix, 1),
            adx_at_entry=round(adx, 1),
            rsi_at_entry=round(rsi_val, 1),
            overlay_modifier=round(overlay_mod, 3),
            overlay_detail=overlay_detail,
            conviction_score=round(overlay_mod * (1 + structural_conf) / 2, 3),
            day_of_week=td.weekday(),
            month=td.month,
            year=td.year,
            max_favorable=round(max_fav * 100, 2),
            max_adverse=round(max_adv * 100, 2),
            is_synthetic=is_synthetic,
            structural_confidence=round(structural_conf, 3),
            structural_strength=structural_str,
        )
        self.trades.append(trade)
        return trade

    # ----------------------------------------------------------
    # MAIN RUN
    # ----------------------------------------------------------
    def run(self, df: pd.DataFrame):
        """Run the comprehensive backtest over all trading days."""
        n = len(df)
        dates = df['date'].values if 'date' in df.columns else df.index.values
        closes = df['close'].values.astype(float)

        print(f"\n  Running backtest over {n} bars...")
        print(f"  Evaluating: 7 daily rules + {len(self.structural_instances)} structural "
              f"+ {len(self.overlay_instances)} overlays")

        np.random.seed(42)  # Reproducibility for synthetic fields

        daily_signals_fired = 0
        structural_signals_fired = 0
        skipped_conflict = 0

        for i in range(1, n):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            c = float(closes[i])

            try:
                dt_date = pd.Timestamp(dates[i]).date()
            except Exception:
                dt_date = date(2024, 1, 1)

            lot_size = get_lot_size(dt_date)
            regime = detect_regime(row)
            vix = float(row.get('india_vix', 15)) if pd.notna(row.get('india_vix')) else 15.0
            adx_val = float(row.get('adx_14', 20)) if pd.notna(row.get('adx_14')) else 20.0
            rsi_val = float(row.get('rsi_14', 50)) if pd.notna(row.get('rsi_14')) else 50.0

            # Build market data context
            market_data = build_synthetic_market_data(row, prev, df, i)

            # Evaluate overlays
            overlay_mod, overlay_detail = self.evaluate_overlays(market_data)

            # Track daily equity
            d_str = str(dates[i])[:10]
            self.daily_equity[d_str] = self.equity

            # ── CHECK EXITS for open daily positions ──
            for sig_id, state in self.sig_state.items():
                if state['position'] is None:
                    continue

                state['days_in'] += 1
                rules = SIGNAL_RULES[sig_id]
                pos = state['position']
                ep = state['entry_price']
                sl_pct = rules.get('stop_loss_pct', 0.02)
                tp_pct = rules.get('take_profit_pct', 0)
                hold = rules.get('hold_days', 0)
                max_hold = rules.get('max_hold', 25)

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
                    elif state['days_in'] >= max_hold:
                        exit_reason = 'MAX_HOLD'
                else:  # SHORT
                    if sl_pct > 0 and c >= ep * (1 + sl_pct):
                        exit_reason, exit_price = 'SL', ep * (1 + sl_pct)
                    elif tp_pct > 0 and c <= ep * (1 - tp_pct):
                        exit_reason, exit_price = 'TP', ep * (1 - tp_pct)
                    elif rules.get('exit_short') and _eval_conditions(row, prev, rules['exit_short']):
                        exit_reason = 'RULE'
                    elif hold > 0 and state['days_in'] >= hold:
                        exit_reason = 'TIME'
                    elif state['days_in'] >= max_hold:
                        exit_reason = 'MAX_HOLD'

                if exit_reason:
                    self.record_trade(
                        sig_id=sig_id, sig_type='DAILY',
                        entry_date=dates[state['entry_idx']],
                        exit_date=dates[i],
                        direction=pos, entry_price=ep, exit_price=exit_price,
                        lots=state['lots'], lot_size=lot_size,
                        exit_reason=exit_reason, days_held=state['days_in'],
                        regime=state['regime'], vix=state['vix'],
                        adx=state['adx'], rsi_val=state['rsi'],
                        overlay_mod=state['overlay_mod'],
                        overlay_detail='',
                        max_fav=0, max_adv=0,
                    )
                    state['position'] = None
                    state['last_exit'] = i
                    self.open_positions.pop(sig_id, None)

            # ── CHECK ENTRIES for daily signals ──
            for sig_id, rules in SIGNAL_RULES.items():
                state = self.sig_state[sig_id]
                if state['position'] is not None:
                    continue
                cooldown = rules.get('cooldown_days', 1)
                if i - state['last_exit'] < cooldown:
                    continue
                if len(self.open_positions) >= MAX_POSITIONS:
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
                same_dir = sum(1 for p in self.open_positions.values() if p == fired_dir)
                if same_dir >= MAX_SAME_DIR:
                    continue

                # Conflict check: if structural fired opposing direction, skip
                # (done later in structural section)

                lots = self.compute_lots(c, rules.get('stop_loss_pct', 0.02),
                                         dt_date, overlay_mod)

                state['position'] = fired_dir
                state['entry_price'] = c
                state['entry_idx'] = i
                state['days_in'] = 0
                state['lots'] = lots
                state['direction'] = fired_dir
                state['regime'] = regime
                state['vix'] = vix
                state['adx'] = adx_val
                state['rsi'] = rsi_val
                state['overlay_mod'] = overlay_mod
                self.open_positions[sig_id] = fired_dir
                daily_signals_fired += 1

            # ── EVALUATE STRUCTURAL SIGNALS ──
            if len(self.open_positions) < MAX_POSITIONS:
                structural_fired = self.evaluate_structural(market_data)
                for sf in structural_fired:
                    s_id = sf['signal_id']
                    s_dir = sf['direction']

                    # Conflict check
                    opposing = sum(1 for d in self.open_positions.values()
                                   if (d == 'LONG' and s_dir == 'SHORT') or
                                      (d == 'SHORT' and s_dir == 'LONG'))
                    if opposing > 0:
                        skipped_conflict += 1
                        continue

                    same_dir = sum(1 for d in self.open_positions.values() if d == s_dir)
                    if same_dir >= MAX_SAME_DIR:
                        continue
                    if len(self.open_positions) >= MAX_POSITIONS:
                        break

                    # Use simulate_exit with default structural rules
                    struct_rules = {
                        'stop_loss_pct': 0.02,
                        'take_profit_pct': 0.03,
                        'max_hold': 10,
                    }
                    exit_idx, exit_price, exit_reason, max_fav, max_adv = \
                        self.simulate_exit(df, i, s_dir, c, struct_rules)

                    if exit_idx > i:
                        days_held = exit_idx - i
                        self.record_trade(
                            sig_id=s_id, sig_type='STRUCTURAL',
                            entry_date=dates[i], exit_date=dates[exit_idx],
                            direction=s_dir, entry_price=c, exit_price=exit_price,
                            lots=self.compute_lots(c, 0.02, dt_date, overlay_mod),
                            lot_size=lot_size,
                            exit_reason=exit_reason, days_held=days_held,
                            regime=regime, vix=vix, adx=adx_val, rsi_val=rsi_val,
                            overlay_mod=overlay_mod, overlay_detail=overlay_detail,
                            max_fav=max_fav, max_adv=max_adv,
                            is_synthetic=True,
                            structural_conf=sf.get('confidence', 0.5),
                            structural_str=sf.get('strength', 'MODERATE'),
                        )
                        structural_signals_fired += 1

            # Daily P&L tracking
            day_pnl = self.equity - self.daily_equity.get(
                str(dates[max(0, i-1)])[:10], self.equity)
            self.daily_pnl[d_str] = day_pnl

            # Progress
            if i % 500 == 0:
                print(f"    Bar {i}/{n} | Equity: {self.equity:,.0f} | "
                      f"Trades: {len(self.trades)} | DD: {self.max_dd*100:.1f}%")

        print(f"\n  Backtest complete: {len(self.trades)} trades")
        print(f"  Daily signals fired: {daily_signals_fired}")
        print(f"  Structural signals fired: {structural_signals_fired}")
        print(f"  Skipped (conflict): {skipped_conflict}")

    # ----------------------------------------------------------
    # METRICS COMPUTATION
    # ----------------------------------------------------------
    def compute_metrics(self) -> Dict:
        """Compute comprehensive portfolio metrics."""
        if not self.trades:
            return {}

        net_pnls = [t.net_pnl for t in self.trades]
        gross_pnls = [t.gross_pnl for t in self.trades]
        wins = [p for p in net_pnls if p > 0]
        losses = [p for p in net_pnls if p <= 0]

        total_net = sum(net_pnls)
        total_gross = sum(gross_pnls)
        total_costs = sum(t.costs for t in self.trades)
        win_rate = len(wins) / len(net_pnls) * 100 if net_pnls else 0
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 1
        wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        expectancy = np.mean(net_pnls) if net_pnls else 0

        # Time-based metrics
        dates_list = sorted(self.daily_equity.keys())
        if len(dates_list) >= 2:
            first_date = pd.Timestamp(dates_list[0])
            last_date = pd.Timestamp(dates_list[-1])
            years = (last_date - first_date).days / 365.25
        else:
            years = 1

        final_equity = self.equity
        cagr = (final_equity / self.initial_capital) ** (1 / max(years, 0.5)) - 1
        calmar = cagr / self.max_dd if self.max_dd > 0 else 0

        # Sharpe from daily equity changes
        eq_series = pd.Series(self.daily_equity)
        if len(eq_series) > 1:
            daily_rets = eq_series.pct_change().dropna()
            sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)
                      if daily_rets.std() > 0 else 0)
            sortino_downside = daily_rets[daily_rets < 0].std()
            sortino = (daily_rets.mean() / sortino_downside * np.sqrt(252)
                       if sortino_downside > 0 else 0)
        else:
            sharpe = 0
            sortino = 0

        return {
            'total_trades': len(self.trades),
            'total_net_pnl': round(total_net),
            'total_gross_pnl': round(total_gross),
            'total_costs': round(total_costs),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(pf, 2),
            'avg_win': round(avg_win),
            'avg_loss': round(avg_loss),
            'wl_ratio': round(wl_ratio, 2),
            'expectancy': round(expectancy),
            'max_drawdown': round(self.max_dd * 100, 2),
            'cagr': round(cagr * 100, 2),
            'sharpe': round(sharpe, 2),
            'sortino': round(sortino, 2),
            'calmar': round(calmar, 2),
            'final_equity': round(final_equity),
            'years': round(years, 1),
            'trades_per_year': round(len(self.trades) / max(years, 0.5), 1),
            'avg_days_held': round(np.mean([t.days_held for t in self.trades]), 1),
            'cost_pct_of_gross': round(total_costs / max(abs(total_gross), 1) * 100, 2),
        }

    # ----------------------------------------------------------
    # REPORTING
    # ----------------------------------------------------------
    def print_report(self):
        """Print all 12 report sections."""
        metrics = self.compute_metrics()
        if not metrics:
            print("  NO TRADES — nothing to report.")
            return

        W = 95

        # ── SECTION 1: Portfolio Summary ──
        print("\n" + "=" * W)
        print("  SECTION 1: PORTFOLIO SUMMARY")
        print("=" * W)
        print(f"  {'Period':<30s} {metrics['years']:.1f} years")
        print(f"  {'Initial Capital':<30s} Rs {self.initial_capital:>15,.0f}")
        print(f"  {'Final Equity':<30s} Rs {metrics['final_equity']:>15,.0f}")
        print(f"  {'Total Net P&L':<30s} Rs {metrics['total_net_pnl']:>15,.0f}")
        print(f"  {'Total Costs':<30s} Rs {metrics['total_costs']:>15,.0f}")
        print(f"  {'CAGR':<30s} {metrics['cagr']:>15.2f}%")
        print(f"  {'Sharpe Ratio':<30s} {metrics['sharpe']:>15.2f}")
        print(f"  {'Sortino Ratio':<30s} {metrics['sortino']:>15.2f}")
        print(f"  {'Max Drawdown':<30s} {metrics['max_drawdown']:>15.2f}%")
        print(f"  {'Calmar Ratio':<30s} {metrics['calmar']:>15.2f}")
        print(f"  {'Profit Factor':<30s} {metrics['profit_factor']:>15.2f}")
        print(f"  {'Win Rate':<30s} {metrics['win_rate']:>15.1f}%")
        print(f"  {'W/L Ratio':<30s} {metrics['wl_ratio']:>15.2f}")
        print(f"  {'Expectancy (per trade)':<30s} Rs {metrics['expectancy']:>13,.0f}")
        print(f"  {'Total Trades':<30s} {metrics['total_trades']:>15d}")
        print(f"  {'Trades/Year':<30s} {metrics['trades_per_year']:>15.1f}")
        print(f"  {'Avg Days Held':<30s} {metrics['avg_days_held']:>15.1f}")
        print(f"  {'Costs as % of Gross':<30s} {metrics['cost_pct_of_gross']:>15.2f}%")

        # ── SECTION 2: Per-Signal Breakdown ──
        print("\n" + "=" * W)
        print("  SECTION 2: PER-SIGNAL BREAKDOWN")
        print("=" * W)
        sig_groups = defaultdict(list)
        for t in self.trades:
            sig_groups[t.signal_id].append(t)

        print(f"  {'Signal':<28s} {'Trades':>6s} {'WR%':>6s} {'PF':>6s} "
              f"{'Net PnL':>12s} {'Avg PnL':>10s} {'AvgDays':>7s} {'Type':>10s}")
        print(f"  {'-'*85}")

        for sig_id in sorted(sig_groups.keys()):
            trades = sig_groups[sig_id]
            pnls = [t.net_pnl for t in trades]
            w = [p for p in pnls if p > 0]
            l = [p for p in pnls if p <= 0]
            wr = len(w) / len(pnls) * 100 if pnls else 0
            pf_ = sum(w) / abs(sum(l)) if l and sum(l) != 0 else 0
            net = sum(pnls)
            avg = np.mean(pnls)
            avg_d = np.mean([t.days_held for t in trades])
            sig_type = trades[0].signal_type
            print(f"  {sig_id:<28s} {len(trades):>6d} {wr:>5.1f}% {pf_:>6.2f} "
                  f"Rs{net:>10,.0f} Rs{avg:>8,.0f} {avg_d:>6.1f} {sig_type:>10s}")

        # ── SECTION 3: Monthly P&L Table ──
        print("\n" + "=" * W)
        print("  SECTION 3: MONTHLY P&L TABLE")
        print("=" * W)
        monthly_pnl = defaultdict(float)
        for t in self.trades:
            key = t.exit_date[:7]
            monthly_pnl[key] += t.net_pnl

        months_sorted = sorted(monthly_pnl.keys())
        # Group by year
        year_months = defaultdict(dict)
        for m in months_sorted:
            yr = m[:4]
            mo = int(m[5:7])
            year_months[yr][mo] = monthly_pnl[m]

        header = "  Year  " + " ".join(f"{'M'+str(m):>8s}" for m in range(1, 13)) + f"  {'TOTAL':>10s}"
        print(header)
        print(f"  {'-'*len(header)}")

        for yr in sorted(year_months.keys()):
            vals = []
            for m in range(1, 13):
                v = year_months[yr].get(m, 0)
                vals.append(v)
            total = sum(vals)
            parts = [f"{yr}  "]
            for v in vals:
                if v == 0:
                    parts.append(f"{'--':>8s}")
                elif v > 0:
                    parts.append(f"{v:>8,.0f}")
                else:
                    parts.append(f"{v:>8,.0f}")
            parts.append(f"  {total:>10,.0f}")
            print("  " + " ".join(parts))

        # ── SECTION 4: Yearly P&L Table ──
        print("\n" + "=" * W)
        print("  SECTION 4: YEARLY P&L TABLE")
        print("=" * W)
        yearly_pnl = defaultdict(lambda: {'pnl': 0, 'trades': 0, 'wins': 0})
        for t in self.trades:
            yr = t.exit_date[:4]
            yearly_pnl[yr]['pnl'] += t.net_pnl
            yearly_pnl[yr]['trades'] += 1
            if t.net_pnl > 0:
                yearly_pnl[yr]['wins'] += 1

        print(f"  {'Year':<8s} {'Net PnL':>12s} {'Trades':>8s} {'WR%':>7s} {'Equity':>12s}")
        print(f"  {'-'*50}")
        running_eq = self.initial_capital
        for yr in sorted(yearly_pnl.keys()):
            d = yearly_pnl[yr]
            wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
            running_eq += d['pnl']
            print(f"  {yr:<8s} Rs{d['pnl']:>10,.0f} {d['trades']:>8d} {wr:>6.1f}% Rs{running_eq:>10,.0f}")

        # ── SECTION 5: Regime Performance ──
        print("\n" + "=" * W)
        print("  SECTION 5: REGIME PERFORMANCE")
        print("=" * W)
        regime_groups = defaultdict(list)
        for t in self.trades:
            regime_groups[t.regime].append(t)

        print(f"  {'Regime':<22s} {'Trades':>7s} {'WR%':>6s} {'PF':>6s} "
              f"{'Net PnL':>12s} {'Avg PnL':>10s}")
        print(f"  {'-'*65}")
        for reg in sorted(regime_groups.keys()):
            trades = regime_groups[reg]
            pnls = [t.net_pnl for t in trades]
            w = [p for p in pnls if p > 0]
            l = [p for p in pnls if p <= 0]
            wr = len(w) / len(pnls) * 100 if pnls else 0
            pf_ = sum(w) / abs(sum(l)) if l and sum(l) != 0 else 0
            net = sum(pnls)
            avg = np.mean(pnls)
            print(f"  {reg:<22s} {len(trades):>7d} {wr:>5.1f}% {pf_:>6.2f} "
                  f"Rs{net:>10,.0f} Rs{avg:>8,.0f}")

        # ── SECTION 6: Exit Reason Analysis ──
        print("\n" + "=" * W)
        print("  SECTION 6: EXIT REASON ANALYSIS")
        print("=" * W)
        exit_groups = defaultdict(list)
        for t in self.trades:
            exit_groups[t.exit_reason].append(t)

        print(f"  {'Exit Reason':<15s} {'Trades':>7s} {'WR%':>6s} {'Net PnL':>12s} {'Avg PnL':>10s}")
        print(f"  {'-'*55}")
        for reason in sorted(exit_groups.keys()):
            trades = exit_groups[reason]
            pnls = [t.net_pnl for t in trades]
            w = [p for p in pnls if p > 0]
            wr = len(w) / len(pnls) * 100 if pnls else 0
            net = sum(pnls)
            avg = np.mean(pnls)
            print(f"  {reason:<15s} {len(trades):>7d} {wr:>5.1f}% Rs{net:>10,.0f} Rs{avg:>8,.0f}")

        # ── SECTION 7: Day of Week Analysis ──
        print("\n" + "=" * W)
        print("  SECTION 7: DAY OF WEEK ANALYSIS")
        print("=" * W)
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        dow_groups = defaultdict(list)
        for t in self.trades:
            dow_groups[t.day_of_week].append(t)

        print(f"  {'Day':<12s} {'Trades':>7s} {'WR%':>6s} {'Net PnL':>12s} {'Avg PnL':>10s}")
        print(f"  {'-'*50}")
        for d in range(5):
            trades = dow_groups.get(d, [])
            if not trades:
                continue
            pnls = [t.net_pnl for t in trades]
            w = [p for p in pnls if p > 0]
            wr = len(w) / len(pnls) * 100 if pnls else 0
            net = sum(pnls)
            avg = np.mean(pnls)
            name = dow_names[d] if d < len(dow_names) else f'Day{d}'
            print(f"  {name:<12s} {len(trades):>7d} {wr:>5.1f}% Rs{net:>10,.0f} Rs{avg:>8,.0f}")

        # ── SECTION 8: Conviction / Overlay Impact ──
        print("\n" + "=" * W)
        print("  SECTION 8: CONVICTION / OVERLAY IMPACT")
        print("=" * W)
        high_mod = [t for t in self.trades if t.overlay_modifier > 1.10]
        low_mod = [t for t in self.trades if t.overlay_modifier < 0.90]
        neutral_mod = [t for t in self.trades if 0.90 <= t.overlay_modifier <= 1.10]

        for label, subset in [('High (>1.10)', high_mod), ('Neutral (0.90-1.10)', neutral_mod),
                               ('Low (<0.90)', low_mod)]:
            if not subset:
                pnls, wr_, net_ = [], 0, 0
            else:
                pnls = [t.net_pnl for t in subset]
                w = [p for p in pnls if p > 0]
                wr_ = len(w) / len(pnls) * 100
                net_ = sum(pnls)
            print(f"  {label:<22s} Trades: {len(subset):>5d}  WR: {wr_:>5.1f}%  "
                  f"Net: Rs{net_:>10,.0f}")

        # ── SECTION 9: Drawdown Periods (>5%) ──
        print("\n" + "=" * W)
        print("  SECTION 9: DRAWDOWN PERIODS (>5%)")
        print("=" * W)
        eq_series = pd.Series(self.daily_equity)
        if len(eq_series) > 0:
            eq_series = eq_series.sort_index()
            peak_s = eq_series.cummax()
            dd_s = (eq_series - peak_s) / peak_s * 100

            in_dd = False
            dd_start = None
            dd_periods = []
            for dt_str, dd_val in dd_s.items():
                if dd_val < -5 and not in_dd:
                    in_dd = True
                    dd_start = dt_str
                elif dd_val >= -1 and in_dd:
                    dd_periods.append((dd_start, dt_str, dd_s.loc[dd_start:dt_str].min()))
                    in_dd = False

            if in_dd and dd_start:
                dd_periods.append((dd_start, dd_s.index[-1], dd_s.loc[dd_start:].min()))

            if dd_periods:
                print(f"  {'Start':<12s} {'End':<12s} {'Max DD':>8s}")
                print(f"  {'-'*35}")
                for start, end, max_dd_val in dd_periods[:15]:
                    print(f"  {start:<12s} {end:<12s} {max_dd_val:>7.1f}%")
            else:
                print("  No drawdown periods > 5% detected.")
        else:
            print("  No equity data available.")

        # ── SECTION 10: Capital Sensitivity ──
        print("\n" + "=" * W)
        print("  SECTION 10: CAPITAL SENSITIVITY (5L / 10L / 30L)")
        print("=" * W)
        for cap_label, cap_mult in [('5L', 0.5), ('10L', 1.0), ('30L', 3.0)]:
            cap = self.initial_capital * cap_mult
            total_net = sum(t.net_pnl * cap_mult for t in self.trades)
            final_eq = cap + total_net
            cagr_val = (final_eq / cap) ** (1 / max(metrics['years'], 0.5)) - 1
            print(f"  Capital Rs {cap:>10,.0f} -> Final Rs {final_eq:>12,.0f}  "
                  f"CAGR: {cagr_val*100:>6.2f}%  Net: Rs {total_net:>12,.0f}")

        # ── SECTION 11: Top 20 Best / Worst Trades ──
        print("\n" + "=" * W)
        print("  SECTION 11: TOP 20 BEST / WORST TRADES")
        print("=" * W)
        sorted_trades = sorted(self.trades, key=lambda t: t.net_pnl, reverse=True)

        print("  -- TOP 20 BEST --")
        print(f"  {'#':>3s} {'Signal':<25s} {'Dir':>5s} {'Entry':>10s} {'Exit':>10s} "
              f"{'Net PnL':>10s} {'Days':>5s} {'Regime':<14s}")
        for i, t in enumerate(sorted_trades[:20]):
            print(f"  {i+1:>3d} {t.signal_id:<25s} {t.direction:>5s} "
                  f"{t.entry_date:>10s} {t.exit_date:>10s} "
                  f"Rs{t.net_pnl:>8,.0f} {t.days_held:>5d} {t.regime:<14s}")

        print("\n  -- TOP 20 WORST --")
        for i, t in enumerate(sorted_trades[-20:]):
            rank = len(sorted_trades) - 19 + i
            print(f"  {rank:>3d} {t.signal_id:<25s} {t.direction:>5s} "
                  f"{t.entry_date:>10s} {t.exit_date:>10s} "
                  f"Rs{t.net_pnl:>8,.0f} {t.days_held:>5d} {t.regime:<14s}")

        # ── SECTION 12: Per-Signal Walk-Forward (252/63 windows) ──
        print("\n" + "=" * W)
        print("  SECTION 12: PER-SIGNAL WALK-FORWARD (252 train / 63 test)")
        print("=" * W)
        self._walk_forward_analysis(sig_groups)

    # ----------------------------------------------------------
    # WALK-FORWARD ANALYSIS
    # ----------------------------------------------------------
    def _walk_forward_analysis(self, sig_groups: Dict[str, List[Trade]]):
        """Run walk-forward validation per signal (252 train / 63 test)."""
        TRAIN_DAYS = 252
        TEST_DAYS = 63

        print(f"  {'Signal':<28s} {'Windows':>8s} {'IS WR%':>7s} {'OOS WR%':>8s} "
              f"{'IS PF':>6s} {'OOS PF':>7s} {'Degrade':>8s} {'Verdict':>8s}")
        print(f"  {'-'*85}")

        for sig_id in sorted(sig_groups.keys()):
            trades = sorted(sig_groups[sig_id], key=lambda t: t.entry_date)
            if len(trades) < 10:
                print(f"  {sig_id:<28s} {'<10 trades, skipped':>55s}")
                continue

            # Convert to date-indexed
            trade_dates = [pd.Timestamp(t.entry_date) for t in trades]
            if not trade_dates:
                continue

            min_date = min(trade_dates)
            max_date = max(trade_dates)
            total_days = (max_date - min_date).days

            if total_days < TRAIN_DAYS + TEST_DAYS:
                print(f"  {sig_id:<28s} {'insufficient date range':>55s}")
                continue

            windows = 0
            is_wrs = []
            oos_wrs = []
            is_pfs = []
            oos_pfs = []

            current = min_date
            while current + timedelta(days=TRAIN_DAYS + TEST_DAYS) <= max_date:
                train_end = current + timedelta(days=TRAIN_DAYS)
                test_end = train_end + timedelta(days=TEST_DAYS)

                train_trades = [t for t in trades
                                if current <= pd.Timestamp(t.entry_date) < train_end]
                test_trades = [t for t in trades
                               if train_end <= pd.Timestamp(t.entry_date) < test_end]

                if len(train_trades) >= 3 and len(test_trades) >= 1:
                    # In-sample metrics
                    is_pnls = [t.net_pnl for t in train_trades]
                    is_wins = [p for p in is_pnls if p > 0]
                    is_losses = [p for p in is_pnls if p <= 0]
                    is_wr = len(is_wins) / len(is_pnls) * 100
                    is_pf = sum(is_wins) / abs(sum(is_losses)) if is_losses and sum(is_losses) != 0 else 0

                    # Out-of-sample metrics
                    oos_pnls = [t.net_pnl for t in test_trades]
                    oos_wins = [p for p in oos_pnls if p > 0]
                    oos_losses = [p for p in oos_pnls if p <= 0]
                    oos_wr = len(oos_wins) / len(oos_pnls) * 100
                    oos_pf = sum(oos_wins) / abs(sum(oos_losses)) if oos_losses and sum(oos_losses) != 0 else 0

                    is_wrs.append(is_wr)
                    oos_wrs.append(oos_wr)
                    is_pfs.append(is_pf)
                    oos_pfs.append(oos_pf)
                    windows += 1

                current += timedelta(days=TEST_DAYS)

            if windows == 0:
                print(f"  {sig_id:<28s} {'no valid windows':>55s}")
                continue

            avg_is_wr = np.mean(is_wrs)
            avg_oos_wr = np.mean(oos_wrs)
            avg_is_pf = np.mean(is_pfs)
            avg_oos_pf = np.mean(oos_pfs)
            degradation = (avg_is_wr - avg_oos_wr) / avg_is_wr * 100 if avg_is_wr > 0 else 0

            if avg_oos_wr > 50 and avg_oos_pf > 1.0 and degradation < 25:
                verdict = 'PASS'
            elif avg_oos_wr > 45 and avg_oos_pf > 0.8:
                verdict = 'MARGINAL'
            else:
                verdict = 'FAIL'

            print(f"  {sig_id:<28s} {windows:>8d} {avg_is_wr:>6.1f}% {avg_oos_wr:>7.1f}% "
                  f"{avg_is_pf:>6.2f} {avg_oos_pf:>7.2f} {degradation:>7.1f}% {verdict:>8s}")

    # ----------------------------------------------------------
    # EXPORT CSV
    # ----------------------------------------------------------
    def export_csv(self):
        """Export trades and monthly summary to CSV."""
        output_dir = os.path.dirname(os.path.abspath(__file__))

        # All trades detail
        trades_path = os.path.join(output_dir, 'all_trades_detail.csv')
        trades_df = pd.DataFrame([asdict(t) for t in self.trades])
        trades_df.to_csv(trades_path, index=False)
        print(f"\n  Exported: {trades_path} ({len(trades_df)} rows)")

        # Monthly summary
        monthly_path = os.path.join(output_dir, 'monthly_summary.csv')
        monthly_data = defaultdict(lambda: {'trades': 0, 'wins': 0, 'net_pnl': 0,
                                            'gross_pnl': 0, 'costs': 0})
        for t in self.trades:
            key = t.exit_date[:7]
            monthly_data[key]['trades'] += 1
            if t.net_pnl > 0:
                monthly_data[key]['wins'] += 1
            monthly_data[key]['net_pnl'] += t.net_pnl
            monthly_data[key]['gross_pnl'] += t.gross_pnl
            monthly_data[key]['costs'] += t.costs

        rows = []
        for month in sorted(monthly_data.keys()):
            d = monthly_data[month]
            wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
            rows.append({
                'month': month,
                'trades': d['trades'],
                'wins': d['wins'],
                'win_rate': round(wr, 1),
                'net_pnl': round(d['net_pnl']),
                'gross_pnl': round(d['gross_pnl']),
                'costs': round(d['costs']),
            })
        pd.DataFrame(rows).to_csv(monthly_path, index=False)
        print(f"  Exported: {monthly_path} ({len(rows)} rows)")


# ================================================================
# MAIN
# ================================================================

def main():
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s %(levelname)-8s %(message)s',
    )

    t0 = time_mod.perf_counter()

    print("=" * 95)
    print("  COMPREHENSIVE SYSTEM BACKTEST — DEFINITIVE")
    print("  All signal layers: 7 daily + 13 structural + 4 overlays")
    print("  Data: nifty_daily (2015-2026) from PostgreSQL")
    print("  Capital: Rs 10,00,000 | Lot-based sizing | Realistic costs")
    print("  Costs: Rs 20/order x 1.18 GST + STT 0.0125% + Exch 0.0495% + slippage")
    print("=" * 95)

    bt = ComprehensiveBacktest(capital=INITIAL_CAPITAL)

    # Step 1: Load data
    df = bt.load_data()

    # Step 2: Load signal classes
    bt.load_signals()

    # Step 3: Run backtest
    bt.run(df)

    # Step 4: Print comprehensive report
    bt.print_report()

    # Step 5: Export CSVs
    bt.export_csv()

    elapsed = time_mod.perf_counter() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print("=" * 95)


if __name__ == '__main__':
    main()

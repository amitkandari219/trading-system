"""
Intraday Orchestrator — full-session 5-min bar processor for NIFTY + BANKNIFTY.

Integrates:
  - Signal generators: KAUFMAN_BB_MR, GUJRAL_RANGE, ORB_TUNED, gamma signals,
    BankNifty signals (BN_KAUFMAN_BB_MR, BN_GUJRAL_RANGE, L9 suite)
  - Regime filter: blocks/scales signals by real-time volatility regime
  - Risk: DailyLossLimiter (3-tier circuit breaker), CompoundSizer (lot scaling)
  - Execution: OptionsExecutor -> ExecutionEngine -> KiteBridge -> FillMonitor
  - Monitoring: Telegram alerts on entries, exits, regime changes, errors

Modes:
  PAPER  — full logic, simulated fills (default)
  LIVE   — real Kite API orders (requires EXECUTION_MODE=LIVE env var)
  REPLAY — replay historical bars from DB

Usage:
    venv/bin/python3 -m paper_trading.intraday_runner
    venv/bin/python3 -m paper_trading.intraday_runner --live
    venv/bin/python3 -m paper_trading.intraday_runner --replay 2026-03-20
    venv/bin/python3 -m paper_trading.intraday_runner --status
    venv/bin/python3 -m paper_trading.intraday_runner --dry-run
    venv/bin/python3 -m paper_trading.intraday_runner --instrument NIFTY --no-banknifty
"""

import argparse
import logging
import os
import signal as signal_module
import sys
import time as time_mod
import traceback
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

from backtest.indicators import sma, ema, rsi, atr, adx, bollinger_bands, stochastic
from config.settings import (
    DATABASE_DSN, MAX_POSITIONS, NIFTY_LOT_SIZE, TOTAL_CAPITAL,
)
from data.intraday_indicators import (
    add_intraday_indicators, session_vwap, opening_range, session_features,
)
from data.kite_auth import get_kite
from execution.options_executor import OptionsExecutor
from execution.spread_builder import SpreadBuilder, SpreadOrder, GAMMA_SIGNAL_IDS
from execution.spread_executor import SpreadExecutor
from execution.spread_monitor import SpreadMonitor, ExitSignal
from monitoring.telegram_alerter import TelegramAlerter
from paper_trading.intraday_shadows import INTRADAY_SHADOW_SIGNALS
from risk.compound_sizer import CompoundSizer
from risk.daily_loss_limiter import DailyLossLimiter
from signals.banknifty_signals import BankNiftySignalComputer
from signals.expiry_day_detector import ExpiryDayDetector
from data.vix_streamer import VIXStreamer
from signals.dynamic_regime import DynamicRegimeManager
from signals.regime_filter import IntradayRegimeFilter, SIGNAL_REGIME_MATRIX
from signals.fii_overlay import FIIOverlay
from signals.calendar_overlay import CalendarOverlay
from paper_trading.decay_auto_manager import DecayAutoManager
from risk.behavioral_overlay import (
    BehavioralOverlay, OverlayContext, PositionContext, TradeRecord,
)
from signals.iv_rank_filter import IVRankFilter
from signals.entry_window_optimizer import EntryWindowOptimizer
from execution.crash_recovery import CrashRecovery
from risk.india_risk_guard import IndiaRiskGuard

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

BANKNIFTY_LOT_SIZE = 15

SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
PREMARKET_TIME = time(9, 10)
FORCE_EXIT_TIME = time(15, 20)
RECONCILE_TIME = time(15, 35)
BAR_INTERVAL_SECONDS = 300  # 5 minutes

# Kite instrument tokens (NSE index futures)
KITE_TOKENS = {
    'NIFTY': 256265,
    'BANKNIFTY': 260105,
}

# Max positions across both instruments
MAX_TOTAL_POSITIONS = MAX_POSITIONS  # 4

# Kite reconnect policy
KITE_MAX_RETRIES = 3
KITE_RETRY_DELAY = 5  # seconds

# ORB tuned config from intraday_shadows
ORB_TUNED_CONFIG = INTRADAY_SHADOW_SIGNALS.get('ORB_TUNED', {}).get('params', {})


# ═══════════════════════════════════════════════════════════════
# PREMIUM ESTIMATION
# ═══════════════════════════════════════════════════════════════

def estimate_premium(price: float, atr_val: float) -> float:
    """
    Estimate ATM option premium from underlying price and ATR.
    Returns a value between 50 and 500.
    """
    return max(50, min(120 + 0.8 * atr_val, 500))


# ═══════════════════════════════════════════════════════════════
# INTRADAY ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

class IntradayOrchestrator:
    """
    Full-session intraday orchestrator for NIFTY and BANKNIFTY.

    Runs a 5-min bar loop from 9:15 to 15:25, checking signals,
    filtering by regime, sizing via compound sizer, and executing
    via the options executor / execution engine pipeline.
    """

    def __init__(self, mode: str = 'PAPER',
                 instruments: Optional[List[str]] = None):
        """
        Args:
            mode: 'PAPER' or 'LIVE'
            instruments: list of instruments to trade, default ['NIFTY', 'BANKNIFTY']
        """
        self.mode = mode.upper()
        self.instruments = instruments or ['NIFTY', 'BANKNIFTY']
        self.dry_run = False
        self._shutdown_requested = False

        # Set EXECUTION_MODE env var so downstream components respect it
        os.environ['EXECUTION_MODE'] = self.mode

        # ── Database ──
        self.conn = psycopg2.connect(DATABASE_DSN)
        self.conn.autocommit = False

        # ── Kite ──
        self.kite = None
        if self.mode == 'LIVE':
            self.kite = self._connect_kite()

        # ── Telegram ──
        self.alerter = self._init_alerter()

        # ── Risk components ──
        self.loss_limiter = DailyLossLimiter(equity=TOTAL_CAPITAL)
        self.sizer = CompoundSizer(initial_equity=TOTAL_CAPITAL)

        # ── Signal components ──
        self.regime_filter = IntradayRegimeFilter()
        self.expiry_detector = ExpiryDayDetector()
        self.bn_signal_computer = BankNiftySignalComputer()

        # ── FII overlay (overnight signals modify sizing) ──
        self.fii_overlay = FIIOverlay(db_conn=self.conn)
        self._fii_bias: Optional[Dict] = None

        # ── Calendar overlay (event-driven sizing) ──
        self.calendar_overlay = CalendarOverlay(db_conn=self.conn)
        self._calendar_ctx: Optional[Dict] = None

        # ── Decay auto-manager (size overrides from decay detection) ──
        self.decay_manager = DecayAutoManager(conn=self.conn)

        # ── Behavioral overlay (Layer 7 — Kahneman bias corrections) ──
        self.behavioral_overlay = BehavioralOverlay()

        # ── IV rank filter (Layer 8 — options strategy gating) ──
        self.iv_rank_filter = IVRankFilter(db_conn=self.conn)

        # ── Entry window optimizer (Layer 9 — time-of-day gating) ──
        self.entry_window_optimizer = EntryWindowOptimizer()

        # ── India risk guard (circuit breaker proximity, consecutive losses) ──
        self.india_risk_guard = IndiaRiskGuard()

        # ── Crash recovery (check for orphaned orders from prior session) ──
        self._crash_recovery = CrashRecovery(
            kite=self.kite, db_conn=self.conn, alerter=self.alerter
        )

        # ── VIX streaming & dynamic regime ──
        self.vix_streamer: Optional[VIXStreamer] = None
        self.dynamic_regime = DynamicRegimeManager(
            self.regime_filter, telegram_alerter=self.alerter
        )
        self.static_vix: Optional[float] = None  # PAPER mode fallback

        if self.mode == 'LIVE' and self.kite is not None:
            self.vix_streamer = VIXStreamer(
                kite=self.kite,
                on_update_callback=self._on_vix_update,
            )
        else:
            # PAPER mode: VIXStreamer with no kite reads from DB
            self.vix_streamer = VIXStreamer(
                kite=None,
                on_update_callback=self._on_vix_update,
            )

        # ── Execution ──
        self.options_executor = OptionsExecutor(
            equity=TOTAL_CAPITAL,
            kite=self.kite,
            paper_mode=(self.mode == 'PAPER'),
        )

        # ── Spread execution components ──
        self.spread_executor = SpreadExecutor(
            kite_bridge=None,       # set later if LIVE
            fill_monitor=None,      # set later if LIVE
            paper_mode=(self.mode == 'PAPER'),
            alerter=self.alerter,
        )
        self.spread_monitor = SpreadMonitor()

        # ── Per-instrument session state ──
        self.session_state: Dict[str, Dict] = {}
        for inst in self.instruments:
            self.session_state[inst] = {
                'bars': [],               # list of bar dicts
                'bars_df': pd.DataFrame(),  # DataFrame for indicators
                'vwap': None,
                'or_high': None,
                'or_low': None,
                'positions': [],          # active positions
                'daily_atr': None,
                'daily_vix': None,
                'daily_adx': None,
                'expiry_info': None,
                'bar_count': 0,
                'signals_fired': 0,
                'signals_blocked': 0,
                'orders_placed': 0,
                'exits_count': 0,
                'daily_pnl': 0.0,
            }

        # ── Session-level state ──
        self._session_date = None
        self._total_positions = 0
        self._bar_summaries: List[str] = []

        # ── Signal handlers for graceful shutdown ──
        signal_module.signal(signal_module.SIGINT, self._handle_shutdown)
        signal_module.signal(signal_module.SIGTERM, self._handle_shutdown)

        logger.info(
            f"IntradayOrchestrator initialized: mode={self.mode}, "
            f"instruments={self.instruments}"
        )

    # ══════════════════════════════════════════════════════════
    # PUBLIC: Main session loop
    # ══════════════════════════════════════════════════════════

    def run_session(self, session_date: Optional[date] = None):
        """
        Run a complete trading session.

        Args:
            session_date: trading date (defaults to today)
        """
        self._session_date = session_date or date.today()
        logger.info(f"{'='*70}")
        logger.info(f"  SESSION START: {self._session_date} | mode={self.mode}")
        logger.info(f"  Instruments: {self.instruments}")
        logger.info(f"{'='*70}")

        try:
            # ── CRASH RECOVERY (before anything else) ──
            recovery = self._crash_recovery.recover_on_startup()
            if recovery.get('recovered', 0) > 0 or recovery.get('cancelled', 0) > 0:
                logger.warning(
                    f"Crash recovery: {recovery.get('recovered', 0)} recovered, "
                    f"{recovery.get('cancelled', 0)} cancelled"
                )

            # ── PRE-MARKET SETUP (9:10) ──
            self._premarket_setup()

            # ── BAR LOOP (9:15 - 15:25) ──
            self._run_bar_loop()

            # ── FORCE EXIT (15:20) ──
            self._force_exit_all()

            # ── RECONCILIATION (15:35) ──
            self._reconcile()

            # ── SESSION SUMMARY ──
            self._session_summary()

        except Exception as e:
            logger.critical(f"Session error: {e}")
            logger.critical(traceback.format_exc())
            self._alert('CRITICAL', f"Session crashed: {e}")
            self._emergency_exit()
        finally:
            logger.info(f"Session {self._session_date} ended")

    def run_replay(self, replay_date: date):
        """
        Replay historical bars from DB for a given date.
        """
        self._session_date = replay_date
        logger.info(f"REPLAY: {replay_date}")

        self._premarket_setup()

        for inst in self.instruments:
            bars = self._load_replay_bars(inst, replay_date)
            if not bars:
                logger.warning(f"No bars for {inst} on {replay_date}")
                continue

            logger.info(f"Replaying {len(bars)} bars for {inst}")
            prev_bar = None

            for i, bar_dict in enumerate(bars):
                bar_time = bar_dict['datetime']

                # Skip bars outside trading hours
                if bar_time.time() < SESSION_OPEN or bar_time.time() > time(15, 25):
                    continue

                # Append bar and compute indicators
                state = self.session_state[inst]
                state['bars'].append(bar_dict)
                state['bar_count'] += 1
                self._update_indicators(inst)

                bars_df = state['bars_df']
                if bars_df.empty:
                    continue

                current_bar = bars_df.iloc[-1]
                prev_bar_s = bars_df.iloc[-2] if len(bars_df) > 1 else None

                # Check exits
                self._check_exits(inst, current_bar)

                # Check entries
                if bar_time.time() < FORCE_EXIT_TIME:
                    expiry_info = state['expiry_info'] or {}
                    signals = self._check_signals(
                        inst, current_bar, prev_bar_s, bars_df, expiry_info
                    )
                    for sig in signals:
                        self._filter_and_size(sig, inst, current_bar)

                self._log_bar_summary(inst, current_bar)

        # Force exit remaining
        self._force_exit_all()
        self._session_summary()

    # ══════════════════════════════════════════════════════════
    # PRE-MARKET SETUP
    # ══════════════════════════════════════════════════════════

    def _premarket_setup(self):
        """Load daily context, init regime, cache symbols."""
        logger.info("Pre-market setup...")

        for inst in self.instruments:
            state = self.session_state[inst]

            # Load daily ATR, VIX, ADX from DB
            ctx = self._load_daily_context(inst)
            state['daily_atr'] = ctx['daily_atr']
            state['daily_vix'] = ctx['daily_vix']
            state['daily_adx'] = ctx['daily_adx']

            # Feed ATR to regime filter
            self.regime_filter.update_daily_atr(ctx['daily_atr'])

            # Expiry info
            state['expiry_info'] = self.expiry_detector.get_expiry_info(
                self._session_date
            )

            # Reset session state
            state['bars'] = []
            state['bars_df'] = pd.DataFrame()
            state['positions'] = []
            state['bar_count'] = 0
            state['signals_fired'] = 0
            state['signals_blocked'] = 0
            state['orders_placed'] = 0
            state['exits_count'] = 0
            state['daily_pnl'] = 0.0

            logger.info(
                f"  {inst}: ATR={ctx['daily_atr']:.0f} VIX={ctx['daily_vix']:.1f} "
                f"ADX={ctx['daily_adx']:.0f} | "
                f"expiry={'YES' if state['expiry_info'].get('is_any_expiry') else 'no'}"
            )

        # Reset daily loss limiter
        self.loss_limiter.can_trade(self._session_date)
        self._total_positions = 0

        # ── Start VIX streamer & set initial regime ──
        if self.vix_streamer is not None:
            self.vix_streamer.start()

        # Use first instrument's daily VIX as initial value
        initial_vix = self.session_state[self.instruments[0]].get('daily_vix', 15.0)
        self.static_vix = initial_vix

        # Get live VIX if available, else use static
        live_vix = None
        if self.vix_streamer is not None:
            live_vix = self.vix_streamer.get_vix()
        vix_for_regime = live_vix if live_vix is not None else initial_vix

        regime_result = self.dynamic_regime.update_regime(
            vix_for_regime, datetime.now()
        )
        logger.info(
            f"  Initial regime: {regime_result['regime']} "
            f"(VIX={vix_for_regime:.2f}, size_factor={regime_result['size_factor']:.2f})"
        )

        # ── FII overnight signals ──
        fii_signals = self.fii_overlay.load_active_signals(self._session_date)
        self._fii_bias = self.fii_overlay.get_direction_bias(self._session_date)

        fii_msg = ""
        if fii_signals:
            fii_parts = []
            for sig in fii_signals:
                fii_parts.append(
                    f"{sig['signal_id']} ({sig['direction']}, "
                    f"{sig['confidence']:.0%}): {sig['pattern_name']}"
                )
            fii_msg = "\nFII signals:\n" + "\n".join(fii_parts)
            logger.info(
                f"  FII bias: {self._fii_bias['direction']} "
                f"({self._fii_bias['confidence']:.0%}) — "
                f"{len(fii_signals)} active signal(s)"
            )
        else:
            fii_msg = "\nFII signals: NONE"
            logger.info("  FII bias: NEUTRAL (no active signals)")

        # ── Calendar overlay ──
        self._calendar_ctx = self.calendar_overlay.get_daily_context(self._session_date)
        cal_msg = ""
        if self._calendar_ctx.get('block_new_entries'):
            cal_msg = "\nCALENDAR: ENTRIES BLOCKED (event day)"
            logger.warning(
                f"  Calendar: BLOCK new entries — "
                f"events={self._calendar_ctx.get('events_active', [])}"
            )
        elif self._calendar_ctx.get('events_active'):
            events = self._calendar_ctx['events_active']
            mod = self._calendar_ctx.get('composite_modifier', 1.0)
            cal_msg = f"\nCalendar: {', '.join(events)} (modifier={mod:.2f}x)"
            logger.info(
                f"  Calendar events: {events} modifier={mod:.2f}x "
                f"SL_mult={self._calendar_ctx.get('expiry_sl_multiplier', 1.0):.2f}x"
            )
        else:
            logger.info("  Calendar: no events")

        self._alert(
            'INFO',
            f"Session {self._session_date} pre-market complete.\n"
            f"Instruments: {', '.join(self.instruments)}\n"
            f"Regime: {regime_result['regime']} (VIX={vix_for_regime:.1f})\n"
            + '\n'.join(
                f"{inst}: ATR={self.session_state[inst]['daily_atr']:.0f} "
                f"VIX={self.session_state[inst]['daily_vix']:.1f}"
                for inst in self.instruments
            )
            + fii_msg
            + cal_msg
        )

    def _load_daily_context(self, instrument: str) -> Dict:
        """Load daily ATR, VIX, ADX from DB."""
        table = 'nifty_daily'  # use nifty daily for both (VIX is market-wide)
        cur = self.conn.cursor()

        try:
            cur.execute("""
                SELECT india_vix, close FROM nifty_daily
                WHERE date <= %s ORDER BY date DESC LIMIT 1
            """, (self._session_date,))
            row = cur.fetchone()
            vix = float(row[0]) if row and row[0] else 15.0
            close = float(row[1]) if row and row[1] else 25000.0

            # Compute ATR from recent daily bars
            cur.execute("""
                SELECT high, low, close FROM nifty_daily
                WHERE date <= %s ORDER BY date DESC LIMIT 15
            """, (self._session_date,))
            rows = cur.fetchall()

            if len(rows) >= 2:
                trs = []
                for i in range(1, len(rows)):
                    h, l, c_prev = float(rows[i - 1][0]), float(rows[i - 1][1]), float(rows[i - 1][2])
                    c = float(rows[i][2])
                    trs.append(max(h - l, abs(h - c), abs(l - c)))
                atr_val = sum(trs) / len(trs) if trs else close * 0.015
            else:
                atr_val = close * 0.015

            return {
                'daily_atr': atr_val,
                'daily_vix': vix,
                'daily_adx': 20.0,  # will be refined from intraday bars
            }
        except Exception as e:
            logger.error(f"Failed to load daily context: {e}")
            self.conn.rollback()
            return {'daily_atr': 300.0, 'daily_vix': 15.0, 'daily_adx': 20.0}

    # ══════════════════════════════════════════════════════════
    # BAR LOOP
    # ══════════════════════════════════════════════════════════

    def _run_bar_loop(self):
        """Main loop: fetch bar every 5 min, process signals."""
        logger.info("Entering bar loop (9:15 - 15:25)...")

        last_bar_times: Dict[str, Optional[datetime]] = {
            inst: None for inst in self.instruments
        }

        while not self._shutdown_requested:
            now = datetime.now()
            current_time = now.time()

            # Wait for market open
            if current_time < SESSION_OPEN:
                time_mod.sleep(10)
                continue

            # Exit after 15:25 (leave 5 min for force exit)
            if current_time > time(15, 25):
                logger.info("Bar loop ended (15:25)")
                break

            # ── Sleep until next 5-min bar boundary + 15s settle ──
            # Align to 5-min boundaries (9:15, 9:20, 9:25, ...) so we
            # always query right after a bar closes, not at random offsets.
            minutes_since_open = (now.hour * 60 + now.minute) - (9 * 60 + 15)
            minutes_into_bar = minutes_since_open % 5
            seconds_into_bar = minutes_into_bar * 60 + now.second

            # Bar interval is 300s; we want to wake up 15s after bar close
            # to give Kite time to finalize the candle.
            target_offset = 315  # 5min + 15s settle
            sleep_time = target_offset - seconds_into_bar
            if sleep_time <= 5:
                # We're already past the settle window; fetch now and sleep
                # a full cycle next time. Don't add a full 300s -- just
                # a short wait so we don't tight-loop.
                pass  # fall through to fetch
            else:
                # Sleep in 1-second increments to check shutdown flag
                for _ in range(int(sleep_time)):
                    if self._shutdown_requested:
                        break
                    time_mod.sleep(1)
                if self._shutdown_requested:
                    break
                now = datetime.now()
                current_time = now.time()

            # ── VIX update & regime check (once per bar cycle) ──
            vix_now = None
            if self.vix_streamer is not None:
                vix_now = self.vix_streamer.get_vix()
            if vix_now is None:
                vix_now = self.static_vix

            if vix_now is not None:
                regime_result = self.dynamic_regime.update_regime(
                    vix_now, now
                )
                if regime_result['changed']:
                    logger.info(
                        f"REGIME CHANGE: {regime_result['old_regime']} -> "
                        f"{regime_result['regime']} "
                        f"(VIX={vix_now:.2f})"
                    )

                # Check emergency exit
                if self.dynamic_regime.should_emergency_exit(vix_now):
                    logger.critical(
                        f"EMERGENCY EXIT triggered: VIX spike "
                        f"(VIX={vix_now:.2f})"
                    )
                    self._alert(
                        'EMERGENCY',
                        f"VIX spike emergency exit triggered!\n"
                        f"VIX={vix_now:.2f}\n"
                        f"Closing all positions."
                    )
                    self._emergency_exit()
                    break

            for inst in self.instruments:
                try:
                    # In LIVE mode, fetch ALL bars from session open so
                    # we recover any bars missed due to slow cycles or
                    # API hiccups.  In PAPER mode, fall back to single-bar
                    # DB fetch (the DB already has all bars).
                    if self.mode == 'LIVE' and self.kite is not None:
                        all_bars = self._fetch_bars_kite(inst, now)
                    else:
                        single = self._fetch_bar_db(inst, now)
                        all_bars = [single] if single else []

                    for bar_dict in all_bars:
                        bar_time = bar_dict['datetime']

                        # Skip already-processed bars
                        if (last_bar_times[inst] is not None
                                and bar_time <= last_bar_times[inst]):
                            continue

                        last_bar_times[inst] = bar_time
                        state = self.session_state[inst]
                        state['bars'].append(bar_dict)
                        state['bar_count'] += 1

                        # Persist bar to intraday_bars table
                        self._persist_bar(inst, bar_dict)

                        # Update indicators
                        self._update_indicators(inst)

                        bars_df = state['bars_df']
                        if bars_df.empty or len(bars_df) < 2:
                            continue

                        current_bar = bars_df.iloc[-1]
                        prev_bar = bars_df.iloc[-2] if len(bars_df) > 1 else None

                        # Check exits first
                        self._check_exits(inst, current_bar)

                        # Check entries (only before 15:10)
                        if current_time < time(15, 10):
                            expiry_info = state['expiry_info'] or {}
                            signals = self._check_signals(
                                inst, current_bar, prev_bar, bars_df, expiry_info
                            )
                            for sig in signals:
                                self._filter_and_size(sig, inst, current_bar)

                        self._log_bar_summary(inst, current_bar)

                except Exception as e:
                    logger.error(f"Error processing {inst}: {e}")
                    logger.debug(traceback.format_exc())

    # ══════════════════════════════════════════════════════════
    # BAR FETCHING
    # ══════════════════════════════════════════════════════════

    def _fetch_bar(self, instrument: str,
                   timestamp: datetime) -> Optional[Dict]:
        """
        Fetch the latest 5-min bar.

        LIVE mode: Kite historical data API
        PAPER mode: read from DB table
        """
        if self.mode == 'LIVE' and self.kite is not None:
            return self._fetch_bar_kite(instrument, timestamp)
        else:
            return self._fetch_bar_db(instrument, timestamp)

    def _fetch_bars_kite(self, instrument: str,
                         timestamp: datetime) -> List[Dict]:
        """Fetch all bars from session open to now (Kite API with retry).

        Returns a list of bar dicts, oldest first.  The caller uses
        last_bar_times to skip already-processed bars.
        """
        token = KITE_TOKENS.get(instrument)
        if token is None:
            logger.warning(f"No Kite token for {instrument}")
            return []

        # Always fetch from session open so we recover any missed bars
        from_dt = datetime.combine(self._session_date, SESSION_OPEN)

        for attempt in range(KITE_MAX_RETRIES):
            try:
                data = self.kite.historical_data(
                    instrument_token=token,
                    from_date=from_dt,
                    to_date=timestamp,
                    interval='5minute',
                )
                if not data:
                    return []

                bars = []
                for candle in data:
                    bar_time = candle['date']
                    if hasattr(bar_time, 'tzinfo') and bar_time.tzinfo:
                        bar_time = bar_time.replace(tzinfo=None)
                    bars.append({
                        'datetime': bar_time,
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': int(candle.get('volume', 0)),
                    })
                return bars

            except Exception as e:
                logger.warning(
                    f"Kite fetch attempt {attempt + 1}/{KITE_MAX_RETRIES} "
                    f"for {instrument}: {e}"
                )
                if attempt < KITE_MAX_RETRIES - 1:
                    time_mod.sleep(KITE_RETRY_DELAY)
                else:
                    logger.error(
                        f"Kite fetch failed after {KITE_MAX_RETRIES} retries "
                        f"for {instrument}"
                    )
                    self._alert(
                        'CRITICAL',
                        f"Kite API failed for {instrument} after "
                        f"{KITE_MAX_RETRIES} retries. HALTING {instrument}."
                    )
                    return []

    def _fetch_bar_kite(self, instrument: str,
                        timestamp: datetime) -> Optional[Dict]:
        """Fetch latest single bar from Kite (backward-compat wrapper)."""
        bars = self._fetch_bars_kite(instrument, timestamp)
        return bars[-1] if bars else None

    def _fetch_bar_db(self, instrument: str,
                      timestamp: datetime) -> Optional[Dict]:
        """Fetch latest bar from DB (PAPER mode)."""
        table = 'nifty_intraday' if instrument == 'NIFTY' else 'banknifty_intraday'
        try:
            cur = self.conn.cursor()
            cur.execute(f"""
                SELECT datetime, open, high, low, close, volume
                FROM {table}
                WHERE timeframe = '5min'
                  AND datetime::date = %s
                  AND datetime <= %s
                ORDER BY datetime DESC
                LIMIT 1
            """, (self._session_date, timestamp))
            row = cur.fetchone()
            if row:
                return {
                    'datetime': row[0],
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4]),
                    'volume': int(row[5] or 0),
                }
            return None
        except Exception as e:
            logger.error(f"DB bar fetch error for {instrument}: {e}")
            self.conn.rollback()
            return None

    def _persist_bar(self, instrument: str, bar_dict: Dict):
        """Persist a fetched bar to the intraday_bars table for backtesting."""
        try:
            cur = self.conn.cursor()
            cur.execute(
                """INSERT INTO intraday_bars
                       (timestamp, instrument, open, high, low, close, volume, oi)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, 0)
                   ON CONFLICT (timestamp, instrument) DO NOTHING""",
                (bar_dict['datetime'], instrument,
                 bar_dict['open'], bar_dict['high'],
                 bar_dict['low'], bar_dict['close'],
                 bar_dict['volume']),
            )
            self.conn.commit()
        except Exception as e:
            logger.warning(f"Failed to persist bar for {instrument}: {e}")
            self.conn.rollback()

    def _load_replay_bars(self, instrument: str,
                          replay_date: date) -> List[Dict]:
        """Load all bars for a date from DB (for replay mode)."""
        table = 'nifty_intraday' if instrument == 'NIFTY' else 'banknifty_intraday'
        try:
            cur = self.conn.cursor()
            cur.execute(f"""
                SELECT datetime, open, high, low, close, volume
                FROM {table}
                WHERE timeframe = '5min' AND datetime::date = %s
                ORDER BY datetime
            """, (replay_date,))
            rows = cur.fetchall()
            return [
                {
                    'datetime': r[0],
                    'open': float(r[1]),
                    'high': float(r[2]),
                    'low': float(r[3]),
                    'close': float(r[4]),
                    'volume': int(r[5] or 0),
                }
                for r in rows
            ]
        except Exception as e:
            logger.error(f"Replay load error for {instrument}: {e}")
            self.conn.rollback()
            return []

    # ══════════════════════════════════════════════════════════
    # INDICATOR COMPUTATION
    # ══════════════════════════════════════════════════════════

    def _update_indicators(self, instrument: str):
        """Rebuild indicator DataFrame from accumulated bars."""
        state = self.session_state[instrument]
        if not state['bars']:
            return

        df = pd.DataFrame(state['bars'])
        df = df.sort_values('datetime').reset_index(drop=True)

        # Add all intraday indicators (VWAP, ORB, technicals, etc.)
        try:
            df = add_intraday_indicators(df)
        except Exception as e:
            logger.debug(f"Indicator computation partial for {instrument}: {e}")

        state['bars_df'] = df

        # Cache OR levels
        if 'opening_range_high' in df.columns and len(df) >= 3:
            state['or_high'] = float(df['opening_range_high'].iloc[-1])
            state['or_low'] = float(df['opening_range_low'].iloc[-1])
        if 'vwap' in df.columns:
            state['vwap'] = float(df['vwap'].iloc[-1]) if pd.notna(df['vwap'].iloc[-1]) else None

    # ══════════════════════════════════════════════════════════
    # SIGNAL CHECKING
    # ══════════════════════════════════════════════════════════

    def _check_signals(self, instrument: str, bar: pd.Series,
                       prev_bar: Optional[pd.Series],
                       session_bars: pd.DataFrame,
                       expiry_info: Dict) -> List[Dict]:
        """
        Check all signals for the current bar.

        Returns list of signal dicts: [{signal_id, direction, price, reason}, ...]
        """
        fired = []
        state = self.session_state[instrument]

        # ── 1. KAUFMAN_BB_MR — REMOVED ──
        # Tested on real 5-min Kite data: PF 0.22 (Nifty), 0.26 (BankNifty).
        # All 17 parameter configs lose. BB mean-reversion doesn't work intraday.

        # ── 2. GUJRAL_RANGE — REMOVED ──
        # Tested on real 5-min Kite data: PF 0.36 (Nifty), 0.49 (BankNifty).
        # All parameter configs lose. Session-range MR doesn't work intraday.

        # ── 3. ORB_TUNED (NIFTY only, long-only) ──
        if instrument == 'NIFTY':
            sig = self._check_orb_tuned(bar, prev_bar, session_bars, state)
            if sig:
                fired.append(sig)

        # ── 4. Gamma signals on expiry days ──
        if expiry_info.get('is_any_expiry', False):
            gamma_sigs = self.expiry_detector.check_gamma_signals(
                bar, prev_bar, session_bars, expiry_info
            )
            # For BANKNIFTY expiry (Wednesday), prefix gamma signals with BN_
            for gs in gamma_sigs:
                gs_instrument = gs.get('instrument', '')
                if gs_instrument == 'BANKNIFTY':
                    gs['signal_id'] = 'BN_' + gs['signal_id']
                    gs['instrument'] = 'BANKNIFTY'
            fired.extend(gamma_sigs)

        # ── 5. BankNifty signals (via BankNiftySignalComputer) ──
        # All BN_ signals start as SHADOW — logged to DB but not executed live.
        # Use bn_promotion_tracker.py for promotion to SCORING after criteria met.
        if instrument == 'BANKNIFTY':
            bn_sigs = self.bn_signal_computer.compute_all(
                bar, prev_bar, session_bars
            )
            for bs in bn_sigs:
                bs['instrument'] = 'BANKNIFTY'
                # Mark as SHADOW trade for tracking
                bs['trade_type'] = 'SHADOW'
            fired.extend(bn_sigs)

            # BN gamma signals on Wednesday (BN expiry day)
            if expiry_info.get('is_banknifty_expiry', False):
                bn_gamma_sigs = self.expiry_detector.check_gamma_signals(
                    bar, prev_bar, session_bars,
                    {
                        'is_any_expiry': True,
                        'expiry_instruments': ['BANKNIFTY'],
                    }
                )
                for gs in bn_gamma_sigs:
                    gs['signal_id'] = 'BN_' + gs['signal_id']
                    gs['instrument'] = 'BANKNIFTY'
                    gs['trade_type'] = 'SHADOW'
                fired.extend(bn_gamma_sigs)

        # ── Entry window filter (Layer 9) ──
        # Block signals outside their optimal entry windows
        bar_time = bar['datetime'] if 'datetime' in bar.index else None
        if bar_time is not None and fired:
            current_t = bar_time.time() if hasattr(bar_time, 'time') else None
            if current_t is not None:
                filtered = []
                for sig in fired:
                    sid = sig.get('signal_id', '')
                    if self.entry_window_optimizer.is_entry_allowed(sid, current_t):
                        filtered.append(sig)
                    else:
                        state['signals_blocked'] += 1
                        logger.debug(f"  Entry window blocked {sid} at {current_t}")
                fired = filtered

        state['signals_fired'] += len(fired)
        return fired

    def _check_kaufman_bb_mr(self, instrument: str, bar: pd.Series,
                             prev_bar: Optional[pd.Series],
                             session_bars: pd.DataFrame) -> Optional[Dict]:
        """
        KAUFMAN_BB_MR: Bollinger Band mean reversion.
        Long at lower band, short at upper band, only when ADX < 25.
        RSI confirmation: RSI < 35 for longs, RSI > 65 for shorts.
        """
        if pd.isna(bar.get('bb_lower')) or pd.isna(bar.get('adx_14')):
            return None

        close = float(bar['close'])
        bb_lower = float(bar['bb_lower'])
        bb_upper = float(bar['bb_upper'])
        adx_val = float(bar['adx_14'])
        rsi_val = float(bar.get('rsi_14', 50)) if pd.notna(bar.get('rsi_14')) else 50

        if adx_val >= 25:
            return None

        bb_width = bb_upper - bb_lower
        if bb_width <= 0:
            return None

        # Long: close near lower band + RSI confirmation
        lower_zone = bb_lower + bb_width * 0.10
        if close <= lower_zone and close > bb_lower and rsi_val < 35:
            return {
                'signal_id': 'KAUFMAN_BB_MR',
                'direction': 'LONG',
                'price': close,
                'instrument': instrument,
                'reason': (
                    f'BB MR long: close={close:.0f} bb_low={bb_lower:.0f} '
                    f'rsi={rsi_val:.0f} adx={adx_val:.0f}'
                ),
            }

        # Short: close near upper band + RSI confirmation
        upper_zone = bb_upper - bb_width * 0.10
        if close >= upper_zone and close < bb_upper and rsi_val > 65:
            return {
                'signal_id': 'KAUFMAN_BB_MR',
                'direction': 'SHORT',
                'price': close,
                'instrument': instrument,
                'reason': (
                    f'BB MR short: close={close:.0f} bb_up={bb_upper:.0f} '
                    f'rsi={rsi_val:.0f} adx={adx_val:.0f}'
                ),
            }

        return None

    def _check_gujral_range(self, instrument: str, bar: pd.Series,
                            prev_bar: Optional[pd.Series],
                            session_bars: pd.DataFrame) -> Optional[Dict]:
        """
        GUJRAL_RANGE: Pivot breakout with body% and ADX filter.
        Uses session range boundaries for mean reversion.
        """
        if len(session_bars) < 10:
            return None

        close = float(bar['close'])
        adx_val = float(bar.get('adx_14', 0)) if pd.notna(bar.get('adx_14')) else 0
        body_pct = float(bar.get('body_pct', 0)) if pd.notna(bar.get('body_pct')) else 0

        # ADX filter: only trade when not strongly trending
        if adx_val >= 30:
            return None

        # Need a real bar (body > 40%)
        if body_pct < 0.40:
            return None

        session_high = float(session_bars['high'].max())
        session_low = float(session_bars['low'].min())
        session_range = session_high - session_low

        if session_range <= 0:
            return None

        # Compute pivot
        pivot = (session_high + session_low + close) / 3
        position = (close - session_low) / session_range

        # Long: close near session low, reversal candle
        if position <= 0.15 and close > session_low:
            return {
                'signal_id': 'GUJRAL_RANGE',
                'direction': 'LONG',
                'price': close,
                'instrument': instrument,
                'reason': (
                    f'Range long: pos={position:.0%} body={body_pct:.0%} '
                    f'adx={adx_val:.0f} range={session_range:.0f}'
                ),
            }

        # Short: close near session high
        if position >= 0.85 and close < session_high:
            return {
                'signal_id': 'GUJRAL_RANGE',
                'direction': 'SHORT',
                'price': close,
                'instrument': instrument,
                'reason': (
                    f'Range short: pos={position:.0%} body={body_pct:.0%} '
                    f'adx={adx_val:.0f} range={session_range:.0f}'
                ),
            }

        return None

    def _check_orb_tuned(self, bar: pd.Series,
                         prev_bar: Optional[pd.Series],
                         session_bars: pd.DataFrame,
                         state: Dict) -> Optional[Dict]:
        """
        ORB_TUNED: Opening Range Breakout (LONG only).
        Filters: OR/ATR < 0.3, VIX < 17.
        """
        if prev_bar is None or pd.isna(bar.get('opening_range_high')):
            return None

        close = float(bar['close'])
        or_high = state.get('or_high')
        or_low = state.get('or_low')
        daily_atr = state.get('daily_atr', 1)
        daily_vix = state.get('daily_vix', 20)

        if or_high is None or or_low is None or daily_atr <= 0:
            return None

        or_width = or_high - or_low
        or_atr_ratio = or_width / daily_atr if daily_atr > 0 else 999

        # ORB filters from intraday_shadows config
        atr_max_ratio = ORB_TUNED_CONFIG.get('atr_max_ratio', 0.3)
        vix_max = ORB_TUNED_CONFIG.get('vix_max', 17.0)
        long_only = ORB_TUNED_CONFIG.get('long_only', True)

        if or_atr_ratio >= atr_max_ratio:
            return None
        if daily_vix >= vix_max:
            return None

        # Must be after opening range is set (bar >= 3)
        if state['bar_count'] < 4:
            return None

        prev_close = float(prev_bar['close'])

        # Long breakout
        if close > or_high and prev_close <= or_high:
            return {
                'signal_id': 'ORB_TUNED',
                'direction': 'LONG',
                'price': close,
                'instrument': 'NIFTY',
                'reason': (
                    f'ORB long: {close:.0f} > OR_high={or_high:.0f} '
                    f'OR/ATR={or_atr_ratio:.2f} VIX={daily_vix:.1f}'
                ),
            }

        # Short breakout (only if not long_only)
        if not long_only and close < or_low and prev_close >= or_low:
            return {
                'signal_id': 'ORB_TUNED',
                'direction': 'SHORT',
                'price': close,
                'instrument': 'NIFTY',
                'reason': (
                    f'ORB short: {close:.0f} < OR_low={or_low:.0f} '
                    f'OR/ATR={or_atr_ratio:.2f} VIX={daily_vix:.1f}'
                ),
            }

        return None

    # ══════════════════════════════════════════════════════════
    # FILTER, SIZE, EXECUTE
    # ══════════════════════════════════════════════════════════

    def _filter_and_size(self, signal: Dict, instrument: str,
                         bar: pd.Series):
        """
        Pipeline: regime filter -> loss limiter -> compound sizer -> options executor.

        BN SHADOW signals: logged to DB for tracking, not executed.
        """
        state = self.session_state[instrument]
        signal_id = signal.get('signal_id', 'UNKNOWN')

        # ── BN SHADOW: log trade for promotion tracking, skip execution ──
        if signal.get('trade_type') == 'SHADOW' and signal_id.startswith('BN_'):
            self._log_bn_shadow_trade(signal, instrument, bar)
            return

        # ── Step 0b: Calendar overlay — block entries on event days ──
        if self._calendar_ctx and self._calendar_ctx.get('block_new_entries'):
            logger.info(
                f"  BLOCKED {signal_id}: calendar event day "
                f"(events={self._calendar_ctx.get('events_active', [])})"
            )
            state['signals_blocked'] += 1
            return

        # ── Step 0c: Decay override check ──
        decay_factor = self.decay_manager.apply_size_overrides(signal_id)
        if decay_factor <= 0:
            logger.info(
                f"  BLOCKED {signal_id}: decay override (factor={decay_factor})"
            )
            state['signals_blocked'] += 1
            return

        # ── Step 1: Position limit check ──
        if self._total_positions >= MAX_TOTAL_POSITIONS:
            logger.info(
                f"  BLOCKED {signal_id}: max positions "
                f"({self._total_positions}/{MAX_TOTAL_POSITIONS})"
            )
            state['signals_blocked'] += 1
            return

        # ── Step 2: Regime filter (dynamic regime takes priority) ──
        bars_df = state['bars_df']
        daily_atr = state.get('daily_atr')

        # Use dynamic regime if available, fall back to bar-based detection
        dynamic_regime = self.dynamic_regime.get_regime()

        regime_result = self.regime_filter.evaluate(
            bar, bars_df, signal_id, daily_atr=daily_atr
        )

        if not regime_result['allow']:
            logger.info(
                f"  BLOCKED {signal_id}: {regime_result['reason']}"
            )
            state['signals_blocked'] += 1
            return

        # Override regime with dynamic VIX-based regime (more conservative wins)
        from signals.dynamic_regime import REGIME_SEVERITY
        bar_regime = regime_result['regime']
        if REGIME_SEVERITY.get(dynamic_regime, 0) > REGIME_SEVERITY.get(bar_regime, 0):
            regime = dynamic_regime
            from signals.regime_filter import REGIMES as _REGIMES
            regime_config = _REGIMES.get(regime, _REGIMES['NORMAL'])
            size_factor = regime_config['size_factor']
            sl_mult = regime_config['sl_mult']
        else:
            regime = bar_regime
            size_factor = regime_result['size_factor']
            sl_mult = regime_result['sl_multiplier']

        # Apply signal-specific size adjustment from regime filter
        adjusted_factor = self.regime_filter.get_adjusted_size_factor(signal_id, regime)
        if adjusted_factor < size_factor:
            size_factor = adjusted_factor

        # ── Step 2b: FII overlay sizing modifier ──
        # FII signals are OVERLAY only — modify sizing, never standalone.
        trade_direction = signal.get('direction', 'LONG')
        fii_modifier = 1.0
        if self._fii_bias and self._fii_bias.get('signal_id'):
            fii_modifier = self.fii_overlay.get_sizing_modifier(
                self._fii_bias['signal_id'],
                trade_direction,
                as_of_date=self._session_date,
            )
            if fii_modifier != 1.0:
                logger.info(
                    f"  FII modifier for {signal_id} ({trade_direction}): "
                    f"{fii_modifier:.2f}x "
                    f"(bias={self._fii_bias['direction']})"
                )
            size_factor *= fii_modifier

        # ── Step 2c: Calendar overlay sizing modifier ──
        if self._calendar_ctx:
            cal_mod = self._calendar_ctx.get('composite_modifier', 1.0)
            cal_sl_mult = self._calendar_ctx.get('expiry_sl_multiplier', 1.0)

            if cal_mod != 1.0 and cal_mod > 0:
                size_factor *= cal_mod
                logger.info(
                    f"  Calendar modifier for {signal_id}: "
                    f"{cal_mod:.2f}x (events={self._calendar_ctx.get('events_active', [])})"
                )

            # Apply tighter SL on event days (widen SL = larger SL multiplier)
            if cal_sl_mult != 1.0:
                sl_mult *= cal_sl_mult
                logger.info(
                    f"  Calendar SL adjustment: {cal_sl_mult:.2f}x "
                    f"(expiry week widen)"
                )

        # ── Step 3: Loss limiter ──
        if not self.loss_limiter.can_trade(self._session_date):
            logger.warning(f"  BLOCKED {signal_id}: daily loss limit hit")
            state['signals_blocked'] += 1

            # Tier 2: emergency exit
            if self.loss_limiter.tier >= 2:
                self._emergency_exit()
            return

        # Apply loss limiter size factor
        size_factor *= self.loss_limiter.size_factor

        # Apply decay override factor
        if decay_factor < 1.0:
            size_factor *= decay_factor
            logger.info(
                f"  Decay override for {signal_id}: "
                f"factor={decay_factor:.2f} -> size_factor={size_factor:.2f}"
            )

        # ── Step 4: Compound sizer ──
        close = float(bar['close'])
        premium = estimate_premium(close, state.get('daily_atr', 200))
        lots = self.sizer.get_lots(
            instrument=instrument, premium=premium, today=self._session_date
        )

        # Apply regime size factor
        lots = max(1, int(lots * size_factor))

        if lots <= 0:
            logger.info(f"  BLOCKED {signal_id}: sizer returned 0 lots")
            state['signals_blocked'] += 1
            return

        # ── Step 4b: Behavioral overlay (Layer 7) ──
        behavioral_ctx = OverlayContext(
            system_size=size_factor,
            current_dd_pct=self.sizer.drawdown_pct,
            entry_price=close,
            proposed_sl=close * 0.97,  # approximate
            proposed_tgt=close * 1.03,
            direction=trade_direction,
            current_atr=state.get('daily_atr', 200),
        )
        behavioral_result = self.behavioral_overlay.apply_all(behavioral_ctx)
        if behavioral_result.overlays_triggered:
            behavioral_lots = max(1, int(lots * behavioral_result.size_multiplier))
            logger.info(
                f"  Behavioral overlay {signal_id}: {lots} -> {behavioral_lots} lots "
                f"(triggers={behavioral_result.overlays_triggered})"
            )
            lots = behavioral_lots

        # ── Step 4c: IV rank filter (Layer 8 — options strategy gating) ──
        if signal_id.startswith('L8_'):
            current_vix_pre = self.static_vix or 15.0
            if self.vix_streamer is not None:
                live_vix_pre = self.vix_streamer.get_vix()
                if live_vix_pre is not None:
                    current_vix_pre = live_vix_pre
            self.iv_rank_filter.update_vix(current_vix_pre)
            strategy_type = signal.get('strategy_type', '')
            iv_check = self.iv_rank_filter.check_strategy(strategy_type, current_vix_pre)
            if iv_check['action'] == 'BLOCK':
                logger.info(
                    f"  BLOCKED {signal_id}: IV rank filter "
                    f"(iv_rank={iv_check['iv_rank']:.0f}, action=BLOCK)"
                )
                state['signals_blocked'] += 1
                return
            if iv_check['action'] == 'PREFER':
                lots = max(1, int(lots * iv_check['size_boost']))
                logger.info(
                    f"  IV rank PREFER {signal_id}: "
                    f"boosted to {lots} lots (×{iv_check['size_boost']:.1f})"
                )

        # ── Step 5: Dry run check ──
        if self.dry_run:
            logger.info(
                f"  DRY-RUN {signal_id} {signal['direction']}: "
                f"{lots} lots @ {close:.0f} | regime={regime} "
                f"size_factor={size_factor:.2f}"
            )
            return

        # ── Step 6: Determine spread vs naked ──
        expiry_info = state.get('expiry_info', {})
        expiry_date_key = (
            'nifty_expiry_date' if instrument == 'NIFTY'
            else 'banknifty_expiry_date'
        )
        expiry_date = expiry_info.get(expiry_date_key)
        expiry_str = expiry_date.strftime('%y%m%d') if expiry_date else None

        # Get current VIX for spread decision
        current_vix = self.static_vix or 15.0
        if self.vix_streamer is not None:
            live_vix = self.vix_streamer.get_vix()
            if live_vix is not None:
                current_vix = live_vix

        daily_atr = state.get('daily_atr', 300)

        is_gamma = (
            signal_id in GAMMA_SIGNAL_IDS
            or signal_id.startswith('GAMMA_')
        )

        # Use spreads for non-gamma signals
        if not is_gamma:
            # Get spread-aware lot sizing
            lot_size = NIFTY_LOT_SIZE if instrument == 'NIFTY' else BANKNIFTY_LOT_SIZE
            max_loss_per_lot = premium * lot_size  # conservative estimate
            spread_lots = self.sizer.get_spread_lots(
                instrument=instrument,
                net_debit=premium,
                max_loss_per_lot=max_loss_per_lot,
                today=self._session_date,
            )
            spread_lots = max(1, int(spread_lots * size_factor))

            spread_order = self.options_executor.signal_to_orders(
                signal=signal,
                underlying_ltp=close,
                instrument=instrument,
                atm_premium=premium,
                expiry_date=expiry_str,
                use_spreads=True,
                atr=daily_atr,
                vix=current_vix,
                regime=regime,
            )

            if isinstance(spread_order, SpreadOrder):
                # Override lots from spread sizer
                spread_order.lots = spread_lots
                spread_order.buy_leg.lots = spread_lots
                spread_order.buy_leg.quantity = spread_lots * lot_size
                if spread_order.sell_leg:
                    spread_order.sell_leg.lots = spread_lots
                    spread_order.sell_leg.quantity = spread_lots * lot_size

                # Execute spread
                spread_fill = self.spread_executor.execute_spread(spread_order)

                if spread_fill.status in ("FILLED", "DEGRADED", "PARTIAL"):
                    # Track in spread monitor
                    self.spread_monitor.add_spread(
                        signal_id, spread_order, spread_fill,
                        underlying_price=close,
                    )

                    # Record position in session state
                    position = {
                        'signal_id': signal_id,
                        'tradingsymbol': spread_order.buy_leg.tradingsymbol,
                        'direction': signal.get('direction', '').upper(),
                        'entry_premium': spread_fill.net_entry_debit,
                        'sl_premium': spread_order.sl_value,
                        'target_premium': spread_order.tgt_value,
                        'quantity': spread_order.buy_leg.quantity,
                        'lots': spread_lots,
                        'lot_size': lot_size,
                        'instrument': instrument,
                        'entry_time': datetime.now(),
                        'entry_bar': bar.get('datetime'),
                        'bars_held': 0,
                        'underlying_entry': close,
                        'is_spread': True,
                        'spread_strategy': spread_order.strategy,
                    }
                    state['positions'].append(position)
                    state['orders_placed'] += 1
                    self._total_positions += 1

                    # Log premium savings
                    saved = spread_order.premium_saved
                    saved_total = saved * spread_order.buy_leg.quantity
                    logger.info(
                        f"  SPREAD {signal_id} {spread_order.strategy} "
                        f"{spread_order.direction}: "
                        f"buy={spread_order.buy_leg.tradingsymbol} "
                        f"{'sell=' + spread_order.sell_leg.tradingsymbol if spread_order.sell_leg else ''} "
                        f"x{spread_lots}L | net_debit={spread_fill.net_entry_debit:.0f} "
                        f"| saved={saved:.0f}/unit ({saved_total:,.0f} total)"
                    )

                    self._alert(
                        'INFO',
                        f"SPREAD ENTRY: {signal_id} {spread_order.strategy}\n"
                        f"Buy: {spread_order.buy_leg.tradingsymbol} x{spread_lots}L\n"
                        f"{'Sell: ' + spread_order.sell_leg.tradingsymbol if spread_order.sell_leg else 'Naked'}\n"
                        f"Net debit: {spread_fill.net_entry_debit:.0f}\n"
                        f"Premium saved: {saved_total:,.0f}",
                        signal_id=signal_id,
                    )
                    return
                else:
                    logger.warning(
                        f"  Spread execution FAILED for {signal_id}: "
                        f"{spread_fill.status} — falling through to naked"
                    )

        # ── Fallback: naked option order (gamma signals or spread failure) ──
        order = self.options_executor.signal_to_orders(
            signal=signal,
            underlying_ltp=close,
            instrument=instrument,
            atm_premium=premium,
            expiry_date=expiry_str,
        )

        if order is None:
            logger.info(f"  BLOCKED {signal_id}: options executor rejected")
            state['signals_blocked'] += 1
            return

        # Override lots from our sizer
        lot_size = NIFTY_LOT_SIZE if instrument == 'NIFTY' else BANKNIFTY_LOT_SIZE
        order['lots'] = lots
        order['quantity'] = lots * lot_size
        order['sl_multiplier'] = sl_mult

        # Adjust SL by regime multiplier
        if sl_mult != 1.0:
            sl_pct = 0.30 * sl_mult  # widen SL in volatile regimes
            sl_pct = min(sl_pct, 0.60)  # cap at 60%
            order['sl_premium'] = round(premium * (1 - sl_pct), 2)

        # ── Execute naked order (log to DB, paper fill or Kite) ──
        self._execute_order(order, instrument, bar)

    def _log_bn_shadow_trade(self, signal: Dict, instrument: str,
                             bar: pd.Series):
        """
        Log a BN SHADOW signal to the trades table for promotion tracking.

        These are not executed — they accumulate performance data so that
        bn_promotion_tracker.py can evaluate them for promotion to SCORING.
        """
        signal_id = signal.get('signal_id', 'UNKNOWN')
        direction = signal.get('direction', '')
        close = float(bar['close'])
        state = self.session_state[instrument]

        # Compute simulated exit price using SL/TGT from signal config
        from paper_trading.signal_compute import SHADOW_SIGNALS
        sig_config = SHADOW_SIGNALS.get(signal_id, {})
        sl_pct = sig_config.get('stop_loss_pct', 0.004)
        tgt_pct = sig_config.get('take_profit_pct', 0.007)

        lot_size = BANKNIFTY_LOT_SIZE
        premium = estimate_premium(close, state.get('daily_atr', 400))

        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO trades
                    (signal_id, tradingsymbol, direction, entry_date,
                     entry_price, entry_time, quantity, lots, lot_size,
                     instrument, trade_type, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'SHADOW', %s)
            """, (
                signal_id,
                f'BN_SHADOW_{signal_id}',
                direction,
                self._session_date,
                premium,
                datetime.now(),
                lot_size,  # 1 lot for shadow
                1,
                lot_size,
                instrument,
                signal.get('reason', ''),
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"BN shadow trade log failed: {e}")
            self.conn.rollback()

        logger.info(
            f"  SHADOW {signal_id} {direction}: "
            f"{close:.0f} prem~{premium:.0f} | "
            f"SL={sl_pct:.1%} TGT={tgt_pct:.1%}"
        )

    def _execute_order(self, order: Dict, instrument: str,
                       bar: pd.Series):
        """Place order and record position."""
        state = self.session_state[instrument]
        signal_id = order['signal_id']

        # Log to DB
        try:
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO trades
                    (signal_id, tradingsymbol, direction, entry_date,
                     entry_price, entry_time, quantity, lots, lot_size,
                     instrument, trade_type, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                signal_id,
                order.get('tradingsymbol', ''),
                order.get('signal_direction', ''),
                self._session_date,
                order.get('entry_premium', 0),
                datetime.now(),
                order.get('quantity', 0),
                order.get('lots', 1),
                order.get('lot_size', 25),
                instrument,
                'PAPER' if self.mode == 'PAPER' else 'LIVE',
                order.get('signal_reason', ''),
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Trade log failed: {e}")
            self.conn.rollback()

        # Track position
        position = {
            'signal_id': signal_id,
            'tradingsymbol': order.get('tradingsymbol', ''),
            'direction': order.get('signal_direction', ''),
            'entry_premium': order.get('entry_premium', 0),
            'sl_premium': order.get('sl_premium', 0),
            'target_premium': order.get('target_premium', 0),
            'quantity': order.get('quantity', 0),
            'lots': order.get('lots', 1),
            'lot_size': order.get('lot_size', 25),
            'instrument': instrument,
            'entry_time': datetime.now(),
            'entry_bar': bar.get('datetime'),
            'bars_held': 0,
            'underlying_entry': float(bar['close']),
        }
        state['positions'].append(position)
        state['orders_placed'] += 1
        self._total_positions += 1

        logger.info(
            f"  ORDER {signal_id} {position['direction']}: "
            f"{order.get('tradingsymbol', '')} x{position['lots']}L "
            f"@ {position['entry_premium']:.0f} "
            f"| SL={position['sl_premium']:.0f} TGT={position['target_premium']:.0f}"
        )

        self._alert(
            'INFO',
            f"ENTRY: {signal_id} {position['direction']}\n"
            f"{order.get('tradingsymbol', '')} x{position['lots']}L "
            f"@ {position['entry_premium']:.0f}\n"
            f"SL={position['sl_premium']:.0f} TGT={position['target_premium']:.0f}",
            signal_id=signal_id,
        )

    # ══════════════════════════════════════════════════════════
    # EXIT CHECKING
    # ══════════════════════════════════════════════════════════

    def _check_exits(self, instrument: str, bar: pd.Series):
        """
        Check all open positions for exit conditions:
        - Spread positions: via SpreadMonitor (SL 40%, TGT 80%, TIME 15:20)
        - Naked positions: Premium SL (30%), TGT (50%), TIME 15:20
        - Tier 2 emergency
        """
        state = self.session_state[instrument]
        bar_dt = bar['datetime'] if isinstance(bar['datetime'], datetime) else datetime.now()
        now_time = bar_dt.time() if isinstance(bar_dt, datetime) else datetime.now().time()
        underlying_now = float(bar['close'])
        positions_to_remove = []

        # ── Check spread positions via SpreadMonitor ──
        spread_signal_ids = set()
        for pos in state['positions']:
            if pos.get('is_spread'):
                signal_id = pos['signal_id']
                spread_signal_ids.add(signal_id)
                # Update underlying for delta estimation
                self.spread_monitor.update_from_underlying(
                    signal_id, underlying_now
                )

        if spread_signal_ids:
            exit_signals = self.spread_monitor.check_exits(bar_dt)
            for exit_sig in exit_signals:
                if exit_sig.signal_id not in spread_signal_ids:
                    continue

                # Find the position in our list
                for i, pos in enumerate(state['positions']):
                    if pos['signal_id'] == exit_sig.signal_id and pos.get('is_spread'):
                        live_spread = self.spread_monitor.get_spread(exit_sig.signal_id)
                        if live_spread:
                            # Execute spread exit
                            exit_fill = self.spread_executor.exit_spread(
                                live_spread.spread_order,
                                live_spread.spread_fill,
                                exit_sig.reason,
                                current_buy_premium=exit_sig.current_buy_premium,
                                current_sell_premium=exit_sig.current_sell_premium,
                            )

                            # Calculate P&L
                            pnl_result = self.spread_executor.calculate_spread_pnl(
                                live_spread.spread_fill,
                                exit_fill,
                                pos['quantity'],
                            )
                            pnl = pnl_result['net_pnl']

                            # Update risk components
                            self.loss_limiter.record_trade(pnl, today=self._session_date)
                            self.sizer.update_equity(self.sizer.equity + pnl)
                            self.regime_filter.record_trade_result(pnl >= 0)

                            state['exits_count'] += 1
                            state['daily_pnl'] += pnl
                            self._total_positions = max(0, self._total_positions - 1)

                            # Remove from spread monitor
                            self.spread_monitor.remove_spread(exit_sig.signal_id)

                            sign = '+' if pnl >= 0 else ''
                            logger.info(
                                f"  SPREAD EXIT {pos['signal_id']} ({exit_sig.reason}): "
                                f"{sign}{pnl:,.0f} ({exit_sig.pnl_pct:+.1%}) | "
                                f"bars_held={pos['bars_held']} | "
                                f"day_pnl={state['daily_pnl']:,.0f}"
                            )

                            self._alert(
                                'INFO' if pnl >= 0 else 'WARNING',
                                f"SPREAD EXIT: {pos['signal_id']} ({exit_sig.reason})\n"
                                f"Strategy: {pos.get('spread_strategy', 'N/A')}\n"
                                f"P&L: {sign}{pnl:,.0f} ({exit_sig.pnl_pct:+.1%})\n"
                                f"Costs: {pnl_result['total_costs']:,.0f}\n"
                                f"Day P&L: {state['daily_pnl']:,.0f}",
                                signal_id=pos['signal_id'],
                            )

                        positions_to_remove.append(i)
                        break

        # ── Check non-spread (naked) positions ──
        for i, pos in enumerate(state['positions']):
            if i in positions_to_remove:
                continue
            if pos.get('is_spread'):
                continue  # already handled above

            exit_reason = None
            pos['bars_held'] += 1

            # Estimate current premium from underlying movement
            underlying_entry = pos.get('underlying_entry', underlying_now)
            entry_premium = pos['entry_premium']

            # Simple premium estimate based on underlying delta (~0.50)
            delta_move = underlying_now - underlying_entry
            if pos['direction'] == 'LONG':
                current_premium = entry_premium + delta_move * 0.50
            else:
                current_premium = entry_premium - delta_move * 0.50

            current_premium = max(current_premium, 0.05)  # floor

            # ── Time exit (15:20) ──
            if now_time >= FORCE_EXIT_TIME:
                exit_reason = 'FORCE_EXIT'

            # ── Stop loss ──
            elif current_premium <= pos['sl_premium']:
                exit_reason = 'STOP_LOSS'

            # ── Target ──
            elif current_premium >= pos['target_premium']:
                exit_reason = 'TARGET'

            # ── Tier 2 emergency ──
            elif self.loss_limiter.tier >= 2:
                exit_reason = 'TIER2_EMERGENCY'

            # ── Disposition effect check (Layer 7) ──
            elif exit_reason is None:
                unrealized_pct = (current_premium - entry_premium) / entry_premium if entry_premium > 0 else 0
                max_gain_pct = pos.get('max_unrealized_gain_pct', 0.0)
                if unrealized_pct > max_gain_pct:
                    pos['max_unrealized_gain_pct'] = unrealized_pct

                disp_ctx = OverlayContext(
                    position=PositionContext(
                        signal_id=pos['signal_id'],
                        direction=pos['direction'],
                        entry_price=entry_premium,
                        current_price=current_premium,
                        bars_held=pos['bars_held'],
                        max_unrealized_gain_pct=pos.get('max_unrealized_gain_pct', 0.0),
                        unrealized_pnl_pct=unrealized_pct,
                    ),
                )
                disp_result = self.behavioral_overlay.apply_all(disp_ctx)
                if disp_result.force_exit:
                    exit_reason = f'DISPOSITION_{disp_result.force_exit_reason}'
                    logger.info(
                        f"  Disposition exit {pos['signal_id']}: "
                        f"{disp_result.force_exit_reason}"
                    )
                elif disp_result.trailing_stop is not None:
                    # Check if trailing stop was hit
                    if pos['direction'] == 'LONG' and current_premium <= disp_result.trailing_stop:
                        exit_reason = 'TRAILING_STOP'
                    elif pos['direction'] == 'SHORT' and current_premium >= disp_result.trailing_stop:
                        exit_reason = 'TRAILING_STOP'

            if exit_reason:
                self._close_position(
                    instrument, pos, current_premium, exit_reason
                )
                positions_to_remove.append(i)

        # Remove closed positions (reverse order to maintain indices)
        for idx in sorted(set(positions_to_remove), reverse=True):
            state['positions'].pop(idx)

    def _close_position(self, instrument: str, position: Dict,
                        exit_premium: float, reason: str):
        """Close a position and record P&L."""
        state = self.session_state[instrument]
        entry_premium = position['entry_premium']
        quantity = position['quantity']

        pnl = (exit_premium - entry_premium) * quantity
        pnl_pct = (exit_premium - entry_premium) / entry_premium if entry_premium > 0 else 0

        # Update risk components
        self.loss_limiter.record_trade(pnl, today=self._session_date)
        self.sizer.update_equity(self.sizer.equity + pnl)
        self.regime_filter.record_trade_result(pnl >= 0)

        state['exits_count'] += 1
        state['daily_pnl'] += pnl
        self._total_positions = max(0, self._total_positions - 1)

        # Record trade in behavioral overlay for streak tracking
        self.behavioral_overlay.record_trade(TradeRecord(
            pnl=pnl,
            pnl_pct=pnl_pct,
            entry_time=position.get('entry_time', datetime.now()),
            exit_time=datetime.now(),
            holding_bars=position['bars_held'],
            signal_id=position['signal_id'],
        ))

        sign = '+' if pnl >= 0 else ''
        logger.info(
            f"  EXIT {position['signal_id']} ({reason}): "
            f"{sign}{pnl:,.0f} ({pnl_pct:+.1%}) | "
            f"bars_held={position['bars_held']} | "
            f"day_pnl={state['daily_pnl']:,.0f}"
        )

        # Log exit to DB
        try:
            cur = self.conn.cursor()
            cur.execute("""
                UPDATE trades
                SET exit_date = %s, exit_price = %s, exit_time = NOW(),
                    pnl = %s, exit_reason = %s
                WHERE signal_id = %s AND exit_date IS NULL
                  AND trade_type = %s
                LIMIT 1
            """, (
                self._session_date, exit_premium, pnl, reason,
                position['signal_id'],
                'PAPER' if self.mode == 'PAPER' else 'LIVE',
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Exit log failed: {e}")
            self.conn.rollback()

        self._alert(
            'INFO' if pnl >= 0 else 'WARNING',
            f"EXIT: {position['signal_id']} ({reason})\n"
            f"P&L: {sign}{pnl:,.0f} ({pnl_pct:+.1%})\n"
            f"Day P&L: {state['daily_pnl']:,.0f}",
            signal_id=position['signal_id'],
        )

    def _force_exit_all(self):
        """Force-exit all open positions at 15:20."""
        for inst in self.instruments:
            state = self.session_state[inst]
            if not state['positions']:
                continue

            logger.info(f"Force-exiting {len(state['positions'])} positions for {inst}")
            bars_df = state['bars_df']
            if bars_df.empty:
                continue

            bar = bars_df.iloc[-1]
            for pos in list(state['positions']):
                underlying_now = float(bar['close'])
                underlying_entry = pos.get('underlying_entry', underlying_now)
                entry_premium = pos['entry_premium']
                delta_move = underlying_now - underlying_entry
                if pos['direction'] == 'LONG':
                    current_premium = entry_premium + delta_move * 0.50
                else:
                    current_premium = entry_premium - delta_move * 0.50
                current_premium = max(current_premium, 0.05)

                self._close_position(inst, pos, current_premium, 'FORCE_EXIT')

            state['positions'] = []

    def _emergency_exit(self):
        """Emergency exit all positions (Tier 2 or crash)."""
        logger.critical("EMERGENCY EXIT — closing all positions")
        self._alert('EMERGENCY', 'Emergency exit triggered — closing all positions')
        self._force_exit_all()

    # ══════════════════════════════════════════════════════════
    # RECONCILIATION
    # ══════════════════════════════════════════════════════════

    def _reconcile(self):
        """Post-session reconciliation at 15:35."""
        logger.info("Running post-session reconciliation...")

        total_pnl = 0
        total_trades = 0
        for inst in self.instruments:
            state = self.session_state[inst]
            total_pnl += state['daily_pnl']
            total_trades += state['orders_placed']

            if state['positions']:
                logger.warning(
                    f"  {inst}: {len(state['positions'])} positions still open "
                    f"after force exit!"
                )

        logger.info(
            f"Reconciliation complete: total P&L={total_pnl:,.0f} "
            f"trades={total_trades}"
        )

    # ══════════════════════════════════════════════════════════
    # LOGGING
    # ══════════════════════════════════════════════════════════

    def _log_bar_summary(self, instrument: str, bar: pd.Series):
        """One-line per bar: signals fired/blocked/orders/exits/P&L."""
        state = self.session_state[instrument]
        bar_time = bar['datetime']
        close = float(bar['close'])

        regime = self.dynamic_regime.get_regime()

        # Also show live VIX if available
        vix_str = ""
        if self.vix_streamer is not None:
            vix_val = self.vix_streamer.get_vix()
            if vix_val is not None:
                vix_str = f" VIX={vix_val:.1f}"

        ts = bar_time.strftime('%H:%M') if isinstance(bar_time, datetime) else str(bar_time)

        summary = (
            f"  {ts} {instrument:10s} {close:>8.0f} | "
            f"regime={regime:8s}{vix_str} | "
            f"fired={state['signals_fired']:2d} "
            f"blocked={state['signals_blocked']:2d} "
            f"orders={state['orders_placed']:2d} "
            f"exits={state['exits_count']:2d} | "
            f"pos={len(state['positions'])} "
            f"pnl={state['daily_pnl']:>+8,.0f}"
        )
        logger.info(summary)

    def _session_summary(self):
        """End-of-session summary."""
        # Stop VIX streamer
        if self.vix_streamer is not None:
            self.vix_streamer.stop()

        logger.info(f"\n{'='*70}")
        logger.info(f"  SESSION SUMMARY: {self._session_date}")
        logger.info(f"{'='*70}")

        total_pnl = 0
        total_trades = 0
        total_signals = 0
        total_blocked = 0

        for inst in self.instruments:
            state = self.session_state[inst]
            total_pnl += state['daily_pnl']
            total_trades += state['orders_placed']
            total_signals += state['signals_fired']
            total_blocked += state['signals_blocked']

            logger.info(
                f"  {inst:12s}: bars={state['bar_count']:3d} "
                f"signals={state['signals_fired']:2d} "
                f"blocked={state['signals_blocked']:2d} "
                f"orders={state['orders_placed']:2d} "
                f"exits={state['exits_count']:2d} "
                f"P&L={state['daily_pnl']:>+10,.0f}"
            )

        # Regime summary
        regime_summary = self.dynamic_regime.get_session_summary()
        logger.info(f"  {'─'*60}")
        logger.info(
            f"  REGIME: {regime_summary['current_regime']} | "
            f"transitions={regime_summary['transitions']} | "
            f"VIX range=[{regime_summary.get('min_vix', '?')}-"
            f"{regime_summary.get('max_vix', '?')}]"
        )
        if regime_summary.get('time_per_regime'):
            time_parts = [
                f"{r}={mins:.0f}m"
                for r, mins in regime_summary['time_per_regime'].items()
            ]
            logger.info(f"  Time: {' | '.join(time_parts)}")

        # Spread monitor summary
        spread_pnl = self.spread_monitor.get_live_pnl()
        premium_saved = spread_pnl['total_premium_saved']

        logger.info(f"  {'─'*60}")
        logger.info(
            f"  TOTAL: signals={total_signals} blocked={total_blocked} "
            f"trades={total_trades} P&L={total_pnl:>+10,.0f}"
        )
        if premium_saved > 0:
            logger.info(
                f"  Spreads: premium saved={premium_saved:,.0f}"
            )
        logger.info(
            f"  Equity: {self.sizer.equity:,.0f} "
            f"(DD={self.sizer.drawdown_pct:.1%})"
        )
        logger.info(f"{'='*70}\n")

        # VIX session stats
        vix_change = {}
        if self.vix_streamer is not None:
            vix_change = self.vix_streamer.get_vix_change()

        regime_detail = ""
        if regime_summary.get('transitions', 0) > 0:
            regime_detail = (
                f"\nRegime transitions: {regime_summary['transitions']}"
            )
            for t in regime_summary.get('transition_log', []):
                ts = t['timestamp'].strftime('%H:%M') if hasattr(t['timestamp'], 'strftime') else str(t['timestamp'])
                regime_detail += f"\n  {ts}: {t['from']} -> {t['to']} (VIX={t['vix']:.1f})"

        vix_detail = ""
        if vix_change.get('current') is not None:
            vix_detail = (
                f"\nVIX: {vix_change['current']:.1f} "
                f"(open={vix_change.get('open', '?')}, "
                f"hi={vix_change.get('high', '?')}, "
                f"lo={vix_change.get('low', '?')}, "
                f"chg={vix_change.get('change_abs', 0):+.1f})"
            )

        spread_detail = ""
        if premium_saved > 0:
            spread_detail = f"\nPremium saved (spreads): {premium_saved:,.0f}"

        self._alert(
            'INFO',
            f"Session {self._session_date} complete.\n"
            f"Trades: {total_trades} | Signals: {total_signals}\n"
            f"P&L: {total_pnl:+,.0f}\n"
            f"Equity: {self.sizer.equity:,.0f}\n"
            f"Regime: {regime_summary['current_regime']}"
            + spread_detail + vix_detail + regime_detail
        )

    # ══════════════════════════════════════════════════════════
    # STATUS
    # ══════════════════════════════════════════════════════════

    def show_status(self):
        """Show recent trades and current state."""
        cur = self.conn.cursor()
        try:
            cur.execute("""
                SELECT signal_id, direction, entry_date, entry_price,
                       exit_date, exit_price, pnl, exit_reason, instrument,
                       trade_type
                FROM trades
                WHERE trade_type IN ('PAPER', 'LIVE')
                  AND entry_date >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY entry_date DESC, entry_time DESC
                LIMIT 30
            """)
            rows = cur.fetchall()

            print(f"\n{'='*80}")
            print(f"  INTRADAY ORCHESTRATOR — Recent Trades (last 7 days)")
            print(f"{'='*80}")

            if not rows:
                print("  No trades found.")
            else:
                print(f"  {'Date':<12s} {'Signal':<20s} {'Dir':<6s} "
                      f"{'Entry':>8s} {'Exit':>8s} {'P&L':>10s} "
                      f"{'Reason':<12s} {'Inst':<10s}")
                print(f"  {'─'*76}")
                for r in rows:
                    sig, dir_, edate, eprice, xdate, xprice, pnl_, reason, inst, _ = r
                    pnl_str = f"{pnl_:+,.0f}" if pnl_ else "open"
                    xprice_str = f"{xprice:.0f}" if xprice else "—"
                    reason_str = reason or "—"
                    print(
                        f"  {edate!s:<12s} {sig:<20s} {dir_:<6s} "
                        f"{eprice:>8.0f} {xprice_str:>8s} {pnl_str:>10s} "
                        f"{reason_str:<12s} {inst:<10s}"
                    )

            # Loss limiter status
            print(f"\n  Loss Limiter:")
            status = self.loss_limiter.get_status()
            for k in ('equity', 'daily_total_pnl', 'tier', 'can_trade'):
                print(f"    {k}: {status[k]}")

            # Sizer status
            print(f"\n  Compound Sizer:")
            ss = self.sizer.get_status()
            for k in ('equity', 'drawdown_pct', 'current_lots', 'deploy_fraction'):
                print(f"    {k}: {ss[k]}")

            print(f"{'='*80}\n")

        except Exception as e:
            print(f"Error: {e}")
            self.conn.rollback()

    # ══════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════

    def _connect_kite(self):
        """Connect to Kite with retry."""
        for attempt in range(KITE_MAX_RETRIES):
            try:
                kite = get_kite()
                if kite is None:
                    raise RuntimeError(
                        "Not authenticated. Run: "
                        "venv/bin/python3 -m data.kite_auth --login"
                    )
                # Verify connection
                kite.profile()
                logger.info("Kite connected successfully")
                return kite
            except Exception as e:
                logger.warning(
                    f"Kite connect attempt {attempt + 1}/{KITE_MAX_RETRIES}: {e}"
                )
                if attempt < KITE_MAX_RETRIES - 1:
                    time_mod.sleep(KITE_RETRY_DELAY)

        raise RuntimeError(
            f"Failed to connect to Kite after {KITE_MAX_RETRIES} attempts"
        )

    def _init_alerter(self) -> Optional[TelegramAlerter]:
        """Initialize Telegram alerter from env vars."""
        token = os.environ.get('TELEGRAM_BOT_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        if token and chat_id:
            return TelegramAlerter(token, chat_id)
        logger.info("Telegram alerter not configured (missing env vars)")
        return None

    def _alert(self, level: str, message: str, signal_id: str = None):
        """Send alert via Telegram (non-blocking, best-effort)."""
        if self.alerter is None:
            return
        try:
            self.alerter.send(level, message, signal_id=signal_id)
        except Exception as e:
            logger.debug(f"Telegram alert failed: {e}")

    def _on_vix_update(self, vix: float, timestamp: datetime):
        """Callback from VIXStreamer when VIX changes by >= 0.5 pts."""
        logger.info(f"VIX update: {vix:.2f}")

    def _handle_shutdown(self, signum, frame):
        """Handle SIGINT/SIGTERM gracefully."""
        logger.info(f"Shutdown signal received (sig={signum})")
        self._shutdown_requested = True

        # Close positions
        logger.info("Closing positions before shutdown...")
        self._force_exit_all()

        # Session summary
        self._session_summary()

        logger.info("Graceful shutdown complete")

    def close(self):
        """Clean up resources."""
        try:
            if self.vix_streamer is not None:
                self.vix_streamer.stop()
        except Exception:
            pass
        try:
            if self.conn and not self.conn.closed:
                self.conn.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Intraday Orchestrator — NIFTY + BANKNIFTY signal runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper mode, both instruments, today:
  venv/bin/python3 -m paper_trading.intraday_runner

  # Live mode:
  venv/bin/python3 -m paper_trading.intraday_runner --live

  # Replay a specific date:
  venv/bin/python3 -m paper_trading.intraday_runner --replay 2026-03-20

  # NIFTY only:
  venv/bin/python3 -m paper_trading.intraday_runner --instrument NIFTY --no-banknifty

  # Dry run (signals computed but no orders):
  venv/bin/python3 -m paper_trading.intraday_runner --dry-run

  # Check recent trades:
  venv/bin/python3 -m paper_trading.intraday_runner --status
        """,
    )

    parser.add_argument(
        '--live', action='store_true',
        help='Run in LIVE mode (requires EXECUTION_MODE=LIVE env var)',
    )
    parser.add_argument(
        '--replay', type=str, default=None,
        help='Replay date (YYYY-MM-DD)',
    )
    parser.add_argument(
        '--instrument', type=str, default=None,
        help='Single instrument to trade (NIFTY or BANKNIFTY)',
    )
    parser.add_argument(
        '--no-banknifty', action='store_true',
        help='Exclude BANKNIFTY',
    )
    parser.add_argument(
        '--status', action='store_true',
        help='Show recent trades and system status',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Compute signals but do not execute orders',
    )

    args = parser.parse_args()

    # ── Logging ──
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S',
    )

    # ── Mode ──
    mode = 'PAPER'
    if args.live:
        # Safety: require explicit EXECUTION_MODE env var for live trading
        env_mode = os.environ.get('EXECUTION_MODE', '').upper()
        if env_mode != 'LIVE':
            print(
                "ERROR: --live requires EXECUTION_MODE=LIVE environment variable.\n"
                "Set it explicitly: EXECUTION_MODE=LIVE venv/bin/python3 -m "
                "paper_trading.intraday_runner --live"
            )
            sys.exit(1)
        mode = 'LIVE'

    # ── Instruments ──
    instruments = ['NIFTY', 'BANKNIFTY']
    if args.instrument:
        instruments = [args.instrument.upper()]
    elif args.no_banknifty:
        instruments = ['NIFTY']

    # ── Build orchestrator ──
    try:
        orch = IntradayOrchestrator(mode=mode, instruments=instruments)
    except Exception as e:
        print(f"ERROR: Failed to initialize orchestrator: {e}")
        traceback.print_exc()
        sys.exit(1)

    if args.dry_run:
        orch.dry_run = True

    try:
        if args.status:
            orch.show_status()
        elif args.replay:
            replay_date = date.fromisoformat(args.replay)
            orch.run_replay(replay_date)
        else:
            orch.run_session()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        orch.close()


if __name__ == '__main__':
    main()

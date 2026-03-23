"""
Max OI Strike Barrier Signal.

Uses options open interest to identify intraday support/resistance levels
for Nifty futures.  The strike with the highest Put OI acts as support
(put writers defend it); the strike with the highest Call OI acts as
resistance (call writers defend it).

Signal logic:
    1. Scan option chain snapshot → find max_put_oi_strike, max_call_oi_strike.

    2. LONG bounce:
       - Nifty approaches max_put_strike from above, within 30 pts.
       - Put OI at that strike > 50 lakh (5M).
       - Entry near support, SL 40 pts below, TGT 60-80 pts.

    3. SHORT rejection:
       - Nifty approaches max_call_strike from below, within 30 pts.
       - Call OI at that strike > 50 lakh (5M).
       - Entry near resistance, SL 40 pts above, TGT 60-80 pts.

    4. Breakout:
       - Price breaks support/resistance by 20+ pts → follow breakout.
       - SL 40 pts behind the broken barrier, TGT 60-80 pts.

    Filters:
       - Max OI strike must have OI > 50 lakh.
       - Max put and max call strikes must be ≥ 100 pts apart.
       - Not in the first or last 15 min of session.
       - Max hold: 2 hours.

Data source:
    - Historical: `nifty_options` table (date, expiry, strike,
      option_type, oi, volume, close, etc.)
    - Live: Kite Connect option chain snapshot.

Usage:
    from signals.structural.max_oi_barrier import MaxOIBarrierSignal

    sig = MaxOIBarrierSignal()

    # Live
    result = sig.evaluate(trade_date, current_time, nifty_price,
                          option_chain_snapshot, session_bars)

    # Backtest
    results = sig.backtest_evaluate(trade_date, nifty_bars, option_chain_df)

Academic basis: Option seller hedging (gamma exposure) creates
gravitational pull near high-OI strikes.  Empirically, max-OI strikes
act as barriers 60-70% of the time on non-trending days.
"""

import logging
import math
from datetime import date, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ================================================================
# CONSTANTS
# ================================================================

SIGNAL_ID = 'MAX_OI_BARRIER'

# OI thresholds
MIN_OI_THRESHOLD = 5_000_000    # 50 lakh = 5M contracts (OI must exceed this)
MIN_STRIKE_SPREAD = 100         # Max put & call strikes must be ≥ 100 pts apart

# Proximity thresholds (Nifty points)
APPROACH_DISTANCE = 30          # Price within 30 pts of barrier → signal zone
BREAKOUT_DISTANCE = 20          # Price breaks barrier by 20+ pts → breakout

# Risk / reward (Nifty points)
SL_POINTS = 40                  # Stop loss: 40 pts beyond barrier
TGT_MIN_POINTS = 60             # Minimum target: 60 pts
TGT_MAX_POINTS = 80             # Maximum target: 80 pts

# Time filters
SESSION_START = time(9, 15)
SESSION_END = time(15, 30)
NO_TRADE_OPEN = time(9, 30)     # No trade in first 15 min
NO_TRADE_CLOSE = time(15, 15)   # No trade in last 15 min

# Max hold
MAX_HOLD_BARS = 24              # 24 × 5-min = 2 hours

# Confidence
CONF_BOUNCE = 0.58              # Bounce off barrier
CONF_BREAKOUT = 0.52            # Breakout through barrier (lower confidence)
OI_MASSIVE_BOOST = 0.06         # OI > 100 lakh adds confidence
OI_MODERATE_PENALTY = -0.03     # OI barely above threshold
VOLUME_CONF_BOOST = 0.05        # High volume at barrier adds conviction

# Size
BASE_SIZE_MODIFIER = 1.0
MAX_SIZE_MODIFIER = 1.4
MIN_SIZE_MODIFIER = 0.6

# Signal types
SIGNAL_BOUNCE = 'BOUNCE'
SIGNAL_BREAKOUT = 'BREAKOUT'


# ================================================================
# HELPERS
# ================================================================

def _safe_float(val: Any, default: float = float('nan')) -> float:
    """Safely cast to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely cast to int."""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _bar_close(bar: Dict) -> float:
    return _safe_float(bar.get('close'))


def _bar_high(bar: Dict) -> float:
    return _safe_float(bar.get('high'))


def _bar_low(bar: Dict) -> float:
    return _safe_float(bar.get('low'))


def _bar_time_obj(bar: Dict) -> Optional[time]:
    """Extract time from bar timestamp."""
    ts = bar.get('timestamp') or bar.get('time') or bar.get('datetime')
    if ts is None:
        return None
    if hasattr(ts, 'time'):
        return ts.time()
    if isinstance(ts, time):
        return ts
    try:
        from datetime import datetime as _dt
        return _dt.fromisoformat(str(ts)).time()
    except (ValueError, TypeError):
        return None


def _find_max_oi_strikes(
    option_chain: List[Dict],
) -> Tuple[Optional[int], Optional[int], int, int]:
    """
    Scan option chain and find the strikes with max Put OI and max Call OI.

    Parameters
    ----------
    option_chain : list of dicts
        Each dict: {strike, option_type ('CE'/'PE'), oi, volume, ...}

    Returns
    -------
    (max_put_strike, max_call_strike, max_put_oi, max_call_oi)
    """
    max_put_strike = None
    max_call_strike = None
    max_put_oi = 0
    max_call_oi = 0

    for row in option_chain:
        strike = _safe_int(row.get('strike'))
        oi = _safe_int(row.get('oi'))
        opt_type = str(row.get('option_type', '')).upper().strip()

        if strike <= 0 or oi <= 0:
            continue

        if opt_type in ('PE', 'PUT', 'P'):
            if oi > max_put_oi:
                max_put_oi = oi
                max_put_strike = strike
        elif opt_type in ('CE', 'CALL', 'C'):
            if oi > max_call_oi:
                max_call_oi = oi
                max_call_strike = strike

    return max_put_strike, max_call_strike, max_put_oi, max_call_oi


def _find_max_oi_strikes_df(option_chain_df) -> Tuple[Optional[int], Optional[int], int, int]:
    """
    DataFrame variant of _find_max_oi_strikes.

    Parameters
    ----------
    option_chain_df : pandas DataFrame with columns: strike, option_type, oi

    Returns
    -------
    (max_put_strike, max_call_strike, max_put_oi, max_call_oi)
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error('pandas not available for DataFrame processing')
        return None, None, 0, 0

    if option_chain_df is None or option_chain_df.empty:
        return None, None, 0, 0

    df = option_chain_df.copy()

    # Normalise column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ('strike', 'strike_price'):
            col_map[c] = 'strike'
        elif cl in ('oi', 'open_interest', 'openinterest'):
            col_map[c] = 'oi'
        elif cl in ('option_type', 'optiontype', 'type', 'opt_type'):
            col_map[c] = 'option_type'
    df = df.rename(columns=col_map)

    required = {'strike', 'oi', 'option_type'}
    if not required.issubset(set(df.columns)):
        logger.warning('Option chain DF missing columns: %s', required - set(df.columns))
        return None, None, 0, 0

    df['oi'] = pd.to_numeric(df['oi'], errors='coerce').fillna(0).astype(int)
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce').fillna(0).astype(int)
    df['option_type'] = df['option_type'].astype(str).str.upper().str.strip()

    # Map common aliases
    df['option_type'] = df['option_type'].replace({
        'PUT': 'PE', 'P': 'PE', 'CALL': 'CE', 'C': 'CE',
    })

    puts = df[df['option_type'] == 'PE']
    calls = df[df['option_type'] == 'CE']

    max_put_strike, max_put_oi = None, 0
    max_call_strike, max_call_oi = None, 0

    if not puts.empty:
        idx = puts['oi'].idxmax()
        max_put_strike = int(puts.loc[idx, 'strike'])
        max_put_oi = int(puts.loc[idx, 'oi'])

    if not calls.empty:
        idx = calls['oi'].idxmax()
        max_call_strike = int(calls.loc[idx, 'strike'])
        max_call_oi = int(calls.loc[idx, 'oi'])

    return max_put_strike, max_call_strike, max_put_oi, max_call_oi


def _compute_target(direction: str, entry_price: float, barrier_strike: int) -> float:
    """
    Compute target between TGT_MIN_POINTS and TGT_MAX_POINTS.
    Use distance from barrier to scale target.
    """
    dist_from_barrier = abs(entry_price - barrier_strike)
    # Scale: closer entry → larger target (more room); farther → smaller
    scale = max(0.0, min(1.0, 1.0 - dist_from_barrier / 60.0))
    tgt_pts = TGT_MIN_POINTS + (TGT_MAX_POINTS - TGT_MIN_POINTS) * scale

    if direction == 'LONG':
        return entry_price + tgt_pts
    else:
        return entry_price - tgt_pts


# ================================================================
# SIGNAL CLASS
# ================================================================

class MaxOIBarrierSignal:
    """
    Max OI Strike Barrier signal for intraday Nifty futures trading.

    Identifies support (max put OI) and resistance (max call OI),
    then generates bounce or breakout signals when price interacts
    with these barriers.
    """

    SIGNAL_ID = SIGNAL_ID

    def __init__(self) -> None:
        self._last_fire_date: Optional[date] = None
        self._last_fire_time: Optional[time] = None
        self._daily_trade_count: int = 0
        self._max_daily_trades: int = 3
        logger.info('MaxOIBarrierSignal initialised')

    # ----------------------------------------------------------
    # LIVE evaluate
    # ----------------------------------------------------------
    def evaluate(
        self,
        trade_date: date,
        current_time: time,
        nifty_price: float,
        option_chain_snapshot: List[Dict],
        session_bars: Optional[List[Dict]] = None,
    ) -> Optional[Dict]:
        """
        Evaluate Max OI barrier signal on live / paper data.

        Parameters
        ----------
        trade_date            : Trading date.
        current_time          : Current IST time.
        nifty_price           : Current Nifty spot / futures price.
        option_chain_snapshot : List of dicts with strike, option_type, oi, volume.
        session_bars          : Optional list of 5-min bars so far today
                                (used for volume confirmation).

        Returns
        -------
        dict with signal details, or None if no trade.
        """
        try:
            return self._evaluate_inner(
                trade_date, current_time, nifty_price,
                option_chain_snapshot, session_bars,
            )
        except Exception as e:
            logger.error('MaxOIBarrierSignal.evaluate error: %s', e, exc_info=True)
            return None

    def _evaluate_inner(
        self,
        trade_date: date,
        current_time: time,
        nifty_price: float,
        option_chain_snapshot: List[Dict],
        session_bars: Optional[List[Dict]],
    ) -> Optional[Dict]:
        # ── Reset daily counter on new day ────────────────────────
        if self._last_fire_date != trade_date:
            self._daily_trade_count = 0

        # ── Max trades per day ────────────────────────────────────
        if self._daily_trade_count >= self._max_daily_trades:
            return None

        # ── Time filter ───────────────────────────────────────────
        if current_time < NO_TRADE_OPEN or current_time >= NO_TRADE_CLOSE:
            logger.debug('Outside trading window: %s', current_time)
            return None

        # ── Validate price ────────────────────────────────────────
        nifty_price = _safe_float(nifty_price)
        if math.isnan(nifty_price) or nifty_price <= 0:
            logger.debug('Invalid nifty_price: %s', nifty_price)
            return None

        # ── Validate option chain ─────────────────────────────────
        if not option_chain_snapshot:
            logger.debug('Empty option chain snapshot')
            return None

        # ── Find max OI strikes ───────────────────────────────────
        max_put_strike, max_call_strike, max_put_oi, max_call_oi = \
            _find_max_oi_strikes(option_chain_snapshot)

        if max_put_strike is None or max_call_strike is None:
            logger.debug('Could not determine max OI strikes')
            return None

        # ── Filter: OI threshold ──────────────────────────────────
        if max_put_oi < MIN_OI_THRESHOLD and max_call_oi < MIN_OI_THRESHOLD:
            logger.debug(
                'Both max OI below threshold: put=%d call=%d',
                max_put_oi, max_call_oi,
            )
            return None

        # ── Filter: strike spread ─────────────────────────────────
        strike_spread = abs(max_call_strike - max_put_strike)
        if strike_spread < MIN_STRIKE_SPREAD:
            logger.debug(
                'Strikes too close: put=%d call=%d spread=%d',
                max_put_strike, max_call_strike, strike_spread,
            )
            return None

        # ── Determine signal type ─────────────────────────────────
        signal = self._classify_signal(
            nifty_price, max_put_strike, max_call_strike,
            max_put_oi, max_call_oi,
        )
        if signal is None:
            return None

        signal_type, direction, barrier_strike, barrier_oi = signal

        # ── Compute entry / SL / TGT ─────────────────────────────
        entry_price = nifty_price

        if direction == 'LONG':
            stop_loss = barrier_strike - SL_POINTS
        else:
            stop_loss = barrier_strike + SL_POINTS

        target = _compute_target(direction, entry_price, barrier_strike)

        # ── Confidence ────────────────────────────────────────────
        if signal_type == SIGNAL_BOUNCE:
            confidence = CONF_BOUNCE
        else:
            confidence = CONF_BREAKOUT

        if barrier_oi > MIN_OI_THRESHOLD * 2:
            confidence += OI_MASSIVE_BOOST
        elif barrier_oi < MIN_OI_THRESHOLD * 1.2:
            confidence += OI_MODERATE_PENALTY

        # Volume confirmation from recent bars
        if session_bars and len(session_bars) >= 2:
            recent_vol = _safe_float(session_bars[-1].get('volume'), 0)
            prev_vol = _safe_float(session_bars[-2].get('volume'), 0)
            if prev_vol > 0 and recent_vol > prev_vol * 1.5:
                confidence += VOLUME_CONF_BOOST

        confidence = min(0.90, max(0.10, confidence))

        # ── Size modifier ─────────────────────────────────────────
        size_modifier = BASE_SIZE_MODIFIER
        if barrier_oi > MIN_OI_THRESHOLD * 2:
            size_modifier = min(MAX_SIZE_MODIFIER, 1.2)
        elif barrier_oi < MIN_OI_THRESHOLD * 1.2:
            size_modifier = max(MIN_SIZE_MODIFIER, 0.8)

        # ── Risk / reward ─────────────────────────────────────────
        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        rr = reward / risk if risk > 0 else 0.0

        # ── Build reason ──────────────────────────────────────────
        reason_parts = [
            f"MAX_OI_BARRIER ({signal_type})",
            f"Dir={direction}",
            f"Barrier={barrier_strike}",
            f"OI={barrier_oi:,}",
            f"Price={nifty_price:.2f}",
            f"PutOI_Strike={max_put_strike} ({max_put_oi:,})",
            f"CallOI_Strike={max_call_strike} ({max_call_oi:,})",
            f"Spread={strike_spread}",
            f"R:R={rr:.1f}",
        ]

        self._last_fire_date = trade_date
        self._last_fire_time = current_time
        self._daily_trade_count += 1

        logger.info(
            '%s signal: %s %s %s price=%.1f barrier=%d oi=%d conf=%.3f',
            self.SIGNAL_ID, signal_type, direction, trade_date,
            nifty_price, barrier_strike, barrier_oi, confidence,
        )

        return {
            'signal_id': self.SIGNAL_ID,
            'signal_type': signal_type,
            'direction': direction,
            'confidence': round(confidence, 3),
            'size_modifier': round(size_modifier, 2),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'barrier_strike': barrier_strike,
            'barrier_oi': barrier_oi,
            'max_put_strike': max_put_strike,
            'max_put_oi': max_put_oi,
            'max_call_strike': max_call_strike,
            'max_call_oi': max_call_oi,
            'strike_spread': strike_spread,
            'max_hold_bars': MAX_HOLD_BARS,
            'reason': ' | '.join(reason_parts),
        }

    # ----------------------------------------------------------
    # Signal classification
    # ----------------------------------------------------------
    def _classify_signal(
        self,
        price: float,
        max_put_strike: int,
        max_call_strike: int,
        max_put_oi: int,
        max_call_oi: int,
    ) -> Optional[Tuple[str, str, int, int]]:
        """
        Classify whether the current price setup is a bounce, breakout,
        or no signal.

        Returns
        -------
        (signal_type, direction, barrier_strike, barrier_oi) or None.
        """
        dist_to_support = price - max_put_strike
        dist_to_resistance = max_call_strike - price

        # ── BOUNCE off support (LONG) ─────────────────────────────
        if (0 < dist_to_support <= APPROACH_DISTANCE
                and max_put_oi >= MIN_OI_THRESHOLD):
            return (SIGNAL_BOUNCE, 'LONG', max_put_strike, max_put_oi)

        # ── BOUNCE off resistance (SHORT) ─────────────────────────
        if (0 < dist_to_resistance <= APPROACH_DISTANCE
                and max_call_oi >= MIN_OI_THRESHOLD):
            return (SIGNAL_BOUNCE, 'SHORT', max_call_strike, max_call_oi)

        # ── BREAKOUT below support (SHORT) ────────────────────────
        if (dist_to_support < -BREAKOUT_DISTANCE
                and max_put_oi >= MIN_OI_THRESHOLD):
            return (SIGNAL_BREAKOUT, 'SHORT', max_put_strike, max_put_oi)

        # ── BREAKOUT above resistance (LONG) ──────────────────────
        if (dist_to_resistance < -BREAKOUT_DISTANCE
                and max_call_oi >= MIN_OI_THRESHOLD):
            return (SIGNAL_BREAKOUT, 'LONG', max_call_strike, max_call_oi)

        return None

    # ----------------------------------------------------------
    # BACKTEST evaluate
    # ----------------------------------------------------------
    def backtest_evaluate(
        self,
        trade_date: date,
        nifty_bars: List[Dict],
        option_chain_df=None,
    ) -> Optional[List[Dict]]:
        """
        Backtest variant — processes an entire day's 5-min bars against
        the day's option chain OI data.

        Parameters
        ----------
        trade_date      : Date being backtested.
        nifty_bars      : List of 5-min bar dicts for the full session.
                          Each: {open, high, low, close, volume, timestamp}.
        option_chain_df : pandas DataFrame with columns:
                          strike, option_type ('CE'/'PE'), oi, volume.
                          Snapshot of OI for the day (e.g. EOD or mid-session).

        Returns
        -------
        List of trade result dicts, or None if no trades.
        """
        try:
            return self._backtest_inner(trade_date, nifty_bars, option_chain_df)
        except Exception as e:
            logger.error(
                'MaxOIBarrierSignal.backtest_evaluate error: %s', e, exc_info=True
            )
            return None

    def _backtest_inner(
        self,
        trade_date: date,
        nifty_bars: List[Dict],
        option_chain_df,
    ) -> Optional[List[Dict]]:
        # ── Validate inputs ───────────────────────────────────────
        if not nifty_bars or len(nifty_bars) < 4:
            return None

        # ── Find max OI strikes from DataFrame ────────────────────
        max_put_strike, max_call_strike, max_put_oi, max_call_oi = \
            _find_max_oi_strikes_df(option_chain_df)

        if max_put_strike is None or max_call_strike is None:
            logger.debug('%s: Could not find max OI strikes for %s', self.SIGNAL_ID, trade_date)
            return None

        # ── Filter: OI threshold ──────────────────────────────────
        if max_put_oi < MIN_OI_THRESHOLD and max_call_oi < MIN_OI_THRESHOLD:
            return None

        # ── Filter: strike spread ─────────────────────────────────
        strike_spread = abs(max_call_strike - max_put_strike)
        if strike_spread < MIN_STRIKE_SPREAD:
            return None

        # ── Walk through bars, look for entries ───────────────────
        trades: List[Dict] = []
        in_trade = False
        trade_direction = None
        trade_entry_price = 0.0
        trade_sl = 0.0
        trade_tgt = 0.0
        trade_entry_idx = 0
        trade_barrier = 0
        trade_barrier_oi = 0
        trade_signal_type = ''
        bars_held = 0

        for i, bar in enumerate(nifty_bars):
            bar_t = _bar_time_obj(bar)
            bar_cls = _bar_close(bar)
            bar_hi = _bar_high(bar)
            bar_lo = _bar_low(bar)

            if math.isnan(bar_cls) or bar_cls <= 0:
                continue

            # Skip first/last 15 min
            if bar_t is not None:
                if bar_t < NO_TRADE_OPEN or bar_t >= NO_TRADE_CLOSE:
                    # If in trade and session ending, force close
                    if in_trade and bar_t >= NO_TRADE_CLOSE:
                        pnl = self._calc_pnl(trade_direction, trade_entry_price, bar_cls)
                        trades.append(self._build_backtest_result(
                            trade_date, trade_direction, trade_signal_type,
                            trade_entry_price, trade_sl, trade_tgt,
                            bar_cls, i, 'SESSION_END',
                            trade_barrier, trade_barrier_oi,
                            max_put_strike, max_put_oi,
                            max_call_strike, max_call_oi,
                            strike_spread, pnl,
                        ))
                        in_trade = False
                    continue

            # ── If in trade: check SL / TGT / max hold ───────────
            if in_trade:
                bars_held += 1
                exit_price = None
                exit_reason = None

                if not math.isnan(bar_hi) and not math.isnan(bar_lo):
                    if trade_direction == 'LONG':
                        if bar_lo <= trade_sl:
                            exit_price = trade_sl
                            exit_reason = 'SL_HIT'
                        elif bar_hi >= trade_tgt:
                            exit_price = trade_tgt
                            exit_reason = 'TGT_HIT'
                    else:  # SHORT
                        if bar_hi >= trade_sl:
                            exit_price = trade_sl
                            exit_reason = 'SL_HIT'
                        elif bar_lo <= trade_tgt:
                            exit_price = trade_tgt
                            exit_reason = 'TGT_HIT'

                if exit_price is None and bars_held >= MAX_HOLD_BARS:
                    exit_price = bar_cls
                    exit_reason = 'MAX_HOLD'

                if exit_price is not None:
                    pnl = self._calc_pnl(trade_direction, trade_entry_price, exit_price)
                    trades.append(self._build_backtest_result(
                        trade_date, trade_direction, trade_signal_type,
                        trade_entry_price, trade_sl, trade_tgt,
                        exit_price, i, exit_reason,
                        trade_barrier, trade_barrier_oi,
                        max_put_strike, max_put_oi,
                        max_call_strike, max_call_oi,
                        strike_spread, pnl,
                    ))
                    in_trade = False

                continue  # Don't look for new entries while in a trade

            # ── Look for new entry ────────────────────────────────
            if len(trades) >= self._max_daily_trades:
                break  # Enough trades for the day

            signal = self._classify_signal(
                bar_cls, max_put_strike, max_call_strike,
                max_put_oi, max_call_oi,
            )
            if signal is None:
                continue

            trade_signal_type, trade_direction, trade_barrier, trade_barrier_oi = signal
            trade_entry_price = bar_cls
            trade_entry_idx = i
            bars_held = 0

            if trade_direction == 'LONG':
                trade_sl = trade_barrier - SL_POINTS
            else:
                trade_sl = trade_barrier + SL_POINTS

            trade_tgt = _compute_target(trade_direction, trade_entry_price, trade_barrier)
            in_trade = True

            logger.debug(
                '%s backtest entry: %s %s bar=%d price=%.1f barrier=%d',
                self.SIGNAL_ID, trade_signal_type, trade_direction, i, bar_cls, trade_barrier,
            )

        # ── If still in trade at end of data, force close ─────────
        if in_trade and nifty_bars:
            last_bar = nifty_bars[-1]
            last_cls = _bar_close(last_bar)
            if not math.isnan(last_cls) and last_cls > 0:
                pnl = self._calc_pnl(trade_direction, trade_entry_price, last_cls)
                trades.append(self._build_backtest_result(
                    trade_date, trade_direction, trade_signal_type,
                    trade_entry_price, trade_sl, trade_tgt,
                    last_cls, len(nifty_bars) - 1, 'END_OF_DATA',
                    trade_barrier, trade_barrier_oi,
                    max_put_strike, max_put_oi,
                    max_call_strike, max_call_oi,
                    strike_spread, pnl,
                ))

        if not trades:
            return None

        logger.info(
            '%s backtest %s: %d trades, barriers put=%d(%d) call=%d(%d)',
            self.SIGNAL_ID, trade_date, len(trades),
            max_put_strike, max_put_oi, max_call_strike, max_call_oi,
        )

        return trades

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------
    @staticmethod
    def _calc_pnl(direction: str, entry: float, exit_price: float) -> float:
        if direction == 'LONG':
            return exit_price - entry
        else:
            return entry - exit_price

    @staticmethod
    def _build_backtest_result(
        trade_date: date,
        direction: str,
        signal_type: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        exit_price: float,
        exit_bar_idx: int,
        exit_reason: str,
        barrier_strike: int,
        barrier_oi: int,
        max_put_strike: int,
        max_put_oi: int,
        max_call_strike: int,
        max_call_oi: int,
        strike_spread: int,
        pnl_pts: float,
    ) -> Dict:
        """Build a standardised backtest result dict."""
        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        rr = reward / risk if risk > 0 else 0.0

        # Confidence for backtest records
        if signal_type == SIGNAL_BOUNCE:
            confidence = CONF_BOUNCE
        else:
            confidence = CONF_BREAKOUT
        if barrier_oi > MIN_OI_THRESHOLD * 2:
            confidence += OI_MASSIVE_BOOST
        confidence = min(0.90, max(0.10, confidence))

        reason_parts = [
            f"MAX_OI_BARRIER ({signal_type})",
            f"Dir={direction}",
            f"Barrier={barrier_strike}",
            f"OI={barrier_oi:,}",
            f"Exit={exit_reason}",
            f"PnL={pnl_pts:+.1f}",
            f"R:R={rr:.1f}",
        ]

        return {
            'signal_id': SIGNAL_ID,
            'trade_date': trade_date.isoformat(),
            'signal_type': signal_type,
            'direction': direction,
            'confidence': round(confidence, 3),
            'entry_price': round(entry_price, 2),
            'stop_loss': round(stop_loss, 2),
            'target': round(target, 2),
            'exit_price': round(exit_price, 2),
            'exit_bar_idx': exit_bar_idx,
            'exit_reason': exit_reason,
            'pnl_pts': round(pnl_pts, 2),
            'barrier_strike': barrier_strike,
            'barrier_oi': barrier_oi,
            'max_put_strike': max_put_strike,
            'max_put_oi': max_put_oi,
            'max_call_strike': max_call_strike,
            'max_call_oi': max_call_oi,
            'strike_spread': strike_spread,
            'max_hold_bars': MAX_HOLD_BARS,
            'reason': ' | '.join(reason_parts),
        }

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        self._last_fire_date = None
        self._last_fire_time = None
        self._daily_trade_count = 0

    def __repr__(self) -> str:
        return f"MaxOIBarrierSignal(signal_id='{self.SIGNAL_ID}')"
